#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""魔搭创空间入口：在原服务外面包一层门和一道额度闸。

三条硬约束（2026-09-16 作者定）：
  1. **key 只从环境变量来**，代码里一个字符都不许有。创空间的 Secrets 面板填。
  2. **门**：进站要口令。口令也从环境变量来。
  3. **超额只许变笨，不许花超**。额度用完之后服务照常开着，只是一档一档降下去。

原 scripts/server.py 一行没动——本机 ./run.sh 跑法照旧，页面也是同一份，
所以书里写的操作步骤在线上线下是一样的。
"""
import hashlib, hmac, json, os, sys, time, threading
from datetime import datetime
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

import server as S                      # noqa: E402  原服务，不改
import answer                           # noqa: E402

# ── 配置全部来自环境变量 ──────────────────────────────────────
口令      = os.environ.get("SITE_PASSWORD", "")
每天问答   = int(os.environ.get("DAILY_ASK", "150"))      # 全站每天最多调几次生成
每天向量   = int(os.environ.get("DAILY_EMBED", "1200"))   # 全站每天最多调几次向量化
每人问答   = int(os.environ.get("SESSION_ASK", "12"))     # 单个访客每天最多问几次
额度文件   = Path(os.environ.get("QUOTA_FILE", "/tmp/gzh-rag-quota.json"))
COOKIE    = "wgmy"
有效天数   = 7


def 密钥() -> bytes:
    # 拿口令派生一把签名用的密钥。口令本身不进 cookie。
    return hashlib.sha256(("gzh-rag|" + 口令).encode("utf-8")).digest()


def 签(到期: int) -> str:
    mac = hmac.new(密钥(), str(到期).encode(), hashlib.sha256).hexdigest()[:32]
    return f"{到期}.{mac}"


def 验(票: str) -> bool:
    try:
        到期, mac =票.split(".", 1)
        if int(到期) < time.time():
            return False
        return hmac.compare_digest(签(int(到期)), 票)
    except Exception:
        return False


class 额度:
    """全站每日计数。进程重启会从文件读回来，读不回来就从 0 开始——
    这是尽力而为，真正的硬顶是账户那头的消费限额。"""

    def __init__(self):
        self.锁 = threading.Lock()
        self.日 = ""
        self.问 = 0
        self.量 = 0
        self.每人 = {}
        self._读盘()

    def _今天(self):
        return datetime.now().strftime("%Y-%m-%d")

    def _读盘(self):
        try:
            d = json.loads(额度文件.read_text(encoding="utf-8"))
            if d.get("日") == self._今天():
                self.日, self.问, self.量 = d["日"], d["问"], d["量"]
                self.每人 = d.get("每人", {})
                return
        except Exception:
            pass
        self.日 = self._今天()

    def _落盘(self):
        try:
            额度文件.write_text(json.dumps(
                {"日": self.日, "问": self.问, "量": self.量, "每人": self.每人},
                ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass                                    # 写不进去不影响服务

    def _翻天(self):
        if self.日 != self._今天():
            self.日, self.问, self.量, self.每人 = self._今天(), 0, 0, {}

    def 档(self):
        """0=正常  1=只检索不生成  2=连向量化都停，纯字面匹配"""
        with self.锁:
            self._翻天()
            if self.量 >= 每天向量:
                return 2
            if self.问 >= 每天问答:
                return 1
            return 0

    def 记一次问(self, 谁):
        with self.锁:
            self._翻天()
            self.问 += 1
            self.每人[谁] = self.每人.get(谁, 0) + 1
            self._落盘()

    def 记一次量(self):
        with self.锁:
            self._翻天()
            self.量 += 1
            self._落盘()

    def 这人超了吗(self, 谁):
        with self.锁:
            self._翻天()
            return self.每人.get(谁, 0) >= 每人问答

    def 快照(self):
        with self.锁:
            self._翻天()
            return {"档": 0 if self.量 < 每天向量 and self.问 < 每天问答
                    else (2 if self.量 >= 每天向量 else 1),
                    "今日问答": self.问, "问答上限": 每天问答,
                    "今日检索": self.量, "检索上限": 每天向量}


闸 = 额度()
撞门 = {}          # ip -> [失败次数, 起算时刻]，防暴力猜口令


门页 = """<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>编程一生谢晓静　写过没有</title>
<style>
:root{color-scheme:light}
body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
 background:#f7f7f5;font:16px/1.7 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;color:#23262b;
 padding:24px}
.box{width:100%;max-width:420px}
h1{font-size:21px;margin:0 0 6px;letter-spacing:.5px}
p{margin:0 0 22px;color:#6b7280;font-size:14.5px}
input{width:100%;box-sizing:border-box;padding:13px 14px;font-size:16px;border:1.5px solid #d6d9de;
 border-radius:9px;background:#fff;outline:none}
input:focus{border-color:#1a5fb4}
button{width:100%;margin-top:12px;padding:13px;font-size:16px;border:0;border-radius:9px;
 background:#1a5fb4;color:#fff;cursor:pointer}
.err{color:#b4472e;font-size:14px;margin-top:12px;min-height:20px}
</style>
<div class="box">
<h1>写过没有</h1>
<p>把一个观点丢进来，看我以前写过没有。要口令。</p>
<form method="post" action="/gate">
<input name="p" type="password" placeholder="口令" autofocus autocomplete="off">
<button type="submit">进去</button>
<div class="err">__ERR__</div>
</form>
</div></html>"""


class 门内Handler(S.Handler):
    """继承原 Handler，只在最前面加一道门和一道闸，路由逻辑全用原来的。"""

    def _客(self):
        票 = SimpleCookie(self.headers.get("Cookie", "")).get(COOKIE)
        return (票.value if 票 else "")[:40]

    def _进得来吗(self):
        if not 口令:                                  # 没配口令就是不设门（本机调试用）
            return True
        return 验(self._客())

    def _发门页(self, 提示=""):
        body = 门页.replace("__ERR__", 提示).encode("utf-8")
        self._send(200, body, "text/html; charset=utf-8")

    def do_POST(self):
        if urlparse(self.path).path != "/gate":
            return self._json({"error": "not found"}, 404)
        ip = self.client_address[0]
        次, 起 = 撞门.get(ip, (0, time.time()))
        if time.time() - 起 > 3600:
            次, 起 = 0, time.time()
        if 次 >= 10:
            return self._发门页("试得太多了，等一小时再来。")
        n = int(self.headers.get("Content-Length") or 0)
        猜 = parse_qs(self.rfile.read(n).decode("utf-8", "ignore")).get("p", [""])[0]
        if not hmac.compare_digest(猜.strip(), 口令):
            撞门[ip] = (次 + 1, 起)
            return self._发门页("口令不对。")
        撞门.pop(ip, None)
        票 = 签(int(time.time()) + 有效天数 * 86400)
        self.send_response(303)
        self.send_header("Location", "/")
        self.send_header("Set-Cookie",
                         f"{COOKIE}={票}; Path=/; Max-Age={有效天数*86400}; HttpOnly; SameSite=Lax")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if not self._进得来吗():
            if path in ("/", "/index.html", "/gate"):
                return self._发门页()
            return self._json({"error": "要口令"}, 401)

        档 = 闸.档()

        if path == "/api/stats":                       # 把当前档位一并告诉前端
            原 = self.__class__.__mro__[1]
            # 直接复用父类逻辑太绕，这里自己拼一份，字段跟原来一模一样再加两个
            d = self.searcher.docs
            return self._json({
                "articles": len(d), "chunks": len(self.searcher.chunks),
                "comics": sum(1 for x in d.values() if x.get("kind") == "comic"),
                "model": self.searcher.index["model"],
                "built_at": self.searcher.index["built_at"],
                "can_answer": answer.has_key() and 档 == 0,
                "started_at": S.STARTED_AT,
                "earliest": min((x["date"] for x in d.values() if x["date"]), default=""),
                "latest": max((x["date"] for x in d.values() if x["date"]), default=""),
                "额度": 闸.快照(),
            })

        if path == "/api/search":
            qs = parse_qs(urlparse(self.path).query)
            if 档 >= 2:
                qs["mode"] = ["lex"]                   # 二档：一次接口都不调
                self.path = f"/api/search?{_拼(qs)}"
            else:
                闸.记一次量()
            return super().do_GET()

        if path == "/api/ask":
            qs = parse_qs(urlparse(self.path).query)
            q = (qs.get("q") or [""])[0]
            谁 = self._客()[:16] or self.client_address[0]
            if 档 >= 1 or 闸.这人超了吗(谁):
                return self._只检索(q, qs, 档, 闸.这人超了吗(谁))
            闸.记一次量(); 闸.记一次问(谁)
            return super().do_GET()

        return super().do_GET()

    def _只检索(self, q, qs, 档, 个人超了):
        """降级路径：照常把检索结果推给前端，把「替你读一遍」那一步省掉。
        走的是前端已有的 sources / delta / done 三个事件，页面一个字都不用改。"""
        q = (q or "").strip()
        if not q:
            return self._json({"error": "empty query"}, 400)
        mode = "lex" if 档 >= 2 else (qs.get("mode") or ["hybrid"])[0]
        if 档 < 2:
            闸.记一次量()
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        def push(o):
            self.wfile.write(f"data: {json.dumps(o, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()
        try:
            r = self.searcher.search(q, limit=answer.CONTEXT_DOCS, with_chunks=True,
                                     mode=mode, kind=(qs.get("kind") or ["all"])[0])
            push({"type": "sources", "terms": S.useful_terms(q), "started_at": S.STARTED_AT,
                  "results": [{k: v for k, v in x.items() if k != "hit_texts"} for x in r]})
            if 个人超了:
                话 = "你今天问得有点多，先歇一歇。下面是检索到的原文，明天再来它就接着替你读。"
            elif 档 >= 2:
                话 = ("今天的额度用完了，现在只按字面匹配找，换个说法可能就搜不到。"
                      "下面是找到的原文，明天恢复。")
            else:
                话 = "今天让它替我读的次数用完了。下面这些原文照样能翻，明天再来就能让它替你读。"
            push({"type": "delta", "text": 话})
            push({"type": "done"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            try:
                push({"type": "error", "text": str(exc)})
            except Exception:
                pass


def _拼(qs):
    from urllib.parse import urlencode
    return urlencode([(k, v) for k, vs in qs.items() for v in vs])


def main():
    S.STARTED_AT = int(time.time())
    port = int(os.environ.get("PORT", "7860"))
    print("正在加载索引…", flush=True)
    S.Handler.searcher = S.Searcher()
    门内Handler.searcher = S.Handler.searcher
    gp = S.DATA / "graph.json"
    g = json.loads(gp.read_text(encoding="utf-8")) if gp.exists() else None
    S.Handler.graph = 门内Handler.graph = g
    print(f"就绪：0.0.0.0:{port}　门{'开着' if 口令 else '没设（没配 SITE_PASSWORD）'}　"
          f"额度 问答{每天问答}/天 检索{每天向量}/天", flush=True)
    S.ThreadingHTTPServer(("0.0.0.0", port), 门内Handler).serve_forever()


if __name__ == "__main__":
    main()
