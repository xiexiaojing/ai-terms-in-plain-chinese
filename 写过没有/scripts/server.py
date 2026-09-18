#!/usr/bin/env python3
"""公众号文章语义检索服务：本机跑，浏览器访问。

检索是向量 + 字面双路：
  向量路（模型由 .env 决定，见 embedding.py）负责「换个说法也能搜到」，
  字面路（中文 bigram BM25）负责「术语、人名、书名必须命中」，
两路分数融合后按文章聚合。缺了任一路都会漏——只有向量时搜「省 token」会给出
讲 JWT 鉴权的文章，只有字面时搜「怎么判断该不该重构」什么也搜不到。
"""
import json
import math
import os
import re
import socket
import sys
import threading
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

import time

import answer
import embedding

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

STARTED_AT = None          # 服务启动时刻，用来提醒开着的旧页面刷新

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WEB = ROOT / "web"

VEC_WEIGHT = 0.72          # 语义权重
BM25_WEIGHT = 0.28         # 字面权重
LEX_SATURATION = 12.0      # BM25 归一化的饱和常数，见 bm25_norm()
TITLE_BONUS = 0.18         # 标题命中加成
TITLE_MATCH_GATE = 0.70    # 「有一篇专门写这个」的判据，见 Searcher.title_match
MULTI_HIT_BONUS = 0.035    # 一篇里多处命中的加成（每处，封顶 3 处）


# 高亮用：两个字都是虚词的 bigram 没有信息量，全高亮出来满屏是黄的
STOP_CHARS = set("的了是在我你他她它们这那有和就都也不要么什怎为以上下会能还说吧呢啊很把被让给对从到与及或但而且如自己个之其")


def useful_terms(query: str):
    """从查询里挑出值得高亮的词：实词 bigram 与英文数字词。"""
    picked = [t for t in set(bigrams(query))
              if len(t) >= 2 and not (len(t) == 2 and (t[0] in STOP_CHARS or t[1] in STOP_CHARS))]
    return sorted(picked or [t for t in set(bigrams(query)) if len(t) >= 2], key=len, reverse=True)


def bigrams(text: str):
    """中文按 2-gram 切，英文数字按词切。省掉分词依赖，短查询上够用。"""
    text = re.sub(r"[\s　]+", " ", (text or "").lower())
    terms = []
    for token in re.findall(r"[a-z0-9_.+#-]+|[一-鿿]+", text):
        if re.match(r"^[a-z0-9]", token):
            terms.append(token)
        else:
            terms.extend(token[i:i + 2] for i in range(max(len(token) - 1, 1)))
    return terms


def bm25_norm(lex: np.ndarray) -> np.ndarray:
    """把 BM25 压到 [0,1)，用饱和函数而不是「除以本次查询的最大值」。

    除以本次最大值的坏处是分数只在查询内部可比：搜「马拉松跑鞋怎么选」这种根本没写过的，
    也必然有一块拿到 0.97——《思维发散的双刃剑》里有句「人生是一场马拉松」——
    融合后总分 0.67，比真写过的「布隆过滤器」(0.55) 还高。「写过没写过」就此判不准。
    饱和归一化后分数有绝对含义：命中得多、命中词罕见，分才高。
    """
    return lex / (lex + LEX_SATURATION)


class Searcher:
    def __init__(self):
        self.index = json.loads((DATA / "index.json").read_text(encoding="utf-8"))
        self.chunks = self.index["chunks"]
        self.vectors = np.load(DATA / "vectors.npy")
        tv = DATA / "title_vectors.npy"
        self.title_vectors = np.load(tv) if tv.exists() else None
        self.title_pos = {d: i for i, d in enumerate(self.index.get("title_doc_ids", []))}
        self.docs = json.loads((DATA / "docs.json").read_text(encoding="utf-8"))["docs"]
        self.fulltext = json.loads((DATA / "fulltext.json").read_text(encoding="utf-8"))
        self._build_bm25()
        self._model = None
        self._lock = threading.Lock()

    # ---------- 字面路 ----------
    def _build_bm25(self):
        self.postings = defaultdict(list)
        self.doc_len = np.zeros(len(self.chunks), dtype=np.float32)
        for i, c in enumerate(self.chunks):
            # 标题要并进每块：搜「杀猪盘」时，那三个字只在标题里，
            # 不并进来字面路一个词都命中不了，全靠语义扛，分数低到被判成「没写过」。
            doc = self.docs.get(c["doc_id"], {})
            terms = bigrams(f"{doc.get('title', '')} {c['heading']} {c['text']}")
            self.doc_len[i] = len(terms) or 1
            for term, tf in Counter(terms).items():
                self.postings[term].append((i, tf))
        self.avg_len = float(self.doc_len.mean()) or 1.0
        self.n_docs = len(self.chunks)
        self.title_terms = {
            doc_id: set(bigrams(" ".join([d["title"], *d.get("aka", [])])))
            for doc_id, d in self.docs.items()
        }

    def bm25(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        k1, b = 1.5, 0.75
        for term in set(bigrams(query)):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = math.log(1 + (self.n_docs - len(posting) + 0.5) / (len(posting) + 0.5))
            for i, tf in posting:
                norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * self.doc_len[i] / self.avg_len))
                scores[i] += idf * norm
        return scores

    # ---------- 语义路 ----------
    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    if self.index["model"].startswith("BAAI/"):
                        self._model = embedding.get("bge")
                    else:
                        self._model = embedding.ApiEmbedder(self.index.get("protocol", ""))
        return self._model

    @property
    def can_vector(self) -> bool:
        """索引是接口算的、手上又没 key，就只能走字面路。"""
        if self.index["model"].startswith("BAAI/"):
            return True                    # 本机模型算的索引，不需要 key
        return embedding.has_key()

    def check_model(self):
        """索引是哪个模型算的，现在配的还是不是它。换了模型不重建索引，
        查询向量跟库里的根本不在一个空间，检索结果会安静地全乱，不报错。"""
        built = self.index.get("model", "")
        if built.startswith("BAAI/") or not embedding.has_key():
            return
        now = embedding.model_name()
        if now and built and now != built:
            print(f"⚠️  索引是用 {built} 建的，现在 .env 配的是 {now}。"
                  f"\n    这两个向量对不上，检索会全乱。跑 ./refresh.sh 重建索引，"
                  f"或把 EMBEDDING_MODEL 改回 {built}。", file=sys.stderr)

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.query(query)

    def title_match(self, doc_id: str, q_terms: set, qvec) -> float:
        """这篇的标题是不是在讲这件事。字面和语义取大，两条各有各的盲区：

        「婆媳关系怎么处」对上《我的婆媳相处之道》，字面只重叠 0.17，语义 0.95；
        「熔断的原理」对上《稳定性五件套-熔断的原理和实现》，字面 1.00，语义反而只有 0.34。
        取最大值，两种情况都接得住。

        这个信号是用来分「专门写过」和「顺带提过一句」的——后者的第一名往往是
        《面试题总结》这种大杂烩，正文命中一大片，标题跟问题毫无关系。
        """
        lex = len(q_terms & self.title_terms.get(doc_id, set())) / len(q_terms) if q_terms else 0.0
        sem = 0.0
        if self.title_vectors is not None and qvec is not None and doc_id in self.title_pos:
            sem = float(self.title_vectors[self.title_pos[doc_id]] @ qvec)
        return max(lex, sem)

    # ---------- 融合与聚合 ----------
    def search(self, query: str, limit: int = 20, order: str = "relevance", kind: str = "all",
               mode: str = "hybrid", with_chunks: bool = False):
        """三种检索模式，课上要当场切着看差别：

        hybrid  两路融合，正常用的就是它
        vector  「朴素向量 RAG」：只有余弦，关掉 BM25 和标题加成
        lex     只有 BM25 字面路，一次接口都不调（没配 key 时系统也自动落到这儿）
        """
        query = (query or "").strip()
        if not query:
            return []

        vector_only = mode == "vector"
        # 没 key 又是接口索引：向量这一路整个用不了，只能落到纯字面。
        # 效果差一截，但学员没配 key 时至少还能搜，而不是打不开。
        lex_only = mode == "lex" or not self.can_vector
        qvec = None
        if lex_only:
            lex_norm = bm25_norm(self.bm25(query))
            sims = np.zeros(len(self.chunks), dtype=np.float32)
            fused = lex_norm
        elif vector_only:
            qvec = self.embed_query(query)
            sims = self.vectors @ qvec
            lex_norm = np.zeros_like(sims)
            fused = sims
        else:
            qvec = self.embed_query(query)
            sims = self.vectors @ qvec
            lex_norm = bm25_norm(self.bm25(query))
            fused = VEC_WEIGHT * sims + BM25_WEIGHT * lex_norm

        q_terms = set(bigrams(query))
        top_idx = np.argsort(-fused)[:400]

        grouped = defaultdict(list)
        for i in top_idx:
            c = self.chunks[i]
            doc = self.docs.get(c["doc_id"])
            if not doc:
                continue
            dk = doc.get("kind", "article")
            # 「我的文件」是读者自己的库，跟公众号那堆分开：all 只搜公众号（文章＋漫画），file 只搜文件
            if kind == "all" and dk == "file":
                continue
            if kind in ("article", "comic", "file") and dk != kind:
                continue
            grouped[c["doc_id"]].append((float(fused[i]), float(sims[i]), float(lex_norm[i]), c))

        results = []
        for doc_id, hits in grouped.items():
            hits.sort(key=lambda h: -h[0])
            doc = self.docs[doc_id]
            score = hits[0][0] + MULTI_HIT_BONUS * min(len(hits) - 1, 3)
            tmatch = self.title_match(doc_id, q_terms, qvec)
            if q_terms and not vector_only:
                overlap = len(q_terms & self.title_terms.get(doc_id, set())) / len(q_terms)
                score += TITLE_BONUS * overlap
            results.append({
                "id": doc_id,
                "title": doc["title"],
                "date": doc["date"],
                "link": doc["link"],
                "kind": doc.get("kind", "article"),
                "has_text": doc["source"] == "本地全文",
                "read_num": doc.get("read_num"),
                "topic": doc.get("topic", ""),
                "score": round(score, 4),
                "title_match": round(tmatch, 4),
                "vec_score": round(hits[0][1], 4),
                "lex_score": round(hits[0][2], 4),
                "hit_count": len(hits),
                "snippets": [{"heading": h[3]["heading"], "text": snippet(h[3]["text"], q_terms),
                              "score": round(h[0], 4)} for h in hits[:2]],
                "thumbs": len(comic_images(doc_id)),
            })
            if with_chunks:
                # 喂给模型的是命中的那几块原文，不是文章开头——命中常在中部，
                # 喂开头等于让模型对着不相关的段落回答问题。
                results[-1]["hit_texts"] = [
                    (f"（{h[3]['heading']}）\n" if h[3]["heading"] else "") + h[3]["text"]
                    for h in hits[:3]
                ]

        results.sort(key=lambda r: (r["date"] or "", r["score"]) if order == "date" else r["score"],
                     reverse=True)
        return results[:limit]


# 漫画的缩略图（2026-09-12 加）：漫画条目本来只有标题和一行 digest，卡片空荡荡，
# 比纯文字的文章还难看。漫画的原图就在 wechat-comic/<子路径>/images/ 下，doc id
# 形如 `comic:已归档/wechat-comic-daxie`，冒号后面那截正好是子路径。
COMIC_ROOT = (Path.home() / "workspace" / "wechat-comic").resolve()
THUMB_MAX = 3


def comic_images(doc_id: str):
    """按 doc id 找出这篇漫画的前几张图，找不到返回空列表。"""
    if not doc_id.startswith("comic:"):
        return []
    sub = doc_id.split(":", 1)[1]
    d = (COMIC_ROOT / sub / "images").resolve()
    # 只认 COMIC_ROOT 底下的路径，挡掉 id 里带 ../ 的情况
    if not str(d).startswith(str(COMIC_ROOT)) or not d.is_dir():
        return []
    return sorted(q for q in d.iterdir()
                  if q.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))[:THUMB_MAX]


def snippet(text: str, q_terms, width: int = 260) -> str:
    """截取片段：尽量从命中词附近开始，别从半句话中间起。

    查询词经过 bigrams() 已经是小写，正文要保留原文大小写给人看，
    定位必须大小写不敏感——不然搜 mqtt 对不上正文里的 MQTT，
    窗口会从文首切，命中词落在 260 字之外，摘要变成「使用 M…」。
    """
    if len(text) <= width:
        return clean_marks(text)
    haystack = text.lower()
    pos = -1
    hit_end = -1
    for term in q_terms:
        if len(term) < 2:
            continue
        found = haystack.find(term.lower())
        if found != -1 and (pos == -1 or found < pos):
            pos = found
            hit_end = found + len(term)
    if pos < 0:
        return clean_marks(text[:width]) + "…"
    start = max(0, pos - 60)
    if hit_end > start + width:          # 命中贴在文末：窗口后移，把词完整留下
        start = max(0, hit_end - width + 8)
    for punct in "。！？\n":
        p = text.rfind(punct, max(0, start - 40), pos)
        if p != -1:
            start = p + 1
            break
    if hit_end > start + width:          # 往前对齐到句首后，再确认一次词还在窗口里
        start = max(0, hit_end - width + 8)
    return ("…" if start > 0 else "") + clean_marks(text[start:start + width].strip()) + "…"


def clean_marks(text: str) -> str:
    """片段里的 markdown 记号（**加粗**、`代码`、> 引用）读起来碍事，展示前抹掉。"""
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", text)   # 正文里的互链只留标题
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = text.replace("**", "")      # 截断处会留下落单的加粗记号
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.M)
    return text.strip()


class Handler(BaseHTTPRequestHandler):
    searcher: Searcher = None
    graph = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        # send_error 会把 HTTPStatus 放进 args[0]，不能直接做 "/api/" in ...
        first = args[0] if args else ""
        if not isinstance(first, str):
            first = str(first)
        if "/api/" in first or "/api/" in (fmt or ""):
            try:
                sys.stderr.write("  %s\n" % (fmt % args))
            except Exception:
                sys.stderr.write("  %s %s\n" % (fmt, args))

    def _send(self, status, body: bytes, ctype="application/json; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=200):
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _ask(self, query, mode="hybrid", kind="all"):
        """SSE：先把检索到的文章推给前端（立刻有东西看），再流式吐答案。

        mode/kind 必须跟着传下来。之前这里漏了，页面上切「只用向量」对「问一句」
        毫无反应——两种模式返回一模一样的结果，看着像检索坏了。
        """
        query = (query or "").strip()
        if not query:
            return self._json({"error": "empty query"}, 400)
        self.close_connection = True          # SSE 没有 Content-Length，靠关连接收尾
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        def push(obj):
            self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        try:
            results = self.searcher.search(query, limit=answer.CONTEXT_DOCS,
                                           with_chunks=True, mode=mode, kind=kind)
            push({"type": "sources", "terms": useful_terms(query), "started_at": STARTED_AT,
                  "results": [{k: v for k, v in r.items() if k != "hit_texts"} for r in results]})
            # 分不够就不调模型：这个库的阈值是标定过的，「没写过」比硬凑一段可信得多
            if not results or results[0]["score"] < answer.ANSWER_FLOOR:
                push({"type": "no_answer"})
                return push({"type": "done"})
            for ev in answer.stream_answer(query, results, self.searcher.fulltext, kind):
                push(ev)
            push({"type": "done"})
        except (BrokenPipeError, ConnectionResetError):
            pass                                  # 用户关页面了，正常收摊
        except Exception as exc:
            try:
                push({"type": "error", "text": str(exc)})
            except Exception:
                pass

    def do_GET(self):
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        path = url.path

        if path in ("/", "/index.html"):
            return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/stats":
            docs = self.searcher.docs.values()
            return self._json({
                "articles": len(self.searcher.docs),
                "chunks": len(self.searcher.chunks),
                "comics": sum(1 for d in self.searcher.docs.values() if d.get("kind") == "comic"),
                "files": sum(1 for d in self.searcher.docs.values() if d.get("kind") == "file"),
                "model": self.searcher.index["model"],
                "built_at": self.searcher.index["built_at"],
                "can_answer": answer.has_key(),
                "started_at": STARTED_AT,
                "earliest": min((d["date"] for d in self.searcher.docs.values() if d["date"]), default=""),
                "latest": max((d["date"] for d in self.searcher.docs.values() if d["date"]), default=""),
            })
        if path == "/api/ask":
            return self._ask((qs.get("q") or [""])[0],
                             mode=(qs.get("mode") or ["hybrid"])[0],
                             kind=(qs.get("kind") or ["all"])[0])
        if path == "/api/search":
            q = (qs.get("q") or [""])[0]
            try:
                results = self.searcher.search(
                    q,
                    limit=min(int((qs.get("limit") or [20])[0]), 50),
                    order=(qs.get("order") or ["relevance"])[0],
                    kind=(qs.get("kind") or ["all"])[0],
                    mode=(qs.get("mode") or ["hybrid"])[0],
                )
            except Exception as exc:                      # 检索失败也要给前端一个能显示的错
                return self._json({"error": str(exc)}, 500)
            return self._json({"query": q, "count": len(results), "results": results,
                               "terms": useful_terms(q), "started_at": STARTED_AT})
        if path == "/api/doc":
            doc_id = (qs.get("id") or [""])[0]
            doc = self.searcher.docs.get(doc_id)
            if not doc:
                return self._json({"error": "not found"}, 404)
            return self._json({"title": doc["title"], "link": doc["link"], "date": doc["date"],
                               "text": self.searcher.fulltext.get(doc_id, "")})
        if path == "/api/thumb":
            imgs = comic_images((qs.get("id") or [""])[0])
            try:
                f = imgs[int((qs.get("n") or ["0"])[0])]
            except (IndexError, ValueError):
                return self._json({"error": "no such thumb"}, 404)
            ctype = {".png": "image/png", ".webp": "image/webp"}.get(f.suffix.lower(), "image/jpeg")
            return self._send(200, f.read_bytes(), ctype)
        if path == "/api/graph":
            if not self.graph:
                return self._json({"error": "图谱还没建，跑 scripts/build_graph.py"}, 404)
            return self._json({**self.graph, "started_at": STARTED_AT})
        return self._json({"error": "not found"}, 404)


def pick_port(preferred: int) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise SystemExit("没有空闲端口")


def host() -> str:
    """默认只听本机。要让学员从局域网访问，起服务时加 HOST=0.0.0.0——
    开之前想清楚：同一个网络里的人都能打开，没有口令。"""
    return os.environ.get("HOST", "127.0.0.1")


def main():
    global STARTED_AT
    STARTED_AT = int(time.time())
    port = pick_port(int(sys.argv[1]) if len(sys.argv) > 1 else 7788)
    print("正在加载索引与模型…")
    Handler.searcher = Searcher()
    Handler.searcher.check_model()
    Handler.searcher.search("预热", limit=1)      # 提前把模型载进来，首次搜索不卡
    gp = DATA / "graph.json"
    Handler.graph = json.loads(gp.read_text(encoding="utf-8")) if gp.exists() else None
    bind = host()
    shown = "127.0.0.1" if bind in ("127.0.0.1", "localhost") else bind
    print(f"就绪：http://{shown}:{port}  （{len(Handler.searcher.docs)} 篇 / "
          f"{len(Handler.searcher.chunks)} 块）")
    ThreadingHTTPServer((bind, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
