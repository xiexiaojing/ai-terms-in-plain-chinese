"""用检索到的片段，替她本人回答问题——RAG 的最后一步。

生成走 MiniMax（OpenAI 兼容接口）。两条硬约束，都是冲着「别编」去的：

1. 检索分数不够就不调模型。这个库已经标定过阈值，「没写过」是个可靠判断，
   与其让模型对着一堆沾边的片段硬凑，不如直接说没写过——省钱，也不会骗人。
2. 喂给模型的只有她自己的原文，答案里每句话都要能指回某一篇。
"""
import json
import os

import httpx

ANSWER_TITLE_GATE = 0.70  # 没有一篇标题在讲这事，就按「只是提过」来答，别说得像专门写过
ANSWER_FLOOR = 0.30      # 低于这个分就不调模型，直接说没写过（阈值标定见 web/index.html）
CONTEXT_DOCS = 6         # 喂几篇进去
CHARS_PER_DOC = 1500     # 每篇最多喂多少字（喂的是命中块，不是文章开头）

# 她别处配的是完整路径（.../v1/chat/completions），也有人配到 /v1 为止，两种都认
def _env_first(names, default=""):
    for n in names:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    return default


# 变量名兼容旧写法，学员的 .env 不用改
_BASE = _env_first(("LLM_BASE_URL", "MINIMAX_BASE_URL", "OPENAI_BASE_URL"),
                   "https://api.minimaxi.com/v1").rstrip("/")
CHAT_URL = _BASE if _BASE.endswith("/chat/completions") else f"{_BASE}/chat/completions"
MODEL = _env_first(("LLM_MODEL", "ANSWER_MODEL"), "MiniMax-M2.7-highspeed")
TIMEOUT = float(os.environ.get("ANSWER_TIMEOUT", "120"))
# 本机的 HTTPS_PROXY 指着翻墙代理，MiniMax 是国内服务，走代理反而不通。
# 真要走代理起服务时加 MINIMAX_USE_PROXY=1。
USE_PROXY = _env_first(("LLM_USE_PROXY", "MINIMAX_USE_PROXY")) == "1"

SYSTEM = """你在替谢晓静回答读者的提问。她是有二十年经验的女程序员，写了六百多篇技术与生活文章。

用户给你的是从她**已发布**文章里检索出来的片段。回答规则：

1. 只用片段里她说过的话。片段里没有的一个字都别加，包括你自己知道的正确答案。
2. 用她的第一人称（「我」），保留她原本的立场、说法和语气，别改写成中性的教科书表述。
3. **每一段末尾都必须带 [编号]**，编号对应片段序号，同一段用了两篇就写 [1][3]。
   这是硬要求：读者要靠它点回原文核对。示例——「加缓存救不了这类问题，用户等的是能不能开始干活。[2]」
   一段都不带编号的回答等于没完成。
4. 这些片段答不了这个问题，就直说「这个我没正面写过」，再说清沾边的是哪几篇、它们各讲什么。别硬凑。
5. 三到五句话，说完就停。不写总结段，不写建议，不写「希望对你有帮助」。
6. 中文正文用「」当引号，不用双引号。直接给答案，不要复述问题。"""


# 2026-09-17 「我的文件」库：读者问的是自己公司的制度、说明书，不是她的文章，口径换成中性的。
SYSTEM_FILE = """用户给你的是从**用户自己的文件**里检索出来的片段，你按片段替他回答。回答规则：
1. 只用片段里写着的话。片段里没有的一个字都别加，包括你自己知道的正确答案。
2. **每一段末尾都必须带 [编号]**，编号对应片段序号。这是硬要求：读者要靠它点回原文核对。
3. 这些片段答不了这个问题，就直说「文件里没有写这件事」，再说清沾边的是哪一段、它讲的是什么。别硬凑，别推测。
4. 两到四句话，说完就停。不写总结，不写建议。
5. 中文正文用「」当引号。直接给答案，不要复述问题。"""


def build_messages(query: str, results: list, fulltext: dict, kind: str = "all") -> list:
    parts = []
    for i, r in enumerate(results, 1):
        blocks = r.get("hit_texts") or [s["text"] for s in r.get("snippets", [])]
        body = "\n\n".join(blocks).strip() or (fulltext.get(r["id"]) or "")[:CHARS_PER_DOC]
        label = ("制度版本：" + r["version"]) if kind == "file" and r.get("version") else ("文件日期：" if kind == "file" else "发布日期：") + r["date"]
        parts.append(f"[{i}]《{r['title']}》（{label}）\n{body[:CHARS_PER_DOC]}")
    # 角标要求放在 user 消息最后：只写在 system 里，M2.7 实测会漏掉，
    # 挪到紧挨生成位置才稳。
    user = ("片段：\n\n" + "\n\n".join(parts) + f"\n\n问题：{query}\n\n"
            "按上面的规则回答。每段末尾必须标出处编号，像这样：「……用户等的是能不能开始干活。[2]」")
    system = SYSTEM_FILE if kind == "file" else SYSTEM
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class ThinkStripper:
    """剥掉推理模型吐出来的 <think>…</think>。

    流式下标签会被切成两截（`<thi` 一个 chunk，`nk>` 下一个），所以不能逐块 replace：
    没闭合的尾巴要留在缓冲里等下一块，只把确定安全的部分放出去。
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.buf = ""
        self.in_think = False

    def feed(self, text: str) -> str:
        self.buf += text
        out = []
        while True:
            if self.in_think:
                i = self.buf.find(self.CLOSE)
                if i == -1:
                    self.buf = self.buf[-(len(self.CLOSE) - 1):]      # 只留可能是半个闭标签的尾巴
                    break
                self.buf = self.buf[i + len(self.CLOSE):]
                self.in_think = False
                continue
            i = self.buf.find(self.OPEN)
            if i == -1:
                keep = len(self.OPEN) - 1                             # 尾巴可能是半个开标签
                if len(self.buf) > keep:
                    out.append(self.buf[:-keep])
                    self.buf = self.buf[-keep:]
                break
            out.append(self.buf[:i])
            self.buf = self.buf[i + len(self.OPEN):]
            self.in_think = True
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self.in_think else self.buf
        self.buf = ""
        return rest


def has_key() -> bool:
    return bool(_env_first(("LLM_API_KEY", "MINIMAX_API_KEY", "OPENAI_API_KEY")))


def stream_answer(query: str, results: list, fulltext: dict, kind: str = "all"):
    key = _env_first(("LLM_API_KEY", "MINIMAX_API_KEY", "OPENAI_API_KEY"))
    if not key:
        yield {"type": "no_key"}
        return

    payload = {"model": MODEL, "stream": True, "max_tokens": 1024, "temperature": 0.3,
               "messages": build_messages(query, results, fulltext, kind)}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    stripper = ThinkStripper()
    try:
        with httpx.Client(timeout=TIMEOUT, trust_env=USE_PROXY) as client:
            with client.stream("POST", CHAT_URL, headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    resp.read()
                    yield {"type": "error", "text": _explain(resp.status_code, resp.text)}
                    return
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        tail = stripper.flush()
                        if tail:
                            yield {"type": "delta", "text": tail}
                        return
                    try:
                        ev = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    # MiniMax 走 OpenAI 兼容格式，但结束那条 chunk 里 choices 可能没有 delta
                    for choice in ev.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("reasoning_content"):
                            continue                                  # 有的模型把思考单独放这个字段
                        clean = stripper.feed(delta.get("content") or "")
                        if clean:
                            yield {"type": "delta", "text": clean}
                    if ev.get("base_resp", {}).get("status_code") not in (0, None):
                        yield {"type": "error",
                               "text": ev["base_resp"].get("status_msg") or "MiniMax 返回了错误"}
                        return
                # 2026-09-17 修：MiniMax 有时发完 finish_reason=stop 就关流，不给 [DONE]，
                # 缓冲里留着的最后几个字（防半个 <think> 标签的那截）从没吐出来，答案总缺个尾巴。
                tail = stripper.flush()
                if tail:
                    yield {"type": "delta", "text": tail}
    except httpx.TimeoutException:
        yield {"type": "error", "text": f"MiniMax 超过 {TIMEOUT:.0f} 秒没回，先当它挂了。"}
    except httpx.HTTPError as exc:
        yield {"type": "error", "text": f"连不上 MiniMax：{exc}"}


def _explain(status: int, body: str) -> str:
    """接口的原始报错甩给用户看不懂，翻译成能照做的一句。"""
    if status in (401, 403):
        return "模型服务说这个 key 不对或没权限，检查 .env 里的 LLM_API_KEY。"
    if status == 429:
        return "MiniMax 限流了，等一下再问。"
    if status == 404:
        return f"模型服务说没有 {MODEL} 这个模型，改 .env 里的 LLM_MODEL 再起服务。"
    return f"MiniMax 返回 {status}：{body[:200]}"
