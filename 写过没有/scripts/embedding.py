"""向量化。不绑任何一家：OpenAI 兼容协议和 MiniMax 私有协议都认，本机模型也留着。

为什么默认调接口而不是本机跑模型：这套要发给学员，他们得在几分钟内自己搭起来。
`pip install torch` 是 2GB，模型权重另外 391MB，网速好也得十几分钟。
改调接口后依赖只剩 numpy + httpx，装完就能跑。

协议两种，`EMBEDDING_PROTOCOL` 不填就自动探测（先试 OpenAI 格式，再试 MiniMax）：

  openai   POST /embeddings  {"model":…, "input":[…]}  →  {"data":[{"embedding":[…]}]}
           OpenAI、DeepSeek、通义、智谱、月之暗面、硅基流动、本地 Ollama / vLLM 都是这个

  minimax  POST /embeddings  {"model":…, "texts":[…], "type":"db"|"query"}  →  {"vectors":[…]}
           MiniMax 独有，而且是非对称的：入库传 db、查询传 query，混用会掉精度

换模型必须重建索引——维度和向量空间都变了，旧向量跟新查询根本不在一个空间里。
server.py 启动时会核对，对不上会直接说出来。
"""
import os
import time

import numpy as np

BATCH = int(os.environ.get("EMBEDDING_BATCH", "64"))
MAX_CHARS = int(os.environ.get("EMBEDDING_MAX_CHARS", "1500"))

BGE_MODEL = "BAAI/bge-base-zh-v1.5"
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# 兼容早先只支持 MiniMax 时写的变量名，学员的 .env 不用改
_KEY_VARS = ("LLM_API_KEY", "EMBEDDING_API_KEY", "MINIMAX_API_KEY", "OPENAI_API_KEY")
_BASE_VARS = ("EMBEDDING_BASE_URL", "LLM_BASE_URL", "MINIMAX_BASE_URL", "OPENAI_BASE_URL")


def env_first(names, default=""):
    for n in names:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    return default


def api_key() -> str:
    return env_first(_KEY_VARS)


def has_key() -> bool:
    return bool(api_key())


def base_url() -> str:
    base = env_first(_BASE_VARS, "https://api.minimaxi.com/v1").rstrip("/")
    # .env 里常常填的是聊天端点，回退到根再拼 /embeddings
    for suffix in ("/chat/completions", "/text/chatcompletion_v2", "/embeddings"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


def model_name() -> str:
    """不填就按 base_url 猜一个常见的，猜不出来让用户自己填。"""
    named = env_first(("EMBEDDING_MODEL",))
    if named:
        return named
    host = base_url()
    for frag, guess in (("minimax", "embo-01"), ("dashscope", "text-embedding-v3"),
                        ("deepseek", "deepseek-embedding"), ("bigmodel", "embedding-3"),
                        ("siliconflow", "BAAI/bge-m3"), ("moonshot", "moonshot-embedding-v1"),
                        ("openai", "text-embedding-3-small")):
        if frag in host:
            return guess
    return "text-embedding-3-small"


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


class ApiEmbedder:
    """走 HTTP 的向量化。协议不确定时先试 OpenAI 格式，被拒了再试 MiniMax。"""

    def __init__(self, protocol: str = ""):
        import httpx
        self._httpx = httpx
        self.key = api_key()
        if not self.key:
            raise RuntimeError("没有 API key。把 LLM_API_KEY 填进 .env。")
        self.url = f"{base_url()}/embeddings"
        self.name = model_name()
        self.protocol = protocol or os.environ.get("EMBEDDING_PROTOCOL", "").strip()
        self.dim = None

    # ---------- 两种协议的收发 ----------
    def _payload(self, texts, kind, protocol):
        if protocol == "minimax":
            return {"model": self.name, "texts": texts, "type": kind}
        return {"model": self.name, "input": texts}          # OpenAI 兼容

    def _parse(self, data, protocol):
        if protocol == "minimax":
            return data.get("vectors")
        rows = data.get("data")
        if not isinstance(rows, list):
            return None
        # OpenAI 不保证按请求顺序返回，按 index 排回去
        rows = sorted(rows, key=lambda r: r.get("index", 0))
        return [r.get("embedding") for r in rows]

    def _try(self, texts, kind, protocol):
        r = self._httpx.post(self.url, headers={"Authorization": f"Bearer {self.key}"},
                             json=self._payload(texts, kind, protocol),
                             timeout=90, trust_env=False)
        if r.status_code != 200:
            return None, f"{r.status_code} {r.text[:160]}"
        try:
            vecs = self._parse(r.json(), protocol)
        except ValueError:
            return None, "返回的不是 JSON"
        if not vecs or vecs[0] is None:
            return None, f"返回里没有向量：{str(r.json())[:160]}"
        return vecs, ""

    def _post(self, texts, kind):
        order = [self.protocol] if self.protocol else ["openai", "minimax"]
        last = ""
        for attempt in range(3):
            for proto in order:
                vecs, err = self._try(texts, kind, proto)
                if vecs:
                    if not self.protocol:                    # 探测成功就钉死，后面不再试错
                        self.protocol = proto
                        order = [proto]
                    return vecs
                last = f"[{proto}] {err}"
            time.sleep(2 * (attempt + 1))                    # 两种都不成，可能只是网络抖
        raise RuntimeError(
            f"向量化接口调不通：{last}\n"
            f"  端点 {self.url}\n  模型 {self.name}\n"
            f"  对不上就在 .env 里改 EMBEDDING_BASE_URL / EMBEDDING_MODEL，"
            f"或指定 EMBEDDING_PROTOCOL=openai|minimax")

    # ---------- 对外 ----------
    def documents(self, texts, on_progress=None):
        out = []
        texts = [t[:MAX_CHARS] for t in texts]
        for i in range(0, len(texts), BATCH):
            out.extend(self._post(texts[i:i + BATCH], "db"))
            if on_progress:
                on_progress(min(i + BATCH, len(texts)), len(texts))
        arr = normalize(np.array(out, dtype=np.float32))
        self.dim = int(arr.shape[1])
        return arr

    def query(self, text: str) -> np.ndarray:
        v = np.array(self._post([text[:MAX_CHARS]], "query")[0], dtype=np.float32)
        return normalize(v)


class BgeEmbedder:
    """本机模型，完全离线。要多装 torch（2GB）和权重（391MB）。"""
    name = BGE_MODEL

    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(BGE_MODEL, device=device)
        self.model.max_seq_length = 512
        self.dim = self.model.get_sentence_embedding_dimension()

    def documents(self, texts, on_progress=None):
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def query(self, text: str) -> np.ndarray:
        return self.model.encode([BGE_QUERY_PREFIX + text], normalize_embeddings=True,
                                 convert_to_numpy=True)[0].astype(np.float32)


def get(prefer: str = "") -> object:
    """prefer 留空时按环境挑：有 key 走接口，没有就退本机 bge。"""
    prefer = prefer or os.environ.get("EMBEDDING_BACKEND", "")
    if prefer == "bge" or prefer.startswith("BAAI/"):
        return BgeEmbedder()
    if prefer and prefer != "api":
        return ApiEmbedder()                                 # 索引里记的模型名，照着用
    return ApiEmbedder() if has_key() else BgeEmbedder()
