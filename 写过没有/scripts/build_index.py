#!/usr/bin/env python3
"""把检索块向量化，产出 vectors.npy + index.json。

向量化走哪家由 .env 决定，OpenAI 兼容协议和 MiniMax 私有协议都认，
`EMBEDDING_BACKEND=bge` 可切回本机模型。细节见 embedding.py 顶部。
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedding                                  # noqa: E402

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DATA = Path(__file__).resolve().parent.parent / "data"
MAX_CHARS = 1500

# 系列前缀会稀释标题的语义向量：《漫画：儿童心理学系列 ⑤ · 我讲了三遍你还不会，那是我讲错了》
# 整条编码，跟「孩子学不会是我讲错了」只有 0.56 的相似度，剥掉前缀就上去了。
# 《稳定性五件套-熔断的原理和实现》对「熔断的原理」只有 0.34，同一个原因。
SERIES_PREFIX = re.compile(
    r"^(?:漫画|视频|音频)\s*[:：]\s*"                       # 漫画：
    r"|^【[^】]{1,12}】\s*"                                  # 【硬核】
    r"|^[^·:：\-—]{1,12}系列\s*[①-⑳0-9]*\s*[·・]\s*"      # 儿童心理学系列 ⑤ ·
    r"|^[^\-—:：]{2,10}(?:五件套|三件套|之路|系列)\s*[\-—]\s*"  # 稳定性五件套-
)


def title_for_embedding(title: str) -> str:
    """剥掉系列前缀再编码。剥完为空就退回原标题，别把标题剥没了。"""
    stripped = title
    for _ in range(3):                                    # 前缀可能叠着，最多剥三层
        new = SERIES_PREFIX.sub("", stripped, count=1).strip()
        if new == stripped:
            break
        stripped = new
    return stripped if len(stripped) >= 4 else title                                  # 接口对单条长度有上限，超了整批会被拒


def main():
    chunks = [json.loads(l) for l in (DATA / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    embedder = embedding.get()
    print(f"待向量化 {len(chunks)} 块，用 {embedder.name}")

    t0 = time.time()

    def progress(done, total):
        print(f"\r  {done}/{total}  {time.time() - t0:.0f}s", end="", flush=True)

    vecs = embedder.documents([c["embed_text"][:MAX_CHARS] for c in chunks], on_progress=progress)
    print(f"\r向量化完成 {time.time() - t0:.1f}s，形状 {vecs.shape}          ")

    # 标题单独编码一份：判断「有没有一篇专门写这个」全靠它，见 server.py 的 title_match
    docs = json.loads((DATA / "docs.json").read_text(encoding="utf-8"))["docs"]
    doc_ids = list(docs)
    print(f"再编码 {len(doc_ids)} 个标题…")
    tvecs = embedder.documents([title_for_embedding(docs[i]["title"]) for i in doc_ids])

    np.save(DATA / "vectors.npy", vecs)
    np.save(DATA / "title_vectors.npy", tvecs)
    (DATA / "index.json").write_text(json.dumps({
        "model": embedder.name,
        "dim": int(vecs.shape[1]),
        "count": len(chunks),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "title_doc_ids": doc_ids,
        "protocol": getattr(embedder, "protocol", ""),   # 记下探测结果，服务启动就不用再试一次
        "chunks": [{k: c[k] for k in ("chunk_id", "doc_id", "seq", "heading", "text")} for c in chunks],
    }, ensure_ascii=False), encoding="utf-8")
    print(f"索引已写入 {DATA}")


if __name__ == "__main__":
    main()
