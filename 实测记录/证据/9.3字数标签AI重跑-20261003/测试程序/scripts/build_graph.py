#!/usr/bin/env python3
"""从已发布文章抽出知识图谱：文中互链 + 标题语义近邻，再按题义聚成主题簇。

产出 data/graph.json，页面「知识图谱」视图直接画，不在浏览器里现算。
"""
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest import normalize  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"

MP_LINK = re.compile(r"https?://mp\.weixin\.qq\.com/s/([A-Za-z0-9_-]+)")
BOOK = re.compile(r"《([^》]{4,40})》")
SERIES_PREFIX = re.compile(
    r"^(?:漫画|视频|音频)\s*[:：]\s*"
    r"|^【[^】]{1,12}】\s*"
    r"|^[^·:：\-—]{1,12}系列\s*[①-⑳0-9]*\s*[·・]\s*"
    r"|^[^\-—:：]{2,10}(?:五件套|三件套|之路|系列)\s*[\-—]\s*"
)

SIM_FLOOR = 0.64          # 标题余弦低于这个不连，避免全图糊成一团
SIM_TOPK = 3              # 每篇最多留几条语义边
KMEANS_ITERS = 18


def link_id(url: str) -> str:
    m = MP_LINK.search(url or "")
    return m.group(1) if m else ""


def short_title(title: str) -> str:
    t = title or ""
    for _ in range(3):
        n = SERIES_PREFIX.sub("", t, count=1).strip()
        if n == t:
            break
        t = n
    t = re.sub(r"^漫画\s*[:：]\s*", "", t).strip()
    return t if len(t) >= 4 else (title or "")


def kmeans(X: np.ndarray, k: int, rng):
    n = len(X)
    k = max(2, min(k, n))
    centers = X[rng.choice(n, k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(KMEANS_ITERS):
        sim = X @ centers.T
        labels = sim.argmax(axis=1)
        for i in range(k):
            m = labels == i
            if not m.any():
                centers[i] = X[rng.integers(0, n)]
                continue
            c = X[m].mean(axis=0)
            nrm = np.linalg.norm(c)
            centers[i] = c / nrm if nrm > 1e-8 else X[rng.integers(0, n)]
    return labels, centers


def community_label(members, docs, centroid, tvecs, id_pos):
    """簇名用离中心最近那篇的短标题。2-gram 拼出来会变成「公众众号」这种半截词。"""
    best, best_sim = members[0], -1.0
    for did in members:
        i = id_pos.get(did)
        if i is None or centroid is None:
            continue
        s = float(tvecs[i] @ centroid)
        if s > best_sim:
            best, best_sim = did, s
    t = short_title(docs[best]["title"])
    return t[:16] + ("…" if len(t) > 16 else "")


def extract_cites(docs, fulltext):
    by_link, by_title = {}, {}
    for did, d in docs.items():
        lid = link_id(d.get("link") or "")
        if lid:
            by_link[lid] = did
        key = normalize(d["title"])
        if key:
            by_title.setdefault(key, did)
        for aka in d.get("aka") or []:
            k = normalize(aka)
            if k and k not in by_title:
                by_title[k] = did

    edges = {}  # (src, dst) -> weight; cite 记 1
    for did, text in fulltext.items():
        if did not in docs:
            continue
        seen = set()
        for lid in MP_LINK.findall(text or ""):
            dst = by_link.get(lid)
            if dst and dst != did and dst not in seen:
                seen.add(dst)
                edges[(did, dst)] = "cite"
        for name in BOOK.findall(text or ""):
            dst = by_title.get(normalize(name))
            if dst and dst != did and dst not in seen:
                seen.add(dst)
                edges[(did, dst)] = "cite"
    return edges


def extract_similar(docs, tvecs, id_list):
    if tvecs is None or len(id_list) == 0:
        return {}
    sim = tvecs @ tvecs.T
    np.fill_diagonal(sim, -1)
    edges = {}
    for i, did in enumerate(id_list):
        idx = np.argsort(-sim[i])[:SIM_TOPK]
        for j in idx:
            if sim[i, j] < SIM_FLOOR:
                continue
            other = id_list[int(j)]
            a, b = (did, other) if did < other else (other, did)
            # 无向：同一对只留更高的分；cite 优先在 merge 时处理
            prev = edges.get((a, b))
            w = float(sim[i, j])
            if prev is None or w > prev:
                edges[(a, b)] = w
    return edges


def layout(communities, nodes):
    """主题簇围一圈，簇内文章再绕自己的中心。坐标落在 0–1000。"""
    n_c = max(len(communities), 1)
    for i, c in enumerate(communities):
        ang = 2 * math.pi * i / n_c - math.pi / 2
        ring = 330
        c["x"] = round(500 + ring * math.cos(ang), 1)
        c["y"] = round(500 + ring * math.sin(ang), 1)
    by_c = defaultdict(list)
    for n in nodes:
        by_c[n["community"]].append(n)
    for c in communities:
        group = by_c.get(c["id"], [])
        cx, cy = c["x"], c["y"]
        n = len(group)
        rad = 22 + 7.5 * math.sqrt(n)
        for j, node in enumerate(group):
            if n == 1:
                node["x"], node["y"] = cx, cy
                continue
            a = 2 * math.pi * j / n
            node["x"] = round(cx + rad * math.cos(a), 1)
            node["y"] = round(cy + rad * math.sin(a), 1)


def main():
    docs = json.loads((DATA / "docs.json").read_text(encoding="utf-8"))["docs"]
    fulltext = json.loads((DATA / "fulltext.json").read_text(encoding="utf-8"))
    index = json.loads((DATA / "index.json").read_text(encoding="utf-8"))
    id_list = index.get("title_doc_ids") or list(docs)
    tvecs = np.load(DATA / "title_vectors.npy") if (DATA / "title_vectors.npy").exists() else None
    if tvecs is not None and len(tvecs) != len(id_list):
        print(f"标题向量 {len(tvecs)} 和文章 {len(id_list)} 对不上，语义边先不加")
        tvecs = None

    id_pos = {d: i for i, d in enumerate(id_list)}
    cites = extract_cites(docs, fulltext)
    sims = extract_similar(docs, tvecs, id_list) if tvecs is not None else {}

    # 合并边：互链盖过语义，同一对只出一条
    edge_list = []
    seen_pair = set()
    for (src, dst), _ in cites.items():
        pair = (src, dst) if src < dst else (dst, src)
        seen_pair.add(pair)
        edge_list.append({"source": src, "target": dst, "type": "cite"})
    for (a, b), w in sims.items():
        if (a, b) in seen_pair:
            continue
        if (a, b) in cites or (b, a) in cites:
            continue
        edge_list.append({"source": a, "target": b, "type": "similar", "weight": round(w, 3)})

    k = min(16, max(8, len(docs) // 42))
    if tvecs is None:
        labels = np.zeros(len(id_list), dtype=np.int32)
        centers = None
        k = 1
    else:
        rng = np.random.default_rng(7)
        labels, centers = kmeans(tvecs, k, rng)

    groups = defaultdict(list)
    for did, lab in zip(id_list, labels):
        if did in docs:
            groups[int(lab)].append(did)

    communities = []
    remap = {}
    for old, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        cid = len(communities)
        remap[old] = cid
        centroid = centers[old] if centers is not None else None
        communities.append({
            "id": cid,
            "label": community_label(members, docs, centroid, tvecs, id_pos) if centroid is not None
                     else "文章",
            "size": len(members),
        })

    degree = Counter()
    for e in edge_list:
        degree[e["source"]] += 1
        degree[e["target"]] += 1

    nodes = []
    for did, d in docs.items():
        i = id_pos.get(did, 0)
        lab = int(labels[i]) if i < len(labels) else 0
        nodes.append({
            "id": did,
            "title": d["title"],
            "short": short_title(d["title"]),
            "date": d.get("date") or "",
            "kind": d.get("kind", "article"),
            "link": d.get("link") or "",
            "community": remap.get(lab, 0),
            "degree": degree.get(did, 0),
        })

    layout(communities, nodes)

    graph = {
        "built_at": index.get("built_at"),
        "nodes": nodes,
        "edges": edge_list,
        "communities": communities,
        "stats": {
            "articles": len(nodes),
            "cites": sum(1 for e in edge_list if e["type"] == "cite"),
            "similar": sum(1 for e in edge_list if e["type"] == "similar"),
            "communities": len(communities),
        },
    }
    (DATA / "graph.json").write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    st = graph["stats"]
    print(f"图谱 {st['articles']} 点 / 互链 {st['cites']} / 近邻 {st['similar']} / 主题簇 {st['communities']}")


if __name__ == "__main__":
    main()
