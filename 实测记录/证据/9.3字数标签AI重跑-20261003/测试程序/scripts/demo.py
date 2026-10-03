#!/usr/bin/env python3
"""课堂演示脚本。每条命令当场跑，输出就是课件里那张表。

    .venv/bin/python scripts/demo.py 两路     缓存救不了性能问题
    .venv/bin/python scripts/demo.py 归一化   马拉松跑鞋怎么选
    .venv/bin/python scripts/demo.py 判定     Redis 缓存穿透
    .venv/bin/python scripts/demo.py 飘移     维果斯基
    .venv/bin/python scripts/demo.py 分离度
    .venv/bin/python scripts/demo.py 留一
    .venv/bin/python scripts/demo.py 相似     配偶生育的，享有护理假 15 天 | 陪产假多少天 | 生日假怎么请
        （书第 6 章 6.6「你照着做」：第一句是原文，后面每句跟它比，用 | 分开）
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server as S
import embedding

HI, LO, GATE = 0.60, 0.30, 0.70


def _load():
    sr = S.Searcher()
    sr.model                                    # 提前把 embedder 备好
    return sr


def _top(sr, q, *, use_vec=True, use_lex=True, relative_norm=False, gate=True):
    """按开关跑一次完整检索，返回第一名。开关就是课上要演示的那几个变量。"""
    qvec = sr.embed_query(q)
    sims = (sr.vectors @ qvec) if use_vec else np.zeros(len(sr.chunks), dtype=np.float32)
    raw = sr.bm25(q)
    if not use_lex:
        lex = np.zeros_like(sims)
    elif relative_norm:
        lex = raw / raw.max() if raw.max() > 0 else raw      # 旧写法：除以本次最大值
    else:
        lex = S.bm25_norm(raw)                               # 现在的写法：饱和归一化
    fused = (S.VEC_WEIGHT * sims + S.BM25_WEIGHT * lex) if (use_vec and use_lex) else (sims + lex)

    terms = set(S.bigrams(q))
    grp = defaultdict(list)
    for i in np.argsort(-fused)[:400]:
        grp[sr.chunks[i]["doc_id"]].append(float(fused[i]))
    best = None
    for did, hs in grp.items():
        hs.sort(reverse=True)
        sc = hs[0] + S.MULTI_HIT_BONUS * min(len(hs) - 1, 3)
        if terms and use_lex:
            sc += S.TITLE_BONUS * len(terms & sr.title_terms.get(did, set())) / len(terms)
        tm = sr.title_match(did, terms, qvec)
        if best is None or sc > best[0]:
            best = (sc, tm, len(hs), sr.docs[did]["title"])
    sc, tm, n, title = best
    if sc >= HI and (tm >= GATE or not gate):
        v = "写过"
    elif sc >= LO:
        v = "擦边"
    else:
        v = "没写过"
    return {"score": sc, "title_match": tm, "hits": n, "title": title, "verdict": v}


def cmd_两路(sr, q):
    print(f"\n问题：{q}\n")
    print(f"{'配置':16s} {'分数':>6s} {'判定':6s}  检索到的第一篇")
    print("─" * 78)
    for name, kw in (("混合检索", {}), ("只用向量", {"use_lex": False}), ("只用字面 BM25", {"use_vec": False})):
        r = _top(sr, q, **kw)
        print(f"{name:16s} {r['score']:6.2f} {r['verdict']:6s}  {r['title'][:34]}")


def cmd_归一化(sr, q):
    print(f"\n问题：{q}\n")
    print(f"{'BM25 归一化方式':22s} {'分数':>6s} {'判定':6s}  第一篇")
    print("─" * 84)
    for name, rel in (("饱和 lex/(lex+12)  现在", False), ("除以本次最大值    旧写法", True)):
        r = _top(sr, q, relative_norm=rel)
        print(f"{name:22s} {r['score']:6.2f} {r['verdict']:6s}  {r['title'][:32]}")


def cmd_判定(sr, q):
    r = _top(sr, q)
    ng = _top(sr, q, gate=False)
    print(f"\n问题：{q}")
    print(f"\n  第一名        《{r['title']}》")
    print(f"  检索分数      {r['score']:.2f}   {'≥ 0.60 ✓' if r['score'] >= HI else '< 0.60'}")
    print(f"  标题匹配      {r['title_match']:.2f}   {'≥ 0.70 ✓ 有一篇标题就在讲这个' if r['title_match'] >= GATE else '< 0.70 ✗ 没有一篇标题在讲这个'}")
    print(f"  命中块数      {r['hits']}")
    print(f"\n  结论          {r['verdict']}")
    if r["verdict"] != ng["verdict"]:
        print(f"  （只看分数不看标题的话会判成「{ng['verdict']}」——标题这一条就是拦这个的）")


def cmd_飘移(sr, word=None):
    """低频专名在向量空间里落在哪儿。课上用来解释纯向量为什么会选错第一名。"""
    word = word or "维果斯基"
    e = sr.model
    ref = {
        "维果斯基": [("Vygotsky", "同一个人的英文原名"), ("维戈茨基", "同一个人的另一种译名"),
                  ("最近发展区", "他最著名的理论"), ("儿童心理学", "他的领域"),
                  ("数据建模", "毫不相干"), ("本体论", "毫不相干")],
    }.get(word, [("相对论", "参照"), ("数据建模", "毫不相干")])

    print(f"\n「{word}」跟这些词的余弦相似度\n")
    qv = e.query(word)
    for other, note in ref:
        print(f"   {other:12s} {float(qv @ e.query(other)):+.3f}   {note}")

    print("\n换几个高频专名对照（人名 ↔ 他的代表作）\n")
    for name, work in (("爱因斯坦", "相对论"), ("鲁迅", "杂文"), (word, ref[2][0] if len(ref) > 2 else "相对论")):
        print(f"   {name:8s} ↔ {work:10s} {float(e.query(name) @ e.query(work)):+.3f}")
    print("\n  高频人名跟自己的代表作能到 0.5~0.7，低频专名连 0.15 都不到。")
    print("  这不是模型不会处理人名，是这个名字在训练语料里出现得太少。")


def cmd_分离度(sr, _=None):
    POS = ["杀猪盘", "CSRF", "江南七怪", "线程池参数怎么配", "分布式事务怎么做", "省 token",
           "缓存救不了性能问题", "AI 写的代码要不要逐行看", "长腿蟹", "达梦数据库"]
    NEG_FAR = ["怎么挑选羽绒服", "猫藓怎么治", "插花入门教程", "螺蛳粉哪家好吃", "怎么挑西瓜"]
    NEG_HARD = ["Redis 缓存穿透", "MySQL 索引什么时候失效", "线程死锁怎么排查"]
    for name, qs in (("确定写过", POS), ("完全没写过", NEG_FAR), ("难负例·沾边但没专门写", NEG_HARD)):
        xs = [_top(sr, q) for q in qs]
        s = [x["score"] for x in xs]
        t = [x["title_match"] for x in xs]
        print(f"\n【{name}】{len(qs)} 个")
        print(f"  检索分数  {min(s):.2f} ~ {max(s):.2f}")
        print(f"  标题匹配  {min(t):.2f} ~ {max(t):.2f}")
        print(f"  判定分布  " + "、".join(f"{v}×{sum(1 for x in xs if x['verdict']==v)}"
                                       for v in ("写过", "擦边", "没写过")
                                       if sum(1 for x in xs if x["verdict"] == v)))


def cmd_留一(sr, _=None):
    truth = json.loads(Path("/tmp/holdout_truth.json").read_text()) if Path("/tmp/holdout_truth.json").exists() else {}
    if not truth:
        print("没有 /tmp/holdout_truth.json，跳过"); return
    hard = 0
    print(f"\n{'问题':24s} {'真值':6s} {'分数':>5s} {'标题':>5s} {'判定':6s}")
    print("─" * 62)
    for q, real in truth.items():
        r = _top(sr, q)
        bad = (real and r["verdict"] == "没写过") or ((not real) and r["verdict"] == "写过")
        hard += bad
        print(f"{q[:22]:24s} {'写过' if real else '没写过':6s} {r['score']:5.2f} "
              f"{r['title_match']:5.2f} {r['verdict']:6s} {'✗' if bad else ''}")
    print(f"\n判错 {hard}/{len(truth)}")


def cmd_相似(sr, arg):
    """几句话两两跟第一句比相似度。书 6.6 让读者自己跑的那一步：换个说法还像不像、不沾边的有多不像。"""
    if not arg or "|" not in arg:
        print("用法：demo.py 相似  原文一句 | 换个说法 | 再一句 …"); return
    ss = [x.strip() for x in arg.split("|") if x.strip()]
    V = np.asarray(sr.model.documents(ss)); V = V / np.linalg.norm(V, axis=1, keepdims=True)
    print(f"原文：{ss[0]}\n每句有 {V.shape[1]} 个数，长度都缩放成 1.0\n")
    for s_, v in zip(ss[1:], V[1:]):
        print(f"{float(v @ V[0]):.3f}  {s_}")
    print("\n1 是一模一样，0 是毫不相干；这就是页面上那个“匹配 N%”背后的数。")


CMDS = {"两路": cmd_两路, "相似": cmd_相似, "归一化": cmd_归一化, "判定": cmd_判定, "飘移": cmd_飘移,
        "分离度": cmd_分离度, "留一": cmd_留一}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print(__doc__); raise SystemExit(1)
    name = sys.argv[1]
    arg = " ".join(sys.argv[2:]) or None
    CMDS[name](_load(), arg)
