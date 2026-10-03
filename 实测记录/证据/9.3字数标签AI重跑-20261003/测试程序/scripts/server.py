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
from datetime import datetime
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

# 问一句每分钟最多几次。超了回 429，不再作答。检索不走这道。
# 看见超限那次用 ASK_PER_MIN=2 起服务；平时 20，不然这页问两句就停。
ASK_LOCK = threading.Lock()
ASK_HITS = []
ASK_WINDOW = 60
ASK_PER_MIN = int(os.environ.get("ASK_PER_MIN", "20"))


def ask_over_limit() -> bool:
    now = time.monotonic()
    with ASK_LOCK:
        ASK_HITS[:] = [t for t in ASK_HITS if now - t < ASK_WINDOW]
        if len(ASK_HITS) >= ASK_PER_MIN:
            return True
        ASK_HITS.append(now)
        return False

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("RAG_DATA_DIR", str(ROOT / "data")))
FILES_DIR = Path(os.environ.get("RAG_FILES_DIR", str(ROOT / "我的文件")))
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


# ---------- 设置 ----------
# 页面右上角齿轮能调的东西全在这儿，契约见 docs/设置契约.md。
SETTINGS_FILE = DATA / "settings.json"


def env_num(name: str, fallback):
    """出厂值允许被环境变量盖一层：换成设置面板之前，.env 里已经写过这些名字的人还在，
    那一行不能因为改版突然失效。写歪了就当没写过——import 期抛异常会把整个服务带走。"""
    try:
        return type(fallback)(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return fallback


# 出厂值只有这一份，GET /api/settings 的 defaults 段直接发它。前端的「恢复默认」也读这份，
# 不许两边各抄一遍：抄了就会有一天服务端改了数、页面上的「默认」还是老的，而且谁也不会发现。
DEFAULTS = {
    # appearance 服务端一个都不用，只负责存和发；贴到 CSS 变量上是前端的事
    "appearance": {
        "theme": "auto",
        # 2026-09-19 换成 test81（公司产品测试环境）那套令牌：作者说「我看我们 test81 的
        # 比这个还好看些呢」。它赢在有纪律——冷灰中性色、8px 圆角、4/8/12/16 的间距刻度。
        # 她之前说的「框太大太圆」正好落在圆角上，8 对 14。
        "accent": "#419eff",
        "radius": 8,
        "density": "normal",
        "font_size": 14,
        "max_width": 860,
    },
    "search": {
        "limit": 20,
        # 第二段摘要的门槛：分数低于第一段这个比例就不显示（2026-09-19 加）。
        # 0.9 是拿五个真查询测出来的，不是拍的：
        #   年假能不能留到明年 0.99、试用期年假 0.97 —— 第二段都是同一话题的另一段，有用
        #   护理假有几天      0.78 —— 第二段是「几条常被问到的」，只蹭到了「护理假」三个字，没用
        # 有用的挤在 0.97 以上，没用的掉到 0.78，中间空一大段，所以门槛放 0.9。
        # 先定过 0.75，太松，那条没用的照样过了。
        "second_snippet_ratio": env_num("SECOND_SNIPPET_RATIO", 0.9),
        "snippet_width": 260,
        # 下面五个服务端不读：判定档和高亮上限都在前端算。放一起是因为它们是同一件事的旋钮，
        # 调松紧的人不该被「这个归谁管」绊住。
        "verdict_yes": 0.6,
        "verdict_maybe": 0.3,
        "title_match_min": 0.7,
        "highlight_cap": 6,
        "fulltext_highlight_cap": 40,
    },
    # 这个工具在书里是一步一步长出来的。第 1 章那几张截图拍的时候还没有「我的文件」，
    # 第 5 章才有。要重截第 1 章的图就得让页面退回那一版——改代码退、改完再改回来，
    # 来回折腾还容易忘记改回去。做成开关，截图前拨一下就行。
    # 读者也一样：把开关一个个打开，就看得见这个工具是怎么长起来的。
    "features": {
        "kind_article": True,
        "kind_comic": True,
        "kind_file": True,
        "order_switch": True,
        "answer": True,
        "search_only_btn": True,
        # 2026-09-19 作者定：出厂关。这排胶囊是给第一次来的人看的，
        # 但她拿这个页面截书里的图，每张都得先关掉它——默认关，要用的时候再开。
        "examples": False,
        "verdict": True,
        "score_tag": True,
        "hit_count_tag": True,
        "thumbs": True,
        "copy_actions": True,
        "fulltext": True,
        "gear": True,
        # 这几行 2026-09-19 作者定：默认关。她的原话是「页面上有很多多余的解释，
        # 比如多少文章什么的都不要」，跟着点名了副标题和那句大字，「都多余」。
        # 「611 篇文章 + 60 篇漫画」在页面上出现两遍，「按意思搜」也出现两遍。
        # 但不能直接删：第 5 章那八张截图拍着统计行，图 7-10 的首页空态拍着大字和副行，
        # 删了就重截不出来。所以留开关，截图前打开。
        "subtitle": False,
        "hero": False,
        "stats_line": False,
        # 2026-09-19 作者问「向量模型 embo-01 · 索引建于 … 这个小字需要吗」——不需要。
        # 模型名读者不认识，摆着不说明任何事；那个时间戳唯一有用的场合已经改成
        # 过期时自己冒出来了（见 index_stale）。所以这行默认关。
        "foot_note": False,
        # 第 1 章截图里发送键是个文字胶囊，7.7 讲的正是它换成圆箭头这件事，两版都得留。
        "send_button": "arrow",
    },
}

# 每个键认什么值。类型和范围抄自契约那两张表，改范围先改契约。
SPEC = {
    "appearance": {
        "theme": ("enum", ("auto", "light", "dark")),
        "accent": ("hex", None),
        "radius": ("int", (0, 24)),
        "density": ("enum", ("loose", "normal", "compact")),
        "font_size": ("int", (13, 18)),
        "max_width": ("int", (640, 1200)),
    },
    "search": {
        "limit": ("int", (5, 50)),
        "second_snippet_ratio": ("float", (0.0, 1.0)),
        "snippet_width": ("int", (120, 400)),
        "verdict_yes": ("float", (0.0, 2.0)),
        "verdict_maybe": ("float", (0.0, 2.0)),
        "title_match_min": ("float", (0.0, 1.0)),
        "highlight_cap": ("int", (1, 40)),
        "fulltext_highlight_cap": ("int", (1, 200)),
    },
    "features": {
        "kind_article": ("bool", None),
        "kind_comic": ("bool", None),
        "kind_file": ("bool", None),
        "order_switch": ("bool", None),
        "answer": ("bool", None),
        "search_only_btn": ("bool", None),
        "examples": ("bool", None),
        "verdict": ("bool", None),
        "score_tag": ("bool", None),
        "hit_count_tag": ("bool", None),
        "thumbs": ("bool", None),
        "copy_actions": ("bool", None),
        "fulltext": ("bool", None),
        "gear": ("bool", None),
        "subtitle": ("bool", None),
        "hero": ("bool", None),
        "stats_line": ("bool", None),
        "foot_note": ("bool", None),
        "send_button": ("enum", ("arrow", "text")),
    },
}

# 取色控件给的是 6 位，手写的人爱写 3 位，两种都收
HEX_COLOR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")

_settings_lock = threading.Lock()


def check(section: str, key: str, value):
    """回 (认不认, 收下的值)。

    bool 要单独挡：Python 里 isinstance(True, int) 为真，不挡的话 radius 能存成 true，
    发到前端就是 CSS 里一个 border-radius: truepx，页面上只看到圆角没了，查不到是谁干的。
    """
    kind, arg = SPEC[section][key]
    if kind == "bool":
        return isinstance(value, bool), value
    if kind == "enum":
        return value in arg, value
    if kind == "hex":
        return bool(isinstance(value, str) and HEX_COLOR.match(value)), value
    if isinstance(value, bool):
        return False, value
    if kind == "int":
        if not isinstance(value, int):
            return False, value
    elif not isinstance(value, (int, float)):
        return False, value
    return arg[0] <= value <= arg[1], float(value) if kind == "float" else value


def load_settings() -> dict:
    """当前设置 = 出厂值盖上盘里存的那份。

    盘里没有、是空的、JSON 烂了、少几个键、被手改成离谱的数，一律当那一项没写过。
    设置是个附属功能，它坏掉不该让检索跟着打不开——读者从 GitHub 拉下来第一次跑，
    data/ 里本来就没有这个文件。
    """
    saved = {}
    try:
        raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        saved = raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        pass
    out = {}
    for section, factory in DEFAULTS.items():
        merged = dict(factory)
        got = saved.get(section)
        for key, value in (got.items() if isinstance(got, dict) else ()):
            if key not in factory:
                continue
            ok, value = check(section, key, value)
            if ok:
                merged[key] = value
        out[section] = merged
    return out


def save_settings(patch: dict):
    """两层浅合并，回 (合并后的全量, 没收下的键名)。

    一个键写坏不退回整次保存：面板上是改一项发一次，整批退回等于用户点了没反应。
    读-改-写整段上锁，因为服务是 ThreadingHTTPServer——拖滑块会连着发好几个 POST，
    不锁的话后发的那次读到的是改之前的全量，先发的那次被无声吞掉。
    先写临时文件再改名，是为了别让别的线程读到写了一半的 JSON：那会儿它就成了「文件坏了」，
    整页悄悄跳回出厂值，比报错难查得多。
    """
    ignored = []
    with _settings_lock:
        merged = load_settings()
        for section, part in patch.items():
            if section not in SPEC or not isinstance(part, dict):
                ignored.append(section)
                continue
            for key, value in part.items():
                ok, value = check(section, key, value) if key in SPEC[section] else (False, value)
                if ok:
                    merged[section][key] = value
                else:
                    ignored.append(f"{section}.{key}")
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_FILE.with_name(SETTINGS_FILE.name + ".tmp")
        tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(SETTINGS_FILE)
    return merged, ignored


def reset_settings() -> dict:
    """恢复默认是把文件删掉，不是往里写一份出厂值的副本。
    写副本的话，以后改了 DEFAULTS，点过「恢复默认」的那些机器会被旧副本一直钉在老值上。"""
    with _settings_lock:
        SETTINGS_FILE.unlink(missing_ok=True)
    return load_settings()


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
    def search(self, query: str, limit: int = None, order: str = "relevance", kind: str = "all",
               mode: str = "hybrid", with_chunks: bool = False):
        """三种检索模式，课上要当场切着看差别：

        hybrid  两路融合，正常用的就是它
        vector  「朴素向量 RAG」：只有余弦，关掉 BM25 和标题加成
        lex     只有 BM25 字面路，一次接口都不调（没配 key 时系统也自动落到这儿）
        """
        query = (query or "").strip()
        if not query:
            return []

        # 每次查询现读一次设置：面板上调完松紧，不重启服务下一次搜索就得是新的。
        # 这个文件几百字节，比下面任何一路都便宜。limit 没人指定才按配置来——
        # 「问一句」那条要的是 answer.CONTEXT_DOCS，不能被页面上的「一次出几条」带跑。
        cfg = load_settings()["search"]
        if limit is None:
            limit = cfg["limit"]

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
                "has_text": bool(self.fulltext.get(doc_id)),
                "chars": len(self.fulltext.get(doc_id, "").strip()) if doc.get("kind") == "file" else None,
                "version": (re.search(r"(?:样例版本|版本)[：:]\s*([0-9-]+)", self.fulltext.get(doc_id, "")[:250]).group(1) if re.search(r"(?:样例版本|版本)[：:]\s*([0-9-]+)", self.fulltext.get(doc_id, "")[:250]) else ""),
                "read_num": doc.get("read_num"),
                "topic": doc.get("topic", ""),
                "score": round(score, 4),
                "title_match": round(tmatch, 4),
                "vec_score": round(hits[0][1], 4),
                "lex_score": round(hits[0][2], 4),
                "hit_count": len(hits),
                # 第二段只在它跟第一段够接近的时候才给。2026-09-19 作者看到
                # 「护理假有几天」的结果里，第二段是「几条常被问到的」——它被捞进来
                # 只因为正文别处也有「护理假」三个字，对「这篇是不是我要的」毫无帮助，
                # 白白让结果页长一截。摘要的活是让人判断要不要点进去，不是把内容读给他听。
                "snippets": [{"heading": h[3]["heading"],
                              "text": snippet(h[3]["text"], q_terms, cfg["snippet_width"]),
                              "score": round(h[0], 4)}
                             for h in hits[:2]
                             if h is hits[0]
                             or h[0] >= hits[0][0] * cfg["second_snippet_ratio"]],
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


def missing_indexed_file(results) -> str:
    """索引里还有、磁盘上已经没有的「我的文件」。有就回文件名，没有回空串。

    问一句读的是建索引时拷进去的那份。文件改了名、索引没重建，检索照样命中，
    模型还会按旧片段作答。这一问先看磁盘：源文件不在，就别再假装答完了。
    """
    if not results:
        return ""
    doc_id = results[0].get("id") or ""
    if not str(doc_id).startswith("file:"):
        return ""
    name = str(doc_id).split(":", 1)[1]
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return ""
    root = FILES_DIR.resolve()
    path = (root / name).resolve()
    if not str(path).startswith(str(root)) or path.is_file():
        return ""
    return name


def ingest_preview() -> dict:
    """「我的文件」里现在有什么。只读磁盘，不重建索引。

    读进库之前先把这一页摆出来等人点头。没点头，库不动。
    """
    root = FILES_DIR.resolve()
    rows = []
    if root.is_dir():
        for p in sorted(root.iterdir()):
            if not p.is_file() or p.name.startswith("."):
                continue
            if p.suffix.lower() not in (".md", ".txt"):
                continue
            if not str(p.resolve()).startswith(str(root)):
                continue
            text = p.read_text(encoding="utf-8")
            title = p.stem
            m = re.match(r"^#\s+(.+)$", text, flags=re.M)
            if m:
                title = m.group(1).strip()
            basis = ""
            for line in text.splitlines():
                line = line.strip()
                if "结转" in line:
                    basis = re.sub(r"^\d+\.\s*", "", line).strip()
                    break
            rows.append({
                "title": title,
                "name": p.name,
                "date": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d"),
                "chars": len(text.strip()),
                "basis": basis,
            })
    return {"count": len(rows), "files": rows, "waiting": True}


def index_stale(built_at: str) -> str:
    """索引比源文件旧了就回一句话，没旧回空串。

    2026-09-19：页脚原来常年挂一行「向量模型 X · 索引建于 Y」。作者问「这个小字需要吗」——
    不需要。那个时间戳唯一真正有用的场合，是你往 我的文件/ 丢了一份新文件、忘了跑
    ./refresh.sh，然后搜不到、一脸问号。平时占一行当装饰，等于把一个警告伪装成签名，
    真出事的时候反倒没人看。所以改成只在真过期时才冒出来。

    只看 我的文件/ ——公众号那批是脚本抓的，作者不会手动往里放东西。
    """
    try:
        built = datetime.fromisoformat(built_at).timestamp()
    except Exception:
        return ""
    newer = [q.name for q in FILES_DIR.rglob("*")
             if q.is_file() and not q.name.startswith(".") and q.stat().st_mtime > built]
    if not newer:
        return ""
    head = "、".join(newer[:2]) + ("…" if len(newer) > 2 else "")
    return f"{head} 比索引新，还搜不到。跑一下 ./refresh.sh"


def snippet(text: str, q_terms, width: int = None) -> str:
    """截取片段：尽量从命中词附近开始，别从半句话中间起。

    查询词经过 bigrams() 已经是小写，正文要保留原文大小写给人看，
    定位必须大小写不敏感——不然搜 mqtt 对不上正文里的 MQTT，
    窗口会从文首切，命中词落在 260 字之外，摘要变成「使用 M…」。
    """
    # 宽度不再写死在签名里：search() 一次读好配置传进来，单独调它的（试脚本、REPL）现读一次
    if width is None:
        width = load_settings()["search"]["snippet_width"]
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
        if ask_over_limit():
            return self._json({"error": "太频繁"}, 429)
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
            if kind == "file" and missing_indexed_file(results):
                push({"type": "missing_file"})
                return push({"type": "done"})
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
                "stale": index_stale(self.searcher.index["built_at"]),
                "started_at": STARTED_AT,
                "earliest": min((d["date"] for d in self.searcher.docs.values() if d["date"]), default=""),
                "latest": max((d["date"] for d in self.searcher.docs.values() if d["date"]), default=""),
            })
        if path == "/api/settings":
            return self._json({**load_settings(), "defaults": DEFAULTS})
        if path == "/api/ingest/preview":
            return self._json(ingest_preview())
        if path == "/api/ask":
            return self._ask((qs.get("q") or [""])[0],
                             mode=(qs.get("mode") or ["hybrid"])[0],
                             kind=(qs.get("kind") or ["all"])[0])
        if path == "/api/search":
            q = (qs.get("q") or [""])[0]
            want = (qs.get("limit") or [""])[0].strip()
            try:
                results = self.searcher.search(
                    q,
                    # 不带 limit 就按设置里的「一次出几条」来；带了还是最多 50 条，
                    # 免得有人手敲 ?limit=5000 把一整页结果压进浏览器
                    limit=min(int(want), 50) if want else None,
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

    def do_POST(self):
        # 不管认不认这个路径，先把请求体读干净：protocol_version 是 HTTP/1.1，连接要复用，
        # body 剩在管子里的话下一个请求会从它中间开始解析，报出来的错跟真因八竿子打不着。
        path = urlparse(self.path).path
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))

        if path == "/api/settings/reset":
            return self._json({**reset_settings(), "defaults": DEFAULTS})
        if path == "/api/settings":
            try:
                patch = json.loads(body or b"{}")
            except ValueError:
                return self._json({"ok": False, "error": "请求体不是 JSON"}, 400)
            if not isinstance(patch, dict):
                return self._json({"ok": False, "error": "请求体要是个对象"}, 400)
            try:
                merged, ignored = save_settings(patch)
            except OSError as exc:            # 目录只读、盘满：告诉前端存不下，别装作存上了
                return self._json({"ok": False, "error": str(exc)}, 500)
            out = {"ok": True, "settings": merged}
            if ignored:                       # 全收下时不出这个字段，省得前端把空数组当成有东西被扔
                out["ignored"] = ignored
            return self._json(out)
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
