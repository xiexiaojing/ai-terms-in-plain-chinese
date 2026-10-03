#!/usr/bin/env python3
"""扫描本机公众号语料，产出文章清单 docs.json 与检索块 chunks.jsonl。

语料来源：
  1. wechat-article/已归档/*/article.md   —— 已发布（多数篇头部自带 mp 链接）
  2. wechat-article/制作中/*/article.md   —— 未发布草稿（无链接，标记 draft）
  3. feishu-claude-bot/data/gzh-history/all-articles.json —— 线上全量档案（631 篇，篇篇带 link）
     本地没有正文的，用标题 + 摘要建轻量条目，保证「搜得到、点得开」。
"""
import json
import re
import unicodedata
from difflib import SequenceMatcher
from datetime import datetime, timezone, timedelta
from pathlib import Path

WORKSPACE = Path("/Users/xiexiaojing/workspace")
ARCHIVE_DIR = WORKSPACE / "wechat-article" / "已归档"
DRAFT_DIR = WORKSPACE / "wechat-article" / "制作中"
COMIC_ARCHIVE = WORKSPACE / "wechat-comic" / "已归档"
COMIC_DRAFT = WORKSPACE / "wechat-comic" / "制作中"
# wechat-comic/禁止外发/ 有意不扫：那三篇含真实小区名、社工电话、中标单位，
# 目录 README 写死「一律不发布」。这系统的按钮是「一键复制走」，不该把它们放进这条流水线。
HISTORY_JSON = WORKSPACE / "feishu-claude-bot" / "data" / "gzh-history" / "all-articles.json"
SNAPSHOT_DIR = WORKSPACE / "feishu-claude-bot" / "data" / "gzh"   # 每日抓的快照，用来补建档之后新发的文章
OUT_DIR = Path(__file__).resolve().parent.parent / "data"
# 2026-09-17 书 6.2 动手做：读者把自己公司的制度、说明书放进这个目录，跟公众号文章分开成一个库。
# 认 .md / .txt / .docx；docx 只抽正文段落。文件名（去后缀）当标题，正文第一行是「# 标题」就用它。
MY_FILES_DIR = Path(__file__).resolve().parent.parent / "我的文件"

CST = timezone(timedelta(hours=8))
CHUNK_TARGET = 480      # 每块目标字数（中文字符）
CHUNK_MAX = 900         # 单段超长时的硬切上限

META_LINE = re.compile(r"^\s*[-*]\s*(正式标题|发布日期|链接|抓取方式|说明|公众号发布日期|题材)\s*[：:]\s*(.*)$")
# 「标题…」开头的小节一律是创作过程的候选清单，不是正文
DROP_SECTIONS = ("元信息", "题材", "标题", "封面", "配图", "发布记录", "改版说明",
                 "头条备稿说明", "备稿说明", "草稿箱", "复盘", "互链", "站外", "转化入口",
                 # 发文前的核稿清单和运营手法，读者从没见过，学员更不该在 demo 里看到
                 "发文前", "事实点", "fact_notes", "作者置顶评论", "置顶评论", "选题", "数据回顾")

# 小节名是写稿时的内部叫法，展示前换成人话
HEADING_ALIASES = {"摘要位": "摘要", "脉脉位": "摘要", "摘要位金句": "摘要",
                   "朋友圈 / 群转发语": "导语", "群转发语": "导语", "转发语": "导语",
                   "正文（公众号可发）": "", "正文": "", "头条版正文": ""}

# 只在她自己的稿子里出现、读者从没见过的东西：灌稿命令、后台 appmsgid、仓库路径、写作规范条目号。
# 行级过滤，不整段丢——漫画分镜里「桌牌『今日待办：把 curl 权限收了』」是画面道具文字，读者看得见，得留。
INTERNAL_LINE = re.compile(
    r"mp_draft|appmsgid|\.venv|openspec|feishu-claude-bot|wechat-article/|wechat-comic/|vip-course/"
    r"|规范\s*§|§\s*\d|ledger|静姐确认|灌进草稿箱|灌入\s*`?\d{6,}")


def normalize(text: str) -> str:
    """标题规范化：全角转半角、去空白与标点，用于跨来源匹配。"""
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"[\s　]+", "", text)
    text = re.sub(r"[·・,，.。:：;；!！?？、\"'“”‘’()（）\[\]【】<>《》~～\-—_/\\|*#]", "", text)
    return text.lower()


def fuzzy_match(title: str, hist_index, body: str = ""):
    """标题与线上档案对不上时的兜底匹配。

    「三言丨…」这类标题成批雷同，只看相似度必然配错，所以要过三关：
    分数够高、和第二名拉得开、或者线上摘要的开头确实出现在本地正文里。
    宁可留空也不配错——引用到别人那篇比没有链接更糟。
    """
    key = normalize(title)
    if len(key) < 8:
        return None
    scored = []
    for cand_key, art in hist_index.items():
        if len(cand_key) < 8:
            continue
        if key in cand_key or cand_key in key:
            score = min(len(key), len(cand_key)) / max(len(key), len(cand_key))
        else:
            score = SequenceMatcher(None, key, cand_key).ratio()
        scored.append((score, cand_key, art))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    best_score, _, best = scored[0]
    if best_score < 0.86:
        return None
    second = scored[1][0] if len(scored) > 1 else 0.0

    body_key = normalize(body)[:4000]
    digest_key = normalize(best.get("digest") or "")[:16]
    digest_hit = bool(digest_key) and digest_key in body_key
    if digest_hit:
        return best, best_score, "fuzzy+digest"
    if best_score - second >= 0.04:
        return best, best_score, "fuzzy"
    return None


def load_history():
    """线上文章总表 → {规范化标题: 记录}，以及原始列表。

    全量档案（631 篇，2026-08-18 建）+ 每日快照。档案是死的，快照是每天更新的，
    只读档案会让最近发的文章全都搜不到链接。
    """
    by_link = {}
    if HISTORY_JSON.exists():
        for a in json.loads(HISTORY_JSON.read_text(encoding="utf-8")).get("articles", []):
            if a.get("link"):
                by_link[a["link"]] = a
    for snap in sorted(SNAPSHOT_DIR.glob("*.json")) if SNAPSHOT_DIR.exists() else []:
        try:
            data = json.loads(snap.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        arts = data.get("articles") if isinstance(data, dict) else data
        if not isinstance(arts, list):
            continue
        for a in arts:
            if isinstance(a, dict) and a.get("link"):
                old = by_link.get(a["link"])
                by_link[a["link"]] = {**old, **{k: v for k, v in a.items() if v is not None}} if old else a

    arts = list(by_link.values())
    index = {}
    for a in arts:
        key = normalize(a.get("title", ""))
        if key and key not in index:
            index[key] = a

    # 副标题索引：线上标题常是「正标题：系列名 · 第几篇」，而本地稿名只写冒号后那半截
    # （本地「省 token 故事 · 一字千金」＝线上《在开头插了个时间戳，后面全部作废：省 token 故事 · 一字千金》）。
    # 一段被多篇共用（比如光写「省 token 故事」）就作废，只留能唯一定位一篇的段。
    seg_index = {}
    for a in arts:
        for seg in re.split(r"[：:｜|—]", a.get("title", "")):
            key = normalize(seg)
            if len(key) < 6 or key in index:
                continue
            if key in seg_index and seg_index[key] is not a:
                seg_index[key] = None
            else:
                seg_index[key] = a
    index.update({k: v for k, v in seg_index.items() if v is not None})
    return index, arts


# 稿子里夹着写给自己看的编辑说明（「2026-08-17 增长改版：原标题是索取姿态…」「已发布版本保留备查」）。
# 它们和正文一样是引用块，但读者从没看过——这个库要发给学员做 demo，展开全文时不能冒出来。
EDITORIAL_HINTS = ("改版", "备查", "动机", "结构调整", "本次只动", "保持不动", "写作规范",
                   "实测公众号", "修改额度", "发布记录", "存档", "旧稿", "作废",
                   "仍未发表", "别当它已发", "灌进", "灌入", "草稿箱", "重灌", "口径见", "记账")


def drop_editorial_quotes(text: str) -> str:
    """连续的引用行算一块；整块里出现编辑说明的词，整块丢掉。

    按行判会把说明块切碎（留下半句「原标题是索取姿态 + 格言体」这种），所以要成块判。
    「脉脉位」那种引用块是正文，不含这些词，照常留着。
    """
    out, block = [], []

    def flush():
        if block:
            joined = "\n".join(block)
            if not any(hint in joined for hint in EDITORIAL_HINTS):
                out.extend(block)
            block.clear()

    for line in text.splitlines():
        if INTERNAL_LINE.search(line):
            continue
        if line.lstrip().startswith(">"):
            block.append(line)
            continue
        flush()
        out.append(line)
    flush()
    return "\n".join(out)


def parse_comic(path: Path):
    """漫画稿的正文在两个地方：摘要位（观点）和分镜表（台词、图内文字）。

    引用块里全是画风指令和人物设定（「细腻干净的细线稿」「米色针织衫」），
    那是给出图工具看的，进了索引只会把观点冲淡，一律丢掉。
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()

    title = ""
    for line in lines:
        if line.startswith("# "):
            title = re.sub(r"[*`]", "", line[2:]).strip()
            break
    title = title or path.parent.name
    candidates = [title]
    stripped = re.sub(r"^漫画\s*[:：·]\s*", "", title).strip()
    if stripped != title:
        candidates.append(stripped)
    candidates.append(f"漫画：{title}")           # 线上次条一般带这个前缀
    candidates.append(path.parent.name)

    body, skipping, header = [], False, None
    for line in lines:
        text = line.strip()
        h = re.match(r"^#{1,4}\s+(.*)$", text)
        if h:
            heading = h.group(1).strip()
            header = None
            # 漫画稿里混着一堆给工具看的段落：画风指令、人物设定、灌草稿箱的命令行。
            # 它们不是内容，进了索引就会在结果里冒出 mp_draft.py 和一串目录名。
            skipping = any(k in heading for k in (
                "画风", "出图", "人设", "角色设定", "红线", "排版", "封面",
                "发布指引", "发布流程", "草稿箱", "灌稿", "合集", "待办", "检查", "质检", "改图"))
            if text.startswith("# ") or skipping:
                continue
            body.append(f"## {heading}")
            continue
        if skipping or text.startswith(">"):        # 引用块＝出图指令
            continue
        if text.startswith("|"):
            cells = [c.strip() for c in text.strip("|").split("|")]
            if all(re.fullmatch(r"[-:\s]*", c) for c in cells):
                continue                            # 表头分隔线
            if header is None:
                header = cells                      # 第一行是表头，用来认列
                continue
            body.append(format_panel(header, cells))
            continue
        line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)
        body.append(line)

    text = drop_editorial_quotes("\n".join(body))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return {"title": title, "candidates": [c for c in dict.fromkeys(candidates) if c],
            "date": "", "link": "", "topic": "漫画", "text": text}


def format_panel(header, cells) -> str:
    """一格分镜排成「台词｜背景字（画面：…）」。

    台词和图内文字是读者真正看到的，画面描述是给出图工具的。两者混在一起等长排列时，
    「静姐站立交桥下双手不拿手机」这种描述会把观点冲淡——搜「杀猪盘怎么套人」，
    画过那篇也只排 0.57，低到被判成「没写过」。所以台词在前，画面截短跟在后面。
    """
    text_cols, bg_cols, visual_cols = [], [], []
    for i, name in enumerate(header):
        if i >= len(cells) or not cells[i].strip():
            continue
        value = cells[i].strip()
        if re.fullmatch(r"[#\d\s格]*", value):        # 序号列
            continue
        if any(k in name for k in ("文案", "文字", "台词", "对白", "正文", "旁白", "字幕")):
            text_cols.append(value)
        elif "背景" in name:
            bg_cols.append(value)
        elif any(k in name for k in ("画面", "镜头", "构图", "场景", "图")):
            visual_cols.append(value)
        else:
            text_cols.append(value)
    parts = text_cols + bg_cols
    if visual_cols and len("".join(parts)) < 200:
        shot = "／".join(visual_cols)[:60]
        parts.append(f"（画面：{shot}）")
    return "｜".join(parts)


def parse_html_title(path: Path) -> str:
    """有几篇漫画只有 index.html（图片壳），标题是里面唯一能拿的文本。"""
    try:
        html = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    m = re.search(r"<title>(.*?)</title>", html, re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def parse_article(path: Path):
    """从 article.md 解析标题、日期、链接与干净正文。"""
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()

    meta = {}
    for line in lines[:60]:
        m = META_LINE.match(line)
        if m:
            meta.setdefault(m.group(1), m.group(2).strip())
        # 手写稿把「正式标题」写在引用块里：> - 正式标题：xxx
        m2 = re.match(r"^\s*>\s*[-*]\s*(正式标题|题材)\s*[：:]\s*(.*)$", line)
        if m2:
            meta.setdefault(m2.group(1), m2.group(2).strip())

    # 标题候选按可信度排序，最终用哪个由线上档案裁决（发布标题才是真相）
    def clean_title(t):
        t = re.sub(r"[*`]", "", t or "").strip()
        t = re.sub(r"^\s*\d+[.、)]\s*", "", t)
        t = re.sub(r"[（(]\s*(推荐|备选|保留|定稿)\s*[)）]\s*$", "", t).strip()
        return t

    candidates = []
    if meta.get("正式标题"):
        candidates.append(clean_title(meta["正式标题"]))
    for line in lines:
        if line.startswith("# "):
            cand = re.sub(r"^公众号文章\s*[·・]\s*", "", line[2:].strip())
            cand = re.sub(r"[（(](历史回填|草稿|定稿|已发布归档.*?)[)）]\s*$", "", cand).strip()
            candidates.append(clean_title(cand))
            break
    candidates.append(clean_title(path.parent.name))
    # 「标题 N 选 1」里带（推荐）的一条，可信度最低：它可能只是备选，未必发出去了
    in_title_section = False
    for line in lines:
        h = re.match(r"^#{2,3}\s+(.*)$", line)
        if h:
            in_title_section = h.group(1).strip().startswith("标题")
            continue
        if in_title_section and "推荐" in line and re.match(r"^\s*\d+[.、)]\s*", line):
            candidates.append(clean_title(line))
    candidates = [c for c in dict.fromkeys(candidates) if c]
    title = candidates[0] if candidates else path.parent.name

    date = meta.get("发布日期") or meta.get("公众号发布日期") or ""
    date = date.strip()
    dm = re.search(r"(20\d{2})\D(\d{1,2})\D(\d{1,2})", date)
    date = f"{dm.group(1)}-{int(dm.group(2)):02d}-{int(dm.group(3)):02d}" if dm else ""

    link = meta.get("链接", "").strip()
    if not re.match(r"^https?://", link):
        link = ""

    # 正文清洗：丢掉纯创作元数据的小节。
    #
    # 两类小节，收手方式不一样：
    # 「元信息」后面往往直接就是正文（老的历史回填稿一个小标题都没有），所以遇到第一段正文就收手——
    # 不这么做会整篇被吞，曾因此丢掉 400 多篇。
    # 「标题 / 题材 / 封面」这些只出现在手写稿里，稿子有完整小标题结构，MUST 一直跳到下一个同级标题：
    # 提前收手会把「这篇是转化文，钱花在引导语上（见规范 §4.6）」这类写给自己看的分析放进正文，
    # 学员点开全文就看见了。
    body, skipping, drop_level, drop_name = [], False, 0, ""
    for line in lines:
        h = re.match(r"^(#{1,4})\s+(.*)$", line)
        if h:
            level, heading = len(h.group(1)), h.group(2).strip()
            if level == 1:
                continue                      # H1 是文件说明行，标题已单取
            if skipping and level > drop_level:
                continue                      # 被丢小节里的子标题
            skipping = any(heading.startswith(s) for s in DROP_SECTIONS)
            drop_level, drop_name = (level, heading) if skipping else (0, "")
            if skipping:
                continue
        elif skipping:
            stripped = line.strip()
            looks_like_meta = (
                not stripped
                or META_LINE.match(line)
                or re.match(r"^\s*([-*>]|\d+[.、)])", line)
                or re.match(r"^\s*(题材|公众号发布日期|内容线|字数|状态|系列)\s*[：:]", stripped)
                or re.match(r"^\s*(-{3,}|\*{3,}|={3,})\s*$", stripped)
            )
            if looks_like_meta or not drop_name.startswith("元信息"):
                continue
            skipping = False                  # 元信息段后面就是正文了
        if META_LINE.match(line) or re.match(r"^\s*>\s*[-*]\s*(正式标题|题材)\s*[：:]", line):
            continue
        if re.match(r"^\s*(-{3,}|\*{3,}|={3,})\s*$", line):
            continue
        line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)      # 图片
        line = re.sub(r"<!--.*?-->", "", line)                 # 注释
        body.append(line)

    text = drop_editorial_quotes("\n".join(body))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return {"title": title, "candidates": candidates, "date": date, "link": link,
            "topic": meta.get("题材", ""), "text": text}


def clean_heading(raw: str) -> str:
    """小节名做展示用，去掉序号和内部叫法。"""
    name = re.sub(r"^[一二三四五六七八九十\d]+[、.．)]\s*", "", raw).strip()
    if name in HEADING_ALIASES:
        return HEADING_ALIASES[name]
    if name.startswith("分镜表"):
        return "分镜"
    if "转发语" in name:
        return "导语"
    return name if len(name) <= 30 else name[:30] + "…"   # 老文章拿整句当小标题，展示要截


def normalize_headings(text: str) -> str:
    """展开全文时正文里的小标题也要是人话——片段上换了、正文里没换，等于白换。"""
    out = []
    for line in text.splitlines():
        h = re.match(r"^(#{2,4})\s+(.*)$", line)
        if not h:
            out.append(line)
            continue
        name = clean_heading(h.group(2).strip())
        if name:
            out.append(f"{h.group(1)} {name}")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out))


def split_chunks(text: str):
    """按段落聚合成 ~480 字的块，块内保留所在小节标题作为上下文。"""
    chunks, buf, buf_len, heading = [], [], 0, ""

    def flush():
        nonlocal buf, buf_len
        if buf:
            content = "\n".join(buf).strip()
            if len(content) >= 30:
                chunks.append({"heading": heading, "text": content})
            buf, buf_len = [], 0

    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        h = re.match(r"^#{2,4}\s+(.*)$", para)
        if h:
            flush()
            heading = clean_heading(h.group(1).strip())
            continue
        # 超长段落硬切
        while len(para) > CHUNK_MAX:
            cut = para.rfind("。", 0, CHUNK_MAX) + 1 or CHUNK_MAX
            flush()
            chunks.append({"heading": heading, "text": para[:cut]})
            para = para[cut:].strip()
        if buf_len + len(para) > CHUNK_TARGET and buf:
            tail = buf[-1] if len(buf) > 1 else None   # 与下一块重叠一段，避免语义被切断
            flush()
            if tail and len(tail) < 200:
                buf, buf_len = [tail], len(tail)
        buf.append(para)
        buf_len += len(para)
    flush()
    return chunks



def read_docx(path: Path) -> str:
    """不引第三方库：docx 是 zip，正文在 word/document.xml，每个 <w:p> 是一段。"""
    import zipfile
    from xml.etree import ElementTree as ET
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    paras = []
    for p in root.iterfind(".//w:p", ns):
        t = "".join(x.text or "" for x in p.iterfind(".//w:t", ns)).strip()
        if t:
            paras.append(t)
    return "\n\n".join(paras)


def collect_my_files():
    """「我的文件」库：读者自己放进去的东西。一份文件一条，kind=file，没有链接，打开就是看全文。"""
    docs = []
    if not MY_FILES_DIR.exists():
        return docs
    for p in sorted(MY_FILES_DIR.iterdir()):
        if p.name.startswith(".") or p.suffix.lower() not in (".md", ".txt", ".docx"):
            continue
        try:
            text = read_docx(p) if p.suffix.lower() == ".docx" else p.read_text(encoding="utf-8")
        except Exception as exc:                       # 一份坏文件别拖垮整库
            print(f"跳过 {p.name}：{exc}")
            continue
        text = text.strip()
        if len(text) < 20:
            continue
        m = re.match(r"^#\s+(.+)$", text, flags=re.M)
        title = (m.group(1).strip() if m else p.stem)
        docs.append({
            "id": f"file:{p.name}",
            "title": title,
            "date": datetime.fromtimestamp(p.stat().st_mtime, CST).strftime("%Y-%m-%d"),
            "link": "",
            "topic": "我的文件",
            "read_num": None,
            "digest": "",
            "source": "本地全文",
            "kind": "file",
            "match": "",
            "aka": [p.stem] if m else [],
            "text": text,
        })
    return docs


def collect():
    hist_index, hist_all = load_history()
    docs, used_links = [], set()

    for source_dir, status in ((ARCHIVE_DIR, "published"), (DRAFT_DIR, "draft")):
        if not source_dir.exists():
            continue
        for md in sorted(source_dir.glob("*/article.md")):
            art = parse_article(md)
            if len(art["text"]) < 120:
                continue
            hit, match_kind = None, ""
            for cand in art["candidates"]:
                hit = hist_index.get(normalize(cand))
                if hit:
                    match_kind = "exact"
                    break
            if not hit:
                for cand in art["candidates"]:
                    fz = fuzzy_match(cand, hist_index, art["text"])
                    if fz:
                        hit, match_kind = fz[0], f"{fz[2]}:{fz[1]:.2f}"
                        break
            if hit and hit.get("title"):
                art["title"] = hit["title"].strip()   # 发布标题是真相，本地标题只当别名
            if hit:
                art["link"] = art["link"] or hit.get("link", "")
                art["read_num"] = hit.get("read_num")
                art["digest"] = hit.get("digest") or ""
                if not art["date"] and hit.get("sent_time"):
                    art["date"] = datetime.fromtimestamp(hit["sent_time"], CST).strftime("%Y-%m-%d")
            if art["link"]:
                used_links.add(art["link"])
            docs.append({
                "id": f"local:{source_dir.name}/{md.parent.name}",
                "title": art["title"],
                "date": art["date"],
                "link": art["link"],
                "topic": art.get("topic", ""),
                "read_num": art.get("read_num"),
                "digest": art.get("digest", ""),
                "source": "本地全文",
                "kind": "article",
                "match": match_kind,
                "aka": [c for c in art["candidates"] if normalize(c) != normalize(art["title"])][:3],
                "text": art["text"],
            })

    for source_dir, status in ((COMIC_ARCHIVE, "published"), (COMIC_DRAFT, "draft")):
        if not source_dir.exists():
            continue
        for folder in sorted(d for d in source_dir.iterdir() if d.is_dir()):
            md = folder / "article.md"
            if md.exists():
                art = parse_comic(md)
                path_str, min_len = str(md), 60
            else:
                # 只有图片壳的老漫画：标题也值得搜，正文就靠线上摘要补
                title = parse_html_title(folder / "index.html") or folder.name
                art = {"title": title, "candidates": [title, f"漫画：{title}", folder.name],
                       "date": "", "link": "", "topic": "漫画", "text": title}
                path_str, min_len = str(folder / "index.html"), 0
            if len(art["text"]) < min_len:
                continue

            hit, match_kind = None, ""
            for cand in art["candidates"]:
                hit = hist_index.get(normalize(cand))
                if hit:
                    match_kind = "exact"
                    break
            if not hit:
                for cand in art["candidates"]:
                    fz = fuzzy_match(cand, hist_index, art["text"])
                    if fz:
                        hit, match_kind = fz[0], f"{fz[2]}:{fz[1]:.2f}"
                        break
            if hit and hit.get("title"):
                art["title"] = hit["title"].strip()
            if hit:
                art["link"] = hit.get("link", "")
                art["read_num"] = hit.get("read_num")
                art["digest"] = hit.get("digest") or ""
                if hit.get("sent_time"):
                    art["date"] = datetime.fromtimestamp(hit["sent_time"], CST).strftime("%Y-%m-%d")
            if art["link"]:
                used_links.add(art["link"])

            docs.append({
                "id": f"comic:{source_dir.name}/{folder.name}",
                "title": art["title"],
                "date": art["date"],
                "link": art["link"],
                "topic": "漫画",
                "read_num": art.get("read_num"),
                "digest": art.get("digest", ""),
                "source": "本地全文" if md.exists() else "仅标题",
                "kind": "comic",
                "match": match_kind,
                "aka": [c for c in art["candidates"] if normalize(c) != normalize(art["title"])][:3],
                "text": art["text"],
            })

    # 线上有、本地没正文的，用标题 + 摘要补进来
    for a in hist_all:
        link = a.get("link") or ""
        if not link or link in used_links or a.get("is_deleted"):
            continue
        title = (a.get("title") or "").strip()
        if not title:
            continue
        digest = (a.get("digest") or "").strip()
        docs.append({
            "id": f"online:{link.rsplit('/', 1)[-1]}",
            "title": title,
            "date": datetime.fromtimestamp(a["sent_time"], CST).strftime("%Y-%m-%d") if a.get("sent_time") else "",
            "link": link,
            "topic": "",
            "read_num": a.get("read_num"),
            "digest": digest,
            "source": "线上摘要",
            "kind": "comic" if title.startswith("漫画") else "article",
            "text": f"{title}\n\n{digest}".strip(),
        })

    # 只留已发布的：这个库要发给学员做 demo，每条都得点得开。
    # 没链接的一律不进——未发布草稿、只剩图片壳又没匹配上线上记录的老漫画，都在这一步落地。
    before = len(docs)
    docs = [d for d in docs if d["link"]]
    print(f"只保留已发布：{before} → {len(docs)} 篇（去掉 {before - len(docs)} 篇没有公众号链接的）")

    docs = arbitrate_links(docs)
    docs.sort(key=lambda d: d.get("date") or "0000-00-00", reverse=True)
    mine = collect_my_files()
    print(f"我的文件 {len(mine)} 份（{MY_FILES_DIR.name}/ 目录）")
    docs.extend(mine)
    return docs


def arbitrate_links(docs):
    """一个 mp 链接只能归一篇。

    两种情况会撞：同一篇文章本地存了两份稿（制作中 + 已归档），以及近似匹配配错了人。
    精确匹配优先，近似匹配让位（宁可这篇没链接）；剩下真同一篇的多份稿，
    保留正文最全的那份，其余不进索引——否则搜出来两条一模一样的结果。
    """
    by_link = {}
    for d in docs:
        if d["link"]:
            by_link.setdefault(d["link"], []).append(d)

    kept = []
    dropped = set()
    for link, group in by_link.items():
        if len(group) == 1:
            continue
        exact = [d for d in group if d.get("match") == "exact" or d["source"] == "线上摘要"]
        if exact and len(exact) < len(group):
            for d in group:
                if d not in exact:
                    d["link"], d["match"] = "", "conflict-dropped"
            group = exact
        if len(group) > 1:      # 同一篇的多份稿：留正文最全的
            group.sort(key=lambda d: (d["source"] == "本地全文", len(d["text"])), reverse=True)
            for d in group[1:]:
                dropped.add(d["id"])
                kept.append((d["title"], group[0]["title"]))

    by_title = {}
    for d in docs:
        if d["id"] in dropped:
            continue
        by_title.setdefault(normalize(d["title"]), []).append(d)
    for group in by_title.values():
        if len(group) < 2:
            continue
        # 同名的多份稿：带链接的、正文最全的留下
        group.sort(key=lambda d: (bool(d["link"]), d["source"] == "本地全文", len(d["text"])), reverse=True)
        for d in group[1:]:
            dropped.add(d["id"])

    if dropped:
        print(f"合并重复稿 {len(dropped)} 篇（同一篇文章的多份本地稿，保留最全的一份）")
    return [d for d in docs if d["id"] not in dropped]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    docs = collect()

    chunks = []
    for doc in docs:
        pieces = split_chunks(doc["text"]) or [{"heading": "", "text": doc["text"][:CHUNK_MAX]}]
        for i, p in enumerate(pieces):
            ctx = f"《{doc['title']}》"
            if p["heading"]:
                ctx += f"｜{p['heading']}"
            chunks.append({
                "chunk_id": f"{doc['id']}#{i}",
                "doc_id": doc["id"],
                "seq": i,
                "heading": p["heading"],
                "text": p["text"],
                "embed_text": f"{ctx}\n{p['text']}",
            })

    meta = {d["id"]: {k: v for k, v in d.items() if k != "text"} for d in docs}
    (OUT_DIR / "docs.json").write_text(
        json.dumps({"built_at": datetime.now(CST).isoformat(), "count": len(docs), "docs": meta},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    with (OUT_DIR / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    full = {d["id"]: normalize_headings(d["text"]) for d in docs}
    (OUT_DIR / "fulltext.json").write_text(json.dumps(full, ensure_ascii=False), encoding="utf-8")

    n_comic = sum(1 for d in docs if d.get("kind") == "comic")
    n_local = sum(1 for d in docs if d["source"] == "本地全文")
    print(f"已发布 {len(docs)} 篇：文章 {len(docs) - n_comic}，漫画 {n_comic}")
    print(f"本地有全文 {n_local} 篇，其余 {len(docs) - n_local} 篇只有标题与摘要")
    print(f"检索块 {len(chunks)} 个，平均 {sum(len(c['text']) for c in chunks) // max(len(chunks), 1)} 字")


if __name__ == "__main__":
    main()
