"""Turn one page (markdown) into typed, dated chunks: prose passages and table rows."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from urllib.parse import urlparse

from .text import (EFFECTIVE_RE, IMG_RE, LINK_RE, URL_DATE_RE, clean_cell, is_junk_line, norm_space, parse_long_date,
                   parse_num_date, strip_markdown_line)

# ---------------------------------------------------------------------------
# Page classification and authority
# ---------------------------------------------------------------------------
# Authority = how much we trust a page for *current product facts*.
AUTHORITY = {
    "rates": 1.00,      # the official rate sheet(s)
    "product": 0.90,    # product / service pages
    "contact": 0.95,
    "pdf": 0.55,        # disclosures, fee schedules, newsletters
    "landing": 0.55,    # promo landing pages (often left up after the promo ends)
    "news": 0.55,       # press releases (dated)
    "blog": 0.45,       # articles (dated)
    "resource": 0.40,   # business resource articles
    "calculator": 0.30, # calculators carry illustrative / stale rates
    "index": 0.08,      # tag / author / listing pages (snippets of other pages)
}


def classify(url: str) -> str:
    p = urlparse(url)
    path = p.path.lower().rstrip("/")
    if path in ("/rates", "/business-rates"):
        return "rates"
    if path in ("/contact", "/connect/contact-us"):
        return "contact"
    if re.search(r"/(tag|author|category|page)/", path + "/") or path in ("/learn/blog", "/newsroom",
                                                                             "/business/resources"):
        return "index"
    if path.endswith((".pdf", ".xlsx", ".xls", ".docx")):
        return "pdf"
    if "calculator" in path or path.startswith("/calculators") or path == "/business-calculators":
        return "calculator"
    if path.startswith("/newsroom"):
        return "news"
    if path.startswith("/learn/blog") or path.startswith("/dupacodaily") or re.match(r"^/20\d\d/", path):
        return "blog"
    if path.startswith("/business/resources"):
        return "resource"
    if path.endswith("-lp") or path in ("/switch-and-earn", "/earn-more", "/offer-expired",
                                         "/money-makeover", "/21-day-sprint-to-savings-challenge",
                                         "/sprint-to-savings-challenge", "/mortgage-refi"):
        return "landing"
    return "product"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    kind: str                 # 'para' | 'row' | 'table'
    text: str                 # display text
    heading: str              # heading path / table caption
    effective: date | None = None
    header: list[str] = field(default_factory=list)   # table header (rows/tables)
    cells: list[str] = field(default_factory=list)    # table row cells (rows)
    table_no: int = -1
    row_no: int = -1
    ctx: str = ""             # rows: words just before the table + nearby link paths (product context)

    def index_text(self, title: str) -> str:
        t = title.split("|")[0]
        return f"{t} {t} {self.heading} {self.ctx} {self.text}"


@dataclass
class ParsedPage:
    url: str
    title: str
    ptype: str
    pub_date: date | None
    chunks: list[Chunk]
    links: list[tuple[str, str]]
    clean_text: str


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# "6-month certificate APY: 3.50%" / "Lost debit card replacement fee: $2.50"
KV_LINE_RE = re.compile(r"^([A-Za-z0-9][^:|]{2,80}?):\s*(\$?\d[\d,]*(?:\.\d+)?\s?%?(?:\s*(?:APY|APR))?)\s*\.?$")
SEP_ROW_RE = re.compile(r"^\|?\s*:?-{2,}")
READ_TIME_RE = re.compile(r"^\d+ min read$", re.I)
MAX_PARA = 850


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    cells = re.split(r"(?<!\\)\|", s)
    return [clean_cell(strip_markdown_line(c.replace("<br>", " "))) for c in cells]


def _is_caption(line: str) -> bool:
    """Short plain line that names the following table (e.g. 'Share Savings')."""
    return 2 <= len(line) <= 70 and not line.endswith((".", ":", "!", "?")) and "|" not in line


def _pub_date(url: str, lines: list[str]) -> date | None:
    for ln in lines[:14]:
        d = parse_long_date(ln)
        if d and len(ln) < 40:
            return d
    m = URL_DATE_RE.search(url)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3) or 1))
        except ValueError:
            return None
    return None


def _drop_form_lists(lines: list[str]) -> list[str]:
    """Remove form dropdowns (runs of many tiny bullets: states, countries) and the article header
    lines ('2 min read', a lone date)."""
    out, run = [], []
    for ln in lines:
        st = ln.strip()
        if re.match(r"^[-*] .{1,28}$", st) or (run and not st):
            run.append(ln)
            continue
        if sum(1 for r in run if r.strip()) < 8:
            out.extend(run)
        run = []
        out.append(ln)
    if sum(1 for r in run if r.strip()) < 8:
        out.extend(run)
    head = 0
    for k, ln in enumerate(out):
        st = ln.strip()
        if not st:
            continue
        head += 1
        if head > 15:
            break
        if READ_TIME_RE.match(st) or (len(st) < 40 and parse_long_date(st)):
            out[k] = ""
    return out


def parse_page(url: str, title: str, markdown: str) -> ParsedPage:
    ptype = classify(url)
    lines0 = markdown.splitlines()
    if ptype in ("news", "blog", "resource"):
        # article pages end with 'related article' cards that start with their own "N min read":
        # everything after the second such marker belongs to other articles.
        marks = [k for k, l in enumerate(lines0) if READ_TIME_RE.match(l.strip())]
        if len(marks) >= 2:
            lines0 = lines0[:marks[1]]
    raw_lines = _drop_form_lists(lines0)
    pub = _pub_date(url, [l.strip() for l in markdown.splitlines() if l.strip()]) if ptype in (
        "news", "blog", "resource", "pdf", "landing", "product") else None
    if ptype in ("product", "landing") and pub and pub.year < 2015:
        pub = None

    links: list[tuple[str, str]] = []
    chunks: list[Chunk] = []
    headings: dict[int, str] = {}
    para_buf: list[str] = []
    recent_plain: list[str] = []          # recent short plain lines (table captions)
    recent_links: list[str] = []          # targets of recent link-only lines ("Explore -> /certificates/")
    last_text = ""
    clean_lines: list[str] = []
    table_no = 0

    def heading_path() -> str:
        return " > ".join(headings[k] for k in sorted(headings))

    def flush_para():
        if not para_buf:
            return
        text = norm_space(" ".join(para_buf))
        para_buf.clear()
        if len(text) < 3:
            return
        hp = heading_path()
        # split long text on sentence boundaries into ~MAX_PARA windows
        if len(text) <= MAX_PARA:
            chunks.append(Chunk("para", text, hp))
            return
        sents = re.split(r"(?<=[.!?])\s+", text)
        cur = ""
        for s in sents:
            if cur and len(cur) + len(s) > MAX_PARA:
                chunks.append(Chunk("para", cur.strip(), hp))
                cur = ""
            cur += " " + s
        if cur.strip():
            chunks.append(Chunk("para", cur.strip(), hp))

    i = 0
    n = len(raw_lines)
    header_zone = ptype in ("news", "blog", "resource")
    while i < n:
        line = raw_lines[i]
        st = line.strip()
        # ---- tables -------------------------------------------------------
        if st.startswith("|") and i + 1 < n and SEP_ROW_RE.match(raw_lines[i + 1].strip().lstrip("|").strip()
                                                                     or "x"):
            flush_para()
            header = _split_row(st)
            j = i + 2
            rows = []
            while j < n and raw_lines[j].strip().startswith("|"):
                rows.append(_split_row(raw_lines[j]))
                j += 1
            # effective date within the next few non-empty lines
            eff = None
            k, seen = j, 0
            while k < n and seen < 5:
                t = raw_lines[k].strip()
                if t:
                    seen += 1
                    if t.startswith("|") or HEADING_RE.match(t):
                        break
                    m = EFFECTIVE_RE.search(t)
                    if m:
                        eff = parse_num_date(m.group(1))
                        break
                k += 1
            caption_bits = [c for c in recent_plain[-2:] if c not in heading_path()]
            caption = norm_space(" > ".join([heading_path()] + caption_bits)).strip(" >")
            table_no += 1
            tbl_lines = []
            ctx = " ".join(re.sub(r"[/\-_?=#.]", " ", urlparse(u).path) for u in recent_links[-2:])
            ctx = norm_space(f"{ctx} {last_text[-160:]}")
            for r_i, cells in enumerate(rows):
                if not any(cells):
                    continue
                pairs = [f"{h}: {c}" if h and h.lower() not in ("col1", "col2", "col3") else c
                         for h, c in zip(header + [""] * (len(cells) - len(header)), cells) if c]
                text = "; ".join(pairs)
                tbl_lines.append(" | ".join(c for c in cells if c))
                chunks.append(Chunk("row", text, caption, eff, header, cells, table_no, r_i, ctx))
            if tbl_lines:
                body = " || ".join(tbl_lines)
                if len(body) > 2500:
                    body = body[:2500] + " ..."
                chunks.append(Chunk("table", body, caption, eff, header, [], table_no, -1))
                clean_lines.append(f"[table] {caption}: {body}")
            recent_plain = []
            i = j
            continue
        # ---- headings ----------------------------------------------------
        hm = HEADING_RE.match(st)
        if hm:
            flush_para()
            level = len(hm.group(1))
            text = strip_markdown_line(hm.group(2), links)
            for lv in [k for k in headings if k >= level]:
                del headings[lv]
            if text:
                headings[level] = text[:120]
                clean_lines.append(text)
            recent_plain = []
            header_zone = False
            i += 1
            continue
        # ---- prose -------------------------------------------------------
        text = strip_markdown_line(st, links)
        if header_zone and (READ_TIME_RE.match(text) or (parse_long_date(text) and len(text) < 40)):
            i += 1
            continue
        if not st:
            i += 1
            continue
        if is_junk_line(text):
            i += 1
            continue
        clean_lines.append(text)
        link_only = st.startswith("[") and not LINK_RE.sub("", IMG_RE.sub("", st)).strip(" \\")
        if _is_caption(text) and not text.startswith(("-", "*")) and not link_only:
            recent_plain.append(text)
            recent_plain = recent_plain[-3:]
        if link_only:
            recent_links.extend(u for _, u in LINK_RE.findall(st) if "dupaco" in u or u.startswith("/"))
            recent_links = recent_links[-4:]
        if link_only and len(text) < 60:
            i += 1
            continue
        last_text = text
        kv = KV_LINE_RE.match(text)
        if kv and not text.startswith(("-", "*", "•")):
            flush_para()
            label, value = kv.group(1).strip(), clean_cell(kv.group(2))
            chunks.append(Chunk("row", f"{label}: {value}", heading_path(), None, ["", ""], [label, value],
                                1000 + len(chunks), 0))
            i += 1
            continue
        if text.startswith(("- ", "* ")):
            text = "• " + text[2:]
        para_buf.append(text)
        if sum(len(p) for p in para_buf) > MAX_PARA:
            carry = para_buf[-1] if len(para_buf) > 1 and len(para_buf[-1]) < 300 else None
            if carry:
                para_buf.pop()
            flush_para()
            if carry:
                para_buf.append(carry)
        i += 1
    flush_para()
    return ParsedPage(url, title, ptype, pub, chunks, links, "\n".join(clean_lines))
