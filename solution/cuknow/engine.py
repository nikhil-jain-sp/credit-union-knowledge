"""Deterministic answering: retrieval with authority/recency priors, structured rate lookup,
passage extraction, link lookup and explicit "not on the site" checks."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .parse import AUTHORITY, classify
from .query import QueryInfo, analyze
from .store import ChunkRec, Store
from .text import SYNONYMS, _stem, parse_iso, sentences, tokenize

# Words that carry no product identity; ignored when checking whether evidence covers the question.
GENERIC = set(tokenize(
    "rate rates apr apy yield interest dividend earn pay paying charge charged loan loans account accounts "
    "current standard regular normal typical purchase price cost much what get open today now available "
    "percent percentage annual information info tell know member members bank banking option options "
    "website site page online service services product products number amount term terms fixed "
    "general typical usually usual new old provide give show need want like looking see find tell list "
    "quote highest lowest best top maximum minimum cheapest right now currently between earn earning "
    "much amount up least over under around approximately exactly still today"))

TIME_RE = re.compile(r"\b\d{1,2}(?::\d\d)?\s?(?:a\.?m\.?|p\.?m\.?)|\bnoon\b", re.I)
ADDR_RE = re.compile(r"\b\d{2,6}\s+[A-Z][\w.]*(?:\s+[A-Z][\w.]*){0,4}\s+(?:St|Street|Rd|Road|Ave|Avenue|Dr|Drive|"
                     r"Ln|Lane|Blvd|Way|Ct|Hwy|Highway|Pl|Pkwy)\b|\b\d{5}\b")
PHONE_RE = re.compile(r"\b\d{3}[-.]\d{3}[-.]\d{4}\b|\(\d{3}\)\s?\d{3}-\d{4}")
DATE_RE = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|"
                     r"December)\s+\d{1,2},?\s+20\d\d\b")
NUM_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
PCT_IN = re.compile(r"\d+(?:\.\d+)?\s?%")

DROP_COLS = re.compile(r"min\.?\s*to|payment example|^points$", re.I)
SHORT_HDR = [(re.compile(r"annual percentage yield.*", re.I), "APY"),
             (re.compile(r"annual percentage rate.*as low as.*", re.I), "APR (as low as)"),
             (re.compile(r"annual percentage rate.*", re.I), "APR")]

INTENT_WORDS = set(tokenize("apply application online sign up now today open start get started"))

MONTH_TOKENS = {_stem(m) for m in ["january", "february", "march", "april", "may", "june", "july", "august",
                                  "september", "october", "november", "december", "share", "month", "year"]}

REFUSAL_TEXT = ("I don't know: the website doesn't list {what}. "
                "Please contact the credit union directly to confirm.")


@dataclass
class Result:
    text: str
    sources: list[tuple[str, str | None, str | None]] = field(default_factory=list)   # (url, snapshot, quote)
    confident: bool = True
    conflict_note: str | None = None
    evidence: list[ChunkRec] = field(default_factory=list)
    kind: str = "passage"


def _short_header(h: str) -> str:
    for pat, rep in SHORT_HDR:
        if pat.match(h.strip()):
            return rep
    return h.strip()


def _money(s: str) -> float:
    return float(s.replace("$", "").replace(",", ""))


def row_tier(cells: list[str], header: list[str]) -> tuple[float, float] | None:
    """Balance / loan-amount tier of a table row, as a [lo, hi] range."""
    for h, c in zip(header + [""] * len(cells), cells):
        if DROP_COLS.search(h or "") or "%" in c:
            continue
        m = re.search(r"\$\s?([\d,]+(?:\.\d+)?)\s*-\s*\$?\s?([\d,]+(?:\.\d+)?)", c)
        if m:
            return _money(m.group(1)), _money(m.group(2))
        m = re.search(r"\$\s?([\d,]+(?:\.\d+)?)\s*\+", c)
        if m:
            return _money(m.group(1)), math.inf
        m = re.search(r"\$\s?([\d,]+(?:\.\d+)?)\s*or less", c)
        if m:
            return 0.0, _money(m.group(1))
    return None


def row_model_years(text: str) -> tuple[int, int] | None:
    m = re.search(r"\b((?:19|20)\d\d)\s*-\s*(\d{2,4})\b", text)
    if m and "model" in text.lower():
        a = int(m.group(1))
        b = m.group(2)
        b = int(b) if len(b) == 4 else int(str(a)[:2] + b)
        return min(a, b), max(a, b)
    m = re.search(r"\b((?:19|20)\d\d) model year \(and older\)", text, re.I)
    if m:
        return 1900, int(m.group(1))
    return None


class Engine:
    def __init__(self, store: Store):
        self.s = store
        self._links_cache: dict[int, list] = {}

    # ------------------------------------------------------------------ retrieval
    def view(self, as_of: str | None, qi: QueryInfo) -> int:
        d = parse_iso(as_of) if as_of else None
        if d is None and qi.as_of_hint is not None:
            d = qi.as_of_hint
        return self.s.snap_index_for(d)

    def scores(self, qi: QueryInfo, snap: int, recency_strength: float = 0.12) -> np.ndarray:
        s = self.s
        base = s.bm25(qi.tokens)
        visible = (s.mask & (1 << snap)) != 0
        view_ord = s.snaps[snap].taken_on.toordinal()
        age = np.where(s.pub > 0, np.maximum(0, view_ord - s.pub) / 365.0, 0.0)
        recency = 1.0 / (1.0 + recency_strength * age)
        if qi.asks_rate:
            kindf = np.choose(s.kind, [0.75, 1.35, 0.35])
            auth = s.auth ** 1.5
        else:
            kindf = np.choose(s.kind, [1.0, 0.85, 0.35])
            auth = s.auth
        return np.where(visible, base * auth * recency * kindf * s.stale, 0.0)

    def top(self, sc: np.ndarray, k: int) -> list[int]:
        k = min(k, len(sc))
        idx = np.argpartition(-sc, k - 1)[:k]
        idx = idx[np.argsort(-sc[idx])]
        return [int(i) for i in idx if sc[i] > 0]

    # ------------------------------------------------------------------ coverage
    def content_terms(self, qi: QueryInfo) -> list[str]:
        out = []
        for t, w in qi.tokens:
            if w < 1.0 or t in GENERIC or re.fullmatch(r"\d+(?:\.\d+)?%?|\$[\d.]+", t):
                continue
            if re.fullmatch(r"\d+(mo|yr)", t) or t in MONTH_TOKENS:
                continue
            out.append(t)
        return out

    def coverage(self, terms: list[str], text_tokens: set[str]) -> float:
        if not terms:
            return 1.0
        tot = got = 0.0
        for t in terms:
            w = self.s.idf(t) if self.s.df.get(t) else 6.0
            tot += w
            alts = {t} | {_stem(x) for x in SYNONYMS.get(t, [])}
            if alts & text_tokens:
                got += w
        return got / tot if tot else 1.0

    def unknown_names(self, qi: QueryInfo) -> list[str]:
        out = []
        for w in qi.proper_nouns:
            t = _stem(w.lower())
            if not self.s.df.get(t) and not self.s.df.get(w.lower()):
                out.append(w)
        return out

    # ------------------------------------------------------------------ sources
    def _src(self, rec: ChunkRec, snap: int, quote: str | None = None):
        return (rec.url, self.s.snaps[snap].sid, (quote or rec.chunk.text)[:200])

    def agreeing_sources(self, value_tokens: list[str], cand: list[int], snap: int, skip: set[str],
                         limit: int = 5) -> list:
        out = []
        for cid in cand:
            rec = self.s.chunks[cid]
            if rec.url in skip or self.s.page_of(rec).ptype in ("index", "calculator"):
                continue
            txt = rec.chunk.text
            if value_tokens and all(v in txt for v in value_tokens):
                out.append(self._src(rec, snap))
                skip.add(rec.url)
                if len(out) >= limit:
                    break
        return out

    # ------------------------------------------------------------------ main entry
    def answer(self, question: str, as_of: str | None = None) -> Result:
        qi = analyze(question)
        snap = self.view(as_of, qi)
        unknown = self.unknown_names(qi)
        if unknown:
            return Result(REFUSAL_TEXT.format(what=f"anything about '{' '.join(unknown)}'"), confident=False,
                          kind="refuse")
        if qi.asks_link:
            r = self.answer_link(qi, snap)
            if r:
                return r
        if qi.asks_rate:
            r = self.answer_rate(qi, snap)
            if r:
                return r
        return self.answer_passage(qi, snap)

    # ------------------------------------------------------------------ rates
    def row_match(self, rec: ChunkRec, qi: QueryInfo) -> float:
        ch = rec.chunk
        text = f"{ch.heading} {ch.text}"
        low = text.lower()
        if not PCT_IN.search(ch.text):
            return 0.0
        f = 1.0
        for m in qi.months:
            if not re.search(rf"\b{m}\s*-?\s*(?:months?|mo)\b", low):
                return 0.0
        for y in qi.years:
            if not re.search(rf"\b{y}\s*-?\s*(?:years?|yr)\b", low):
                return 0.0
        if qi.model_year:
            rng = row_model_years(text)
            if rng is None:
                f *= 0.4
            elif not rng[0] <= qi.model_year <= rng[1]:
                return 0.0
        if qi.amounts:
            tier = row_tier(ch.cells, ch.header)
            if tier is not None:
                lo = min(qi.amounts)
                hi = max(qi.amounts)
                if not (tier[0] <= lo <= tier[1]):
                    return 0.0
                if len(qi.amounts) > 1 and not (tier[0] <= hi <= tier[1] + 1):
                    f *= 0.6
        if qi.jumbo and "jumbo" not in low:
            return 0.0
        if not qi.jumbo and "jumbo" in low:
            f *= 0.5
        is_ira = bool(re.search(r"\bira\b|roth|traditional", low))
        if qi.ira != is_ira:
            f *= 0.25
        is_biz = "business" in low or "/business" in rec.url
        if qi.business != is_biz:
            f *= 0.8 if (qi.asks_list and not qi.business) else 0.45
        if qi.special and "special" not in low:
            f *= 0.6
        if not qi.special and "special" in low and (qi.months or qi.years):
            f *= 0.9
        if re.search(r"intro|introductory", low) and not re.search(r"intro|promo", qi.norm.lower()):
            f *= 0.5
        return f

    def fmt_row(self, rec: ChunkRec) -> str:
        ch = rec.chunk
        parts = []
        for h, c in zip(ch.header + [""] * len(ch.cells), ch.cells):
            if not c or c in ("-", "—") or DROP_COLS.search(h or ""):
                continue
            h2 = _short_header(h) if h else ""
            parts.append(f"{h2}: {c}" if h2 and h2.lower() not in ("account type", "certificate", "ira type",
                                                                     "card type", "program", "product",
                                                                     "loan program", "loan type") else c)
        return ", ".join(parts)

    def table_key(self, rec: ChunkRec) -> tuple:
        return (rec.url, rec.vkey, rec.chunk.table_no)

    def answer_rate(self, qi: QueryInfo, snap: int) -> Result | None:
        s = self.s
        sc = self.scores(qi, snap)
        cand = self.top(sc, 400)
        rows = []
        for cid in cand:
            rec = s.chunks[cid]
            if rec.chunk.kind != "row":
                continue
            m = self.row_match(rec, qi)
            if m <= 0:
                continue
            rows.append((sc[cid] * m, rec))
        terms = self.content_terms(qi)
        if not rows:
            return self._rate_refusal_or_none(qi, snap, sc, cand, terms)

        # group rows into tables; a table's score is its best row
        tables: dict[tuple, list] = defaultdict(list)
        for score, rec in rows:
            tables[self.table_key(rec)].append((score, rec))
        ranked = sorted(tables.values(), key=lambda rs: -max(r[0] for r in rs))

        # take the best-ranked table that actually talks about the product asked for
        def covers(rs) -> bool:
            rec = max(rs, key=lambda r: r[0])[1]
            pg = s.page_of(rec)
            return self.coverage(terms, set(tokenize(
                f"{pg.title} {rec.chunk.heading} {rec.chunk.ctx} {rec.chunk.text}"))) >= 0.5
        ranked = [rs for rs in ranked[:12] if covers(rs)]
        if not ranked:
            return self._rate_refusal_or_none(qi, snap, sc, cand, terms, force_check=True)
        best_rows = ranked[0]
        best_score = max(r[0] for r in best_rows)
        best_rec = max(best_rows, key=lambda r: r[0])[1]

        chosen = [best_rows]
        sig = lambda rs: (tuple(h.lower() for h in rs[0][1].chunk.header), rs[0][1].chunk.heading.split(">")[-1]
                          .strip().lower())
        seen_sig = {sig(best_rows)}
        seen_hdr_caps = {sig(best_rows)[1]}
        seen_labels = {row_signature(r)[1] for r in self._table_rows(best_rec)}
        if qi.asks_list and not (qi.months or qi.years or qi.amounts or qi.model_year):
            for rs in ranked[1:6]:
                sc_rs = max(r[0] for r in rs)
                rec0 = rs[0][1]
                if sc_rs < 0.55 * best_score:
                    break
                labels = {row_signature(r)[1] for r in self._table_rows(rec0)}
                if sig(rs) in seen_sig or sig(rs)[1] in seen_hdr_caps or labels & seen_labels:
                    continue          # same product from another page: resolved by ranking
                ev = set(tokenize(f"{s.page_of(rec0).title} {rec0.chunk.heading} {rec0.chunk.text}"))
                if self.coverage(terms, ev) < 0.5:
                    continue
                chosen.append(rs)
                seen_sig.add(sig(rs))
                seen_hdr_caps.add(sig(rs)[1])
                seen_labels |= labels
                if len(chosen) >= 4:
                    break

        lines, srcs, used_urls = [], [], set()
        for rs in chosen:
            rs_sorted = sorted(rs, key=lambda r: r[1].chunk.row_no)
            top_sc = max(r[0] for r in rs)
            keep = [r for r in rs_sorted if r[0] >= 0.45 * top_sc]
            if qi.superlative:
                vals = []
                for r in keep:
                    pv = [float(x.rstrip("% ")) for x in PCT_IN.findall(r[1].chunk.text)]
                    if pv:
                        vals.append((max(pv), r))
                if vals:
                    pick = max(vals, key=lambda v: v[0]) if qi.superlative == "max" else min(vals, key=lambda v: v[0])
                    keep = [pick[1]]
            # complete the table if the member asked for "the rates" without a tier
            if not (qi.months or qi.years or qi.amounts or qi.model_year or qi.superlative):
                full = [c for c in self._table_rows(rs[0][1]) if self.row_match(c, qi) > 0]
                if full and len(full) <= 12:
                    keep = [(0, c) for c in full]
            keep = keep[:10]
            rec0 = keep[0][1]
            caption = rec0.chunk.heading.split(">")[-1].strip() or s.page_of(rec0).title.split("|")[0].strip()
            body = "; ".join(self.fmt_row(r[1]) for r in keep)
            eff = rec0.chunk.effective
            eff_s = f" (rates effective as of {eff.month:02d}/{eff.day:02d}/{eff.year})" if eff else ""
            lines.append(f"{caption}: {body}{eff_s}.")
            if rec0.url not in used_urls:
                srcs.append(self._src(rec0, snap, body))
                used_urls.add(rec0.url)

        # pages that agree with the chosen value(s) are cited too; disagreeing ones are noted
        key_vals = PCT_IN.findall(self.fmt_row(best_rec))[:1]
        srcs += self.agreeing_sources(key_vals, [r[1].cid for r in rows], snap, used_urls)
        note = self._competing_note(best_rec, rows, qi)
        text = " ".join(lines)   # competing (stale) values live only in conflict_note, never in the text
        return Result(text, srcs, True, note, [r[1] for r in chosen[0]], kind="rate")

    def _table_rows(self, rec: ChunkRec) -> list[ChunkRec]:
        page = self.s.versions[rec.vkey]
        out = []
        for ch in page.chunks:
            if ch.kind == "row" and ch.table_no == rec.chunk.table_no:
                cid = self.s._chunk_key.get((rec.url, ch.kind, ch.heading, ch.text))
                if cid is not None:
                    out.append(self.s.chunks[cid])
        return out

    def _competing_note(self, best: ChunkRec, rows, qi) -> str | None:
        """Explain values from other pages that we deliberately did not use."""
        best_vals = set(PCT_IN.findall(best.chunk.text))
        notes = []
        for score, rec in rows:
            if rec.url == best.url or rec.chunk.row_no < 0:
                continue
            if row_signature(rec) != row_signature(best):
                continue
            vals = set(PCT_IN.findall(rec.chunk.text))
            diff = vals - best_vals
            if diff:
                ptype = self.s.page_of(rec).ptype
                notes.append(f"Ignored {', '.join(sorted(diff))} on {rec.url} ({ptype} page"
                             f"{', older effective date' if (rec.chunk.effective or date.min) < (best.chunk.effective or date.min) else ''}"
                             f"); used the {self.s.page_of(best).ptype} page {best.url}.")
            if len(notes) >= 2:
                break
        return " ".join(notes) or None

    def _rate_refusal_or_none(self, qi, snap, sc, cand, terms, force_check=False) -> Result | None:
        """No table row matches. Refuse when no authoritative passage states a rate for this product."""
        s = self.s
        for cid in cand[:40]:
            rec = s.chunks[cid]
            page = s.page_of(rec)
            if rec.chunk.kind != "para" or page.ptype in ("index", "calculator", "blog", "resource"):
                continue
            for sent in sentences(rec.chunk.text):
                if not PCT_IN.search(sent) or not re.search(r"apr|apy|rate|dividend", sent, re.I):
                    continue
                toks = set(tokenize(f"{page.title} {rec.chunk.heading} {sent}"))
                if self.coverage(terms, toks) >= 0.6 and all(
                        re.search(rf"\b{m}\s*-?\s*(?:months?|mo)\b", sent.lower()) for m in qi.months):
                    return None      # let the passage answerer handle it
        what = "a published rate for that" if terms else "that rate"
        if terms:
            what = f"a published rate for {' '.join(t for t in qi.raw.split() if _stem(re.sub(r'[^a-z0-9/-]', '', t.lower())) in terms) or 'that product'}"
        return Result(REFUSAL_TEXT.format(what=what), confident=False, kind="refuse")

    # ------------------------------------------------------------------ links
    def links_for(self, snap: int):
        if snap in self._links_cache:
            return self._links_cache[snap]
        out = []
        for url, vkey in self.s.snaps[snap].pages.items():
            page = self.s.versions[vkey]
            if page.ptype in ("index",):
                continue
            for text, target in page.links:
                if not target.startswith("http") or "#" == target[-1:]:
                    continue
                tgt = target.split("?")[0]
                if re.search(r"#dupaco-locator-widget", tgt):
                    tgt = tgt.split("#")[0]
                tgt = tgt.rstrip("/") if not tgt.endswith("#rates") else tgt
                out.append((set(tokenize(text)), tgt))
        self._links_cache[snap] = out
        return out

    def answer_link(self, qi: QueryInfo, snap: int) -> Result | None:
        terms = [t for t, w in qi.tokens if t not in {"link", "url", "use", "see", "website", "page", "go",
                                                   "web", "site", "online", "address"}]
        tset = set(terms) | {_stem(x) for t in terms for x in SYNONYMS.get(t, [])}
        votes: dict[str, float] = defaultdict(float)
        topic_votes: dict[str, float] = defaultdict(float)
        want_articles = bool(re.search(r"\b(blog|article|story|news|press|calculator)\b", qi.norm.lower()))
        for anchor, tgt in self.links_for(snap):
            if "dupaco.com" not in tgt:
                continue
            if not want_articles and classify(tgt) in ("blog", "news", "resource", "index", "calculator"):
                continue      # link questions are navigational: product pages, rate sheets, apps
            ov = anchor & tset
            if ov:
                votes[tgt] += sum(self.s.idf(t) for t in ov)
                topical = ov - INTENT_WORDS
                if topical:
                    topic_votes[tgt] += sum(self.s.idf(t) for t in topical)
        if not votes:
            return None
        low_q = qi.norm.lower()
        final = {}
        for tgt, v in votes.items():
            path_toks = set(tokenize(re.sub(r"[/\-_.#]", " ", tgt.split("//", 1)[-1])))
            v += sum(self.s.idf(t) for t in (path_toks & tset))
            host = tgt.split("//", 1)[-1].split("/")[0]
            auth = AUTHORITY.get(classify(tgt), 0.5) if host.startswith("www.") else 0.9
            if re.search(r"\brates?\b", low_q) and (tgt.endswith("#rates") or tgt.rstrip("/").endswith("/rates")):
                v *= 1.5
            if re.search(r"\bonline\b|\bapply\b|\bapplication\b", low_q) and not host.startswith("www."):
                v *= 1.6
            final[tgt] = v * auth
        best = max(final.items(), key=lambda kv: kv[1])[0]
        if re.search(r"\bonline\b|\bapply\b|\bapplication\b", low_q):
            apps = [(topic_votes[t], t) for t, v in final.items()
                    if not t.split("//", 1)[-1].startswith("www.") and v >= 0.5 * final[best]
                    and topic_votes[t] > 0]
            if apps:
                best = max(apps)[1]
        if final[best] < 4:
            return None
        clean = best.replace("http://", "https://")
        srcs = [(clean, self.s.snaps[snap].sid, None)] if "www.dupaco.com" in clean else []
        return Result(f"Use {clean} .", srcs, True, kind="link")

    # ------------------------------------------------------------------ passages
    def answer_passage(self, qi: QueryInfo, snap: int) -> Result:
        s = self.s
        rs = 0.6 if (qi.asks_count or re.search(r"\b(latest|recent|newest|last year)\b", qi.norm.lower())) \
            else 0.12
        sc = self.scores(qi, snap, recency_strength=rs)
        cand = self.top(sc, 60)
        if not cand:
            return Result(REFUSAL_TEXT.format(what="that information"), confident=False, kind="refuse")
        terms = self.content_terms(qi)
        low_q = qi.norm.lower()
        count_noun = None
        if qi.asks_count:
            m = re.search(r"how many ([a-z]+)(?: [a-z]+)?", low_q)
            if m:
                count_noun = _stem(m.group(1))
        asks_open = bool(re.search(r"\b(open|close[sd]?|closing|closure|still)\b", low_q))

        if (qi.asks_hours or qi.asks_address) and qi.proper_nouns:
            names = [w.lower() for w in qi.proper_nouns]
            ok = [cid for cid in cand[:60]
                  if all(n in s.chunks[cid].chunk.text.lower() for n in names)
                  and (TIME_RE.search(s.chunks[cid].chunk.text) if qi.asks_hours
                       else ADDR_RE.search(s.chunks[cid].chunk.text))]
            if not ok:
                return Result(REFUSAL_TEXT.format(what=f"{'hours' if qi.asks_hours else 'an address'} for "
                                                       f"{' '.join(qi.proper_nouns)}"), confident=False,
                              kind="refuse")
            cand = ok
        elif qi.asks_hours:
            r = self.answer_hours(qi, snap, sc, cand)
            if r:
                return r

        def pattern_bonus(text: str) -> float:
            b = 1.0
            if count_noun and re.search(rf"(?<![\d,])\d{{1,3}}(?:,\d{{3}})?\+?\s+(?:[a-z-]+\s+){{0,2}}{count_noun}|"
                                        rf"\b{count_noun}\w*\s+(?:[a-z-]+\s+){{1,5}}\d{{1,3}}(?:,\d{{3}})+", text.lower()):
                b *= 3.0
            if re.search(r"\bnumber\b", low_q) and re.search(r"\b\d{7,}\b|\b\d{3}-\d{3}-\d{4}\b", text):
                b *= 2.0
            if asks_open and DATE_RE.search(text):
                b *= 1.8
            if qi.asks_hours and TIME_RE.search(text):
                b *= 2.0
            if qi.asks_address and ADDR_RE.search(text):
                b *= 1.6
            if qi.asks_phone and PHONE_RE.search(text):
                b *= 1.6
            if qi.asks_when and DATE_RE.search(text):
                b *= 1.6
            if qi.asks_count and NUM_RE.search(text):
                b *= 1.3
            return b

        seen_txt = set()
        rescored = []
        for cid in cand:
            rec = s.chunks[cid]
            if rec.chunk.kind == "table":
                continue
            key = rec.chunk.text[:160]
            if key in seen_txt:
                continue
            seen_txt.add(key)
            rescored.append((sc[cid] * pattern_bonus(rec.chunk.text), rec))
        rescored.sort(key=lambda x: -x[0])
        if count_noun and rescored:
            # Counts change over time (branches, members...). Among statements of the count, trust the
            # most recently published one; undated pages only when nothing dated states it.
            pat = re.compile(rf"(?<![\d,])\d{{1,3}}(?:,\d{{3}})?\+?\s+(?:[a-z-]+\s+){{0,2}}{count_noun}|"
                             rf"\b{count_noun}\w*\s+(?:[a-z-]+\s+){{1,5}}\d{{1,3}}(?:,\d{{3}})+")
            def states_own_count(text: str) -> bool:
                for sn in sentences(text):
                    if pat.search(sn.lower()) and not re.search(
                            r"nationwide|shared branch|co-op|participating|across the u\.?s|industry|"
                            r"in the (?:u\.?s|country|nation)", sn, re.I):
                        return True
                return False
            pool = self.top(sc, 250)
            stating = [(float(sc[c]), s.chunks[c]) for c in pool
                       if s.chunks[c].chunk.kind == "para" and states_own_count(s.chunks[c].chunk.text)]
            if stating:
                top_x = max(x for x, _ in stating)
                stating = [x for x in stating if x[0] >= 0.25 * top_x]
            dated = [(x, r) for x, r in stating if s.page_of(r).pub_date]
            if dated:
                
                newest = max(dated, key=lambda d: (s.page_of(d[1]).pub_date, d[0]))
                rescored = [newest] + [x for x in rescored if x[1].cid != newest[1].cid]
        if not rescored:
            return Result(REFUSAL_TEXT.format(what="that information"), confident=False, kind="refuse")
        best_score, best = rescored[0]
        page = s.page_of(best)
        ev_tokens = set(tokenize(f"{page.title} {best.chunk.heading} {best.chunk.text}"))
        if self.coverage(terms, ev_tokens) < 0.2:
            return Result(REFUSAL_TEXT.format(what="information about that"), confident=False, kind="refuse")

        text = self.extract(best.chunk.text, qi)
        heading_last = best.chunk.heading.split(">")[-1].strip()
        if best.chunk.kind == "row" and heading_last:
            text = f"{heading_last}: {self.fmt_row(best)}."
        srcs = [self._src(best, snap, text)]
        used = {best.url}
        keyvals = [v for v in (PHONE_RE.findall(text) + DATE_RE.findall(text) +
                               re.findall(r"\b\d{6,}\b", text))][:1]
        if keyvals:
            srcs += self.agreeing_sources(keyvals, [r.cid for _, r in rescored[:30]], snap, used)
        else:
            for sc2, rec in rescored[1:4]:
                if rec.url not in used and sc2 >= 0.7 * best_score and s.page_of(rec).ptype != "index":
                    srcs.append(self._src(rec, snap))
                    used.add(rec.url)
        return Result(text, srcs, True, None, [r for _, r in rescored[:8]], kind="passage")

    HOURS_BLOCK = re.compile(r"((?:Branch hours are:?\s*)?Lobby\b.*?)(?:ABOUT DUPACO|$)", re.I | re.S)

    def answer_hours(self, qi: QueryInfo, snap: int, sc, cand) -> Result | None:
        """General hours question: the site has no single schedule, so report the most common
        schedule(s) across branch announcements and say hours vary by branch."""
        s = self.s
        sc2 = self.scores(analyze("branch hours lobby drive-up monday friday saturday"), snap, 0.0)
        pool = self.top(sc2, 80)
        sched: dict[str, list] = defaultdict(list)
        for cid in pool:
            rec = s.chunks[cid]
            m = self.HOURS_BLOCK.search(rec.chunk.text)
            if not m or not TIME_RE.search(m.group(1)) or not re.search(r"monday|saturday", m.group(1), re.I):
                continue
            block = re.sub(r"\s+", " ", m.group(1)).strip()
            block = re.sub(r"^Branch hours are:?\s*", "", block)[:260]
            key = re.sub(r"[^a-z0-9:]", "", block.lower().replace("–", "-"))
            if rec.url not in [r.url for r in sched[key]]:
                sched[key].append(rec)
        if not sched:
            return None
        ranked = sorted(sched.values(), key=lambda v: (-len(v), v[0].url))
        blocks = []
        for recs in ranked[:3]:
            m = self.HOURS_BLOCK.search(recs[0].chunk.text)
            blocks.append(re.sub(r"^Branch hours are:?\s*", "", re.sub(r"\s+", " ", m.group(1)).strip())[:260])
        text = ("Hours vary by branch. The most common schedule is: " + blocks[0].rstrip(" .") + ".")
        for b in blocks[1:]:
            text += " Other branches use: " + b.rstrip(" .") + "."
        text += " Tell me which branch and I can check its hours."
        srcs = [self._src(r, snap) for recs in ranked[:2] for r in recs[:2]]
        return Result(text, srcs, True, kind="hours")

    def extract(self, chunk_text: str, qi: QueryInfo, budget: int = 520) -> str:
        if len(chunk_text) <= budget:
            return chunk_text
        sents = sentences(chunk_text)
        qset = {t for t, _ in qi.tokens}
        scored = []
        for i, sn in enumerate(sents):
            toks = set(tokenize(sn))
            sc = sum(self.s.idf(t) * w for t, w in qi.tokens if t in toks)
            if qi.asks_hours and TIME_RE.search(sn):
                sc += 6
            if (qi.asks_when or re.search(r"\b(open|close|closing)\b", qi.norm.lower())) and DATE_RE.search(sn):
                sc += 4
            if (qi.asks_count or qi.asks_rate) and NUM_RE.search(sn):
                sc += 1.5
            scored.append((sc, i, sn))
        # a sentence that follows a relevant one often carries the answer ("This initiative provides...")
        raw = [x[0] for x in scored]
        scored = [(sc + 0.5 * (raw[i - 1] if i else 0) + (0.3 if i < 2 else 0), i, sn) for sc, i, sn in scored]
        scored.sort(key=lambda x: (-x[0], x[1]))
        picked, total = [], 0
        for sc, i, sn in scored:
            if total + len(sn) > budget and picked:
                continue
            picked.append((i, sn))
            total += len(sn)
            if total > budget * 0.75:
                break
        if qi.asks_hours:   # keep the whole schedule block together
            idxs = [i for i, _ in picked]
            lo, hi = min(idxs), max(idxs)
            while hi + 1 < len(sents) and TIME_RE.search(sents[hi + 1]) and total < 700:
                hi += 1
                total += len(sents[hi])
            picked = [(i, sents[i]) for i in range(lo, hi + 1)]
        picked.sort()
        return " ".join(sn for _, sn in picked)


def row_signature(rec: ChunkRec) -> tuple:
    """Identity of a table row independent of its values (product + tier/term label)."""
    ch = rec.chunk
    labels = []
    for h, c in zip(ch.header + [""] * len(ch.cells), ch.cells):
        if "%" in c or DROP_COLS.search(h or ""):
            continue
        labels.append(re.sub(r"[^a-z0-9]+", " ", c.lower()).strip())
    cap = re.sub(r"[^a-z0-9]+", " ", ch.heading.split(">")[-1].lower()).strip()
    return (cap, tuple(labels))
