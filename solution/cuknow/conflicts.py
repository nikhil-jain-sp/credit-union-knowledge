"""Find places where the website contradicts itself and decide which value is right.

Three detectors, each producing (topic, values, sources, resolution, reason):

1. temporal   the same table row / the same sentence on the same page shows different numbers in
              different snapshots          -> the newest snapshot wins ("page was updated").
2. cross-page in one snapshot, the same product row appears on several pages with different numbers
              -> the page with the newest 'effective as of' date wins, then the more authoritative
                 page type (rate sheet > product page > landing page > calculator).
3. org facts  organisation-level counts and recognitions repeated in press-release boilerplate
              ("22 branch office locations", "fifth time Forbes recognized") -> newest release wins.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from .engine import DROP_COLS, PCT_IN, row_model_years, row_signature
from .parse import AUTHORITY
from .store import ChunkRec, Store
from .text import sentences, tokenize

FACT_PAGES = ("rates", "product", "landing", "pdf", "contact")
NOISE_SENT = re.compile(r"if you borrow|over the life of the loan|payment example|minimum amount required|"
                        r"redemptions? start|\bexample\b", re.I)


def _num(v: str) -> float:
    try:
        return float(re.sub(r"[^\d.]", "", v) or 0)
    except ValueError:
        return 0.0


def _footnote_variant(a: str, b: str) -> bool:
    """'$3002' vs '$300' (footnote digit glued on) is a formatting artefact, not a change."""
    da, db = re.sub(r"\D", "", a), re.sub(r"\D", "", b)
    return da.startswith(db) and len(da) == len(db) + 1 or db.startswith(da) and len(db) == len(da) + 1


MONEY_PCT = re.compile(r"\$\s?\d[\d,]*(?:\.\d\d)?|\d+(?:\.\d+)?\s?%")


@dataclass
class Found:
    topic: str
    values: list[str]
    sources: list[tuple[str, str | None, str | None]]
    resolution: str | None
    reason: str
    weight: float


def _pcts(rec: ChunkRec) -> list[str]:
    out = []
    for h, c in zip(rec.chunk.header + [""] * len(rec.chunk.cells), rec.chunk.cells):
        if DROP_COLS.search(h or ""):
            continue
        out += [p.replace(" ", "") for p in PCT_IN.findall(c)]
    return out


def _family(h: str) -> str:
    h = (h or "").lower()
    if "yield" in h or "apy" in h:
        return "APY"
    if "apr" in h or "percentage rate" in h:
        return "APR"
    if "dividend" in h:
        return "dividend rate"
    if "rate" in h:
        return "rate"
    return h[:20] or "value"


def _labels_key(rec: ChunkRec) -> tuple:
    labs = []
    for h, c in zip(rec.chunk.header + [""] * len(rec.chunk.cells), rec.chunk.cells):
        if "%" in c or DROP_COLS.search(h or ""):
            continue
        labs.append(re.sub(r"[^a-z0-9]+", " ", c.lower().replace(",", "")).strip())
    return tuple(labs)


def _label(rec: ChunkRec) -> str:
    cap = rec.chunk.heading.split(">")[-1].strip()
    lab = " ".join(c for c in rec.chunk.cells if c and "%" not in c and not c.startswith("$") or
                   (c.startswith("$") and "-" in c))
    return re.sub(r"\s+", " ", f"{cap} {lab}").strip()[:120]


def _metric_values(rec: ChunkRec) -> dict[str, str]:
    """{column header: % value} for a row."""
    out = {}
    for h, c in zip(rec.chunk.header + [""] * len(rec.chunk.cells), rec.chunk.cells):
        if DROP_COLS.search(h or ""):
            continue
        h = _family(h)
        rng = re.search(r"\d+(?:\.\d+)?%\s*-\s*\d+(?:\.\d+)?%", c)
        m = PCT_IN.search(c)
        if rng:
            out[h] = rng.group(0).replace(" ", "")
        elif m:
            out[h] = m.group(0).replace(" ", "")
    return out


def find_conflicts(store: Store, limit: int = 45) -> list[Found]:
    found: list[Found] = []
    found += _temporal_rows(store)
    found += _cross_page_rows(store)
    found += _temporal_sentences(store)
    found += _org_facts(store)
    # de-duplicate: same value set on the same topic is reported once (keep the strongest)
    best: dict[tuple, Found] = {}
    for f in found:
        key = (frozenset(f.values), re.sub(r"[^a-z0-9]", "", f.topic.lower())[:40])
        key2 = frozenset(f.values)
        if key2 in {k[0] for k in best} and any(k[0] == key2 and _similar(k[1], key[1]) for k in best):
            continue
        if key not in best or best[key].weight < f.weight:
            best[key] = f
    out = sorted(best.values(), key=lambda f: -f.weight)
    per_group: dict[str, int] = defaultdict(int)
    kept = []
    for f in out:
        g = (f.sources[0][0], re.sub(r"[^a-z]", "", f.topic.lower())[:12])
        if per_group[g] >= 3:
            continue
        per_group[g] += 1
        kept.append(f)
    return kept[:limit]


def _similar(a: str, b: str) -> bool:
    return a[:15] == b[:15]


# ---------------------------------------------------------------------------------------------
def _sid(store: Store, idx: int) -> str:
    return store.snaps[idx].sid


def _temporal_rows(store: Store) -> list[Found]:
    """Same page + same row signature, values change between snapshots."""
    out = []
    for url, hist in store.url_versions.items():
        page0 = store.versions[hist[0][1]]
        if page0.ptype not in FACT_PAGES:
            continue
        # row signature -> [(snap idx, metric->value, rec)]
        series: dict[tuple, list] = defaultdict(list)
        for snap_idx, vkey in hist:
            page = store.versions[vkey]
            for ch in page.chunks:
                if ch.kind != "row" or not PCT_IN.search(ch.text):
                    continue
                cid = store._chunk_key.get((url, ch.kind, ch.heading, ch.text))
                if cid is None:
                    continue
                rec = store.chunks[cid]
                series[_labels_key(rec)].append((snap_idx, _metric_values(rec), rec))
        for sig, pts in series.items():
            if not any(sig) or len({p[0] for p in pts}) < len(pts):
                continue      # unlabeled or ambiguous rows (several rows share the label in one version)
            pts.sort(key=lambda p: p[0])
            # use the most informative metric: APY, then APR, then any
            metrics = set().union(*(p[1].keys() for p in pts))
            metric = next((m for m in ("APY", "APR", "rate", "dividend rate") if m in metrics), None) or \
                next(iter(sorted(metrics)), None)
            if not metric:
                continue
            seq = []
            for snap_idx, mv, rec in pts:
                v = mv.get(metric)
                if v and (not seq or seq[-1][1] != v):
                    seq.append((snap_idx, v, rec))
            if len({v for _, v, _ in seq}) < 2:
                continue
            first_idx = {}
            for snap_idx, v, rec in seq:
                first_idx.setdefault(v, snap_idx)
            latest = seq[-1]
            # the row must still be on the latest version of the page to resolve to it
            vals = []
            for _, v, _ in seq:
                if v not in vals:
                    vals.append(v)
            for _, mv, _ in pts:          # other metrics of the same row (e.g. rate next to APR)
                for m2, v2 in mv.items():
                    if m2 != metric and v2 not in vals and "-" not in v2:
                        vals.append(v2)
            label = _label(latest[2])
            srcs = [(url, _sid(store, si), f"{_label(r)}: {v}") for si, v, r in seq]
            reason = (f"{url} changed this value over time ({' -> '.join(f'{v} [{_sid(store, si)[:13]}]' for si, v, _ in seq)});"
                      f" the newest snapshot is current")
            w = 3.0 * AUTHORITY.get(page0.ptype, 0.5) + 0.2 * len(seq)
            cur = " / ".join(v for v in latest_metrics(pts[-1][1]))
            out.append(Found(f"{label} ({_short(metric)})", vals, srcs, cur or latest[1], reason, w))
    return out


def _short(metric: str) -> str:
    m = metric.lower()
    return "APY" if "yield" in m or m == "apy" else "APR" if "apr" in m or "percentage rate" in m else metric


def latest_metrics(mv: dict) -> list[str]:
    return [v for v in mv.values() if "-" not in v]


def _my_key(rec: ChunkRec) -> tuple:
    """Row identity for cross-page matching. Model-year ranges are matched by overlap later."""
    cap, labels = row_signature(rec)
    labels = tuple(re.sub(r"\b(19|20)\d\d\s*\d{2,4}\b", "<my>", l) for l in labels)
    return labels


def _cross_page_rows(store: Store) -> list[Found]:
    out = []
    last = len(store.snaps) - 1
    bit = 1 << last
    groups: dict[tuple, list[ChunkRec]] = defaultdict(list)
    for rec in store.chunks:
        if not (rec.mask & bit) or rec.chunk.kind != "row" or not PCT_IN.search(rec.chunk.text):
            continue
        page = store.page_of(rec)
        if page.ptype in ("index", "blog", "news", "resource"):
            continue
        key = _my_key(rec)
        if not key or all(not k for k in key):
            continue
        groups[key].append(rec)
    for key, recs in groups.items():
        urls = {r.url for r in recs}
        if len(urls) < 2:
            continue
        # pick the winner: newest effective date, then authority
        def rank(r):
            return ((r.chunk.effective or date.min), AUTHORITY.get(store.page_of(r).ptype, 0.5))
        winner = max(recs, key=rank)
        wv = _pcts(winner)
        if not wv:
            continue
        wy = row_model_years(winner.chunk.text)
        losers = []
        for r in recs:
            if r.url == winner.url:
                continue
            ry = row_model_years(r.chunk.text)
            if (wy is None) != (ry is None):
                continue
            if wy and ry:
                overlap = min(wy[1], ry[1]) - max(wy[0], ry[0]) + 1
                if overlap < 2:
                    continue
            # same product: caption words must overlap
            cw = set(tokenize(winner.chunk.heading.split(">")[-1])) | set(tokenize(" ".join(winner.chunk.cells)))
            cr = set(tokenize(r.chunk.heading.split(">")[-1])) | set(tokenize(" ".join(r.chunk.cells)))
            if not (cw & cr - {"month", "year", "apr"}):
                continue
            rv = _pcts(r)
            if rv and rv[0] != wv[0]:
                losers.append((r, rv[0]))
        if not losers:
            continue
        lv = []
        for r, _ in losers:
            for v in _pcts(r):
                if v not in wv and v not in lv:
                    lv.append(v)
        vals = wv + lv
        srcs = [(winner.url, _sid(store, last), f"{_label(winner)}: {wv[0]}")] + \
               [(r.url, _sid(store, last), f"{_label(r)}: {v}") for r, v in losers]
        lp = {store.page_of(r).ptype for r, _ in losers}
        reason = (f"{winner.url} ({store.page_of(winner).ptype} page"
                  f"{', effective ' + winner.chunk.effective.isoformat() if winner.chunk.effective else ''})"
                  f" is authoritative; {', '.join(sorted(lp))} page(s) show stale values")
        w = 4.0 + 0.5 * len(losers)
        out.append(Found(_label(winner), vals, srcs, " / ".join(wv), reason, w))
    return out


def _temporal_sentences(store: Store) -> list[Found]:
    """A sentence on a product page whose $/% figure changed between snapshots."""
    out = []
    for url, hist in store.url_versions.items():
        page0 = store.versions[hist[0][1]]
        if page0.ptype not in ("product", "landing", "rates", "contact"):
            continue
        prev_sents: list[tuple[set, str, str, int]] = []
        reported = set()
        for snap_idx, vkey in hist:
            page = store.versions[vkey]
            cur = []
            for ch in page.chunks:
                if ch.kind != "para":
                    continue
                for sn in sentences(ch.text):
                    nums = MONEY_PCT.findall(sn)
                    if not nums or len(sn) > 400 or NOISE_SENT.search(sn):
                        continue
                    cur.append((set(tokenize(re.sub(r"[\d$%.,]", " ", sn))), sn, tuple(n.replace(" ", "") for n in nums),
                                snap_idx))
            if prev_sents:
                prev_texts = {p[1] for p in prev_sents}
                for toks, sn, nums, si in cur:
                    if sn in prev_texts or len(toks) < 5:
                        continue
                    for ptoks, psn, pnums, psi in prev_sents:
                        if not ptoks:
                            continue
                        j = len(toks & ptoks) / len(toks | ptoks)
                        if j < 0.6 or set(nums) == set(pnums):
                            continue
                        if len(nums) != len(pnums):
                            continue
                        pairs = [(a, b) for a, b in zip(pnums, nums) if _num(a) != _num(b)
                                 and a[0] == b[0] and not _footnote_variant(a, b)]
                        if len(pairs) != 1:
                            continue
                        changed_old, changed_new = [pairs[0][0]], [pairs[0][1]]
                        key = (changed_old[0], changed_new[0])
                        if key in reported:
                            break
                        reported.add(key)
                        topic = " ".join(sn.split()[:12])
                        out.append(Found(
                            topic, [changed_old[0], changed_new[0]],
                            [(url, _sid(store, psi), psn[:200]), (url, _sid(store, si), sn[:200])],
                            changed_new[0],
                            f"{url} was updated between {_sid(store, psi)} and {_sid(store, si)}; newest wording is current",
                            2.0))
                        break
            prev_sents = cur
    return out


ORG_PATTERNS = [
    ("number of branch office locations", re.compile(r"(\d{1,3}) branch office locations")),
    ("number of members (membership)", re.compile(r"membership has grown to (?:more than |almost |nearly )?([\d,]{6,})")),
    ("total assets", re.compile(r"assets (?:exceeding|of nearly|of) (\$[\d.]+ billion)")),
    ("times Forbes recognized Dupaco", re.compile(r"(first|second|third|fourth|fifth|sixth|seventh) time Forbes recognized")),
]


def _org_facts(store: Store) -> list[Found]:
    out = []
    last = len(store.snaps) - 1
    bit = 1 << last
    for topic, pat in ORG_PATTERNS:
        obs = []   # (pub date, value, url)
        for rec in store.chunks:
            if not (rec.mask & bit) or rec.chunk.kind != "para":
                continue
            m = pat.search(rec.chunk.text)
            if not m:
                continue
            page = store.page_of(rec)
            obs.append((page.pub_date or date.min, m.group(1), rec.url))
        vals = {v for _, v, _ in obs}
        if len(vals) < 2:
            continue
        obs.sort()
        newest = obs[-1]
        ordered = []
        for d, v, u in sorted(obs, key=lambda o: o[0], reverse=True):
            if v not in ordered:
                ordered.append(v)
        srcs = []
        for v in ordered:
            d, _, u = max(o for o in obs if o[1] == v)
            srcs.append((u, _sid(store, last), f"{topic}: {v} (published {d.isoformat() if d != date.min else 'undated'})"))
        ordered, srcs = ordered[:5], srcs[:5]
        out.append(Found(topic, ordered, srcs, newest[1],
                         f"press-release boilerplate repeats older figures; the newest release "
                         f"({newest[0].isoformat()}, {newest[2]}) says {newest[1]}", 3.5))
    return out
