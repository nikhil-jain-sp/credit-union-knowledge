"""Versioned page store + BM25 index with per-snapshot visibility masks.

Every distinct page version is parsed once. Every distinct chunk (url, heading, text) is stored once
with a bitmask of the snapshots it appears in, so an `as_of` query simply filters by bit.
"""
from __future__ import annotations

import hashlib
import re
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .parse import AUTHORITY, Chunk, ParsedPage, parse_page
from .text import parse_iso, tokenize

EXPIRED_RE = re.compile(r"\b(?:offer|promotion|promo|special|deal)\s+(?:has\s+)?(?:ended|expired)|"
                        r"\bno longer (?:available|offered)|\bthis offer has expired", re.I)


@dataclass
class SnapInfo:
    sid: str
    taken_on: date
    bit: int
    pages: dict[str, str] = field(default_factory=dict)   # url -> version key
    carried: set = field(default_factory=set)             # urls carried over from an earlier crawl


@dataclass
class ChunkRec:
    cid: int
    url: str
    vkey: str            # page version where first seen
    chunk: Chunk
    mask: int = 0        # snapshots containing this chunk
    first_snap: int = 99
    last_snap: int = -1


class Store:
    K1, B = 1.2, 0.75

    def __init__(self):
        self.snaps: list[SnapInfo] = []
        self.versions: dict[str, ParsedPage] = {}       # vkey -> parsed page
        self.version_snaps: dict[str, int] = defaultdict(int)   # vkey -> mask
        self.url_versions: dict[str, list[tuple[int, str]]] = defaultdict(list)  # url -> [(snap idx, vkey)]
        self.chunks: list[ChunkRec] = []
        self._chunk_key: dict[tuple, int] = {}
        self._dirty = True
        self.titles: dict[str, str] = {}

    # ------------------------------------------------------------------ ingest
    def add_snapshot(self, snapshot) -> None:
        idx = len(self.snaps)
        info = SnapInfo(snapshot.snapshot_id, parse_iso(snapshot.taken_on) or date.today(), 1 << idx)
        for p in snapshot.pages:
            vkey = hashlib.sha1((p.url + "\0" + p.markdown).encode("utf-8")).hexdigest()[:16]
            info.pages[p.url] = vkey
            self.titles[p.url] = p.title or ""
            if vkey not in self.versions:
                self.versions[vkey] = parse_page(p.url, p.title or "", p.markdown)
            self.version_snaps[vkey] |= info.bit
            self.url_versions[p.url].append((idx, vkey))
            page = self.versions[vkey]
            for ch in page.chunks:
                key = (p.url, ch.kind, ch.heading, ch.text)
                cid = self._chunk_key.get(key)
                if cid is None:
                    cid = len(self.chunks)
                    self._chunk_key[key] = cid
                    self.chunks.append(ChunkRec(cid, p.url, vkey, ch))
                rec = self.chunks[cid]
                rec.mask |= info.bit
                rec.first_snap = min(rec.first_snap, idx)
                rec.last_snap = max(rec.last_snap, idx)
        # A page missing from this crawl keeps its last known version (the crawl may simply have
        # missed it); it is marked as carried so ranking can trust it a little less.
        if self.snaps:
            prev = self.snaps[-1]
            for url, vkey in prev.pages.items():
                if url in info.pages:
                    continue
                info.pages[url] = vkey
                info.carried.add(url)
                self.version_snaps[vkey] |= info.bit
                page = self.versions[vkey]
                for ch in page.chunks:
                    cid = self._chunk_key.get((url, ch.kind, ch.heading, ch.text))
                    if cid is not None:
                        self.chunks[cid].mask |= info.bit
        self.snaps.append(info)
        self._dirty = True

    # ------------------------------------------------------------------ index
    def build(self) -> None:
        if not self._dirty:
            return
        n = len(self.chunks)
        postings: dict[str, dict[int, int]] = defaultdict(dict)
        dl = np.zeros(n, dtype=np.float32)
        auth = np.zeros(n, dtype=np.float32)
        pub = np.zeros(n, dtype=np.float32)          # ordinal of publication date (0 = undated)
        eff = np.zeros(n, dtype=np.float32)          # ordinal of 'effective as of' date
        kind = np.zeros(n, dtype=np.int8)            # 0 para, 1 row, 2 table
        mask = np.zeros(n, dtype=np.int32)
        stale = np.ones(n, dtype=np.float32)
        for rec in self.chunks:
            page = self.versions[rec.vkey]
            toks = tokenize(rec.chunk.index_text(page.title))
            dl[rec.cid] = len(toks) or 1
            for t in toks:
                d = postings[t]
                d[rec.cid] = d.get(rec.cid, 0) + 1
            auth[rec.cid] = AUTHORITY.get(page.ptype, 0.5)
            pub[rec.cid] = page.pub_date.toordinal() if page.pub_date else 0
            eff[rec.cid] = rec.chunk.effective.toordinal() if rec.chunk.effective else 0
            kind[rec.cid] = {"para": 0, "row": 1, "table": 2}[rec.chunk.kind]
            mask[rec.cid] = rec.mask
            if EXPIRED_RE.search(rec.chunk.text) or page.url.rstrip("/").endswith("offer-expired"):
                stale[rec.cid] = 0.3
        self.post: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.df: dict[str, int] = {}
        for t, d in postings.items():
            ids = np.fromiter(d.keys(), dtype=np.int32, count=len(d))
            tfs = np.fromiter(d.values(), dtype=np.float32, count=len(d))
            self.post[t] = (ids, tfs)
            self.df[t] = len(d)
        self.dl, self.auth, self.pub, self.eff, self.kind, self.mask = dl, auth, pub, eff, kind, mask
        self.stale = stale
        self.avgdl = float(dl.mean()) if n else 1.0
        self.N = n
        self._dirty = False

    def idf(self, t: str) -> float:
        df = self.df.get(t, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def bm25(self, qtoks: list[tuple[str, float]]) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float32)
        norm = self.K1 * (1 - self.B + self.B * self.dl / self.avgdl)
        for t, w in qtoks:
            p = self.post.get(t)
            if p is None:
                continue
            ids, tfs = p
            idf = self.idf(t)
            scores[ids] += w * idf * tfs * (self.K1 + 1) / (tfs + norm[ids])
        return scores

    # ------------------------------------------------------------------ views
    def snap_index_for(self, as_of: date | None) -> int:
        if as_of is None or not self.snaps:
            return len(self.snaps) - 1
        best = 0
        for i, s in enumerate(self.snaps):
            if s.taken_on <= as_of:
                best = i
        return best

    def page_at(self, url: str, snap_idx: int) -> ParsedPage | None:
        vkey = self.snaps[snap_idx].pages.get(url)
        return self.versions.get(vkey) if vkey else None

    def page_of(self, rec: ChunkRec) -> ParsedPage:
        return self.versions[rec.vkey]
