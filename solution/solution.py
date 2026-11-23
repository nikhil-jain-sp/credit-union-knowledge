"""
Knowledge system for the "what's true now" hackathon.

    build() -> CreditUnionKnowledge with ingest(snapshot), answer(question, as_of=None), conflicts()

Pipeline (see NOTES.md):
  ingest    every page version is parsed once into heading-aware passages and table rows (with
            'effective as of' dates); chunks are de-duplicated across snapshots and carry a bitmask of
            the snapshots they appear in, so `as_of` is a cheap filter.
  answer    BM25 over the as-of view, weighted by page authority (rate sheet > product page > landing >
            news > blog > calculator) and recency. Rate questions are answered from table rows matched
            on term / tier / model year / jumbo / IRA / business qualifiers; questions whose specifics
            the site does not cover get an explicit "I don't know". If ANTHROPIC_API_KEY is set, Claude
            rewrites the answer from the same evidence (optional; the deterministic answer is the fallback).
  conflicts table rows and numeric sentences aligned across snapshots (same page) and across pages,
            plus a few organisation-level facts from press-release boilerplate, each resolved by
            effective date / authority / recency.
"""
from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:  # use the harness's own dataclasses when available
    _h = next((d / "harness" for d in HERE.parents if (d / "harness" / "contract.py").exists()), None)
    if _h is not None:
        sys.path.insert(0, str(_h))
    from contract import Answer, Conflict, Source  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover - fallback mirrors harness/contract.py
    @dataclass
    class Source:
        url: str
        snapshot_id: Optional[str] = None
        quote: Optional[str] = None

    @dataclass
    class Answer:
        text: str
        sources: list = field(default_factory=list)
        confident: bool = True
        conflict_note: Optional[str] = None

    @dataclass
    class Conflict:
        topic: str
        values: list
        sources: list
        resolution: Optional[str] = None
        reason: Optional[str] = None

from cuknow.conflicts import find_conflicts  # noqa: E402
from cuknow.engine import Engine  # noqa: E402
from cuknow.query import analyze  # noqa: E402
from cuknow.store import Store  # noqa: E402


class CreditUnionKnowledge:
    def __init__(self):
        self.store = Store()
        self.engine = Engine(self.store)
        self._cache: dict = {}
        self._lock = threading.Lock()
        self._conflicts = None
        self._llm = None
        if os.getenv("ANTHROPIC_API_KEY") and os.getenv("CUKNOW_DISABLE_LLM") != "1":
            try:
                from cuknow.llm import LLMComposer
                self._llm = LLMComposer()
            except Exception as e:  # missing package etc. -> deterministic mode
                print(f"[cuknow] LLM disabled: {e}", file=sys.stderr)

    # ------------------------------------------------------------------ contract
    def ingest(self, snapshot) -> None:
        with self._lock:
            self.store.add_snapshot(snapshot)
            self._cache.clear()
            self._conflicts = None

    @staticmethod
    def _route_to_llm(question: str, r) -> bool:
        """Only free-text answers go to the LLM. Rate lookups, hours aggregation, links, counts and
        refusals are computed by the engine from structure the evidence passages don't show, and a
        rewrite there was measured to lose accuracy (0.83 -> 0.75 on the public set)."""
        if r.kind != "passage":
            return False
        qi = analyze(question)
        return not (qi.asks_count or qi.asks_phone or qi.asks_address or qi.asks_hours or qi.asks_link)

    def answer(self, question: str, as_of: Optional[str] = None):
        key = (" ".join(question.lower().split()), as_of)
        with self._lock:
            if key in self._cache:           # asking twice gives the same answer
                return self._cache[key]
            self.store.build()
        r = self.engine.answer(question, as_of)
        if self._llm is not None and self._route_to_llm(question, r):
            try:
                r = self._llm.compose(question, as_of, r, self.engine)
            except Exception as e:
                print(f"[cuknow] LLM failed, using deterministic answer: {e}", file=sys.stderr)
        ans = Answer(
            text=r.text,
            sources=[Source(url=u, snapshot_id=sid, quote=q) for u, sid, q in r.sources],
            confident=r.confident,
            conflict_note=r.conflict_note,
        )
        with self._lock:
            self._cache[key] = ans
        return ans

    def conflicts(self):
        with self._lock:
            self.store.build()
            if self._conflicts is None:
                self._conflicts = [
                    Conflict(topic=c.topic, values=c.values,
                             sources=[Source(url=u, snapshot_id=sid, quote=q) for u, sid, q in c.sources],
                             resolution=c.resolution, reason=c.reason)
                    for c in find_conflicts(self.store)
                ]
            return list(self._conflicts)


def build():
    return CreditUnionKnowledge()
