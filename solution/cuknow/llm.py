"""Optional Claude answer composer (enabled when ANTHROPIC_API_KEY is set).

The deterministic engine stays in charge of *what is true*: it picks the view (as_of snapshot), ranks
evidence by authority / recency / effective date, and resolves stale values. Claude only rewrites the
answer from that evidence - and guardrails reject any rewrite that reintroduces a value the engine
already identified as stale, or that states numbers in a refusal. Any API error -> deterministic answer.

Env:
  CUKNOW_MODEL   model id (default claude-opus-5-5)
  CUKNOW_EFFORT  low|medium|high (default low - answers must stay fast)
"""
from __future__ import annotations

import json
import os
import re

import anthropic

from .engine import PCT_IN, Result
from .query import analyze

SYSTEM = """You answer credit-union members' questions using ONLY the evidence passages provided, \
which come from the credit union's own website as it stood on the date given.

Rules:
- Use the CURRENT value. Evidence is ranked: official rate sheet > product page > landing/promo page > \
press release > blog > calculator. A newer 'effective' date beats an older one; a newer press release \
beats an older one. Never repeat an outdated or superseded value, not even to say it is outdated.
- Respect qualifiers: term (6-month vs 12-month), tier (balance or loan amount, model year), rate vs APY/APR, \
regular vs jumbo vs special, personal vs business, IRA vs regular certificate. Give the tier that matches the \
question; if the member did not specify one, list the tiers briefly (lead with the most common one).
- The 'Engine answer' is a careful deterministic lookup. Prefer its values unless the evidence clearly shows \
it picked the wrong product or tier.
- If the evidence does not contain the answer, set confident=false and say you don't know and that the member \
should contact the credit union. A refusal must not contain any digits.
- Be concise: 1-3 sentences, plain text, no markdown. Include the key value(s) verbatim (e.g. "5.39% APR").
- source_urls: the URLs of the passages you used (copied exactly)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confident": {"type": "boolean"},
        "source_urls": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "confident", "source_urls"],
    "additionalProperties": False,
}


class LLMComposer:
    def __init__(self):
        self.model = os.getenv("CUKNOW_MODEL", "claude-opus-5-5")
        self.effort = os.getenv("CUKNOW_EFFORT", "low")
        self.client = anthropic.Anthropic(timeout=20.0, max_retries=1)
        self._use_fallbacks = True

    # ------------------------------------------------------------------ evidence
    def evidence(self, question: str, as_of, det: Result, engine, k: int = 10) -> tuple[list[dict], str]:
        s = engine.s
        qi = analyze(question)
        snap = engine.view(as_of, qi)
        sc = engine.scores(qi, snap)
        recs, seen = list(det.evidence[:6]), {r.cid for r in det.evidence[:6]}
        for cid in engine.top(sc, 40):
            r = s.chunks[cid]
            if cid in seen or r.chunk.kind == "table" or s.page_of(r).ptype == "index":
                continue
            recs.append(r)
            seen.add(cid)
            if len(recs) >= k:
                break
        out = []
        for r in recs:
            p = s.page_of(r)
            out.append({
                "url": r.url,
                "page_type": p.ptype,
                "published": p.pub_date.isoformat() if p.pub_date else None,
                "effective": r.chunk.effective.isoformat() if r.chunk.effective else None,
                "section": r.chunk.heading[:120],
                "text": r.chunk.text[:900],
            })
        return out, s.snaps[snap].taken_on.isoformat()

    # ------------------------------------------------------------------ compose
    def _call(self, prompt: str):
        kwargs = dict(
            model=self.model,
            max_tokens=2000,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        )
        if self._use_fallbacks:
            try:   # server-side refusal fallback (beta); dropped if the account/model rejects it
                return self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
            except anthropic.BadRequestError:
                self._use_fallbacks = False
        return self.client.messages.create(**kwargs)

    def compose(self, question: str, as_of, det: Result, engine) -> Result:
        ev, view_date = self.evidence(question, as_of, det, engine)
        prompt = (
            f"Website as of: {view_date}\n"
            f"Member question: {question}\n\n"
            f"Engine answer ({'refused' if not det.confident else det.kind}): {det.text}\n"
            f"Engine note on rejected values: {det.conflict_note or 'none'}\n\n"
            f"Evidence passages (JSON):\n{json.dumps(ev, ensure_ascii=False)}"
        )
        resp = self._call(prompt)
        if resp.stop_reason != "end_turn":
            return det
        text = next((b.text for b in resp.content if b.type == "text"), "")
        data = json.loads(text)
        answer = (data.get("answer") or "").strip()
        if not answer:
            return det

        # guardrails ----------------------------------------------------------------
        rejected = set(PCT_IN.findall(det.conflict_note or "")) - set(PCT_IN.findall(det.text))
        if any(v in answer for v in rejected):
            return det                                   # reintroduced a stale value
        if not data.get("confident", True):
            if re.search(r"\d", answer) or det.confident and det.kind == "rate":
                return det                               # refusal with numbers, or refusing a found rate
            return Result(answer, [], False, det.conflict_note, det.evidence, "refuse")
        if det.kind == "rate":
            det_vals = set(PCT_IN.findall(det.text))
            if det_vals and not (set(PCT_IN.findall(answer)) & det_vals):
                return det                               # disagrees with the rate-sheet lookup

        known = {e["url"] for e in ev}
        srcs = [(u, None, None) for u in data.get("source_urls", []) if u in known]
        for src in det.sources:
            if src[0] not in {x[0] for x in srcs}:
                srcs.append(src)
        return Result(answer, srcs[:6], True, det.conflict_note, det.evidence, det.kind)
