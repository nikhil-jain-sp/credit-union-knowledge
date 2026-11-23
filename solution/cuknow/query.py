"""Question analysis: intent, qualifiers (term, tier, model year, amount...), date hints."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from .text import MONTHS, NUM_WORDS, STOP, month_year_in_question, query_tokens

RATE_WORDS = re.compile(r"\b(rates?|aprs?|apys?|yields?|dividends?|interest|earn|pays?|paying)\b", re.I)
LIST_WORDS = re.compile(r"\b(all|list|every|each|what (?:kinds?|types?)|(?:which|what) .{0,40}(?:do you|does dupaco) "
                        r"(?:offer|have)|options|rates\b(?! on| for a))", re.I)


@dataclass
class QueryInfo:
    raw: str
    norm: str
    tokens: list[tuple[str, float]]
    as_of_hint: date | None = None
    asks_rate: bool = False
    metric: str | None = None            # 'apy' | 'apr' | 'dividend' | None
    months: set[int] = field(default_factory=set)
    years: set[int] = field(default_factory=set)
    model_year: int | None = None
    amounts: list[float] = field(default_factory=list)
    jumbo: bool = False
    ira: bool = False
    business: bool = False
    special: bool = False
    superlative: str | None = None       # 'max' | 'min'
    asks_list: bool = False
    asks_link: bool = False
    asks_hours: bool = False
    asks_address: bool = False
    asks_phone: bool = False
    asks_count: bool = False
    asks_when: bool = False
    asks_yesno: bool = False
    proper_nouns: list[str] = field(default_factory=list)


def _normalize(q: str) -> str:
    s = q.replace("–", "-").replace("—", "-").replace("’", "'")
    for w, n in NUM_WORDS.items():
        s = re.sub(rf"\b{w}\b", n, s, flags=re.I)
    return s


def analyze(q: str) -> QueryInfo:
    norm = _normalize(q)
    low = norm.lower()
    qi = QueryInfo(raw=q, norm=norm, tokens=query_tokens(norm))
    qi.as_of_hint = month_year_in_question(norm)

    qi.asks_link = bool(re.search(r"\b(link|url|web ?page|website address|web address|which page|"
                                  r"where (?:can|do|should) i (?:go|find|see|look|apply|sign up|join)|"
                                  r"online\b.*\b(?:join|apply|sign up|become a member)|"
                                  r"(?:join|apply|sign up|become a member)\b.*\bonline)\b", low))
    qi.asks_hours = bool(re.search(r"\bhours?\b|\bopen (?:on|at|until)\b|\bclose[sd]? at\b|"
                                   r"what time", low))
    qi.asks_address = bool(re.search(r"\baddress\b|\blocated\b|\bwhere is\b", low))
    qi.asks_phone = bool(re.search(r"\bphone\b|\bcall\b|\bnumber to\b|\btelephone\b", low))
    qi.asks_count = bool(re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", low))
    qi.asks_when = bool(re.search(r"^\s*when\b|\bwhat date\b|\bwhich date\b", low))
    qi.asks_yesno = bool(re.match(r"\s*(is|are|do|does|did|can|could|will|has|have)\b", low))

    qi.asks_rate = bool(RATE_WORDS.search(low)) and not qi.asks_link and not re.search(
        r"\brouting\b|\bexchange rate\b", low)
    if re.search(r"\bapys?\b|\byield\b", low):
        qi.metric = "apy"
    elif re.search(r"\baprs?\b", low):
        qi.metric = "apr"
    elif re.search(r"\bdividend\b", low):
        qi.metric = "dividend"

    qi.months = {int(m) for m in re.findall(r"\b(\d{1,3})\s*-?\s*(?:months?|mos?)\b", low)}
    qi.years = {int(y) for y in re.findall(r"\b(\d{1,2})\s*-?\s*(?:years?|yrs?)\b(?!\s*old)", low)}
    if re.search(r"\b(auto|car|vehicle|truck|motorcycle|model[- ]year|suv)\b", low):
        ys = [int(y) for y in re.findall(r"\b(20[0-3]\d|19[89]\d)\b", low)]
        if ys:
            qi.model_year = ys[0]
    for m in re.finditer(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(k|thousand|million|m)?\b", low):
        v = float(m.group(1).replace(",", ""))
        if m.group(2) in ("k", "thousand"):
            v *= 1000
        elif m.group(2) in ("m", "million"):
            v *= 1_000_000
        qi.amounts.append(v)
    qi.jumbo = "jumbo" in low
    qi.ira = bool(re.search(r"\b(ira|iras|roth|traditional ira|retirement)\b", low))
    qi.business = bool(re.search(r"\bbusiness", low))
    qi.special = bool(re.search(r"\b(special|promo|promotion|promotional)\b", low))
    if re.search(r"\b(highest|best|top|maximum|max|most)\b", low):
        qi.superlative = "max"
    elif re.search(r"\b(lowest|cheapest|minimum|min|least)\b", low):
        qi.superlative = "min"
    qi.asks_list = bool(LIST_WORDS.search(low)) or bool(re.search(r"\brates\b", low) and not (
        qi.months or qi.years or qi.model_year or qi.amounts))

    # Capitalised words that are not sentence-initial: candidate proper nouns ("Mullan").
    words = re.findall(r"(?<![.?!]\s)(?<!^)\b([A-Z][a-z]{2,})\b", q.strip())
    skip = {"dupaco", "visa", "platinum", "rewards", "apy", "apr"} | STOP | set(MONTHS) | {
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
    qi.proper_nouns = [w for w in words if w.lower() not in skip]
    if qi.proper_nouns:   # names carry the question: weight them up
        pn = {t for w in qi.proper_nouns for t in query_tokens(w) for t in [t[0]]}
        qi.tokens = [(t, w * 1.6 if t in pn else w) for t, w in qi.tokens]
    return qi
