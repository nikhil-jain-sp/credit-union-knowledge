"""Text utilities: markdown cleaning, tokenising, dates and numeric values."""
from __future__ import annotations

import re
from datetime import date

# ---------------------------------------------------------------------------
# Markdown cleaning
# ---------------------------------------------------------------------------
IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK_RE = re.compile(r"\[([^\]]*)\]\((\S*?)(?:\s+\"[^\"]*\")?\)")
FOOTNOTE_LINK_RE = re.compile(r"\s?\[(\d)\]\([^)]*\)")         # "[1](...#fine-print)"
ESCAPES_RE = re.compile(r"\\([\\_*#|$\[\]()~`>+\-.!])")
CRLF_LITERAL_RE = re.compile(r"(\\\\r\\\\n|\\r\\n|\\n)+")
SPACES_RE = re.compile(r"[ \t ]+")


def strip_markdown_line(line: str, links: list | None = None) -> str:
    """Remove images, turn links into their text, drop escapes. Collects (text, url) into `links`."""
    line = IMG_RE.sub("", line)
    line = FOOTNOTE_LINK_RE.sub("", line)
    if links is not None:
        for m in LINK_RE.finditer(line):
            links.append((m.group(1).strip(" *\\"), m.group(2)))
    line = LINK_RE.sub(lambda m: m.group(1), line)
    line = CRLF_LITERAL_RE.sub(" ", line)
    line = ESCAPES_RE.sub(r"\1", line)
    line = line.replace("**", "").replace("__", "")
    line = SPACES_RE.sub(" ", line)
    return line.strip()


FLAG_RE = re.compile("[\U0001F1E6-\U0001F1FF]")


def is_junk_line(line: str) -> bool:
    """Chart axis dumps, map widgets, form widgets and other text with no prose value."""
    s = line.strip()
    if not s:
        return True
    if FLAG_RE.search(s):
        return True
    longest = max((len(t) for t in s.split()), default=0)
    if longest > 60 and not s.startswith("http"):
        return True                      # "$1K$1K$1K..." chart labels
    if s in {"* * *", "---", "Loading locations...", "Go", "Cancel", "Continue"}:
        return True
    if re.fullmatch(r"[\W_]+", s):
        return True
    return False


# ---------------------------------------------------------------------------
# Numbers / values
# ---------------------------------------------------------------------------
PCT_RE = re.compile(r"(?<![\d.])(\d{1,2}(?:\.\d{1,3})?)\s?%")
MONEY_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{2}))?")
FOOTNOTE_AFTER_PCT = re.compile(r"(\d%)(\d)(?!\d)")        # "0.20%1" -> "0.20%"
FOOTNOTE_AFTER_WORD = re.compile(r"\b(APR|APY|Rate|Yield|\))(\d)(?!\d)")


def clean_cell(c: str) -> str:
    c = FOOTNOTE_AFTER_PCT.sub(r"\1", c.strip())
    c = FOOTNOTE_AFTER_WORD.sub(r"\1", c)
    c = c.replace("–", "-").replace("—", "-")
    return c.strip()


def pct_values(text: str) -> list[float]:
    return [float(m.group(1)) for m in PCT_RE.finditer(text or "")]


def fmt_pct(v: str) -> str:
    return v


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------
MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
MONTH_ABBR = {k[:3]: v for k, v in MONTHS.items()}
LONG_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2}),?\s+(20\d\d|19\d\d)\b")
NUM_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(20\d\d)\b")
EFFECTIVE_RE = re.compile(r"effective (?:as of )?(\d{1,2}/\d{1,2}/20\d\d)", re.I)
URL_DATE_RE = re.compile(r"/(20\d\d|19\d\d)/(\d\d)/(?:(\d\d)/)?")
MONTH_YEAR_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+(?:(\d{1,2}),?\s+)?(20\d\d)\b", re.I)


def parse_long_date(s: str) -> date | None:
    m = LONG_DATE_RE.search(s or "")
    if not m:
        return None
    try:
        return date(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2)))
    except ValueError:
        return None


def parse_num_date(s: str) -> date | None:
    m = NUM_DATE_RE.search(s or "")
    if not m:
        return None
    try:
        return date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
    except ValueError:
        return None


def parse_iso(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def month_year_in_question(q: str) -> date | None:
    """'as of May 2026' / 'in June 2026' -> last day of that month (used when as_of is absent)."""
    m = re.search(r"\b(?:as of|in|during|back in|on)\s+" + MONTH_YEAR_RE.pattern, q, re.I)
    if not m:
        return None
    mon = MONTH_ABBR[m.group(1)[:3].lower()]
    year = int(m.group(3))
    day = int(m.group(2)) if m.group(2) else 28
    if not m.group(2):
        import calendar
        day = calendar.monthrange(year, mon)[1]
    try:
        return date(year, mon, day)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Tokenising
# ---------------------------------------------------------------------------
STOP = set("""
a an the and or of to in on for with at by from is are was were be been being it its this that these those
what whats which who whom whose when where why how do does did can could would should will shall may might must
i me my we our you your yours he she they them their there here than then so if as about into over under
any all some each per vs via not no yes get got give tell please know find show want need like also just
has have had dupaco dupacos credit union cu current currently right now today offer offers offering much many more most
""".split())

NUM_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
             "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "fifteen": "15",
             "eighteen": "18", "twenty": "20", "thirty": "30", "thirty-six": "36", "forty-eight": "48",
             "sixty": "60", "seventy-two": "72", "eighty-four": "84"}

SYNONYMS = {
    "cd": ["certificate"], "cds": ["certificate"], "certificates": ["certificate"],
    "mortgage": ["home", "loan", "conventional"], "mortgages": ["home", "loan", "conventional"],
    "heloc": ["home", "equity", "line"], "apy": ["yield"], "apr": ["annual", "percentage"],
    "car": ["auto", "vehicle"], "vehicle": ["auto"], "automobile": ["auto"],
    "savings": ["saving"], "checking": ["check"], "atm": ["atm"],
    "lobby": ["lobby"], "hours": ["hours"], "phone": ["call"], "telephone": ["call", "phone"],
    "card": ["visa"], "cards": ["visa", "card"], "visa": ["card"],
    "routing": ["aba"], "aba": ["routing"],
    "dividend": ["dividend"], "interest": ["dividend"],
    "join": ["membership", "member"], "membership": ["member", "join"], "member": ["membership"], "eligible": ["eligibility", "charter"], "eligibility": ["eligible", "charter"],
    "fee": ["fees", "charge"], "fees": ["fee", "charge"],
}

TOKEN_RE = re.compile(r"\$?\d+(?:[.,/]\d+)*%?|[a-z][a-z0-9'&$]*")
UNIT_WORDS = {"month": "mo", "months": "mo", "mo": "mo", "mos": "mo", "year": "yr", "years": "yr",
              "yr": "yr", "yrs": "yr"}


_SUFFIXES = ("ations", "ation", "itions", "ition", "izing", "ized", "izes", "ize", "ings", "ing", "ed")


def _stem(t: str) -> str:
    """Light stemmer: plurals plus a few derivational suffixes (recognition/recognized -> recogn)."""
    if not t[:1].isalpha() or "$" in t:
        return t
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    for suf in _SUFFIXES:
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            t = t[: -len(suf)]
            if t.endswith(("iz", "it")) and len(t) > 5:
                t = t[:-2]
            return t
    if len(t) > 4 and t.endswith(("ches", "shes", "sses", "xes", "zzes")):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss") and not t.endswith("us"):
        return t[:-1]
    return t


def tokenize(text: str, for_query: bool = False) -> list[str]:
    """Lower-cased word/number tokens. Emits fused number+unit tokens ('60mo', '30yr')."""
    s = (text or "").lower().replace("–", "-").replace("—", "-").replace("’", "'")
    for w, n in NUM_WORDS.items():
        if w in s:
            s = re.sub(rf"\b{w}\b", n, s)
    raw = TOKEN_RE.findall(s)
    out: list[str] = []
    prev = None
    for t in raw:
        t = t.strip("'")
        if t.endswith("'s"):
            t = t[:-2]
        if not t:
            continue
        if t[0].isdigit() or t[0] == "$":
            t = t.rstrip(".,")
            if prev is not None:
                pass
            out.append(t.replace(",", ""))
            prev = t
            continue
        u = UNIT_WORDS.get(t)
        if u and prev is not None and re.fullmatch(r"\d+", prev):
            out.append(prev + u)
        prev = None
        if t in STOP:
            continue
        t = _stem(t)
        if len(t) < 2:
            continue
        out.append(t)
    return out


def query_tokens(q: str) -> list[tuple[str, float]]:
    """Query terms with weights (synonym expansions get lower weight)."""
    base = tokenize(q, for_query=True)
    out: dict[str, float] = {}
    for t in base:
        out[t] = max(out.get(t, 0.0), 1.0)
        for syn in SYNONYMS.get(t, []):
            st = _stem(syn)
            if st not in out:
                out[st] = 0.4
    return list(out.items())


def norm_space(s: str) -> str:
    return SPACES_RE.sub(" ", s or "").strip()


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"“(])|\n+", text or "")
    return [p.strip() for p in parts if p and p.strip()]
