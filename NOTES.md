# NOTES: a knowledge system that knows what's true *now*

Submission folder: **`solution/`** (`solution.py` → `build()`; the code lives in `solution/cuknow/`).

```bash
pip install -r requirements.txt          # numpy (+ anthropic, optional)
python harness/run_eval.py --submission solution --eval eval/public_eval.jsonl
```

**API keys: none required.** The system is fully deterministic and local by default. If
`ANTHROPIC_API_KEY` is set, an optional Claude layer rewrites answers (see below). Set
`CUKNOW_DISABLE_LLM=1` to force the deterministic path even when a key is present.

## Results (public practice set, deterministic mode)

| | |
|---|---|
| Accuracy (qa_score) | **0.83** (starter: 0.09) |
| Stale-value trap hits | **0** of 40 |
| should_refuse | 3 / 3 |
| Consistency (paraphrase / repeat) | 1.0 / 1.0 |
| Latency p95 | < 0.1 s per answer |
| Ingest (5 snapshots, ~3.3k pages) | ~3 s |
| Harness self-test (`harness/selftest`, incl. conflicts gold) | 1.0 |

The remaining misses are mostly free-text questions where the right sentence sits in a long article
(e.g. "what is Home$tart?"). That is where the optional LLM layer helps most.

## How it works

**1. Ingest: keep history, not just the latest copy.** Every distinct page version is parsed once.
The parser removes images, link URLs, chart dumps, form widgets (state and country dropdowns) and the
"related article" cards that bleed other articles into news pages. It then emits:
- *passages*: heading-aware windows of about 850 chars;
- *table rows*: each row becomes a fact carrying its table caption, column headers, the product context
  just before the table, and the table's **"Rates are effective as of …"** date. `label: value` lines
  are parsed as rows too.

Each page gets a **type** (rate sheet, product, contact, landing/promo, news, blog, business resource,
PDF, calculator, tag/author index) and a **publication date** (from the article header or URL). Chunks
are de-duplicated across snapshots and carry a bitmask of the snapshots they appear in. An `as_of`
query is therefore just a filter on the latest snapshot taken on or before that date. A page missing
from a later crawl keeps its last known version.

**2. Deciding what "current" means.** These are the rules, in order:
1. **The as-of view.** Only content visible in the snapshot for that date (newest snapshot by default;
   "as of May 2026" in the question text is honoured when `as_of` is absent).
2. **Authority.** Official rate sheet (`/rates`) > product page > contact > landing/promo page, PDF and
   press release > blog > business article > calculator > tag/index pages. Calculators and promo pages
   are where stale rates linger (the 5.34% vs 5.39% auto loan).
3. **Effective date.** Between tables for the same product, the newer "effective as of" date wins (the
   `/homebuying-guidance` mortgage table, effective 04/22/2026, loses to `/rates`, effective 09/25/2026).
4. **Recency of statements.** Dated pages decay with age. For "how many …" questions, the most recently
   *published* statement of the count wins. Press-release boilerplate says 19, 20, 22, 23 branches over
   the years, and the newest release says 22.
5. **Expired offers.** Text such as "offer ended" or "no longer available" is demoted.

**3. Answering.**
- *Rate questions* are answered from table rows, not prose. The question's qualifiers are parsed and
  matched against each row: term (6-month vs 12-month, 30-year), balance or loan-amount tier
  ($10,000–$24,999.99), model-year ranges (a 2025 vehicle falls in "2024-27"), jumbo vs regular,
  special vs regular, IRA vs certificate, personal vs business, and highest/lowest. The best-ranked
  table that actually covers the product asked about wins. Rows for the same product on other pages
  are suppressed, never mixed in, so a stale value is never printed. With no tier given, the
  member gets the whole table. "Rates" questions without a tier (e.g. credit cards) can pull in
  sibling tables.
- *Other questions* use the best passage (BM25 × authority × recency, plus intent bonuses for
  hours, addresses, phone numbers, dates and counts) and extract the 2–3 most relevant sentences.
- *General hours* questions: the site has no single schedule, so the answer gives the most common
  schedules across branch announcements and asks which branch.
- *Link questions* ("what URL…"): the site's own anchor text votes for a target, restricted to
  navigational pages (e.g. "today's rates" → `/personal/savings-account#rates`; "Join Dupaco" →
  `join.dupaco.com`).
- *Saying "I don't know"*. The system declines when the evidence doesn't cover the question's
  specific words: a 7/1 ARM, a construction-loan rate, a 60-month *jumbo* certificate, or a named
  branch with no hours on file. Refusals contain no numbers.
- *Sources*: the chosen page plus other non-calculator pages that state the same value.
- *Consistency*: answers are deterministic and cached per (question, as_of).

**4. Conflicts.** `conflicts()` returns at most 45 entries, ranked by importance and capped per
table to keep precision. There are three detectors:
- *temporal*: the same table row or the same sentence on the same page changes value between
  snapshots. The newest snapshot wins (e.g. 1st Rate Checking 3.00% → 2.50% APY, hardship grant
  $2,000 → $3,500, 17-month special 3.85% → 3.95% → 4.10%).
- *cross-page*: the same product row (model-year ranges matched by overlap) shows different values on
  different pages in the latest snapshot. Newest effective date, then authority, wins (auto loan
  5.39% rate sheet vs 5.34% calculator; mortgage 7.25%/7.29% vs a stale 6.000%/6.037% page).
- *org facts*: counts and recognitions in press-release boilerplate (branch count, members, assets,
  "fifth/fourth time Forbes recognized"). The newest release wins.

Each conflict lists every value, the source page and snapshot of each, the resolution, and the reason.

## Optional LLM layer (`solution/cuknow/llm.py`)

When `ANTHROPIC_API_KEY` is set, Claude (`claude-opus-5-5` at `effort: low` by default; override with
`CUKNOW_MODEL` / `CUKNOW_EFFORT`) gets the question, the as-of date, the deterministic answer and about
10 ranked evidence passages labelled with page type, publication date and effective date. It returns
JSON (`answer`, `confident`, `source_urls`) via structured outputs. Server-side refusal fallback is
enabled when the account supports it. The engine stays in charge of the truth, and the rewrite is
discarded in favour of the deterministic answer if it:
- repeats a value the engine rejected as stale;
- disagrees with a rate-sheet lookup;
- gives a refusal that contains digits;
- fails or times out.

This layer has only been checked against a stubbed client, not live API calls, and it adds about 1–3 s
of latency per answer. Grading works without it.

## Known limitations
- Extractive answers can pick a neighbouring sentence in long articles. Retrieval is lexical (BM25
  plus synonyms), so heavy rewording of non-rate questions can miss.
- Expected answers that depend on "last year" relative to an older article (294 VTO users in 2023
  vs 256 in 2024) follow the newest statement.
- Conflict precision versus the hidden gold list is unmeasured; the cap of 45 is a guess.

## Dev tools (`tools/`)
- `grep_snapshots.py REGEX [url-regex]`: which snapshots contain a line.
- `debug_query.py "question" [as_of]`: top-ranked chunks and the engine's answer.
- `show_misses.py RESULTS_DIR`: misses from a harness run.
