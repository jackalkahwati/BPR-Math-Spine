# STARFIRE Literature Validation Loop

The repo rule is "never invent numbers — all claims must trace to a pass."
This adds the missing rung: **does recently published literature contradict
or move any canonical assumption?** It is the Sci-Bot pattern — retrieve,
ground, cite — over **legal open corpora only**: the arXiv API and the
OSTI.gov records API. No keys, no paywalled or unauthorized full-text
sources, ever.

## Pieces

- `claims.yaml` — registry of canonical STARFIRE numbers (TBR, REBCO fluence
  limit, dpa lifetimes, LCOE/CAPEX, MHD/FCI, bootstrap target...), each
  traced to its source pass (A/B/C/ext) with tuned search terms.
- `lit_validate.py` — the loop. Queries arXiv (`cat:physics.plasm-ph` +
  claim terms) and OSTI (fusion subject filter) for papers newer than each
  claim's `last_checked` (365 d lookback on first run), rate-limited to
  1 request / 3 s with a descriptive User-Agent.
- `lit_index.jsonl` — append-only paper index (claim_id, source, id/DOI,
  title, date, abstract, fetched_at). Idempotent: already-indexed papers
  are skipped.

## Run

```bash
python3 validation/lit_validate.py                  # dry-run, all claims
python3 validation/lit_validate.py --claim LCOE-PILOT --sources arxiv
python3 validation/lit_validate.py --apply          # persist + dql write
python3 validation/lit_validate.py --offline        # registry state, no network (CI-safe)
```

Per-claim verdicts: **FRESH** (checked, no contrary signal), **REVIEW**
(new relevant papers — listed with links), **STALE** (>90 d unchecked).
The relevance pass is a keyword/numeric heuristic isolated behind
`score_paper()` so an LLM judge can be swapped in later. It flags papers
for human review; **it does not verdict physics**.

On `--apply`, the run also upserts `decision_quality_log` in the work-hub
knowledge DB (module=`starfire`): claims checked / FRESH / REVIEW counts.
If the DB is unreachable it skips with a message and never creates it.

## How the RSI loop closes

1. Loop flags a claim **REVIEW** with specific arXiv/OSTI links.
2. A human reads the papers. If the literature moved (e.g., a new REBCO
   irradiation result shifts the 1e18 n/cm² fluence limit), the relevant
   pass is re-run and the design record updated.
3. The claim's value/statement is updated in `claims.yaml`, traced to the
   new pass result, and the next sweep re-verifies it as FRESH.

Recursion is human-in-the-loop by design: retrieval is automated,
judgment is not.
