#!/usr/bin/env python3
"""
lit_validate.py — literature-grounded validation loop for BPR-STARFIRE claims.

Pattern: retrieve -> ground -> cite, over LEGAL OPEN corpora only:
  * arXiv API        http://export.arxiv.org/api/query       (no key needed)
  * OSTI.gov API     https://www.osti.gov/api/v1/records     (no key needed)

For each claim in claims.yaml the loop pulls recent papers, scores abstracts
with a keyword/numeric heuristic, and flags candidates for HUMAN review.
The heuristic does NOT verdict physics — it only surfaces papers a human
should read. REVIEW means "go read these", not "the claim is wrong".

Statuses reported per claim:
  FRESH  — checked recently, no contrary-relevant signal above threshold
  REVIEW — new relevant papers found (listed with links)
  STALE  — not checked in > 90 days (or never)

Dry-run by default. --apply persists: claims.yaml last_checked/status,
lit_index.jsonl appends, and a decision_quality_log upsert in the work-hub
knowledge DB (skipped silently-with-message if the DB is unreachable).

Usage:
  python3 lit_validate.py                      # dry-run, all claims, both sources
  python3 lit_validate.py --claim REBCO-FLUENCE-LIMIT --sources arxiv
  python3 lit_validate.py --apply              # persist results
  python3 lit_validate.py --offline            # no network; registry state only
"""

import argparse
import json
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    print("ERROR: PyYAML required (pip install pyyaml)", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------- constants

VALIDATION_DIR = Path(__file__).resolve().parent
CLAIMS_PATH = VALIDATION_DIR / "claims.yaml"
INDEX_PATH = VALIDATION_DIR / "lit_index.jsonl"

ARXIV_API = "https://export.arxiv.org/api/query"  # http:// 301s here; go direct
OSTI_API = "https://www.osti.gov/api/v1/records"

USER_AGENT = (
    "BPR-STARFIRE-lit-validate/1.0 "
    "(fusion design claim validation; open-corpora only; contact: jack@thestardrive.com)"
)

RATE_LIMIT_SECONDS = 3.0   # 1 request / 3 s — be polite
HTTP_TIMEOUT = 30
STALE_DAYS = 90
DEFAULT_LOOKBACK_DAYS = 365
DEFAULT_MAX_RESULTS = 25
DEFAULT_THRESHOLD = 3.0
TOP_HITS_PER_CLAIM = 5

WORKHUB_DB = Path(
    "/Volumes/T9 Backup/UserData/Projects/work-hub/knowledge/knowledge.db"
)

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

STOPWORDS = frozenset(
    """a an the and or of to in for with on at by is are was were be been over
    under about from this that these its their it as not no than then we
    our which has have had do does did but if can could may might will would
    should into out up down per via using use used based new study studies
    results analysis paper present presented show shown""".split()
)

# ------------------------------------------------------------- rate limiter


class RateLimiter:
    """Global politeness limiter: at most one request per RATE_LIMIT_SECONDS."""

    def __init__(self, min_interval=RATE_LIMIT_SECONDS):
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self):
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


def http_get(url, params, limiter):
    """GET with descriptive UA and rate limiting. Returns bytes or None."""
    limiter.wait()
    full = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(full, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        print("    [warn] request failed ({}): {}".format(url, exc), file=sys.stderr)
        return None


# ---------------------------------------------------------------- retrieval


def query_arxiv(claim, since_iso, max_results, limiter):
    """Query arXiv (plasma physics category + claim search terms).

    One request per claim: terms OR-combined, newest first, filtered to
    papers submitted on/after since_iso (YYYY-MM-DD).
    """
    terms = claim.get("search_terms") or []
    if not terms:
        return []
    # Phrase search (all:"...") is too strict for 3-5 word terms — it requires a
    # verbatim match and returns ~0 hits. Instead each term becomes an AND-group
    # of its significant words; groups are OR'd together.
    groups = []
    for t in terms:
        words = [
            w
            for w in re.findall(r"[A-Za-z][A-Za-z0-9]+", t)
            if len(w) >= 3 and w.lower() not in STOPWORDS
        ]
        if words:
            groups.append("(" + " AND ".join("all:{}".format(w) for w in words) + ")")
    if not groups:
        return []
    search_query = "cat:physics.plasm-ph AND ({})".format(" OR ".join(groups))
    raw = http_get(
        ARXIV_API,
        {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": 0,
            "max_results": max_results,
        },
        limiter,
    )
    if raw is None:
        return []
    papers = []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        print("    [warn] arXiv XML parse error: {}".format(exc), file=sys.stderr)
        return []
    for entry in root.findall("atom:entry", ATOM_NS):
        arxiv_id = _text(entry, "atom:id").rsplit("/", 1)[-1]
        published = _text(entry, "atom:published")[:10]
        if published and published < since_iso:
            continue
        papers.append(
            {
                "source": "arxiv",
                "paper_id": arxiv_id,
                "doi": "",
                "title": _clean(_text(entry, "atom:title")),
                "date": published,
                "abstract": _clean(_text(entry, "atom:summary")),
                "url": "https://arxiv.org/abs/{}".format(arxiv_id),
            }
        )
    return papers


def query_osti(claim, since_iso, max_results, limiter):
    """Query OSTI.gov records API with fusion subject filter."""
    terms = claim.get("search_terms") or []
    if not terms:
        return []
    since_dt = datetime.strptime(since_iso, "%Y-%m-%d")
    raw = http_get(
        OSTI_API,
        {
            "q": " OR ".join('"{}"'.format(t) for t in terms),
            "subject": "fusion energy",
            "publication_date_start": since_dt.strftime("%m/%d/%Y"),
            "rows": max_results,
            "sort": "publication_date desc",
        },
        limiter,
    )
    if raw is None:
        return []
    try:
        records = json.loads(raw)
    except (ValueError, TypeError) as exc:
        print("    [warn] OSTI JSON parse error: {}".format(exc), file=sys.stderr)
        return []
    if isinstance(records, dict):  # some error payloads come back as dicts
        records = records.get("records") or []
    papers = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        osti_id = str(rec.get("osti_id", ""))
        if not osti_id:
            continue
        date = _osti_date(rec.get("publication_date", ""))
        if date and date < since_iso:
            continue
        papers.append(
            {
                "source": "osti",
                "paper_id": "osti:{}".format(osti_id),
                "doi": rec.get("doi") or "",
                "title": _clean(rec.get("title", "")),
                "date": date,
                "abstract": _clean(rec.get("description", "") or ""),
                "url": "https://www.osti.gov/biblio/{}".format(osti_id),
            }
        )
    return papers


def _text(node, path):
    found = node.find(path, ATOM_NS)
    return (found.text or "") if found is not None else ""


def _clean(text):
    return re.sub(r"\s+", " ", text or "").strip()


def _osti_date(raw):
    """Normalize OSTI publication dates to YYYY-MM-DD (best effort)."""
    raw = (raw or "").strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%m/%d/%Y", "%Y"):
        try:
            return datetime.strptime(raw[:19] if "T" in raw else raw, fmt).strftime(
                "%Y-%m-%d"
            )
        except ValueError:
            continue
    return ""


# ----------------------------------------------------------------- scoring
#
# v1: keyword/numeric heuristic. The scorer is deliberately isolated behind
# score_paper() so a future LLM judge can be swapped in:
#
#   def score_paper(claim, paper):
#       return llm_judge(claim, paper)   # same signature, returns float
#


def _tokens(text):
    return {
        w
        for w in re.findall(r"[a-z][a-z0-9\-]{2,}", (text or "").lower())
        if w not in STOPWORDS
    }


def _numbers(text):
    out = []
    for m in re.findall(r"\d+\.?\d*(?:[eE][+-]?\d+)?", text or ""):
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


def heuristic_score(claim, paper):
    """Keyword + numeric proximity heuristic. Higher = more relevant.

    Components:
      +2.0  per search-term phrase appearing verbatim in title+abstract
      +0.2  per claim-statement keyword found in abstract (capped at 2.0)
      +1.0  if the abstract mentions a number within 15% of the claim value
    """
    text = "{} {}".format(paper.get("title", ""), paper.get("abstract", "")).lower()
    score = 0.0

    for term in claim.get("search_terms") or []:
        if term.lower() in text:
            score += 2.0

    claim_kw = _tokens(claim.get("statement", ""))
    paper_kw = _tokens(text)
    score += min(2.0, 0.2 * len(claim_kw & paper_kw))

    try:
        value = float(claim.get("value"))
    except (TypeError, ValueError):
        value = None
    if value:
        for n in _numbers(text):
            if n > 0 and abs(n - value) / abs(value) <= 0.15:
                score += 1.0
                break
    return round(score, 2)


def score_paper(claim, paper):
    """Swap point: replace heuristic_score with an LLM judge later."""
    return heuristic_score(claim, paper)


# ------------------------------------------------------------------- index


def load_index_ids():
    """Set of (claim_id, paper_id) already indexed — keeps the loop idempotent."""
    seen = set()
    if not INDEX_PATH.exists():
        return seen
    with INDEX_PATH.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                seen.add((rec.get("claim_id"), rec.get("paper_id")))
            except ValueError:
                continue
    return seen


def append_index(entries):
    with INDEX_PATH.open("a") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- registry


def load_claims():
    if not CLAIMS_PATH.exists():
        print("ERROR: {} not found".format(CLAIMS_PATH), file=sys.stderr)
        sys.exit(1)
    with CLAIMS_PATH.open() as fh:
        doc = yaml.safe_load(fh)
    return doc


def save_claims(doc):
    header = (
        "# BPR-STARFIRE claims registry — literature validation loop\n"
        "# Managed by lit_validate.py --apply. status: unverified | fresh | review\n"
        "# Rule: never invent numbers — all claims must trace to a pass.\n"
    )
    with CLAIMS_PATH.open("w") as fh:
        fh.write(header)
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True, width=100)


def staleness(claim, today):
    lc = claim.get("last_checked")
    if not lc:
        return None  # never checked
    try:
        return (today - datetime.strptime(str(lc)[:10], "%Y-%m-%d")).days
    except ValueError:
        return None


# --------------------------------------------------------------- dql write


def write_dql(checked, fresh, review, summary):
    """Upsert decision_quality_log in the work-hub knowledge DB.

    recommendations_made        = claims checked this run
    recommendations_accepted    = claims FRESH (no contrary signal)
    actions_escalated_correctly = claims flagged REVIEW (escalated to human)

    Unreachable DB -> skip with message. NEVER creates the DB (uri mode=rw).
    """
    if not WORKHUB_DB.exists():
        print("[dql] work-hub knowledge.db not found — skipping cross-repo write")
        return False
    try:
        conn = sqlite3.connect("file:{}?mode=rw".format(WORKHUB_DB), uri=True, timeout=10)
        week_of = datetime.now().strftime("%G-W%V")
        conn.execute(
            """
            INSERT INTO decision_quality_log
                (week_of, module, recommendations_made, recommendations_accepted,
                 actions_escalated_correctly, notes)
            VALUES (?, 'starfire', ?, ?, ?, ?)
            ON CONFLICT(week_of, module) DO UPDATE SET
                recommendations_made        = excluded.recommendations_made,
                recommendations_accepted    = excluded.recommendations_accepted,
                actions_escalated_correctly = excluded.actions_escalated_correctly,
                notes                       = excluded.notes
            """,
            (week_of, checked, fresh, review, json.dumps(summary)),
        )
        conn.commit()
        conn.close()
        print("[dql] decision_quality_log upserted (week_of={}, module=starfire)".format(week_of))
        return True
    except sqlite3.Error as exc:
        print("[dql] knowledge.db unreachable ({}) — skipping".format(exc))
        return False


# ------------------------------------------------------------------ report


def report_offline(claims, today):
    print("=== BPR-STARFIRE claims registry (offline) ===")
    counts = {"FRESH": 0, "REVIEW": 0, "STALE": 0}
    for c in claims:
        days = staleness(c, today)
        if days is None or days > STALE_DAYS:
            state = "STALE"
        elif c.get("status") == "review":
            state = "REVIEW"
        else:
            state = "FRESH"
        counts[state] += 1
        lc = c.get("last_checked") or "never"
        print(
            "  [{:6}] {:24} {} {} (pass {}, last checked: {})".format(
                state, c["id"], c.get("value"), c.get("unit", ""), c.get("source_pass"), lc
            )
        )
    print(
        "\n{} claims — FRESH {}, REVIEW {}, STALE {}".format(
            len(claims), counts["FRESH"], counts["REVIEW"], counts["STALE"]
        )
    )


def run(args):
    doc = load_claims()
    claims = doc.get("claims") or []
    today = datetime.now()

    if args.claim:
        claims_to_check = [c for c in claims if c["id"] == args.claim]
        if not claims_to_check:
            print("ERROR: claim id '{}' not in registry".format(args.claim), file=sys.stderr)
            print("Known ids: {}".format(", ".join(c["id"] for c in claims)), file=sys.stderr)
            sys.exit(1)
    else:
        claims_to_check = claims

    if args.offline:
        report_offline(claims_to_check, today)
        return

    limiter = RateLimiter()
    seen = load_index_ids()
    new_index_entries = []
    fetched_at = today.strftime("%Y-%m-%dT%H:%M:%S")
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    results = {}  # claim_id -> {"state":..., "hits":[...]}

    print(
        "=== lit_validate — {} claim(s), sources: {}, {} ===".format(
            len(claims_to_check), "+".join(sources), "APPLY" if args.apply else "dry-run"
        )
    )

    for claim in claims_to_check:
        days = staleness(claim, today)
        lookback = args.lookback if days is None else min(args.lookback, days + 1)
        since = (today - timedelta(days=lookback)).strftime("%Y-%m-%d")
        print("\n--- {} (pass {}) — papers since {} ---".format(claim["id"], claim["source_pass"], since))
        print("    claim: {}".format(claim["statement"]))

        papers = []
        if "arxiv" in sources:
            papers += query_arxiv(claim, since, args.max_results, limiter)
        if "osti" in sources:
            papers += query_osti(claim, since, args.max_results, limiter)

        # idempotency: index only never-seen papers; score everything fetched
        fresh_papers = [p for p in papers if (claim["id"], p["paper_id"]) not in seen]
        for p in fresh_papers:
            new_index_entries.append(
                {
                    "claim_id": claim["id"],
                    "source": p["source"],
                    "paper_id": p["paper_id"],
                    "doi": p["doi"],
                    "title": p["title"],
                    "date": p["date"],
                    "abstract": p["abstract"],
                    "fetched_at": fetched_at,
                }
            )
            seen.add((claim["id"], p["paper_id"]))

        scored = sorted(
            ((score_paper(claim, p), p) for p in papers), key=lambda x: -x[0]
        )
        top = scored[:TOP_HITS_PER_CLAIM]
        hits = [(s, p) for s, p in top if s >= args.threshold]

        state = "REVIEW" if hits else "FRESH"
        results[claim["id"]] = {"state": state, "hits": len(hits), "fetched": len(papers)}
        print(
            "    fetched {} paper(s) ({} new to index) -> {}".format(
                len(papers), len(fresh_papers), state
            )
        )
        if hits:
            print("    flagged for HUMAN review (heuristic — not a physics verdict):")
        elif top:
            print("    top-scored (below threshold {}):".format(args.threshold))
        for s, p in top:
            flag = "*" if s >= args.threshold else " "
            print("     {}[{:>5}] {} ({}, {})".format(flag, s, p["title"][:88], p["source"], p["date"]))
            print("              {}".format(p["url"]))

        if args.apply:
            claim["last_checked"] = today.strftime("%Y-%m-%d")
            claim["status"] = state.lower()

    # ----- summary
    n_review = sum(1 for r in results.values() if r["state"] == "REVIEW")
    n_fresh = sum(1 for r in results.values() if r["state"] == "FRESH")
    print(
        "\n=== summary: {} checked — {} FRESH, {} REVIEW, {} new index entries ===".format(
            len(results), n_fresh, n_review, len(new_index_entries)
        )
    )

    if args.apply:
        append_index(new_index_entries)
        save_claims(doc)
        print("[apply] claims.yaml + lit_index.jsonl updated")
        summary = {
            "checked": len(results),
            "fresh": n_fresh,
            "review": n_review,
            "new_index_entries": len(new_index_entries),
            "review_claims": [k for k, v in results.items() if v["state"] == "REVIEW"],
            "run_at": fetched_at,
        }
        write_dql(len(results), n_fresh, n_review, summary)
    else:
        print("[dry-run] nothing persisted — re-run with --apply to commit")


def main():
    ap = argparse.ArgumentParser(
        description="Literature validation loop for BPR-STARFIRE design claims "
        "(arXiv + OSTI.gov open corpora only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--claim", help="check a single claim id (default: all)")
    ap.add_argument("--apply", action="store_true", help="persist results (default: dry-run)")
    ap.add_argument("--offline", action="store_true", help="no network; report registry state only")
    ap.add_argument("--sources", default="arxiv,osti", help="comma list: arxiv,osti")
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK_DAYS, help="days back on first check")
    ap.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS, help="max papers per source per claim")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="relevance score cutoff for REVIEW")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
