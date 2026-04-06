"""Evidence discovery and prediction mapping helpers."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable, Protocol

from bpr.experimental_data import MEASUREMENTS, Measurement

from .evidence import PaperCandidate, SearchQuery
from .policy import DEFAULT_AUTONOMY_POLICY

_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "below",
    "bound",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[A-Za-z0-9\-\+\.]+", text.lower())
        if token not in _STOPWORDS and len(token) > 1
    ]


def _measurement_keywords(measurement: Measurement) -> list[str]:
    tokens = _tokenize(measurement.name)
    source_tokens = _tokenize(measurement.source)
    merged = tokens + [token for token in source_tokens if token not in tokens]
    return merged[:8]


def _measurement_query_fragments(
    prediction_id: str, measurement: Measurement
) -> list[str]:
    hints = list(DEFAULT_AUTONOMY_POLICY.query_hints.get(prediction_id, ()))
    base_phrase = measurement.name.replace("Δ", "delta ").replace("θ", "theta ")
    fragments = [base_phrase]
    if hints:
        fragments.extend(hints)
    fragments.append(measurement.source)
    return fragments


def build_search_queries(
    prediction_ids: list[str] | None = None,
    max_queries_per_prediction: int = 2,
    source: str = "arxiv",
) -> list[SearchQuery]:
    """Build prioritized search queries from measurement targets."""
    if max_queries_per_prediction < 1:
        raise ValueError("max_queries_per_prediction must be >= 1")
    ids = prediction_ids or list(DEFAULT_AUTONOMY_POLICY.flagship_prediction_ids)
    queries: list[SearchQuery] = []
    for prediction_id in ids:
        if prediction_id not in MEASUREMENTS:
            raise ValueError(f"Unknown prediction_id: {prediction_id}")
        measurement = MEASUREMENTS[prediction_id]
        fragments = _measurement_query_fragments(prediction_id, measurement)
        for priority, fragment in enumerate(fragments[:max_queries_per_prediction]):
            keywords = " ".join(_measurement_keywords(measurement)[:3])
            query = f"{fragment} {keywords}".strip()
            queries.append(
                SearchQuery(
                    source=source,
                    query=query,
                    prediction_id=prediction_id,
                    measurement_name=measurement.name,
                    priority=priority,
                    tags=tuple(_measurement_keywords(measurement)[:4]),
                )
            )
    return queries


@dataclass(frozen=True)
class CandidateMatch:
    """Best-effort mapping of a paper candidate to a BPR prediction."""

    prediction_id: str
    measurement_name: str
    score: float
    matched_terms: tuple[str, ...]


def _candidate_text(candidate: PaperCandidate) -> str:
    return f"{candidate.title} {candidate.abstract}".lower()


def score_candidate_against_measurement(
    candidate: PaperCandidate, measurement: Measurement
) -> CandidateMatch:
    """Compute a simple token overlap score for one measurement."""
    measurement_terms = set(_measurement_keywords(measurement))
    policy_terms = set(DEFAULT_AUTONOMY_POLICY.query_hints.get(measurement.prediction_id, ()))
    policy_tokens = {
        token
        for term in policy_terms
        for token in _tokenize(term)
    }
    candidate_terms = set(_tokenize(_candidate_text(candidate)))
    matched = sorted((measurement_terms | policy_tokens) & candidate_terms)
    measurement_score = len(measurement_terms & candidate_terms)
    policy_score = 1.5 * len(policy_tokens & candidate_terms)
    specific_bonus = 0.25 * len(
        {
            token
            for token in (measurement_terms | policy_tokens) & candidate_terms
            if len(token) >= 4
        }
    )
    score = measurement_score + policy_score + specific_bonus
    return CandidateMatch(
        prediction_id=measurement.prediction_id,
        measurement_name=measurement.name,
        score=float(score),
        matched_terms=tuple(matched),
    )


def map_candidate_to_prediction(
    candidate: PaperCandidate, prediction_ids: list[str] | None = None
) -> CandidateMatch:
    """Map a paper candidate to the most likely BPR prediction."""
    ids = prediction_ids or list(MEASUREMENTS.keys())
    best_match = CandidateMatch("", "", -1.0, ())
    for prediction_id in ids:
        match = score_candidate_against_measurement(candidate, MEASUREMENTS[prediction_id])
        if match.score > best_match.score:
            best_match = match
    return best_match


class EvidenceSource(Protocol):
    """Protocol for evidence discovery backends."""

    name: str

    def fetch_candidates(
        self, query: SearchQuery, max_results: int = 5
    ) -> list[PaperCandidate]:
        ...


class InMemoryEvidenceSource:
    """Deterministic source used by tests and dry-run workflows."""

    name = "memory"

    def __init__(self, candidates: Iterable[PaperCandidate]) -> None:
        self._candidates = list(candidates)

    def fetch_candidates(
        self, query: SearchQuery, max_results: int = 5
    ) -> list[PaperCandidate]:
        terms = set(_tokenize(query.query))
        ranked: list[tuple[int, PaperCandidate]] = []
        for candidate in self._candidates:
            overlap = len(terms & set(_tokenize(_candidate_text(candidate))))
            ranked.append((overlap, candidate))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [candidate for score, candidate in ranked[:max_results] if score > 0]


def _read_limited_bytes(response: object, max_bytes: int = 2_000_000) -> bytes:
    """Read a bounded number of bytes from a response-like object."""
    payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(f"Response exceeded {max_bytes} bytes")
    return payload


class ArxivEvidenceSource:
    """arXiv API client using the standard library only."""

    name = "arxiv"

    def fetch_candidates(
        self, query: SearchQuery, max_results: int = 5
    ) -> list[PaperCandidate]:
        params = urllib.parse.urlencode(
            {
                "search_query": f"all:{query.query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "lastUpdatedDate",
                "sortOrder": "descending",
            }
        )
        url = f"https://export.arxiv.org/api/query?{params}"
        with urllib.request.urlopen(url, timeout=15) as response:
            payload = _read_limited_bytes(response)

        root = ET.fromstring(payload)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        candidates: list[PaperCandidate] = []
        for entry in root.findall("atom:entry", ns):
            source_id = entry.findtext("atom:id", default="", namespaces=ns).rsplit("/", 1)[-1]
            title = " ".join((entry.findtext("atom:title", default="", namespaces=ns)).split())
            abstract = " ".join(
                (entry.findtext("atom:summary", default="", namespaces=ns)).split()
            )
            published = entry.findtext("atom:published", default="", namespaces=ns)
            authors = [
                author.findtext("atom:name", default="", namespaces=ns)
                for author in entry.findall("atom:author", ns)
            ]
            candidates.append(
                PaperCandidate(
                    source=self.name,
                    source_id=source_id,
                    title=title,
                    abstract=abstract,
                    url=f"https://arxiv.org/abs/{source_id}",
                    published=published,
                    authors=authors,
                )
            )
        return candidates


class CrossrefEvidenceSource:
    """Crossref works search using only the standard library."""

    name = "crossref"

    def fetch_candidates(
        self, query: SearchQuery, max_results: int = 5
    ) -> list[PaperCandidate]:
        params = urllib.parse.urlencode(
            {"query.bibliographic": query.query, "rows": max_results}
        )
        request = urllib.request.Request(
            f"https://api.crossref.org/works?{params}",
            headers={"User-Agent": "BPR-Math-Spine/continuous-research-loop"},
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(_read_limited_bytes(response).decode("utf-8"))

        candidates: list[PaperCandidate] = []
        for item in payload.get("message", {}).get("items", []):
            title = " ".join(item.get("title", [""]))
            abstract = re.sub(r"<[^>]+>", " ", item.get("abstract", "") or "")
            published_parts = item.get("published-print", item.get("published-online", {}))
            date_parts = published_parts.get("date-parts", [[]])
            flat_parts = [str(part) for part in date_parts[0]]
            published = "-".join(flat_parts)
            authors = []
            for author in item.get("author", []):
                given = author.get("given", "")
                family = author.get("family", "")
                authors.append(" ".join(part for part in (given, family) if part).strip())
            doi = item.get("DOI", "")
            candidates.append(
                PaperCandidate(
                    source=self.name,
                    source_id=doi or item.get("URL", title),
                    title=title,
                    abstract=" ".join(abstract.split()),
                    url=item.get("URL", ""),
                    published=published,
                    authors=authors,
                    metadata={"doi": doi, "container_title": item.get("container-title", [])},
                )
            )
        return candidates
