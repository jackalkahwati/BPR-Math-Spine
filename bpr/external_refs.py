"""Loader and validator for commit-pinned external cross-reference indexes.

The only catalogue so far is ``doc/external/openai_math_index.json``: results from
openai/math (Apache-2.0, pinned to one commit) that BPR has considered. Policy,
citation template and the pin-moving procedure are in ``doc/external/OPENAI_MATH.md``.

What this module checks is bookkeeping, never mathematics: commit pins, licence
handling, vocabulary, references into the repo and the unification map, usage
caps, Lean-scope labels and citation rendering (rules 1-10 below, plus rule 0 for
the schema). A clean ``validate`` means the index is internally consistent. It says
nothing about whether any upstream result is correct.

Statuses are never computed from an index. The unification-map loader does not
read this file; ``validate`` only checks that the map does not lean on an entry
more than the entry's ``usage_cap`` allows (rule 7).

Rules
-----
0. Schema: required fields, types, key format, unique keys, hash formats.
1. Pin: 40-hex commit with matching ``commit_short``; every upstream URL is a
   ``blob``/``tree`` permalink at that commit; ``blob/main`` never appears.
2. Licence: Apache-2.0 for openai/math; redistributed files (none today) need the
   licence text, change notices and, if upstream has one, the NOTICE.
3. Vocabulary: every labelled field takes a value from ``vocab``.
4. References: every ``bpr_nodes`` id is a map node; every ``bpr_files`` path exists.
5. Usage cap: ``rigorous_input`` needs a Lean main statement, ``bpr_check == "built"``,
   a fidelity check, an active upstream status and no external dependencies;
   ``manuscript_only`` entries stay ``context_only`` unless justified.
6. Inactive entries (withdrawn, or considered but not cited) carry no relation,
   no nodes and the lowest cap.
7. The map never leans on an entry beyond its cap (input/counterexample entries
   below ``rigorous_input`` only touch conditional/open nodes or named hypotheses;
   context-like entries never appear among premises).
8. Every rendered citation names the verification label and says "Unrefereed".
9. An entry that overlaps a classical or refereed result names that fallback.
10. A ``lean_*`` label is valid only if the cited statement lies inside the
    recorded Lean scope (mechanical proxy: anchor terms shared with the scope
    excerpt, and no term recorded as outside the Lean scope).

Standard library only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Union

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENAI_MATH = ROOT / "doc" / "external" / "openai_math_index.json"

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Entry field -> vocab list that constrains it (rule 3).
VOCAB_FIELDS = {
    "upstream_status": "upstream_status",
    "verification": "verification",
    "verification_source": "verification_source",
    "bpr_check": "bpr_check",
    "relation": "relation",
    "applicability": "applicability",
    "usage_cap": "usage_cap",
}
REQUIRED_UPSTREAM = (
    "name", "url", "key_prefix", "author", "commit", "commit_short", "retrieved",
    "license", "license_url", "license_sha256", "notice_file", "redistributed_files",
    "history_sections",
)
REQUIRED_ENTRY = {
    "key": str, "family_id": str, "manuscript_dir": str, "title": str, "date": str,
    "result_ref": str, "permalink": str, "source_file": str, "source_sha256": str,
    "statement_bpr": str, "not_covered": list, "upstream_status": str, "refereed": bool,
    "verification": str, "lean_scope_terms": list, "outside_lean_scope": list,
    "bpr_check": str, "external_dependencies": list, "classical_fallback": list,
    "supersedes_classical": bool, "bpr_nodes": list, "bpr_files": list, "relation": str,
    "applicability": str, "usage_cap": str, "review": dict,
}
# Keys that must be present but may be null.
NULLABLE_ENTRY = ("verification_source", "lean_scope_excerpt", "lean", "lean_doc",
                  "lean_doc_sha256", "fidelity_check")
LEAN_LABELS = ("lean_main_statement", "lean_supporting_only")
STRONG_RELATIONS = ("input", "counterexample")
REVIEWERS = ("single_agent", "multi_agent")
SOFT_STATUSES = ("conditional", "open")

VERIFICATION_GLOSS = {
    "lean_main_statement": "cited statement is a Lean comparator statement",
    "lean_supporting_only": "only a supporting result is Lean-formalized; the cited statement is paper-only",
    "manuscript_only": "no Lean formalization",
}
SOURCE_GLOSS = {
    "formalization_yaml": "listed in lean/formalization.yaml main_results",
    "lean_docs_only": "recorded in lean/docs only, not in lean/formalization.yaml",
}
CHECK_GLOSS = {
    "none": "not re-checked by BPR",
    "statement_read": "comparator statement read by BPR, not built or re-checked",
    "built": "comparator re-run by BPR",
}

MapLike = Mapping[str, Any]


# --------------------------------------------------------------------------- loading


def _reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


@dataclass(frozen=True)
class Entry:
    """One indexed upstream result, with the upstream block it was pinned under."""

    data: Mapping[str, Any]
    upstream: Mapping[str, Any] = field(default_factory=dict)

    def __getitem__(self, name: str) -> Any:
        return self.data[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self.data.get(name, default)

    @property
    def key(self) -> str:
        return self.data["key"]

    @property
    def verification(self) -> str:
        return self.data["verification"]

    @property
    def usage_cap(self) -> str:
        return self.data["usage_cap"]

    @property
    def relation(self) -> str:
        return self.data["relation"]

    @property
    def bpr_nodes(self) -> list:
        return list(self.data.get("bpr_nodes") or [])

    @property
    def bpr_files(self) -> list:
        return list(self.data.get("bpr_files") or [])


@dataclass(frozen=True)
class ExternalIndex:
    """A parsed cross-reference index (generic over catalogues with this schema)."""

    data: Mapping[str, Any]
    path: Union[Path, None] = None
    entries: tuple = field(init=False)

    def __post_init__(self):
        upstream = self.data.get("upstream") or {}
        raw = self.data.get("entries") or []
        object.__setattr__(self, "entries", tuple(Entry(e, upstream) for e in raw if isinstance(e, Mapping)))

    @property
    def upstream(self) -> Mapping[str, Any]:
        return self.data.get("upstream") or {}

    @property
    def vocab(self) -> Mapping[str, Any]:
        return self.data.get("vocab") or {}

    def entry(self, key: str) -> Entry:
        for e in self.entries:
            if e.get("key") == key:
                return e
        raise KeyError(key)

    def entries_for_node(self, node_id: str) -> list:
        return [e for e in self.entries if node_id in e.bpr_nodes]

    def entries_for_file(self, repo_path: Union[str, Path]) -> list:
        target = _normalise_repo_path(repo_path)
        return [e for e in self.entries if target in {_normalise_repo_path(p) for p in e.bpr_files}]


def _normalise_repo_path(repo_path: Union[str, Path]) -> str:
    p = Path(repo_path)
    if p.is_absolute():
        try:
            p = p.resolve().relative_to(ROOT)
        except ValueError:
            return p.as_posix()
    text = p.as_posix()
    while text.startswith("./"):
        text = text[2:]
    return text


def load_index(path: Union[str, Path] = DEFAULT_OPENAI_MATH) -> ExternalIndex:
    """Read an index file (duplicate JSON keys are an error)."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicates)
    return ExternalIndex(data, p)


# --------------------------------------------------------------------------- citation


def citation(entry: Entry, with_fallback: bool = True) -> str:
    """Render the standard citation string for one entry (rule 8).

    Considered-but-not-cited entries refuse to render until a fidelity check is
    recorded: such results must not reach prose on the strength of the index alone.
    """
    d = entry.data
    up = entry.upstream
    review = d.get("review") or {}
    if review.get("verdict") == "considered_not_cited" and not d.get("fidelity_check"):
        raise ValueError(
            f"{d.get('key')}: considered, not cited; record a fidelity check before any prose cites it"
        )
    label = d["verification"]
    detail = [VERIFICATION_GLOSS.get(label, label)]
    lean = d.get("lean") or {}
    if lean.get("declarations"):
        what = "supporting declaration" if label == "lean_supporting_only" else "comparator"
        detail.append(f"{what} " + ", ".join(lean["declarations"]))
    if d.get("verification_source"):
        detail.append(SOURCE_GLOSS.get(d["verification_source"], d["verification_source"]))
    check = CHECK_GLOSS.get(d.get("bpr_check"), "not re-checked by BPR")
    prefix = "WITHDRAWN upstream; " if d.get("upstream_status") == "withdrawn" else ""
    revised = " (a revised edition exists upstream)" if d.get("upstream_status") == "revised" else ""
    text = (
        f'{prefix}{up.get("author", "OpenAI")}, "{d["title"]}" ({up.get("name", "upstream")} preprint, '
        f'{d["date"]}; commit {up.get("commit_short", "?")}; {up.get("license", "?")}){revised}, '
        f'{d["result_ref"]}. Verification: {label} ({"; ".join(detail)}); {check}. Unrefereed. '
        f'Used as {d["relation"]}, capped at {d["usage_cap"]}. {d["permalink"]}'
    )
    if with_fallback and d.get("classical_fallback"):
        text += " Classical or refereed: " + "; ".join(d["classical_fallback"]) + "."
    return text


# --------------------------------------------------------------------------- validation


def _strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, Mapping):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def _err(rule: int, key: str, message: str) -> str:
    return f"rule{rule}: {key}: {message}"


def _rule0_schema(index: ExternalIndex) -> list:
    errors = []
    data = index.data
    for top in ("schema_version", "upstream", "vocab", "entries"):
        if top not in data:
            errors.append(_err(0, "<index>", f"missing top-level field {top!r}"))
    up = index.upstream
    for name in REQUIRED_UPSTREAM:
        if name not in up:
            errors.append(_err(0, "<upstream>", f"missing field {name!r}"))
    if not isinstance(data.get("entries"), list):
        errors.append(_err(0, "<index>", "entries must be a list"))
        return errors
    prefix = up.get("key_prefix", "")
    key_re = re.compile(rf"^{re.escape(prefix)}:(\d{{3}}):[a-z0-9]+(?:-[a-z0-9]+)*$")
    seen = set()
    for i, raw in enumerate(data["entries"]):
        if not isinstance(raw, Mapping):
            errors.append(_err(0, f"<entry {i}>", "entry must be an object"))
            continue
        key = str(raw.get("key", f"<entry {i}>"))
        for name, typ in REQUIRED_ENTRY.items():
            if name not in raw:
                errors.append(_err(0, key, f"missing field {name!r}"))
            elif not isinstance(raw[name], typ):
                errors.append(_err(0, key, f"field {name!r} must be {typ.__name__}"))
        for name in NULLABLE_ENTRY:
            if name not in raw:
                errors.append(_err(0, key, f"missing field {name!r} (may be null)"))
        if key in seen:
            errors.append(_err(0, key, "duplicate key"))
        seen.add(key)
        m = key_re.match(key)
        if not m:
            errors.append(_err(0, key, f"key must look like {prefix}:NNN:slug"))
        elif m.group(1) != raw.get("family_id"):
            errors.append(_err(0, key, "family_id does not match the key"))
        if not ISO_DATE.match(str(raw.get("date", ""))):
            errors.append(_err(0, key, "date must be YYYY-MM-DD"))
        mdir = str(raw.get("manuscript_dir", ""))
        if not mdir.startswith("preprints/") or ".." in mdir.split("/"):
            errors.append(_err(0, key, "manuscript_dir must be preprints/<dir>"))
        if not str(raw.get("source_file", "")).startswith(mdir + "/"):
            errors.append(_err(0, key, "source_file must lie inside manuscript_dir"))
        for name in ("source_sha256", "lean_doc_sha256"):
            value = raw.get(name)
            if value is not None and not HEX64.match(str(value)):
                errors.append(_err(0, key, f"{name} must be a sha256 hex digest"))
        if (raw.get("lean_doc") is None) != (raw.get("lean_doc_sha256") is None):
            errors.append(_err(0, key, "lean_doc and lean_doc_sha256 must be both set or both null"))
        if raw.get("refereed") is not False:
            errors.append(_err(0, key, "upstream preprints are unrefereed; refereed must be false"))
        review = raw.get("review") if isinstance(raw.get("review"), Mapping) else {}
        if not ISO_DATE.match(str(review.get("date", ""))):
            errors.append(_err(0, key, "review.date must be YYYY-MM-DD"))
        if review.get("reviewers") not in REVIEWERS:
            errors.append(_err(0, key, f"review.reviewers must be one of {REVIEWERS}"))
        for name in ("not_covered", "lean_scope_terms", "outside_lean_scope", "external_dependencies",
                     "classical_fallback", "bpr_nodes", "bpr_files"):
            value = raw.get(name)
            if isinstance(value, list) and not all(isinstance(x, str) and x.strip() for x in value):
                errors.append(_err(0, key, f"{name} must hold non-empty strings"))
        if not str(raw.get("statement_bpr", "")).strip():
            errors.append(_err(0, key, "statement_bpr must be non-empty"))
    return errors


def _rule1_pin(index: ExternalIndex) -> list:
    errors = []
    up = index.upstream
    commit = str(up.get("commit", ""))
    short = str(up.get("commit_short", ""))
    if not HEX40.match(commit):
        errors.append(_err(1, "<upstream>", "commit must be 40 lowercase hex characters"))
    if len(short) < 7 or not commit.startswith(short):
        errors.append(_err(1, "<upstream>", "commit_short must be a prefix (>= 7 chars) of commit"))
    url = str(up.get("url", "")).rstrip("/")
    if not url.startswith("https://github.com/"):
        errors.append(_err(1, "<upstream>", "url must be a https://github.com/ repository URL"))
        return errors
    pinned = re.compile(rf"^/(blob|tree)/{re.escape(commit)}/\S+$")
    url_re = re.compile(re.escape(url) + r"(?P<rest>[^\s\"'<>)\]`]*)")
    for text in _strings(index.data):
        if re.search(r"/(blob|tree)/(main|master|HEAD)/", text):
            errors.append(_err(1, "<index>", f"unpinned URL: {text[:120]!r}"))
        for m in url_re.finditer(text):
            rest = m.group("rest")
            if rest and not pinned.match(rest):
                errors.append(_err(1, "<index>", f"URL not pinned to {commit[:9]}: {m.group(0)[:120]!r}"))
    if up.get("license_url") != f"{url}/blob/{commit}/LICENSE":
        errors.append(_err(1, "<upstream>", "license_url must be the pinned LICENSE permalink"))
    for e in index.entries:
        expect = f"{url}/tree/{commit}/{e.get('manuscript_dir')}"
        if e.get("permalink") != expect:
            errors.append(_err(1, e.get("key"), f"permalink must be {expect}"))
    return errors


def _rule2_licence(index: ExternalIndex, repo_root: Path) -> list:
    errors = []
    up = index.upstream
    if not str(up.get("license", "")).strip():
        errors.append(_err(2, "<upstream>", "license must be recorded"))
    if up.get("name") == "openai/math" and up.get("license") != "Apache-2.0":
        errors.append(_err(2, "<upstream>", "openai/math is Apache-2.0"))
    if up.get("license_sha256") is not None and not HEX64.match(str(up.get("license_sha256"))):
        errors.append(_err(2, "<upstream>", "license_sha256 must be a sha256 hex digest"))
    files = up.get("redistributed_files") or []
    if not isinstance(files, list):
        return errors + [_err(2, "<upstream>", "redistributed_files must be a list")]
    if not files:
        return errors
    slug = str(up.get("name", "")).replace("/", "_")
    roots = [repo_root / "third_party" / slug, repo_root / "lean" / "ThirdParty" / slug]
    for rel in files:
        path = repo_root / rel
        home = next((r for r in roots if r in path.parents), None)
        if home is None:
            errors.append(_err(2, rel, f"redistributed files must live under third_party/{slug}/ or lean/ThirdParty/{slug}/"))
            continue
        if not path.is_file():
            errors.append(_err(2, rel, "redistributed file is missing"))
        if not (path.parent / "LICENSE").is_file() and not (home / "LICENSE").is_file():
            errors.append(_err(2, rel, "a copy of the upstream LICENSE must sit next to redistributed files"))
        if not (home / "MODIFICATIONS.md").is_file():
            errors.append(_err(2, rel, "MODIFICATIONS.md with change notices (Apache-2.0 section 4(b)) is required"))
        if up.get("notice_file") and not (home / "NOTICE").is_file():
            errors.append(_err(2, rel, "upstream has a NOTICE file; a copy must accompany redistributed files"))
    return errors


def _rule3_vocab(index: ExternalIndex) -> list:
    errors = []
    vocab = index.vocab
    for name in set(VOCAB_FIELDS.values()) | {"review_verdict"}:
        if not isinstance(vocab.get(name), list) or not vocab.get(name):
            errors.append(_err(3, "<vocab>", f"vocab.{name} must be a non-empty list"))
    for e in index.entries:
        for fld, vkey in VOCAB_FIELDS.items():
            value = e.get(fld)
            if fld == "verification_source" and value is None:
                continue
            if value not in (vocab.get(vkey) or []):
                errors.append(_err(3, e.get("key"), f"{fld}={value!r} not in vocab.{vkey}"))
        verdict = (e.get("review") or {}).get("verdict")
        if verdict not in (vocab.get("review_verdict") or []):
            errors.append(_err(3, e.get("key"), f"review.verdict={verdict!r} not in vocab.review_verdict"))
    return errors


def _rule4_references(index: ExternalIndex, unification_map: MapLike, repo_root: Path) -> list:
    errors = []
    node_ids = {n.get("id") for n in unification_map.get("nodes", [])}
    for e in index.entries:
        for node in e.bpr_nodes:
            if node not in node_ids:
                errors.append(_err(4, e.key, f"bpr_nodes id {node!r} is not a unification-map node"))
        for rel in e.bpr_files:
            p = PurePosixPath(rel)
            if p.is_absolute() or ".." in p.parts or "\\" in rel:
                errors.append(_err(4, e.key, f"bpr_files path {rel!r} must be repo-relative POSIX"))
            elif not (repo_root / rel).exists():
                errors.append(_err(4, e.key, f"bpr_files path {rel!r} does not exist"))
    return errors


def _rule5_cap(index: ExternalIndex) -> list:
    errors = []
    for e in index.entries:
        cap = e.get("usage_cap")
        if cap == "rigorous_input":
            needs = {
                "verification == lean_main_statement": e.get("verification") == "lean_main_statement",
                "bpr_check == built": e.get("bpr_check") == "built",
                "a fidelity_check": bool(e.get("fidelity_check")),
                "upstream_status == active": e.get("upstream_status") == "active",
                "no external_dependencies": not e.get("external_dependencies"),
                "a verification_source": bool(e.get("verification_source")),
            }
            for what, ok in needs.items():
                if not ok:
                    errors.append(_err(5, e.key, f"rigorous_input requires {what}"))
        elif cap == "conditional_input":
            if e.get("verification") == "manuscript_only" and not str(e.get("cap_justification", "")).strip():
                errors.append(_err(5, e.key, "manuscript_only entries stay context_only unless cap_justification is given"))
    return errors


def _rule6_inactive(index: ExternalIndex) -> list:
    errors = []
    for e in index.entries:
        verdict = (e.get("review") or {}).get("verdict")
        inactive = e.get("upstream_status") == "withdrawn" or verdict == "considered_not_cited"
        why = "withdrawn" if e.get("upstream_status") == "withdrawn" else "considered_not_cited"
        if inactive:
            if e.get("relation") != "none":
                errors.append(_err(6, e.key, f"{why} entries must have relation 'none'"))
            if e.bpr_nodes:
                errors.append(_err(6, e.key, f"{why} entries must have empty bpr_nodes"))
            if e.get("usage_cap") != "context_only":
                errors.append(_err(6, e.key, f"{why} entries must be capped at context_only"))
        if verdict == "considered_not_cited" and e.bpr_files:
            errors.append(_err(6, e.key, "considered_not_cited entries are cited nowhere: bpr_files must be empty"))
        if e.get("relation") == "none" and e.bpr_nodes:
            errors.append(_err(6, e.key, "relation 'none' cannot touch any node"))
    return errors


def _ref_ids(items: Any) -> list:
    ids = []
    for item in items or []:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, Mapping):
            for name in ("key", "id", "node"):
                if isinstance(item.get(name), str):
                    ids.append(item[name])
    return ids


def _rule7_map(index: ExternalIndex, unification_map: MapLike) -> list:
    errors = []
    nodes = {n.get("id"): n for n in unification_map.get("nodes", []) if isinstance(n, Mapping)}
    by_key = {e.key: e for e in index.entries}
    for e in index.entries:
        for node_id in e.bpr_nodes:
            node = nodes.get(node_id)
            if node is None:
                continue  # reported by rule 4
            if e.get("relation") in STRONG_RELATIONS and e.get("usage_cap") != "rigorous_input":
                if node.get("status") not in SOFT_STATUSES and e.key not in _ref_ids(node.get("hypotheses")):
                    errors.append(_err(7, e.key, f"{e.get('relation')} below rigorous_input touches {node_id} "
                                                 f"(status {node.get('status')!r}) without naming it in hypotheses"))
    for node_id, node in nodes.items():
        for ref in node.get("external_refs") or []:
            key = ref.get("key") if isinstance(ref, Mapping) else ref
            role = ref.get("role") if isinstance(ref, Mapping) else None
            entry = by_key.get(key)
            if entry is None:
                if isinstance(key, str) and key.startswith(str(index.upstream.get("key_prefix", "")) + ":"):
                    errors.append(_err(7, key, f"node {node_id} references a key missing from the index"))
                continue
            if node_id not in entry.bpr_nodes:
                errors.append(_err(7, key, f"node {node_id} references the entry but bpr_nodes omits it"))
            if role not in ("premise", "hypothesis", "context"):
                errors.append(_err(7, key, f"node {node_id}: role must be premise, hypothesis or context"))
            if role == "premise" and not (entry.get("usage_cap") == "rigorous_input"
                                          and entry.get("relation") in STRONG_RELATIONS):
                errors.append(_err(7, key, f"node {node_id}: a premise needs relation input/counterexample "
                                           "and usage_cap rigorous_input"))
            if role == "hypothesis" and node.get("status") not in SOFT_STATUSES:
                errors.append(_err(7, key, f"node {node_id}: an external hypothesis forces status conditional or open"))
        for key in _ref_ids(node.get("premises")):
            entry = by_key.get(key)
            if entry is not None and not (entry.get("relation") in STRONG_RELATIONS
                                          and entry.get("usage_cap") == "rigorous_input"):
                errors.append(_err(7, key, f"node {node_id}: {entry.get('relation')!r} entries may never be premises"))
    return errors


def _rule8_citation(index: ExternalIndex) -> list:
    errors = []
    for e in index.entries:
        verdict = (e.get("review") or {}).get("verdict")
        try:
            text = citation(e)
        except ValueError:
            if verdict == "considered_not_cited" and not e.get("fidelity_check"):
                continue
            errors.append(_err(8, e.key, "citation could not be rendered"))
            continue
        except (KeyError, TypeError) as exc:
            errors.append(_err(8, e.key, f"citation could not be rendered ({exc!r})"))
            continue
        if str(e.get("verification")) not in text or "Unrefereed" not in text:
            errors.append(_err(8, e.key, "citation must contain the verification label and 'Unrefereed'"))
    return errors


def _rule9_fallback(index: ExternalIndex) -> list:
    return [
        _err(9, e.key, "supersedes_classical entries must name a classical_fallback")
        for e in index.entries
        if e.get("supersedes_classical") is True and not e.get("classical_fallback")
    ]


def _contains(haystack: str, needle: str) -> bool:
    return needle.casefold() in haystack.casefold()


def _rule10_lean_scope(index: ExternalIndex) -> list:
    errors = []
    vocab_sources = index.vocab.get("verification_source") or []
    for e in index.entries:
        label = e.get("verification")
        lean = e.get("lean")
        statement = str(e.get("statement_bpr", ""))
        excerpt = str(e.get("lean_scope_excerpt") or "")
        inside = e.get("lean_scope_terms") or []
        outside = e.get("outside_lean_scope") or []
        if label in LEAN_LABELS:
            if not isinstance(lean, Mapping):
                errors.append(_err(10, e.key, f"{label} needs a lean record"))
                continue
            if e.get("verification_source") not in vocab_sources:
                errors.append(_err(10, e.key, f"{label} needs verification_source in {vocab_sources}"))
            if not excerpt.strip():
                errors.append(_err(10, e.key, f"{label} needs a lean_scope_excerpt"))
            if not lean.get("declarations"):
                errors.append(_err(10, e.key, "lean.declarations must be non-empty"))
            if not HEX64.match(str(lean.get("statement_sha256", ""))):
                errors.append(_err(10, e.key, "lean.statement_sha256 must be a sha256 hex digest"))
            in_main = lean.get("in_main_results")
            if not isinstance(in_main, bool):
                errors.append(_err(10, e.key, "lean.in_main_results must be true or false"))
            elif in_main != (e.get("verification_source") == "formalization_yaml"):
                errors.append(_err(10, e.key, "verification_source must be formalization_yaml exactly "
                                              "when the comparator is in main_results"))
            if label == "lean_main_statement":
                if not inside:
                    errors.append(_err(10, e.key, "lean_main_statement needs lean_scope_terms"))
                for term in inside:
                    if not (_contains(statement, term) and _contains(excerpt, term)):
                        errors.append(_err(10, e.key, f"scope term {term!r} must appear in both "
                                                      "statement_bpr and lean_scope_excerpt"))
                for term in outside:
                    if _contains(statement, term):
                        errors.append(_err(10, e.key, f"statement_bpr mentions {term!r}, which is outside the "
                                                      "Lean scope; use lean_supporting_only or narrow the statement"))
            else:
                if not outside or not any(_contains(statement, t) for t in outside):
                    errors.append(_err(10, e.key, "lean_supporting_only must name, in outside_lean_scope, what "
                                                  "statement_bpr claims beyond the Lean scope"))
        else:
            if lean is not None:
                errors.append(_err(10, e.key, f"{label} entries must have lean = null"))
            if e.get("verification_source") is not None:
                errors.append(_err(10, e.key, f"{label} entries must have verification_source = null"))
            if inside:
                errors.append(_err(10, e.key, f"{label} entries cannot carry lean_scope_terms"))
    return errors


def validate(index: Union[ExternalIndex, MapLike], unification_map: MapLike,
             repo_root: Union[str, Path] = ROOT) -> list:
    """Return a list of rule violations; an empty list means the index is consistent."""
    if not isinstance(index, ExternalIndex):
        index = ExternalIndex(index)
    root = Path(repo_root)
    errors = _rule0_schema(index)
    if not isinstance(index.data.get("entries"), list):
        return errors
    errors += _rule1_pin(index)
    errors += _rule2_licence(index, root)
    errors += _rule3_vocab(index)
    errors += _rule4_references(index, unification_map, root)
    errors += _rule5_cap(index)
    errors += _rule6_inactive(index)
    errors += _rule7_map(index, unification_map)
    errors += _rule8_citation(index)
    errors += _rule9_fallback(index)
    errors += _rule10_lean_scope(index)
    return errors


__all__ = [
    "DEFAULT_OPENAI_MATH", "Entry", "ExternalIndex", "citation", "load_index", "validate",
]
