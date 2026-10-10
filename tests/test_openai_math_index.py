"""Checks for doc/external/openai_math_index.json and its policy (doc/external/OPENAI_MATH.md).

These are bookkeeping checks: the index validates against the unification map, every validation rule
fires on a targeted corruption (so the validator is not vacuous), citations carry the verification label,
and the guardrail holds (no build file depends on openai/math; no upstream file is copied into the repo).
None of this says anything about whether an upstream result is correct.
"""

import copy
import json
import re
from pathlib import Path

import pytest

import bpr.external_refs as er

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "doc" / "derivations" / "unification_map_2026-09-25.json"
POLICY = ROOT / "doc" / "external" / "OPENAI_MATH.md"
COMMIT = "fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb"


@pytest.fixture(scope="module")
def raw():
    return json.loads(er.DEFAULT_OPENAI_MATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def umap():
    return json.loads(MAP.read_text(encoding="utf-8"))


def _errors(data, umap):
    return er.validate(er.ExternalIndex(copy.deepcopy(data)) if not isinstance(data, er.ExternalIndex) else data, umap, ROOT)


def _rules(errors):
    return {int(re.match(r"rule(\d+):", e).group(1)) for e in errors}


def test_real_index_validates(raw, umap):
    assert er.validate(er.load_index(), umap, ROOT) == []
    assert raw["upstream"]["commit"] == COMMIT and raw["upstream"]["license"] == "Apache-2.0"
    assert raw["upstream"]["redistributed_files"] == []


def test_bottom_line_nothing_above_context(raw, umap):
    """No entry is usable as an input, and no node's status depends on the index."""
    assert all(e["usage_cap"] == "context_only" for e in raw["entries"])
    assert all(e["relation"] not in ("input", "counterexample") for e in raw["entries"])
    nodes = {n["id"]: n for n in umap["nodes"]}
    for node in nodes.values():
        for link in node.get("premises", []):
            assert not str(link["id"]).startswith("omath:")
    # The map loader never reads the index: the map does not mention index keys at all.
    assert "omath:" not in MAP.read_text(encoding="utf-8")


@pytest.mark.parametrize("rule, mutate", [
    (1, lambda d: d["upstream"].__setitem__("commit", "main")),
    (1, lambda d: d["entries"][0].__setitem__("permalink", d["entries"][0]["permalink"].replace(COMMIT, "main"))),
    (2, lambda d: d["upstream"].__setitem__("license", "MIT")),
    (2, lambda d: d["upstream"].__setitem__("redistributed_files", ["doc/copied_paper.pdf"])),
    (3, lambda d: d["entries"][0].__setitem__("relation", "proof")),
    (4, lambda d: d["entries"][0].__setitem__("bpr_nodes", ["no_such_node"])),
    (4, lambda d: d["entries"][0].__setitem__("bpr_files", ["bpr/no_such_file.py"])),
    (5, lambda d: d["entries"][1].__setitem__("usage_cap", "rigorous_input")),
])
def test_each_rule_fires_on_a_targeted_corruption(raw, umap, rule, mutate):
    data = copy.deepcopy(raw)
    mutate(data)
    assert rule in _rules(_errors(data, umap))


def test_every_citation_names_its_label_and_pin():
    for entry in er.load_index().entries:
        if entry["review"]["verdict"] == "considered_not_cited":
            with pytest.raises(ValueError):  # inert entries may not be cited at all (e.g. 003, quasi-Riemann)
                er.citation(entry)
            continue
        text = er.citation(entry)
        assert "Unrefereed" in text and entry.verification in text and COMMIT[:9] in text


def test_rigorous_input_requires_a_build_by_bpr(raw, umap):
    """The strongest cap needs a Lean main statement re-run by BPR (bpr_check == built), not just read."""
    data = copy.deepcopy(raw)
    lean_entry = next(e for e in data["entries"] if e["verification"] == "lean_main_statement" and e["relation"] == "context")
    lean_entry["usage_cap"] = "rigorous_input"
    assert 5 in _rules(_errors(data, umap))


def test_considered_not_cited_entries_are_inert(raw, umap):
    inert = [e for e in raw["entries"] if e["review"]["verdict"] == "considered_not_cited"]
    assert inert and all(e["relation"] == "none" and not e["bpr_nodes"] for e in inert)
    data = copy.deepcopy(raw)
    target = next(e for e in data["entries"] if e["review"]["verdict"] == "considered_not_cited")
    target["bpr_nodes"] = ["thermodynamic_condensation"]
    assert 6 in _rules(_errors(data, umap))


def test_lean_label_must_match_the_lean_scope(raw, umap):
    """Rule 10: the headline-versus-formal-scope trap. Family 267's Lean scope is ground-state condensation only;
    claiming its positive-temperature headline under the Lean label must fail."""
    data = copy.deepcopy(raw)
    e = next(x for x in data["entries"] if x["family_id"] == "267")
    e["statement_bpr"] = "Bose-Einstein condensation at positive temperature with exact quantum depletion."
    assert 10 in _rules(_errors(data, umap))


def test_family_level_lean_badges_are_avoided(raw):
    """Labels are per cited statement, with their source recorded; lean_docs_only comparators are not in
    formalization.yaml main_results, and every Lean entry names its comparator declarations."""
    for e in raw["entries"]:
        if e["verification"].startswith("lean_"):
            assert e["verification_source"] in ("formalization_yaml", "lean_docs_only")
            assert e["lean"]["declarations"] and e["lean_scope_excerpt"]
            assert e["bpr_check"] in ("none", "statement_read")  # BPR never built anything here
        else:
            assert e["lean"] is None


def test_index_queries(raw):
    idx = er.load_index()
    keys = {e["key"] for e in raw["entries"]}
    assert len(keys) == len(raw["entries"])
    assert {x.key for x in idx.entries_for_node("thermodynamic_condensation")} <= keys
    assert all("bpr/optimization.py" in x.bpr_files for x in idx.entries_for_file("bpr/optimization.py"))
    with pytest.raises(KeyError):
        idx.entry("omath:999:missing")


BUILD_FILES = ["pyproject.toml", "Dockerfile", "docker-compose.yml"] + [p.name for p in ROOT.glob("requirements*.txt")] \
    + [p.name for p in ROOT.glob("environment*.yml")]


def test_guardrail_no_build_dependency_on_openai_math():
    for name in BUILD_FILES:
        text = (ROOT / name).read_text(encoding="utf-8", errors="ignore").lower()
        assert "openai/math" not in text and "openai-math" not in text, name
    for lake in list(ROOT.rglob("lakefile.lean")) + list(ROOT.rglob("lakefile.toml")):
        if ".git" not in lake.parts:
            assert "openai" not in lake.read_text(encoding="utf-8", errors="ignore").lower(), lake


def test_guardrail_no_upstream_files_copied():
    """Upstream preprint READMEs carry 'OpenAI Math Release preprint' BibTeX; no such file may be committed, and
    no upstream blob/main link may appear anywhere in the repo's text files."""
    hits = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix in (".pyc", ".png", ".pdf", ".npy", ".npz"):
            continue
        if path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if path.name == Path(__file__).name:
            continue
        if "OpenAI Math Release preprint" in text or "github.com/openai/math/blob/main" in text:
            hits.append(str(path.relative_to(ROOT)))
    assert hits == []


def test_policy_document_states_the_rules():
    text = POLICY.read_text(encoding="utf-8")
    for needle in (COMMIT, "Apache", "never", "python3 -I", "lean_main_statement", "context_only",
                   "No openai/math result changes the status"):
        assert needle in text, needle
