"""Integrity checks for doc/derivations/unification_map_2026-09-25.json.

These are bookkeeping checks (valid references, acyclic dependencies, status
vocabulary, no completion claim). They do not validate any physics.

Schema 2 (2026-10-09) adds premises/context/hypotheses, a premise audit (an
analogue of Lean's ``#print axioms``), per-node verification records, a
status snapshot guard and a release history. The oracles here do not trust
the map: review anchors are resolved against the notes' actual headings,
exact_checks against the test files' actual function definitions, and claim
hashes are recomputed. The audit and the snapshot guard each have a negative
control that must fail.
"""

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAP_JSON = ROOT / "doc" / "derivations" / "unification_map_2026-09-25.json"
MAP_MD = ROOT / "doc" / "derivations" / "unification_map_2026-09-25.md"
SNAPSHOT = ROOT / "doc" / "derivations" / "status_snapshot.json"
HISTORY = ROOT / "doc" / "HISTORY.md"

LEVELS = ["none", "argument", "ai_review", "exact_computation", "formal_statement", "formal_proof"]
AUDITED = {"exact_theorem", "controlled_limit", "obstructed"}
PERMITTED = {"stipulated", "exact_theorem", "controlled_limit", "obstructed"}
NODE_KEYS = {"id", "layer", "status", "claim", "sources", "depends_on", "premises", "context", "hypotheses",
             "verification", "limitation"}
VERIFICATION_KEYS = {"level", "review", "additional_reviews", "exact_checks", "formal", "scope"}
REVIEW_KEYS = {"kind", "date", "blockers_found", "anchor"}
CHANGE_KEYS = {"id", "from", "to", "claim_sha256_from", "claim_sha256_to", "reason"}
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
BPR6D_RESULTS = ("flux_vacuum", "family_symmetry", "flux_selection", "yukawa_sector", "spin10_breaking",
                 "cc_self_tuning")


@pytest.fixture(scope="module")
def data():
    return json.loads(MAP_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def nodes(data):
    return {node["id"]: node for node in data["nodes"]}


@pytest.fixture(scope="module")
def snapshot():
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def claim_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def link_ids(node, key):
    if key == "depends_on":
        return list(node["depends_on"])
    return [link["id"] for link in node[key]]


def closure(nodes, start, key="premises"):
    """Every node reachable from start through links of the given kind."""
    seen, stack = set(), link_ids(nodes[start], key)
    while stack:
        current = stack.pop()
        if current not in seen:
            seen.add(current)
            stack.extend(link_ids(nodes[current], key))
    return seen


def audit(nodes, key="premises", honor_hypotheses=True):
    """Map audited node -> sorted non-permitted nodes it reaches (the premise audit)."""
    hits = {}
    for nid, node in nodes.items():
        if node["status"] not in AUDITED:
            continue
        named = {h["node"] for h in node["hypotheses"] if h["node"]} if honor_hypotheses else set()
        bad = sorted(t for t in closure(nodes, nid, key) if nodes[t]["status"] not in PERMITTED and t not in named)
        if bad:
            hits[nid] = bad
    return hits


def axioms(nodes, nid):
    reach = closure(nodes, nid) | {nid}
    return sorted(t for t in reach if nodes[t]["status"] == "stipulated")


def github_slug(heading):
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def sections(path):
    """GitHub-style heading slug -> text under that heading (code fences skipped)."""
    lines = path.read_text(encoding="utf-8").split("\n")
    found, fenced, counts = [], False, {}
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        match = None if fenced else re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if match:
            slug = github_slug(match.group(2))
            n = counts.get(slug, 0)
            counts[slug] = n + 1
            found.append((slug if n == 0 else f"{slug}-{n}", len(match.group(1)), i))
    result = {}
    for k, (slug, level, start) in enumerate(found):
        end = next((s for _, lv, s in found[k + 1:] if lv <= level), len(lines))
        result[slug] = "\n".join(lines[start + 1:end])
    return result


def pending_changes_problems(snapshot, revisions, current_nodes):
    """Forward guard: current status/claim must equal the snapshot after replaying later node_changes."""
    problems = []
    state = {nid: dict(entry) for nid, entry in snapshot["nodes"].items()}
    for revision in revisions[snapshot["revisions_included"]:]:
        for change in revision["node_changes"]:
            nid = change["id"]
            before = state.get(nid)
            if change["from"] is None:
                if before is not None:
                    problems.append(f"{nid}: added but already present")
            elif before is None or before["status"] != change["from"]:
                problems.append(f"{nid}: change from {change['from']} does not match {before}")
            elif change.get("claim_sha256_from") and change["claim_sha256_from"] != before["claim_sha256"]:
                problems.append(f"{nid}: claim_sha256_from does not match the recorded claim")
            if change["to"] is None:
                state.pop(nid, None)
            else:
                state[nid] = {"status": change["to"],
                              "claim_sha256": change.get("claim_sha256_to") or (before or {}).get("claim_sha256")}
    current = {n["id"]: {"status": n["status"], "claim_sha256": claim_hash(n["claim"])} for n in current_nodes}
    for nid in sorted(set(state) | set(current)):
        if state.get(nid) != current.get(nid):
            problems.append(f"{nid}: map {current.get(nid)} differs from snapshot+node_changes {state.get(nid)}")
    return problems


# ---------------------------------------------------------------------------
# Schema-1 checks (kept; only the schema version changed).
# ---------------------------------------------------------------------------

def test_top_level_flags(data):
    assert data["schema_version"] == 2
    assert data["empirical_validation"] is False
    assert data["toe_complete"] is False
    assert "not a proof score" in data["scope"].lower()


def test_nodes_are_well_formed(data):
    vocabulary = set(data["status_vocabulary"])
    layers = set(data["layers"])
    ids = [node["id"] for node in data["nodes"]]
    assert len(ids) == len(set(ids))
    for node in data["nodes"]:
        assert node["status"] in vocabulary, node["id"]
        assert node["layer"] in layers, node["id"]
        assert node["claim"].strip() and node["limitation"].strip(), node["id"]
        assert node["sources"], node["id"]
        for dep in node["depends_on"]:
            assert dep in ids, (node["id"], dep)


def test_every_source_exists(data):
    for node in data["nodes"]:
        for source in node["sources"]:
            assert (ROOT / source).exists(), (node["id"], source)


def test_dependencies_are_acyclic(data):
    graph = {node["id"]: node["depends_on"] for node in data["nodes"]}
    state = {}

    def visit(node):
        if state.get(node) == 1:
            raise AssertionError("dependency cycle through " + node)
        if state.get(node) == 2:
            return
        state[node] = 1
        for dep in graph[node]:
            visit(dep)
        state[node] = 2

    for node in graph:
        visit(node)


def test_every_layer_has_an_open_or_obstructed_link(data):
    # A layer with only proved links would contradict toe_complete=false.
    unresolved = {"open", "obstructed", "stipulated", "calibrated", "phenomenological_fit", "withdrawn"}
    for layer in data["layers"]:
        statuses = {node["status"] for node in data["nodes"] if node["layer"] == layer}
        assert statuses & unresolved, layer


def test_new_links_cite_their_derivations_and_modules(data):
    nodes = {node["id"]: node for node in data["nodes"]}
    for key in ("cubic_sector_vacuum", "cubic_mean_field_condensate", "acoustic_lorentz_window", "acoustic_metric"):
        assert "doc/derivations/cubic_condensate_regime_2026-09-25.md" in nodes[key]["sources"]
        assert "bpr/cubic_condensate_regime.py" in nodes[key]["sources"]
    assert "bpr/chiral_parent_completion.py" in nodes["chiral_parent_completion"]["sources"]
    assert nodes["chiral_parent"]["status"] == "obstructed"
    assert nodes["family_count"]["status"] == "conditional"


def test_markdown_links_json_and_disclaims_completion():
    text = MAP_MD.read_text(encoding="utf-8")
    assert "unification_map_2026-09-25.json" in text
    assert "not a completed theory of everything" in text
    assert "cubic_condensate_regime_2026-09-25.md" in text
    assert "chiral_parent_completion_2026-09-25.md" in text


def test_round_three_architecture_nodes(data):
    nodes = {node["id"]: node for node in data["nodes"]}
    assert data["architecture_decision"] == "doc/derivations/architecture_decision_2026-09-26.md"
    assert nodes["bpr6d_architecture"]["status"] == "stipulated"
    assert nodes["six_d_gravity"]["status"] == "stipulated"
    assert nodes["flux_vacuum"]["status"] == "exact_theorem"
    assert nodes["family_symmetry"]["status"] == "exact_theorem"
    # Negative round-3 results must stay recorded as obstructions, not silently dropped.
    assert nodes["flux_selection"]["status"] == "obstructed"
    assert nodes["yukawa_sector"]["status"] == "obstructed"
    assert nodes["common_limiting_speed"]["status"] == "obstructed"
    for key in ("flux_vacuum", "flux_selection", "six_d_gravity"):
        assert "doc/derivations/flux_compactification_2026-09-26.md" in nodes[key]["sources"]
    for key in ("family_symmetry", "yukawa_sector"):
        assert "doc/derivations/family_symmetry_from_flux_2026-09-26.md" in nodes[key]["sources"]
        assert "bpr/sphere_family_structure.py" in nodes[key]["sources"]
    text = MAP_MD.read_text(encoding="utf-8")
    assert "architecture_decision_2026-09-26.md" in text


def test_round_four_quantization_node(data):
    nodes = {node["id"]: node for node in data["nodes"]}
    gs = nodes["gs_quantization"]
    assert gs["status"] == "exact_theorem" and "chiral_parent_completion" in gs["depends_on"]
    assert "bpr/green_schwarz_quantization.py" in gs["sources"]
    assert "gs_quantization" in nodes["family_count"]["depends_on"]
    assert "multiple of three" in nodes["family_count"]["claim"]


def test_round_five_global_anomaly_node(data):
    nodes = {node["id"]: node for node in data["nodes"]}
    ga = nodes["global_anomalies"]
    assert ga["status"] == "exact_theorem" and "gs_quantization" in ga["depends_on"]
    assert "bpr/global_anomaly_bordism.py" in ga["sources"]
    assert "global_anomalies" in nodes["bpr6d_architecture"]["depends_on"]


def test_phase_1d_and_phase_2_nodes(data):
    nodes = {node["id"]: node for node in data["nodes"]}
    bp, ff = nodes["brane_positions"], nodes["fermion_fit"]
    assert bp["status"] == "conditional" and "bpr6d_minimal_model" in bp["depends_on"]
    assert "bpr/brane_stabilization.py" in bp["sources"]
    # Phase 2 is a fit, never a validation; the pinned failure must stay recorded.
    assert ff["status"] == "phenomenological_fit" and "brane_positions" in ff["depends_on"]
    assert "bpr/fermion_fit.py" in ff["sources"]
    assert "No empirical validation" in ff["limitation"] and "pinned" in ff["claim"]
    text = MAP_MD.read_text(encoding="utf-8")
    assert "brane_positions_2026-09-27.md" in text and "phase2_fermion_fit_2026-09-27.md" in text


def test_phase_2b_node(data):
    nodes = {node["id"]: node for node in data["nodes"]}
    kc = nodes["kinetic_corrections"]
    assert kc["status"] == "phenomenological_fit" and "fermion_fit" in kc["depends_on"]
    assert "bpr/kinetic_corrections.py" in kc["sources"] and "not a proof" in kc["limitation"]
    assert "vertex" in kc["claim"] and "not predicted" in kc["claim"]
    assert "kinetic_corrections_2026-10-01.md" in MAP_MD.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Schema 2: premises, context and hypotheses.
# ---------------------------------------------------------------------------

def test_node_fields_are_exactly_the_schema(data):
    # No other node fields; in particular no external references in this release.
    for node in data["nodes"]:
        assert set(node) == NODE_KEYS, node["id"]
        assert set(node["verification"]) == VERIFICATION_KEYS, node["id"]


def test_depends_on_is_the_union_of_premises_and_context(nodes):
    for nid, node in nodes.items():
        premises, context = link_ids(node, "premises"), link_ids(node, "context")
        for link in node["premises"] + node["context"]:
            assert set(link) == {"id", "reason"}, nid
            assert link["id"] in nodes, (nid, link["id"])
            assert isinstance(link["reason"], str) and len(link["reason"].strip()) > 10, (nid, link["id"])
        assert len(premises) == len(set(premises)) and len(context) == len(set(context)), nid
        assert not set(premises) & set(context), nid
        assert sorted(premises + context) == sorted(node["depends_on"]), nid


def test_stipulated_nodes_have_no_premises(nodes):
    # A stipulated node is an input; its depends_on entries are motivation (context).
    for nid, node in nodes.items():
        if node["status"] == "stipulated":
            assert node["premises"] == [], nid


def test_hypotheses_are_well_formed(nodes):
    for nid, node in nodes.items():
        names = [h["name"] for h in node["hypotheses"]]
        assert len(names) == len(set(names)), nid
        for hyp in node["hypotheses"]:
            assert set(hyp) == {"name", "node", "statement"}, nid
            assert re.match(r"^[a-z0-9_]+$", hyp["name"]), (nid, hyp["name"])
            assert len(hyp["statement"].strip()) > 20, (nid, hyp["name"])
            if hyp["node"] is not None:
                # A named node must actually be used through premises, or the hypothesis is misattached.
                assert hyp["node"] in closure(nodes, nid), (nid, hyp["node"])
        if node["status"] == "conditional":
            assert node["hypotheses"], nid  # "Follows if named, unproven hypotheses hold."


def test_premise_closure_is_permitted(nodes):
    # Lean '#print axioms' analogue: audited nodes rest only on permitted nodes or named hypotheses.
    assert audit(nodes) == {}


def test_premise_audit_detects_the_schema_one_chain(nodes):
    # Negative control: read every depends_on link as a premise (schema 1). The audit must flag the chain
    # six_d_gravity -> bpr6d_architecture -> common_limiting_speed -> emergent_gauge_field and the flavor fit.
    raw = audit(nodes, key="depends_on", honor_hypotheses=False)
    assert set(raw) == set(BPR6D_RESULTS) | {"common_limiting_speed", "flavor_source_selection"}
    for nid in BPR6D_RESULTS:
        assert raw[nid] == ["emergent_gauge_field"], nid
    assert raw["flavor_source_selection"] == ["emergent_gauge_field", "family_count", "flavor_formulas"]
    # Honoring hypotheses alone does not clear the BPR-6D results: the reclassification to context does.
    assert set(audit(nodes, key="depends_on")) == set(BPR6D_RESULTS) | {"flavor_source_selection"}


def test_audit_resolutions_are_recorded(nodes):
    arch = nodes["bpr6d_architecture"]
    assert "common_limiting_speed" in link_ids(arch, "context")
    assert "bpr6d_architecture" in link_ids(nodes["six_d_gravity"], "context")
    for nid in BPR6D_RESULTS:
        assert "emergent_gauge_field" not in closure(nodes, nid), nid
        assert "emergent_gauge_field" in closure(nodes, nid, "depends_on"), nid  # link kept, reclassified
    # The genuine use is a premise named as a hypothesis, not hidden as context.
    speed = nodes["common_limiting_speed"]
    assert "emergent_gauge_field" in link_ids(speed, "premises")
    assert [h["node"] for h in speed["hypotheses"]] == ["emergent_gauge_field"]
    assert speed["status"] == "obstructed"
    flavor = nodes["flavor_source_selection"]
    assert link_ids(flavor, "context") == ["flavor_formulas"] and link_ids(flavor, "premises") == ["ring_substrate"]


# ---------------------------------------------------------------------------
# Schema 2: verification records.
# ---------------------------------------------------------------------------

def test_verification_vocabulary(data):
    assert data["verification_levels"] == LEVELS
    vocabulary = data["verification_vocabulary"]
    assert list(vocabulary) == LEVELS
    assert "AI review of scoped mathematics, not certification" in vocabulary["ai_review"]
    for level in ("formal_statement", "formal_proof"):
        assert "Not used" in vocabulary[level]
    assert set(data["review_kinds"]) == {"ai_adversarial", "human"}
    audit_spec = data["premise_audit"]
    assert set(audit_spec["audited_statuses"]) == AUDITED and set(audit_spec["permitted_statuses"]) == PERMITTED


def test_verification_rules(data, nodes):
    for nid, node in nodes.items():
        ver = node["verification"]
        level, review = ver["level"], ver["review"]
        assert level in LEVELS, nid
        rank = LEVELS.index(level)
        if node["status"] in {"stipulated", "open", "withdrawn"}:
            assert level == "none" and review is None, nid
        if node["status"] in AUDITED | {"conditional"}:
            assert rank >= LEVELS.index("argument"), nid
        if node["status"] in AUDITED:
            assert review is not None, nid  # every exact, controlled or obstructed node cites its review
        # A recorded review is what ai_review and above mean, and nothing below it has one.
        assert (review is not None) == (rank >= LEVELS.index("ai_review")), nid
        assert review is not None or ver["additional_reviews"] == [], nid
        for rec in ([review] if review else []) + ver["additional_reviews"]:
            assert set(rec) == REVIEW_KEYS, nid
            assert rec["kind"] in data["review_kinds"] and DATE.match(rec["date"]), nid
            assert rec["blockers_found"] is None or (isinstance(rec["blockers_found"], int)
                                                     and rec["blockers_found"] >= 0), nid
        if level == "exact_computation":
            assert ver["exact_checks"] and ver["scope"] and ver["scope"].strip(), nid
        if rank < LEVELS.index("exact_computation"):
            assert ver["exact_checks"] == [], nid
        if level.startswith("formal_"):
            # A formal level needs a catalogue entry with a comparator configuration on disk.
            assert isinstance(ver["formal"], dict), nid
            assert (ROOT / ver["formal"]["comparator_config"]).is_file(), nid
        else:
            assert ver["formal"] is None, nid


def test_review_anchors_resolve_to_review_headings(nodes):
    cache = {}
    for nid, node in nodes.items():
        ver = node["verification"]
        for rec in ([ver["review"]] if ver["review"] else []) + ver["additional_reviews"]:
            path, _, slug = rec["anchor"].partition("#")
            assert path in node["sources"], (nid, path)
            assert slug, (nid, rec["anchor"])
            if path not in cache:
                cache[path] = sections(ROOT / path)
            assert slug in cache[path], (nid, rec["anchor"], sorted(cache[path]))
            assert re.search(r"review", cache[path][slug], re.I), (nid, rec["anchor"])


def test_exact_checks_name_real_tests(nodes):
    for nid, node in nodes.items():
        for check in node["verification"]["exact_checks"]:
            path, _, name = check.partition("::")
            assert path.startswith("tests/") and name.startswith("test_"), (nid, check)
            source = (ROOT / path).read_text(encoding="utf-8")
            assert re.search(rf"^def {re.escape(name)}\(", source, re.M), (nid, check)


def test_no_verified_wording_without_a_formal_proof(nodes):
    pattern = re.compile(r"\b(verified|certified|formally proved)\b", re.I)
    for nid, node in nodes.items():
        if node["verification"]["level"] == "formal_proof":
            continue
        texts = [node["claim"], node["limitation"], node["verification"]["scope"] or ""]
        for text in texts:
            assert not pattern.search(text), (nid, pattern.search(text).group(0))


def test_no_percentage_verified_is_displayed():
    claims = re.compile(r"\d+(\.\d+)?\s*%[^.\n]{0,40}\b(verified|formali[sz]ed|reviewed|proved|certified)"
                        r"|\b(verified|formali[sz]ed|reviewed|proved|certified)\b[^.\n%]{0,40}\d+(\.\d+)?\s*%"
                        r"|\d+\s*/\s*\d+\s+(nodes\s+)?(verified|formali[sz]ed)", re.I)
    for path in (MAP_JSON, MAP_MD, HISTORY):
        match = claims.search(path.read_text(encoding="utf-8"))
        assert match is None, (path.name, match and match.group(0))


def test_three_notes_have_standard_review_headings():
    for note in ("substrate_local_symmetry_2026-09-13.md", "flavor_source_selection_2026-09-12.md",
                 "foundation_observation_gate_2026-09-13.md"):
        found = sections(ROOT / "doc" / "derivations" / note)
        assert "independent-review" in found, note
        assert re.search(r"review", found["independent-review"], re.I), note


def test_markdown_renders_axioms_levels_and_hypotheses(nodes):
    text = MAP_MD.read_text(encoding="utf-8")
    rows = {}
    for line in text.split("\n"):
        match = re.match(r"^\| `([a-z0-9_]+)` \| ([a-z_]+) \| ([a-z_]+) \| (.*?) \| (.*?) \|$", line)
        if match:
            rows[match.group(1)] = match.groups()[1:]
    assert set(rows) == set(nodes)
    for nid, node in nodes.items():
        status, level, axiom_cell, hyp_cell = rows[nid]
        assert status == node["status"] and level == node["verification"]["level"], nid
        expected = ", ".join(f"`{a}`" for a in axioms(nodes, nid)) or "none"
        assert axiom_cell == expected, nid
        hyps = ", ".join(f"`{h['name']}`" + (f" (node `{h['node']}`)" if h["node"] else "")
                         for h in node["hypotheses"]) or "none"
        assert hyp_cell == hyps, nid
    # The BPR-6D results rest on the stipulated six-dimensional gravity, not on the link-boson amendment.
    for nid in BPR6D_RESULTS:
        assert axioms(nodes, nid) == ["six_d_gravity"], nid


def test_markdown_points_to_external_index_and_history():
    text = MAP_MD.read_text(encoding="utf-8")
    assert "external/OPENAI_MATH.md" in text and re.search(r"No\s+node\s+depends\s+on\s+it", text)
    assert "HISTORY.md" in text and "status_snapshot.json" in text


# ---------------------------------------------------------------------------
# Schema 2: revisions, snapshot guard and release history.
# ---------------------------------------------------------------------------

def test_revisions_are_well_formed(data, nodes):
    vocabulary = set(data["status_vocabulary"])
    dates = [revision["date"] for revision in data["revisions"]]
    assert dates == sorted(dates)
    for revision in data["revisions"]:
        assert set(revision) == {"date", "pre_change_commit", "summary", "node_changes"}
        assert DATE.match(revision["date"]) and revision["summary"].strip()
        assert revision["pre_change_commit"] is None or HEX40.match(revision["pre_change_commit"])
        for change in revision["node_changes"]:
            assert set(change) <= CHANGE_KEYS and {"id", "from", "to", "reason"} <= set(change), change
            assert change["from"] in vocabulary | {None} and change["to"] in vocabulary | {None}, change
            assert change["from"] is not None or change["to"] is not None, change
            assert change["to"] is None or change["id"] in nodes, change
            assert len(change["reason"].strip()) > 10, change
            for key in ("claim_sha256_from", "claim_sha256_to"):
                assert change.get(key) is None or HEX64.match(change[key]), change
            if change["from"] is None:
                assert change.get("claim_sha256_from") is None and change.get("claim_sha256_to"), change
            elif change["from"] == change["to"]:
                # A same-status entry must record a claim change, or it records nothing.
                assert change.get("claim_sha256_from") and change.get("claim_sha256_to"), change
                assert change["claim_sha256_from"] != change["claim_sha256_to"], change


def test_status_snapshot_guard(data, snapshot):
    assert snapshot["map"] == "doc/derivations/unification_map_2026-09-25.json"
    assert 0 < snapshot["revisions_included"] <= len(data["revisions"])
    assert data["revisions"][snapshot["revisions_included"] - 1]["date"] == snapshot["last_included_revision_date"]
    assert data["status_snapshot"] == "doc/derivations/status_snapshot.json"
    assert pending_changes_problems(snapshot, data["revisions"], data["nodes"]) == []


def test_snapshot_guard_rejects_unrecorded_changes(data, snapshot):
    # Negative controls: a silent status change or claim edit must fail; a recorded one must pass.
    edited = copy.deepcopy(data)
    target = next(n for n in edited["nodes"] if n["id"] == "flux_selection")
    target["status"] = "exact_theorem"
    assert pending_changes_problems(snapshot, edited["revisions"], edited["nodes"])
    target["status"] = "obstructed"
    target["claim"] += " Edited."
    assert pending_changes_problems(snapshot, edited["revisions"], edited["nodes"])
    old_hash = snapshot["nodes"]["flux_selection"]["claim_sha256"]
    edited["revisions"].append({"date": "2099-01-01", "pre_change_commit": None, "summary": "control",
                                "node_changes": [{"id": "flux_selection", "from": "obstructed", "to": "obstructed",
                                                  "claim_sha256_from": old_hash,
                                                  "claim_sha256_to": claim_hash(target["claim"]),
                                                  "reason": "negative-control edit"}]})
    assert pending_changes_problems(snapshot, edited["revisions"], edited["nodes"]) == []


def test_recorded_history_chains_into_the_snapshot(data, snapshot):
    # Backward check: within the folded-in revisions each node's changes chain, and end at the snapshot.
    last = {}
    for revision in data["revisions"][:snapshot["revisions_included"]]:
        for change in revision["node_changes"]:
            nid, previous = change["id"], last.get(change["id"])
            if previous is not None:
                assert change["from"] == previous["status"], change
                if change.get("claim_sha256_from") and previous["claim"]:
                    assert change["claim_sha256_from"] == previous["claim"], change
            claim = change.get("claim_sha256_to") or (previous or {}).get("claim")
            last[nid] = {"status": change["to"], "claim": claim}
    for nid, final in last.items():
        assert snapshot["nodes"][nid]["status"] == final["status"], nid
        if final["claim"]:
            assert snapshot["nodes"][nid]["claim_sha256"] == final["claim"], nid


def test_thermodynamic_condensation_rescoped(data, nodes):
    node = nodes["thermodynamic_condensation"]
    assert node["status"] == "open"
    for phrase in ("superfluid region", "Mott lobes", "29.3", "Capogrosso-Sansone", "numerical"):
        assert phrase in node["claim"], phrase
    for phrase in ("reflection positivity", "Dyson-Lieb-Simon 1978", "Kennedy-Lieb-Shastry 1988"):
        assert phrase in node["limitation"], phrase
    changes = [c for r in data["revisions"] for c in r["node_changes"] if c["id"] == "thermodynamic_condensation"]
    assert len(changes) == 1
    change = changes[0]
    assert change["from"] == change["to"] == "open"
    assert change["claim_sha256_to"] == claim_hash(node["claim"])
    assert "openai/math is not the basis" in change["reason"]


def test_history_file_records_every_revision(data):
    text = HISTORY.read_text(encoding="utf-8")
    for label in ("Withdrawals and downgrades", "Repairs after review", "Status upgrades",
                  "External-input changes", "Formalizations"):
        assert f"**{label}.**" in text, label
    headings = re.findall(r"^## (\d{4}-\d{2}-\d{2}):", text, re.M)
    assert sorted(set(headings)) == sorted({r["date"] for r in data["revisions"]})
    for revision in data["revisions"]:
        if revision["pre_change_commit"]:
            assert revision["pre_change_commit"] in text, revision["date"]
    assert data["history"] == "doc/HISTORY.md"
