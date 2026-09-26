"""Integrity checks for doc/derivations/unification_map_2026-09-25.json.

These are bookkeeping checks (valid references, acyclic dependencies, status
vocabulary, no completion claim). They do not validate any physics.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAP_JSON = ROOT / "doc" / "derivations" / "unification_map_2026-09-25.json"
MAP_MD = ROOT / "doc" / "derivations" / "unification_map_2026-09-25.md"


@pytest.fixture(scope="module")
def data():
    return json.loads(MAP_JSON.read_text(encoding="utf-8"))


def test_top_level_flags(data):
    assert data["schema_version"] == 1
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
