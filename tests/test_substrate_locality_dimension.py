"""Independent tests of the frozen 2026-09-13 locality counting contract.

Oracles use tiny coordinate BFS or explicit cycle adjacency, not implementation
helpers.  No embeddings are searched.  The largest oracle has 129 vertices;
the 4096-site public box-cap check does not construct an oracle graph.
Authored without reading or executing the implementation or demo.
"""

import builtins
from collections import deque
from fractions import Fraction
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

import bpr.substrate_locality_dimension as locality


MAX_INTEGER = 10 ** 6
MAX_BOX_SITES = 4096


class IntSubclass(int):
    pass


class ListSubclass(list):
    pass


class TupleSubclass(tuple):
    pass


class ConversionTrap:
    def __int__(self):
        raise AssertionError("integer coercion must not run")

    def __index__(self):
        raise AssertionError("index coercion must not run")

    def __float__(self):
        raise AssertionError("float coercion must not run")


BAD_SCALARS = [
    True, False, 1.0, float("nan"), float("inf"), "1", b"1", None,
    Fraction(1, 1), 1 + 0j, IntSubclass(1), ConversionTrap(), [1], (1,),
]


def coordinate_ball(dimension, radius, shape=None, center=None):
    """Discover vertices by unit edges; clip only when a box was supplied."""
    start = (0,) * dimension if center is None else tuple(center)
    distances = {start: 0}
    pending = deque([start])
    while pending:
        point = pending.popleft()
        distance = distances[point]
        if distance == radius:
            continue
        for axis in range(dimension):
            for step in (-1, 1):
                neighbor = list(point)
                neighbor[axis] += step
                neighbor = tuple(neighbor)
                if shape is not None and any(
                    value < 0 or value >= side
                    for value, side in zip(neighbor, shape)
                ):
                    continue
                if neighbor not in distances:
                    distances[neighbor] = distance + 1
                    pending.append(neighbor)
    return set(distances)


def cycle_graph(length):
    return {
        vertex: {(vertex - 1) % length, (vertex + 1) % length}
        for vertex in range(length)
    }


def graph_ball(graph, start, radius):
    reached = {start}
    frontier = {start}
    for unused in range(radius):
        frontier = {
            neighbor
            for vertex in frontier
            for neighbor in graph[vertex]
            if neighbor not in reached
        }
        reached.update(frontier)
        if not frontier:
            break
    return reached


def assert_finite_json(value):
    """Reject non-native containers/scalars as well as nonfinite JSON numbers."""
    if type(value) is dict:
        for key, item in value.items():
            assert type(key) is str
            assert_finite_json(item)
    elif type(value) is list:
        for item in value:
            assert_finite_json(item)
    elif type(value) is float:
        assert math.isfinite(value)
    else:
        assert value is None or type(value) in (str, int, bool)
    encoded = json.dumps(value, allow_nan=False, sort_keys=True)
    assert json.loads(encoded) == value


def assert_scope(scope):
    assert type(scope) is dict
    for flag in (
        "map_existence_established", "physical_dimension_derived",
        "empirical_validation",
    ):
        assert scope[flag] is False
    # The wording/key for the bounded-fiber assumption is not frozen.
    text = json.dumps(scope).lower().replace("_", " ").replace("-", " ")
    assert "bound" in text and "fiber" in text


def assert_growth_report(report, case, source_count=None, target_count=None):
    dimension, radius, length, numerator, denominator, multiplicity = case
    effective = numerator // denominator
    if source_count is None:
        assert radius <= 4, "large-input checks must supply an algebraic oracle"
        source_count = len(coordinate_ball(dimension, radius))
    if target_count is None:
        assert length <= 31 and effective * radius <= 40
        target_count = len(graph_ball(cycle_graph(length), 0, effective * radius))
    capacity = multiplicity * target_count
    excess = source_count - capacity
    assert type(report) is dict
    assert set(report) == {
        "source_kind", "dimension", "radius", "target_sites", "dilation",
        "multiplicity", "source_count", "target_ball_count", "capacity",
        "signed_excess", "status", "scope",
    }
    assert report["source_kind"] == "integer_lattice_ball"
    expected = {
        "dimension": dimension, "radius": radius, "target_sites": length,
        "multiplicity": multiplicity, "source_count": source_count,
        "target_ball_count": target_count, "capacity": capacity,
        "signed_excess": excess,
    }
    for key, value in expected.items():
        assert type(report[key]) is int
        assert report[key] == value
    assert type(report["dilation"]) is dict
    assert report["dilation"] == {
        "numerator": numerator, "denominator": denominator,
        "effective_integer": effective,
    }
    assert all(type(value) is int for value in report["dilation"].values())
    assert report["status"] == (
        "excluded_by_ball_count" if excess > 0 else "not_excluded_by_ball_count"
    )
    assert_scope(report["scope"])
    assert_finite_json(report)


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("radius", [0, 1, 2, 3, 4])
def test_lattice_counts_against_tiny_coordinate_bfs(dimension, radius):
    actual = locality.lattice_ball_count(dimension, radius)
    assert type(actual) is int
    assert actual == len(coordinate_ball(dimension, radius))


@pytest.mark.parametrize("length", [3, 4, 5, 6, 7, 8, 9, 10])
@pytest.mark.parametrize("radius", [0, 1, 2, 3, 4, 5, 7])
def test_cycle_counts_against_explicit_odd_even_cycles(length, radius):
    graph = cycle_graph(length)
    actual = locality.ring_ball_count(length, radius)
    assert type(actual) is int
    # Check wraparound from the last vertex as well as the distinguished zero.
    assert actual == len(graph_ball(graph, 0, radius))
    assert actual == len(graph_ball(graph, length - 1, radius))


@pytest.mark.parametrize("shape,center", [
    ([1], [0]), ([5], [0]), ([5], [2]), ([2, 2], [0, 0]),
    ([3, 4], [1, 0]), ([3, 3, 3], [1, 1, 1]),
    ([1, 1, 5], [0, 0, 2]), ([2, 3, 4], [1, 1, 0]),
    ([5, 5, 5], [2, 2, 2]),
])
@pytest.mark.parametrize("radius", [0, 1, 2, 3, 4])
def test_box_counts_against_boundary_aware_bfs(shape, center, radius):
    before_shape = shape[:]
    before_center = center[:]
    actual = locality.box_ball_count(shape, center, radius)
    assert type(actual) is int
    assert actual == len(coordinate_ball(len(shape), radius, shape, center))
    assert shape == before_shape and center == before_center


@pytest.mark.parametrize("shape_type", [list, tuple])
@pytest.mark.parametrize("center_type", [list, tuple])
def test_builtin_container_combinations(shape_type, center_type):
    assert locality.box_ball_count(shape_type([3, 3]), center_type([1, 1]), 1) == 5


def test_interior_formula_is_not_substituted_at_box_boundaries():
    assert locality.box_ball_count([5, 5, 5], [2, 2, 2], 2) == 25
    assert locality.box_ball_count([3, 3, 3], [1, 1, 1], 2) == 19
    assert locality.box_ball_count([3, 3, 3], [1, 1, 1], 3) == 27
    assert locality.lattice_ball_count(3, 2) == 25
    assert locality.lattice_ball_count(3, 3) == 63


@pytest.mark.parametrize("case,excess", [
    ((2, 1, 31, 1, 1, 1), 2),
    ((1, 1, 31, 1, 1, 1), 0),
    ((1, 1, 31, 1, 1, 2), -3),
    ((3, 0, 31, 0, 1, 1), 0),
    ((3, 0, 31, 0, 1, 2), -1),
    ((3, 1, 31, 0, 1, 1), 6),
    ((1, 2, 9, 0, 1, 5), 0),
    ((1, 2, 9, 0, 1, 6), -1),
])
def test_exact_positive_equality_negative_and_zero_cases(case, excess):
    report = locality.growth_report(*case)
    assert_growth_report(report, case)
    assert report["signed_excess"] == excess


def test_growth_defaults_match_explicit_unit_parameters():
    assert locality.growth_report(2, 2, 31) == locality.growth_report(2, 2, 31, 1, 1, 1)


@pytest.mark.parametrize("case,expected_target", [
    ((3, 2, 31, 3, 2, 1), 5),
    ((3, 2, 31, 6, 4, 1), 5),
    ((2, 3, 31, 1, 2, 1), 1),
    ((2, 3, 31, 2, 3, 1), 1),
    ((1, 2, 31, 5, 2, 1), 9),
])
def test_rational_dilation_floors_each_edge_not_total_radius(case, expected_target):
    report = locality.growth_report(*case)
    assert_growth_report(report, case)
    assert report["target_ball_count"] == expected_target


def test_ring_saturation_enforces_global_capacity():
    case = (3, 2, 3, 10, 1, 8)
    report = locality.growth_report(*case)
    assert_growth_report(report, case)
    assert report["target_ball_count"] == 3
    assert report["capacity"] == 24
    assert report["signed_excess"] == 1
    assert report["status"] == "excluded_by_ball_count"


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_scalar_radius_ceiling_and_exact_large_output(dimension):
    radius = MAX_INTEGER
    algebraic_counts = {
        1: 2 * radius + 1,
        2: 2 * radius ** 2 + 2 * radius + 1,
        3: (4 * radius ** 3 + 6 * radius ** 2 + 8 * radius + 3) // 3,
    }
    count = locality.lattice_ball_count(dimension, radius)
    assert type(count) is int
    assert count == algebraic_counts[dimension]
    assert count > MAX_INTEGER
    assert locality.ring_ball_count(MAX_INTEGER, radius) == MAX_INTEGER


def test_internal_radius_and_capacity_may_exceed_public_input_ceiling():
    radius = MAX_INTEGER
    case = (3, radius, MAX_INTEGER, MAX_INTEGER, 1, MAX_INTEGER)
    source_count = (4 * radius ** 3 + 6 * radius ** 2 + 8 * radius + 3) // 3
    # k*r = 10**12 is internal, not a new public radius input.
    report = locality.growth_report(*case)
    assert_growth_report(report, case, source_count, MAX_INTEGER)
    assert report["capacity"] == MAX_INTEGER ** 2


def test_numerator_denominator_and_multiplicity_input_extrema():
    case = (1, 1, MAX_INTEGER, MAX_INTEGER, MAX_INTEGER, MAX_INTEGER)
    report = locality.growth_report(*case)
    assert_growth_report(report, case, 3, 3)
    assert report["dilation"]["effective_integer"] == 1
    assert report["signed_excess"] == 3 - 3 * MAX_INTEGER
    case = (1, 1, 3, 1, MAX_INTEGER, 1)
    assert_growth_report(locality.growth_report(*case), case)


SCALAR_SLOTS = [
    ("lattice_ball_count", (2, 2), 0),
    ("lattice_ball_count", (2, 2), 1),
    ("ring_ball_count", (7, 2), 0),
    ("ring_ball_count", (7, 2), 1),
    ("box_ball_count", ([3, 3], [1, 1], 2), 2),
] + [("growth_report", (2, 2, 7, 3, 2, 1), slot) for slot in range(6)]


@pytest.mark.parametrize("function_name,arguments,slot", SCALAR_SLOTS)
@pytest.mark.parametrize("bad", BAD_SCALARS)
def test_all_public_scalar_positions_reject_nonbuiltin_ints(function_name, arguments, slot, bad):
    arguments = list(arguments)
    arguments[slot] = bad
    with pytest.raises(ValueError):
        getattr(locality, function_name)(*arguments)


@pytest.mark.parametrize("function_name,arguments", [
    ("lattice_ball_count", (0, 0)),
    ("lattice_ball_count", (4, 0)),
    ("lattice_ball_count", (1, -1)),
    ("lattice_ball_count", (1, MAX_INTEGER + 1)),
    ("ring_ball_count", (2, 0)),
    ("ring_ball_count", (MAX_INTEGER + 1, 0)),
    ("ring_ball_count", (3, -1)),
    ("ring_ball_count", (3, MAX_INTEGER + 1)),
    ("box_ball_count", ([1], [0], -1)),
    ("box_ball_count", ([1], [0], MAX_INTEGER + 1)),
])
def test_count_scalar_ranges_raise_valueerror(function_name, arguments):
    with pytest.raises(ValueError):
        getattr(locality, function_name)(*arguments)


@pytest.mark.parametrize("slot,bad", [
    (0, 0), (0, 4), (0, MAX_INTEGER + 1),
    (1, -1), (1, MAX_INTEGER + 1),
    (2, 2), (2, MAX_INTEGER + 1),
    (3, -1), (3, MAX_INTEGER + 1),
    (4, 0), (4, -1), (4, MAX_INTEGER + 1),
    (5, 0), (5, -1), (5, MAX_INTEGER + 1),
])
def test_growth_scalar_ranges_raise_valueerror(slot, bad):
    arguments = [2, 2, 7, 3, 2, 1]
    arguments[slot] = bad
    with pytest.raises(ValueError):
        locality.growth_report(*arguments)


@pytest.mark.parametrize("bad", BAD_SCALARS)
def test_shape_and_center_elements_require_builtin_ints(bad):
    with pytest.raises(ValueError):
        locality.box_ball_count([bad], [0], 0)
    with pytest.raises(ValueError):
        locality.box_ball_count([3], [bad], 0)


@pytest.mark.parametrize("shape,center", [
    ([], []), ([1, 1, 1, 1], [0, 0, 0, 0]),
    ([3], []), ([3, 3], [0]), ([3], [0, 0]),
    ([0], [0]), ([-1], [0]), ([MAX_INTEGER + 1], [0]),
    ([3], [-1]), ([3], [3]), ([3], [MAX_INTEGER + 1]),
    ([[3]], [0]), ([3], [[0]]),
])
def test_invalid_box_shapes_and_centers(shape, center):
    with pytest.raises(ValueError):
        locality.box_ball_count(shape, center, 0)


@pytest.mark.parametrize("factory", [
    lambda: None, lambda: "3", lambda: b"3", lambda: {3},
    lambda: {0: 3}, lambda: range(3), lambda: iter([3]),
    lambda: (value for value in [3]),
    lambda: ListSubclass([3]), lambda: TupleSubclass([3]),
    lambda: ConversionTrap(),
])
def test_shape_and_center_containers_are_strict_builtins(factory):
    with pytest.raises(ValueError):
        locality.box_ball_count(factory(), [0], 0)
    with pytest.raises(ValueError):
        locality.box_ball_count([4], factory(), 0)


def test_box_site_cap_is_inclusive_and_radius_can_saturate():
    assert locality.MAX_INTEGER == MAX_INTEGER
    assert locality.MAX_BOX_SITES == MAX_BOX_SITES
    assert locality.box_ball_count([16, 16, 16], [8, 8, 8], 0) == 1
    assert locality.box_ball_count([16, 16, 16], [8, 8, 8], MAX_INTEGER) == MAX_BOX_SITES
    assert locality.box_ball_count([MAX_BOX_SITES], [0], MAX_INTEGER) == MAX_BOX_SITES


def test_box_cap_precedes_cartesian_or_coordinate_enumeration(monkeypatch):
    """Trap public stdlib seams by identity, not a guessed private helper name.

    Trapping product catches its eager input pooling even at radius zero.
    Trapping large ranges also bounds an implementation using nested loops.
    Valid shapes are deliberately not constructed while these traps are active.
    """
    original_product = itertools.product
    original_range = builtins.range

    def forbidden_product(*arguments, **keywords):
        raise AssertionError("oversized box reached Cartesian enumeration")

    def guarded_range(*arguments):
        result = original_range(*arguments)
        if len(result) > 16:
            raise AssertionError("oversized box reached coordinate enumeration")
        return result

    # Cover `from itertools import product` (including aliases) as well as
    # `import itertools`; do not depend on arbitrary private function names.
    for name, value in list(vars(locality).items()):
        if value is original_product:
            monkeypatch.setattr(locality, name, forbidden_product)
        elif value is original_range:
            monkeypatch.setattr(locality, name, guarded_range)
    monkeypatch.setattr(itertools, "product", forbidden_product)
    monkeypatch.setattr(locality, "range", guarded_range, raising=False)
    for shape in ([17, 17, 17], [64, 65], [4097], [MAX_INTEGER] * 3):
        with pytest.raises(ValueError):
            locality.box_ball_count(shape, [0] * len(shape), 0)


GROWTH_CASES = [
    (dimension, radius, 31, 1, 1, 1)
    for dimension in (1, 2, 3)
    for radius in (0, 1, 2, 3)
] + [
    (3, 2, 31, 3, 2, 1), (3, 1, 31, 1, 2, 1),
    (1, 3, 4, 1, 1, 1), (1, 2, 9, 0, 1, 5),
]


def test_demo_retains_every_frozen_growth_and_box_slot():
    report = locality.demonstration_report()
    assert type(report) is dict
    assert set(report) == {"module", "growth_cases", "finite_box_controls", "limitations"}
    assert type(report["module"]) is str and report["module"]
    assert report["limitations"]
    assert type(report["growth_cases"]) is list
    assert len(report["growth_cases"]) == 16
    for actual, case in zip(report["growth_cases"], GROWTH_CASES):
        assert_growth_report(actual, case)
    assert {item["status"] for item in report["growth_cases"]} == {
        "excluded_by_ball_count", "not_excluded_by_ball_count",
    }
    assert report["growth_cases"][-1]["signed_excess"] == 0
    assert type(report["finite_box_controls"]) is list
    assert len(report["finite_box_controls"]) == 12
    expected_boxes = []
    for shape, center in (
        ([3, 3, 3], [1, 1, 1]), ([1, 1, 5], [0, 0, 2]), ([2, 2], [0, 0]),
    ):
        for radius in (0, 1, 2, 3):
            expected_boxes.append({
                "shape": shape[:], "center": center[:], "radius": radius,
                "count": len(coordinate_ball(len(shape), radius, shape, center)),
            })
    assert report["finite_box_controls"] == expected_boxes
    for record in report["finite_box_controls"]:
        assert type(record["radius"]) is int and type(record["count"]) is int
        assert all(type(value) is int for value in record["shape"] + record["center"])
    assert [item["count"] for item in expected_boxes[:4]] == [1, 7, 19, 27]
    assert [item["count"] for item in expected_boxes[4:8]] == [1, 3, 5, 5]
    assert [item["count"] for item in expected_boxes[8:]] == [1, 3, 4, 4]
    assert_finite_json(report)


def mutable_container_ids(value):
    if type(value) is dict:
        result = {id(value)}
        for item in value.values():
            result.update(mutable_container_ids(item))
        return result
    if type(value) is list:
        result = {id(value)}
        for item in value:
            result.update(mutable_container_ids(item))
        return result
    return set()


def test_reports_are_detached_across_calls_and_demo_records():
    first = locality.growth_report(2, 2, 31)
    second = locality.growth_report(2, 2, 31)
    snapshot = json.dumps(second, sort_keys=True, allow_nan=False)
    first["dilation"]["numerator"] = 999
    first["scope"]["map_existence_established"] = True
    assert json.dumps(second, sort_keys=True, allow_nan=False) == snapshot
    assert json.dumps(locality.growth_report(2, 2, 31), sort_keys=True, allow_nan=False) == snapshot

    first_demo = locality.demonstration_report()
    second_demo = locality.demonstration_report()
    assert mutable_container_ids(first_demo).isdisjoint(mutable_container_ids(second_demo))
    seen_record_containers = set()
    for record in first_demo["growth_cases"] + first_demo["finite_box_controls"]:
        record_containers = mutable_container_ids(record)
        assert seen_record_containers.isdisjoint(record_containers)
        seen_record_containers.update(record_containers)
    baseline = json.dumps(second_demo, sort_keys=True, allow_nan=False)
    other_growth = json.dumps(first_demo["growth_cases"][1:], sort_keys=True)
    other_boxes = json.dumps(first_demo["finite_box_controls"][1:], sort_keys=True)
    first_demo["growth_cases"][0]["dilation"]["numerator"] = 999
    first_demo["growth_cases"][0]["scope"]["map_existence_established"] = True
    first_demo["finite_box_controls"][0]["shape"][0] = 999
    first_demo["finite_box_controls"][0]["center"][0] = 999
    assert json.dumps(first_demo["growth_cases"][1:], sort_keys=True) == other_growth
    assert json.dumps(first_demo["finite_box_controls"][1:], sort_keys=True) == other_boxes
    assert json.dumps(second_demo, sort_keys=True, allow_nan=False) == baseline
    assert json.dumps(locality.demonstration_report(), sort_keys=True, allow_nan=False) == baseline


def test_counting_pass_does_not_establish_c3_to_c4_map_existence():
    """A finite structural counterexample, not an embedding/permutation scan."""
    triangle = cycle_graph(3)
    square = cycle_graph(4)
    # Radius zero and one cover all distinct source balls; larger radii retain
    # the whole triangle and cannot reduce target capacity.
    for source_vertex in triangle:
        for radius in (0, 1, 2, 3):
            source_count = len(graph_ball(triangle, source_vertex, radius))
            assert source_count <= locality.ring_ball_count(4, radius)
    # C4 is bipartite. With multiplicity one, the adjacent vertices of C3
    # require distinct images. All three pairwise edges cannot cross only two
    # color classes: after two edges the first/third colors coincide.
    colors = {0: 0, 1: 1, 2: 0, 3: 1}
    assert all(colors[v] != colors[w] for v in square for w in square[v])
    assert triangle == {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
    first_color = 0
    second_color = 1 - first_color
    third_color = 1 - second_color
    assert third_color == first_color
    # No arbitrary-graph API is presumed; a public non-exclusion report must
    # still explicitly deny establishment of an actual map or physical claim.
    diagnostic = locality.growth_report(1, 1, 4)
    assert diagnostic["status"] == "not_excluded_by_ball_count"
    assert_scope(diagnostic["scope"])


def run_isolated_demo(tmp_path, *arguments):
    script = Path(__file__).resolve().parents[1] / "scripts" / "demo_substrate_locality_dimension.py"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    assert list(tmp_path.iterdir()) == []
    completed = subprocess.run(
        [sys.executable, "-B", str(script)] + list(arguments),
        cwd=str(tmp_path), env=environment, capture_output=True, text=True,
        timeout=15, check=False,
    )
    assert list(tmp_path.iterdir()) == []
    return completed


def reject_json_constant(token):
    raise AssertionError("nonfinite JSON token: " + token)


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, "duplicate JSON object key: " + key
        result[key] = value
    return result


def test_cli_json_is_one_strict_detached_report_from_isolated_cwd(tmp_path):
    completed = run_isolated_demo(tmp_path, "--json")
    assert completed.returncode == 0
    assert completed.stderr == ""
    # json.loads rejects extra JSON documents or any trailing diagnostic text.
    report = json.loads(
        completed.stdout, parse_constant=reject_json_constant,
        object_pairs_hook=unique_json_object,
    )
    assert_finite_json(report)
    assert report == locality.demonstration_report()
    assert len(report["growth_cases"]) == 16
    assert len(report["finite_box_controls"]) == 12


def test_cli_default_is_text_with_conditional_and_nonphysical_scope(tmp_path):
    completed = run_isolated_demo(tmp_path)
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout.strip()
    with pytest.raises(json.JSONDecodeError):
        json.loads(completed.stdout)
    # Exact wording is not frozen; both prose and printed scope keys qualify.
    text = completed.stdout.lower().replace("_", " ").replace("-", " ")
    assert "conditional" in text
    assert "physical" in text and "dimension" in text
    assert "empirical" in text
    assert any(word in text for word in ("not", "false", "no "))


def test_cli_unknown_arguments_are_rejected_without_creating_files(tmp_path):
    completed = run_isolated_demo(tmp_path, "--unrecognized-locality-option")
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "usage:" in completed.stderr.lower()
    assert "unrecognized" in completed.stderr.lower()
