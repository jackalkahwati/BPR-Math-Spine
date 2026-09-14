"""Contract-only, bounded exact checks for finite-range three-body geometry."""

from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
import importlib.util
from io import StringIO
from itertools import combinations
import json
from math import comb
from pathlib import Path
import unittest
from unittest import mock


_CHECKER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "check_cubic_finite_range_three_body.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "finite_range_three_body_checker_under_test", _CHECKER_PATH
)
_CHECKER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CHECKER)

O = (0, 0, 0)
X = (1, 0, 0)
Y = (0, 1, 0)
ONE = Fraction(1)
HALF = Fraction(1, 2)

# Literal targets, transcribed from section 5; never obtained from the checker.
FIXTURES = (
    ("empty", (), ONE, 0),
    ("single", (O,), ONE, 0),
    ("coincident_pair", (O, O), ONE, 0),
    ("separated_pair", (O, (2, 0, 0)), ONE, 0),
    ("coincident_triple", (O, O, O), ONE, 1),
    ("doubled_near", (O, O, X), HALF, 1),
    ("doubled_boundary", (O, O, X), ONE, 1),
    ("doubled_far", (O, O, (2, 0, 0)), ONE, 0),
    ("distinct_near", (O, X, Y), HALF, 1),
    ("diagonal_far", (O, X, Y), ONE, 0),
    ("anchor_trap", ((-1, 0, 0), O, X), ONE, 0),
    ("six_near", (O, O, O, X, X, Y), HALF, 20),
    ("split_clusters", (O, O, O, (3, 0, 0), (3, 0, 0), (3, 0, 0)), ONE, 2),
)
LIMITATIONS = [
    "Range and strength are supplied fixed physical parameters, not fitted predictions.",
    "Triples count particle labels, including coincident positions, using all three distances.",
    "Two-particle scattering is unchanged, but bounded strength does not cure continuum collapse.",
    "Multiplier convergence is not convergence of the tuned many-body dynamics.",
    "Exact geometry checks are not empirical validation or a theory of everything.",
]


def occupation_oracle(positions, spacing):
    """Equation (3): site multiplicities, not enumeration of label triples."""
    occupations = Counter(positions)
    sites = sorted(occupations)
    distances = {}
    for left, right in combinations(sites, 2):
        distances[left, right] = spacing * spacing * sum(
            (left[axis] - right[axis]) ** 2 for axis in range(3)
        )

    def close(left, right):
        return distances[tuple(sorted((left, right)))] <= 1

    onsite = sum(comb(number, 3) for number in occupations.values())
    doubled = sum(
        comb(occupations[left], 2) * occupations[right]
        for left in sites
        for right in sites
        if left != right and close(left, right)
    )
    distinct = sum(
        occupations[left] * occupations[middle] * occupations[right]
        for left, middle, right in combinations(sites, 3)
        if close(left, middle) and close(left, right) and close(middle, right)
    )
    return onsite, onsite + doubled + distinct


def rational(value):
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def expected_case(label, positions, spacing, target):
    onsite, _ = occupation_oracle(positions, spacing)
    return {
        "id": label,
        "positions": [list(position) for position in positions],
        "spacing": rational(spacing),
        "particle_count": len(positions),
        "triple_count": target,
        "onsite_triple_count": onsite,
        "triple_upper_bound": comb(len(positions), 3),
        "repulsion_energy": rational(target),
    }


def expected_report():
    return {
        "schema_version": 1,
        "status": "conditional_fixed_range_three_body_diagnostics",
        "empirical_validation": False,
        "range": rational(1),
        "strength": rational(1),
        "cases": [expected_case(*fixture) for fixture in FIXTURES],
        "limitations": list(LIMITATIONS),
    }


class TupleSubclass(tuple):
    pass


class IntSubclass(int):
    pass


class FractionSubclass(Fraction):
    pass


class StringSubclass(str):
    pass


class IterationTripwire:
    def __iter__(self):
        raise AssertionError("invalid input was iterated")

    def __len__(self):
        raise AssertionError("invalid input length was inspected")

    def __getitem__(self, key):
        raise AssertionError("invalid input was indexed")


class CubicFiniteRangeThreeBodyTests(unittest.TestCase):
    def assert_rejected(self, positions, spacing):
        with mock.patch.object(
            _CHECKER, "_squared_distance", side_effect=AssertionError("distance before admission")
        ) as distance:
            for function, arguments in (
                (_CHECKER._admit, (positions, spacing)),
                (_CHECKER._count_triples, (positions, spacing)),
                (_CHECKER._case_report, ("empty", positions, spacing)),
            ):
                with self.subTest(function=function.__name__):
                    with self.assertRaises(ValueError):
                        function(*arguments)
            distance.assert_not_called()

    def assert_native_json(self, value):
        if type(value) is dict:
            for key, child in value.items():
                self.assertIs(type(key), str)
                self.assert_native_json(child)
        elif type(value) is list:
            for child in value:
                self.assert_native_json(child)
        else:
            self.assertIn(type(value), (str, int, bool, type(None)))

    def assert_canonical_rational(self, value, expected):
        self.assertIs(type(value), dict)
        self.assertEqual(set(value), {"numerator", "denominator"})
        self.assertIs(type(value["numerator"]), int)
        self.assertIs(type(value["denominator"]), int)
        self.assertGreater(value["denominator"], 0)
        self.assertEqual(value, rational(expected))

    def container_ids(self, value):
        if type(value) not in (dict, list):
            return set()
        found = {id(value)}
        children = value.values() if type(value) is dict else value
        for child in children:
            nested = self.container_ids(child)
            self.assertTrue(found.isdisjoint(nested), "aliased mutable JSON containers")
            found.update(nested)
        return found

    def test_fixtures_match_all_thirteen_literal_rows(self):
        fixtures = _CHECKER._fixtures()
        self.assertEqual(len(fixtures), 13)
        self.assertEqual(tuple(fixtures), tuple(row[:3] for row in FIXTURES))
        for fixture in fixtures:
            self.assertIs(type(fixture), tuple)
            label, positions, spacing = fixture
            self.assertIs(type(label), str)
            self.assertIs(type(positions), tuple)
            self.assertIs(type(spacing), Fraction)
            for position in positions:
                self.assertIs(type(position), tuple)
                self.assertEqual(len(position), 3)
                for coordinate in position:
                    self.assertIs(type(coordinate), int)

    def test_literal_targets_and_independent_occupation_oracle(self):
        for label, positions, spacing, target in FIXTURES:
            with self.subTest(label=label):
                onsite, oracle = occupation_oracle(positions, spacing)
                self.assertEqual(oracle, target)
                self.assertIsNone(_CHECKER._admit(positions, spacing))
                result = _CHECKER._count_triples(positions, spacing)
                self.assertIs(type(result), int)
                self.assertEqual(result, target)
                case = _CHECKER._case_report(label, positions, spacing)
                self.assertEqual(case, expected_case(label, positions, spacing, target))
                self.assertEqual(case["onsite_triple_count"], onsite)

    def test_one_reversal_and_one_bounded_translation_per_fixture(self):
        shift = (1, -1, 1)
        for label, positions, spacing, target in FIXTURES:
            translated = tuple(
                tuple(position[axis] + shift[axis] for axis in range(3))
                for position in positions
            )
            for variant in (positions[::-1], translated):
                with self.subTest(label=label, positions=variant):
                    self.assertTrue(all(-4 <= c <= 4 for point in variant for c in point))
                    self.assertEqual(occupation_oracle(variant, spacing)[1], target)
                    self.assertEqual(_CHECKER._count_triples(variant, spacing), target)
                    self.assertEqual(
                        _CHECKER._case_report(label, variant, spacing),
                        expected_case(label, variant, spacing, target),
                    )

    def test_onsite_lower_bound_upper_bound_and_small_number_zero(self):
        geometries = [(row[1], row[2]) for row in FIXTURES]
        geometries.extend(((O,) * number, HALF) for number in range(7))
        geometries.extend((points, ONE) for points in (
            ((-4, -4, -4),),
            ((-4, -4, -4), (4, 4, 4)),
        ))
        for positions, spacing in geometries:
            with self.subTest(positions=positions, spacing=spacing):
                onsite, oracle = occupation_oracle(positions, spacing)
                result = _CHECKER._count_triples(positions, spacing)
                self.assertEqual(result, oracle)
                self.assertLessEqual(onsite, result)
                self.assertLessEqual(result, comb(len(positions), 3))
                if len(positions) <= 2:
                    self.assertEqual(result, 0)

    def test_squared_distance_is_exact_physical_euclidean_distance(self):
        for left, right, spacing, target in (
            (O, O, HALF, Fraction(0)),
            (O, X, ONE, Fraction(1)),
            (O, X, HALF, Fraction(1, 4)),
            (X, Y, ONE, Fraction(2)),
            (X, Y, HALF, Fraction(1, 2)),
            ((-4, -4, -4), (4, 4, 4), HALF, Fraction(48)),
        ):
            with self.subTest(left=left, right=right, spacing=spacing):
                self.assertIsNone(_CHECKER._admit((left, right), spacing))
                result = _CHECKER._squared_distance(left, right, spacing)
                self.assertIs(type(result), Fraction)
                self.assertEqual(result, target)
                self.assertEqual(_CHECKER._squared_distance(right, left, spacing), target)

    def test_inclusive_boundary_diagonal_and_anchor_traps(self):
        for positions, spacing, target in (
            ((O, O, X), ONE, 1),
            ((O, O, (2, 0, 0)), HALF, 1),
            ((O, X, Y), ONE, 0),
            (((-1, 0, 0), O, X), ONE, 0),
            ((O, X, Y), HALF, 1),
            ((O, X, (2, 0, 0)), HALF, 1),
            (((-4, 0, 0), (-4, 0, 0), (4, 0, 0)), ONE, 0),
        ):
            with self.subTest(positions=positions, spacing=spacing):
                self.assertEqual(occupation_oracle(positions, spacing)[1], target)
                self.assertEqual(_CHECKER._count_triples(positions, spacing), target)

    def test_outer_container_requires_native_tuple_without_iteration(self):
        for positions in ([], [O], TupleSubclass((O,)), IterationTripwire(), None, ""):
            with self.subTest(type=type(positions).__name__):
                self.assert_rejected(positions, ONE)

    def test_coordinate_shapes_and_types_reject_before_arithmetic(self):
        bad_points = (
            [0, 0, 0], TupleSubclass(O), (), (0, 0), (0, 0, 0, 0),
            (True, 0, 0), (0, False, 0), (0, 0, IntSubclass(0)),
            (0.0, 0, 0), (Fraction(0), 0, 0), ("0", 0, 0),
            (None, 0, 0), (5, 0, 0), (0, -5, 0), (0, 0, 5),
            IterationTripwire(),
        )
        for point in bad_points:
            with self.subTest(point=point):
                self.assert_rejected((O, X, point), ONE)

    def test_spacing_requires_exact_fraction_and_supported_value(self):
        for spacing in (
            1, True, 1.0, 0.5, "1/2", None, FractionSubclass(1),
            Fraction(0), Fraction(-1), Fraction(1, 3), Fraction(2),
        ):
            with self.subTest(spacing=spacing):
                self.assert_rejected((O, O, X), spacing)
        for spacing in (Fraction(2, 2), Fraction(2, 4)):
            self.assertIsNone(_CHECKER._admit((O, O, X), spacing))

    def test_lowered_live_caps_reject_before_iteration_or_distances(self):
        self.assertEqual(
            (_CHECKER.MAX_PARTICLES, _CHECKER.MAX_PAIRS, _CHECKER.MAX_TRIPLES),
            (6, 15, 20),
        )
        for cap, value in (("MAX_PARTICLES", 2), ("MAX_PAIRS", 2), ("MAX_TRIPLES", 0)):
            with self.subTest(cap=cap), mock.patch.object(_CHECKER, cap, value):
                self.assert_rejected((O, O, X), ONE)
                # Rejection must not attempt to traverse these invalid points.
                self.assert_rejected((IterationTripwire(),) * 3, ONE)

    def test_zero_workloads_and_raised_caps_do_not_expand_fixed_domain(self):
        with mock.patch.multiple(_CHECKER, MAX_PARTICLES=0, MAX_PAIRS=0, MAX_TRIPLES=0):
            self.assertIsNone(_CHECKER._admit((), ONE))
            self.assertEqual(_CHECKER._count_triples((), ONE), 0)
            self.assert_rejected((O,), ONE)
        with mock.patch.multiple(_CHECKER, MAX_PAIRS=0, MAX_TRIPLES=0):
            self.assertIsNone(_CHECKER._admit((O,), ONE))
            self.assertEqual(_CHECKER._count_triples((O,), ONE), 0)
            self.assert_rejected((O, O), ONE)
        with mock.patch.object(_CHECKER, "MAX_TRIPLES", 0):
            self.assertIsNone(_CHECKER._admit((O, O), ONE))
            self.assertEqual(_CHECKER._count_triples((O, O), ONE), 0)
        with mock.patch.multiple(_CHECKER, MAX_PARTICLES=100, MAX_PAIRS=100, MAX_TRIPLES=100):
            # Oversized input is admission-only: no seven-label configuration is evaluated.
            self.assert_rejected((O,) * 7, ONE)

    def test_each_unordered_label_pair_distance_is_computed_once(self):
        for positions in (
            (O, X, Y, (0, 0, 1), (-1, 0, 0), (0, -1, 0)),
            (O, O, O, X, X, Y),
        ):
            with self.subTest(positions=positions):
                with mock.patch.object(
                    _CHECKER, "_squared_distance", wraps=_CHECKER._squared_distance
                ) as distance:
                    self.assertEqual(_CHECKER._count_triples(positions, HALF), 20)
                self.assertEqual(distance.call_count, 15)
                observed = Counter()
                for call in distance.call_args_list:
                    left, right, spacing = call.args
                    self.assertIs(type(spacing), Fraction)
                    self.assertEqual(spacing, HALF)
                    observed[tuple(sorted((left, right)))] += 1
                expected = Counter(tuple(sorted(pair)) for pair in combinations(positions, 2))
                self.assertEqual(observed, expected)

    def test_rational_encoding_is_canonical_and_strictly_typed(self):
        for value in (0, 1, -3, Fraction(0, 7), Fraction(4, 2), Fraction(6, -8)):
            with self.subTest(value=value):
                self.assert_canonical_rational(_CHECKER._rational(value), value)
        for value in (True, False, IntSubclass(1), FractionSubclass(1), 1.0, "1", None):
            with self.subTest(invalid=value):
                with self.assertRaises(ValueError):
                    _CHECKER._rational(value)
        first = _CHECKER._rational(1)
        second = _CHECKER._rational(1)
        self.assertIsNot(first, second)
        first["numerator"] = 99
        self.assert_canonical_rational(second, 1)

    def test_case_labels_are_native_frozen_ids_and_admission_comes_first(self):
        for label in ("unknown", "", 1, True, None, StringSubclass("empty")):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    _CHECKER._case_report(label, (), ONE)
        failure = ValueError("admission must precede label validation")
        with mock.patch.object(_CHECKER, "_admit", side_effect=failure) as admit:
            with self.assertRaises(ValueError) as caught:
                _CHECKER._case_report("unknown", (), ONE)
            self.assertIs(caught.exception, failure)
            admit.assert_called_once_with((), ONE)
        self.assertEqual(_CHECKER._case_report("empty", (O, O, O), ONE)["triple_count"], 1)

    def test_complete_report_schema_uses_native_json_and_exact_rationals(self):
        report = _CHECKER.demonstration_report()
        self.assert_native_json(report)
        self.assertEqual(report, expected_report())
        self.assertIs(type(report["schema_version"]), int)
        self.assertIs(report["empirical_validation"], False)
        self.assert_canonical_rational(report["range"], 1)
        self.assert_canonical_rational(report["strength"], 1)
        for case, (label, positions, spacing, target) in zip(report["cases"], FIXTURES):
            self.assertIs(type(case["id"]), str)
            for key in ("particle_count", "triple_count", "onsite_triple_count", "triple_upper_bound"):
                self.assertIs(type(case[key]), int)
            self.assert_canonical_rational(case["spacing"], spacing)
            self.assert_canonical_rational(case["repulsion_energy"], target)
            for position in case["positions"]:
                self.assertIs(type(position), list)
                self.assertEqual(len(position), 3)
                for coordinate in position:
                    self.assertIs(type(coordinate), int)
        encoded = json.dumps(report, allow_nan=False)
        self.assertEqual(json.loads(encoded), expected_report())

    def test_reports_and_cases_have_fresh_detached_mutable_containers(self):
        first = _CHECKER.demonstration_report()
        second = _CHECKER.demonstration_report()
        self.assertTrue(self.container_ids(first).isdisjoint(self.container_ids(second)))
        case_one = _CHECKER._case_report("coincident_triple", (O, O, O), ONE)
        case_two = _CHECKER._case_report("coincident_triple", (O, O, O), ONE)
        self.assertTrue(self.container_ids(case_one).isdisjoint(self.container_ids(case_two)))
        self.assertTrue(self.container_ids(case_one).isdisjoint(self.container_ids(first)))
        first["range"]["numerator"] = 99
        first["cases"][4]["positions"][0][0] = 99
        first["cases"][4]["repulsion_energy"]["numerator"] = 99
        first["limitations"].clear()
        case_one["positions"][0][0] = 99
        case_one["spacing"]["numerator"] = 99
        self.assertEqual(second, expected_report())
        self.assertEqual(case_two, expected_case("coincident_triple", (O, O, O), ONE, 1))
        self.assertEqual(_CHECKER.demonstration_report(), expected_report())
        self.assertEqual(tuple(_CHECKER._fixtures()), tuple(row[:3] for row in FIXTURES))

    def test_cli_text_uses_mocked_report_and_prints_all_cases_and_limitations(self):
        output = StringIO()
        with mock.patch.object(_CHECKER, "demonstration_report", return_value=expected_report()) as report:
            with redirect_stdout(output):
                result = _CHECKER.main([])
        self.assertIs(type(result), int)
        self.assertEqual(result, 0)
        report.assert_called_once_with()
        text = output.getvalue()
        self.assertTrue(text.startswith("Finite-range three-body repulsion: exact geometry checks"))
        for label, _, _, target in FIXTURES:
            matching = [line for line in text.splitlines() if label in line]
            self.assertTrue(matching, label)
            self.assertTrue(any(str(target) in line for line in matching), label)
        offsets = [text.index(limitation) for limitation in LIMITATIONS]
        self.assertEqual(offsets, sorted(offsets))
        self.assertGreater(offsets[0], text.index("split_clusters"))

    def test_cli_json_uses_mocked_report_and_strict_serialization(self):
        output = StringIO()
        payload = expected_report()
        with mock.patch.object(_CHECKER, "demonstration_report", return_value=payload) as report:
            with redirect_stdout(output):
                result = _CHECKER.main(["--json"])
        self.assertIs(type(result), int)
        self.assertEqual(result, 0)
        report.assert_called_once_with()
        self.assertEqual(json.loads(output.getvalue()), payload)

    def test_cli_help_and_invalid_arguments_parse_before_report_generation(self):
        for argv, exit_code in (
            (["--help"], 0), (["-h"], 0), (["--j"], 2), (["--js"], 2),
            (["--jso"], 2), (["--JSON"], 2), (["--unknown"], 2),
            (["--json=true"], 2), (["unexpected"], 2),
        ):
            with self.subTest(argv=argv):
                with mock.patch.object(
                    _CHECKER, "demonstration_report", side_effect=AssertionError("report before parse")
                ) as report:
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        with self.assertRaises(SystemExit) as caught:
                            _CHECKER.main(argv)
                    self.assertEqual(caught.exception.code, exit_code)
                    report.assert_not_called()

    def test_cli_json_rejects_mocked_nonfinite_values(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                payload = expected_report()
                payload["cases"][0]["repulsion_energy"]["numerator"] = value
                output = StringIO()
                with mock.patch.object(_CHECKER, "demonstration_report", return_value=payload) as report:
                    with redirect_stdout(output), redirect_stderr(StringIO()):
                        with self.assertRaises(ValueError):
                            _CHECKER.main(["--json"])
                    report.assert_called_once_with()
                self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
