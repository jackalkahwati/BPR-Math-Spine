"""Independent, bounded scalar oracles for the frozen section 6 contract.

No production code is imported until unittest runs. These finite controls do
not prove the all-occupation bound or extract a fixed particle-number sector.
"""

import importlib.util
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from itertools import combinations, product
from math import gcd, isqrt
from pathlib import Path
import sys
import unittest
from unittest import mock


LIMITATIONS = [
    "The onsite three-body penalty is a supplied assumption and leaves N<=2 unchanged.",
    "Fixed-spacing finite-volume boundedness is not uniform continuum stability.",
    "The continuum obstruction is proved by a trial state with zero triple occupancy.",
    "The fixed particle number exists analytically; no large block or sector is constructed.",
    "Exact scalar checks are not empirical validation or a theory of everything.",
]
ONSITE_ENERGIES = {1: (0, 0, -1, -2, -2, 0, 5), 2: (0, 0, -1, -1, 2, 10, 25)}
BLOCK_ENERGIES = {1: Fraction(2378, 1681), 2: Fraction(9424, 1681),
                  128: Fraction(-6815744, 1681)}


def rational(value):
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def unpack(value):
    return Fraction(value["numerator"], value["denominator"])


def local_oracle():
    return {
        "x": rational(Fraction(1, 4)),
        "normalization": rational(Fraction(41, 32)),
        "mean_number": rational(Fraction(10, 41)),
        "annihilation": rational(Fraction(20, 41)),
        "pair_mean": rational(Fraction(1, 41)),
        "triple_mean": rational(0),
        "kinetic_difference": rational(Fraction(10, 1681)),
        "probabilities": [rational(Fraction(weight, 41)) for weight in (32, 8, 1)],
    }


def onsite_oracle(n, w):
    # At most six labels: fifteen pairs and twenty triples, never Fock states.
    pairs = sum(1 for _ in combinations(range(n), 2))
    triples = sum(1 for _ in combinations(range(n), 3))
    bound = {1: Fraction(2, 3), 2: Fraction(25, 48)}[w]
    return {
        "n": n, "w": w, "pair_count": pairs, "triple_count": triples,
        "energy": rational(ONSITE_ENERGIES[w][n]),
        "bound_coefficient": rational(bound),
        "square_remainder": rational(
            Fraction(n * (2 * w * (n - 1) - (3 + w)) ** 2, 24 * w)
        ),
    }


def block_counts(b):
    if b == 128:
        # Count slices and bonds per axis, not sites or particle sectors.
        face = 128 * 128
        return 128 * face, 3 * (127 * face), 6 * face
    if b not in (1, 2):
        raise ValueError("Only the two tiny blocks may be enumerated")
    sites = tuple(product(range(b), repeat=3))  # At most eight sites.
    edges = sum(
        sum(abs(x - y) for x, y in zip(left, right)) == 1
        for left, right in combinations(sites, 2)  # At most 28 pairs.
    )
    crossing = sum(
        int(site[axis] == 0) + int(site[axis] == b - 1)
        for site in sites for axis in range(3)
    )
    return len(sites), edges, crossing


def block_oracle(b):
    sites, edges, crossing = block_counts(b)
    mean, alpha = Fraction(10, 41), Fraction(20, 41)
    # Independent edge-form kinetic energy, including missing neighbors.
    kinetic = 2 * edges * (mean - alpha * alpha) + crossing * mean
    return {
        "b": b, "sites": sites, "internal_edges": edges,
        "crossing_edges": crossing, "max_number_support": 2 * sites,
        "mean_number": rational(sites * mean),
        "shifted_kinetic": rational(kinetic),
        "pair_energy": rational(Fraction(-2 * sites, 41)),
        "triple_energy": rational(0),
        "reference_energy": rational(BLOCK_ENERGIES[b]),
    }


def report_oracle():
    return {
        "schema_version": 1,
        "status": "conditional_onsite_three_body_stability_diagnostics",
        "empirical_validation": False,
        "onsite_cases": [onsite_oracle(n, w) for w in (1, 2) for n in range(7)],
        "local_state": local_oracle(),
        "block_cases": [block_oracle(b) for b in (1, 2, 128)],
        "limitations": list(LIMITATIONS),
    }


class IntSubclass(int):
    pass


class FractionSubclass(Fraction):
    pass


class CubicThreeBodyStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "scripts" / "check_cubic_three_body_stability.py"
        spec = importlib.util.spec_from_file_location("three_body_scalar_contract", path)
        cls.checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.checker)

    def assert_native_equal(self, actual, expected):
        self.assertIs(type(actual), type(expected))
        if type(expected) is dict:
            self.assertEqual(set(actual), set(expected))
            if set(expected) == {"numerator", "denominator"}:
                self.assertIs(type(actual["numerator"]), int)
                self.assertIs(type(actual["denominator"]), int)
                self.assertGreater(actual["denominator"], 0)
                self.assertEqual(gcd(actual["numerator"], actual["denominator"]), 1)
            for key in expected:
                self.assert_native_equal(actual[key], expected[key])
        elif type(expected) is list:
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_native_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_onsite_labelled_pair_and_triple_counts(self):
        for w in (1, 2):
            for n in range(7):
                with self.subTest(n=n, w=w):
                    row = self.checker._onsite_case(n, w)
                    pairs = tuple(combinations(range(n), 2))
                    triples = tuple(combinations(range(n), 3))
                    self.assertLessEqual(len(triples), 20)
                    self.assertEqual(row["pair_count"], len(pairs))
                    self.assertEqual(row["triple_count"], len(triples))
                    self.assertEqual(unpack(row["energy"]), -len(pairs) + w * len(triples))
                    if n <= 2:
                        self.assertEqual(row["triple_count"], 0)

    def test_all_fourteen_onsite_fields_and_literal_energies(self):
        for w in (1, 2):
            for n in range(7):
                with self.subTest(n=n, w=w):
                    self.assert_native_equal(self.checker._onsite_case(n, w), onsite_oracle(n, w))

    def test_full_square_completion_including_vacuum(self):
        for w in (1, 2):
            for n in range(7):
                row = self.checker._onsite_case(n, w)
                energy, bound, remainder = (
                    unpack(row[key]) for key in ("energy", "bound_coefficient", "square_remainder")
                )
                full_square = Fraction(n * (2 * w * (n - 1) - (3 + w)) ** 2, 24 * w)
                self.assertEqual(bound, Fraction((3 + w) ** 2, 24 * w))
                self.assertEqual(energy + n * bound, full_square)
                self.assertEqual(remainder, full_square)
                self.assertGreaterEqual(remainder, 0)
                if n == 0:
                    self.assertEqual((energy, remainder), (0, 0))
                else:
                    center = Fraction(3 + w, 2 * w)
                    self.assertEqual(energy / n, Fraction(w, 6) * (n - 1 - center) ** 2 - bound)

    def test_local_fields_and_three_probability_moments(self):
        actual = self.checker._local_moments()
        self.assert_native_equal(actual, local_oracle())
        x = Fraction(1, 4)
        weights = (Fraction(1), x, x * x / 2)
        normalization = sum(weights)
        probabilities = tuple(weight / normalization for weight in weights)
        self.assertEqual(normalization, unpack(actual["normalization"]))
        self.assertEqual(sum(probabilities), 1)
        self.assertEqual(probabilities, tuple(unpack(p) for p in actual["probabilities"]))
        self.assertEqual(sum(n * p for n, p in enumerate(probabilities)), unpack(actual["mean_number"]))
        self.assertEqual(probabilities[2], unpack(actual["pair_mean"]))
        self.assertEqual(unpack(actual["triple_mean"]), 0)

    def test_ladder_weighted_products_have_exact_integer_roots(self):
        # Ordinary normalized occupation inner product: ladder factors 1 and 2.
        weights = (32, 8, 1)
        products = (weights[0] * weights[1], 2 * weights[1] * weights[2])
        self.assertEqual(products, (256, 16))
        roots = tuple(isqrt(value) for value in products)
        for root, value in zip(roots, products):
            self.assertEqual(root * root, value)
        alpha = Fraction(sum(roots), 41)
        mean = Fraction(weights[1] + 2 * weights[2], 41)
        self.assertEqual(alpha, Fraction(20, 41))
        actual = self.checker._local_moments()
        self.assertEqual(unpack(actual["annihilation"]), alpha)
        self.assertEqual(mean - alpha * alpha, Fraction(10, 1681))
        self.assertEqual(unpack(actual["kinetic_difference"]), mean - alpha * alpha)

    def test_small_block_counts_from_unordered_site_pairs(self):
        for b, expected in ((1, (1, 0, 6)), (2, (8, 12, 24))):
            self.assertEqual(block_counts(b), expected)
            row = self.checker._block_case(b)
            self.assertEqual(tuple(row[key] for key in ("sites", "internal_edges", "crossing_edges")), expected)
            self.assertEqual(2 * row["internal_edges"] + row["crossing_edges"], 6 * row["sites"])

    def test_large_block_uses_only_factored_scalar_counts(self):
        row = self.checker._block_case(128)
        self.assertEqual(block_counts(128), (2097152, 6242304, 98304))
        for key, expected in (("sites", 2097152), ("internal_edges", 6242304),
                              ("crossing_edges", 98304), ("max_number_support", 4194304)):
            self.assertIs(type(row[key]), int)
            self.assertEqual(row[key], expected)

    def test_all_block_fields_literal_energies_and_edge_kinetic(self):
        for b in (1, 2, 128):
            row = self.checker._block_case(b)
            self.assert_native_equal(row, block_oracle(b))
            total = sum(unpack(row[key]) for key in ("shifted_kinetic", "pair_energy", "triple_energy"))
            self.assertEqual(total, BLOCK_ENERGIES[b])
            self.assertEqual(total, unpack(row["reference_energy"]))
            self.assertEqual(total < 0, b == 128)

    def test_report_exact_schema_order_native_values_and_limitations(self):
        self.assert_native_equal(self.checker.demonstration_report(), report_oracle())

    def test_every_report_container_is_detached_between_calls(self):
        first, second = self.checker.demonstration_report(), self.checker.demonstration_report()

        def container_ids(value):
            if type(value) not in (dict, list):
                return set()
            children = value.values() if type(value) is dict else value
            result = {id(value)}
            for child in children:
                result.update(container_ids(child))
            return result

        self.assertTrue(container_ids(first).isdisjoint(container_ids(second)))
        first["onsite_cases"][0]["energy"].clear()
        first["local_state"]["probabilities"][0].clear()
        first["block_cases"][0]["mean_number"].clear()
        first["limitations"].clear()
        self.assert_native_equal(second, report_oracle())
        self.assert_native_equal(self.checker.demonstration_report(), report_oracle())

    def test_rational_exact_inputs_reduction_and_freshness(self):
        for value in (0, 1, -3, Fraction(0, 7), Fraction(6, 3), Fraction(6, -8)):
            with self.subTest(value=value):
                first, second = self.checker._rational(value), self.checker._rational(value)
                self.assert_native_equal(first, rational(value))
                self.assertIsNot(first, second)
                first.clear()
                self.assert_native_equal(second, rational(value))

    def test_rational_rejects_nonexact_types_and_subclasses(self):
        for value in (True, False, IntSubclass(1), FractionSubclass(1, 2),
                      "1", None, [], {}, complex(1), float("1")):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    self.checker._rational(value)

    def test_onsite_strict_inputs_and_frozen_label_limits(self):
        bad_n = (-1, 7, True, False, IntSubclass(1), Fraction(1), FractionSubclass(1),
                 float("1"), complex(1), "1", None, [], {})
        bad_w = (0, 3, True, False, IntSubclass(1), Fraction(1), FractionSubclass(1),
                 float("1"), complex(1), "1", None, [], {})
        for n in bad_n:
            with self.subTest(n=repr(n)), self.assertRaises(ValueError):
                self.checker._onsite_case(n, 1)
        for w in bad_w:
            with self.subTest(w=repr(w)), self.assertRaises(ValueError):
                self.checker._onsite_case(1, w)
        with mock.patch.object(self.checker, "MAX_OCCUPATION", 7):
            with self.assertRaises(ValueError):
                self.checker._onsite_case(7, 1)

    def test_block_strict_inputs_and_frozen_label_set(self):
        for b in (-1, 0, 3, 127, 129, True, False, IntSubclass(2), Fraction(2),
                  FractionSubclass(2), float("2"), complex(2), "2", None, [], {}):
            with self.subTest(b=repr(b)), self.assertRaises(ValueError):
                self.checker._block_case(b)
        with mock.patch.object(self.checker, "MAX_BLOCK_SIDE", 129):
            with self.assertRaises(ValueError):
                self.checker._block_case(129)

    def test_admission_precedes_even_input_validation(self):
        sentinel = ValueError("admission must be first")
        for name, args in (("_onsite_case", (None, None)), ("_local_moments", ()),
                           ("_block_case", (None,)), ("demonstration_report", ())):
            with self.subTest(name=name):
                with mock.patch.object(self.checker, "_admit", side_effect=sentinel) as admit:
                    with self.assertRaises(ValueError) as caught:
                        getattr(self.checker, name)(*args)
                    self.assertIs(caught.exception, sentinel)
                    admit.assert_called_once_with()

    def test_live_caps_fail_before_scalar_or_report_construction(self):
        for cap, required in (("MAX_OCCUPATION", 6), ("MAX_BLOCK_SIDE", 128), ("MAX_LOCAL_ENTRIES", 3)):
            self.assertIs(type(getattr(self.checker, cap)), int)
            self.assertEqual(getattr(self.checker, cap), required)
            with self.subTest(cap=cap), mock.patch.object(self.checker, cap, required - 1):
                with self.assertRaises(ValueError):
                    self.checker._admit()
                with mock.patch.object(self.checker, "_rational", side_effect=AssertionError("construction before admission")) as encode:
                    for name, args in (("_onsite_case", (0, 1)), ("_local_moments", ()), ("_block_case", (1,))):
                        with self.assertRaises(ValueError):
                            getattr(self.checker, name)(*args)
                    encode.assert_not_called()
                with mock.patch.object(self.checker, "_onsite_case") as onsite, \
                     mock.patch.object(self.checker, "_local_moments") as local, \
                     mock.patch.object(self.checker, "_block_case") as block:
                    with self.assertRaises(ValueError):
                        self.checker.demonstration_report()
                    for generator in (onsite, local, block):
                        generator.assert_not_called()
            with mock.patch.object(self.checker, cap, required + 1):
                self.checker._admit()

    def test_cli_default_text_prints_every_mocked_case_and_limitation(self):
        report = report_oracle()
        # Unique mocked energies test row coverage without freezing whitespace.
        markers = []
        for base, rows in ((10001, report["onsite_cases"]), (20001, report["block_cases"])):
            key = "energy" if base == 10001 else "reference_energy"
            for index, row in enumerate(rows):
                marker = base + index
                row[key] = rational(marker)
                markers.append(str(marker))
        for argv in ([], None):
            output = io.StringIO()
            with mock.patch.object(self.checker, "demonstration_report", return_value=report) as generate, \
                 mock.patch.object(sys, "argv", ["checker"]), redirect_stdout(output):
                result = self.checker.main(argv)
            self.assertIs(type(result), int)
            self.assertEqual(result, 0)
            generate.assert_called_once_with()
            text = output.getvalue()
            self.assertTrue(text.startswith("Cubic three-body stability: exact scalar checks"))
            for phrase in markers + LIMITATIONS:
                self.assertIn(phrase, text)

    def test_cli_exact_json_uses_mocked_generation(self):
        report, output = report_oracle(), io.StringIO()
        with mock.patch.object(self.checker, "demonstration_report", return_value=report) as generate, \
             redirect_stdout(output):
            result = self.checker.main(["--json"])
        self.assertIs(type(result), int)
        self.assertEqual(result, 0)
        generate.assert_called_once_with()
        self.assert_native_equal(json.loads(output.getvalue()), report)

    def test_cli_no_abbreviation_and_help_parse_before_generation(self):
        cases = [([flag], 2) for flag in ("--j", "--js", "--jso", "--unknown", "positional", "--json=yes")]
        cases += [([flag], 0) for flag in ("-h", "--help")]
        for argv, status in cases:
            with self.subTest(argv=argv):
                output, error = io.StringIO(), io.StringIO()
                with mock.patch.object(self.checker, "demonstration_report") as generate, \
                     redirect_stdout(output), redirect_stderr(error):
                    with self.assertRaises(SystemExit) as caught:
                        self.checker.main(argv)
                self.assertEqual(caught.exception.code, status)
                generate.assert_not_called()
                if status == 0:
                    self.assertIn("--json", output.getvalue())

    def test_cli_json_rejects_all_nonfinite_mock_payloads(self):
        # Nonfinite values exist only as deliberately invalid serializer inputs.
        for spelling in ("nan", "inf", "-inf"):
            with self.subTest(value=spelling):
                report = report_oracle()
                report["local_state"]["x"] = float(spelling)
                with mock.patch.object(self.checker, "demonstration_report", return_value=report) as generate, \
                     redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    with self.assertRaises(ValueError):
                        self.checker.main(["--json"])
                generate.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
