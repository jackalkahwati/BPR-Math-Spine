"""Independent exact tests of the frozen 2026-09-13 contact contract.

Authored from the derivation, without reading the implementation. Oracles use
729 position pairs only at n=3, at most 343 inverse partners per lattice, and
81 one-coordinate collision quadruples. No matrices or numerical propagation.
"""

import copy
import importlib.util
import io
import json
import math
import re
import sys
import unittest
from collections import Counter
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from fractions import Fraction
from itertools import product
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_cubic_contact_continuum.py"
LABELS = ("zero", "fixed", "formal-contact")
LIMITATIONS = [
    "Geometry, quantization and coupling scalings are supplied assumptions.",
    "The proved continuum comparison is free, fixed-band and two-particle only.",
    "Formal contact matching does not define an interacting continuum Hamiltonian.",
    "Generator leakage is not a lower bound on finite-time evolution error.",
    "Exact finite identities are not empirical validation or matter/gravity completion.",
]
COUNT_KEYS = {
    "inverse_pairs", "retained_pairs", "excluded_pairs", "self_inverse_pairs",
    "retained_unordered_pairs", "excluded_unordered_pairs",
    "integer_collisions_1d", "modular_collisions_1d", "modular_collisions_3d",
    "max_fiber_3d",
}
RATIONAL_KEYS = (
    "expectation", "contact_norm_squared", "projected_norm_squared",
    "leakage_norm_squared", "band_contact_norm_squared",
    "interaction_duhamel_coefficient_squared",
)
# Literal targets, not outputs of any implementation seam or formula helper.
TARGETS = {
    (3, "zero"): ("0", "0", "0", "0", "1", "0"),
    (3, "fixed"): ("1/27", "1/27", "1/27", "0", "1", "1"),
    (3, "formal-contact"): ("1", "27", "27", "0", "1", "729"),
    (5, "zero"): ("0", "0", "0", "0", "27/125", "0"),
    (5, "fixed"): ("1/125", "1/125", "27/15625", "98/15625", "27/125", "27/125"),
    (5, "formal-contact"): ("1", "125", "27", "98", "27/125", "3375"),
    (7, "zero"): ("0", "0", "0", "0", "27/343", "0"),
    (7, "fixed"): ("1/343", "1/343", "27/117649", "316/117649", "27/343", "27/343"),
    (7, "formal-contact"): ("1", "343", "27", "316", "27/343", "9261"),
}


class IntSubclass(int):
    pass


class TupleSubclass(tuple):
    pass


class FractionSubclass(Fraction):
    pass


def container_ids(value):
    if type(value) is dict:
        result = {id(value)}
        for child in value.values():
            result.update(container_ids(child))
        return result
    if type(value) is list:
        result = {id(value)}
        for child in value:
            result.update(container_ids(child))
        return result
    return set()


class CubicContactContinuumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # File-path loading deliberately avoids importing the bpr package.
        name = "_cubic_contact_continuum_contract_test"
        spec = importlib.util.spec_from_file_location(name, SCRIPT)
        if spec is None or spec.loader is None:
            raise AssertionError("Cannot construct the contact script loader")
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {name: module}):
            spec.loader.exec_module(module)
        cls.module = module

    def assert_rational(self, encoded, expected):
        self.assertIs(type(encoded), dict)
        self.assertEqual(set(encoded), {"numerator", "denominator"})
        numerator, denominator = encoded["numerator"], encoded["denominator"]
        self.assertIs(type(numerator), int)
        self.assertIs(type(denominator), int)
        self.assertGreater(denominator, 0)
        self.assertEqual(math.gcd(numerator, denominator), 1)
        self.assertEqual(Fraction(numerator, denominator), Fraction(expected))

    def assert_json_native(self, value):
        self.assertIn(type(value), (dict, list, str, int, bool, type(None)))
        if type(value) is dict:
            for key, child in value.items():
                self.assertIs(type(key), str)
                self.assert_json_native(child)
        elif type(value) is list:
            for child in value:
                self.assert_json_native(child)

    def forbid_enumeration(self, stack):
        # Only patch the module binding when product was imported there.
        if hasattr(self.module, "product"):
            return stack.enter_context(mock.patch.object(
                self.module, "product",
                side_effect=AssertionError("enumeration preceded rejection"),
            ))
        return None

    def test_report_schema_ordering_labels_and_limitations(self):
        report = self.module.demonstration_report()
        self.assertIs(type(report), dict)
        self.assertEqual(set(report), {
            "schema_version", "status", "empirical_validation", "controls",
            "cases", "limitations",
        })
        self.assertIs(type(report["schema_version"]), int)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["status"], "conditional_contact_scaling_diagnostics")
        self.assertIs(report["empirical_validation"], False)
        self.assertEqual(report["controls"], {
            "n": [3, 5, 7], "J": 1, "ell": 1, "coupling_labels": list(LABELS),
        })
        for key in ("J", "ell"):
            self.assertIs(type(report["controls"][key]), int)
        for n in report["controls"]["n"]:
            self.assertIs(type(n), int)
        self.assertEqual(report["limitations"], LIMITATIONS)
        self.assertIs(type(report["cases"]), list)
        expected_order = [(n, label) for n in (3, 5, 7) for label in LABELS]
        self.assertEqual([(c["n"], c["coupling_label"]) for c in report["cases"]], expected_order)
        case_keys = {
            "n", "M", "J", "K", "g", "coupling_label", "band_saturated",
            "no_alias_condition_met", "counts", *RATIONAL_KEYS,
        }
        for case in report["cases"]:
            with self.subTest(n=case["n"], label=case["coupling_label"]):
                self.assertEqual(set(case), case_keys)
                for key in ("n", "M", "J", "K", "g"):
                    self.assertIs(type(case[key]), int)
                n = case["n"]
                self.assertEqual((case["M"], case["J"], case["K"]), (n ** 3, 1, 27))
                self.assertEqual(case["g"], dict(zip(LABELS, (0, 1, n ** 3)))[case["coupling_label"]])
                self.assertIs(case["band_saturated"], n == 3)
                self.assertIs(case["no_alias_condition_met"], n > 4)
                self.assertEqual(set(case["counts"]), COUNT_KEYS)
                for value in case["counts"].values():
                    self.assertIs(type(value), int)
                self.assertEqual(case, self.module._case_report(n, case["g"], case["coupling_label"]))
                self.assertEqual(case["counts"], self.module._counts(n))
        self.assert_json_native(report)
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)

    def test_all_nine_literal_rational_targets(self):
        cases = self.module.demonstration_report()["cases"]
        self.assertEqual(len(cases), 9)
        for case in cases:
            key = (case["n"], case["coupling_label"])
            for field, expected in zip(RATIONAL_KEYS, TARGETS[key]):
                with self.subTest(case=key, field=field):
                    self.assert_rational(case[field], expected)

    def test_reports_and_private_containers_are_fresh(self):
        first = self.module.demonstration_report()
        second = self.module.demonstration_report()
        pristine = copy.deepcopy(second)
        self.assertEqual(first, second)
        self.assertFalse(container_ids(first) & container_ids(second))
        first["controls"]["n"].append(99)
        first["controls"]["coupling_labels"][0] = "changed"
        first["cases"][0]["counts"]["inverse_pairs"] = -1
        first["cases"][0]["expectation"]["numerator"] = -1
        first["limitations"].append("changed")
        first["cases"].pop()
        self.assertEqual(second, pristine)
        self.assertEqual(self.module.demonstration_report(), pristine)
        for n in (3, 5, 7):
            for g, label in zip((0, 1, n ** 3), LABELS):
                left = self.module._case_report(n, g, label)
                right = self.module._case_report(n, g, label)
                self.assertEqual(left, right)
                self.assertFalse(container_ids(left) & container_ids(right))
            counts_a, counts_b = self.module._counts(n), self.module._counts(n)
            self.assertEqual(counts_a, counts_b)
            self.assertIsNot(counts_a, counts_b)
        rational_a = self.module._rational(Fraction(2, 3))
        rational_b = self.module._rational(Fraction(2, 3))
        self.assertIsNot(rational_a, rational_b)

    def test_uniform_contact_from_729_ordered_position_pairs(self):
        sites = tuple(product(range(3), repeat=3))
        amplitude = Fraction(1, 27)
        normalization = expectation = norm_squared = Fraction(0)
        visited = diagonal = 0
        for x, y in product(sites, repeat=2):
            visited += 1
            diagonal += int(x == y)
            contact_amplitude = amplitude if x == y else Fraction(0)
            normalization += amplitude * amplitude
            expectation += amplitude * contact_amplitude
            norm_squared += contact_amplitude * contact_amplitude
        self.assertEqual((visited, diagonal), (729, 27))
        self.assertEqual(normalization, 1)
        self.assertEqual(expectation, Fraction(1, 27))
        self.assertEqual(norm_squared, Fraction(1, 27))
        for g, label in zip((0, 1, 27), LABELS):
            case = self.module._case_report(3, g, label)
            self.assert_rational(case["expectation"], g * expectation)
            self.assert_rational(case["contact_norm_squared"], g * g * norm_squared)

    def test_inverse_partners_and_weighted_unordered_projection(self):
        for n in (3, 5, 7):
            with self.subTest(n=n):
                residues = {0, 1, n - 1}
                ordered = []
                for p in product(range(n), repeat=3):
                    q = tuple((-coordinate) % n for coordinate in p)
                    retained = all(c in residues for c in p + q)
                    ordered.append((p, q, retained))
                self.assertLessEqual(len(ordered), 343)
                retained_count = sum(keep for _, _, keep in ordered)
                self_inverse = sum(p == q for p, q, _ in ordered)
                unordered = {}
                for p, q, keep in ordered:
                    pair = tuple(sorted((p, q)))
                    if pair in unordered:
                        self.assertEqual(unordered[pair], keep)
                    unordered[pair] = keep
                kept_unordered = sum(unordered.values())
                weights = [(1 if p == q else 2, keep) for (p, q), keep in unordered.items()]
                retained_weight = sum(weight for weight, keep in weights if keep)
                excluded_weight = sum(weight for weight, keep in weights if not keep)
                self.assertEqual(self_inverse, 1)
                self.assertEqual(retained_count, 27)
                self.assertEqual(kept_unordered, 14)
                self.assertEqual(retained_weight, retained_count)
                self.assertEqual(retained_weight + excluded_weight, len(ordered))
                expected = {
                    "inverse_pairs": len(ordered), "retained_pairs": retained_count,
                    "excluded_pairs": len(ordered) - retained_count,
                    "self_inverse_pairs": self_inverse,
                    "retained_unordered_pairs": kept_unordered,
                    "excluded_unordered_pairs": len(unordered) - kept_unordered,
                }
                counts = self.module._counts(n)
                for key, value in expected.items():
                    self.assertEqual(counts[key], value, key)
                for g, label in zip((0, 1, n ** 3), LABELS):
                    amplitude_squared = Fraction(g, n ** 3) ** 2
                    ordered_total = sum((amplitude_squared for _ in ordered), Fraction(0))
                    ordered_projected = sum((amplitude_squared for _, _, keep in ordered if keep), Fraction(0))
                    unordered_projected = amplitude_squared * retained_weight
                    unordered_leakage = amplitude_squared * excluded_weight
                    self.assertEqual(ordered_projected, unordered_projected)
                    self.assertEqual(ordered_total, unordered_projected + unordered_leakage)
                    case = self.module._case_report(n, g, label)
                    self.assert_rational(case["contact_norm_squared"], ordered_total)
                    self.assert_rational(case["projected_norm_squared"], unordered_projected)
                    self.assert_rational(case["leakage_norm_squared"], unordered_leakage)
                    if n == 3:
                        self.assertEqual(unordered_leakage, 0)
                    elif g:
                        # A uniform-ray complement would wrongly discard the other 26 band terms.
                        self.assertNotEqual(unordered_leakage, amplitude_squared * (n ** 3 - 1))

    def test_one_coordinate_collision_oracle_and_cubic_counts(self):
        quadruples = tuple(product((-1, 0, 1), repeat=4))
        self.assertEqual(len(quadruples), 81)
        exact = sum(p + q == r + s for p, q, r, s in quadruples)
        self.assertEqual(exact, 19)
        for n, target_1d, target_3d in ((3, 27, 19683), (5, 19, 6859), (7, 19, 6859)):
            modular = sum((p + q - r - s) % n == 0 for p, q, r, s in quadruples)
            self.assertEqual(modular, target_1d)
            counts = self.module._counts(n)
            self.assertEqual(counts["integer_collisions_1d"], exact)
            self.assertEqual(counts["modular_collisions_1d"], modular)
            self.assertEqual(counts["modular_collisions_3d"], modular ** 3)
            self.assertEqual(counts["modular_collisions_3d"], target_3d)
            self.assertEqual(modular == exact, n > 4)

    def test_momentum_fiber_maximum_by_coordinate_factorization(self):
        coordinate_pairs = tuple(product((-1, 0, 1), repeat=2))
        self.assertEqual(len(coordinate_pairs), 9)
        for n in (3, 5, 7):
            fibers = Counter((p + q) % n for p, q in coordinate_pairs)
            self.assertEqual(sum(fibers.values()), 9)
            self.assertEqual(fibers[0], 3)
            self.assertEqual(max(fibers.values()), 3)
            # Coordinate fibers are Cartesian factors; no 3D pair enumeration.
            cubic_max = max(fibers.values()) ** 3
            self.assertEqual(cubic_max, 27)
            self.assertEqual(self.module._counts(n)["max_fiber_3d"], cubic_max)
            for g, label in zip((0, 1, n ** 3), LABELS):
                case = self.module._case_report(n, g, label)
                self.assert_rational(case["band_contact_norm_squared"], Fraction(cubic_max, n ** 3))
                self.assert_rational(case["interaction_duhamel_coefficient_squared"], Fraction(g * g * cubic_max, n ** 3))

    def test_admission_accepts_only_frozen_cases_at_exact_caps(self):
        self.assertIs(type(self.module.MAX_MOMENTA), int)
        self.assertIs(type(self.module.MAX_ALIAS_QUADRUPLES), int)
        self.assertEqual(self.module.MAX_MOMENTA, 343)
        self.assertEqual(self.module.MAX_ALIAS_QUADRUPLES, 81)
        with ExitStack() as stack:
            enumeration = self.forbid_enumeration(stack)
            for n in (3, 5, 7):
                for g in (0, 1, n ** 3):
                    with mock.patch.object(self.module, "MAX_MOMENTA", n ** 3):
                        self.assertIsNone(self.module._admit_case(n, g))
            if enumeration is not None:
                enumeration.assert_not_called()

    def test_strict_admission_rejects_before_enumeration(self):
        bad_n = (True, False, IntSubclass(3), 3.0, Fraction(3), "3", None, [], 0, 2, 4, 9, -3, 10 ** 9)
        bad_g = (True, False, IntSubclass(1), 1.0, Fraction(1), "1", None, [], -1, 2, 26, 28)
        zero = (0, 0, 0)
        with ExitStack() as stack:
            enumeration = self.forbid_enumeration(stack)
            for n in bad_n:
                for operation in (
                    lambda n=n: self.module._admit_case(n, 0),
                    lambda n=n: self.module._counts(n),
                    lambda n=n: self.module._case_report(n, 0, "zero"),
                    lambda n=n: self.module._ordered_contact_element(n, 0, zero, zero, zero, zero),
                ):
                    with self.subTest(n=repr(n)), self.assertRaises(ValueError):
                        operation()
            for g in bad_g:
                for operation in (
                    lambda g=g: self.module._admit_case(3, g),
                    lambda g=g: self.module._case_report(3, g, "fixed"),
                    lambda g=g: self.module._ordered_contact_element(3, g, zero, zero, zero, zero),
                ):
                    with self.subTest(g=repr(g)), self.assertRaises(ValueError):
                        operation()
            if enumeration is not None:
                enumeration.assert_not_called()

    def test_live_caps_reject_all_enumerating_seams_before_product(self):
        zero = (0, 0, 0)
        for n in (3, 5, 7):
            for cap, value in (("MAX_MOMENTA", n ** 3 - 1), ("MAX_ALIAS_QUADRUPLES", 80)):
                with self.subTest(n=n, cap=cap), ExitStack() as stack:
                    stack.enter_context(mock.patch.object(self.module, cap, value))
                    enumeration = self.forbid_enumeration(stack)
                    for operation in (
                        lambda: self.module._admit_case(n, 0),
                        lambda: self.module._counts(n),
                        lambda: self.module._case_report(n, 0, "zero"),
                        lambda: self.module._ordered_contact_element(n, 0, zero, zero, zero, zero),
                    ):
                        with self.assertRaises(ValueError):
                            operation()
                    if enumeration is not None:
                        enumeration.assert_not_called()
        # The public report starts at n=3, so block its first case rather than
        # treating legitimate smaller-case enumeration as a failure.
        for cap, value in (("MAX_MOMENTA", 26), ("MAX_ALIAS_QUADRUPLES", 80)):
            with self.subTest(public_cap=cap), ExitStack() as stack:
                stack.enter_context(mock.patch.object(self.module, cap, value))
                enumeration = self.forbid_enumeration(stack)
                with self.assertRaises(ValueError):
                    self.module.demonstration_report()
                if enumeration is not None:
                    enumeration.assert_not_called()

    def test_rational_encoder_canonical_exact_values(self):
        for value, expected in (
            (0, {"numerator": 0, "denominator": 1}),
            (7, {"numerator": 7, "denominator": 1}),
            (-9, {"numerator": -9, "denominator": 1}),
            (Fraction(6, -8), {"numerator": -3, "denominator": 4}),
            (Fraction(0, 19), {"numerator": 0, "denominator": 1}),
            (Fraction(8, 4), {"numerator": 2, "denominator": 1}),
        ):
            with self.subTest(value=value):
                encoded = self.module._rational(value)
                self.assertEqual(encoded, expected)
                self.assert_rational(encoded, value)

    def test_rational_encoder_rejects_coercions_and_subclasses(self):
        for value in (True, False, IntSubclass(2), FractionSubclass(1, 2), 0.0, 0.5, "1/2", None, [1, 2], (1, 2), {"numerator": 1, "denominator": 2}, complex(1, 0)):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                self.module._rational(value)

    def test_case_labels_require_the_literal_coupling_mapping(self):
        for n in (3, 5, 7):
            for expected_g, label in zip((0, 1, n ** 3), LABELS):
                for g in (0, 1, n ** 3):
                    if g != expected_g:
                        with self.subTest(n=n, g=g, label=label), self.assertRaises(ValueError):
                            self.module._case_report(n, g, label)
            for label in ("", "Zero", "formal_contact", "formal-contact ", "unknown", None, 0, [], {}):
                with self.subTest(n=n, label=repr(label)), self.assertRaises(ValueError):
                    self.module._case_report(n, 0, label)

    def test_ordered_contact_genuine_forbidden_and_modular_transitions(self):
        zero = (0, 0, 0)
        for n in (3, 5, 7):
            for g in (0, 1, n ** 3):
                for axis in range(3):
                    plus = tuple(int(j == axis) for j in range(3))
                    minus = tuple(-c for c in plus)
                    other = tuple(int(j == (axis + 1) % 3) for j in range(3))
                    other_minus = tuple(-c for c in other)
                    transitions = (
                        ((plus, minus, other, other_minus), Fraction(g, n ** 3)),
                        ((plus, zero, zero, zero), Fraction(0)),
                        ((plus, plus, minus, zero), Fraction(g, 27) if n == 3 else Fraction(0)),
                    )
                    for momenta, expected in transitions:
                        p, q, r, s = momenta
                        for args in ((p, q, r, s), (q, p, r, s), (r, s, p, q)):
                            with self.subTest(n=n, g=g, axis=axis, momenta=args):
                                result = self.module._ordered_contact_element(n, g, *args)
                                self.assertIs(type(result), Fraction)
                                self.assertEqual(result, expected)

    def test_ordered_contact_rejects_every_malformed_momentum_slot(self):
        zero = (0, 0, 0)
        bad = (
            None, [0, 0, 0], TupleSubclass(zero), (), (0, 0), (0, 0, 0, 0),
            (True, 0, 0), (0, False, 0), (0, 0, IntSubclass(1)),
            (0.0, 0, 0), (0, Fraction(0), 0), (0, 0, "0"),
            (2, 0, 0), (0, -2, 0), (0, 0, 3), "000",
        )
        for slot in range(4):
            for value in bad:
                momenta = [zero] * 4
                momenta[slot] = value
                for g in (0, 1):
                    with self.subTest(slot=slot, value=repr(value), g=g), self.assertRaises(ValueError):
                        self.module._ordered_contact_element(3, g, *momenta)

    def test_counts_and_ordered_element_perform_admission_first(self):
        sentinel = ValueError("admission sentinel")
        with mock.patch.object(self.module, "_admit_case", side_effect=sentinel) as admission:
            with self.assertRaises(ValueError) as caught:
                self.module._counts(5)
            self.assertIs(caught.exception, sentinel)
            admission.assert_called_once_with(5, 0)
        with mock.patch.object(self.module, "_admit_case", side_effect=sentinel) as admission:
            with self.assertRaises(ValueError) as caught:
                self.module._ordered_contact_element(5, 1, None, None, None, None)
            self.assertIs(caught.exception, sentinel)
            admission.assert_called_once_with(5, 1)

    def test_cli_json_roundtrip_and_none_argv_without_subprocess(self):
        report = self.module.demonstration_report()
        for argv in (["--json"], None):
            output = io.StringIO()
            errors = io.StringIO()
            with mock.patch.object(self.module, "demonstration_report", return_value=report) as generate, mock.patch.object(sys, "argv", [str(SCRIPT), "--json"]), redirect_stdout(output), redirect_stderr(errors):
                result = self.module.main(argv)
            self.assertIs(type(result), int)
            self.assertEqual(result, 0)
            generate.assert_called_once_with()
            self.assertEqual(errors.getvalue(), "")
            self.assertEqual(json.loads(output.getvalue()), report)

    def test_cli_default_text_has_nine_cases_rationals_and_limitations(self):
        report = self.module.demonstration_report()
        output = io.StringIO()
        errors = io.StringIO()
        with mock.patch.object(self.module, "demonstration_report", return_value=report) as generate, redirect_stdout(output), redirect_stderr(errors):
            result = self.module.main([])
        self.assertEqual(result, 0)
        generate.assert_called_once_with()
        self.assertEqual(errors.getvalue(), "")
        text = output.getvalue()
        self.assertTrue(text.startswith("Cubic contact continuum: exact finite identities"))
        for label in LABELS:
            self.assertGreaterEqual(len(re.findall(r"\b" + re.escape(label) + r"\b", text)), 3)
        for rational in ("1/27", "1/125", "1/343", "27/15625", "98/15625", "27/117649", "316/117649"):
            numerator, denominator = rational.split("/")
            self.assertRegex(text, r"(?<!\d)" + numerator + r"\s*/\s*" + denominator + r"(?!\d)")
        positions = [text.index(limitation) for limitation in LIMITATIONS]
        self.assertEqual(positions, sorted(positions))

    def test_cli_rejects_extra_controls_before_generating_report(self):
        for argv in (["--n", "5"], ["--J", "1"], ["--ell", "1"], ["--g", "1"], ["--json", "extra"], ["--unknown"], ["--j"], ["--js"], ["--jso"]):
            with self.subTest(argv=argv), mock.patch.object(self.module, "demonstration_report") as generate, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    self.module.main(argv)
                self.assertEqual(caught.exception.code, 2)
                generate.assert_not_called()

    def test_cli_json_rejects_nonfinite_values_in_mocked_report(self):
        # Deliberately invalid mock payloads test allow_nan=False, not arithmetic.
        for spelling in ("nan", "inf", "-inf"):
            with self.subTest(value=spelling), mock.patch.object(self.module, "demonstration_report", return_value={"invalid": float(spelling)}) as generate, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(ValueError):
                    self.module.main(["--json"])
                generate.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
