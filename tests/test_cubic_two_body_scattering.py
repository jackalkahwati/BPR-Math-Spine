"""Independent, exact tests of section 8's frozen finite algebra contract.

The literal two-dimensional fixture is not a cubic-lattice approximation.
No continuum calculation or optical-theorem check is performed here.
"""

import contextlib
from fractions import Fraction
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


F = Fraction
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_cubic_two_body_scattering.py"
LIMITATIONS = [
    "This fixture checks finite rank-one algebra, not a cubic lattice approximation.",
    "The scattering limit rests on the reviewed analytic boundary proof, not these tests.",
    "Nonnegative onsite couplings have vanishing continuum scattering in the stated model.",
    "The attractive two-body tuning is not an adopted stable many-body model.",
    "No empirical validation, matter/gravity completion or TOE claim is made.",
]
TARGETS = (
    (-2, -1, F(1), F(-3), F(-2)),
    (-2, 0, F(3), F(0), F(-2, 3)),
    (-2, 1, F(5), F(3, 5), F(-2, 5)),
    (2, -1, F(5), F(-3, 5), F(2, 5)),
    (2, 0, F(3), F(0), F(2, 3)),
    (2, 1, F(1), F(3), F(2)),
)
CASE_KEYS = {
    "z", "g", "status", "free_source", "denominator",
    "interacting_determinant", "tau", "interacting_source",
}
IDENTITY = [[F(1), F(0)], [F(0), F(1)]]
PROJECTOR = [[F(1), F(0)], [F(0), F(0)]]


class IntSubclass(int):
    pass


class FractionSubclass(Fraction):
    pass


def encoded(value):
    """Test-owned serialization; never delegate an oracle to the checker."""
    value = F(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def determinant2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def adjugate_inverse2(matrix):
    """Only the explicit 2x2 adjugate, not a general inverse engine."""
    a, b = matrix[0]
    c, d = matrix[1]
    determinant = a * d - b * c
    if determinant == 0:
        raise ValueError("singular test fixture")
    return [[d / determinant, -b / determinant],
            [-c / determinant, a / determinant]]


def multiply2(left, right):
    return [
        [left[0][0] * right[0][0] + left[0][1] * right[1][0],
         left[0][0] * right[0][1] + left[0][1] * right[1][1]],
        [left[1][0] * right[0][0] + left[1][1] * right[1][0],
         left[1][0] * right[0][1] + left[1][1] * right[1][1]],
    ]


def fixture_matrices(z, g):
    """Build zI-A and zI-A-gP directly from the frozen literal entries."""
    z, g = F(z), F(g)
    return [[z, F(-1)], [F(-1), z]], [[z - g, F(-1)], [F(-1), z]]


def expected_case(z, g, determinant, tau, source):
    free = F(-2, 3) if z == -2 else F(2, 3)
    return {
        "z": z,
        "g": encoded(g),
        "status": "regular" if tau is not None else "expected_interacting_pole",
        "free_source": encoded(free),
        "denominator": encoded(1 - F(g) * free),
        "interacting_determinant": encoded(determinant),
        "tau": None if tau is None else encoded(tau),
        "interacting_source": None if source is None else encoded(source),
    }


def expected_report():
    """Fresh mocked CLI payload, independent of production report generation."""
    return {
        "schema_version": 1,
        "status": "abstract_rank_one_resolvent_diagnostics",
        "empirical_validation": False,
        "fixture": {"A": [[0, 1], [1, 0]], "e": [1, 0], "dimension": 2},
        "regular_cases": [expected_case(*row) for row in TARGETS],
        "pole_case": expected_case(2, F(3, 2), F(0), None, None),
        "limitations": list(LIMITATIONS),
    }


def container_ids(value):
    if type(value) is dict:
        return [id(value)] + [item for child in value.values() for item in container_ids(child)]
    if type(value) is list:
        return [id(value)] + [item for child in value for item in container_ids(child)]
    return []


def spoil_containers(value):
    if type(value) is dict:
        for child in list(value.values()):
            spoil_containers(child)
        value["test_mutation"] = True
    elif type(value) is list:
        for child in list(value):
            spoil_containers(child)
        value.append("test_mutation")


class CubicTwoBodyScatteringContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("cubic_two_body_scattering_contract_subject", SCRIPT)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load the contracted standalone checker")
        cls.subject = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.subject)

    def assertRational(self, value, expected):
        self.assertIs(type(value), dict)
        self.assertEqual(set(value), {"numerator", "denominator"})
        self.assertIs(type(value["numerator"]), int)
        self.assertIs(type(value["denominator"]), int)
        self.assertGreater(value["denominator"], 0)
        reduced = F(value["numerator"], value["denominator"])
        self.assertEqual(value, encoded(reduced))
        self.assertEqual(reduced, expected)
        return reduced

    def assertNativeJSON(self, value):
        if type(value) is dict:
            for key, child in value.items():
                self.assertIs(type(key), str)
                self.assertNativeJSON(child)
        elif type(value) is list:
            for child in value:
                self.assertNativeJSON(child)
        else:
            self.assertIn(type(value), (int, str, bool, type(None)))

    def run_cli(self, argv, payload):
        output, errors = io.StringIO(), io.StringIO()
        with mock.patch.object(self.subject, "demonstration_report", return_value=payload) as report:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                result = self.subject.main(argv)
            report.assert_called_once_with()
        self.assertIs(type(result), int)
        self.assertEqual(result, 0)
        self.assertEqual(errors.getvalue(), "")
        return output.getvalue()

    def test_rational_encoding_is_reduced_positive_and_exact(self):
        for value in (0, -3, 7, F(0, 9), F(6, -8), F(-12, -4), F(7, 11)):
            with self.subTest(value=value):
                self.assertRational(self.subject._rational(value), F(value))

    def test_rational_rejects_non_native_types_and_subclasses(self):
        for value in (True, False, 1.0, "1", None, [], {}, object(),
                      IntSubclass(1), FractionSubclass(1, 2)):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    self.subject._rational(value)

    def test_free_resolvent_exact_values_and_distinct_free_poles(self):
        for z, expected in ((-2, F(-2, 3)), (2, F(2, 3))):
            result = self.subject._free_resolvent(z)
            self.assertIs(type(result), Fraction)
            self.assertEqual(result, expected)
        for z in (-1, 1):
            with self.subTest(z=z):
                with self.assertRaisesRegex(ValueError, "free pole"):
                    self.subject._free_resolvent(z)
                with self.assertRaises(ValueError):
                    self.subject._case_report(z, 0)

    def test_free_resolvent_rejects_off_grid_and_non_native_inputs(self):
        for value in (-3, 0, 3, True, False, 2.0, F(2), "2", None,
                      IntSubclass(2), FractionSubclass(2), [], {}):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    self.subject._free_resolvent(value)

    def test_case_admission_rejects_invalid_inputs_before_scalar_work(self):
        invalid_z = (-3, -1, 0, 1, 3, True, False, -2.0, F(-2), "-2", None,
                     IntSubclass(-2), FractionSubclass(-2), [], {})
        invalid_g = (-2, 2, True, False, 1.0, F(-1), F(0), F(1), F(2, 3),
                     "1", None, IntSubclass(1), FractionSubclass(3, 2), [], {})
        pairs = [(z, 0) for z in invalid_z]
        pairs += [(z, g) for z in (-2, 2) for g in invalid_g]
        pairs.append((-2, F(3, 2)))
        for z, g in pairs:
            with self.subTest(z=repr(z), g=repr(g)):
                with mock.patch.object(self.subject, "_free_resolvent", side_effect=AssertionError("arithmetic before admission")) as free:
                    with mock.patch.object(self.subject, "_rational", side_effect=AssertionError("encoding before admission")) as rational:
                        with self.assertRaises(ValueError):
                            self.subject._case_report(z, g)
                        free.assert_not_called()
                        rational.assert_not_called()

    def test_dimension_cap_is_live_and_admitted_before_case_entry(self):
        self.assertIs(type(self.subject.MAX_DIMENSION), int)
        self.assertEqual(self.subject.MAX_DIMENSION, 2)
        self.subject._admit_dimension()
        for cap in (-1, 0, 1):
            with self.subTest(cap=cap), mock.patch.object(self.subject, "MAX_DIMENSION", cap):
                with self.assertRaises(ValueError):
                    self.subject._admit_dimension()
                with mock.patch.object(self.subject, "_case_report", side_effect=AssertionError("case entered before dimension admission")) as case:
                    with self.assertRaises(ValueError):
                        self.subject.demonstration_report()
                    case.assert_not_called()
                for z, g in ((-2, -1), (-2, 0), (-2, 1), (2, -1), (2, 0), (2, 1), (2, F(3, 2))):
                    with mock.patch.object(self.subject, "_free_resolvent", side_effect=AssertionError("free calculation before cap")) as free:
                        with mock.patch.object(self.subject, "_rational", side_effect=AssertionError("encoding before cap")) as rational:
                            with self.assertRaises(ValueError):
                                self.subject._case_report(z, g)
                            free.assert_not_called()
                            rational.assert_not_called()

    def test_admission_failure_stops_report_and_case_construction(self):
        class AdmissionStopped(ValueError):
            pass

        def stop_before_owned_containers():
            # Inspect only active checker frames at its declared admission seam.
            # A fixture allocated before this seam is a contract violation even
            # when no case or scalar helper has been called yet.
            frame = sys._getframe(1)
            try:
                while frame is not None:
                    if frame.f_code.co_filename == str(SCRIPT):
                        for name, value in frame.f_locals.items():
                            self.assertNotIn(type(value), (list, dict),
                                             f"{name} constructed before dimension admission")
                    frame = frame.f_back
            finally:
                del frame
            raise AdmissionStopped("admission stopped")

        for helper, args in ((self.subject.demonstration_report, ()),
                             (self.subject._case_report, (2, 1)),
                             (self.subject._case_report, (2, F(3, 2)))):
            with self.subTest(helper=helper.__name__, args=args):
                with mock.patch.object(self.subject, "_admit_dimension", side_effect=stop_before_owned_containers) as admission:
                    with mock.patch.object(self.subject, "_rational", side_effect=AssertionError("construction before admission")) as rational:
                        with mock.patch.object(self.subject, "_free_resolvent", side_effect=AssertionError("calculation before admission")) as free:
                            with self.assertRaises(AdmissionStopped):
                                helper(*args)
                            admission.assert_called_once_with()
                            rational.assert_not_called()
                            free.assert_not_called()

    def test_all_six_regular_cases_match_literal_targets(self):
        for row in TARGETS:
            z, g, determinant, tau, source = row
            with self.subTest(z=z, g=g):
                result = self.subject._case_report(z, g)
                self.assertEqual(result, expected_case(*row))
                self.assertEqual(set(result), CASE_KEYS)
                self.assertIs(type(result["z"]), int)
                self.assertRational(result["g"], F(g))
                self.assertRational(result["free_source"], F(-2, 3) if z == -2 else F(2, 3))
                self.assertRational(result["denominator"], 1 - F(g) * (F(-2, 3) if z == -2 else F(2, 3)))
                self.assertRational(result["interacting_determinant"], determinant)
                self.assertRational(result["tau"], tau)
                self.assertRational(result["interacting_source"], source)

    def test_full_rank_one_resolvent_and_both_inverse_residuals(self):
        for z, g, _, _, _ in TARGETS:
            with self.subTest(z=z, g=g):
                free_matrix, matrix = fixture_matrices(z, g)
                free_inverse = adjugate_inverse2(free_matrix)
                direct_inverse = adjugate_inverse2(matrix)
                case = self.subject._case_report(z, g)
                # This tau oracle comes from the direct inverse, not g/(1-gG).
                independent_tau = F(g) + F(g) ** 2 * direct_inverse[0][0]
                tau = self.assertRational(case["tau"], independent_tau)
                self.assertRational(case["free_source"], free_inverse[0][0])
                self.assertRational(case["interacting_source"], direct_inverse[0][0])
                correction = multiply2(multiply2(free_inverse, PROJECTOR), free_inverse)
                rank_one_inverse = [[free_inverse[i][j] + tau * correction[i][j]
                                     for j in range(2)] for i in range(2)]
                self.assertEqual(rank_one_inverse, direct_inverse)
                for operator, inverse in ((free_matrix, free_inverse),
                                          (matrix, direct_inverse),
                                          (matrix, rank_one_inverse)):
                    self.assertEqual(multiply2(operator, inverse), IDENTITY)
                    self.assertEqual(multiply2(inverse, operator), IDENTITY)

    def test_determinant_lemma_including_the_separate_pole(self):
        for z, g in [(row[0], row[1]) for row in TARGETS] + [(2, F(3, 2))]:
            with self.subTest(z=z, g=g):
                free_matrix, matrix = fixture_matrices(z, g)
                free_inverse = adjugate_inverse2(free_matrix)
                denominator = F(1) - F(g) * free_inverse[0][0]
                self.assertEqual(determinant2(matrix), determinant2(free_matrix) * denominator)
                case = self.subject._case_report(z, g)
                self.assertRational(case["denominator"], denominator)
                self.assertRational(case["interacting_determinant"], determinant2(matrix))

    def test_tau_identity_uses_direct_interacting_source(self):
        for z, g, _, _, _ in TARGETS:
            with self.subTest(z=z, g=g):
                _, matrix = fixture_matrices(z, g)
                source = adjugate_inverse2(matrix)[0][0]
                case = self.subject._case_report(z, g)
                self.assertRational(case["interacting_source"], source)
                self.assertRational(case["tau"], F(g) + F(g) * F(g) * source)

    def test_zero_coupling_is_regular_without_dividing_by_g(self):
        for z in (-2, 2):
            case = self.subject._case_report(z, 0)
            free_matrix, matrix = fixture_matrices(z, 0)
            self.assertEqual(adjugate_inverse2(matrix), adjugate_inverse2(free_matrix))
            self.assertEqual(case["status"], "regular")
            self.assertRational(case["tau"], F(0))
            self.assertRational(case["denominator"], F(1))
            self.assertEqual(case["interacting_source"], case["free_source"])

    def test_expected_interacting_pole_has_no_finite_amplitudes(self):
        free_matrix, matrix = fixture_matrices(2, F(3, 2))
        free_inverse = adjugate_inverse2(free_matrix)
        self.assertEqual(multiply2(free_matrix, free_inverse), IDENTITY)
        self.assertEqual(multiply2(free_inverse, free_matrix), IDENTITY)
        self.assertEqual(determinant2(matrix), F(0))
        with self.assertRaises(ValueError):
            adjugate_inverse2(matrix)
        case = self.subject._case_report(2, F(3, 2))
        self.assertEqual(case, expected_case(2, F(3, 2), F(0), None, None))
        self.assertEqual(set(case), CASE_KEYS)
        self.assertIsNone(case["tau"])
        self.assertIsNone(case["interacting_source"])
        self.assertRational(case["free_source"], F(2, 3))
        self.assertRational(case["denominator"], F(0))
        self.assertRational(case["interacting_determinant"], F(0))

    def test_complete_report_schema_order_limitations_and_native_json(self):
        report = self.subject.demonstration_report()
        self.assertEqual(report, expected_report())
        self.assertEqual(set(report), set(expected_report()))
        self.assertIs(type(report["schema_version"]), int)
        self.assertIs(report["empirical_validation"], False)
        self.assertEqual(len(report["regular_cases"]), 6)
        self.assertEqual([(case["z"], case["g"]) for case in report["regular_cases"]],
                         [(row[0], encoded(row[1])) for row in TARGETS])
        for case in report["regular_cases"] + [report["pole_case"]]:
            self.assertEqual(set(case), CASE_KEYS)
            self.assertIs(type(case["z"]), int)
            for key in CASE_KEYS - {"z", "status"}:
                if case[key] is not None:
                    self.assertRational(case[key], F(case[key]["numerator"], case[key]["denominator"]))
        self.assertEqual(report["limitations"], LIMITATIONS)
        self.assertNativeJSON(report)
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)

    def test_report_and_case_containers_are_detached_and_fresh(self):
        first = self.subject.demonstration_report()
        second = self.subject.demonstration_report()
        self.assertFalse(set(container_ids(first)) & set(container_ids(second)))
        self.assertEqual(len(container_ids(first)), len(set(container_ids(first))))
        standalone = self.subject._case_report(-2, -1)
        repeated = self.subject._case_report(-2, -1)
        self.assertFalse(set(container_ids(standalone)) & set(container_ids(first)))
        self.assertFalse(set(container_ids(standalone)) & set(container_ids(repeated)))
        spoil_containers(first)
        spoil_containers(standalone)
        self.assertEqual(second, expected_report())
        self.assertEqual(repeated, expected_case(*TARGETS[0]))
        self.assertEqual(self.subject.demonstration_report(), expected_report())
        self.assertEqual(self.subject._case_report(-2, -1), expected_case(*TARGETS[0]))
        pole = self.subject._case_report(2, F(3, 2))
        spoil_containers(pole)
        self.assertEqual(self.subject._case_report(2, F(3, 2)), expected_case(2, F(3, 2), F(0), None, None))

    def test_cli_default_text_with_mocked_report(self):
        text = self.run_cli([], expected_report())
        self.assertTrue(text.startswith("Cubic two-body scattering: finite resolvent algebra"))
        for z, g, _, _, _ in TARGETS:
            # Whitespace around '=' is presentation, not a scientific control.
            self.assertRegex(text, rf"(?m)^.*z\s*=\s*{z}\b.*g\s*=\s*{g}\b.*$")
        self.assertRegex(text.lower(), r"expected[^\n]*pole")
        for limitation in LIMITATIONS:
            self.assertIn(limitation, text)

    def test_cli_exact_json_flag_with_mocked_report(self):
        payload = expected_report()
        text = self.run_cli(["--json"], payload)
        self.assertEqual(json.loads(text), payload)
        self.assertNativeJSON(json.loads(text))
        self.assertNotIn("Cubic two-body scattering: finite resolvent algebra", text)

    def test_cli_none_argv_and_json_disallows_nan(self):
        payload = expected_report()
        with mock.patch("sys.argv", [str(SCRIPT), "--json"]):
            self.assertEqual(json.loads(self.run_cli(None, payload)), payload)
        # A spy verifies the serialization policy without supplying float data.
        with mock.patch.object(self.subject, "demonstration_report", return_value=payload) as report:
            with mock.patch.object(json, "dumps", wraps=json.dumps) as dumps:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(self.subject.main(["--json"]), 0)
                self.assertTrue(dumps.called)
                self.assertTrue(any(call.kwargs.get("allow_nan") is False for call in dumps.call_args_list))
            report.assert_called_once_with()

    def test_cli_rejects_abbreviations_and_controls_before_report(self):
        invalid = (["--j"], ["--js"], ["--jso"], ["--JSON"], ["--json=true"],
                   ["--z", "2"], ["--g", "1"], ["--dimension", "2"],
                   ["--max-dimension", "2"], ["--samples", "2"],
                   ["--tolerance", "0"], ["--energy", "2"],
                   ["--json", "--z", "2"], ["2"])
        for argv in invalid:
            with self.subTest(argv=argv):
                with mock.patch.object(self.subject, "demonstration_report", side_effect=AssertionError("report generated before CLI admission")) as report:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as stopped:
                            self.subject.main(argv)
                    self.assertEqual(stopped.exception.code, 2)
                    report.assert_not_called()

    def test_cli_help_does_not_generate_report(self):
        for argv in (["--help"], ["-h"]):
            with self.subTest(argv=argv):
                with mock.patch.object(self.subject, "demonstration_report", side_effect=AssertionError("help generated report")) as report:
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as stopped:
                            self.subject.main(argv)
                    self.assertEqual(stopped.exception.code, 0)
                    self.assertIn("--json", output.getvalue())
                    report.assert_not_called()


if __name__ == "__main__":
    unittest.main()
