"""Independent, fixed-control witnesses derived from the frozen 2026-09-13 contract.

Authored without reading the implementation. These checks are finite software
checks, not empirical validation or certified floating-point error bounds.
"""

import cmath
import copy
import importlib.util
import io
import itertools
import json
import math
from collections import Counter
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
import re
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ATOL = 1e-11
LIMITATIONS = [
    "Three-dimensional geometry is supplied, not derived.",
    "The continuum theorem is nonrelativistic and one-particle only.",
    "The empty vacuum is not the ground state of the unshifted model.",
    "No physical fermions, matter, gravity, calibration or empirical distinction is established.",
    "Finite floating checks are not certified roundoff bounds or empirical validation.",
]
CONTROLS = {
    "n": 3, "C": 1, "populations": [0, 1, 2], "couplings": [0, 1],
    "ell": 3, "kappa": 1, "J": 1, "times": [0, 0.1],
    "absolute_allowance": ATOL,
}
CASES = [(number, coupling) for number in range(3) for coupling in range(2)]


def _load_file(name, relative_path):
    """File loading must not run the scientific package initializer."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise ImportError(str(relative_path))
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"bpr": None, name: module}):
        spec.loader.exec_module(module)
    return module


model = _load_file("supplied_cubic_bose", "bpr/supplied_cubic_bose.py")


def _own_admit(dimension):
    """Bound oracle allocations independently of the production admission seam."""
    if dimension > 378 or dimension > model.MAX_DIMENSION:
        raise ValueError("test oracle dimension exceeds live cap")


def _particle_occupations(number):
    """Enumerate unordered particle positions, then sort their occupations."""
    dimension = math.comb(27 + number - 1, number)
    _own_admit(dimension)
    states = []
    for positions in itertools.combinations_with_replacement(range(27), number):
        occupation = tuple(positions.count(site) for site in range(27))
        states.append((occupation, positions))
    return sorted(states)


def _pair_terms(pair):
    """Sparse coefficients of a normalized symmetric first-quantized ket."""
    x, y = pair
    if x == y:
        return ((x, y, 1.0),)
    weight = 1.0 / math.sqrt(2.0)
    return ((x, y, weight), (y, x, weight))


class SuppliedCubicBoseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sites = tuple(itertools.product(range(3), repeat=3))
        cls.edge_set = {
            (i, j)
            for i, x in enumerate(cls.sites)
            for j, y in enumerate(cls.sites)
            if i < j and sum(min(abs(a - b), 3 - abs(a - b))
                             for a, b in zip(x, y)) == 1
        }
        _own_admit(27)
        cls.adjacency = np.array([
            [float((min(i, j), max(i, j)) in cls.edge_set)
             for j in range(27)] for i in range(27)
        ], dtype=np.float64)
        cls.occupation_oracles = {
            number: _particle_occupations(number) for number in range(3)
        }
        cls.sectors = {}
        cls.sector_calls = []
        cls.fourier_cases = []
        original_sector = model.cubic_sector
        original_fourier = model._fourier_diagnostics

        def capture_sector(number, coupling):
            result = original_sector(number, coupling)
            cls.sector_calls.append((number, coupling))
            cls.sectors[number, coupling] = result
            return result

        def capture_fourier(basis, hamiltonian):
            cls.fourier_cases.append([
                key for key, value in cls.sectors.items()
                if value[0] is basis and value[1] is hamiltonian
            ])
            return original_fourier(basis, hamiltonian)

        with mock.patch.object(model, "cubic_sector", side_effect=capture_sector), \
                mock.patch.object(model, "_fourier_diagnostics", side_effect=capture_fourier):
            cls.report = model.demonstration_report()

    def assertClose(self, actual, expected):
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)

    def assertNativeFinite(self, value):
        if type(value) is dict:
            for key, child in value.items():
                self.assertIs(type(key), str)
                self.assertNativeFinite(child)
        elif type(value) is list:
            for child in value:
                self.assertNativeFinite(child)
        elif type(value) in (int, float):
            self.assertTrue(math.isfinite(value))
        else:
            self.assertIn(type(value), (str, bool, type(None)))

    def fresh_sector(self, number, coupling):
        basis, matrix = self.sectors[number, coupling]
        _own_admit(len(basis))
        return list(basis), matrix.copy()

    def deny_allocations(self, stack):
        mocks = []
        for name in ("array", "asarray", "zeros", "ones", "empty", "full", "eye",
                     "identity", "zeros_like", "ones_like", "empty_like", "full_like"):
            mocks.append(stack.enter_context(mock.patch.object(
                np, name, side_effect=AssertionError("allocation before rejection")
            )))
        return mocks

    def test_01_strict_inputs_rejected_before_enumeration_or_allocation(self):
        class IntSubclass(int):
            pass

        invalid_numbers = [True, False, -1, 3, 1000000, 1.0, float("nan"),
                           float("inf"), -float("inf"), "1", None, [], {},
                           complex(1, 0), np.int64(1), np.float64(1), IntSubclass(1)]
        invalid_couplings = invalid_numbers + [2]
        with ExitStack() as stack:
            enumeration = stack.enter_context(mock.patch.object(
                model, "_occupations", side_effect=AssertionError("enumerated invalid input")
            ))
            allocations = self.deny_allocations(stack)
            for bad in invalid_numbers:
                for coupling in (0, 1):
                    with self.subTest(N=repr(bad), g=coupling):
                        with self.assertRaises(ValueError):
                            model.cubic_sector(bad, coupling)
            for bad in invalid_couplings:
                for number in (0, 1, 2):
                    with self.subTest(N=number, g=repr(bad)):
                        with self.assertRaises(ValueError):
                            model.cubic_sector(number, bad)
            enumeration.assert_not_called()
            for allocation in allocations:
                allocation.assert_not_called()

    def test_02_live_sector_cap_precedes_enumeration_and_later_allocation(self):
        self.assertEqual(model.MAX_DIMENSION, 378)
        for number, dimension in ((0, 1), (1, 27), (2, 378)):
            with self.subTest(N=number), ExitStack() as stack:
                stack.enter_context(mock.patch.object(model, "MAX_DIMENSION", dimension - 1))
                enumeration = stack.enter_context(mock.patch.object(
                    model, "_occupations", side_effect=AssertionError("enumerated over cap")
                ))
                self.deny_allocations(stack)
                with self.assertRaises(ValueError):
                    model.cubic_sector(number, 0)
                enumeration.assert_not_called()
        original_occupations = model._occupations

        with ExitStack() as stack:
            def lower_after_enumeration(*args, **kwargs):
                if args[0] != 27:
                    return original_occupations(*args, **kwargs)
                occupations = list(original_occupations(*args, **kwargs))
                model.MAX_DIMENSION = 0
                self.deny_allocations(stack)
                return occupations

            stack.enter_context(mock.patch.object(model, "MAX_DIMENSION", 378))
            stack.enter_context(mock.patch.object(
                model, "_occupations", side_effect=lower_after_enumeration
            ))
            with self.assertRaises(ValueError):
                model.cubic_sector(2, 1)
        with mock.patch.object(model, "MAX_DIMENSION", 26):
            with self.assertRaises(ValueError):
                model._admit(27)
        model._admit(378)
        with self.assertRaises(ValueError):
            model._admit(379)

    def test_03_full_graph_matches_pairwise_periodic_distance(self):
        sites, edges = model._graph()
        self.assertIs(type(sites), tuple)
        self.assertIs(type(edges), tuple)
        self.assertEqual(sites, self.sites)
        site_index = {site: index for index, site in enumerate(self.sites)}
        actual_edges = []
        for edge in edges:
            self.assertIs(type(edge), tuple)
            self.assertEqual(len(edge), 2)
            endpoints = tuple(site_index[v] if type(v) is tuple else v for v in edge)
            self.assertTrue(all(type(v) is int and 0 <= v < 27 for v in endpoints))
            self.assertNotEqual(*endpoints)
            actual_edges.append(tuple(sorted(endpoints)))
        self.assertEqual(len(actual_edges), 81)
        self.assertEqual(len(set(actual_edges)), 81)
        self.assertEqual(set(actual_edges), self.edge_set)
        self.assertEqual(Counter(v for edge in actual_edges for v in edge),
                         Counter({v: 6 for v in range(27)}))
        self.assertIn((0, 18), self.edge_set)  # (0,0,0) to (2,0,0), wraparound.

    def test_04_all_six_complete_bases_and_owned_float_matrices(self):
        self.assertEqual(set(self.sectors), set(CASES))
        for number, coupling in CASES:
            with self.subTest(N=number, g=coupling):
                basis, matrix = self.sectors[number, coupling]
                expected = [state for state, _ in self.occupation_oracles[number]]
                self.assertIs(type(basis), list)
                self.assertEqual(basis, expected)
                self.assertEqual(len(basis), (1, 27, 378)[number])
                self.assertEqual(len(set(basis)), len(basis))
                for state in basis:
                    self.assertIs(type(state), tuple)
                    self.assertEqual(len(state), 27)
                    self.assertTrue(all(type(n) is int and n >= 0 for n in state))
                    self.assertEqual(sum(state), number)
                self.assertEqual(matrix.shape, (len(basis), len(basis)))
                self.assertEqual(matrix.dtype, np.dtype("float64"))
                self.assertTrue(matrix.flags.owndata)
                self.assertIsNone(matrix.base)
                self.assertTrue(np.isfinite(matrix).all())
                self.assertClose(matrix, matrix.T)

    def test_05_vacuum_and_full_one_particle_hopping(self):
        for coupling in (0, 1):
            self.assertClose(self.sectors[0, coupling][1], [[0.0]])
            basis, matrix = self.sectors[1, coupling]
            row_for_site = [basis.index(tuple(int(j == site) for j in range(27)))
                            for site in range(27)]
            self.assertEqual(row_for_site, list(reversed(range(27))))
            _own_admit(27)
            in_site_order = matrix[np.ix_(row_for_site, row_for_site)]
            self.assertClose(in_site_order, -self.adjacency)
            uniform = np.ones(27, dtype=np.float64) / math.sqrt(27)
            self.assertClose(matrix @ uniform, -6 * uniform)
        self.assertClose(self.sectors[1, 0][1], self.sectors[1, 1][1])

    def test_06_full_two_particle_matrices_from_symmetric_pair_terms(self):
        states = self.occupation_oracles[2]
        _own_admit(len(states))
        expected = np.empty((378, 378), dtype=np.float64)
        terms = [_pair_terms(pair) for _, pair in states]
        for row, bra in enumerate(terms):
            for column, ket in enumerate(terms):
                value = 0.0
                for a, b, bra_weight in bra:
                    for c, d, ket_weight in ket:
                        # <a,b|(-A tensor I - I tensor A)|c,d>.
                        value += bra_weight * ket_weight * (
                            -self.adjacency[a, c] * (b == d)
                            -(a == c) * self.adjacency[b, d]
                        )
                expected[row, column] = value
        self.assertClose(self.sectors[2, 0][1], expected)
        # Contact projector in the normalized symmetric-pair basis.
        for row, (_, (a, b)) in enumerate(states):
            expected[row, row] += float(a == b)
        self.assertClose(self.sectors[2, 1][1], expected)

    def test_07_doublon_diagonal_hopping_and_signed_commutator(self):
        basis, free = self.sectors[2, 0]
        interacting = self.sectors[2, 1][1]
        doublons = [i for i, state in enumerate(basis) if 2 in state]
        split = [i for i, state in enumerate(basis) if 2 not in state]
        self.assertEqual(len(doublons), 27)
        self.assertEqual(len(split), 351)
        self.assertClose(np.diag(free), 0.0)
        self.assertClose(np.diag(interacting)[doublons], 1.0)
        self.assertClose(np.diag(interacting)[split], 0.0)
        x, y = self.sites.index((0, 0, 0)), self.sites.index((1, 0, 0))
        d = basis.index(tuple(2 * int(i == x) for i in range(27)))
        s = basis.index(tuple(int(i == x) + int(i == y) for i in range(27)))
        for coupling, matrix in ((0, free), (1, interacting)):
            self.assertClose(matrix[s, d], -math.sqrt(2.0))
            self.assertClose(matrix[d, s], -math.sqrt(2.0))
            self.assertClose((matrix[s, s] - matrix[d, d]) * matrix[s, d],
                             coupling * math.sqrt(2.0))

    def test_08_sector_results_are_detached_between_calls(self):
        first_basis, first_matrix = model.cubic_sector(1, 0)
        second_basis, second_matrix = model.cubic_sector(1, 0)
        self.assertIsNot(first_basis, second_basis)
        self.assertFalse(np.shares_memory(first_matrix, second_matrix))
        first_basis.clear()
        first_matrix[0, 0] = 123.0
        self.assertEqual(second_basis, self.sectors[1, 0][0])
        self.assertClose(second_matrix, self.sectors[1, 0][1])

    def test_09_independent_characters_orthogonality_and_row_permutation(self):
        basis, matrix = self.sectors[1, 0]
        modes = tuple(itertools.product((-1, 0, 1), repeat=3))
        position_for_row = [self.sites[state.index(1)] for state in basis]
        _own_admit(27)
        characters = np.array([
            [cmath.exp(2j * math.pi * sum(m[j] * x[j] for j in range(3)) / 3)
             / math.sqrt(27) for m in modes] for x in position_for_row
        ], dtype=np.complex128)
        identity = np.eye(27)
        energies = np.array([3.0 * sum(component != 0 for component in m) for m in modes])
        gram_error = characters.conj().T @ characters - identity
        eigen_error = (matrix + 6 * identity) @ characters - characters * energies
        self.assertClose(gram_error, 0.0)
        self.assertClose(eigen_error, 0.0)
        self.assertClose(self.report["fourier"]["orthogonality_residual"],
                         np.max(np.abs(gram_error)))
        self.assertClose(self.report["fourier"]["eigenvector_residual"],
                         np.max(np.abs(eigen_error)))
        site_characters = characters[[basis.index(tuple(int(j == i) for j in range(27)))
                                      for i in range(27)], :]
        self.assertClose((6 * identity - self.adjacency) @ site_characters,
                         site_characters * energies)
        # A non-geometric row swap catches a hard-coded site/occupation order.
        permutation = [1, 0] + list(range(2, 27))
        permuted_basis = [basis[i] for i in permutation]
        permuted_matrix = matrix[np.ix_(permutation, permutation)]
        result = model._fourier_diagnostics(permuted_basis, permuted_matrix)
        self.assertClose(result["orthogonality_residual"], 0.0)
        self.assertClose(result["eigenvector_residual"], 0.0)
        self.assertEqual([item["m"] for item in result["modes"]], [list(m) for m in modes])

    def test_10_fourier_all_energies_and_fixed_multiplicities(self):
        fourier = self.report["fourier"]
        self.assertEqual(set(fourier), {"orthogonality_residual", "eigenvector_residual", "modes"})
        for field in ("orthogonality_residual", "eigenvector_residual"):
            self.assertGreaterEqual(fourier[field], 0.0)
            self.assertClose(fourier[field], 0.0)
        expected_modes = list(itertools.product((-1, 0, 1), repeat=3))
        self.assertEqual([item["m"] for item in fourier["modes"]], [list(m) for m in expected_modes])
        energy_counts = Counter()
        for item, mode in zip(fourier["modes"], expected_modes):
            target = 3 * sum(component != 0 for component in mode)
            self.assertClose(item["lattice_energy"], target)
            energy_counts[target] += 1
        self.assertEqual(energy_counts, {0: 1, 3: 6, 6: 12, 9: 8})

    def test_11_all_mode_generator_and_phase_targets(self):
        for item in self.report["fourier"]["modes"]:
            with self.subTest(m=item["m"]):
                self.assertEqual(set(item), {
                    "m", "lattice_energy", "continuum_energy", "generator_error",
                    "generator_bound", "generator_lower_excess", "generator_upper_excess", "phases",
                })
                q = [2 * math.pi * component / 3 for component in item["m"]]
                # Adjacency/character eigenvalue, independent of the production sine formula.
                lattice = 6 - 2 * sum(math.cos(2 * math.pi * component / 3)
                                      for component in item["m"])
                continuum = sum(component ** 2 for component in q)
                bound = sum(component ** 4 for component in q) / 12
                error = continuum - lattice
                for key, target in (("lattice_energy", lattice), ("continuum_energy", continuum),
                                    ("generator_error", error), ("generator_bound", bound),
                                    ("generator_lower_excess", -error),
                                    ("generator_upper_excess", error - bound)):
                    self.assertClose(item[key], target)
                self.assertLessEqual(item["generator_lower_excess"], ATOL)
                self.assertLessEqual(item["generator_upper_excess"], ATOL)
                self.assertEqual([phase["time"] for phase in item["phases"]], [0, 0.1])
                for phase, time in zip(item["phases"], (0, 0.1)):
                    self.assertEqual(set(phase), {"time", "error", "bound", "signed_excess"})
                    phase_error = abs(cmath.exp(-1j * time * lattice)
                                      - cmath.exp(-1j * time * continuum))
                    phase_bound = min(2.0, abs(time) * bound)
                    self.assertClose(phase["error"], phase_error)
                    self.assertClose(phase["bound"], phase_bound)
                    self.assertClose(phase["signed_excess"], phase_error - phase_bound)
                    self.assertLessEqual(phase["signed_excess"], ATOL)

    def test_12_zero_controls_and_negative_excesses_remain_raw(self):
        for item in self.report["fourier"]["modes"]:
            zero_time = item["phases"][0]
            for field in ("time", "error", "bound", "signed_excess"):
                self.assertClose(zero_time[field], 0.0)
            self.assertClose(item["generator_lower_excess"], -item["generator_error"])
            self.assertClose(item["generator_upper_excess"],
                             item["generator_error"] - item["generator_bound"])
            for phase in item["phases"]:
                self.assertClose(phase["signed_excess"], phase["error"] - phase["bound"])
            if item["m"] == [0, 0, 0]:
                for field in ("lattice_energy", "continuum_energy", "generator_error",
                              "generator_bound", "generator_lower_excess", "generator_upper_excess"):
                    self.assertClose(item[field], 0.0)
                for phase in item["phases"]:
                    for field in ("error", "bound", "signed_excess"):
                        self.assertClose(phase[field], 0.0)
            else:
                self.assertLess(item["generator_lower_excess"], 0.0)
                self.assertLess(item["generator_upper_excess"], 0.0)
                self.assertLess(item["phases"][1]["signed_excess"], 0.0)

    def test_13_exact_report_fields_status_and_sector_summaries(self):
        report = self.report
        self.assertEqual(set(report), {"schema_version", "status", "empirical_validation",
                                     "derived_dimension", "controls", "graph", "sectors",
                                     "fourier", "limitations"})
        self.assertIs(type(report["schema_version"]), int)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["status"], "supplied_geometry_nonrelativistic_demonstrator")
        self.assertIs(report["empirical_validation"], False)
        self.assertIs(report["derived_dimension"], False)
        self.assertEqual(report["controls"], CONTROLS)
        self.assertEqual(report["graph"], {"sites": 27, "edges": 81, "degrees": [6] * 27})
        self.assertEqual(report["limitations"], LIMITATIONS)
        self.assertEqual([(item["N"], item["g"]) for item in report["sectors"]], CASES)
        for item in report["sectors"]:
            self.assertEqual(set(item), {"N", "g", "dimension", "hermiticity_residual",
                                        "interaction_trace", "doublons", "commutator_sd"})
            number, coupling = item["N"], item["g"]
            basis, matrix = self.sectors[number, coupling]
            _own_admit(len(basis))
            self.assertEqual(item["dimension"], (1, 27, 378)[number])
            self.assertClose(item["hermiticity_residual"], np.max(np.abs(matrix - matrix.T)))
            self.assertClose(item["interaction_trace"], 27 * coupling if number == 2 else 0.0)
            self.assertEqual(item["doublons"], 27 if number == 2 else 0)
            if number == 2:
                self.assertClose(item["commutator_sd"], coupling * math.sqrt(2.0))
            else:
                self.assertIsNone(item["commutator_sd"])
        self.assertNativeFinite(report)
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)

    def test_14_report_builds_each_control_once_and_one_fourier_case(self):
        self.assertEqual(self.sector_calls, CASES)
        self.assertEqual(self.fourier_cases, [[(1, 0)]])

    def test_15_reports_have_no_shared_mutable_containers(self):
        other = model.demonstration_report()
        self.assertEqual(other, self.report)

        def mutable_ids(value):
            if type(value) is dict:
                return {id(value)}.union(*(mutable_ids(v) for v in value.values()))
            if type(value) is list:
                return {id(value)}.union(*(mutable_ids(v) for v in value))
            return set()

        self.assertTrue(mutable_ids(other).isdisjoint(mutable_ids(self.report)))
        other["controls"]["times"].append(99)
        other["graph"]["degrees"][0] = -1
        other["sectors"][0]["dimension"] = -1
        other["fourier"]["modes"][0]["m"][0] = 99
        other["limitations"].clear()
        self.assertEqual(self.report["controls"], CONTROLS)
        self.assertEqual(self.report["graph"]["degrees"], [6] * 27)
        self.assertEqual(self.report["sectors"][0]["dimension"], 1)
        self.assertEqual(self.report["fourier"]["modes"][0]["m"], [-1, -1, -1])
        self.assertEqual(self.report["limitations"], LIMITATIONS)

    def test_16_finite_seam_recurses_through_json_and_rejects_nonfinite(self):
        valid = {"a": [None, True, False, 0, -2, 0.1, "text", {"nested": [3.0]}]}
        model._finite(valid)
        model._finite(self.report)
        for bad in (float("nan"), float("inf"), -float("inf")):
            for value in (bad, [bad], {"a": [0, {"b": bad}]}):
                with self.subTest(value=repr(value)):
                    with self.assertRaisesRegex(ValueError, "numerical failure"):
                        model._finite(value)

    def test_17_nonfinite_injection_fails_and_structural_errors_propagate(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            broken = copy.deepcopy(self.report["fourier"])
            broken["modes"][0]["phases"][1]["error"] = bad
            with mock.patch.object(model, "cubic_sector", side_effect=self.fresh_sector), \
                    mock.patch.object(model, "_fourier_diagnostics", return_value=broken):
                with self.assertRaisesRegex(ValueError, "numerical failure"):
                    model.demonstration_report()

        def broken_sector(number, coupling):
            basis, matrix = self.fresh_sector(number, coupling)
            if (number, coupling) == (0, 0):
                matrix[0, 0] = float("nan")
            return basis, matrix

        with mock.patch.object(model, "cubic_sector", side_effect=broken_sector), \
                mock.patch.object(model, "_fourier_diagnostics", return_value=self.report["fourier"]):
            with self.assertRaisesRegex(ValueError, "numerical failure"):
                model.demonstration_report()
        with mock.patch.object(model, "_fourier_diagnostics", side_effect=RuntimeError("structural sentinel")), \
                mock.patch.object(model, "cubic_sector", side_effect=self.fresh_sector):
            with self.assertRaisesRegex(RuntimeError, "structural sentinel"):
                model.demonstration_report()

    def test_18_fourier_and_report_respect_live_cap_before_allocation(self):
        basis, matrix = self.sectors[1, 0]
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(model, "MAX_DIMENSION", 26))
            allocations = self.deny_allocations(stack)
            with self.assertRaises(ValueError):
                model._fourier_diagnostics(basis, matrix)
            for allocation in allocations:
                allocation.assert_not_called()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(model, "MAX_DIMENSION", 0))
            enumeration = stack.enter_context(mock.patch.object(
                model, "_occupations", side_effect=AssertionError("report enumerated over cap")
            ))
            self.deny_allocations(stack)
            with self.assertRaises(ValueError):
                model.demonstration_report()
            enumeration.assert_not_called()

    def test_19_cli_text_and_json_mock_report_and_block_package_initializer(self):
        with mock.patch.dict(sys.modules, {"bpr": None, "supplied_cubic_bose": model}), \
                mock.patch.object(model, "demonstration_report", side_effect=AssertionError("ran on import")):
            cli = _load_file("_supplied_bose_cli_test", "scripts/demo_supplied_cubic_bose.py")
        for argv in ([], ["--json"]):
            stdout, stderr = io.StringIO(), io.StringIO()
            with self.subTest(argv=argv), \
                    mock.patch.object(cli, "demonstration_report", return_value=copy.deepcopy(self.report)) as report_call, \
                    redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(cli.main(argv), 0)
            report_call.assert_called_once_with()
            self.assertEqual(stderr.getvalue(), "")
            if argv:
                self.assertEqual(json.loads(stdout.getvalue()), self.report)
            else:
                text = stdout.getvalue()
                self.assertIn("Supplied cubic Bose demonstrator", text)
                for count in ("27", "81", "378"):
                    self.assertIn(count, text)
                for limitation in LIMITATIONS:
                    self.assertIn(limitation, text)
                for label, field in (("orthogonality", "orthogonality_residual"),
                                     ("eigenvector", "eigenvector_residual")):
                    lines = [line for line in text.splitlines() if label in line.lower()]
                    self.assertTrue(lines, label)
                    numbers = [float(token) for line in lines for token in re.findall(
                        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", line
                    )]
                    self.assertTrue(any(abs(number - self.report["fourier"][field]) <= ATOL
                                        for number in numbers), label)

    def test_20_cli_rejects_scientific_controls_and_nonfinite_json(self):
        with mock.patch.dict(sys.modules, {"bpr": None, "supplied_cubic_bose": model}):
            cli = _load_file("_supplied_bose_cli_rejection_test", "scripts/demo_supplied_cubic_bose.py")
        for argv in (["--N", "2"], ["--g", "1"], ["--n", "3"], ["--C", "1"],
                     ["--time", "0.1"], ["--ell", "3"], ["--kappa", "1"],
                     ["--J", "1"], ["extra"]):
            with self.subTest(argv=argv), mock.patch.object(cli, "demonstration_report") as report_call, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as rejected:
                    cli.main(argv)
                self.assertEqual(rejected.exception.code, 2)
                report_call.assert_not_called()
        broken = copy.deepcopy(self.report)
        broken["fourier"]["eigenvector_residual"] = float("nan")
        with mock.patch.object(cli, "demonstration_report", return_value=broken), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(ValueError):
                cli.main(["--json"])


if __name__ == "__main__":
    unittest.main()
