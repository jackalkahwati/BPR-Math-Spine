"""Independent checks for doc/derivations/cubic_condensate_regime_2026-09-25.md.

Oracles below rebuild the lattice Hamiltonian, Bogoliubov quadratic form and
BdG generator from the stated definitions rather than from the module's
helpers. They are bounded floating checks, not certified error bounds or
empirical validation.
"""

import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import scipy.sparse  # noqa: F401  (load before the isolated module import)
import scipy.sparse.linalg  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "cubic_condensate_regime", ROOT / "bpr" / "cubic_condensate_regime.py")
    module = importlib.util.module_from_spec(spec)
    # The module must not need the eager scientific package initializer.
    with mock.patch.dict(sys.modules, {"bpr": None, "cubic_condensate_regime": module}):
        spec.loader.exec_module(module)
    return module


c = _load()


# ---------------------------------------------------------------------------
# Independent oracles
# ---------------------------------------------------------------------------

def _sites(n):
    return list(itertools.product(range(n), repeat=3))


def _neighbors(n):
    sites = _sites(n)
    pairs = set()
    for i, s in enumerate(sites):
        for j, t in enumerate(sites):
            if i < j:
                d = sum(min((a - b) % n, (b - a) % n) for a, b in zip(s, t))
                if d == 1:
                    pairs.add((i, j))
    return pairs


def _two_particle_hamiltonian(n, g, C=1.0):
    """First-quantized symmetric two-boson Hamiltonian, independent of occupations."""
    M = n ** 3
    pairs = _neighbors(n)
    adjacency = np.zeros((M, M))
    for i, j in pairs:
        adjacency[i, j] = adjacency[j, i] = 1.0
    one = -C * adjacency
    eye = np.eye(M)
    h2 = np.kron(one, eye) + np.kron(eye, one)
    h2 += g * np.diag([1.0 if x == y else 0.0 for x in range(M) for y in range(M)])
    swap = np.zeros((M * M, M * M))
    for x in range(M):
        for y in range(M):
            swap[x * M + y, y * M + x] = 1.0
    sym = 0.5 * (np.eye(M * M) + swap)
    w, v = np.linalg.eigh(sym)
    basis = v[:, w > 0.5]
    return basis, basis.T @ h2 @ basis, one


def _bogoliubov_numerical(n, lam, C=1.0):
    """Ground energy and frequencies of H_B from its symplectic spectrum."""
    modes = [m for m in itertools.product(range(n), repeat=3) if any(m)]
    index = {m: i for i, m in enumerate(modes)}
    K = len(modes)
    mu = lam / n ** 3
    eps = [2 * C * sum(1 - math.cos(2 * math.pi * x / n) for x in m) for m in modes]
    A = np.diag([e + mu for e in eps])
    B = np.zeros((K, K))
    for m in modes:
        partner = tuple((-x) % n for x in m)
        B[index[m], index[partner]] += mu
    generator = np.block([[A, B], [-B, -A]])
    omega = np.sort(np.linalg.eigvals(generator).real)
    positive = omega[K:]
    return 0.5 * (positive.sum() - np.trace(A)), positive


def _bdg_oracle(n, m_p, mu, C=1.0):
    M = n ** 3
    sites = _sites(n)
    adjacency = np.zeros((M, M))
    for i, j in _neighbors(n):
        adjacency[i, j] = adjacency[j, i] = 1.0
    p = [2 * math.pi * x / n for x in m_p]
    phi = np.array([np.exp(1j * sum(a * b for a, b in zip(p, s))) for s in sites]) / math.sqrt(M)
    lam = mu * M
    chem = 2 * C * sum(1 - math.cos(x) for x in p) - 6 * C + mu
    D = -C * adjacency + np.diag(2 * lam * np.abs(phi) ** 2) - chem * np.eye(M)
    O = np.diag(lam * phi ** 2)
    return np.block([[D, O], [-O.conj(), -D.conj()]])


def _match(a, b):
    """Greedy nearest matching of two complex multisets; returns the worst distance."""
    remaining = list(b)
    worst = 0.0
    for x in sorted(a, key=lambda z: (round(z.real, 6), round(z.imag, 6))):
        distances = [abs(x - y) for y in remaining]
        i = int(np.argmin(distances))
        worst = max(worst, distances[i])
        remaining.pop(i)
    return worst


# ---------------------------------------------------------------------------
# Theorem 1: sector vacuum
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N,g", [(1, 0), (2, 0), (2, 1), (2, 5), (3, 1), (3, 5)])
def test_sector_vacuum_simple_positive_symmetric(N, g):
    report = c.sector_vacuum_report(3, N, g)
    assert report["first_gap"] > 1e-6
    assert report["min_component"] > 0
    for residual in report["symmetry_residuals"].values():
        assert residual < 1e-9
    assert report["condensate_overlap"] > 0
    assert report["product_state_upper_bound_margin"] >= -1e-9


def test_two_particle_ground_matches_first_quantized_oracle():
    for g in (0.0, 1.0, 5.0):
        _, H, _ = _two_particle_hamiltonian(3, g)
        oracle = np.linalg.eigvalsh(H)[0]
        assert c.sector_vacuum_report(3, 2, g)["ground_energy"] == pytest.approx(oracle, abs=1e-10)


def test_vacuum_on_hypercube_lattice_is_simple_and_positive():
    # n=4 gives the 6-cube; the positive ground vector must still be unique.
    report = c.sector_vacuum_report(4, 1, 0)
    assert report["first_gap"] > 0.1
    assert report["min_component"] > 0


def test_vacuum_invariant_under_non_product_hypercube_automorphism():
    """C_4^3 is the 6-cube via a Gray code; swap bits across two coordinates."""
    gray = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
    inverse = {bits: i for i, bits in gray.items()}

    def automorphism(site):
        bits = [list(gray[x]) for x in site]
        bits[0][1], bits[1][1] = bits[1][1], bits[0][1]
        return tuple(inverse[tuple(b)] for b in bits)

    sites = _sites(4)
    index = {s: i for i, s in enumerate(sites)}
    site_map = [index[automorphism(s)] for s in sites]
    edges = {frozenset(e) for e in _neighbors(4)}
    assert {frozenset((site_map[i], site_map[j])) for i, j in edges} == edges
    # Not a product of per-coordinate cycle automorphisms.
    assert automorphism((1, 0, 0))[0] != automorphism((1, 2, 0))[0]
    import scipy.sparse.linalg as sla
    basis, H = c.position_sector(4, 2, 3.0)
    w, v = sla.eigsh(H, k=2, which="SA", tol=1e-13)
    order = np.argsort(w)
    ground = v[:, order[0]]
    ground = ground if ground.sum() > 0 else -ground
    assert w[order[1]] - w[order[0]] > 1e-6
    assert ground.min() > 0
    rows = {state: i for i, state in enumerate(basis)}
    image = np.empty_like(ground)
    for i, state in enumerate(basis):
        moved = [0] * len(state)
        for x, count in enumerate(state):
            moved[site_map[x]] = count
        image[rows[tuple(moved)]] = ground[i]
    assert np.max(np.abs(image - ground)) < 1e-9


def test_position_basis_rejects_oversized_sector():
    with pytest.raises(ValueError):
        c.position_sector(3, 4, 1.0)


# ---------------------------------------------------------------------------
# Momentum basis equals position basis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("g", [0.0, 2.0])
def test_momentum_sectors_reproduce_full_two_particle_spectrum(g):
    _, H, _ = _two_particle_hamiltonian(3, g)
    full = np.sort(np.linalg.eigvalsh(H))
    pieces = []
    for K in range(27):
        _, block = c.momentum_sector(3, 2, g, K)
        pieces.extend(np.linalg.eigvalsh(block.toarray()))
    assert len(pieces) == len(full)
    assert np.max(np.abs(np.sort(pieces) - full)) < 1e-9


def test_three_particle_momentum_sectors_match_position_basis():
    basis, H = c.position_sector(3, 3, 1.5)
    full = np.sort(np.linalg.eigvalsh(H.toarray()))
    pieces = []
    for K in range(27):
        _, block = c.momentum_sector(3, 3, 1.5, K)
        pieces.extend(np.linalg.eigvalsh(block.toarray()))
    assert len(pieces) == len(basis)
    assert np.max(np.abs(np.sort(pieces) - full)) < 1e-8


# ---------------------------------------------------------------------------
# Lemma 2 and Bogoliubov data
# ---------------------------------------------------------------------------

def test_hartree_uniform_minimizer_against_random_trials():
    rng = np.random.default_rng(20260925)
    for n in (3, 4):
        for lam in (0.0, 3.0, 40.0):
            floor = -6 + lam / (2 * n ** 3)
            assert c.hartree_minimum(n, 1.0, lam) == pytest.approx(floor)
            uniform = np.ones(n ** 3)
            assert c.hartree_energy(uniform, n, 1.0, lam) == pytest.approx(floor, abs=1e-12)
            for _ in range(40):
                trial = rng.normal(size=n ** 3) + 1j * rng.normal(size=n ** 3)
                trial = trial / np.linalg.norm(trial)
                mixed = uniform / np.linalg.norm(uniform) + 0.05 * trial
                assert c.hartree_energy(trial, n, 1.0, lam) >= floor - 1e-12
                assert c.hartree_energy(mixed, n, 1.0, lam) >= floor - 1e-12


@pytest.mark.parametrize("n,lam", [(3, 0.0), (3, 20.0), (4, 7.0), (5, 50.0)])
def test_bogoliubov_formula_matches_symplectic_diagonalization(n, lam):
    data = c.bogoliubov_modes(n, 1.0, lam)
    energy, frequencies = _bogoliubov_numerical(n, lam)
    assert data["E_B"] == pytest.approx(energy, abs=1e-9)
    formula = np.sort([m["e"] for m in data["modes"]])
    assert np.max(np.abs(formula - frequencies)) < 1e-8
    for m in data["modes"]:
        assert m["u2"] - m["v2"] == pytest.approx(1.0, abs=1e-12)
        assert m["residue_per_particle"] <= 1.0 + 1e-12


def test_gap_constant_is_smallest_nonzero_dispersion():
    for n in range(3, 10):
        eps = c.dispersion(n)
        assert min(e for e in eps if e > 1e-12) == pytest.approx(c.gap_constant(n))


# ---------------------------------------------------------------------------
# Theorem 3, Theorem 4, Corollary 1: bounded exact diagonalization
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ladder():
    return {lam: [c.mean_field_case(N, lam) for N in (2, 3, 4)] for lam in (5.0, 20.0)}


def test_coercivity_and_depletion_bounds_hold(ladder):
    for cases in ladder.values():
        for case in cases:
            assert case["coercivity_min_eigenvalue"] >= -1e-9
            assert 0 <= case["depletion"] <= case["depletion_bound"] + 1e-12


def test_ed_approaches_bogoliubov_monotonically(ladder):
    for cases in ladder.values():
        gaps = [abs(case["gap_error"]) for case in cases]
        energies = [abs(case["energy_error"]) for case in cases]
        residues = [abs(case["residue_per_particle"] - case["bogoliubov_residue"]) for case in cases]
        assert gaps[0] > gaps[1] > gaps[2]
        assert energies[0] > energies[1] > energies[2]
        assert residues[0] > residues[1] > residues[2]
        # Diagnostic 1/N shape of the gap error, not a proven rate.
        scaled = [case["N"] * abs(case["gap_error"]) for case in cases]
        assert max(scaled) / min(scaled) < 1.5


def test_fsum_identity_is_exact(ladder):
    for cases in ladder.values():
        for case in cases:
            assert abs(case["fsum_residual"]) < 1e-9 * max(1.0, abs(case["fsum_kinetic"]))


def test_fsum_identity_independent_position_basis():
    n, g = 3, 2.0
    basis, H, one = _two_particle_hamiltonian(n, g)
    w, v = np.linalg.eigh(H)
    ground = basis @ v[:, 0]
    sites = _sites(n)
    M = n ** 3
    q = [2 * math.pi / 3, 0.0, 0.0]
    phase = np.array([np.exp(1j * sum(a * b for a, b in zip(q, s))) for s in sites])
    rho = np.kron(np.diag(phase), np.eye(M)) + np.kron(np.eye(M), np.diag(phase))
    raised = basis.T @ (rho @ ground)
    spectral = sum((w[i] - w[0]) * abs(np.vdot(v[:, i], raised)) ** 2 for i in range(len(w)))
    kinetic = 0.0
    for axis in range(3):
        T = np.zeros((M, M))
        for i, s in enumerate(sites):
            t = list(s)
            t[axis] = (t[axis] + 1) % n
            j = sites.index(tuple(t))
            T[i, j] += -1.0
            T[j, i] += -1.0
        T2 = np.kron(T, np.eye(M)) + np.kron(np.eye(M), T)
        kinetic += (1 - math.cos(q[axis])) * float(-(ground @ T2 @ ground))
    assert spectral == pytest.approx(kinetic, rel=1e-10)


def test_five_particle_ladder_continues_to_converge():
    case4 = c.mean_field_case(4, 20.0)
    case5 = c.mean_field_case(5, 20.0)
    assert abs(case5["gap_error"]) < abs(case4["gap_error"])
    assert abs(case5["residue_per_particle"] - case5["bogoliubov_residue"]) < \
        abs(case4["residue_per_particle"] - case4["bogoliubov_residue"])


# ---------------------------------------------------------------------------
# Theorem 5: acoustic window
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [3, 4, 7, 12, 16])
@pytest.mark.parametrize("mu", [0.05, 1.0, 9.0])
@pytest.mark.parametrize("a", [1.0, 0.25])
def test_acoustic_inequalities_hold_for_every_mode(n, mu, a):
    report = c.acoustic_window(n, 1.0, mu, a)
    for margin in report["minimum_margins"].values():
        assert margin >= -1e-10


def test_window_requirement_example():
    req = c.lorentz_window_requirements(0.01)
    assert req["n_min"] == 19
    assert (math.pi ** 2 / 3) / 19 ** 2 <= 0.01 < (math.pi ** 2 / 3) / 18 ** 2
    report = c.acoustic_window(19, 1.0, req["mu_min_at_n_min"])
    lowest = 2 * math.pi / 19
    e = math.sqrt(4 * math.sin(lowest / 2) ** 2 * (4 * math.sin(lowest / 2) ** 2 + 2 * req["mu_min_at_n_min"]))
    assert abs(e / (report["c_s"] * lowest) - 1) <= 0.01 + 1e-12


# ---------------------------------------------------------------------------
# Theorems 6-7: moving condensates
# ---------------------------------------------------------------------------

CURRENT_CASES = [(n, m_p, mu) for n in (3, 4, 5, 6)
                 for m_p in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1), (n // 2, 0, 0))
                 for mu in (0.0, 0.3, 2.0)]


@pytest.mark.parametrize("n,m_p,mu", CURRENT_CASES)
def test_bdg_formula_matches_full_generator(n, m_p, mu):
    report = c.moving_condensate(n, m_p, mu)
    formula = []
    for (r1, r2), (i1, i2) in zip(report["frequency_real_parts"], report["frequency_imaginary_parts"]):
        formula.extend([complex(r1, i1), complex(r2, i2)])
    formula.extend([0.0, 0.0])
    numerical = np.linalg.eigvals(_bdg_oracle(n, m_p, mu))
    assert _match(formula, list(numerical)) < 1e-5


@pytest.mark.parametrize("n,m_p,mu", CURRENT_CASES)
def test_stability_and_signature_equivalences(n, m_p, mu):
    report = c.moving_condensate(n, m_p, mu)
    cosines = [math.cos(2 * math.pi * x / n) for x in m_p]
    if report["energetic_stability"] == "stable":
        assert report["dynamical_stability"] == "stable"
    if mu > 0 and all(x > 1e-9 for x in cosines):
        assert report["signature"] == "lorentzian"
        assert report["dynamical_stability"] == "stable"
    if mu > 0 and any(x < -1e-9 for x in cosines):
        assert report["signature"] == "non_lorentzian"
        worst = min(cosines)
        if 2 * abs(worst) * (1 - math.cos(2 * math.pi / n)) < 2 * mu:
            assert report["dynamical_stability"] == "unstable"
    if m_p == (0, 0, 0) and mu > 0:
        assert report["energetic_stability"] == "stable"
        assert report["ergoregion"] == "absent"


def test_metric_is_inverse_and_cone_is_long_wave_limit():
    for n, m_p, mu in [(8, (1, 0, 0), 1.0), (10, (1, 2, 0), 0.4), (12, (2, 1, 1), 3.0)]:
        report = c.moving_condensate(n, m_p, mu)
        assert report["signature"] == "lorentzian"
        upper = np.array(report["inverse_metric"])
        lower = np.zeros((4, 4))
        lower[0, 0] = report["metric"]["g00"]
        lower[0, 1:] = lower[1:, 0] = report["metric"]["g0i"]
        lower[1:, 1:] = np.diag(report["metric"]["gij_diagonal"])
        assert np.max(np.abs(upper @ lower - np.eye(4))) < 1e-10
        eig = np.linalg.eigvalsh(upper)
        assert (eig < 0).sum() == 1 and (eig > 0).sum() == 3
        p = [2 * math.pi * x / n for x in m_p]
        v = np.array(report["velocity"])
        W = np.diag(report["W_diagonal"])
        rng = np.random.default_rng(7)
        for _ in range(5):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            for t in (1e-3, 1e-4):
                k = t * direction
                S = 2 * sum(math.cos(p[j]) * (1 - math.cos(k[j])) for j in range(3))
                D = 2 * sum(math.sin(p[j]) * math.sin(k[j]) for j in range(3))
                omega = D + math.sqrt(S * (S + 2 * mu))
                cone = (omega - v @ k) ** 2 / (k @ W @ k)
                assert cone == pytest.approx(1.0, rel=20 * t)
        # Cauchy-Schwarz: sup_k (v.k)^2/(k W k) = v W^{-1} v, attained at k = W^{-1} v.
        best = np.linalg.solve(W, v)
        assert (v @ best) ** 2 / (best @ W @ best) == pytest.approx(report["v_Winv_v"], rel=1e-12)
        assert (report["ergoregion"] == "absent") == (report["metric"]["g00"] < 0)


def test_supercritical_flow_is_landau_unstable_on_fine_lattice():
    report = c.moving_condensate(32, (2, 0, 0), 0.02)
    assert report["signature"] == "lorentzian"
    assert report["ergoregion"] == "present"
    assert report["energetic_stability"] == "unstable"
    assert report["dynamical_stability"] == "stable"


# ---------------------------------------------------------------------------
# Report contract
# ---------------------------------------------------------------------------

def test_demonstration_report_is_strict_json_and_scoped():
    report = c.demonstration_report((2, 3))
    text = json.dumps(report, allow_nan=False)
    assert json.loads(text) == report
    assert report["empirical_validation"] is False
    assert report["thermodynamic_limit"] is False
    assert report["status"] == "mean_field_condensate_regime_demonstrator"
    assert report["limitations"] == c.LIMITATIONS
    assert len(report["sector_vacuum"]) == 9
    assert len(report["mean_field_ladder"]) == 4


@pytest.mark.parametrize("bad", [True, 2.0, "3", None])
def test_inputs_are_type_checked(bad):
    with pytest.raises((TypeError, ValueError)):
        c.position_sector(bad, 2, 1.0)
    with pytest.raises((TypeError, ValueError)):
        c.demonstration_report((bad,))
