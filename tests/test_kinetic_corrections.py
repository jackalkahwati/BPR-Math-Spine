"""Checks for doc/derivations/kinetic_corrections_2026-10-01.md (Phase 2b).

Oracles: orthonormality of the flux-sphere zero modes (numerical quadrature), the tight-frame identity of the
tetrahedral coherent states, Schur's lemma for the A4 action (the rotations permute the brane states), the exact
action of a single-brane kinetic term on its own brane matrix, invariance of the SO(10) relations, and recomputation
of the stored scan points from scratch.
"""

import json

import numpy as np
import pytest

import bpr.fermion_fit as ff
import bpr.kinetic_corrections as kc


@pytest.fixture(scope="module")
def tg():
    return ff.targets(2e16)


def test_zero_modes_are_orthonormal_and_their_brane_value_is_the_coherent_state():
    G, density_dev = kc.zero_mode_gram()
    assert np.allclose(G, np.eye(3), atol=1e-4)  # quadrature accuracy
    assert density_dev < 1e-12  # sum_m |f_m|^2 = 3/(4 pi r^2) everywhere, so psi(z_a) = sqrt(3/(4 pi r^2)) u_a^T psi
    for u in kc.brane_states():
        assert np.linalg.norm(u) == pytest.approx(1.0)


def test_equal_brane_kinetic_terms_do_nothing():
    u = kc.brane_states()
    assert np.allclose(sum(np.outer(a, a.conj()) for a in u), 4 / 3 * np.eye(3))  # a tight frame
    m = ff.pinned_matrices(ff.BEST_FIT_PINNED)
    n = kc.normalized(m, kc.bkt_kinetic_matrix([0.2] * 4))
    for key in ("Mu", "Md", "Me", "MD", "F"):
        assert np.allclose(n[key], m[key] / (1 + 4 * 0.2 / 3))  # a common rescaling, absorbed by the couplings


def test_a4_symmetric_kinetic_matrices_are_trivial():
    """The 12 tetrahedral rotations permute the brane states, so any A4-invariant K is a multiple of 1 (Schur)."""
    from bpr.brane_stabilization import tetrahedral_rotations
    from bpr.minimal_model import spin_matrices
    _, pts = ff.tetrahedral_brane_matrices()
    u = kc.brane_states()
    Ds = kc._spin1_rotations(tetrahedral_rotations(pts), spin_matrices)
    assert len(Ds) == 12
    for D in Ds:
        for a in u:
            assert max(abs(np.vdot(b, D @ a)) for b in u) == pytest.approx(1.0, abs=1e-12)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    K = X @ X.conj().T
    assert np.allclose(kc.a4_average(K), np.trace(K).real / 3 * np.eye(3), atol=1e-12)


def test_a_single_brane_kinetic_term_dilutes_its_own_yukawa_through_normalized():
    """Through normalized() and at a brane with complex z, so the Kbar-versus-K convention is discriminated."""
    u = kc.brane_states()
    P, _ = ff.tetrahedral_brane_matrices()
    assert np.max(abs(u[0].imag)) > 0.1
    eps = 0.3
    m = {"H": P[0], "F": 0.5 * P[0], "r": 1.0, "s": 0.3}
    n = kc.normalized(m, kc.bkt_kinetic_matrix([eps, 0, 0, 0], u))
    assert np.allclose(n["H"], P[0] / (1 + eps)) and np.allclose(n["F"], 0.5 * P[0] / (1 + eps))
    wrong = kc._inverse_sqrt(kc.bkt_kinetic_matrix([eps, 0, 0, 0], u).conj())  # the other convention fails
    assert not np.allclose(wrong @ P[0] @ wrong.T, P[0] / (1 + eps))
    # The normalized matrices keep the SO(10) relations and stay symmetric.
    m = kc.normalized(ff.pinned_matrices(ff.BEST_FIT_PINNED), kc.bkt_kinetic_matrix([0.1, -0.2, 0.05, 0.3], u))
    assert np.allclose(m["Md"], m["H"] + m["F"]) and np.allclose(m["Me"], m["H"] - 3 * m["F"])
    assert np.allclose(m["Mu"], m["r"] * (m["H"] + m["s"] * m["F"]))
    for key in ("Mu", "Md", "Me", "MD", "F"):
        assert np.allclose(m[key], m[key].T)


def test_brane_kinetic_terms_are_a_special_general_kinetic_matrix():
    from scipy.linalg import logm
    Kb = kc.bkt_kinetic_matrix([0.1, -0.2, 0.05, 0.3])
    det = np.linalg.det(Kb).real
    h = logm(Kb / det ** (1 / 3))
    t = [np.trace(h @ L).real for L in kc.gell_mann()]
    m = ff.pinned_matrices(ff.BEST_FIT_PINNED)
    a, b = kc.normalized(m, Kb), kc.normalized(m, kc.general_kinetic_matrix(t))
    assert np.allclose(a["H"], b["H"] / det ** (1 / 3)) and np.allclose(a["F"], b["F"] / det ** (1 / 3))


def test_an_order_one_kinetic_matrix_erases_pinning(tg):
    """Positive control: any four-brane realization maps onto the tetrahedron with a GL(3) kinetic transformation,
    so the generic best fit is reproduced exactly by the pinned model plus an O(1) kinetic matrix."""
    x, size = kc.kinetic_map_to_tetrahedron(ff.build(ff.BEST_FIT_GENERIC, tg), seed=1)
    assert kc.corrected_chi2(x, "general_K", tg) == pytest.approx(ff.chi2(ff.BEST_FIT_GENERIC, tg), rel=1e-6)
    assert size == pytest.approx(kc.correction_size("general_K", x)) and size > 10 * kc.NATURAL_BOUND["general_K"]


def test_derivative_vertex_structure(tg):
    """The one-derivative vertex: w_a = D(R_a) e_0 is orthogonal to u_a = D(R_a) e_1; with delta = 0 the pinned model
    is recovered. S_a is pure J = 2 like P_a but is not annihilated by the condensate, so with delta unbounded the P_a
    and S_a span the whole J = 2 space: the free-position model of Phase 1. Only the size of delta constrains."""
    from bpr.minimal_model import wigner, basis_vector
    _, pts = ff.tetrahedral_brane_matrices()
    u, w = kc.brane_states(), kc.derivative_states()
    for a, b, n in zip(u, w, pts):
        D = wigner(1, np.arccos(np.clip(n[2], -1, 1)), np.arctan2(n[1], n[0]))
        assert abs(abs(np.vdot(D @ basis_vector(1, 1), a)) - 1) < 1e-12 and abs(np.vdot(a, b)) < 1e-12
    x = np.r_[ff.BEST_FIT_PINNED, np.zeros(16)]
    assert kc.corrected_chi2(x, "vertex", tg) == pytest.approx(ff.pinned_chi2(ff.BEST_FIT_PINNED, tg), rel=1e-12)
    P, S = kc._vertex_basis()
    span = np.array([M[np.triu_indices(3)] for M in list(P) + list(S)])
    assert np.linalg.matrix_rank(span, tol=1e-10) == 5  # all of J = 2
    for M in S:
        j0, chi = ff.w_conditions(M)
        assert abs(j0) < 1e-12 and abs(chi) > 1e-3  # pure J = 2, but outside W
    nat = kc.vertex_epsilon_estimate(3.9), kc.vertex_epsilon_estimate(3.0, 2.24)
    assert 0.06 < nat[0] < 0.07 and 0.24 < nat[1] < 0.26


def test_natural_size_of_brane_kinetic_terms():
    nat = kc.natural_epsilon_range()
    assert nat["rM_range"][0] == pytest.approx(3.0)
    lo, hi = nat["eps_range"]
    assert 0.01 < lo < hi < 0.03
    assert kc.epsilon_from_bkt(1.0, 3.0) == pytest.approx(3 / (4 * np.pi * 9))


def test_zero_corrections_reproduce_the_phase_2_pinned_fits(tg):
    for kind, n in kc.KINDS.items():
        x = np.r_[ff.BEST_FIT_PINNED, np.zeros(n)]
        assert kc.corrected_chi2(x, kind, tg) == pytest.approx(ff.pinned_chi2(ff.BEST_FIT_PINNED, tg), rel=1e-12)
        xc = np.r_[ff.BEST_FIT_PINNED_CHARGED, np.zeros(n)]
        assert kc.corrected_chi2(xc, kind, tg, with_nu=False) == pytest.approx(
            ff.pinned_chi2(ff.BEST_FIT_PINNED_CHARGED, tg, False), rel=1e-12)


def test_stored_scan_points_are_recomputed_and_respect_their_bounds(tg):
    for (kind, setting, bound), (c, x) in kc.STORED_POINTS.items():
        x = np.asarray(x)
        assert kc.within_bounds(kind, bound, x)
        assert kc.corrected_chi2(x, kind, tg, with_nu=(setting == "nu")) == pytest.approx(c, rel=1e-6, abs=1e-8)


def test_verdict_from_recomputed_points(tg):
    """Normalization corrections at natural size leave the pinned model far from the data; the derivative vertex at
    its largest scanned size fits. All numbers recomputed from the stored points (upper bounds on chi^2_min)."""
    P = kc.STORED_POINTS
    for kind in ("bkt", "general_K", "displacement", "combined"):
        b = kc.NATURAL_BOUND[kind]
        assert kc.corrected_chi2(np.asarray(P[(kind, "charged", b)][1]), kind, tg, False) > 30
        assert kc.corrected_chi2(np.asarray(P[(kind, "nu", b)][1]), kind, tg, True) > 80
    vmax = max(b for (k, s, b) in P if k == "vertex" and s == "nu")
    assert kc.corrected_chi2(np.asarray(P[("vertex", "nu", vmax)][1]), "vertex", tg, True) <= 17
    assert kc.within_bounds("vertex", vmax, P[("vertex", "nu", vmax)][1]) and vmax <= 0.3


def test_stored_natural_points_are_local_minima(tg):
    from scipy.optimize import least_squares
    for key in (("general_K", "charged", 0.05), ("vertex", "charged", 0.1)):
        kind, setting, b = key
        x = np.asarray(kc.STORED_POINTS[key][1])
        lo, hi = kc._bounds(kind, b)
        c0 = kc.corrected_chi2(x, kind, tg, setting == "nu")
        sol = least_squares(kc.corrected_residuals, np.clip(x, lo, hi), args=(kind, tg, setting == "nu"),
                            bounds=(lo, hi), max_nfev=200, x_scale="jac")
        assert c0 - 2 * sol.cost < 0.05 * c0 + 0.1


def test_report():
    report = kc.demonstration_report()
    json.loads(json.dumps(report, default=str))
    assert report["empirical_validation"] is False and report["limitations"] == kc.LIMITATIONS
