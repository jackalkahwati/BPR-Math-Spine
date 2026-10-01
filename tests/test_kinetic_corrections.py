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


def test_a_single_brane_kinetic_term_dilutes_its_own_yukawa():
    u = kc.brane_states()
    P, _ = ff.tetrahedral_brane_matrices()
    eps = 0.3
    A = kc._inverse_sqrt(kc.bkt_kinetic_matrix([eps, 0, 0, 0], u))
    assert np.allclose(A @ P[0] @ A.T, P[0] / (1 + eps))
    # The normalized matrices keep the SO(10) relations and stay symmetric.
    m = kc.normalized(ff.pinned_matrices(ff.BEST_FIT_PINNED), kc.bkt_kinetic_matrix([0.1, -0.2, 0.05, 0.3], u))
    assert np.allclose(m["Md"], m["H"] + m["F"]) and np.allclose(m["Me"], m["H"] - 3 * m["F"])
    assert np.allclose(m["Mu"], m["r"] * (m["H"] + m["s"] * m["F"]))
    for key in ("Mu", "Md", "Me", "MD", "F"):
        assert np.allclose(m[key], m[key].T)


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


def test_natural_size_corrections_do_not_rescue_the_pinned_model():
    nat = kc.natural_epsilon_range()["eps_range"][1]
    scan = dict(kc.SCANS["bkt"]["charged"])
    assert min(c for b, c in scan.items() if b <= nat + 1e-12) > 60  # charged sector still fails at natural size
    for kind in kc.SCANS:
        for setting, rows in kc.SCANS[kind].items():
            chis = [c for _, c in sorted(rows)]
            assert all(b <= a + 1e-6 for a, b in zip(chis, chis[1:]))  # larger bounds never fit worse (homotopy)


def test_report():
    report = kc.demonstration_report()
    json.loads(json.dumps(report, default=str))
    assert report["empirical_validation"] is False and report["limitations"] == kc.LIMITATIONS
