"""Checks for doc/derivations/brane_positions_2026-09-27.md (Phase 1d).

Oracles: the spin-weighted harmonics and eth (lowest level), the known spin-2 tetrahedral state
(|2,2> + sqrt2 |2,-1>)/sqrt3, the exact values found by the independent review via Clebsch-Gordan algebra
(minimum 25/(84 pi), square 5/(14 pi), coherent 25/(36 pi), Hessian 10/(21 pi) and 100/(63 pi), stiffness 5/(3 pi),
barrier 5/(16 pi)), direct evaluation of the profile at the polynomial roots, a Burnside irreducibility test, and
reachability with a constructed positive control.
"""

import json

import numpy as np
import pytest

import bpr.brane_stabilization as b


@pytest.fixture(scope="module")
def minimum():
    c, values = b.minimize_quartic(starts=40)
    return c, values


def test_lowest_level_is_a_spin_two_multiplet_with_four_zeros():
    level = b.lowest_level_checks()
    assert level["eth_annihilates"] and level["level"] == 2 and level["states"] == 5
    rng = np.random.default_rng(0)
    for _ in range(5):
        c = rng.normal(size=5) + 1j * rng.normal(size=5)
        pts = b.zeros(c)
        assert len(pts) == 4
        for p in pts:  # the polynomial roots really are zeros of the profile
            assert abs(b.profile(c, np.arccos(np.clip(p[2], -1, 1)), np.arctan2(p[1], p[0]))) < 1e-9 * np.linalg.norm(c)


def test_zeros_at_the_south_pole():
    c = np.zeros(5, complex)
    c[-1] = 1.0  # m = -2 only: all four zeros at one pole
    pts = b.zeros(c)
    assert len(pts) == 4 and all(abs(abs(p[2]) - 1) < 1e-9 for p in pts)
    c = np.array([0, 1.0, 0, 0, 0], complex)  # zeros split between the poles
    pts = b.zeros(c)
    assert len(pts) == 4 and all(abs(abs(p[2]) - 1) < 1e-6 for p in pts)


def test_quartic_minimum_is_unique_and_tetrahedral(minimum):
    c, values = minimum
    assert max(values) - min(values) < 1e-9  # every start reaches the same minimum
    assert min(values) == pytest.approx(25 / (84 * np.pi), rel=1e-10)
    assert b.tetrahedron_test(b.zeros(c)) < 1e-6
    alt = b.alternative_configurations()
    assert alt["equatorial square"] == pytest.approx(5 / (14 * np.pi), rel=1e-10)
    assert alt["double zeros at both poles"] == pytest.approx(5 / (14 * np.pi), rel=1e-10)
    assert alt["coherent (one quadruple zero)"] == pytest.approx(25 / (36 * np.pi), rel=1e-10)


def test_known_tetrahedral_state_is_the_minimum(minimum):
    _, values = minimum
    tet = np.array([1, 0, 0, np.sqrt(2), 0], complex) / np.sqrt(3)  # (|2,2> + sqrt2 |2,-1>)/sqrt3, m = 2..-2
    assert b.quartic_energy(tet) == pytest.approx(min(values), rel=1e-10)
    assert b.tetrahedron_test(b.zeros(tet)) < 1e-12


def test_stability_three_rotations_and_five_massive_modes(minimum):
    c, _ = minimum
    eig = np.sort(b.quartic_hessian(c))
    assert np.all(abs(eig[:3]) < 1e-6)
    assert np.allclose(eig[3:5], 10 / (21 * np.pi), rtol=1e-5) and np.allclose(eig[5:], 100 / (63 * np.pi), rtol=1e-5)
    stiff = b.pinning_stiffness(c, b.zeros(c)[0])
    assert np.allclose(stiff, 5 / (3 * np.pi), rtol=1e-5)  # isotropy is automatic at a simple holomorphic zero


def test_pinning_is_only_metastable(minimum):
    c, _ = minimum
    assert b.pinning_barrier(c) == pytest.approx(5 / (16 * np.pi), rel=1e-6)  # the edge-midpoint saddle
    assert b.stacking_pinning_energy(c) < 1e-20  # two branes on one zero cost no pinning energy


def test_residual_family_symmetry_is_A4_times_Z4(minimum):
    c, _ = minimum
    group = b.residual_family_group(c)
    assert group["order_on_families"] == 48 and group["projective_order"] == 12
    assert group["families_irreducible"]
    assert sorted(set(round(x, 6) for x in group["rotation_phases"])) == [-0.333333, 0.0, 0.333333]


def test_fixed_tetrahedral_positions_constrain_the_yukawas(minimum):
    c, _ = minimum
    pts = b.zeros(c)
    out = b.tetrahedral_yukawa_rank(pts)
    assert out["span"] == 4 and out["rank_mod_U3"] == 24  # the image contains an open set ...
    # ... but is a proper subset: Phase 1's hierarchical example is unreachable (the minimum over U is ~5e-4, so any
    # search returns at least that), while a pair constructed from the tetrahedral branes is reached exactly.
    assert b.reachability_cost(*_phase1_pair(), pts, starts=10) > 1e-4
    rng = np.random.default_rng(3)
    zs = [np.tan(np.arccos(np.clip(v[2], -1, 1)) / 2) * np.exp(1j * np.arctan2(v[1], v[0])) for v in pts]
    from scipy.linalg import expm
    import bpr.minimal_model as mm
    U0 = expm(sum(x * g for x, g in zip(rng.normal(size=9), mm._U3)))
    A0 = sum(z * mm.brane_matrix(zz) for z, zz in zip(rng.normal(size=4) + 1j * rng.normal(size=4), zs))
    B0 = sum(z * mm.brane_matrix(zz) for z, zz in zip(rng.normal(size=4) + 1j * rng.normal(size=4), zs))
    Ui = U0.conj().T
    assert b.reachability_cost(Ui.T @ A0 @ Ui, Ui.T @ B0 @ Ui, pts, starts=20) < 1e-12


def _phase1_pair():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    return np.diag([1e-5, 3e-3, 1.0]).astype(complex), 0.02 * (A + A.T)


def test_scale_window_is_narrow_once_higgs_brane_forces_are_included():
    w = b.scale_window()
    assert w["window_open_casimir_only"] and w["vr_min_casimir_only"] < 0.3
    # With the crude Higgs brane-term estimate the lower end moves up to ~1.75, close to vr_max ~ 2.45.
    assert 1.5 < w["vr_min_with_higgs_brane_terms"] < w["vr_max"] < 2.5
    assert not b.scale_window(kappa=0.5)["window_open_with_higgs_brane_terms"]
    assert w["flux_bound_vr"] == pytest.approx(1 / (np.sqrt(2) * 0.03 * 4 / 3))
    assert b.type_ii()["type_II"] and not b.type_ii(lam=1e-4)["type_II"]


def test_report():
    report = b.demonstration_report()
    json.loads(json.dumps(report, default=str))
    assert report["empirical_validation"] is False and report["limitations"] == b.LIMITATIONS
