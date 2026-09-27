"""Checks for doc/derivations/brane_positions_2026-09-27.md (Phase 1d).

Oracles: the spin-weighted harmonics and eth (lowest level), the known spin-2 tetrahedral state
(|2,2> + sqrt2 |2,-1>)/sqrt3, direct evaluation of the profile at the polynomial roots, a Burnside
irreducibility test, and the independent Yukawa-rank computation.
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


def test_quartic_minimum_is_unique_and_tetrahedral(minimum):
    c, values = minimum
    assert max(values) - min(values) < 1e-9  # every start reaches the same minimum
    assert b.tetrahedron_test(b.zeros(c)) < 1e-6
    for energy in b.alternative_configurations().values():
        assert energy > min(values) + 1e-3


def test_known_tetrahedral_state_is_the_minimum(minimum):
    _, values = minimum
    tet = np.array([1, 0, 0, np.sqrt(2), 0], complex) / np.sqrt(3)  # (|2,2> + sqrt2 |2,-1>)/sqrt3, m = 2..-2
    assert b.quartic_energy(tet) == pytest.approx(min(values), rel=1e-10)
    assert b.tetrahedron_test(b.zeros(tet)) < 1e-12


def test_stability_three_rotations_and_five_massive_modes(minimum):
    c, _ = minimum
    eig = b.quartic_hessian(c)
    assert np.sum(abs(eig) < 1e-6) == 3
    assert np.all(eig[abs(eig) >= 1e-6] > 0.1)
    stiff = b.pinning_stiffness(c, b.zeros(c)[0])
    assert np.all(stiff > 0.1) and stiff[1] / stiff[0] == pytest.approx(1.0, rel=1e-6)


def test_residual_family_symmetry_is_A4_times_Z4(minimum):
    c, _ = minimum
    group = b.residual_family_group(c)
    assert group["order_on_families"] == 48 and group["projective_order"] == 12
    assert group["families_irreducible"]
    assert sorted(set(round(x, 6) for x in group["rotation_phases"])) == [-0.333333, 0.0, 0.333333]


def test_fixed_tetrahedral_positions_still_reach_generic_yukawas(minimum):
    c, _ = minimum
    out = b.tetrahedral_yukawa_rank(b.zeros(c))
    assert out["span"] == 4 and out["rank_mod_U3"] == 24


def test_scale_window():
    w = b.scale_window()
    assert w["window_open"] and w["vr_min"] < 1 < w["vr_max"]
    assert not b.scale_window(kappa=1e-3)["window_open"]  # weak pinning loses to Casimir forces
    assert w["modulus_mass_times_r_at_vr_1"] > 0.1  # moduli near the compactification scale


def test_report():
    report = b.demonstration_report()
    json.loads(json.dumps(report, default=str))
    assert report["empirical_validation"] is False and report["limitations"] == b.LIMITATIONS
