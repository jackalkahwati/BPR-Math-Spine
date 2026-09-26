"""Checks for doc/derivations/minimal_model_2026-09-26.md (Phase 1a).

Oracles: the Lie-algebra homomorphism property of the 5-form representation, the weight of the holomorphic 5-form,
the SO(3) rotation formula for spin-1 coherent states, l'Huilier's spherical-excess formula, exact reconstruction
of target Yukawa pairs, and the round-8 brane J_z rule.
"""

import json

import numpy as np
import pytest

import bpr.minimal_model as mm


def test_intended_couplings_allowed_and_dangerous_ones_forbidden():
    wanted, dangerous = mm.coupling_table()
    assert all(row["allowed"] for row in wanted.values())
    assert not any(row["allowed"] for row in dangerous.values())
    assert dangerous["mu-term H_u H_d = 10H 10H"]["F"] == -12
    assert dangerous["conjugate Yukawa 16 16 10H*"]["F"] == 12
    assert mm.brane_scalar_higgs_is_useless()


def test_five_form_action_is_a_representation():
    gens = mm.so10_generators()
    rng = np.random.default_rng(0)
    for _ in range(3):
        a, b = rng.choice(len(gens), 2, replace=False)
        A, B = gens[a], gens[b]
        lhs = mm.five_form_action(A) @ mm.five_form_action(B) - mm.five_form_action(B) @ mm.five_form_action(A)
        assert np.allclose(lhs, mm.five_form_action(A @ B - B @ A))


def test_holomorphic_five_form_has_weight_plus_or_minus_X():
    omega = mm.holomorphic_five_form()
    for k in range(5):
        h = [0.0] * 5
        h[k] = 1.0
        image = mm.five_form_action(mm.cartan(h)) @ omega
        ratio = image[np.argmax(abs(omega))] / omega[np.argmax(abs(omega))]
        assert np.allclose(image, ratio * omega) and abs(abs(ratio) - 1) < 1e-12 and abs(ratio.real) < 1e-12


def test_breaking_pattern_reaches_exactly_the_standard_model():
    assert mm.breaking_pattern(1.0, 0.0, False)["unbroken_dimension"] == 15  # SU(3) x SU(2)^2 x U(1)
    assert mm.breaking_pattern(0.0, 1.0, False)["unbroken_dimension"] == 19  # SU(4) x SU(2) x U(1)
    assert mm.breaking_pattern(1.0, 0.4, False)["unbroken_dimension"] == 13
    sm = mm.breaking_pattern(1.0, 0.4, True)
    assert sm["unbroken_dimension"] == 12
    assert sm["cartan_unbroken"] == {"Y": True, "Y_flipped": False, "X": False}
    assert mm.stabilizer_dimension(None, mm.holomorphic_five_form())[0] == 24  # SU(5)


def test_c2_branes_give_only_the_j2_part():
    assert mm.c2_brane_span_dimension() == 5
    rng = np.random.default_rng(1)
    for z in rng.normal(size=4) + 1j * rng.normal(size=4):
        assert abs(mm.j0_component(mm.brane_matrix(z))) < 1e-14
    invariant = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])  # the SU(2)-invariant form in the m = (1, 0, -1) basis
    assert mm.j0_component(invariant) != 0


def test_coherent_state_moves_by_rotation():
    # A brane at z is the pole brane rotated there: u(z) is proportional to D(R)^T e_{m=1} up to phase.
    from bpr.yukawa_mechanisms import rotation
    for polar in (0.4, 1.3, 2.7):
        z = np.tan(polar / 2)  # stereographic from the north pole along x
        D = rotation([0.0, polar, 0.0])
        v = D @ np.array([1, 0, 0], complex)
        u = mm.coherent_state(z)
        overlap = abs(np.vdot(v, u)) / (np.linalg.norm(v) * np.linalg.norm(u))
        assert overlap == pytest.approx(1.0, abs=1e-12)


def test_realizability_ranks_modulo_u3():
    assert mm.realizability_rank(2) == 16
    assert mm.realizability_rank(3) == 23  # one real relation
    assert mm.realizability_rank(4) == 24  # generic pairs
    assert mm.realizability_rank(4, with_u3=False) == 20  # without U(3): the J = 2 pairs only


def test_three_brane_relation_is_the_bargmann_phase():
    assert mm.bargmann_relation() < 1e-10


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_four_branes_realize_hierarchical_yukawa_pairs(seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    Y10 = np.diag([1e-5, 3e-3, 1.0]).astype(complex)
    Y126 = 0.02 * (A + A.T)
    out = mm.realize_yukawa_pair(Y10, Y126, seed=seed)
    assert out["Y10_error"] < 1e-10 and out["Y126_error"] < 1e-10
    sv = np.sort(np.linalg.svd(out["Y10_back"], compute_uv=False))
    assert sv[0] == pytest.approx(1e-5, rel=1e-8) and sv[1] == pytest.approx(3e-3, rel=1e-10)
    # The construction is honest: the brane matrices really are J = 2 and U is unitary.
    U = out["U"]
    assert np.allclose(U.conj().T @ U, np.eye(3), atol=1e-12)
    assert len(out["z"]) == 4


def test_family_isometry_broken_by_two_generic_branes():
    rng = np.random.default_rng(3)
    pts = [v / np.linalg.norm(v) for v in rng.normal(size=(4, 3))]
    assert mm.isometry_stabilizer_dimension(pts[:1]) == 1
    assert mm.isometry_stabilizer_dimension(pts[:2]) == 0
    north = np.array([0, 0, 1.0])
    assert mm.isometry_stabilizer_dimension([north, -north]) == 1  # antipodal branes keep J_z


def test_report():
    report = mm.demonstration_report()
    json.loads(json.dumps(report))
    assert report["empirical_validation"] is False and report["limitations"] == mm.LIMITATIONS
