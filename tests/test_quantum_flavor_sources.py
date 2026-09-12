"""Independent operator, symmetry and strict-CLI regressions for the toy ansatz."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bpr.chiral_flavor_prototype import (
    HARMONIC_DEGREES, monopole_profiles, overlap_matrix, real_harmonics, sphere_quadrature,
)
from bpr.flavor_source_selection import (
    SelectionParameters, coherent_occupation, density_coefficients, equal_occupation,
    response_weights,
)
from bpr import quantum_flavor_sources as q

ROOT = Path(__file__).resolve().parents[1]


def assert_close(actual, expected, atol=2e-13):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=2e-13)


def test_canonical_anticommutators_all_modes_and_cached_array_isolation():
    c = q.annihilation_operators()
    for i in range(6):
        for j in range(6):
            assert_close(c[i]@c[j]+c[j]@c[i], 0)
            assert_close(c[i]@c[j].conj().T+c[j].conj().T@c[i], np.eye(64) if i == j else 0)
    assert c[3, 1, 9] == -1  # interspecies Jordan-Wigner sign
    c[:] = 99
    assert q.annihilation_operators()[3, 1, 9] == -1


@pytest.mark.parametrize("radius", [0.7, 1., 2.3])
def test_algebraic_overlap_against_independent_quadrature(radius):
    algebraic = q.overlap_operators()
    # Direct profiles+harmonic integral, and independently exposed prototype API.
    theta, phi, weights = sphere_quadrature(radius, 19, 41)
    psi = monopole_profiles(theta, phi, R=radius)
    quadrature = np.einsum("n,ni,nA,nj->Aij", weights, psi.conj(), real_harmonics(theta, phi), psi)
    assert_close(algebraic, quadrature)
    assert_close(algebraic, [overlap_matrix(row, R=radius, n_polar=23, n_azimuth=47) for row in np.eye(9)])


def test_spin_conventions_casimirs_and_invariant_contractions():
    j = q.spin_one_generators()
    assert_close(j[0]@j[1]-j[1]@j[0], 1j*j[2])
    assert_close(sum(matrix@matrix for matrix in j), 2*np.eye(3))
    matrices = q.overlap_operators()
    for ell in range(3):
        subset = matrices[HARMONIC_DEGREES == ell]
        assert_close(sum(t@t for t in subset), q.CASIMIR_COEFFICIENTS[ell]*np.eye(3))
        assert_close(sum(np.kron(t, t) for t in subset), q.invariant_contractions()[ell])
    assert q.overlap_operators()[2, 0, 1].imag > 0  # K_y=-J_y


def test_all_number_blocks_partition_and_dimensions():
    numbers = q.number_operators()
    visited = []
    from math import comb
    for nu in range(4):
        for nd in range(4):
            indices = q.number_sector_indices(nu, nd)
            visited.extend(indices)
            assert len(indices) == comb(3, nu)*comb(3, nd)
            for s, n in enumerate((nu, nd)):
                assert_close(numbers[s][np.ix_(indices, indices)], n*np.eye(len(indices)))
    assert sorted(visited) == list(range(64))
    assert len(set(q.one_plus_one_indices())) == 9
    assert set(q.one_plus_one_indices()) == set(q.number_sector_indices(1, 1))


def test_complex_density_transpose_convention_and_radius():
    v = np.array([1, 1j, 2-1j], complex)/np.sqrt(7)
    w = np.array([2j, 1, 0], complex)/np.sqrt(5)
    state = q.embed_one_plus_one(np.kron(v, w))
    for species, vector in (("u", v), ("d", w)):
        rho = np.outer(vector, vector.conj())
        result = q.one_body_occupation(state, species)
        assert result["number"] == pytest.approx(1)
        assert_close(result["occupation"], rho)
        assert_close(result["normalized_occupation"], rho)
        for radius in (0.8, 2.7):
            assert_close(q.density_expectation(state, species, radius), density_coefficients(rho, radius))
    # Offdiagonal imaginary density catches transposing the one-body convention.
    assert abs(v[0]*v[1].conjugate()) > 0


def test_vacuum_filled_and_indefinite_population_are_distinct():
    vacuum, filled = np.eye(64)[0], np.eye(64)[63]
    for species in ("u", "d"):
        zero = q.one_body_occupation(vacuum, species)
        assert zero["number"] == 0
        assert zero["normalized_occupation"] is None
        assert_close(zero["occupation"], np.zeros((3, 3)))
        full = q.one_body_occupation(filled, species)
        assert full["number"] == 3
        assert_close(full["occupation"], np.eye(3))
        assert_close(full["normalized_occupation"], np.eye(3)/3)
        values = q.density_expectation(filled, species, 1.7)
        assert_close(values, np.r_[3/(np.sqrt(4*np.pi)*1.7**2), np.zeros(8)])
        mixture = 0.4*np.outer(vacuum, vacuum)+0.6*np.outer(filled, filled)
        assert_close(q.one_body_occupation(mixture, species)["occupation"], 0.6*np.eye(3))
    # Actual spatial density is constant; no three-family interpretation.
    theta, phi, _ = sphere_quadrature(1.7, 9, 21)
    psi = monopole_profiles(theta, phi, R=1.7)
    assert_close(np.sum(abs(psi)**2, axis=1), 3/(4*np.pi*1.7**2))


@pytest.mark.parametrize("species", ["u", "d"])
def test_direct_quartic_ordering_identity(species):
    for t in q.overlap_operators():
        bilinear = q.second_quantization(t, species)
        direct = q.normal_ordered_square(t, species)
        assert_close(bilinear@bilinear, direct+q.second_quantization(t@t, species))
        indices = q.number_sector_indices(1, 1)
        assert_close(direct[np.ix_(indices, indices)], 0)


@pytest.mark.parametrize("parameters", [SelectionParameters(), SelectionParameters(R=1.3, g_u=0.7, g_d=1.2, kappa=0, eta=0.3)])
def test_counterterm_hermiticity_number_and_rotation_conservation(parameters):
    full = q.hamiltonian(parameters)
    normal = q.hamiltonian(parameters, "normal_ordered")
    assert_close(normal-full, q.ordering_counterterm(parameters))
    a, _ = response_weights(parameters)
    expected = (np.dot(a, q.CASIMIR_COEFFICIENTS)/(2*parameters.R**2)
                *np.einsum("s,sij->ij", [parameters.g_u**2, parameters.g_d**2], q.number_operators()))
    assert_close(q.ordering_counterterm(parameters), expected)
    for h in (full, normal):
        assert_close(h, h.conj().T)
        for n in q.number_operators():
            assert_close(h@n-n@h, 0)
        for j in q.spin_one_generators():
            total = q.second_quantization(j, "u")+q.second_quantization(j, "d")
            assert_close(h@total-total@h, 0)
    assert not np.allclose(q.ordering_counterterm(parameters), np.eye(64)*q.ordering_counterterm(parameters)[63,63])


def test_noncommuting_density_caveat_is_executable():
    densities = q.density_operators()
    commutator = densities[0, 1]@densities[0, 2]-densities[0, 2]@densities[0, 1]
    assert np.linalg.norm(commutator, 2) > 0.01
    assert_close(densities[0, 1]@densities[1, 2]-densities[1, 2]@densities[0, 1], 0)


@pytest.mark.parametrize("ordering", q.ORDERINGS)
@pytest.mark.parametrize("parameters", [SelectionParameters(), SelectionParameters(eta=0), SelectionParameters(kappa=0), SelectionParameters(R=1.4, eta=0.19, g_u=0.6, g_d=1.8)])
def test_all_sixteen_spectra_match_independent_oracles(parameters, ordering):
    h = q.hamiltonian(parameters, ordering)
    for nu in range(4):
        for nd in range(4):
            result = q.sector_spectrum(nu, nd, parameters, ordering)
            analytic = q.analytic_sector_spectrum(nu, nd, parameters, ordering)
            expected = np.sort(np.concatenate([np.repeat(ch["energy"], ch["multiplicity"])
                                               for ch in analytic["channels"]]))
            assert_close(np.linalg.eigvalsh(q.restrict_to_sector(h, nu, nd)), expected)
            assert result["dimension"] == sum(ch["multiplicity"] for ch in result["channels"])
            if nu in (1, 2) and nd in (1, 2):
                assert result["ground_dimension"] == (9 if parameters.eta == 0 else 5)
            else:
                assert result["ground_dimension"] == result["dimension"]


def test_hole_quadrupole_reversal_not_dipole_reversal():
    # Independent particle-hole isometry: |m>hole=(-1)^(1-m) c_-m |filled>.
    c = q.annihilation_operators()
    basis = np.eye(64)
    hole = np.column_stack([(-1)**i*c[2-i]@basis[7] for i in range(3)])
    particle = np.column_stack([basis[1 << i] for i in range(3)])
    for t, ell in zip(q.overlap_operators(), HARMONIC_DEGREES):
        lifted = q.second_quantization(t, "u")
        expected = (2 if ell == 0 else 1 if ell == 1 else -1)*(particle.conj().T@lifted@particle)
        assert_close(hole.conj().T@lifted@hole, expected)


def test_projectors_casimir_and_independent_fock_block():
    projectors = q.spin_channel_projectors()
    total = [np.kron(j, np.eye(3))+np.kron(np.eye(3), j) for j in q.spin_one_generators()]
    casimir = sum(j@j for j in total)
    assert_close(sum(projectors.values()), np.eye(9))
    indices = q.one_plus_one_indices()
    block = q.hamiltonian()[np.ix_(indices, indices)]
    solution = q.one_plus_one_solution()
    reconstruction = np.zeros((9, 9), complex)
    for j, projector in projectors.items():
        assert_close(projector, projector.conj().T)
        assert_close(projector@projector, projector)
        assert np.trace(projector).real == pytest.approx(2*j+1)
        assert_close(casimir@projector, j*(j+1)*projector)
        reconstruction += solution["channels"][j]["energy"]*projector
        for k, other in projectors.items():
            if j != k:
                assert_close(projector@other, 0)
    assert_close(block, reconstruction)
    assert_close(solution["ground_projector"], projectors[2])


def test_same_sector_squares_are_constants_not_classical_purity():
    indices = q.one_plus_one_indices()
    densities = q.density_operators()
    for s in range(2):
        for ell in range(3):
            square_sum = sum(t@t for t in densities[s, HARMONIC_DEGREES == ell])
            assert_close(square_sum[np.ix_(indices, indices)], q.CASIMIR_COEFFICIENTS[ell]*np.eye(9))


def test_stable_gaps_exact_zero_control_and_kappa_zero_excited_degeneracy():
    p = SelectionParameters()
    solution = q.one_plus_one_solution(p)
    _, b = response_weights(p)
    assert solution["stable_gaps"]["J0_minus_J2"] == pytest.approx(9*(5*b[1]-b[2])/(80*np.pi))
    assert solution["stable_gaps"]["J1_minus_J2"] == pytest.approx(3*(5*b[1]+b[2])/(40*np.pi))
    zero = q.one_plus_one_solution(SelectionParameters(eta=0))
    assert zero["ground_dimension"] == 9
    assert zero["analytic_ground_channels"] == [0, 1, 2]
    assert zero["numerical_status"] == "exact_degeneracy"
    assert_close(zero["ground_ensemble"], np.eye(9)/9)
    flat = q.one_plus_one_solution(SelectionParameters(kappa=0))
    assert flat["channels"][0]["energy"] == pytest.approx(flat["channels"][1]["energy"])


@pytest.mark.parametrize("eta", [1e-20, 1e-100, 1e-280])
def test_tiny_positive_eta_remains_analytically_distinct_and_common_terms_removed(eta):
    p = SelectionParameters(eta=eta)
    result = q.one_plus_one_solution(p)
    assert result["ground_dimension"] == 5
    assert result["analytic_ground_channels"] == [2]
    assert result["numerical_status"] == "full_energy_unresolved"
    assert result["gap"] > 0
    assert result["stable_gaps"]["J0_minus_J2"] > 0
    assert result["projector_residual"] < result["projector_tolerance"]
    assert_close(result["ground_projector"], q.spin_channel_projectors()[2])


@pytest.mark.parametrize("eta", [1e-20, 1e-100, 1e-280])
def test_normal_ordering_does_not_cancel_tiny_cross_energy_at_zero_h0(eta):
    p = SelectionParameters(eta=eta, h0=0)
    result = q.one_plus_one_solution(p, "normal_ordered")
    _, b = response_weights(p)
    expected_common = -b[0]/(4*np.pi)
    assert result["common_energy"] == pytest.approx(expected_common, rel=2e-13, abs=0)
    assert result["energy"] < expected_common
    assert result["numerical_status"] == "resolved"
    assert result["ground_dimension"] == 5


def test_normal_ordering_keeps_tiny_baseline_and_zero_kappa_self_identity():
    p = SelectionParameters(eta=0, h0=1e-20)
    result = q.one_plus_one_solution(p, "normal_ordered")
    assert result["energy"] == pytest.approx(-2e-20, rel=2e-13, abs=0)
    for species in ("u", "d"):
        direct_sum = sum(q.normal_ordered_square(t, species) for t in q.overlap_operators())
        assert_close(direct_sum, 0)
    # Equal response weights have exactly zero net quartic, also in filled blocks.
    for eta in (0., 1e-20, 0.25):
        parameters = SelectionParameters(kappa=0, eta=eta, h0=0)
        for nu in range(4):
            for nd in range(4):
                result = q.sector_spectrum(nu, nd, parameters, "normal_ordered")
                if nu == 0 or nd == 0:
                    assert result["energy"] == 0


def test_unresolved_positive_kappa_not_silently_identified_with_zero():
    p = SelectionParameters(kappa=1e-20, h0=0)
    for evaluate in (lambda: q.hamiltonian(p, "normal_ordered"),
                     lambda: q.analytic_sector_spectrum(2, 0, p, "normal_ordered")):
        with pytest.raises(ValueError, match="unresolved normal-ordered response"):
            evaluate()


def test_huge_common_h0_does_not_erase_channel_gaps():
    ordinary = q.one_plus_one_solution()
    large = q.one_plus_one_solution(SelectionParameters(h0=1e100))
    assert large["numerical_status"] == "full_energy_unresolved"
    assert large["stable_gaps"] == ordinary["stable_gaps"]


def test_ensemble_coherent_entangled_ground_representatives():
    representatives = q.ground_representatives()
    ensemble = representatives["ensemble"]
    coherent = representatives["chosen_coherent"]
    entangled = representatives["chosen_entangled"]
    for rep in representatives.values():
        assert rep["ground_residual"] < 2e-13
        for species in ("u", "d"):
            assert_close(q.one_body_occupation(q.embed_one_plus_one(rep["state"]), species)["occupation"], rep["rho_"+species])
    assert ensemble["purity"] == pytest.approx(1/5)
    assert_close(ensemble["rho_u"], np.eye(3)/3)
    assert_close(ensemble["rho_d"], np.eye(3)/3)
    assert_close(coherent["rho_u"], np.diag([1, 0, 0]))
    assert_close(entangled["rho_u"], np.diag([1, 4, 1])/6)
    assert coherent["purity"] == pytest.approx(1)
    assert entangled["purity"] == pytest.approx(1)
    assert np.trace(entangled["rho_u"]@entangled["rho_u"]).real < 1
    # Nonfactorizing cross correlation for explicit entangled ground state.
    jz = q.spin_one_generators()[2]
    correlation = np.trace(entangled["state"]@np.kron(jz, jz)).real
    product = np.trace(entangled["rho_u"]@jz)*np.trace(entangled["rho_d"]@jz)
    assert correlation == pytest.approx(-1/3)
    assert abs(correlation-product) > 0.1


@pytest.mark.parametrize("rho_u,rho_d", [(coherent_occupation(), coherent_occupation()),
                                          (coherent_occupation(1.1, 0.8), coherent_occupation(2.2, 1.7)),
                                          (equal_occupation(), np.diag([0.2, 0.3, 0.5]))])
def test_product_variance_identity_and_cross_factorization(rho_u, rho_d):
    p = SelectionParameters(R=1.7, g_u=0.7, g_d=1.2)
    result = q.product_variance_comparison(rho_u, rho_d, p)
    assert abs(result["identity_residual"]) < 2e-13
    assert result["weighted_variance_correction"] > 0
    assert result["quantum_energy"] < result["classical_energy"]
    state = q.embed_one_plus_one(np.kron(rho_u, rho_d))
    operators = q.density_operators()
    for a in range(9):
        correlation = np.trace(state@operators[0, a]@operators[1, a])
        assert_close(correlation, np.trace(rho_u@q.overlap_operators()[a])*np.trace(rho_d@q.overlap_operators()[a]))


@pytest.mark.parametrize("bad", [np.ones(64), np.eye(64), np.eye(64)/63, np.eye(64)*np.nan,
                                 np.diag(np.r_[1.1, -0.1, np.zeros(62)])])
def test_invalid_states_not_normalized_clipped_or_repaired(bad):
    with pytest.raises(ValueError):
        q.one_body_occupation(bad, "u")


@pytest.mark.parametrize("radius", [0, -1, float("nan"), float("inf"), 1e-250, 1e250])
def test_bad_or_unresolved_radius_rejected(radius):
    with pytest.raises(ValueError):
        q.density_operators(radius)


def test_bad_indices_species_ordering_and_underflow_rejected():
    with pytest.raises(TypeError):
        q.number_sector_indices(True, 1)
    with pytest.raises(ValueError):
        q.number_sector_indices(4, 1)
    with pytest.raises(ValueError):
        q.one_body_occupation(np.eye(64)[0], "x")
    with pytest.raises(ValueError):
        q.hamiltonian(ordering="arbitrary")
    with pytest.raises(TypeError):
        q.hamiltonian(parameters={})
    for parameters in (SelectionParameters(g_u=1e-250), SelectionParameters(g_u=1e250),
                       SelectionParameters(eta=np.nextafter(0., 1.))):
        with pytest.raises(ValueError, match="unresolved"):
            q.one_plus_one_solution(parameters)
    state = np.zeros((64, 64))
    state[0, 0], state[1, 1] = 1-1e-15, 1e-15
    with pytest.raises(ValueError, match="unresolved nonzero occupation"):
        q.one_body_occupation(state, "u")


def test_frozen_demonstration_ledger_nulls_and_all_filled_oracle():
    report = q.demonstration()
    json.dumps(report, allow_nan=False)
    assert report["physical_masses"] is None
    assert report["physical_mixing"] is None
    assert len(report["assumptions"]) == 6
    for row in report["assumptions"]:
        assert row["missing_input"] and row["existing_evidence"] and row["toy_assumption"]
        assert row["status"] == "missing microscopic matching"
    assert report["operator_algebra"]["density_commutator_norm"] > 0
    for ordering in q.ORDERINGS:
        table = report["orderings"][ordering]
        assert len(table["sectors"]) == 16
        assert table["lowest_sectors"] == [[3, 3]]
    assert report["orderings"]["full_square"]["sectors"][-1]["energy"] == pytest.approx(-12-3/np.pi)
    a, b = response_weights(SelectionParameters())
    assert_close(a, [16/15, 48/143, 112/783])
    assert_close(b, [4/15, 4/143, 4/783])


def test_parent_demo_text_and_strict_json():
    script = ROOT / "scripts" / "demo_quantum_flavor_sources.py"
    text = subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    assert "conditional" in text.lower()
    raw = subprocess.run([sys.executable, str(script), "--json"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    report = json.loads(raw, parse_constant=lambda value: pytest.fail(f"nonfinite JSON {value}"))
    assert report["physical_masses"] is None and report["physical_mixing"] is None
    assert report["one_plus_one"]["full_square"]["ground_dimension"] == 5
