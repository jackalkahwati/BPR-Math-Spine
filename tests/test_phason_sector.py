"""Tests for the proposed phason sector (Postulate 0c extension, speculative)."""
import numpy as np
import pytest

from bpr.phason_sector import (
    ElasticState,
    TopologicalPhasonDefect,
    elastic_free_energy_density,
    perturbative_phason_force,
    topological_phason_force,
    required_substrate_energy_density,
    RHO_LAMBDA,
    RHO_QFT_PLANCK,
    G_EARTH,
)


def test_elastic_energy_has_three_terms():
    s = ElasticState(grad_u=2.0, grad_w=3.0)
    f = elastic_free_energy_density(s, C=1.0, K=0.05, D=0.1)
    assert f["phonon"] == pytest.approx(0.5 * 1.0 * 4.0)
    assert f["phason"] == pytest.approx(0.5 * 0.05 * 9.0)
    assert f["coupling"] == pytest.approx(0.1 * 2.0 * 3.0)
    assert f["total"] == pytest.approx(f["phonon"] + f["phason"] + f["coupling"])


def test_phason_is_softer_so_perturbative_force_is_smaller():
    phonon = 1e-17
    assert perturbative_phason_force(phonon) < phonon  # the wall: NOT a win


def test_topological_force_is_dimensionally_force_and_lambda_free():
    # F = chi * rho * A : [J/m^3]*[m^2] = N, independent of stiffness/coupling
    F = topological_phason_force(rho_substrate=5000.0, area_m2=3.0,
                                 defect=TopologicalPhasonDefect(charge=1))
    assert F == pytest.approx(5000.0 * 3.0)


def test_required_density_solves_lift_condition():
    m, A = 1500.0, 3.0
    rho = required_substrate_energy_density(m, A, TopologicalPhasonDefect(charge=1))
    # plugging it back must exactly balance the weight
    F = topological_phason_force(rho, A, TopologicalPhasonDefect(charge=1))
    assert F == pytest.approx(m * G_EARTH)


def test_cosmological_constant_problem_framing():
    # Honest landing: Lambda is far too small, Planck-cutoff density is plenty.
    rho_req = required_substrate_energy_density(1500.0, 4 * np.pi * 0.5 ** 2)
    assert rho_req > RHO_LAMBDA          # observed Lambda cannot lift a car
    assert rho_req < RHO_QFT_PLANCK      # unrenormalized zero-point could
