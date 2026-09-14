"""Independent controls for the frozen substrate long-wave contract.

Authored from doc/derivations/substrate_longwave_limit_2026-09-12.md,
without consulting the new module or demo.  Neither source report routines nor
source eigensystem/occupation helpers are used as mathematical oracles.

Numerical allowances below are explicitly diagnostic floating tolerances, not
certified errors and not modifications of any analytic bound.  In particular,
the analytic t=0 and g=0 bounds must remain exactly zero.
"""

import ast
import copy
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bpr import substrate_longwave_limit as subject


ROOT = Path(__file__).resolve().parents[1]
CLASSICAL_LENGTHS = (16, 32, 64, 128, 256)
MODES = (1, 2)
ACOUSTIC_COUPLINGS = (0.7, 40.0)
TIMES = (0.0, 0.5, 1.0, 2.0)
ZETAS = (0.5j, 1.0 + 0.5j, 4.0 + 1.0j)
QUANTUM_LENGTHS = (3, 4, 5)
QUANTUM_COUPLINGS = (0.0, 0.7, 40.0)
QUANTUM_TIMES = (0.0, 0.1, 0.5, 1.0)
EPS = np.finfo(float).eps

CLASSICAL_KEYS = {
    'L', 'm', 'g', 'C', 'nbar', 'h', 'p', 'scaling', 'coordinates',
    'time_scaling', 'source_scaling', 'density_output_factor', 'norm_weights',
    'a_h', 'a_0', 'a_second_order', 'a_remainder', 'a_remainder_bound',
    'a_difference_bound', 'generator', 'limit_generator',
    'scaled_frequency_squared', 'limit_frequency_squared', 'propagator_cases',
    'response_cases', 'generator_error_operator_norm', 'generator_error_bound',
    'exactness_certificate',
}
ACOUSTIC_EXTRA_KEYS = {
    'frequency_second_order', 'frequency_remainder', 'frequency_remainder_bound',
    'h_uniform_max', 'gamma', 'propagator_uniform_bound',
    'generator_bound_coefficient',
}
PROPAGATOR_KEYS = {
    'T', 'propagator', 'limit_propagator', 'error_operator_norm', 'error_bound',
    'uncapped_error_bound', 'bound_status',
}
RESPONSE_KEYS = {
    'zeta', 'microscopic_z', 'response', 'limit_response', 'absolute_error',
    'error_bound', 'resolvent_error_operator_norm', 'resolvent_error_bound',
    'resolvent_norm_bound', 'response_contraction_residual', 'bound_kind',
}
QUANTUM_KEYS = {
    'L', 'N', 'C', 'g', 't', 'dimension', 'basis', 'state_normalization_residual',
    'energy', 'expected_energy', 'energy_residual', 'orbital_frequency',
    'hartree_lift_frequency', 'phase_correction_rate', 'variance',
    'expected_variance', 'variance_residual', 'interaction_residual_agreement',
    'phase_corrected_error', 'error_bound', 'uncapped_error_bound', 'bound_status',
    'free_exact_structure', 'eigenpair_frobenius_residual',
    'orthogonality_frobenius_residual', 'eigensystem_tolerance',
    'exactness_certificate',
}


def diagnostic_tolerance(*values, factor=2048.0):
    """Roundoff allowance for comparison only, never an analytic envelope."""
    scale = max([1.0] + [float(np.max(np.abs(v))) for v in values])
    return factor * EPS * scale


def close(actual, expected, atol=None, rtol=2.0e-11):
    if atol is None:
        atol = diagnostic_tolerance(expected)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def within_analytic_envelope(error, bound, scale=1.0):
    assert math.isfinite(error) and math.isfinite(bound)
    assert error >= 0.0 and bound >= 0.0
    assert error <= bound + diagnostic_tolerance(scale, error, bound)


def strict_json_native(value):
    """Reject ndarray/scalar subclasses, nonfinite numbers and shared encoders."""
    if type(value) is dict:
        assert all(type(key) is str for key in value)
        for item in value.values():
            strict_json_native(item)
    elif type(value) is list:
        for item in value:
            strict_json_native(item)
    elif type(value) is float:
        assert math.isfinite(value)
    else:
        assert type(value) in (str, int, bool, type(None)), type(value)


def complex_scalar(value):
    assert type(value) is dict and set(value) == {'real', 'imag'}
    return complex(value['real'], value['imag'])


def matrix(value):
    if isinstance(value, dict):
        assert set(value) == {'shape', 'real', 'imag'}
        result = np.asarray(value['real']) + 1j * np.asarray(value['imag'])
        assert list(result.shape) == value['shape']
        return result
    assert type(value) is list
    result = np.asarray(value)
    assert result.ndim == 2
    return result


def fixed_weights(value, expected):
    assert type(value) is list and len(value) == 2
    close(value, expected)


def weighted_operator_norm(operator, weights):
    similarity = np.diag(np.sqrt(weights))
    transformed = similarity @ operator @ np.diag(1.0 / np.sqrt(weights))
    return float(np.linalg.svd(transformed, compute_uv=False)[0])


def independent_exponential(generator, time):
    """Eigenbasis exponential, deliberately not the source cos/sinc formula."""
    eigenvalues, vectors = np.linalg.eig(np.asarray(generator, dtype=complex))
    return (vectors * np.exp(time * eigenvalues)) @ np.linalg.inv(vectors)


def independent_laplacian_mode(length, mode):
    """Assemble 2I-P-P^T, then contract with a normalized Fourier orbital."""
    shift = np.zeros((length, length))
    for site in range(length):
        shift[(site + 1) % length, site] = 1.0
    laplacian = 2.0 * np.eye(length) - shift - shift.T
    orbital = np.exp(2j * np.pi * mode * np.arange(length) / length)
    orbital /= math.sqrt(length)
    contraction = np.vdot(orbital, laplacian @ orbital)
    close(np.vdot(orbital, orbital), 1.0)
    close(contraction.imag, 0.0)
    close(laplacian @ orbital, contraction.real * orbital)
    return float(contraction.real)


def classical_oracle(length, mode, coupling):
    h = 1.0 / length
    p = 2.0 * np.pi * mode
    epsilon = independent_laplacian_mode(length, mode)
    a = epsilon / h**2
    a0 = p**2
    microscopic = np.array([[0.0, 2.0 * epsilon],
                            [-coupling - epsilon / 2.0, 0.0]])
    if coupling:
        coordinate_map = np.diag([1.0 / h, 1.0])
        time_scale = h
        source_scale = h
        limit = np.array([[0.0, 2.0 * a0], [-coupling, 0.0]])
        weights = np.array([coupling, 2.0 * a0])
        output = 1.0
    else:
        coordinate_map = np.diag([0.5, 1.0])
        time_scale = h**2
        source_scale = h**2
        limit = np.array([[0.0, a0], [-a0, 0.0]])
        weights = np.ones(2)
        output = 2.0
    generator = coordinate_map @ microscopic @ np.linalg.inv(coordinate_map)
    generator /= time_scale
    rescaled_source = coordinate_map @ np.array([0.0, -source_scale])
    rescaled_source /= time_scale
    close(rescaled_source, [0.0, -1.0])
    return {
        'h': h, 'p': p, 'epsilon': epsilon, 'a': a, 'a0': a0,
        'microscopic': microscopic, 'time_scale': time_scale,
        'source_scale': source_scale, 'generator': generator, 'limit': limit,
        'weights': weights, 'output': output,
    }


def check_classical_report(report, length, mode, coupling):
    expected_keys = CLASSICAL_KEYS | (ACOUSTIC_EXTRA_KEYS if coupling else set())
    assert set(report) == expected_keys
    strict_json_native(report)
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report['exactness_certificate'] is False
    assert report['L'] == length and report['m'] == mode
    assert report['g'] == coupling and report['C'] == report['nbar'] == 1.0
    assert report['scaling'] == ('acoustic' if coupling else 'dispersive')
    assert type(report['coordinates']) is list and len(report['coordinates']) == 2
    assert all(type(item) is str and item for item in report['coordinates'])
    for key in ('time_scaling', 'source_scaling'):
        assert type(report[key]) is str and report[key], key
    oracle = classical_oracle(length, mode, coupling)
    h, p, a, a0 = (oracle[key] for key in ('h', 'p', 'a', 'a0'))
    # Fourier contraction suffers cancellation before division by h^2.
    lattice_atol = diagnostic_tolerance(length**2, factor=128.0)
    close(report['h'], h)
    close(report['p'], p)
    close(report['a_h'], a, atol=lattice_atol)
    close(report['a_0'], a0)
    close(report['density_output_factor'], oracle['output'])
    fixed_weights(report['norm_weights'], oracle['weights'])
    assert type(report['generator']) is list
    assert type(report['limit_generator']) is list
    close(matrix(report['generator']), oracle['generator'], atol=lattice_atol)
    close(matrix(report['limit_generator']), oracle['limit'])
    a_second = a0 - h**2 * p**4 / 12.0
    close(report['a_second_order'], a_second)
    close(report['a_remainder'], report['a_h'] - a_second)
    close(report['a_remainder_bound'], h**4 * abs(p)**6 / 360.0)
    close(report['a_difference_bound'], h**2 * p**4 / 12.0)
    within_analytic_envelope(abs(report['a_remainder']),
                             report['a_remainder_bound'], a0)
    within_analytic_envelope(a0 - report['a_h'],
                             report['a_difference_bound'], a0)
    generator, limit, weights = (oracle[key] for key in
                                  ('generator', 'limit', 'weights'))
    generator_error = weighted_operator_norm(generator - limit, weights)
    close(report['generator_error_operator_norm'], generator_error,
          atol=lattice_atol)
    if coupling:
        gamma = 1.0 - np.pi**2 / 192.0
        omega0 = math.sqrt(2.0 * coupling * a0)
        coefficient = omega0 * max(p**2 / 12.0, p**2 / (2.0 * coupling))
        uniform_bound = math.sqrt(1.0 / gamma + (1.0 / 16.0)**2 *
                                  p**2 / (2.0 * coupling))
        frequency0 = 2.0 * coupling * a0
        frequency = a * (2.0 * coupling + h**2 * a)
        frequency_second = frequency0 + h**2 * p**4 * (1.0 - coupling / 6.0)
        close(report['gamma'], gamma)
        close(report['h_uniform_max'], 1.0 / 16.0)
        close(report['generator_bound_coefficient'], coefficient)
        close(report['propagator_uniform_bound'], uniform_bound)
        close(report['frequency_second_order'], frequency_second)
        close(report['frequency_remainder'],
              report['scaled_frequency_squared'] - frequency_second)
        close(report['frequency_remainder_bound'],
              h**4 * abs(p)**6 * (coupling / 180.0 + 1.0 / 6.0))
        within_analytic_envelope(abs(report['frequency_remainder']),
                                 report['frequency_remainder_bound'], frequency0)
        generator_bound = h**2 * coefficient
        trivial_cap = uniform_bound + 1.0
    else:
        frequency0, frequency = a0**2, a**2
        uniform_bound = 1.0
        generator_bound = h**2 * p**4 / 12.0
        trivial_cap = 2.0
    close(report['scaled_frequency_squared'], frequency,
          atol=lattice_atol * max(1.0, 2.0 * a, 2.0 * coupling))
    close(report['limit_frequency_squared'], frequency0)
    close(report['generator_error_bound'], generator_bound)
    within_analytic_envelope(report['generator_error_operator_norm'],
                             report['generator_error_bound'], generator)
    assert len(report['propagator_cases']) == len(TIMES)
    for case, time in zip(report['propagator_cases'], TIMES):
        assert set(case) == PROPAGATOR_KEYS
        assert case['T'] == time
        actual_prop = matrix(case['propagator'])
        actual_limit = matrix(case['limit_propagator'])
        assert actual_prop.shape == actual_limit.shape == (2, 2)
        close(actual_prop, independent_exponential(generator, time),
              atol=8.0 * lattice_atol)
        close(actual_limit, independent_exponential(limit, time))
        raw_error = weighted_operator_norm(actual_prop - actual_limit, weights)
        close(case['error_operator_norm'], raw_error)
        independent_error = weighted_operator_norm(
            independent_exponential(generator, time) -
            independent_exponential(limit, time), weights)
        close(case['error_operator_norm'], independent_error,
              atol=8.0 * lattice_atol)
        uncapped = uniform_bound * abs(time) * generator_bound
        close(case['uncapped_error_bound'], uncapped)
        close(case['error_bound'], min(trivial_cap, uncapped))
        assert case['bound_status'] == (
            'informative' if uncapped < trivial_cap else 'trivial_cap')
        within_analytic_envelope(case['error_operator_norm'], case['error_bound'])
        within_analytic_envelope(weighted_operator_norm(actual_prop, weights),
                                 uniform_bound)
        if time == 0.0:
            assert case['error_bound'] == case['uncapped_error_bound'] == 0.0
            close(actual_prop, np.eye(2))
        if not coupling:
            close(actual_prop.conj().T @ actual_prop, np.eye(2))
            close(independent_exponential(generator, -time) @ actual_prop,
                  np.eye(2), atol=8.0 * lattice_atol)
    assert len(report['response_cases']) == len(ZETAS)
    source = np.array([0.0, -1.0])
    for case, zeta in zip(report['response_cases'], ZETAS):
        assert set(case) == RESPONSE_KEYS
        assert case['bound_kind'] == 'analytic_absolute_envelope_not_accuracy_certificate'
        close(complex_scalar(case['zeta']), zeta)
        microscopic_z = oracle['time_scale'] * zeta
        close(complex_scalar(case['microscopic_z']), microscopic_z)
        resolvent = np.linalg.inv(-1j * zeta * np.eye(2) - generator)
        limit_resolvent = np.linalg.inv(-1j * zeta * np.eye(2) - limit)
        response = oracle['output'] * (resolvent @ source)[0]
        limit_response = oracle['output'] * (limit_resolvent @ source)[0]
        micro_resolvent = np.linalg.inv(
            -1j * microscopic_z * np.eye(2) - oracle['microscopic'])
        micro_response = (micro_resolvent @ source)[0]
        expected_from_micro = micro_response if coupling else h**2 * micro_response
        close(response, expected_from_micro, atol=lattice_atol)
        close(response, 2.0 * a / (zeta**2 - frequency), atol=lattice_atol)
        close(complex_scalar(case['response']), response, atol=lattice_atol)
        close(complex_scalar(case['limit_response']), limit_response)
        # Positive density source gives negative susceptibility on imaginary axis.
        if zeta.real == 0.0:
            assert complex_scalar(case['response']).real < 0.0
        close(case['absolute_error'], abs(complex_scalar(case['response']) -
                                          complex_scalar(case['limit_response'])))
        close(case['resolvent_error_operator_norm'],
              weighted_operator_norm(resolvent - limit_resolvent, weights),
              atol=lattice_atol)
        resolvent_error_bound = uniform_bound * generator_bound / zeta.imag**2
        scalar_factor = math.sqrt(2.0 * a0 / coupling) if coupling else 2.0
        close(case['resolvent_error_bound'], resolvent_error_bound)
        close(case['error_bound'], scalar_factor * resolvent_error_bound)
        close(case['resolvent_norm_bound'], uniform_bound / zeta.imag)
        within_analytic_envelope(case['absolute_error'], case['error_bound'])
        within_analytic_envelope(case['resolvent_error_operator_norm'],
                                 case['resolvent_error_bound'])
        within_analytic_envelope(weighted_operator_norm(resolvent, weights),
                                 case['resolvent_norm_bound'])
        # This is a raw residual, not a value clipped into the envelope.
        assert case['response_contraction_residual'] >= 0.0
        assert case['response_contraction_residual'] <= diagnostic_tolerance(response)


@pytest.mark.parametrize('length,mode,coupling', itertools.product(
    CLASSICAL_LENGTHS, MODES, ACOUSTIC_COUPLINGS))
def test_acoustic_frozen_grid(length, mode, coupling):
    check_classical_report(subject.acoustic_report(length, mode, coupling),
                           length, mode, coupling)


@pytest.mark.parametrize('length,mode', itertools.product(CLASSICAL_LENGTHS, MODES))
def test_free_dispersive_frozen_grid(length, mode):
    check_classical_report(subject.dispersive_report(length, mode),
                           length, mode, 0.0)


def test_fixed_norm_is_not_euclidean_or_frobenius():
    report = subject.acoustic_report(16, 2, 0.7)
    generator = matrix(report['generator'])
    limit = matrix(report['limit_generator'])
    weights = [0.7, 2.0 * (4.0 * np.pi)**2]
    expected = weighted_operator_norm(generator - limit, weights)
    euclidean = np.linalg.norm(generator - limit, ord=2)
    transformed = np.diag(np.sqrt(weights)) @ (generator - limit)
    transformed = transformed @ np.diag(1.0 / np.sqrt(weights))
    frobenius = np.linalg.norm(transformed, ord='fro')
    assert not np.isclose(expected, euclidean, rtol=1.0e-3)
    assert not np.isclose(expected, frobenius, rtol=1.0e-3)
    close(report['generator_error_operator_norm'], expected)


def recursive_occupations(sites, particles):
    """Independent complete weak compositions, lexicographic from recursion."""
    if sites == 1:
        return [(particles,)]
    result = []
    for first in range(particles + 1):
        for rest in recursive_occupations(sites - 1, particles - first):
            result.append((first,) + rest)
    return result


def independent_quantum_system(length, coupling):
    particles = length
    basis = recursive_occupations(length, particles)
    index = {occupation: position for position, occupation in enumerate(basis)}
    dimension = len(basis)
    assert dimension == math.comb(particles + length - 1, particles)
    hopping = np.zeros((dimension, dimension))
    pair_counts = np.array([sum(n * (n - 1) // 2 for n in occupation)
                            for occupation in basis], dtype=float)
    for column, occupation in enumerate(basis):
        for origin in range(length):
            if occupation[origin] == 0:
                continue
            for destination in ((origin - 1) % length, (origin + 1) % length):
                target = list(occupation)
                target[origin] -= 1
                target[destination] += 1
                amplitude = math.sqrt(occupation[origin] *
                                      (occupation[destination] + 1))
                hopping[index[tuple(target)], column] -= amplitude
    hamiltonian = hopping + np.diag(coupling * pair_counts)
    close(hamiltonian, hamiltonian.T)
    orbital = np.ones(length) / math.sqrt(length)
    state = independent_condensate(basis, orbital)
    return basis, hopping, hamiltonian, pair_counts, state


def independent_condensate(basis, orbital):
    particles = sum(basis[0])
    amplitudes = []
    for occupation in basis:
        multiplicity = math.factorial(particles)
        for number in occupation:
            multiplicity /= math.factorial(number)
        amplitude = math.sqrt(multiplicity)
        for component, number in zip(orbital, occupation):
            amplitude *= component**number
        amplitudes.append(amplitude)
    return np.asarray(amplitudes, dtype=complex)


def check_quantum_report(report, length, coupling, time):
    assert set(report) == QUANTUM_KEYS
    strict_json_native(report)
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report['exactness_certificate'] is False
    assert report['L'] == report['N'] == length
    assert report['C'] == 1.0 and report['g'] == coupling and report['t'] == time
    basis, hopping, hamiltonian, pair_counts, state = independent_quantum_system(
        length, coupling)
    dimension = len(basis)
    assert report['dimension'] == dimension
    assert report['basis'] == [list(occupation) for occupation in basis]
    assert dimension <= 512
    close(np.vdot(state, state), 1.0)
    close(hopping @ state, -2.0 * length * state)
    interaction_mean = coupling * length * (length - 1) / (2.0 * length)
    energy = -2.0 * length + interaction_mean
    numerical_energy = float(np.vdot(state, hamiltonian @ state).real)
    orbital_frequency = -2.0 + coupling * (length - 1) / length
    expected_variance = coupling**2 * math.comb(length, 2) * (length - 1) / length**2
    residual = (hamiltonian - energy * np.eye(dimension)) @ state
    interaction_residual = coupling * (pair_counts - (length - 1) / 2.0) * state
    variance = float(np.vdot(residual, residual).real)
    close(residual, interaction_residual)
    close(report['expected_energy'], energy)
    close(report['energy'], numerical_energy)
    close(report['orbital_frequency'], orbital_frequency)
    close(report['hartree_lift_frequency'], length * orbital_frequency)
    close(report['phase_correction_rate'], interaction_mean)
    close(report['variance'], variance)
    close(report['expected_variance'], expected_variance)
    close(report['variance'], expected_variance)
    for key, scale in (('state_normalization_residual', 1.0),
                       ('energy_residual', energy),
                       ('variance_residual', expected_variance),
                       ('interaction_residual_agreement', hamiltonian)):
        assert report[key] >= 0.0
        assert report[key] <= diagnostic_tolerance(scale)
    close(report['energy_residual'], abs(report['energy'] - energy))
    close(report['variance_residual'], abs(report['variance'] - expected_variance))
    # Independent full-H evolution followed by the *fixed* mean-energy phase.
    # The implementation contract instead diagonalizes H-EI for this evolution.
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    exact = eigenvectors @ (np.exp(-1j * time * eigenvalues) *
                            (eigenvectors.conj().T @ state))
    trial = np.exp(-1j * energy * time) * state
    corrected_error = float(np.linalg.norm(exact - trial))
    close(report['phase_corrected_error'], corrected_error,
          atol=diagnostic_tolerance(dimension))
    uncorrected_hartree = np.exp(-1j * length * orbital_frequency * time) * state
    phase_corrected_hartree = np.exp(1j * interaction_mean * time) * uncorrected_hartree
    close(phase_corrected_hartree, trial)
    wrong_phase_residual = (hamiltonian - length * orbital_frequency *
                            np.eye(dimension)) @ state
    close(np.linalg.norm(wrong_phase_residual)**2,
          expected_variance + interaction_mean**2)
    uncapped = abs(time) * math.sqrt(expected_variance)
    close(report['uncapped_error_bound'], uncapped, atol=0.0, rtol=8.0 * EPS)
    close(report['error_bound'], min(2.0, uncapped), atol=0.0, rtol=8.0 * EPS)
    assert report['bound_status'] == ('informative' if uncapped < 2.0 else 'trivial_cap')
    within_analytic_envelope(report['phase_corrected_error'], report['error_bound'],
                             dimension)
    assert report['free_exact_structure'] is (coupling == 0.0)
    tolerance = 256.0 * EPS * dimension
    close(report['eigensystem_tolerance'], tolerance, atol=0.0, rtol=4.0 * EPS)
    centered = hamiltonian - energy * np.eye(dimension)
    assert 0.0 <= report['orthogonality_frobenius_residual'] <= tolerance
    assert 0.0 <= report['eigenpair_frobenius_residual'] <= (
        tolerance * max(1.0, np.linalg.norm(centered, ord='fro')))
    if coupling == 0.0 or time == 0.0:
        assert report['error_bound'] == report['uncapped_error_bound'] == 0.0
        # Never require a raw floating residual to be exactly zero.
        assert report['phase_corrected_error'] <= diagnostic_tolerance(dimension)


@pytest.mark.parametrize('length,coupling,time', itertools.product(
    QUANTUM_LENGTHS, QUANTUM_COUPLINGS, QUANTUM_TIMES))
def test_quantum_complete_frozen_grid(length, coupling, time):
    check_quantum_report(subject.uniform_condensate_bridge(length, coupling, time),
                         length, coupling, time)


def test_hilbert_distance_is_not_projective_or_uncorrected_hartree():
    length, coupling, time = 3, 0.7, 0.5
    report = subject.uniform_condensate_bridge(length, coupling, time)
    _, _, hamiltonian, _, state = independent_quantum_system(length, coupling)
    eigenvalues, vectors = np.linalg.eigh(hamiltonian)
    exact = vectors @ (np.exp(-1j * time * eigenvalues) * (vectors.conj().T @ state))
    energy = -2.0 * length + coupling * (length - 1) / 2.0
    corrected = np.linalg.norm(exact - np.exp(-1j * energy * time) * state)
    overlap = abs(np.vdot(state, exact))
    projective = math.sqrt(max(0.0, 2.0 - 2.0 * overlap))
    hartree_frequency = length * (-2.0 + coupling * (length - 1) / length)
    uncorrected = np.linalg.norm(exact - np.exp(-1j * hartree_frequency * time) * state)
    assert abs(corrected - projective) > 1.0e-5
    assert abs(corrected - uncorrected) > 1.0e-3
    close(report['phase_corrected_error'], corrected)
    assert not np.isclose(report['phase_corrected_error'], projective, rtol=1.0e-5)


def test_nonuniform_multinomial_expectation_algebra_only():
    # A fixed algebraic oracle control, not a new public dynamics grid.
    length, coupling = 3, 0.7
    basis, _, hamiltonian, _, _ = independent_quantum_system(length, coupling)
    orbital = np.array([1.0 + 0.5j, -0.75j, 0.0], dtype=complex)
    orbital /= np.linalg.norm(orbital)
    state = independent_condensate(basis, orbital)
    one_particle = np.zeros((length, length))
    for site in range(length):
        one_particle[site, (site - 1) % length] -= 1.0
        one_particle[site, (site + 1) % length] -= 1.0
    expected = (length * np.vdot(orbital, one_particle @ orbital) +
                coupling * length * (length - 1) / 2.0 * np.sum(abs(orbital)**4))
    close(np.vdot(state, state), 1.0)
    close(np.vdot(state, hamiltonian @ state), expected)
    # N(N-1), not N^2, and positive common uniform phase are load-bearing.
    incorrect = (length * np.vdot(orbital, one_particle @ orbital) +
                 coupling * length**2 / 2.0 * np.sum(abs(orbital)**4))
    assert abs(expected - incorrect) > 0.1


class CoercibleScalar:
    def __float__(self):
        return 0.7

    def __int__(self):
        return 3


BAD_REAL_VALUES = (
    True, False, np.bool_(True), '0.7', '40', 0.7 + 0j, np.complex128(0.7),
    float('nan'), float('inf'), -float('inf'), None, [], {},
    np.array(0.7), np.array([0.7]), CoercibleScalar(),
)
BAD_INTEGER_VALUES = (
    True, False, np.bool_(False), 16.0, np.float64(16), '16', 16 + 0j,
    float('nan'), float('inf'), None, [], {}, np.array(16), CoercibleScalar(),
)


def ordinary_domain_error(function, *args):
    with pytest.raises(ValueError) as error:
        function(*args)
    assert type(error.value) is ValueError


@pytest.mark.parametrize('value', BAD_INTEGER_VALUES)
def test_invalid_integer_types(value):
    ordinary_domain_error(subject.acoustic_report, value, 1, 0.7)
    ordinary_domain_error(subject.acoustic_report, 16, value, 0.7)
    ordinary_domain_error(subject.dispersive_report, value, 1)
    ordinary_domain_error(subject.dispersive_report, 16, value)
    ordinary_domain_error(subject.uniform_condensate_bridge, value, 0.0, 0.0)


@pytest.mark.parametrize('value', BAD_REAL_VALUES)
def test_invalid_real_types_before_structural_zero(value):
    ordinary_domain_error(subject.acoustic_report, 16, 1, value)
    ordinary_domain_error(subject.uniform_condensate_bridge, 3, value, 0.0)
    ordinary_domain_error(subject.uniform_condensate_bridge, 3, 0.0, value)


@pytest.mark.parametrize('length', (-1, 0, 1, 8, 15, 17, 512, 10**100))
def test_classical_lengths_outside_frozen_grid(length):
    ordinary_domain_error(subject.acoustic_report, length, 1, 0.7)
    ordinary_domain_error(subject.dispersive_report, length, 1)


@pytest.mark.parametrize('mode', (-2, -1, 0, 3, 16, 10**100))
def test_no_zero_or_unfrozen_modes(mode):
    ordinary_domain_error(subject.acoustic_report, 16, mode, 0.7)
    ordinary_domain_error(subject.dispersive_report, 16, mode)


@pytest.mark.parametrize('coupling', (-1, 0, 0.1, 1, 39, 41, 1.0e300))
def test_acoustic_rejects_free_and_unfrozen_couplings(coupling):
    ordinary_domain_error(subject.acoustic_report, 16, 1, coupling)


@pytest.mark.parametrize('length', (-1, 0, 1, 2, 6, 16, 10**100))
def test_quantum_domain_before_free_zero_dispatch(length):
    ordinary_domain_error(subject.uniform_condensate_bridge, length, 0, 0)


@pytest.mark.parametrize('coupling', (-1, 0.1, 1, 39, 41, 1.0e300))
def test_quantum_coupling_before_zero_time(coupling):
    ordinary_domain_error(subject.uniform_condensate_bridge, 3, coupling, 0)


@pytest.mark.parametrize('time', (-1, -0.1, 0.01, 0.2, 2, 1.0e300))
def test_quantum_time_before_free_dispatch(time):
    ordinary_domain_error(subject.uniform_condensate_bridge, 3, 0, time)


def test_numpy_real_and_integer_scalars_are_accepted():
    acoustic = subject.acoustic_report(np.int64(16), np.int32(1), np.float64(0.7))
    dispersive = subject.dispersive_report(np.int32(32), np.int64(2))
    quantum = subject.uniform_condensate_bridge(np.int64(3), np.int32(40), np.float64(0.1))
    integer_zero = subject.uniform_condensate_bridge(3, 0, 0)
    for report in (acoustic, dispersive, quantum, integer_zero):
        strict_json_native(report)
    assert acoustic['L'] == 16 and quantum['g'] == 40


def forbid_array_allocation(*args, **kwargs):
    raise AssertionError('array allocation occurred before domain/cap validation')


def test_invalid_domains_rejected_before_numpy_allocation(monkeypatch):
    for name in ('array', 'asarray', 'zeros', 'ones', 'empty', 'full', 'eye'):
        monkeypatch.setattr(np, name, forbid_array_allocation)
    ordinary_domain_error(subject.acoustic_report, 17, 1, 0.7)
    ordinary_domain_error(subject.dispersive_report, 16, 0)
    ordinary_domain_error(subject.uniform_condensate_bridge, 3, 0.0, -0.1)
    ordinary_domain_error(subject.uniform_condensate_bridge, 6, 0.0, 0.0)


@pytest.mark.parametrize('length,coupling,time', (
    (3, 0.0, 0.0), (3, 0.7, 0.1), (4, 40.0, 0.5), (5, 0.0, 1.0),
))
def test_reduced_cap_before_numpy_allocation(monkeypatch, length, coupling, time):
    assert subject.MAX_QUANTUM_DIMENSION == 512
    monkeypatch.setattr(subject, 'MAX_QUANTUM_DIMENSION', 1)
    for name in ('array', 'asarray', 'zeros', 'ones', 'empty', 'full', 'eye'):
        monkeypatch.setattr(np, name, forbid_array_allocation)
    with pytest.raises(ValueError):
        subject.uniform_condensate_bridge(length, coupling, time)


def test_unavailable_is_distinct_from_invalid_domain():
    assert issubclass(subject.NumericalUnavailable, ValueError)
    assert subject.NumericalUnavailable is not ValueError
    ordinary_domain_error(subject.uniform_condensate_bridge, 3, 0.0, -0.1)


def mutable_ids(value):
    result = set()
    if type(value) is dict:
        result.add(id(value))
        for item in value.values():
            result.update(mutable_ids(item))
    elif type(value) is list:
        result.add(id(value))
        for item in value:
            result.update(mutable_ids(item))
    return result


@pytest.mark.parametrize('function,args,nested_path', (
    ('acoustic_report', (16, 1, 0.7), ('generator', 0, 1)),
    ('dispersive_report', (32, 2), ('propagator_cases', 1, 'propagator', 0, 0)),
    ('uniform_condensate_bridge', (3, 0.7, 0.1), ('basis', 0, 0)),
))
def test_reports_are_detached(function, args, nested_path):
    make_report = getattr(subject, function)
    first, second = make_report(*args), make_report(*args)
    assert mutable_ids(first).isdisjoint(mutable_ids(second))
    saved = copy.deepcopy(second)
    target = first
    for component in nested_path[:-1]:
        target = target[component]
    target[nested_path[-1]] = 'intentional caller mutation'
    assert second == saved
    third = make_report(*args)
    assert third == saved


def check_demonstration_shape(report):
    assert set(report) == {'acoustic_cases', 'dispersive_cases', 'quantum_cases',
                           'limitations', 'numerical_domain'}
    strict_json_native(report)
    assert len(report['acoustic_cases']) == 20
    assert len(report['dispersive_cases']) == 10
    assert len(report['quantum_cases']) == 36
    assert {(case['L'], case['m'], case['g']) for case in report['acoustic_cases']} == set(
        itertools.product(CLASSICAL_LENGTHS, MODES, ACOUSTIC_COUPLINGS))
    assert {(case['L'], case['m']) for case in report['dispersive_cases']} == set(
        itertools.product(CLASSICAL_LENGTHS, MODES))
    assert {(case['L'], case['g'], case['t']) for case in report['quantum_cases']} == set(
        itertools.product(QUANTUM_LENGTHS, QUANTUM_COUPLINGS, QUANTUM_TIMES))
    assert report['limitations'] and report['numerical_domain']
    for key in ('acoustic_cases', 'dispersive_cases', 'quantum_cases'):
        accumulated = set()
        for case in report[key]:
            assert case['exactness_certificate'] is False
            current = mutable_ids(case)
            assert current.isdisjoint(accumulated)
            accumulated.update(current)


def test_demonstration_complete_strict_json_and_detached(capsys):
    report = subject.demonstration_report()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ''
    check_demonstration_shape(report)
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    old = copy.deepcopy(report['quantum_cases'][0])
    separate = subject.uniform_condensate_bridge(old['L'], old['g'], old['t'])
    assert mutable_ids(report).isdisjoint(mutable_ids(separate))
    separate['basis'][0][0] = -999
    assert report['quantum_cases'][0] == old


@pytest.mark.parametrize('json_mode', (False, True))
def test_demo_stdout_only_from_empty_directory(tmp_path, json_mode):
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[key] = '1'
    command = [sys.executable, '-W', 'error',
               str(ROOT / 'scripts' / 'demo_substrate_longwave_limit.py')]
    if json_mode:
        command.append('--json')
    completed = subprocess.run(command, cwd=str(tmp_path), env=environment,
                               capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ''
    assert completed.stdout.strip()
    assert list(tmp_path.iterdir()) == []
    if json_mode:
        def reject_nonfinite(token):
            raise AssertionError('nonfinite JSON token: ' + token)
        report = json.loads(completed.stdout, parse_constant=reject_nonfinite)
        check_demonstration_shape(report)
    else:
        text = completed.stdout.lower()
        assert 'acoustic' in text
        assert 'dispersive' in text or 'quadratic' in text
        assert 'condensate' in text or 'quantum' in text
        assert 'limit' in text


@pytest.mark.parametrize('relative_path', (
    'bpr/substrate_longwave_limit.py',
    'scripts/demo_substrate_longwave_limit.py',
    'tests/test_substrate_longwave_limit.py',
))
def test_python38_grammar(relative_path):
    path = ROOT / relative_path
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path),
              feature_version=(3, 8))
