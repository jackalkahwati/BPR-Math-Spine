"""Independent all-population checks of a conditional bosonic Hamiltonian."""
from fractions import Fraction
from itertools import product
from math import factorial, comb
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.linalg import block_diag, expm

from bpr import substrate_fermionization as old
from bpr import substrate_vacuum_selection as v


@pytest.mark.parametrize('L', [3, 4, 5])
@pytest.mark.parametrize('N', range(8))
def test_balanced_minimum_by_independent_occupations(L, N):
    occupations = [s for s in product(range(N+1), repeat=L) if sum(s) == N]
    minimum = min(sum(n*(n-1)//2 for n in s) for s in occupations)
    assert v.balanced_interaction_minimum(L, N) == minimum
    assert 2*L*minimum >= N*N-L*N


def test_large_exact_integer_minimum():
    L, a, r = 10**40, 10**70, 123
    assert v.balanced_interaction_minimum(L, L*a+r) == L*a*(a-1)//2+a*r


@pytest.mark.parametrize('L,N', [(3, 0), (3, 1), (3, 3), (4, 2), (5, 5)])
def test_delegates_unchanged_domain(L, N):
    a = v.all_number_model(L, N, .7, 4.)
    b = old.fixed_number_model(L, N, .7, 4.)
    assert a.basis == b.basis
    for key in ['D', 'H', 'V', 'V_unit', 'p_indices', 'q_indices', 'PHP', 'B', 'QHQ']:
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))


def test_new_domain_does_not_widen_old_api():
    with pytest.raises(ValueError):
        old.fixed_number_model(5, 6)
    a = v.all_number_model(5, 6)
    assert len(a.basis) == 210
    assert a.PHP.shape == (0, 0)
    assert a.B.shape == (0, 210)
    assert a.hard_core_basis == ()
    np.testing.assert_array_equal(a.QHQ, a.H)


def kron_all(operators):
    result = np.ones((1, 1))
    for op in operators:
        result = np.kron(result, op)
    return result


def test_above_unit_filling_independent_tensor_oracle():
    a = np.diag(np.sqrt(np.arange(1, 5)), 1)
    ops = [kron_all([a if x == site else np.eye(5) for x in range(3)])
           for site in range(3)]
    ns = [a.T @ a for a in ops]
    C, g = .8, 2.3
    H = sum(-C*(ops[x].T @ ops[(x+1)%3]+ops[(x+1)%3].T @ ops[x]) for x in range(3))
    H += sum(g/2*n @ (n-np.eye(125)) for n in ns)
    model = v.all_number_model(3, 4, C, g)
    indices = [25*s[0]+5*s[1]+s[2] for s in model.basis]
    np.testing.assert_allclose(model.H, H[np.ix_(indices, indices)], atol=1e-14)


@pytest.mark.parametrize('g', [0., .7, 40.])
def test_six_particle_complete_matrix_elements(g):
    a = v.all_number_model(3, 6, 1., g)
    assert len(a.basis) == 28
    assert min(a.D) == 3
    assert (6, 0, 0) in a.basis
    for col, state in enumerate(a.basis):
        expected = np.zeros(28)
        expected[col] = g*sum(n*(n-1)/2 for n in state)
        for target in range(3):
            for source in range(3):
                if target != source and state[source]:
                    moved = list(state)
                    moved[source] -= 1
                    moved[target] += 1
                    expected[a.basis.index(tuple(moved))] -= np.sqrt(state[source]*(state[target]+1))
        np.testing.assert_allclose(a.H[:, col], expected, atol=1e-14)
    if g == 0:
        assert np.linalg.eigvalsh(a.H)[0] == pytest.approx(-12.)


@pytest.mark.parametrize('L,N', [(3, 0), (3, 2), (3, 3), (3, 6), (4, 4), (4, 5)])
@pytest.mark.parametrize('g', [0., .7, 40.])
def test_variational_bounds_and_trial_states(L, N, g):
    C = .8
    model = v.all_number_model(L, N, C, g)
    bounds = v.sector_energy_bounds(L, N, C, g)
    e0 = np.linalg.eigvalsh(model.H)[0]
    assert float(bounds['lower_bound']) <= e0+2e-12
    assert e0 <= float(bounds['best_upper_bound'])+2e-12
    uniform = np.array([np.sqrt(factorial(N)/(L**N*np.prod([factorial(n) for n in s])))
                        for s in model.basis])
    assert np.linalg.norm(uniform) == pytest.approx(1.)
    assert uniform @ model.H @ uniform == pytest.approx(
        float(bounds['trial_upper_bounds']['uniform_condensate']), abs=2e-12)
    balanced = np.argmin(model.D)
    assert model.H[balanced, balanced] == pytest.approx(
        float(bounds['trial_upper_bounds']['balanced_occupation']))
    if N == L:
        z = np.zeros(len(model.basis))
        z[model.basis.index((1,)*L)] = 1.
        pair = (2, 0)+(1,)*(L-2)
        z[model.basis.index(pair)] = np.sqrt(2)*C/(g+2*C)
        z /= np.linalg.norm(z)
        assert z @ model.H @ z == pytest.approx(
            float(bounds['trial_upper_bounds']['unit_filling_two_state']), abs=1e-14)
        assert bounds['trial_upper_bounds']['unit_filling_two_state'] < 0


def test_exact_binary64_bounds():
    bounds = v.sector_energy_bounds(5, 15, 1., .7)
    assert bounds['trial_upper_bounds']['uniform_condensate'] == 21*Fraction.from_float(.7)-30
    assert bounds['trial_upper_bounds']['uniform_condensate'] != Fraction(-153, 10)
    assert v.sector_energy_bounds(5, 5, 1., .7)['lower_bound'] == -10


@pytest.mark.parametrize('g', [4., np.nextafter(4., 0.), np.nextafter(4., np.inf), 40.])
def test_tail_cutoff_exact_boundary(g):
    tail = v.neutral_tail_bound(5, 5, 1., g)
    T = 5*(1+4/Fraction.from_float(g))
    K = T//5
    assert tail['cutoff'] == T
    assert tail['candidate_count'] == K+1
    assert tail['max_neutral_number'] == K*5
    assert tail['first_excluded_neutral'] == (K+1)*5
    if g == 4:
        assert tail['max_neutral_number'] == 10


@pytest.mark.parametrize('L', [3, 5, 100, 10**30])
@pytest.mark.parametrize('g', [4., 40.])
def test_strong_theorem_independent_of_matrix_caps(L, g):
    report = v.analyze_neutral_sectors(L, L, 1., g, numerical=False)
    assert report['status'] == 'selected'
    assert int(report['selected_sector']) == L
    assert [int(x) for x in report['surviving_sectors']] == [L]
    json.dumps(report, allow_nan=False)


def test_weak_unit_filling_excluded_all_candidates_retained():
    r = v.analyze_neutral_sectors(5, 5, 1., .7)
    assert r['status'] == 'bounded_candidates'
    assert r['selected_sector'] is None
    assert r['surviving_sectors'] == [10, 15, 20]
    assert r['enumeration_complete']
    assert r['best_trial']['N'] == 15
    rows = {x['N']: x for x in r['sectors']}
    assert rows[5]['status'] == 'excluded'
    assert all(rows[N]['status'] == 'retained' for N in [10, 15, 20])
    assert all(rows[N]['numerical']['status'] == 'dimension_cap' for N in [10, 15, 20])


def test_vacuum_and_non_neutral_unit_filling():
    r = v.analyze_neutral_sectors(3, 4, 1., 40.)
    assert r['status'] == 'selected'
    assert r['selected_sector'] == 0
    assert r['surviving_sectors'] == [0]
    assert r['unit_filling_status'] == 'not_neutral'
    assert r['best_trial']['N'] == 0


@pytest.mark.parametrize('q', [2, 5, 10**100])
def test_free_all_neutral_unbounded(q):
    r = v.analyze_neutral_sectors(5, q, 1., 0.)
    assert r['status'] == 'unbounded_below'
    assert r['selected_sector'] is None
    assert r['surviving_sectors'] is None
    t = v.neutral_tail_bound(5, q, 1., 0.)
    assert t['cutoff'] is None
    json.dumps(r, allow_nan=False)


def test_candidate_cap_boundary_and_no_partial_winner():
    # g=4 gives T=2L: with q2 preliminary count=L+1.
    a = v.analyze_neutral_sectors(255, 2, 1., 4., numerical=False)
    b = v.analyze_neutral_sectors(256, 2, 1., 4., numerical=False)
    assert a['enumeration_complete']
    assert b['status'] == 'resource_unresolved'
    assert not b['enumeration_complete']
    assert b['selected_sector'] is None
    assert b['surviving_sectors'] is None
    assert not b['sectors']


def test_equal_lower_bound_is_not_excluded():
    # At L3,q4,g8 the N4 lower bound and vacuum upper bound are both0.
    r = v.analyze_neutral_sectors(3, 4, 1., 8., numerical=False)
    assert r['status'] == 'bounded_candidates'
    assert r['surviving_sectors'] == [0, 4]


@pytest.mark.parametrize('scale', [1e-100, 1e100])
def test_common_scale_selection(scale):
    r = v.analyze_neutral_sectors(5, 5, scale, 40*scale, numerical=False)
    assert r['selected_sector'] == 5
    b = v.sector_energy_bounds(3, 6, scale, 40*scale)
    assert b['lower_bound'] == 3*Fraction.from_float(40*scale)-12*Fraction.from_float(scale)


@pytest.mark.parametrize('value', [Fraction(0), Fraction(1, 3), Fraction(-1, 10**400),
                                   Fraction(10**400), Fraction(1, 10**310)])
def test_exact_rational_serialization(value):
    r = v.rational_record(value)
    assert Fraction(int(r['numerator']), int(r['denominator'])) == value
    if value and (abs(value) < Fraction.from_float(np.finfo(float).tiny)
                  or abs(value) > Fraction.from_float(np.finfo(float).max)):
        assert r['approximate'] is None
    elif value == 0:
        assert r['approximate'] == 0.
    json.dumps(r, allow_nan=False)


def test_tiny_negative_trial_not_false_zero():
    b = v.sector_energy_bounds(5, 5, 1e-200, 1e100)
    u = b['trial_upper_bounds']['unit_filling_two_state']
    assert u < 0
    assert v.rational_record(u)['approximate'] is None
    assert v.analyze_neutral_sectors(5, 5, 1e-200, 1e100, numerical=False)['selected_sector'] == 5


@pytest.mark.parametrize('L,N', [(3, 30), (4, 12), (5, 8)])
def test_dense_dimension_boundaries(L, N):
    assert len(v.all_number_model(L, N).basis) == comb(L+N-1, N)
    with pytest.raises(ValueError):
        v.all_number_model(L, N+1)


def test_capped_binomial_extreme():
    assert v.capped_binomial(100, 0) == 1
    assert v.capped_binomial(8, 2) == 28
    assert v.capped_binomial(10**100, 10**50) == 513


def test_caps_before_enumeration(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('enumeration/allocation reached')
    monkeypatch.setattr(v, '_occupations', fail)
    monkeypatch.setattr(np, 'zeros', fail)
    for L, N in [(3, 31), (5, 10), (13, 0), (3, 10**100)]:
        with pytest.raises(ValueError):
            v.all_number_model(L, N)


@pytest.mark.parametrize('L,N,C,g', [(2, 3, 1., 1.), (3, -1, 1., 1.), (True, 3, 1., 1.),
                                    (3, 4, 0., 1.), (3, 4, 1., -1.),
                                    (3, 4, np.inf, 1.), (3, 4, 1., np.nan),
                                    (3, 4, 1e-320, 1.)])
def test_invalid_inputs(L, N, C, g):
    with pytest.raises((ValueError, TypeError)):
        v.all_number_model(L, N, C, g)
    with pytest.raises((ValueError, TypeError)):
        v.sector_energy_bounds(L, N, C, g)


@pytest.mark.parametrize('q', [1, 0, True, 2.5])
def test_invalid_neutrality(q):
    with pytest.raises((ValueError, TypeError)):
        v.analyze_neutral_sectors(3, q)


def test_diagnostic_failure_cannot_change_analytic_outcome(monkeypatch):
    before = v.analyze_neutral_sectors(5, 5, 1., 40., numerical=False)
    def fail(*args, **kwargs):
        raise ValueError('numerically unresolved injected failure')
    monkeypatch.setattr(v, 'sector_ground_diagnostic', fail)
    after = v.analyze_neutral_sectors(5, 5, 1., 40., numerical=True)
    for key in ['status', 'selected_sector', 'surviving_sectors', 'certificate']:
        assert after[key] == before[key]


def test_reduced_candidate_cap_preserves_independent_theorem():
    r = v.analyze_neutral_sectors(5, 5, 1., 4., candidate_cap=1)
    assert r['status'] == 'selected'
    assert r['selected_sector'] == 5
    assert not r['enumeration_complete']
    assert not r['sectors']
    assert r['certificate']['type'] == 'q_equals_L_g_at_least_4C'


def test_zero_numerical_budget_and_site_cap():
    r = v.analyze_neutral_sectors(3, 3, 1., .7, numerical_budget=0)
    assert r['numerical_diagonalizations'] == 0
    assert all(row['numerical']['status'] == 'budget_exhausted'
               for row in r['sectors'] if row['N'])
    large = v.analyze_neutral_sectors(20, 20, 1., 40.)
    assert large['selected_sector'] == 20
    assert all(row['numerical']['status'] == 'site_cap' for row in large['sectors'])


@pytest.mark.parametrize('kwargs', [{'candidate_cap': 0}, {'candidate_cap': 257},
                                    {'numerical_budget': 5}, {'numerical_budget': -1},
                                    {'numerical': 'yes'}])
def test_invalid_report_budgets(kwargs):
    with pytest.raises((ValueError, TypeError)):
        v.analyze_neutral_sectors(3, 3, **kwargs)


def test_vacuum_and_nonvacuum_diagnostic_contract():
    vacuum = v.sector_ground_diagnostic(v.all_number_model(3, 0))
    assert vacuum['status'] == 'structural_vacuum'
    assert vacuum['ground_energy'] == 0
    model = v.all_number_model(3, 6, 1., .7)
    r = v.sector_ground_diagnostic(model)
    assert r['ground_energy'] == pytest.approx(np.linalg.eigvalsh(model.H)[0])
    assert r['dimension'] == 28
    assert r['multiplicity'] == 1
    assert r['gap'] > 0
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize('g', [0., .7, 40.])
def test_shifted_convention(g):
    r = v.shifted_convention_result(3, 3, 1., g)
    assert r['different_hamiltonian'] is True
    assert r['selected_sector'] == (0 if g else None)
    assert r['status'] == ('selected' if g else 'infinitely_many_tied_sectors')
    for N in [0, 3, 6]:
        a = v.all_number_model(3, N, 1., g)
        shifted = a.H+2*N*np.eye(len(a.basis))
        spectrum = np.linalg.eigvalsh(shifted)
        assert spectrum[0] >= -1e-12
        if g and N:
            assert spectrum[0] > 0
        if not g:
            assert spectrum[0] == pytest.approx(0., abs=1e-12)


def test_number_conservation_no_sector_preparation():
    models = [v.all_number_model(3, N, 1., 40.) for N in [0, 3, 6]]
    H = block_diag(*(m.H for m in models))
    number = block_diag(*(m.N*np.eye(len(m.basis)) for m in models))
    np.testing.assert_allclose(H @ number, number @ H, atol=0)
    z = np.arange(1, len(H)+1, dtype=complex)+.2j
    z /= np.linalg.norm(z)
    evolved = expm(-.3j*H) @ z
    start = 0
    for model in models:
        end = start+len(model.basis)
        assert np.linalg.norm(evolved[start:end])**2 == pytest.approx(np.linalg.norm(z[start:end])**2, abs=2e-14)
        start = end
    vacuum = np.eye(len(H), dtype=complex)[:, 0]
    np.testing.assert_allclose(expm(-.3j*H) @ vacuum, vacuum, atol=0)


@pytest.mark.parametrize('flags', [[], ['--json']])
def test_stdout_demo(tmp_path, flags):
    script = Path(__file__).resolve().parents[1]/'scripts/demo_substrate_vacuum_selection.py'
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    r = subprocess.run([sys.executable, str(script)]+flags, cwd=tmp_path, env=env,
                       text=True, capture_output=True, timeout=90)
    assert r.returncode == 0, r.stderr
    assert not r.stderr
    assert not list(tmp_path.iterdir())
    if flags:
        assert json.loads(r.stdout, parse_constant=lambda x: pytest.fail(x))
    else:
        assert 'sector' in r.stdout.lower()
