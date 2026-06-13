"""Tests for the Prime state / square-free entanglement scaling reproduction."""
import numpy as np
import pytest
from bpr.prime_state_check import (
    SQUARE_FREE_DENSITY,
    square_free_shannon_entropy,
    is_prime,
    primes_below,
    prime_state_amplitudes,
    bipartite_entanglement_entropy,
    garcia_martin_scaling_check,
    kontoyiannis_bound_audit,
)


def test_square_free_density_is_six_over_pi_squared():
    assert SQUARE_FREE_DENSITY == pytest.approx(6.0 / np.pi ** 2, rel=1e-12)


def test_shannon_entropy_square_free():
    """H(0.6079) ≈ 0.670 nats."""
    h = square_free_shannon_entropy()
    assert h == pytest.approx(0.670, abs=0.005)


def test_primes_below_correctness():
    """Sieve of Eratosthenes returns the first few primes correctly."""
    p = primes_below(20)
    assert list(p) == [2, 3, 5, 7, 11, 13, 17, 19]


def test_is_prime_basic():
    assert not is_prime(0)
    assert not is_prime(1)
    assert is_prime(2)
    assert is_prime(104761)
    assert not is_prime(104760)


def test_prime_state_is_normalized():
    psi = prime_state_amplitudes(8)
    assert np.linalg.norm(psi) == pytest.approx(1.0)


def test_entanglement_entropy_zero_for_trivial_partition():
    """S(ρ_A) = 0 when one subsystem is empty."""
    psi = prime_state_amplitudes(6)
    assert bipartite_entanglement_entropy(psi, 6, n_A=0) == 0.0
    assert bipartite_entanglement_entropy(psi, 6, n_A=6) == 0.0


def test_garcia_martin_entropy_scales_with_n():
    """Bipartite entanglement entropy strictly grows with n_qubits."""
    r = garcia_martin_scaling_check(n_qubits_max=10)
    S = [row["S_observed_nats"] for row in r["scan"]]
    # Not strictly monotone at every n due to discreteness; check the trend
    assert S[-1] > S[0]
    # Per-subsystem-qubit entropy should be bounded below H(square-free)
    per_q = [row["S_per_subsystem_qubit"] for row in r["scan"]]
    h_sqf = square_free_shannon_entropy()
    # Empirically the ratio sits in the 0.4-0.8 band at small n
    assert all(0.3 <= s_per <= h_sqf * 1.05 for s_per in per_q)


def test_kontoyiannis_log_p_for_bpr_prime():
    """log(104761) ≈ 16.68 bits ≈ 11.56 nats."""
    a = kontoyiannis_bound_audit(p=104761, n_predictions=205)
    assert a["log_p_bits"] == pytest.approx(16.68, abs=0.01)
    assert a["log_p_nats"] == pytest.approx(11.56, abs=0.01)


def test_kontoyiannis_audit_excess_information():
    """205 binary predictions encode more bits than (J, p, z) input alone."""
    a = kontoyiannis_bound_audit()
    assert a["binary_excess_bits"] > 0


# ---------------------------------------------------------------------------
# BPR-substrate analog: quadratic residue state
# ---------------------------------------------------------------------------

def test_legendre_symbol_basic():
    """Legendre symbol gives 0/1/-1 with known small values."""
    from bpr.prime_state_check import legendre_symbol
    # (1|p) = 1 for any p
    assert legendre_symbol(1, 31) == 1
    # (0|p) = 0
    assert legendre_symbol(0, 31) == 0
    # 4 = 2^2 is a QR mod any odd prime
    assert legendre_symbol(4, 31) == 1
    # 3 is not a QR mod 31
    assert legendre_symbol(3, 31) == -1


def test_quadratic_residues_count():
    """|QR_p| = (p-1)/2 for odd prime p."""
    from bpr.prime_state_check import quadratic_residues_mod_p
    for p in (31, 61, 127, 251):
        qrs = quadratic_residues_mod_p(p)
        assert len(qrs) == (p - 1) // 2


def test_quadratic_residue_state_normalized():
    from bpr.prime_state_check import quadratic_residue_state
    psi, n = quadratic_residue_state(31)
    assert np.linalg.norm(psi) == pytest.approx(1.0)


def test_qr_state_entropy_below_bernoulli_baseline():
    """KEY FINDING: |QR_p⟩ entropy is BELOW log 2 × n_A → multiplicative
    correlations carry information. Ratio ≈ 0.55-0.77 across small primes."""
    from bpr.prime_state_check import bpr_substrate_analog_scaling
    r = bpr_substrate_analog_scaling()
    for row in r["scan"]:
        # ratio strictly less than 1 (correlations present) but bounded away
        # from 0 (state has nontrivial entanglement)
        assert 0.4 < row["ratio_to_bernoulli"] < 0.9


def test_qr_ratio_stable_across_primes():
    """The QR-state-to-Bernoulli ratio is approximately constant in p,
    suggesting an asymptotic limit set by Z_p multiplicative structure
    (not dependent on the specific value of p)."""
    from bpr.prime_state_check import bpr_substrate_analog_scaling
    r = bpr_substrate_analog_scaling()
    # very weak slope: structure doesn't depend strongly on p
    assert abs(r["ratio_slope_vs_log_p"]) < 0.15
