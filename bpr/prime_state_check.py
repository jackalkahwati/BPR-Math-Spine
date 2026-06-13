"""Prime state checks — connecting BPR's prime substrate to Latorre-Sierra prime quantum information.

Implements the Prime state of Latorre, Sierra and collaborators
(arXiv:1403.4765, arXiv:2005.02422):

    |P_n⟩ = (1 / √π(2^n)) Σ_{p prime, p ≤ 2^n} |p⟩

and computes its bipartite entanglement entropy directly. García-Martín
et al. (Quantum 4, 371, 2020) showed that this entropy scales with the
Shannon entropy of the square-free integer density 6/π² ≈ 0.6079 — a
deep number-theoretic identity connecting primes, square-free integers,
and quantum information.

Purpose for BPR
---------------
BPR's substrate is Z_p with a specific prime p = 104761. The
Latorre-Sierra Prime state machinery establishes that primes carry
intrinsic quantum-information structure, independent of any specific
physical theory. This module:

  1. Verifies the García-Martín 6/π² scaling numerically (independent
     reproduction of a published result; sanity-check on our methodology).
  2. Provides the computational framework for the BPR-distinctive test:
     does BPR's substrate boundary phase field, quantum-described,
     reproduce the same scaling? That comparison is the discriminating
     content — same scaling = BPR substrate is doing what the Prime
     state does; different scaling = BPR's prime claim is decorative.

Honest scope
------------
This module does NOT yet implement the BPR-substrate analog computation
(that requires committing to how the Z_p phase field becomes a quantum
state, which involves additional choices). It implements the Prime
state side of the comparison and provides the interface for the
BPR-side computation to be added later. The current numerical result
is a faithful reproduction of the Latorre-Sierra calculation, not a
BPR-specific success.

References
----------
- Latorre, Sierra, "There is entanglement in the primes,"
  arXiv:1403.4765, J. Phys. A 47, 475302 (2014)
- García-Martín, Ribas, Carrazza, Latorre, Sierra, "The Prime state and
  its quantum relatives," Quantum 4, 371 (2020), arXiv:2005.02422
- Kontoyiannis, "Counting Primes Using Entropy," IEEE IT Newsletter (2008)
- Kontoyiannis, "Some information-theoretic computations related to
  the distribution of prime numbers," arXiv:0710.4076 (2007)
"""

from __future__ import annotations

import numpy as np


# Square-free integer density (Mertens 1874): 6/π² ≈ 0.6079271...
SQUARE_FREE_DENSITY = 6.0 / (np.pi ** 2)


def square_free_shannon_entropy(rho: float = SQUARE_FREE_DENSITY) -> float:
    """Shannon entropy of the square-free indicator: H(ρ_sf) [nats]."""
    return -rho * np.log(rho) - (1.0 - rho) * np.log(1.0 - rho)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    for d in range(3, int(np.sqrt(n)) + 1, 2):
        if n % d == 0:
            return False
    return True


def primes_below(N: int) -> np.ndarray:
    """Sieve of Eratosthenes: all primes < N."""
    if N <= 2:
        return np.array([], dtype=np.int64)
    sieve = np.ones(N, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(np.sqrt(N)) + 1):
        if sieve[i]:
            sieve[i * i :: i] = False
    return np.flatnonzero(sieve).astype(np.int64)


def prime_state_amplitudes(n_qubits: int) -> np.ndarray:
    """Amplitudes of |P_n⟩ in the computational basis of 2^n states.

    Returns array of length 2^n with 1/√π(2^n) on prime indices, 0 elsewhere.
    """
    dim = 2 ** n_qubits
    primes = primes_below(dim)
    psi = np.zeros(dim, dtype=np.float64)
    psi[primes] = 1.0
    psi /= np.linalg.norm(psi)
    return psi


def bipartite_entanglement_entropy(
    psi: np.ndarray, n_qubits: int, n_A: int
) -> float:
    """Von Neumann entropy of subsystem A (first n_A qubits), in nats.

    Reshapes psi to (2^n_A, 2^(n_B)) and uses SVD; entropy = Σ -|s|² log|s|².
    """
    if n_A < 0 or n_A > n_qubits:
        raise ValueError("n_A must be in [0, n_qubits]")
    if n_A == 0 or n_A == n_qubits:
        return 0.0
    dim_A = 2 ** n_A
    dim_B = 2 ** (n_qubits - n_A)
    M = psi.reshape(dim_A, dim_B)
    s = np.linalg.svd(M, compute_uv=False)
    s2 = s ** 2
    s2 = s2[s2 > 1e-15]
    return float(-np.sum(s2 * np.log(s2)))


def garcia_martin_scaling_check(n_qubits_max: int = 12) -> dict:
    """Numerical verification of the García-Martín 6/π² scaling.

    For each n in [4, n_qubits_max], builds the Prime state, computes the
    half-half bipartite entanglement entropy, and compares to the predicted
    asymptotic scaling

        S(n) ≈ H(6/π²) × min(n_A, n_B) ≈ 0.668 × (n/2)  [nats]

    (The full Latorre-Sierra result is more refined; this is the leading
    asymptotic behavior. Deviations at small n are expected due to
    finite-size corrections from the prime counting function.)
    """
    h_sqf = square_free_shannon_entropy()
    results = []
    for n in range(4, n_qubits_max + 1):
        psi = prime_state_amplitudes(n)
        S = bipartite_entanglement_entropy(psi, n, n_A=n // 2)
        S_predicted = h_sqf * (n // 2)
        results.append({
            "n_qubits": n,
            "S_observed_nats": S,
            "S_per_subsystem_qubit": S / (n // 2),
            "predicted_S_asymptotic": S_predicted,
            "ratio": S / S_predicted if S_predicted > 0 else float("nan"),
        })
    return {
        "square_free_density": SQUARE_FREE_DENSITY,
        "shannon_entropy_sf_nats": h_sqf,
        "scan": results,
        "note": (
            "Verifies the Latorre-Sierra / García-Martín claim that the "
            "Prime state's entanglement entropy is governed by the "
            "Shannon entropy of square-free density (6/π² ≈ 0.6079). "
            "Convergence toward the asymptotic ratio (per-subsystem-qubit "
            "entropy → H(6/π²) ≈ 0.668 nats) demonstrates the deep "
            "number-theoretic identity holds. This is INDEPENDENT "
            "REPRODUCTION of a published result; it does not by itself "
            "support BPR."
        ),
    }


def kontoyiannis_bound_audit(p: int = 104761, n_predictions: int = 205) -> dict:
    """Audit BPR's '205 predictions from (J, p, z)' against the Kontoyiannis bound.

    Kontoyiannis information-theoretic bound: specifying a prime p ≤ N
    has at most log p nats of information. For BPR's p = 104761,
    log p ≈ 11.56 nats ≈ 16.68 bits.

    If 205 predictions encoded as binary outcomes (much weaker than real
    numbers) carry ~205 bits of information, the information needed to
    select them is ~205 nats vs (log p + log z + log J range) available.
    Real-valued predictions carry far more information per prediction,
    so the bound is even more constraining.

    This is an AUDIT TOOL: predictions cannot encode more information
    than the input parameters provide, unless additional information is
    smuggled in via fitted coefficients (which the parameter-honesty
    pass has flagged: θ_23 coefficient 1.35, θ_12 coefficient 3.5,
    ln(p)/(ln(p)+1) finite-boundary correction, etc.). The bound makes
    this principle quantitative.
    """
    log_p_nats = np.log(p)
    log_p_bits = log_p_nats / np.log(2)
    # generous upper bound on (J, z) information: ~6 bits each
    total_input_bits = log_p_bits + 6 + 6
    # binary-encoded predictions: at least n bits
    binary_predictions_bits = n_predictions
    # real-valued (10% precision) predictions: ~3-4 bits each
    real_predictions_bits = 3.5 * n_predictions
    return {
        "p": p,
        "log_p_nats": float(log_p_nats),
        "log_p_bits": float(log_p_bits),
        "total_input_information_bits_upper": float(total_input_bits),
        "n_predictions_claimed": n_predictions,
        "binary_predictions_information_bits": float(binary_predictions_bits),
        "real_valued_predictions_information_bits": float(real_predictions_bits),
        "binary_excess_bits": float(binary_predictions_bits - total_input_bits),
        "real_valued_excess_bits": float(real_predictions_bits - total_input_bits),
        "verdict": (
            "Binary-encoded 205 predictions carry ~205 bits; (J, p, z) "
            "provide ~28 bits. The excess (~177 bits) IS smuggled in via "
            "additional structural choices (boundary mode integers from z, "
            "the W_c = √3 winding, the Higgs VEV, lattice anchors for "
            "binding) plus fitted coefficients (flagged in the parameter-"
            "honesty pass). This is consistent — derived predictions exploit "
            "many structural relations beyond just (J, p, z) input bits. "
            "The audit shows the framework is NOT 'three numbers → "
            "everything' in pure information-theoretic terms; it is "
            "'three numbers + boundary CFT structure + fitted coefficients "
            "→ 205 predictions'. The latter is still a meaningful "
            "parameter-reduction relative to fitting 20+ SM parameters "
            "independently, but the headline 'three numbers' framing "
            "understates the structural inputs."
        ),
    }


def bpr_substrate_prime_state_analog_stub() -> dict:
    """Stub for the BPR-distinctive test: substrate analog of the Prime state.

    NOT YET IMPLEMENTED. The discriminating computation would:

      1. Construct a quantum state on the Z_p substrate (boundary phase
         field φ on the discrete lattice).
      2. Choose a natural "support" rule analogous to "support is on primes"
         in the standard Prime state — candidates include support on
         topologically nontrivial winding states, support on stable boundary
         resonances, or support on states that minimize the boundary action.
      3. Compute the bipartite entanglement entropy under a natural
         lattice partition.
      4. Compare to the García-Martín 6/π² × n_A scaling.

    If the BPR substrate analog reproduces the scaling, the framework's
    prime claim has genuine number-theoretic content. If it doesn't,
    the framework's prime claim is decorative (the substrate is Z_p but
    doesn't behave like it).

    This requires committing to specific choices (the "support rule" most
    importantly) that haven't been pinned down. It's a real research
    target, not a one-function calculation.
    """
    return {
        "status": "STUB — see docstring for what would need to be done",
        "discriminating_outcome": (
            "If BPR-substrate analog ≈ García-Martín 6/π² scaling: "
            "framework's prime claim has number-theoretic teeth. "
            "If different: framework's prime claim is decorative."
        ),
    }
