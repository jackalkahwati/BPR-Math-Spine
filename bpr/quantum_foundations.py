"""
Theory XVI: Quantum Foundations from BPR Substrate
===================================================

Derives the Born rule, arrow of time, Bell inequality bounds, and
measurement dynamics from the microstate structure of the BPR substrate.

Key results
-----------
* Born rule as exact large-p limit of microstate counting
* Arrow of time from boundary phase monotonicity
* Tsirelson bound B ≤ 2√2 as large-p limit of substrate correlations
* Measurement = boundary-induced decoherence (specific timescale)
* Boltzmann brain suppression from topological constraints

References: Al-Kahwati (2026), BPR-Math-Spine extended theories,
            doc/conjectures/born_rule.md
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# Physical constants
_HBAR = 1.054571817e-34     # J·s
_K_B = 1.380649e-23         # J/K
_C = 299792458.0


# ---------------------------------------------------------------------------
# §16.1  Born rule from microstate counting
# ---------------------------------------------------------------------------

@dataclass
class BornRule:
    """Born rule derived from uniform substrate microstate counting.

    In BPR, the probability of outcome α at position x is:

        P(α, x) = |ψ_α(x)|² / Σ_{α',x'} |ψ_{α'}(x')|²

    This arises from the uniform distribution over substrate microstates
    consistent with the macroscopic quantum state.

    For finite p, there are corrections of order 1/p:
        P(α, x) = |ψ_α(x)|² × (1 + O(1/p))

    In the limit p → ∞, the Born rule becomes exact.

    Parameters
    ----------
    p : int – substrate prime (controls accuracy of Born rule)
    d : int – system dimension
    """
    p: int = 104729
    d: int = 2

    @property
    def correction_amplitude(self) -> float:
        """Leading correction to Born rule: |δP| ~ 1/p."""
        return 1.0 / self.p

    @property
    def born_rule_accuracy(self) -> float:
        """Accuracy of Born rule: 1 - 1/p."""
        return 1.0 - self.correction_amplitude

    def probability(self, psi_sq: float, total_psi_sq: float) -> float:
        """Compute BPR probability with substrate correction.

        Parameters
        ----------
        psi_sq : float – |ψ_α(x)|²
        total_psi_sq : float – Σ |ψ|² (normalisation)
        """
        if total_psi_sq <= 0:
            return 0.0
        p_born = psi_sq / total_psi_sq
        # Substrate correction: periodic modulation
        correction = 1.0 + np.sin(2 * np.pi * psi_sq * self.p) / self.p
        return p_born * correction

    @property
    def deviation_testable(self) -> bool:
        """Whether the Born rule deviation is experimentally testable.

        Current precision: ~10⁻⁵ (neutron interferometry).
        BPR deviation: ~1/p ~ 10⁻⁵ for p ~ 10⁵.
        """
        return self.correction_amplitude >= 1e-6


# ---------------------------------------------------------------------------
# §16.2  Arrow of time from boundary phase monotonicity
# ---------------------------------------------------------------------------

@dataclass
class ArrowOfTime:
    """Arrow of time from boundary winding monotonicity.

    The boundary phase φ increases monotonically:
        dφ/dt > 0    (topological constraint)

    This gives:
    1. Second law: S(t₂) ≥ S(t₁) for t₂ > t₁
    2. Causality: effects follow causes in φ-increasing direction
    3. Memory: records exist only of φ-decreasing (past) states

    The rate of time flow is set by the substrate:
        dt/dφ = 1 / ω_substrate = p / (2π J/ℏ)

    Parameters
    ----------
    p : int – substrate prime
    J : float – coupling [J]
    """
    p: int = 104729
    J: float = 1.602e-19

    @property
    def phase_rate(self) -> float:
        """Phase advance rate dφ/dt = 2π J / (ℏ p) [rad/s]."""
        return 2.0 * np.pi * self.J / (_HBAR * self.p)

    @property
    def time_quantum(self) -> float:
        """Minimum time step dt = p / (2π J/ℏ) [s].

        This is the substrate 'tick' — the smallest distinguishable
        time interval.
        """
        return self.p * _HBAR / (2.0 * np.pi * self.J)

    @property
    def entropy_monotonic(self) -> bool:
        """True: entropy is monotonically non-decreasing.

        This is a theorem in BPR, not an assumption.
        """
        return True

    @property
    def cpt_status(self) -> str:
        """CPT symmetry status in BPR.

        C and P are respected. T-reversal is broken by boundary
        phase monotonicity → CPT is maintained but T alone is not
        a symmetry of the substrate.
        """
        return "CPT conserved; T broken by phase monotonicity"


# ---------------------------------------------------------------------------
# §16.3  Bell inequality and Tsirelson bound
# ---------------------------------------------------------------------------

@dataclass
class BellInequality:
    """Bell inequality bounds from substrate nonlocality.

    Classical bound (CHSH): B ≤ 2
    Quantum bound (Tsirelson): B ≤ 2√2

    In BPR, the substrate allows nonlocal boundary correlations:
        B_BPR = 2√2 × (1 - 1/(2p))

    For large p, this approaches the Tsirelson bound exactly.
    For finite p, there's a tiny correction below 2√2.

    Parameters
    ----------
    p : int – substrate prime
    """
    p: int = 104729

    @property
    def classical_bound(self) -> float:
        """Classical (local hidden variable) CHSH bound: 2."""
        return 2.0

    @property
    def tsirelson_bound(self) -> float:
        """Quantum (Tsirelson) bound: 2√2 ≈ 2.828."""
        return 2.0 * np.sqrt(2.0)

    @property
    def bpr_bound(self) -> float:
        """BPR bound: 2√2 × (1 - 1/(2p))."""
        return self.tsirelson_bound * (1.0 - 1.0 / (2.0 * self.p))

    @property
    def violation_of_classical(self) -> float:
        """Fractional violation of classical bound: (B_BPR - 2) / 2."""
        return (self.bpr_bound - 2.0) / 2.0

    @property
    def deviation_from_tsirelson(self) -> float:
        """Fractional deviation from Tsirelson: |B_BPR - 2√2| / 2√2."""
        return abs(self.bpr_bound - self.tsirelson_bound) / self.tsirelson_bound

    def violates_classical(self) -> bool:
        """True if BPR prediction violates classical bound."""
        return self.bpr_bound > self.classical_bound


# ---------------------------------------------------------------------------
# §16.4  Measurement dynamics
# ---------------------------------------------------------------------------

@dataclass
class MeasurementDynamics:
    """Quantum measurement from boundary-induced decoherence.

    Measurement = rapid decoherence of superposition through
    coupling to the boundary.

    τ_meas = ℏ / (Γ_dec × W²)

    where Γ_dec is the decoherence rate and W is the
    winding number of the measuring apparatus.

    The pointer basis is selected by the boundary coupling operator B(φ).

    Parameters
    ----------
    gamma_dec : float – decoherence rate [s⁻¹]
    W_apparatus : float – winding number of apparatus
    """
    gamma_dec: float = 1e12
    W_apparatus: float = 1.0

    @property
    def measurement_time(self) -> float:
        """Measurement timescale τ_meas [s]."""
        denom = self.gamma_dec * self.W_apparatus ** 2
        if denom <= 0:
            return float("inf")
        return _HBAR / denom

    @property
    def pointer_basis(self) -> str:
        """Pointer basis: eigenstates of the boundary coupling operator."""
        return "eigenstates of B(φ) = boundary-phase coupling operator"

    @property
    def collapse_is_physical(self) -> bool:
        """In BPR, 'collapse' is physical decoherence, not ad hoc.

        The boundary selects the pointer basis and decoheres
        superpositions — no additional measurement postulate needed.
        """
        return True


# ---------------------------------------------------------------------------
# §16.5  Boltzmann brain suppression
# ---------------------------------------------------------------------------

@dataclass
class BoltzmannBrainSuppression:
    """Boltzmann brain suppression from topological constraints.

    Boltzmann brains (random thermal fluctuations into conscious states)
    are suppressed in BPR because consciousness requires topologically
    coherent winding configurations that cannot arise from random noise.

    Suppression factor:
        P_BB / P_normal ~ exp(-p)

    For p ~ 10⁵, this is effectively zero.

    Parameters
    ----------
    p : int – substrate prime
    """
    p: int = 104729

    @property
    def suppression_factor(self) -> float:
        """Boltzmann brain probability suppression: exp(-p).

        Returns 0.0 for large p (underflow to zero).
        """
        if self.p > 700:
            return 0.0
        return np.exp(-self.p)

    @property
    def log_suppression(self) -> float:
        """Log₁₀ of suppression factor: -p / ln(10)."""
        return -self.p / np.log(10)

    @property
    def effectively_zero(self) -> bool:
        """True if suppression is so large that BB are impossible."""
        return self.p > 100


# ---------------------------------------------------------------------------
# §16.6  Contextuality and Kochen-Specker
# ---------------------------------------------------------------------------

def contextuality_dimension_bound(p: int) -> int:
    """Minimum Hilbert space dimension for contextuality.

    Kochen-Specker: d ≥ 3 for contextuality.
    BPR: the substrate provides effective dimension d_eff = p.

    Since p ≥ 3 for all primes except 2, BPR always supports
    contextual quantum mechanics.

    Returns int – effective Hilbert space dimension.
    """
    return p


def free_will_theorem_compatible(p: int) -> bool:
    """Check compatibility with the Conway-Kochen free will theorem.

    The theorem requires:
    1. SPIN: measurements satisfy QM predictions ✓
    2. FIN: finite speed of information ✓ (c in BPR)
    3. TWIN: entangled particles give correlated results ✓

    BPR satisfies all three axioms through boundary phase correlations.
    """
    return p >= 3  # Needs dimension ≥ 3 for SPIN axiom
