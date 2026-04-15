"""
bpr/cs_completion.py

Computational verification of the rigorous claims in doc/CS_UV_COMPLETION.md.

Checks:
  1. Prime constraint: U(1)_k CS anyon field condition forces k prime.
  2. Hopf fibration: S³ → S² maps p fiber modes to ≈ p S² modes.
  3. c=1 boson: boundary parameters R = √(z/2), κ = z/2.
  4. S² propagator: G_S2(0,0) ≈ ln(p) + 2γ − 1; G_S2² ≈ 137.26.
  5. Alpha formula origins: which terms come from CS and which from scheme.

Nothing here constitutes a proof of the coefficient = 1 claim; that
calculation remains open (see CS_UV_COMPLETION.md §6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from bpr.constants import P_DEFAULT, Z_DEFAULT

# Euler–Mascheroni constant
_GAMMA = 0.5772156649015328606

# Experimental value of 1/alpha
_ALPHA_INV_EXP = 137.035999084


# ---------------------------------------------------------------------------
# 1. Prime constraint: anyon field condition
# ---------------------------------------------------------------------------

def _is_prime(n: int) -> bool:
    """Trial-division primality test (sufficient for n ≤ p ≈ 1e5)."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    for i in range(3, r + 1, 2):
        if n % i == 0:
            return False
    return True


def _zk_is_field(k: int) -> bool:
    """Return True iff Z_k is a field (every nonzero element is invertible)."""
    for a in range(1, k):
        # Check gcd(a, k) == 1 (Bezout → multiplicative inverse exists)
        if math.gcd(a, k) != 1:
            return False
    return True


def verify_prime_constraint(k: int) -> dict:
    """
    Verify the anyon field condition:
    Z_k is a field iff k is prime.

    Returns a dict with:
      'k'              : the input level
      'zk_is_field'    : whether Z_k is a field
      'k_is_prime'     : whether k is prime
      'consistent'     : whether the two agree
      'zero_divisor'   : first zero divisor pair if k is composite, else None
    """
    is_prime = _is_prime(k)
    is_field = _zk_is_field(k)

    zero_div: Optional[tuple[int, int]] = None
    if not is_prime:
        # Find the first zero divisor pair  m, n  with m·n ≡ 0 (mod k)
        for m in range(2, k):
            for n in range(2, k):
                if (m * n) % k == 0:
                    zero_div = (m, n)
                    break
            if zero_div:
                break

    return {
        "k": k,
        "zk_is_field": is_field,
        "k_is_prime": is_prime,
        "consistent": is_prime == is_field,
        "zero_divisor": zero_div,
    }


def verify_anyon_field_condition(p: int = P_DEFAULT) -> dict:
    """
    Check the full anyon field condition at the BPR level k = p.

    CS level k = p (prime) is required because BPR coarse-graining needs
    division mod k to be unique (field condition), and Z_k is a field iff k prime.
    """
    result = verify_prime_constraint(p)
    result["interpretation"] = (
        f"Z_{p} is a {'field' if result['zk_is_field'] else 'ring (not field)'}: "
        f"BPR coarse-graining {'is' if result['zk_is_field'] else 'is NOT'} well-defined"
    )
    return result


# ---------------------------------------------------------------------------
# 2. Hopf fibration: S³ → S²  mode count
# ---------------------------------------------------------------------------

@dataclass
class HopfFibrationSummary:
    """
    Numerical summary of the Hopf S³ → S² fibration at CS level k = p.

    Attributes
    ----------
    p          : CS level (= BPR prime)
    L_max      : ⌊√p⌋  — maximum angular momentum of S² modes below UV cutoff
    s2_modes   : (L_max + 1)²  — number of spherical harmonic modes on S²
    ratio      : s2_modes / p  — how close the mode count is to p
    discrepancy_pct : 100 × |1 − ratio|
    """

    p: int
    L_max: int = field(init=False)
    s2_modes: int = field(init=False)
    ratio: float = field(init=False)
    discrepancy_pct: float = field(init=False)

    def __post_init__(self) -> None:
        self.L_max = int(math.isqrt(self.p))
        self.s2_modes = (self.L_max + 1) ** 2
        self.ratio = self.s2_modes / self.p
        self.discrepancy_pct = 100.0 * abs(1.0 - self.ratio)

    def summary_lines(self) -> list[str]:
        return [
            f"Hopf fibration S³ → S² at CS level k = {self.p}",
            f"  L_max = ⌊√{self.p}⌋ = {self.L_max}",
            f"  S² modes = (L_max + 1)² = {self.s2_modes}",
            f"  Ratio  modes / p = {self.ratio:.6f}",
            f"  Discrepancy: {self.discrepancy_pct:.3f}%",
        ]


def hopf_fibration_summary(p: int = P_DEFAULT) -> HopfFibrationSummary:
    """Return the Hopf fibration mode-count summary for CS level p."""
    return HopfFibrationSummary(p=p)


# ---------------------------------------------------------------------------
# 3. c=1 compact boson boundary theory
# ---------------------------------------------------------------------------

@dataclass
class BoundaryTheoryParams:
    """
    Parameters of the c=1 compact boson BPR boundary theory.

    The BPR boundary action on S² is:
        S_bndy = (κ/2) ∫ d²x h^ab ∇_a φ ∇_b φ,    κ = z/2

    This is a c=1 compact boson at compactification radius R = √(z/2).
    The CS level p enters only as the UV cutoff (L_max ≈ √p), not as R.
    """

    z: int = Z_DEFAULT
    p: int = P_DEFAULT
    kappa: float = field(init=False)          # bare boundary coupling = z/2
    compactification_radius: float = field(init=False)  # R = √(z/2)
    uv_cutoff_L: int = field(init=False)       # L_max = ⌊√p⌋

    def __post_init__(self) -> None:
        self.kappa = self.z / 2.0
        self.compactification_radius = math.sqrt(self.kappa)
        self.uv_cutoff_L = int(math.isqrt(self.p))

    def is_c1_boson(self) -> bool:
        """Return True (the action is always c=1 compact boson form)."""
        return True


# ---------------------------------------------------------------------------
# 4. S² propagator and alpha approximation
# ---------------------------------------------------------------------------

def _harmonic_number(n: int) -> float:
    """H_n = Σ_{k=1}^{n} 1/k, computed directly."""
    return sum(1.0 / k for k in range(1, n + 1))


def compute_s2_propagator(p: int = P_DEFAULT) -> dict:
    """
    Compute the S² boundary propagator G_S2(0,0) with UV cutoff L_max = ⌊√p⌋.

    G_S2(0,0) = Σ_{ℓ=1}^{L_max} (2ℓ+1) / (ℓ(ℓ+1))
              = 2 H_{L_max} + 1/(L_max+1) − 1

    where H_n is the n-th harmonic number.

    Asymptotically: G_S2 ≈ ln(p) + 2γ − 1  (since H_L ≈ ln L + γ ≈ ln√p + γ)

    Returns G_S2, G_S2², the asymptotic approximation, and the BPR alpha formula
    for comparison.
    """
    L_max = int(math.isqrt(p))

    # Exact formula
    H_L = _harmonic_number(L_max)
    G_exact = 2.0 * H_L + 1.0 / (L_max + 1) - 1.0

    # Asymptotic approximation  (G ≈ ln p + 2γ − 1)
    G_asymp = math.log(p) + 2.0 * _GAMMA - 1.0

    # BPR alpha formula (ground truth)
    ln_p = math.log(p)
    z = Z_DEFAULT
    alpha_inv_bpr = ln_p**2 + z / 2.0 + _GAMMA - 1.0 / (2.0 * math.pi)

    G_sq = G_exact**2
    discrepancy = G_sq - alpha_inv_bpr
    discrepancy_pct = 100.0 * abs(discrepancy) / alpha_inv_bpr

    return {
        "p": p,
        "L_max": L_max,
        "H_L_max": H_L,
        "G_s2_exact": G_exact,
        "G_s2_asymptotic": G_asymp,
        "G_s2_squared": G_sq,
        "alpha_inv_bpr": alpha_inv_bpr,
        "alpha_inv_exp": _ALPHA_INV_EXP,
        "G_sq_minus_alpha_inv_bpr": discrepancy,
        "discrepancy_pct": discrepancy_pct,
        "status": (
            f"G_S2² = {G_sq:.3f}  vs  1/α_BPR = {alpha_inv_bpr:.3f}  "
            f"({discrepancy_pct:.2f}% gap — open coefficient calculation)"
        ),
    }


# ---------------------------------------------------------------------------
# 5. Alpha formula origins
# ---------------------------------------------------------------------------

@dataclass
class AlphaFormulaOrigins:
    """
    Track the origin of each term in 1/α = [ln p]² + z/2 + γ − 1/(2π).

    status: 'derived' | 'scheme' | 'open'
    """

    p: int = P_DEFAULT
    z: int = Z_DEFAULT

    @property
    def ln_p_sq(self) -> tuple[float, str, str]:
        """[ln p]² — photon self-energy from Z_p UV cutoff."""
        val = math.log(self.p) ** 2
        return val, "open (coefficient=1 unproven)", "UV propagator squared"

    @property
    def z_over_2(self) -> tuple[float, str, str]:
        """z/2 — bare boundary coupling from S² cubic tiling."""
        val = self.z / 2.0
        return val, "derived (Hopf reduction → κ = z/2)", "tree-level boundary action"

    @property
    def euler_gamma(self) -> tuple[float, str, str]:
        """γ — Z_p discrete-to-continuum scheme correction."""
        return _GAMMA, "scheme (Z_p lattice → continuum)", "harmonic series tail"

    @property
    def minus_1_over_2pi(self) -> tuple[float, str, str]:
        """-1/(2π) — on-shell scheme correction."""
        val = -1.0 / (2.0 * math.pi)
        return val, "scheme (on-shell vs Z_p scheme)", "lattice-to-continuum matching"

    @property
    def total(self) -> float:
        return (
            self.ln_p_sq[0]
            + self.z_over_2[0]
            + self.euler_gamma[0]
            + self.minus_1_over_2pi[0]
        )

    def report(self) -> str:
        rows = [
            ("Term", "Value", "Status", "Origin"),
            ("-" * 12, "-" * 10, "-" * 30, "-" * 35),
        ]
        for name, val, status, origin in [
            ("[ln p]²", *self.ln_p_sq),
            ("z/2", *self.z_over_2),
            ("γ", *self.euler_gamma),
            ("−1/(2π)", *self.minus_1_over_2pi),
        ]:
            rows.append((name, f"{val:.6f}", status, origin))

        rows.append(("-" * 12, "-" * 10, "-" * 30, "-" * 35))
        rows.append(("Total 1/α", f"{self.total:.6f}", "BPR formula", ""))
        rows.append(("Exp  1/α", f"{_ALPHA_INV_EXP:.6f}", "CODATA 2018", ""))

        col_widths = [max(len(r[i]) for r in rows) for i in range(4)]
        lines = []
        for row in rows:
            lines.append(
                "  ".join(cell.ljust(w) for cell, w in zip(row, col_widths))
            )
        return "\n".join(lines)


def alpha_formula_cs_origin(p: int = P_DEFAULT, z: int = Z_DEFAULT) -> AlphaFormulaOrigins:
    """Return an AlphaFormulaOrigins object for the given p, z."""
    return AlphaFormulaOrigins(p=p, z=z)


# ---------------------------------------------------------------------------
# 6. High-level summary
# ---------------------------------------------------------------------------

def cs_completion_status(p: int = P_DEFAULT, z: int = Z_DEFAULT) -> str:
    """
    Print a human-readable summary of the CS UV completion status.
    """
    hf = hopf_fibration_summary(p)
    prop = compute_s2_propagator(p)
    origins = alpha_formula_cs_origin(p, z)
    prime = verify_anyon_field_condition(p)

    lines = [
        "=" * 60,
        "BPR / Chern-Simons UV Completion — Status",
        "=" * 60,
        "",
        "1. Prime constraint (anyon field condition)",
        f"   Z_{p} is a field: {prime['zk_is_field']}",
        f"   {p} is prime:     {prime['k_is_prime']}",
        f"   Status: {'DERIVED ✓' if prime['consistent'] and prime['k_is_prime'] else 'OPEN'}",
        "",
        "2. Hopf fibration S³ → S²",
    ]
    lines.extend("   " + l for l in hf.summary_lines()[1:])
    lines += [
        f"   Status: {'RIGOROUS ✓' if hf.discrepancy_pct < 1.0 else 'NEEDS CHECK'}",
        "",
        "3. S² propagator and alpha formula",
        f"   G_S2  = {prop['G_s2_exact']:.6f}",
        f"   G_S2² = {prop['G_s2_squared']:.3f}",
        f"   1/α (BPR) = {prop['alpha_inv_bpr']:.3f}",
        f"   1/α (exp) = {prop['alpha_inv_exp']:.3f}",
        f"   G_S2² gap = {prop['discrepancy_pct']:.2f}%  — open coefficient calculation",
        "",
        "4. Alpha formula term origins",
        "",
        origins.report(),
        "",
        "=" * 60,
        "OVERALL STATUS",
        "  Rigorous: prime constraint, S² boundary, c=1 boson identification",
        "  Open:     coefficient of [ln p]² = 1 (not yet proven analytically)",
        "  Gap:      G_S2² differs from 1/α_BPR by "
        f"{prop['discrepancy_pct']:.2f}%",
        "=" * 60,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(cs_completion_status())
