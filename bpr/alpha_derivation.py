"""
Fine Structure Constant from Substrate
=====================================================

Derives the electromagnetic fine structure constant α from BPR
substrate parameters (p, z) using zero experimental coupling inputs.

STATUS (2026-10): ANSATZ, CALIBRATED.  The formula below is an ansatz:
its four terms are motivated, not derived from a boundary action.  p is
chosen so that the formula reproduces the measured α (inverse
calibration; unification-map node ``fine_structure_constant``, status
"calibrated"), so the agreement with experiment is an input, not a
test.  The real root of the formula is p* ≈ 104749.03, and α fixes p
only to within a prime gap.  The "zero experimental coupling inputs"
wording above is withdrawn.

**MAIN RESULT:**

    1/α = [ln(p)]² + z/2 + γ − 1/(2π)

For p = 104761, z = 6:

    1/α = 133.621 + 3.000 + 0.577 − 0.159 = 137.0386
    Experimental: 1/α = 137.035999084(21)

    Deviation: +0.0019%  (+19.3 ppm)

(Corrected 2026-10.  This block used to print 133.613, 137.031 and
"0.004% (36 ppm)".  Those are the p = 104729 arithmetic, the module's
default before p moved to 104761: 1/α(104729) = 137.0316, truncated to
137.031, which gives 36 ppm instead of 32 ppm.)


DERIVATION (BPR §22)  [ANSATZ: the term origins below are motivations]
────────────────────

The four terms have distinct physical origins:

1. [ln(p)]² — Z_p phase-space screening
   On the Z_p lattice, each node has p discrete phase states.
   The gauge self-energy (vacuum polarization) from summing over
   these states yields a screening contribution proportional to
   [ln(p)]².  Physically, ξ = a√(ln p) is the correlation length;
   the gauge coupling involves two propagators, each scaling as
   ξ²/a² = ln(p).  Their product gives [ln(p)]².

2. z/2 = κ — bare boundary coupling (tree level)
   The dimensionless boundary rigidity κ = z/2 sets the bare
   (unscreened) inverse coupling.  With z = 6 (S² geometry), κ = 3.

3. γ (Euler–Mascheroni) — lattice regularisation constant
   The discrete sum over Z_p modes differs from the continuum
   integral by γ ≈ 0.577.  This arises from the harmonic-number
   asymptotics  Σ_{k=1}^{p−1} 1/k = ln(p−1) + γ + O(1/p)  and is
   the standard lattice→continuum finite part.

4. −1/(2π) — on-shell scheme correction
   The continuum on-shell renormalisation scheme differs from the
   lattice scheme by −1/(2π).  This is the finite constant in the
   one-loop lattice–to–MS̄ matching integral.

Higher-order corrections are O(1/p^{1/3}) ≈ 0.02, consistent
with the observed 36 ppm residual.

(2026-10: the residual at p = 104761 is +0.0026 in 1/α, +19.3 ppm.  It
is the distance from p = 104761 to the real root p* ≈ 104749.03
(Δp ≈ 12, with d(1/α)/dp = 2 ln p / p ≈ 2.2 × 10⁻⁴).  It measures where
the calibrating prime sits, not the size of higher-order corrections.)


Key results
-----------
* 1/α(0) ≈ 137.0386   (exp: 137.036, +19.3 ppm; calibrated, see STATUS)
  [was "137.031, deviation 0.004%": the p = 104729 arithmetic]
* 1/α(M_Z) ≈ 127.955  (exp: 127.952, deviation 0.002%; uses the SM
  vacuum-polarisation input Δ(1/α) = 9.084)
* α_GUT from lattice gauge-link counting
  [ANSATZ α_GUT = π/(p^{1/3} z); the bare forward chain FAILS, see below]
* sin²θ_W consistent with SM running
  [WITHDRAWN 2026-10: the bare chain gives sin²θ_W ≈ 0.274]

Bare GUT chain (2026-10)
------------------------
The ansatz α_GUT = π/(p^{1/3} z) imposed at M_GUT = M_Pl/p^{1/4}
(``alpha_em_from_gut_running``) gives, at p = 104761, z = 6:
1/α_GUT = 90.03, α_s(M_Z) = 0.0203, sin²θ_W = 0.2739 and
1/α_em(M_Z) = 261.4, against 0.1179, 0.23122 and 127.95.  Imposing it at
M_Pl instead gives 0.0217, 0.2667 and 263.1, so the placement does not
rescue it.  This is a correct negative result about an ansatz
(introduced in e7efce9).  The 0.1179 / 0.23122 / 127.952 returned by
``bpr.gauge_unification.GaugeCouplingRunning`` are its M_Z inputs,
returned by a backward fit for every p, not outputs of this chain.
Found in the 2026-10 review of unmerged branches (GUT audit on
claude/audit-bpr-parameters-6fSrQ); see doc/CLOSED_AND_DEPRECATED.md §1c.


References: Al-Kahwati (2026), BPR-Math-Spine §22
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ── Mathematical constants ──────────────────────────────────────────────
_EULER_GAMMA = 0.5772156649015329       # Euler–Mascheroni constant

# ── Experimental reference values (NOT inputs to derivation) ────────────
_INV_ALPHA_0_EXP = 137.035999084        # CODATA 2018  (q² = 0)
_ALPHA_0_EXP = 1.0 / _INV_ALPHA_0_EXP
_INV_ALPHA_MZ_EXP = 127.952             # PDG 2024  (q² = M_Z²)
_ALPHA_MZ_EXP = 1.0 / _INV_ALPHA_MZ_EXP
_ALPHA_S_MZ_EXP = 0.1179                # PDG 2024  strong coupling
_SIN2_TW_EXP = 0.23122                  # PDG 2024  MS̄-bar

# ── QED vacuum polarisation (fermion-loop running, well-measured) ───────
# Δ(1/α) = 1/α(0) − 1/α(M_Z)  ≈  9.08
# Composed of leptonic (≈4.3), hadronic (≈3.8), top (≈−0.01) pieces.
_DELTA_INV_ALPHA_QED = 9.084

# ── SM one-loop beta coefficients (GUT normalisation for α₁) ───────────
_B1 = 41.0 / 10.0     # U(1)_Y
_B2 = -19.0 / 6.0     # SU(2)_L
_B3 = -7.0             # SU(3)_C

# ── Standard scales ────────────────────────────────────────────────────
_M_PL_GEV = 1.22093e19   # Planck mass  [GeV]
_M_Z_GEV = 91.1876        # Z-boson mass [GeV]


# =====================================================================
# §22.1  Core formula:  1/α  from substrate
# =====================================================================

def inverse_alpha_from_substrate(p: int = 104761, z: int = 6) -> float:
    """Derive 1/α_EM(q² = 0) from substrate parameters.

    STATUS (2026-10): ANSATZ, calibrated (p is chosen to reproduce α; see
    the module STATUS).  At p = 104761, z = 6 it returns 137.0386
    (+19.3 ppm vs CODATA 2018).

    FORMULA
    -------
        1/α  =  [ln(p)]²  +  z/2  +  γ  −  1/(2π)

    The four terms are:
        screening  = [ln(p)]²       Z_p phase-space self-energy
        bare       = z/2  = κ       boundary rigidity (tree level)
        lattice    = γ              lattice→continuum constant
        scheme     = −1/(2π)        on-shell matching

    Parameters
    ----------
    p : int
        Substrate prime modulus  (default 104 761; the docstring said
        104 729 until 2026-10).
    z : int
        Lattice coordination number  (default 6, S² topology).

    Returns
    -------
    float
        Predicted 1/α at q² = 0.
    """
    ln_p = np.log(p)

    screening = ln_p ** 2                   # Z_p phase screening
    bare = z / 2.0                          # boundary rigidity  κ = z/2
    lattice = _EULER_GAMMA                  # Euler–Mascheroni
    scheme = -1.0 / (2.0 * np.pi)          # on-shell correction

    return screening + bare + lattice + scheme


def alpha_em_from_substrate(p: int = 104761, z: int = 6) -> float:
    """Derive α_EM(q² = 0) from substrate parameters.

    Returns
    -------
    float
        Predicted fine structure constant at q² = 0.
    """
    return 1.0 / inverse_alpha_from_substrate(p, z)


# =====================================================================
# §22.2  Running to M_Z
# =====================================================================

def inverse_alpha_at_MZ(p: int = 104761, z: int = 6) -> float:
    """Derive 1/α_EM(M_Z) from substrate + standard QED running.

    1/α(M_Z)  =  1/α(0)  −  Δ(1/α)_QED

    where Δ(1/α)_QED ≈ 9.08 from vacuum polarisation (lepton and
    hadron loops).  This running depends on SM particle masses, which
    BPR derives elsewhere.

    (2026-10: Δ(1/α)_QED = 9.084 is a hard-coded SM input here, not
    computed from BPR masses.)

    Returns
    -------
    float
        Predicted 1/α at q² = M_Z².
    """
    return inverse_alpha_from_substrate(p, z) - _DELTA_INV_ALPHA_QED


def alpha_em_at_MZ(p: int = 104761, z: int = 6) -> float:
    """Derive α_EM(M_Z) from substrate + QED running."""
    return 1.0 / inverse_alpha_at_MZ(p, z)


# =====================================================================
# §22.3  Individual gauge couplings at M_Z (top-down)
# =====================================================================

def alpha_gut_from_lattice(p: int = 104761, z: int = 6) -> float:
    """Derive the unified gauge coupling at M_GUT from substrate.

    DERIVATION (BPR §22.3)
    ──────────────────────
    At the GUT scale M_GUT = M_Pl / p^{1/4}, the p^{1/3} ≈ 47
    boundary modes and their z gauge connections define the
    effective 't Hooft coupling:

        N_eff  = p^{1/3} × z / (2π)
        α_GUT  = 1 / (2 × N_eff)
               = π / (p^{1/3} × z)

    This gives  1/α_GUT ≈ p^{1/3} × z / π ≈ 90.0  for the lattice
    bare coupling.

    NOTE: The effective α_GUT *at* M_GUT (after boundary-mode
    threshold corrections between M_Pl and M_GUT) differs from this
    bare value.

    (2026-10: the note above used to point to a function
    ``derive_couplings_top_down`` "for the full chain".  No such function
    exists in this repository, and no code computes a threshold-corrected
    chain from this value.)

    STATUS (2026-10): ANSATZ, and it fails.  Introduced in e7efce9.
    - Scale: the docstring used to call this both the coupling "at M_GUT"
      and the "bare unified coupling at the Planck (lattice) scale".  The
      only code that uses it, ``alpha_em_from_gut_running``, imposes it
      at M_GUT = M_Pl/p^{1/4}; that is the reading kept here.  Imposing it
      at M_Pl fails as well (α_s ≈ 0.022 at M_Z).
    - Failure: at M_GUT it gives α_s(M_Z) ≈ 0.020, sin²θ_W ≈ 0.274 and
      1/α_em(M_Z) ≈ 261 (measured 0.1179, 0.23122, 127.95).
    - It is one of three inconsistent unified couplings in the legacy
      code: 1/α_GUT ≈ 90.0 here, ≈ 48.6 from the backward fit in
      ``bpr.gauge_unification.GaugeCouplingRunning.alpha_gut``, and 40
      hard-coded in ``bpr.gauge_unification.ProtonDecay``.

    Parameters
    ----------
    p : int
        Substrate prime modulus.
    z : int
        Coordination number.

    Returns
    -------
    float
        Bare unified coupling at the Planck (lattice) scale.
        [2026-10: used at M_GUT by ``alpha_em_from_gut_running``; see
        STATUS above.]
    """
    n_modes = p ** (1.0 / 3.0)
    return np.pi / (n_modes * z)


def gut_scale_GeV(p: int = 104761) -> float:
    """M_GUT = M_Pl / p^{1/4}  [GeV].

    ANSATZ (2026-10).  ≈ 6.79 × 10¹⁷ GeV at p = 104761 (not ~2 × 10¹⁶ GeV,
    as some legacy docstrings said).
    """
    return _M_PL_GEV / p ** 0.25


# =====================================================================
# §22.4  Decomposition: term-by-term breakdown
# =====================================================================

@dataclass
class AlphaBreakdown:
    """Term-by-term decomposition of the 1/α prediction."""

    p: int
    z: int

    screening: float        # [ln(p)]²
    bare: float             # z/2
    lattice_correction: float   # γ
    scheme_correction: float    # −1/(2π)

    inv_alpha_predicted: float
    alpha_predicted: float

    inv_alpha_experiment: float = _INV_ALPHA_0_EXP
    alpha_experiment: float = _ALPHA_0_EXP

    @property
    def deviation_ppm(self) -> float:
        """Deviation from experiment in parts per million."""
        return abs(self.inv_alpha_predicted - self.inv_alpha_experiment) / \
            self.inv_alpha_experiment * 1e6

    @property
    def deviation_percent(self) -> float:
        """Deviation from experiment in percent."""
        return abs(self.inv_alpha_predicted - self.inv_alpha_experiment) / \
            self.inv_alpha_experiment * 100.0

    @property
    def deviation_sigma(self) -> float:
        """Deviation in units of experimental uncertainty.

        CODATA 2018:  α⁻¹ = 137.035999084(21)
        """
        exp_unc = 0.000000021
        return abs(self.inv_alpha_predicted - self.inv_alpha_experiment) / exp_unc

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "═══════════════════════════════════════════════════════════",
            "  BPR §22 — Fine Structure Constant from Substrate",
            "═══════════════════════════════════════════════════════════",
            f"  Substrate:  p = {self.p},  z = {self.z}  (κ = {self.z/2})",
            "",
            "  1/α  =  [ln(p)]²  +  z/2  +  γ  −  1/(2π)",
            "",
            f"  [ln({self.p})]²   = {self.screening:12.6f}   (Z_p screening)",
            f"  z/2 = κ          = {self.bare:12.6f}   (bare coupling)",
            f"  γ  (Euler–Masch) = {self.lattice_correction:12.6f}   (lattice reg.)",
            f"  −1/(2π)          = {self.scheme_correction:12.6f}   (scheme corr.)",
            f"  ─────────────────────────────────",
            f"  1/α  (predicted) = {self.inv_alpha_predicted:12.6f}",
            f"  1/α  (CODATA)    = {self.inv_alpha_experiment:12.6f}",
            "",
            f"  Deviation:  {self.deviation_ppm:.1f} ppm  "
            f"({self.deviation_percent:.4f}%)",
            f"  In experimental σ:  {self.deviation_sigma:.0f}σ",
            "  (ANSATZ, calibrated: p is chosen to reproduce α, so the",
            "   residual reflects the choice of prime, not a theory error)",
            "═══════════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


def alpha_breakdown(p: int = 104761, z: int = 6) -> AlphaBreakdown:
    """Compute full term-by-term breakdown of the α derivation.

    Parameters
    ----------
    p : int
        Substrate prime modulus.
    z : int
        Coordination number.

    Returns
    -------
    AlphaBreakdown
        Complete decomposition with comparison to experiment.
    """
    ln_p = np.log(p)
    screening = ln_p ** 2
    bare = z / 2.0
    lattice = _EULER_GAMMA
    scheme = -1.0 / (2.0 * np.pi)
    inv_alpha = screening + bare + lattice + scheme

    return AlphaBreakdown(
        p=p, z=z,
        screening=screening,
        bare=bare,
        lattice_correction=lattice,
        scheme_correction=scheme,
        inv_alpha_predicted=inv_alpha,
        alpha_predicted=1.0 / inv_alpha,
    )


# =====================================================================
# §22.5  Full derivation result
# =====================================================================

@dataclass
class AlphaDerivationResult:
    """Complete result of α derivation from substrate."""

    # ── Substrate inputs ──
    p: int
    z: int

    # ── Predicted at q² = 0 ──
    inv_alpha_0: float
    alpha_0: float

    # ── Predicted at M_Z ──
    inv_alpha_MZ: float
    alpha_MZ: float

    # ── GUT-scale quantities ──
    alpha_gut_bare: float       # bare lattice coupling (ANSATZ; used at M_GUT, see alpha_gut_from_lattice)
    M_gut_GeV: float            # GUT scale

    # ── Experimental comparison ──
    inv_alpha_0_exp: float = _INV_ALPHA_0_EXP
    inv_alpha_MZ_exp: float = _INV_ALPHA_MZ_EXP

    @property
    def deviation_0_percent(self) -> float:
        return abs(self.inv_alpha_0 - self.inv_alpha_0_exp) / \
            self.inv_alpha_0_exp * 100.0

    @property
    def deviation_MZ_percent(self) -> float:
        return abs(self.inv_alpha_MZ - self.inv_alpha_MZ_exp) / \
            self.inv_alpha_MZ_exp * 100.0


def derive_alpha(p: int = 104761, z: int = 6) -> AlphaDerivationResult:
    """End-to-end derivation of α from substrate parameters.

    Chain:
        (p, z)  →  1/α(0)  →  QED running  →  1/α(M_Z)

    Parameters
    ----------
    p : int
        Substrate prime modulus.
    z : int
        Coordination number.

    Returns
    -------
    AlphaDerivationResult
        Complete derivation with comparison to experiment.
    """
    inv_a0 = inverse_alpha_from_substrate(p, z)
    inv_a_mz = inv_a0 - _DELTA_INV_ALPHA_QED
    a_gut = alpha_gut_from_lattice(p, z)
    m_gut = gut_scale_GeV(p)

    return AlphaDerivationResult(
        p=p, z=z,
        inv_alpha_0=inv_a0,
        alpha_0=1.0 / inv_a0,
        inv_alpha_MZ=inv_a_mz,
        alpha_MZ=1.0 / inv_a_mz,
        alpha_gut_bare=a_gut,
        M_gut_GeV=m_gut,
    )


# =====================================================================
# §22.6  Sensitivity analysis:  what prime gives exact α?
# =====================================================================

def prime_for_exact_alpha(z: int = 6, target_inv_alpha: float = _INV_ALPHA_0_EXP) -> float:
    """Compute the prime p that would give *exact* agreement with α.

    Solves:  [ln(p)]² + z/2 + γ − 1/(2π) = target

    Parameters
    ----------
    z : int
        Coordination number.
    target_inv_alpha : float
        Target 1/α (default: CODATA experimental value).

    Returns
    -------
    float
        The (generally non-integer) p that gives exact α.
        Compare with the framework's p = 104 761 (the docstring said
        104 729 until 2026-10).  p* ≈ 104749.03 for z = 6.
    """
    residual = target_inv_alpha - z / 2.0 - _EULER_GAMMA + 1.0 / (2.0 * np.pi)
    if residual < 0:
        return float("nan")
    ln_p = np.sqrt(residual)
    return np.exp(ln_p)


def sensitivity_to_prime(p: int = 104761, z: int = 6) -> dict:
    """How sensitive is α to small changes in p?

    Returns
    -------
    dict
        d(1/α)/dp, d(1/α)/d(ln p), and the p that gives exact α.
    """
    ln_p = np.log(p)
    # d(1/α)/d(ln p) = 2 ln(p)
    d_inv_alpha_d_lnp = 2.0 * ln_p
    # d(1/α)/dp = 2 ln(p) / p
    d_inv_alpha_dp = 2.0 * ln_p / p

    p_exact = prime_for_exact_alpha(z)

    return {
        "d_inv_alpha_d_ln_p": d_inv_alpha_d_lnp,
        "d_inv_alpha_d_p": d_inv_alpha_dp,
        "p_for_exact_alpha": p_exact,
        "delta_p_from_exact": p - p_exact,
        "delta_p_percent": (p - p_exact) / p_exact * 100.0,
    }


# =====================================================================
# §22.7  Consistency check: α from GUT running
# =====================================================================

def alpha_em_from_gut_running(p: int = 104761, z: int = 6) -> dict:
    """Cross-check: derive α_EM via GUT running (top-down).

    This is independent of §22.1 and uses:
        1. α_GUT from lattice gauge-link counting
        2. Threshold corrections from p^{1/3} boundary modes
        3. SM 1-loop running from M_GUT to M_Z

    (2026-10: item 2 is not implemented; the code applies no threshold
    corrections, as the inline comment and the returned note say.)

    RESULT (2026-10): the bare chain FAILS.  At p = 104761, z = 6:
    1/α_GUT = 90.03, α_s(M_Z) = 0.0203, sin²θ_W = 0.2739,
    1/α_em(M_Z) = 261.4 (measured 0.1179, 0.23122, 127.95).  The
    returned ``status`` says so.  Locked by tests/test_alpha_derivation.py.

    Returns
    -------
    dict
        Predicted couplings and comparison with both §22.1 and experiment.
    """
    a_gut = alpha_gut_from_lattice(p, z)
    inv_a_gut = 1.0 / a_gut
    m_gut = gut_scale_GeV(p)
    L = np.log(m_gut / _M_Z_GEV)

    # SM running from M_GUT to M_Z  (no threshold corrections here)
    inv_a1_mz = inv_a_gut + _B1 / (2.0 * np.pi) * L
    inv_a2_mz = inv_a_gut + _B2 / (2.0 * np.pi) * L
    inv_a3_mz = inv_a_gut + _B3 / (2.0 * np.pi) * L

    # α_EM from α₁ and α₂
    if inv_a1_mz > 0 and inv_a2_mz > 0:
        a1 = 1.0 / inv_a1_mz
        a2 = 1.0 / inv_a2_mz
        a_Y = (3.0 / 5.0) * a1           # hypercharge coupling
        sin2_tw = a_Y / (a_Y + a2)
        alpha_em = a2 * sin2_tw
        inv_alpha_em = 1.0 / alpha_em if alpha_em > 0 else float("inf")
    else:
        sin2_tw = float("nan")
        alpha_em = float("nan")
        inv_alpha_em = float("nan")

    return {
        "inv_alpha_gut": inv_a_gut,
        "M_gut_GeV": m_gut,
        "inv_alpha_1_MZ": inv_a1_mz,
        "inv_alpha_2_MZ": inv_a2_mz,
        "inv_alpha_3_MZ": inv_a3_mz,
        "sin2_theta_W": sin2_tw,
        "inv_alpha_em_MZ": inv_alpha_em,
        "alpha_s_MZ": 1.0 / inv_a3_mz if inv_a3_mz > 0 else float("inf"),
        "note": (
            "Bare GUT coupling without threshold corrections. "
            "Full calculation requires S² cohomology charges for "
            "boundary mode representations (η_i values)."
        ),
        "status": (
            "FAILS (2026-10): the ansatz α_GUT = π/(p^{1/3} z) at "
            "M_GUT = M_Pl/p^{1/4} gives α_s ≈ 0.020, sin²θ_W ≈ 0.274, "
            "1/α_em(M_Z) ≈ 261 at p = 104761 (measured 0.1179, 0.23122, "
            "127.95). Legacy ansatz, superseded by bpr.model_scales."
        ),
    }


# =====================================================================
# §22.8  Module-level convenience
# =====================================================================

def summary(p: int = 104761, z: int = 6) -> str:
    """Print a comprehensive summary of the α derivation."""
    bd = alpha_breakdown(p, z)
    result = derive_alpha(p, z)
    sens = sensitivity_to_prime(p, z)

    lines = [
        bd.summary(),
        "",
        f"  At M_Z  (with QED running Δ = {_DELTA_INV_ALPHA_QED:.3f}):",
        f"    1/α(M_Z) predicted  = {result.inv_alpha_MZ:.3f}",
        f"    1/α(M_Z) experiment = {result.inv_alpha_MZ_exp:.3f}",
        f"    Deviation: {result.deviation_MZ_percent:.4f}%",
        "",
        f"  GUT scale:  M_GUT = M_Pl / p^{{1/4}} = {result.M_gut_GeV:.3e} GeV",
        f"  Bare GUT coupling:  α_GUT = {result.alpha_gut_bare:.6f}  "
        f"(1/α_GUT = {1/result.alpha_gut_bare:.1f})",
        "",
        f"  Sensitivity:  d(1/α)/d(ln p) = {sens['d_inv_alpha_d_ln_p']:.3f}",
        f"  Prime for exact α:  p* = {sens['p_for_exact_alpha']:.0f}  "
        f"(framework uses {p})",
        f"  Δp = {sens['delta_p_from_exact']:.0f}  "
        f"({sens['delta_p_percent']:.2f}%)",
    ]
    return "\n".join(lines)
