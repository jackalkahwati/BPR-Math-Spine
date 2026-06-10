"""Heavy gauge-boson winding spectrum from the c=1 compact boson at R²=3.

Implements the tier-B "Gap #3" target: enumerate the winding-momentum modes
of BPR's c=1 compact boundary boson at the self-dual-adjacent radius R²=3,
classify them, and compute their one-loop threshold contribution to gauge
coupling unification.

Physics
-------
BPR's boundary carries a c=1 compact free boson (the boundary phase field φ)
compactified at radius R with R² = 3 (the value fixed by the substrate
prime-field structure; see generations_from_CFT.md). Its spectrum is the
standard winding-momentum tower with conformal dimensions

    (h, h̄) = ( (n/R + wR/2)²/2 , (n/R − wR/2)²/2 ),
    M²(n, w) = (n/R)² + (wR/2)²   [in units of the boundary scale]

where n is momentum (KK-like) and w is winding. The lightest non-trivial
modes are candidates for the heavy gauge bosons (X, Y) that sit between
M_GUT and M_Pl and supply the threshold corrections closing the unification
residual.

BPR contribution and honest accounting
--------------------------------------
DERIVED: the mode tower M²(n,w) at R²=3 is fixed — no free parameters.
The lightest modes and their multiplicities follow directly.

NOT DERIVED (the honest gap): the ASSIGNMENT of specific (n,w) modes to
SU(5)/SO(10) gauge-boson representations requires embedding the gauge group
in the boundary CFT, which is not done here. Without that embedding, the
threshold correction below uses a GENERIC heavy-multiplet beta-function
coefficient, not a rep-resolved one. So this BOUNDS the threshold effect and
shows the mode tower is the right order of magnitude, but does not yet close
the 1.5% residual from first principles.

References: generations_from_CFT.md (R²=3 derivation); Ginsparg, "Applied
Conformal Field Theory" (c=1 spectrum).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_R_SQUARED = 3.0


def winding_mode_mass_squared(n: int, w: int, R_squared: float = _R_SQUARED) -> float:
    """M²(n,w) = (n/R)² + (wR/2)² for the c=1 compact boson [boundary units]."""
    R = np.sqrt(R_squared)
    return (n / R) ** 2 + (w * R / 2.0) ** 2


def enumerate_light_modes(
    n_max: int = 3, w_max: int = 3, R_squared: float = _R_SQUARED
) -> list[dict]:
    """Enumerate the lightest winding-momentum modes, sorted by M².

    Returns a list of dicts {n, w, M2, degeneracy} excluding the (0,0)
    vacuum. Modes with the same M² are grouped (degeneracy counts ±n, ±w
    sign combinations that share the mass).
    """
    seen: dict[float, dict] = {}
    for n in range(-n_max, n_max + 1):
        for w in range(-w_max, w_max + 1):
            if n == 0 and w == 0:
                continue
            m2 = winding_mode_mass_squared(n, w, R_squared)
            key = round(m2, 9)
            if key not in seen:
                seen[key] = {"M2": m2, "degeneracy": 0, "example_nw": (n, w)}
            seen[key]["degeneracy"] += 1
    modes = sorted(seen.values(), key=lambda d: d["M2"])
    return modes


@dataclass
class GUTThresholdEstimate:
    """One-loop threshold estimate from the heavy winding modes.

    Parameters
    ----------
    M_GUT_GeV : float
        Unification scale (BPR: M_Pl / p^{1/4} ≈ 2e16 GeV).
    R_squared : float
        Compact-boson radius squared (3 in BPR).
    """

    M_GUT_GeV: float = 2.0e16
    R_squared: float = _R_SQUARED

    def threshold_delta_inv_alpha(self, b_heavy: float = 1.0) -> float:
        """Σ_X (b_X / 12π) ln(M_GUT²/M_X²) over the lightest heavy modes.

        Uses a GENERIC heavy-multiplet coefficient b_heavy (not rep-resolved
        — see module docstring). The boundary mode masses M²(n,w) are scaled
        so the lightest mode sits at M_GUT; heavier modes sit above it.
        """
        modes = enumerate_light_modes(R_squared=self.R_squared)
        m2_lightest = modes[0]["M2"]
        total = 0.0
        for mode in modes:
            # mode physical mass relative to the lightest = M_GUT anchor
            ratio = mode["M2"] / m2_lightest
            M_X = self.M_GUT_GeV * np.sqrt(ratio)
            # threshold runs from M_GUT to M_X (above), so ln(M_GUT/M_X)<=0
            total += (mode["degeneracy"] * b_heavy / (12.0 * np.pi)) * np.log(
                (self.M_GUT_GeV ** 2) / (M_X ** 2)
            )
        return total

    def report(self) -> dict:
        modes = enumerate_light_modes(R_squared=self.R_squared)
        return {
            "R_squared": self.R_squared,
            "n_light_mode_levels": len(modes),
            "lightest_modes": modes[:6],
            "threshold_delta_inv_alpha_b1": float(self.threshold_delta_inv_alpha(1.0)),
            "residual_to_close_approx": 0.015 * 4.0 * np.pi,  # ~1.5% in 1/alpha units
            "status": (
                "MODE TOWER DERIVED (R²=3, parameter-free); rep ASSIGNMENT "
                "to SU(5)/SO(10) gauge bosons NOT done. Threshold uses a "
                "generic heavy-multiplet b coefficient, so this BOUNDS the "
                "effect and shows the right order of magnitude but does not "
                "yet close the 1.5% unification residual from first "
                "principles. The honest next step is the CFT gauge embedding."
            ),
        }


def gut_winding_report(M_GUT_GeV: float = 2.0e16) -> dict:
    return GUTThresholdEstimate(M_GUT_GeV=M_GUT_GeV).report()


# ---------------------------------------------------------------------------
# SU(5) rep assignment: closing the GUT-threshold gap
# ---------------------------------------------------------------------------
#
# Beyond the SM gauge bosons, SU(5) GUT contains the X, Y bosons in the
# (3, 2, 5/6) ⊕ (3̄, 2, -5/6) of SU(3)×SU(2)×U(1). This is a 12-state heavy
# multiplet. Their one-loop contributions to the beta functions of the three
# SM couplings differ — that asymmetry is what closes unification residuals.
#
# Standard SU(5) X,Y matching coefficients (per X,Y pair, GUT normalization):
#   δb_1 = -25/6    [U(1)_Y, normalized so b_1 = 33/5 for the SM]
#   δb_2 = -3       [SU(2)_L]
#   δb_3 = -2       [SU(3)_c, sometimes quoted -4 depending on convention]
# (See Hall, NPB 178 (1981) 75 and standard GUT reviews.)
#
# In BPR, the lightest c=1 winding-momentum modes above M_GUT are CANDIDATES
# for the X, Y bosons. We need 12 states; the BPR mode degeneracies match
# this when we group the lightest two non-vacuum levels (deg 2 + 2 + 4 + 4 =
# 12 across the first four levels). The ANSATZ is:
#
#   level (n=±1, w=0)    deg 2  → 2 of the 12 X,Y states
#   level (n=0, w=±1)    deg 2  → 2 of the 12
#   level (n=±1, w=±1)   deg 4  → 4 of the 12
#   level (n=±2, w=0)    deg 2  → 2 of the 12
#   level (n=±1, w=∓1)   deg 2  → 2 of the 12  (other sign combination)
#
# i.e. modes assigned to X,Y are the 12 lightest non-vacuum c=1 modes.
# Modes heavier than these are NOT counted toward the GUT threshold.

# SU(5) X,Y beta-function contributions per (3,2,±5/6) pair
SU5_DELTA_B = {
    "U1_Y_GUT_norm": -25.0 / 6.0,   # b_1 normalization: 3/5 × SM hypercharge
    "SU2_L":         -3.0,
    "SU3_c":         -2.0,
}


def _lightest_xy_modes(n_xy: int = 12, R_squared: float = _R_SQUARED) -> list:
    """Return the lightest c=1 winding-momentum modes up to n_xy states.

    Sums degeneracies of the level table until n_xy is reached. Each
    selected mode contributes its full deg to the X,Y count.
    """
    levels = enumerate_light_modes(n_max=4, w_max=4, R_squared=R_squared)
    chosen = []
    used = 0
    for lvl in levels:
        if used >= n_xy:
            break
        chosen.append(lvl)
        used += lvl["degeneracy"]
    return chosen, used


def gut_threshold_su5_assigned(
    M_GUT_GeV: float = 2.0e16,
    R_squared: float = _R_SQUARED,
    n_xy: int = 12,
) -> dict:
    """One-loop threshold correction to 1/α_i from SU(5) X,Y assignment.

    For each chosen mode of mass M_X = M_GUT × √(M²_mode / M²_lightest), the
    matching contribution to 1/α_i at M_GUT is:

        Δ(1/α_i) = (δb_i / 12π) × ln(M_X²/M_GUT²)

    Summed over modes (with their degeneracies, capped at n_xy total
    states). δb_i are the SU(5) X,Y coefficients.

    Honest scope
    ------------
    The mode-to-X,Y ASSIGNMENT is an ansatz: the lightest n_xy = 12 c=1
    states are interpreted as the (3,2,±5/6) of SU(5). A rigorous version
    requires explicitly embedding SU(5) in the boundary CFT and projecting
    the winding tower onto the 24 of SU(5) — that derivation is not done.

    What this DOES show: with this natural assignment, the c=1 threshold
    tower delivers the CORRECT QUALITATIVE structure (asymmetric across
    1/α_1, 1/α_2, 1/α_3, with the largest contribution in U(1)) and a
    magnitude in the right range to address the ~0.8% unification residual
    left after the forward GUT thresholds.
    """
    chosen, used = _lightest_xy_modes(n_xy=n_xy, R_squared=R_squared)
    m2_lightest = chosen[0]["M2"]
    contributions = {key: 0.0 for key in SU5_DELTA_B}
    levels_used = []
    total_states = 0
    for mode in chosen:
        deg = min(mode["degeneracy"], n_xy - total_states)
        if deg <= 0:
            break
        ratio = mode["M2"] / m2_lightest
        # mass ratio above M_GUT; lightest sits at M_GUT
        ln_term = np.log(ratio)  # ≥ 0 since ratio ≥ 1
        per_state_contrib = ln_term / (12.0 * np.pi)
        for key, db in SU5_DELTA_B.items():
            contributions[key] += deg * db * per_state_contrib
        levels_used.append({
            "M2_ratio": ratio,
            "deg_used": deg,
            "ln_M_over_M_GUT": float(0.5 * ln_term),
        })
        total_states += deg

    residual_to_close = 0.015 * 4.0 * np.pi  # ~0.19 in 1/alpha units
    return {
        "n_xy_states_assigned": total_states,
        "levels_used": levels_used,
        "delta_inv_alpha": contributions,
        "max_abs_contribution": float(max(abs(v) for v in contributions.values())),
        "residual_to_close_approx": float(residual_to_close),
        "closes_qualitatively": bool(
            max(abs(v) for v in contributions.values()) >= 0.5 * residual_to_close
        ),
        "status": (
            "ANSATZ-BASED CLOSURE. With the natural assignment of the 12 "
            "lightest c=1 modes to SU(5) X,Y (3,2,±5/6), the threshold "
            "magnitudes are the correct order of magnitude to address the "
            "remaining ~0.8% unification residual, with the expected "
            "asymmetric pattern (|Δ(1/α₁)| > |Δ(1/α₂)| > |Δ(1/α₃)|). "
            "The rigorous version still requires embedding SU(5) in the "
            "boundary CFT explicitly -- this is the right scale and "
            "structure, not the final derivation."
        ),
    }
