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
