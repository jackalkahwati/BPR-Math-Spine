"""Two-gap superconductivity from BPR boundary-mode pairing.

Documents (and honestly bounds) the P4.9 Tc(MgB₂) FRAMEWORK item by making
the σ/π two-band structure explicit, and records which interband-coupling
hypotheses the data support versus reject.

Physics
-------
MgB₂ is the canonical two-gap superconductor: a strongly-coupled σ band
(2D boron px–py, large gap Δ_σ ≈ 7 meV) and a weakly-coupled π band
(3D, small gap Δ_π ≈ 2.7 meV). The two bands share a single Tc because the
interband coupling locks them together.

BPR contribution and honest accounting
--------------------------------------
DERIVED from substrate: the combined-band N(0)V via
``superconductor_n0v_derived`` (the same z²-scaled boundary pairing that
nails Nb to 0.06%). With the combined effective Fermi energy
E_F = 2(E_σ + E_π) = 16.4 eV and T_debye = 900 K (both MgB₂ material
inputs), this gives N(0)V ≈ 0.346 and a single-band Tc that the small
interband correction lifts to the observed 39 K.

INTERBAND FACTOR — what the data say:
The interband correction that reproduces Tc = 38.9 K (0.3% off) is the
SMALL factor f = 1 + 1/(4 ln p) ≈ 1.022. Two "cleaner" coordination-shell
hypotheses were tested and REJECTED:

    f = 1 + 2/z   = 1.333  → Tc = 78.5 K  (2× overshoot)   REJECTED
    f = 1 + 1/(2z) = 1.083 → Tc = 46.0 K  (18% overshoot)  REJECTED
    f = 1 + 1/(4 ln p) = 1.022 → Tc = 38.9 K (0.3% off)    WORKS

This is recorded honestly: the coordination-shell factor that works for the
doubly-charmed hyperfine splitting (2/z) does NOT transfer to the MgB₂
interband coupling. The 1/(4 ln p) factor remains phenomenological — it is
the right magnitude but not derived from a specific boundary mechanism.
Hence MgB₂ stays FRAMEWORK (consistent, small residual, interband factor
not first-principles), NOT promoted to DERIVED.

INHERITED (material inputs, not BPR): σ/π band Fermi energies and the Debye
temperature come from MgB₂ band structure / phonon spectrum, as in any
BCS/Eliashberg calculation. BPR does not derive the boron lattice.

References: Choi et al. Nature 418, 758 (2002) two-gap structure;
Brinkman et al. PRB 65, 180517 (2002).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import phase_transitions as _pt

# Experimental anchors (PDG/materials handbook)
_MGB2_TC_EXP_K = 39.0
_MGB2_GAP_SIGMA_MEV = 7.1
_MGB2_GAP_PI_MEV = 2.7


@dataclass
class TwoGapSuperconductor:
    """Two-band (σ/π) superconductor with BPR-derived combined-band pairing.

    Parameters
    ----------
    E_fermi_sigma_eV, E_fermi_pi_eV : float
        Band Fermi energies [eV]. Material inputs from band structure.
    T_debye : float
        Debye temperature [K]. Material/phonon input.
    p : int
        Substrate structural prime.
    z : int
        Coordination number.
    """

    E_fermi_sigma_eV: float = 2.6
    E_fermi_pi_eV: float = 5.6
    T_debye: float = 900.0
    p: int = 104761
    z: int = 6

    @property
    def E_fermi_combined_eV(self) -> float:
        """Combined effective Fermi energy for the two-band pairing."""
        return 2.0 * (self.E_fermi_sigma_eV + self.E_fermi_pi_eV)

    @property
    def n0v_base(self) -> float:
        """BPR-derived combined-band N(0)V (DERIVED, same mechanism as Nb)."""
        return _pt.superconductor_n0v_derived(
            E_fermi_eV=self.E_fermi_combined_eV,
            T_debye=self.T_debye,
            p=self.p,
            z=self.z,
        )

    @property
    def interband_factor(self) -> float:
        """Phenomenological interband correction 1 + 1/(4 ln p).

        This is the factor that reproduces Tc; it is the right magnitude but
        is NOT derived from a specific boundary mechanism. See module
        docstring for the rejected coordination-shell alternatives.
        """
        return 1.0 + 1.0 / (4.0 * np.log(self.p))

    @property
    def n0v_effective(self) -> float:
        return self.n0v_base * self.interband_factor

    @property
    def Tc_K(self) -> float:
        return _pt.superconductor_tc(N0V=self.n0v_effective, T_debye=self.T_debye)

    def report(self) -> dict:
        # Record the rejected coordination-shell hypotheses explicitly.
        f_2z = 1.0 + 2.0 / self.z
        f_half_z = 1.0 + 1.0 / (2.0 * self.z)
        tc_2z = _pt.superconductor_tc(self.n0v_base * f_2z, self.T_debye)
        tc_half_z = _pt.superconductor_tc(self.n0v_base * f_half_z, self.T_debye)
        return {
            "E_fermi_combined_eV": float(self.E_fermi_combined_eV),
            "n0v_base": float(self.n0v_base),
            "interband_factor": float(self.interband_factor),
            "n0v_effective": float(self.n0v_effective),
            "Tc_predicted_K": float(self.Tc_K),
            "Tc_exp_K": _MGB2_TC_EXP_K,
            "Tc_residual_pct": float(100.0 * (self.Tc_K / _MGB2_TC_EXP_K - 1.0)),
            "rejected_hypotheses": {
                "f=1+2/z": {"factor": f_2z, "Tc_K": float(tc_2z)},
                "f=1+1/(2z)": {"factor": f_half_z, "Tc_K": float(tc_half_z)},
            },
            "status": (
                "FRAMEWORK: combined-band N(0)V DERIVED; interband factor "
                "1+1/(4 ln p) is phenomenological (right magnitude, not "
                "first-principles). Coordination-shell factors 2/z and "
                "1/(2z) tested and REJECTED (overshoot). Band energies and "
                "T_debye are MgB2 material inputs."
            ),
        }


def mgb2_two_gap_tc(p: int = 104761, z: int = 6) -> dict:
    """Convenience wrapper: MgB₂ Tc from the two-gap BPR model."""
    return TwoGapSuperconductor(p=p, z=z).report()
