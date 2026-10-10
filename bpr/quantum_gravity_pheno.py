"""
Quantum Gravity Phenomenology (legacy, SUPERSEDED)
==========================================

Legacy module. It writes down quantum-gravity corrections attributed to the
discrete BPR substrate: modified dispersion relations, a generalized
uncertainty principle, and a Lorentz-violation estimate. None of the
p-dependent values below is derived; each is stipulated on the
alpha-calibrated substrate prime p.

SUPERSEDED (2026-10). The architecture decision
(doc/derivations/architecture_decision_2026-09-26.md, section 4) supersedes
"Lorentz invariance to exp(-p^{1/3})", the ξ₂ = 1/p dispersion and the GUP:
BPR-6D is six-dimensional and exactly Lorentz-invariant, and p has no
counterpart there. BPR-6D therefore predicts no Lorentz violation, the same
as the Standard Model plus general relativity, so ξ₁ = 0 discriminates
nothing. The 2026-10 review of unmerged branches withdrew P20.1, P20.4 and
P20.7 as predictions; see doc/CLOSED_AND_DEPRECATED.md. The code is kept as
a legacy arithmetic record; the registry rows in bpr/experimental_data.py
are left unchanged.

Legacy formulas
---------------
* Modified dispersion: E² = p²c² + m²c⁴ + ξ_n E^{n+2}/(M_Pl c²)^n, with
  ξ₁ = 0 and ξ₂ = 1/p (stipulated)
* Generalized uncertainty principle: ΔxΔp ≥ ℏ/2 [1 + β(Δp/M_Pl c)²], with
  β = 1/p (stipulated)
* Minimum measurable length: Δx_min = l_Pl √β
* Speed-of-light variation: |δc/c| = exp(−p^{1/3}) (stipulated). An earlier
  version of this docstring said O(1/p²) here, which the code never used.
* Photon time delay (quadratic case): Δt = (3/2) ξ₂ (E/M_Pl c²)² L/c

Corrections (2026-10): energy_correction() carried a spurious factor 2 and
grb_time_delay() used the prefactor 1/2 instead of 3/2. Both are fixed; see
the method docstrings.

References: Al-Kahwati (2026), BPR-Math-Spine extended theories
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# Physical constants
_C = 299792458.0
_HBAR = 1.054571817e-34     # J·s
_L_PLANCK = 1.616255e-35    # m
_E_PLANCK_GEV = 1.22093e19  # GeV
_T_PLANCK = 5.391247e-44    # s
_M_PL_KG = 2.176434e-8      # kg


# ---------------------------------------------------------------------------
# §20.1  Modified dispersion relation
# ---------------------------------------------------------------------------

@dataclass
class ModifiedDispersion:
    """Modified dispersion relation (legacy, SUPERSEDED 2026-10).

    E² = p²c² + m²c⁴ + ξ_n × E^{n+2} / (M_Pl c²)^n

    Legacy assignments (stipulated, not derived):
    - n = 1 (linear correction): ξ₁ = 0 (attributed to boundary CPT)
    - n = 2 (quadratic correction): ξ₂ ~ 1/p
    - Higher order: ξ_n ~ 1/p^{n-1}

    The legacy argument was that the boundary preserves CPT, and
    CPT + Lorentz → no odd-power corrections.

    SUPERSEDED (2026-10): BPR-6D is exactly Lorentz-invariant
    (architecture decision, section 4), so it predicts ξ₁ = ξ₂ = 0, as the
    Standard Model plus general relativity do. ξ₁ = 0 is therefore not a
    discriminating prediction, and ξ₂ = 1/p rests on the alpha-calibrated p.

    Parameters
    ----------
    p : int – substrate prime (controls correction magnitude)
    """
    p: int = 104761

    @property
    def xi_1(self) -> float:
        """Linear LIV parameter ξ₁ = 0 (forbidden by CPT)."""
        return 0.0

    @property
    def xi_2(self) -> float:
        """Quadratic LIV parameter ξ₂ ~ 1/p (stipulated, legacy)."""
        return 1.0 / self.p

    def energy_correction(self, E_GeV: float, m_GeV: float = 0.0) -> float:
        """Fractional energy correction from modified dispersion.

        δE / E = ξ₂ × (E / M_Pl)² / 2   (massless, at fixed momentum)

        From E² = p²c² + ξ₂ E⁴/(M_Pl c²)², 2E δE = ξ₂ E⁴/(M_Pl c²)² to
        first order in ξ₂, hence the factor 1/2. Corrected 2026-10: the
        earlier code returned ξ₂ (E/M_Pl)², twice this value.

        Parameters
        ----------
        E_GeV : float – photon/particle energy [GeV]
        m_GeV : float – rest mass [GeV]; unused (massless formula)

        Returns float – fractional correction δE/E.
        """
        return 0.5 * self.xi_2 * (E_GeV / _E_PLANCK_GEV) ** 2

    def grb_time_delay(self, E_GeV: float, distance_Mpc: float) -> float:
        """Time-of-flight difference for GRB photons from modified dispersion [s].

        |Δt| = (3/2) × ξ₂ × E² × L / (M_Pl² c⁵)

        For E² = p²c² + ξ₂ E⁴/(M_Pl c²)², the group velocity is
        v/c = 1 + (3/2) ξ₂ (E/M_Pl c²)² to first order in ξ₂, which gives the
        prefactor 3/2 (the standard (1+n)/2 for n = 2). Corrected 2026-10: the
        earlier code used 1/2, a factor 3 too small. For ξ₂ > 0 the
        higher-energy photon is faster and arrives earlier; the magnitude is
        returned. L is a flat-space distance: no cosmological redshift
        integral is applied.

        Parameters
        ----------
        E_GeV : float – photon energy [GeV]
        distance_Mpc : float – distance to source [Mpc]
        """
        L_m = distance_Mpc * 3.0857e22  # Mpc to metres
        E_J = E_GeV * 1.602e-10         # GeV to J
        M_Pl_c2 = _M_PL_KG * _C ** 2
        return 1.5 * self.xi_2 * E_J ** 2 * L_m / (M_Pl_c2 ** 2 * _C)


# ---------------------------------------------------------------------------
# §20.2  Generalized uncertainty principle (GUP)
# ---------------------------------------------------------------------------

@dataclass
class GeneralizedUncertainty:
    """Generalized uncertainty principle (legacy, SUPERSEDED 2026-10).

    Δx Δp ≥ (ℏ/2) [1 + β (Δp / M_Pl c)²]

    Legacy assignment (stipulated, not derived):
        β = 1/p  (substrate correction)

    SUPERSEDED (2026-10): the architecture decision (section 4) supersedes
    the GUP; BPR-6D has no counterpart of p.

    This modifies the Heisenberg uncertainty at Planck scale:
    - Standard QM: Δx_min = 0 (no minimum length)
    - BPR: Δx_min = l_Pl √β = l_Pl / √p

    Parameters
    ----------
    p : int – substrate prime
    """
    p: int = 104761

    @property
    def beta(self) -> float:
        """GUP parameter β = 1/p."""
        return 1.0 / self.p

    @property
    def minimum_length(self) -> float:
        """Minimum measurable length Δx_min = l_Pl √β [m]."""
        return _L_PLANCK * np.sqrt(self.beta)

    @property
    def minimum_length_over_lp(self) -> float:
        """Δx_min / l_Pl = √β = 1/√p."""
        return np.sqrt(self.beta)

    def uncertainty_product(self, delta_p_GeV: float) -> float:
        """Minimum uncertainty product Δx × Δp [J·m].

        Parameters
        ----------
        delta_p_GeV : float – momentum uncertainty [GeV/c]
        """
        delta_p = delta_p_GeV * 1.602e-10 / _C  # GeV/c to kg·m/s
        M_Pl_c = _M_PL_KG * _C
        factor = 1.0 + self.beta * (delta_p / M_Pl_c) ** 2
        return (_HBAR / 2.0) * factor

    @property
    def experimental_bound(self) -> str:
        """Current experimental bound on β.

        Best bounds: β < 10²¹ (from quantum optics).
        BPR prediction: β = 1/p ≈ 10⁻⁵ (well within bounds).
        """
        return f"β = 1/p ≈ {self.beta:.2e}, well below bound β < 1e21"


# ---------------------------------------------------------------------------
# §20.3  Lorentz invariance violation bounds
# ---------------------------------------------------------------------------

@dataclass
class LorentzInvariance:
    """Legacy Lorentz-violation estimate (SUPERSEDED 2026-10).

    Legacy argument: Lorentz invariance holds to leading order because
    the boundary respects the Killing vectors of S².

    The code returns |δc/c| = exp(−p^{1/3}) (stipulated, not derived):
    3.38e-21 at p = 104729 and 3.36e-21 at p = 104761. An earlier version
    of this docstring said |δc/c| ~ 1/p² (≈ 9.1e-11 at p = 104761) and
    "many orders of magnitude below" the bound; neither matched the code.
    exp(−p^{1/3}) sits only ~0.25 decades below the 6e-21 figure used here.

    SUPERSEDED (2026-10): BPR-6D is exactly Lorentz-invariant and p has no
    counterpart there (architecture decision, section 4), so BPR-6D
    predicts δc/c = 0, as the Standard Model plus general relativity do.
    The comparison with 6e-21 is also a category error: that figure is a
    photon time-of-flight bound on energy-dependent dispersion, which an
    energy-independent offset of the photon speed does not produce. P20.7
    is withdrawn as a prediction (doc/CLOSED_AND_DEPRECATED.md).

    Parameters
    ----------
    p : int – substrate prime
    """
    p: int = 104761

    @property
    def fractional_speed_variation(self) -> float:
        """Fractional variation in speed of light (stipulated, legacy).

        |δc/c| ~ exp(-p^{1/3})

        The legacy argument: the violation requires tunneling through a
        barrier of height p^{1/3} in the boundary winding landscape (3
        Killing vectors of S² must be simultaneously disrupted). No such
        barrier is computed anywhere in the repository.
        """
        return np.exp(-self.p ** (1.0 / 3.0))

    @property
    def experimental_bound(self) -> float:
        """Legacy comparison figure: |δc/c| < 6 × 10⁻²¹.

        Attributed here and in bpr/experimental_data.py to Fermi-LAT
        GRB 090510. The 2026-10 review of unmerged branches traces the figure
        to Nemiroff et al. 2012 (6.94e-21), a low-significance bound on
        energy-dependent dispersion. It does not bound an energy-independent
        offset such as exp(-p^{1/3}). Value kept for the legacy record.
        """
        return 6e-21

    @property
    def within_bounds(self) -> bool:
        """True if BPR prediction is within experimental bounds."""
        return self.fractional_speed_variation < self.experimental_bound

    @property
    def orders_below_bound(self) -> float:
        """Orders of magnitude below experimental bound."""
        return np.log10(self.experimental_bound / self.fractional_speed_variation)


# ---------------------------------------------------------------------------
# §20.4  Planck-scale corrections to atomic spectra
# ---------------------------------------------------------------------------

def hydrogen_gravity_shift(n: int = 1) -> float:
    """Gravitational correction to hydrogen energy levels.

    ΔE_grav / E_n = (m_e / M_Pl)² × f(n)

    This is incredibly tiny (~10⁻⁴⁵). It is a dimensional estimate with no
    dependence on p, so it is not specific to BPR (note added 2026-10).

    Parameters
    ----------
    n : int – principal quantum number

    Returns float – fractional energy shift.
    """
    m_e_MeV = 0.511
    E_Pl_MeV = 1.22093e22
    return (m_e_MeV / E_Pl_MeV) ** 2 / n ** 2


def deformed_commutator(p: int = 104761) -> str:
    """Modified commutation relation from BPR substrate.

    [x̂, p̂] = iℏ(1 + β p̂²/M_Pl²c²)

    with β = 1/p.

    Returns str – the deformed commutator formula.
    """
    return f"[x, p] = iℏ(1 + (1/{p}) p²/M_Pl²c²)"
