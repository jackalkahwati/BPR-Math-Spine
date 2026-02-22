"""
BPR Predictions vs JWST Anomalies
===================================

Computes BPR predictions for three JWST-era anomalies and compares them
honestly against published observations and ΛCDM predictions.

Anomalies addressed
-------------------
1. Too-early massive galaxies  — JWST UV luminosity function at z=9–16
2. Hubble tension              — H₀ = 73 (local) vs 67.4 (CMB)
3. S8 tension                  — σ₈√(Ωm/0.3) lower than Planck+ΛCDM

Status: BPR's corrections are quantified, direction assessed, and the
coupling gap (how far the current theory falls short) is derived.
No parameter tuning is performed.

References
----------
Finkelstein et al. 2023  — JWST CEERS UV LF z=9–12,  ApJL 946, L13
Harikane et al. 2022     — UV LF z=9–16, ApJS 259, 20
Labbe et al. 2023        — massive candidates z=7.4–9.1, Nature 616, 266
Riess et al. 2022        — H0 = 73.04 ± 1.04, ApJL 934, L7
Planck 2020 (Aghanim+)  — H0 = 67.4 ± 0.5, A&A 641 A6
Amon & Efstathiou 2022  — S8 = 0.766 ± 0.020, MNRAS 516, 5355
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


# ── Physical / cosmological constants ──────────────────────────────────────
_H0_PLANCK   = 67.4          # km/s/Mpc (CMB)
_H0_LOCAL    = 73.04         # km/s/Mpc (Cepheid+SN, Riess 2022)
_H0_ERR_LOCAL = 1.04         # 1σ
_OMEGA_M     = 0.315         # Planck matter fraction
_SIGMA8_PLANCK = 0.811       # Planck + ΛCDM
_S8_PLANCK   = _SIGMA8_PLANCK * math.sqrt(_OMEGA_M / 0.3)  # ≈ 0.832
_S8_WL_OBS   = 0.766         # weak-lensing (Amon & Efstathiou 2022)
_S8_WL_ERR   = 0.020
_N_EFF_STD   = 3.044         # standard Neff
_K_PIVOT     = 0.05          # Mpc⁻¹ (CMB pivot scale)

# ── BPR substrate ──────────────────────────────────────────────────────────
_P    = 104729
_N_EF = _P ** (1.0 / 3.0) * (1.0 + 1.0 / 3.0)   # ≈ 62.85

# ── Pre-JWST ΛCDM Schechter parameters ─────────────────────────────────────
# From Bouwens+2021 (Table 5, HST compilation) plus high-z extrapolation.
# These represent the ΛCDM UV LF *before* JWST — the baseline against which
# the JWST excess is measured.
# Format: z → (M_star, phi_star [Mpc⁻³], alpha)
_SCHECHTER_PARAMS: dict = {
    6:  (-20.94, 0.72e-3, -1.87),   # Bouwens+2021
    7:  (-20.69, 0.47e-3, -2.11),   # Bouwens+2021
    8:  (-20.54, 0.18e-3, -2.13),   # Bouwens+2021
    9:  (-20.04, 0.18e-3, -2.17),   # Bouwens+2021
    10: (-20.64, 0.13e-3, -2.27),   # Oesch+2018
    12: (-20.30, 0.036e-3, -2.30),  # semi-analytic extrapolation
    16: (-20.00, 0.001e-3, -2.30),  # extrapolation (ΛCDM expects ~0 galaxies)
}


# ═══════════════════════════════════════════════════════════════════════════
#  JWST OBSERVED DATA POINTS  (published measurements)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class UVLFPoint:
    """One bin of the UV luminosity function."""
    z: float          # redshift
    M_UV: float       # absolute UV magnitude (AB)
    log_phi: float    # log₁₀(φ / Mpc⁻³ mag⁻¹)
    log_phi_err: float
    source: str


# Published JWST UV LF data at the BRIGHT end (z ≥ 9).
# These are the points where JWST observes more galaxies than the pre-JWST
# ΛCDM Schechter extrapolation — the actual "too-early galaxies" anomaly.
# Sources: Finkelstein+2023 ApJL 946 L13; Harikane+2022 ApJS 259 20.
JWST_UV_LF: List[UVLFPoint] = [
    # z ~ 9 — bright end, JWST exceeds pre-JWST predictions by ~1 dex
    UVLFPoint(9.0,  -22.0, -5.40, 0.30, "Harikane+2022"),
    UVLFPoint(9.0,  -21.5, -4.60, 0.25, "Finkelstein+2023"),
    # z ~ 10 — excess grows
    UVLFPoint(10.0, -22.0, -5.70, 0.35, "Harikane+2022"),
    UVLFPoint(10.0, -21.5, -5.10, 0.30, "Harikane+2022"),
    # z ~ 12 — excess grows at the bright end (~1–2 dex above ΛCDM)
    UVLFPoint(12.0, -22.0, -5.80, 0.50, "Harikane+2022"),
    UVLFPoint(12.0, -21.5, -5.40, 0.40, "Harikane+2022"),
    # z ~ 16 — most extreme; ΛCDM extrapolation predicts nearly zero
    UVLFPoint(16.0, -21.0, -6.20, 0.60, "Harikane+2022"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  ΛCDM BASELINE PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════

class LambdaCDM:
    """Analytic ΛCDM estimates for JWST-relevant quantities.

    UV luminosity function: Schechter function with Bouwens+2021 parameters
    (pre-JWST HST baseline).  This is the correct ΛCDM baseline to compare
    against JWST observations.  PS methods are retained for BPR corrections.
    """

    def __init__(
        self,
        H0: float = _H0_PLANCK,
        omega_m: float = _OMEGA_M,
        n_s: float = 0.9649,
        sigma8: float = _SIGMA8_PLANCK,
    ):
        self.H0 = H0
        self.omega_m = omega_m
        self.n_s = n_s
        self.sigma8 = sigma8

    # ── matter power spectrum ──────────────────────────────────────────────

    def sigma_M(self, M_halo_Msun: float) -> float:
        """RMS matter fluctuation in sphere enclosing M_halo at z=0.

        Uses the CDM transfer-function–calibrated power law:
            σ(M) = σ₈ × (M₈/M)^α
        where α = 0.2 for M ~ 10^{10}–10^{12} M_sun (galaxy scales) and
        M₈ = (4π/3) ρ_{m,0} (8 h⁻¹ Mpc)³.

        The exponent α = 0.2 comes from the effective CDM spectral slope
        n_eff ≈ -2.2 at these scales: α = (n_eff+3)/6 = 0.8/6 ≈ 0.13,
        with the additional Window-function correction lifting it to ~0.2.
        Calibrated against Eisenstein-Hu (1998) results (accurate to ~15%).
        """
        h = self.H0 / 100.0
        rho_m0 = 2.775e11 * self.omega_m * h ** 2  # Msun Mpc⁻³
        R8     = 8.0 / h                            # 8 h⁻¹ Mpc
        M8     = (4.0 * math.pi / 3.0) * rho_m0 * R8 ** 3
        # Effective CDM slope exponent for galaxy-formation scales
        alpha  = 0.2
        return self.sigma8 * (M8 / max(M_halo_Msun, 1e6)) ** alpha

    def sigma_M_at_z(self, M_halo_Msun: float, z: float) -> float:
        """σ(M) at redshift z, using linear growth factor D(z)."""
        D_ratio = self._growth_factor(z)   # D(z)/D(0)
        return self.sigma_M(M_halo_Msun) * D_ratio

    def _growth_factor(self, z: float) -> float:
        """Linear growth factor D(z)/D(0), Carroll-Press-Turner (1992).

        The full CPT formula is D(z) = a × g(Ω_m(z), Ω_Λ(z)) where
        g(Ω_m, Ω_Λ) ≈ (5/2) Ω_m / (Ω_m^{4/7} - Ω_Λ + (1+Ω_m/2)(1+Ω_Λ/70)).
        The factor of a = 1/(1+z) is essential; without it the formula returns
        the instantaneous growth rate, not D(z)/D(0).
        """
        a = 1.0 / (1.0 + z)
        om = self.omega_m
        ol = 1.0 - om   # flat
        om_z = om / (om + ol * a ** 3)
        ol_z = ol * a ** 3 / (om + ol * a ** 3)
        g_z = (5.0 / 2.0) * om_z / (
            om_z ** (4.0 / 7.0) - ol_z + (1.0 + om_z / 2.0) * (1.0 + ol_z / 70.0)
        )
        # D(z=0): a=1, om_z=om, ol_z=ol
        g_0 = (5.0 / 2.0) * om / (
            om ** (4.0 / 7.0) - ol + (1.0 + om / 2.0) * (1.0 + ol / 70.0)
        )
        return a * g_z / g_0  # normalized so D(z=0) = 1

    # ── Press-Schechter halo mass function ────────────────────────────────

    def dn_dlnM(self, M_halo_Msun: float, z: float) -> float:
        """Press-Schechter dn/d(lnM) [Mpc⁻³].

        Returns comoving number density per unit ln(M).
        """
        rho_m0 = 2.775e11 * self.omega_m * (self.H0 / 100.0) ** 2
        sigma = self.sigma_M_at_z(M_halo_Msun, z)
        delta_c = 1.686   # collapse threshold

        nu = delta_c / sigma
        # PS: dn/d(lnM) = (rho_m0/M) * |d ln sigma / d ln M| * sqrt(2/pi) * nu * exp(-nu^2/2)
        # sigma ∝ M^{-alpha} with alpha=0.2, so |d ln sigma / d ln M| = alpha
        dlnsigma_dlnM = 0.2  # CDM-calibrated; matches sigma_M() above
        n_ps = (rho_m0 / M_halo_Msun) * abs(dlnsigma_dlnM) * math.sqrt(
            2.0 / math.pi
        ) * nu * math.exp(-nu ** 2 / 2.0)
        return max(n_ps, 1e-30)

    # ── UV luminosity function (Schechter, pre-JWST baseline) ─────────────

    def _schechter_params(self, z: float) -> Tuple[float, float, float]:
        """Interpolate Bouwens+2021 Schechter parameters at redshift z."""
        z_nodes = sorted(_SCHECHTER_PARAMS.keys())
        # Clamp to table range
        z_eff = max(float(z_nodes[0]), min(float(z_nodes[-1]), z))
        for i in range(len(z_nodes) - 1):
            z0, z1 = float(z_nodes[i]), float(z_nodes[i + 1])
            if z0 <= z_eff <= z1:
                t = (z_eff - z0) / (z1 - z0)
                M0, p0, a0 = _SCHECHTER_PARAMS[z_nodes[i]]
                M1, p1, a1 = _SCHECHTER_PARAMS[z_nodes[i + 1]]
                # phi_star interpolated in log space
                log_p = (1.0 - t) * math.log10(p0) + t * math.log10(p1)
                return (1.0 - t) * M0 + t * M1, 10.0 ** log_p, (1.0 - t) * a0 + t * a1
        return _SCHECHTER_PARAMS[z_nodes[-1]]

    def uv_luminosity_function(self, M_UV: float, z: float) -> float:
        """log₁₀(φ) [Mpc⁻³ mag⁻¹] via Schechter function (Bouwens+2021).

        This is the pre-JWST ΛCDM expectation, calibrated to HST surveys.
        φ(M) = (ln10/2.5) φ* × 10^{-0.4(α+1)(M-M*)} × exp(-10^{-0.4(M-M*)})
        """
        M_star, phi_star, alpha = self._schechter_params(z)
        x = 10.0 ** (-0.4 * (M_UV - M_star))
        phi = (math.log(10.0) / 2.5) * phi_star * x ** (alpha + 1.0) * math.exp(-x)
        return math.log10(max(phi, 1e-30))


# ═══════════════════════════════════════════════════════════════════════════
#  BPR PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════

class BPRCosmology:
    """BPR-modified cosmological predictions.

    All modifications derived from substrate parameters (p, N_efolds)
    without any free parameters tuned to JWST data.

    Current BPR modifications:
    ─────────────────────────
    1. n_s = 1 - 2/N_efolds = 0.9682  (vs Planck 0.9649)
    2. ΔN_eff = (4/11)^(4/3) / p^(1/6) ≈ 0.038
    3. Growth rate modified by boundary dissipation Γ_b = H / p^(1/3)
    4. Boundary mode coupling to matter: G_eff(k) = G[1 + (k_H/k)^2/p]

    These are GENUINE BPR predictions. The gap between prediction and
    JWST data is quantified honestly; no post-hoc tuning is applied.
    """

    def __init__(self, p: int = _P):
        self.p  = p
        self.N  = _P ** (1.0 / 3.0) * (1.0 + 1.0 / 3.0)
        # BPR-predicted spectral parameters
        self.n_s    = 1.0 - 2.0 / self.N
        self.delta_Neff = (4.0 / 11.0) ** (4.0 / 3.0) / p ** (1.0 / 6.0)
        # BPR-modified H0
        self._lcdm  = LambdaCDM(n_s=self.n_s)

    # ── spectral index enhancement ─────────────────────────────────────────

    def sigma_ratio(self, k_Mpc: float) -> float:
        """σ_BPR / σ_ΛCDM at wavenumber k [Mpc⁻¹] from Δn_s alone."""
        delta_ns = self.n_s - 0.9649
        return (k_Mpc / _K_PIVOT) ** (delta_ns / 2.0)

    # ── boundary dissipation correction to growth ─────────────────────────

    def boundary_growth_suppression(self, z_form: float) -> float:
        """Factor by which boundary dissipation suppresses growth.

        Γ_b = H / p^(1/3) is the boundary relaxation rate.
        Integrated from matter-radiation equality (z_eq=3400) to z_form:

            f = exp(- ∫ Γ_b/H dt) = exp(- Δln(a) / p^{1/3})

        where Δln(a) = ln((1+z_eq)/(1+z_form)).
        """
        z_eq = 3400.0
        if z_form >= z_eq:
            return 1.0
        delta_lna = math.log((1.0 + z_eq) / (1.0 + z_form))
        return math.exp(-delta_lna / self.p ** (1.0 / 3.0))

    # ── scale-dependent effective gravity ─────────────────────────────────

    def G_eff_ratio(self, k_Mpc: float) -> float:
        """G_eff(k) / G from boundary substrate coupling.

        G_eff / G = 1 + (k_H / k)^2 / p
        k_H = Hubble wavenumber ≈ H0/c ≈ 2.2×10⁻⁴ Mpc⁻¹
        """
        k_H = _H0_PLANCK / (3e5)   # H0/c in Mpc^-1 ≈ 2.24e-4
        return 1.0 + (k_H / max(k_Mpc, 1e-10)) ** 2 / self.p

    # ── combined sigma8 prediction ─────────────────────────────────────────

    @property
    def sigma8_bpr(self) -> float:
        """BPR-predicted σ₈.

        Modified by:
        1. Δn_s shift (small enhancement at k~0.2 Mpc⁻¹)
        2. Boundary growth suppression at late times
        3. Scale-dependent G_eff at k ~ 0.2 Mpc⁻¹
        """
        k_sigma8 = 0.2     # Mpc⁻¹ (scale entering σ₈ integral)
        sigma_ratio = self.sigma_ratio(k_sigma8)
        f_growth = self.boundary_growth_suppression(z_form=0.0)
        G_ratio  = self.G_eff_ratio(k_sigma8)
        return _SIGMA8_PLANCK * sigma_ratio * f_growth * math.sqrt(G_ratio)

    @property
    def S8_bpr(self) -> float:
        """BPR-predicted S₈ = σ₈ √(Ωm / 0.3)."""
        return self.sigma8_bpr * math.sqrt(_OMEGA_M / 0.3)

    # ── Hubble constant prediction ─────────────────────────────────────────

    @property
    def H0_bpr(self) -> float:
        """BPR-predicted H₀ [km/s/Mpc].

        ΔNeff shifts the sound horizon at recombination:
        H0_BPR ≈ H0_Planck × (1 + ΔNeff / (2 N_eff_std))
        """
        return _H0_PLANCK * (1.0 + self.delta_Neff / (2.0 * _N_EFF_STD))

    # ── UV luminosity function (standard BPR) ─────────────────────────────

    def uv_luminosity_function(self, M_UV: float, z: float) -> float:
        """log₁₀(φ_BPR) [Mpc⁻³ mag⁻¹] at given M_UV and z.

        BPR modifies the ΛCDM baseline through:
        1. Enhanced small-scale power from higher n_s
        2. Scale-dependent G_eff at galaxy-scale k
        3. Boundary growth suppression (from z_eq to z_form)
        """
        log_phi_lcdm = self._lcdm.uv_luminosity_function(M_UV, z)

        M_halo = 1e11 * 10.0 ** (-0.4 * (M_UV + 21.0))
        rho_m0 = 2.775e11 * _OMEGA_M * (_H0_PLANCK / 100.0) ** 2
        R = (3.0 * M_halo / (4.0 * math.pi * rho_m0)) ** (1.0 / 3.0)
        k_halo = 2.0 * math.pi / max(R, 0.01)

        sigma_r  = self.sigma_ratio(k_halo)
        G_r      = self.G_eff_ratio(k_halo)
        f_growth = self.boundary_growth_suppression(z_form=z)

        net_sigma_ratio = sigma_r * math.sqrt(G_r) * f_growth
        delta_log_phi   = 3.0 * math.log10(net_sigma_ratio)
        return log_phi_lcdm + delta_log_phi

    # ── UV luminosity function (BPR + Vacuum Impedance Mismatch MOND) ─────────────────────

    @property
    def mond_a0(self) -> float:
        """MOND acceleration scale from Vacuum Impedance Mismatch.

        a₀ = (c H₀ / 2π) × (1 + z_coord / (4 ln p))

        z_coord = 6 is the lattice coordination number; p = 104729.
        Result ≈ 1.18×10⁻¹⁰ m/s² (observed: 1.2×10⁻¹⁰ m/s², 1.5% off).
        """
        H0_si = _H0_PLANCK * 1000.0 / 3.086e22   # s⁻¹
        z_coord = 6.0                              # lattice coordination number
        return (3e8 * H0_si / (2.0 * math.pi)) * (1.0 + z_coord / (4.0 * math.log(self.p)))

    def uv_luminosity_function_mond(self, M_UV: float, z: float) -> float:
        """log₁₀(φ) from BPR + Vacuum Impedance Mismatch MOND collapse modification.

        Vacuum Impedance Mismatch predicts MOND gravity with
        a₀ ≈ 1.18×10⁻¹⁰ m/s² from the impedance formula.

        At galaxy-formation scales:
            a_char = G M / R² ≈ 10⁻¹⁴ m/s²  ≪  a₀ = 1.18×10⁻¹⁰ m/s²

        So ALL galaxy-mass halos are in the deep MOND regime.  Nusser (2002)
        shows that spherical collapse in deep MOND gives δ_c ≈ 1.33 instead
        of the Newtonian 1.686 — a 21% reduction that exponentially boosts
        halo abundance at the rare bright end.

        The boost is:
            log₁₀(φ_MOND / φ_Newton) = log₁₀(ν_M/ν_N) + (ν_N²-ν_M²)/(2 ln10)

        where ν = δ_c / σ(M, z).

        CRITICAL CAVEAT: because a₀ is larger than any cosmological
        acceleration (even cluster scales), MOND applies universally.
        This also boosts σ₈-scale clustering, worsening the S8 tension.
        The MOND fix is therefore NOT consistent with CMB+LSS constraints
        — it is shown here to quantify the gap and point toward the needed
        new physics.
        """
        log_phi_lcdm = self._lcdm.uv_luminosity_function(M_UV, z)

        M_halo  = 1e11 * 10.0 ** (-0.4 * (M_UV + 21.0))
        sigma_z = self._lcdm.sigma_M_at_z(M_halo, z)
        sigma_z = max(sigma_z, 1e-6)

        delta_c_newton = 1.686
        delta_c_mond   = 1.33   # Nusser (2002), deep MOND spherical collapse

        nu_n = delta_c_newton / sigma_z
        nu_m = delta_c_mond   / sigma_z

        # PS ratio in log₁₀:  (ν_M/ν_N) × exp((ν_N²−ν_M²)/2)
        log_ratio = (math.log10(nu_m / nu_n)
                     + (nu_n ** 2 - nu_m ** 2) / (2.0 * math.log(10.0)))
        # Cap at ±4 dex to prevent unphysical blowup at very high z
        log_ratio = max(-4.0, min(4.0, log_ratio))

        # Also include Δn_s power enhancement (same as standard BPR)
        rho_m0 = 2.775e11 * _OMEGA_M * (_H0_PLANCK / 100.0) ** 2
        R      = (3.0 * M_halo / (4.0 * math.pi * rho_m0)) ** (1.0 / 3.0)
        k_halo = 2.0 * math.pi / max(R, 0.01)
        delta_ns_correction = 3.0 * math.log10(self.sigma_ratio(k_halo))

        return log_phi_lcdm + log_ratio + delta_ns_correction


# ═══════════════════════════════════════════════════════════════════════════
#  BPR V2 — THEORY IV PHASE-TRANSITION MECHANISM
# ═══════════════════════════════════════════════════════════════════════════

class BPRCosmologyV2(BPRCosmology):
    """BPR + Universal Phase Transition Taxonomy: epoch-dependent MOND / Newtonian switch.

    Derives a critical phase-transition redshift z_PT where the boundary
    substrate undergoes a Class C (Impedance) transition, gating MOND collapse
    enhancement to early cosmic epochs only.

    DERIVATION OF z_PT
    ------------------
    Vacuum Impedance Mismatch gives MOND frequency ω_MOND = a₀/c.
    Universal Phase Transition Taxonomy Class C: the substrate switches state when
    the boundary dissipation rate Γ_b = H(z)/p^{1/3} equals ω_MOND.

        Γ_b(z_PT) = ω_MOND
        H(z_PT) / p^{1/3} = a₀ / c
        H(z_PT) = p^{1/3} × a₀ / c

    In a flat ΛCDM background, E(z) = H(z)/H₀ = √(Ωm(1+z)³ + ΩΛ):

        z_PT ≈ 5.1  (derived from p and a₀, zero free parameters)

    PHYSICAL BEHAVIOUR
    ------------------
    z > z_PT  (Γ_b > ω_MOND — boundary modes oscillate above MOND scale):
        • Substrate in Class C "frustrated" high-impedance state
        • MOND gravity active: collapse threshold δ_c = 1.33 (Nusser 2002)
        • Boundary dissipation suspended (modes can't damp below a₀/c)
        → Galaxy formation ENHANCED vs ΛCDM  (helps JWST z=9–16)

    z < z_PT  (Γ_b < ω_MOND — boundary modes relax below MOND scale):
        • Class C transition: substrate orders into low-impedance state
        • Gravity reverts to Newtonian: δ_c = 1.686
        • Boundary dissipation resumes, integrated from z_PT (not from z_eq)
        → σ₈ less suppressed than standard BPR  (improves S8 tension)

    QUANTITATIVE IMPROVEMENT
    ------------------------
    Standard BPR:  σ₈ = 0.684 (overshoots WL = 0.748), S8 tension worsened
    This (V2):     σ₈ ≈ 0.781, S8 ≈ 0.800 — closes ~45% of S8 tension
    MOND (no PT):  S8 unchanged, JWST boosted but S8 tension worsened
    """

    @property
    def z_pt(self) -> float:
        """Critical phase-transition redshift from Γ_b(z_PT) = ω_MOND.

        Condition: H(z_PT) = p^{1/3} × a₀ / c

        Solved in ΛCDM:  (1+z_PT)³ = (E_target² − Ω_Λ) / Ω_m
        where E_target = p^{1/3} × a₀ / (c H₀).
        """
        H0_si      = _H0_PLANCK * 1000.0 / 3.086e22   # s⁻¹
        omega_MOND = self.mond_a0 / 3e8                 # a₀/c  [s⁻¹]
        H_target   = self.p ** (1.0 / 3.0) * omega_MOND
        E_target   = H_target / H0_si
        om = _OMEGA_M
        ol = 1.0 - om
        one_plus_z_cubed = max((E_target ** 2 - ol) / om, 1.0)
        return one_plus_z_cubed ** (1.0 / 3.0) - 1.0

    def boundary_growth_suppression_v2(self, z_form: float) -> float:
        """Boundary dissipation integrated from z_PT to z_form (not from z_eq).

        At z > z_PT the boundary is in the MOND-active (high-impedance) state
        and dissipation is suspended.  After the Class C transition at z_PT,
        dissipation resumes from z_PT downward.

            f = exp(−Δln(a) / p^{1/3}),  Δln(a) = ln((1+z_PT)/(1+z_form))
        """
        z_pt = self.z_pt
        if z_form >= z_pt:
            return 1.0   # dissipation suspended during MOND epoch
        delta_lna = math.log((1.0 + z_pt) / (1.0 + z_form))
        return math.exp(-delta_lna / self.p ** (1.0 / 3.0))

    @property
    def sigma8_v2(self) -> float:
        """σ₈ from Universal Phase Transition Taxonomy (V2).

        At z=0 << z_PT: gravity is Newtonian (no MOND boost), dissipation
        integrated only from z_PT ≈ 5.1 to z=0 (shorter lever arm than
        standard BPR which integrates from z_eq=3400).

        Result is less suppressed than standard BPR → improves S8 tension.
        """
        k_sigma8 = 0.2
        sigma_r  = self.sigma_ratio(k_sigma8)
        f_growth = self.boundary_growth_suppression_v2(z_form=0.0)
        G_ratio  = self.G_eff_ratio(k_sigma8)
        return _SIGMA8_PLANCK * sigma_r * f_growth * math.sqrt(G_ratio)

    @property
    def S8_v2(self) -> float:
        """BPR V2  S₈ = σ₈_V2 √(Ωm / 0.3)."""
        return self.sigma8_v2 * math.sqrt(_OMEGA_M / 0.3)

    def uv_luminosity_function_v2(self, M_UV: float, z: float) -> float:
        """log₁₀(φ) with Universal Phase Transition Taxonomy gated MOND.

        z > z_PT : MOND δ_c=1.33, boundary dissipation suspended (f=1)
        z ≤ z_PT : Newtonian δ_c=1.686, dissipation from z_PT to z
        """
        log_phi_lcdm = self._lcdm.uv_luminosity_function(M_UV, z)

        M_halo  = 1e11 * 10.0 ** (-0.4 * (M_UV + 21.0))
        sigma_z = self._lcdm.sigma_M_at_z(M_halo, z)
        sigma_z = max(sigma_z, 1e-6)

        z_pt = self.z_pt
        delta_c_newton = 1.686
        delta_c_eff    = 1.33 if z > z_pt else delta_c_newton

        nu_n = delta_c_newton / sigma_z
        nu_e = delta_c_eff   / sigma_z

        # PS ratio in log₁₀; zero if still Newtonian (z ≤ z_PT)
        if delta_c_eff < delta_c_newton:
            log_ratio = (math.log10(nu_e / nu_n)
                         + (nu_n ** 2 - nu_e ** 2) / (2.0 * math.log(10.0)))
            log_ratio = max(-4.0, min(4.0, log_ratio))
        else:
            log_ratio = 0.0

        # Δn_s spectral correction (always active)
        rho_m0 = 2.775e11 * _OMEGA_M * (_H0_PLANCK / 100.0) ** 2
        R      = (3.0 * M_halo / (4.0 * math.pi * rho_m0)) ** (1.0 / 3.0)
        k_halo = 2.0 * math.pi / max(R, 0.01)
        delta_ns_corr = 3.0 * math.log10(self.sigma_ratio(k_halo))

        # Boundary dissipation (only active at z < z_PT)
        f_growth    = self.boundary_growth_suppression_v2(z_form=z)
        dissip_corr = 3.0 * math.log10(max(f_growth, 1e-10))

        return log_phi_lcdm + log_ratio + delta_ns_corr + dissip_corr


# ═══════════════════════════════════════════════════════════════════════════
#  COUPLING GAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def required_sigma_enhancement(
    log_phi_observed: float,
    log_phi_lcdm: float,
    n_tail: float = 5.0,
) -> float:
    """What σ enhancement factor would BPR need to match JWST?

    At exponential tail of PS function:
        Δ(log φ) ≈ n_tail × Δ(log σ)

    Returns σ_BPR / σ_ΛCDM required.
    """
    delta_log_phi = log_phi_observed - log_phi_lcdm
    return 10.0 ** (delta_log_phi / n_tail)


def required_n_s(
    sigma_ratio_needed: float,
    k_Mpc: float,
    n_s_lcdm: float = 0.9649,
) -> float:
    """What n_s would BPR need to achieve the required σ enhancement?

    σ_BPR/σ_ΛCDM = (k/k_pivot)^(Δn_s / 2)
    → Δn_s = 2 log(ratio) / log(k/k_pivot)
    """
    if k_Mpc <= _K_PIVOT or sigma_ratio_needed <= 0:
        return n_s_lcdm
    delta_ns = 2.0 * math.log(sigma_ratio_needed) / math.log(k_Mpc / _K_PIVOT)
    return n_s_lcdm + delta_ns


def hubble_tension_fraction(H0_bpr: float) -> float:
    """Fraction of Hubble tension explained by BPR."""
    tension = _H0_LOCAL - _H0_PLANCK
    bpr_contribution = H0_bpr - _H0_PLANCK
    return bpr_contribution / tension


# ═══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def run_jwst_comparison(p: int = _P) -> dict:
    """Run all three JWST anomaly comparisons and return a results dict."""
    bpr    = BPRCosmology(p=p)
    bpr_v2 = BPRCosmologyV2(p=p)
    lcdm   = LambdaCDM()

    results = {}

    # ── 1. Hubble tension ──────────────────────────────────────────────────
    H0_bpr  = bpr.H0_bpr
    f_tension = hubble_tension_fraction(H0_bpr)
    results["hubble_tension"] = {
        "H0_planck_km_s_Mpc": _H0_PLANCK,
        "H0_local_km_s_Mpc":  _H0_LOCAL,
        "H0_bpr_km_s_Mpc":    H0_bpr,
        "delta_Neff":          bpr.delta_Neff,
        "fraction_explained":  f_tension,
        "tension_sigma":       (_H0_LOCAL - _H0_PLANCK) / math.sqrt(_H0_ERR_LOCAL**2 + 0.5**2),
    }

    # ── 2. S8 tension ─────────────────────────────────────────────────────
    S8_bpr    = bpr.S8_bpr
    S8_bpr_v2 = bpr_v2.S8_v2
    S8_tension_sigma = (_S8_PLANCK - _S8_WL_OBS) / _S8_WL_ERR
    raw_fraction    = (_S8_PLANCK - S8_bpr)    / (_S8_PLANCK - _S8_WL_OBS)
    raw_fraction_v2 = (_S8_PLANCK - S8_bpr_v2) / (_S8_PLANCK - _S8_WL_OBS)
    results["s8_tension"] = {
        "S8_planck":            _S8_PLANCK,
        "S8_observed_WL":       _S8_WL_OBS,
        "S8_bpr":               S8_bpr,
        "sigma8_bpr":           bpr.sigma8_bpr,
        "S8_bpr_v2":            S8_bpr_v2,
        "sigma8_bpr_v2":        bpr_v2.sigma8_v2,
        "tension_sigma_lcdm":   S8_tension_sigma,
        "tension_sigma_bpr":    abs(S8_bpr    - _S8_WL_OBS) / _S8_WL_ERR,
        "tension_sigma_bpr_v2": abs(S8_bpr_v2 - _S8_WL_OBS) / _S8_WL_ERR,
        "fraction_explained":   raw_fraction,      # may exceed 1.0 (overshoot)
        "fraction_explained_v2": raw_fraction_v2,
        "overshoots":    S8_bpr    < _S8_WL_OBS,
        "overshoots_v2": S8_bpr_v2 < _S8_WL_OBS,
    }

    # ── 3. JWST UV LF ─────────────────────────────────────────────────────
    uv_comparisons = []
    for pt in JWST_UV_LF:
        log_phi_lcdm = lcdm.uv_luminosity_function(pt.M_UV, pt.z)
        log_phi_bpr  = bpr.uv_luminosity_function(pt.M_UV, pt.z)
        log_phi_mond = bpr.uv_luminosity_function_mond(pt.M_UV, pt.z)
        log_phi_v2   = bpr_v2.uv_luminosity_function_v2(pt.M_UV, pt.z)
        gap_lcdm     = pt.log_phi - log_phi_lcdm
        gap_bpr      = pt.log_phi - log_phi_bpr
        gap_mond     = pt.log_phi - log_phi_mond
        gap_v2       = pt.log_phi - log_phi_v2
        bpr_closes   = (gap_lcdm - gap_bpr)  / abs(gap_lcdm) if gap_lcdm != 0 else 0
        mond_closes  = (gap_lcdm - gap_mond) / abs(gap_lcdm) if gap_lcdm != 0 else 0
        v2_closes    = (gap_lcdm - gap_v2)   / abs(gap_lcdm) if gap_lcdm != 0 else 0

        sigma_needed = required_sigma_enhancement(pt.log_phi, log_phi_lcdm)
        n_s_needed   = required_n_s(sigma_needed, k_Mpc=5.0)

        uv_comparisons.append({
            "z":           pt.z,
            "M_UV":        pt.M_UV,
            "log_phi_obs": pt.log_phi,
            "log_phi_lcdm": log_phi_lcdm,
            "log_phi_bpr": log_phi_bpr,
            "log_phi_mond": log_phi_mond,
            "log_phi_v2":  log_phi_v2,
            "gap_lcdm_dex": gap_lcdm,
            "gap_bpr_dex":  gap_bpr,
            "gap_mond_dex": gap_mond,
            "gap_v2_dex":   gap_v2,
            "bpr_fraction_closed":  bpr_closes,
            "mond_fraction_closed": mond_closes,
            "v2_fraction_closed":   v2_closes,
            "sigma_ratio_needed": sigma_needed,
            "n_s_needed_to_explain": n_s_needed,
            "source": pt.source,
        })

    results["jwst_uv_lf"] = uv_comparisons
    results["bpr_params"] = {
        "p":            p,
        "n_s_bpr":      bpr.n_s,
        "n_s_lcdm":     0.9649,
        "delta_n_s":    bpr.n_s - 0.9649,
        "delta_Neff":   bpr.delta_Neff,
        "mond_a0_m_s2": bpr.mond_a0,
        "z_pt":         bpr_v2.z_pt,
    }

    return results
