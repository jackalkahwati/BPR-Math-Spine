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
#  BPR V3 — SUBSTRATE ZONE-BOUNDARY ENHANCEMENT
# ═══════════════════════════════════════════════════════════════════════════

class BPRCosmologyV3(BPRCosmologyV2):
    """BPR V3 — Universal Phase Transition Taxonomy + Substrate Zone-Boundary Enhancement.

    Adds a scale-dependent gravitational coupling derived purely from the
    prime substrate p, on top of the V2 epoch-dependent MOND gate.

    DERIVATION OF k_★
    ------------------
    The prime substrate p=104729 has two characteristic mode numbers:
        n₁ = p^{1/3} ≈ 47.1  →  N_efolds ≈ 62.9  (already used in V1/V2)
        n₂ = p^{2/3} ≈ 2223  →  zone-boundary wavenumber  (this theory)

    The Hubble sphere has comoving radius R_H = c/H₀.
    Mode n₂ of the substrate on the Hubble sphere defines:
        k_★ = 2π × n₂ / R_H = 2π × p^{2/3} × H₀ / c
             ≈ 2π × 2223 × 2.247×10⁻⁴ Mpc⁻¹
             ≈ 3.14 Mpc⁻¹

    This is the "zone-boundary" of the substrate lattice on cosmological
    scales — analogous to the Brillouin zone boundary in crystals, where
    phonon dispersion changes character (Umklapp-type mode coupling).

    PHYSICAL BEHAVIOUR
    ------------------
    k < k_★  (scales > 2 Mpc):
        Bulk-dominated coupling — standard gravitational behaviour.

    k > k_★  (scales < 2 Mpc, galaxy formation band):
        Boundary modes couple with an additional 1/d = 1/3 fraction of
        the gravitational coupling (d=3 spatial dims).  Boundary d.o.f.
        contribute equally to bulk d.o.f. above the zone boundary.

            σ_eff(k)/σ_ΛCDM(k) = σ_V1(k) × [1 + (1/3) × f_ZB(k)]

    SCALE SEPARATION
    ----------------
    σ₈ (k=0.2 Mpc⁻¹) ≪ k_★=3.14 → f_ZB ≈ 0 → S8 tension UNCHANGED
    Galaxy halos (k=4–6 Mpc⁻¹) > k_★    → f_ZB ≈ 1 → σ × 4/3
    JWST log φ boost: 3×log₁₀(4/3) ≈ 0.375 dex per point

    ZERO FREE PARAMETERS
    --------------------
    k_★ derived from p alone; enhancement amplitude 1/d = 1/3 (d=3 fixed);
    logistic sharpness p^{1/3} (same mode number as N_efolds).
    """

    @property
    def k_star(self) -> float:
        """Zone-boundary wavenumber k_★ = 2π p^{2/3} H₀/c [Mpc⁻¹].

        For p=104729: p^{2/3} ≈ 2223, k_H ≈ 2.247×10⁻⁴ Mpc⁻¹
        → k_★ ≈ 3.14 Mpc⁻¹  (galaxy-formation band).
        """
        k_H = _H0_PLANCK / 3e5     # H₀/c in Mpc⁻¹
        return 2.0 * math.pi * self.p ** (2.0 / 3.0) * k_H

    def zone_boundary_factor(self, k_Mpc: float) -> float:
        """Logistic step at k_★ with sharpness p^{1/3}.

        f_ZB(k) = 1 / (1 + exp(−p^{1/3} (k/k_★ − 1)))

        At k=0.2 Mpc⁻¹: f_ZB ≈ 0 (S8 scale unaffected).
        At k=5.0 Mpc⁻¹: f_ZB ≈ 1 (galaxy formation enhanced).
        """
        k_s = self.k_star
        x = self.p ** (1.0 / 3.0) * (k_Mpc / k_s - 1.0)
        x = max(-500.0, min(500.0, x))   # guard against overflow
        return 1.0 / (1.0 + math.exp(-x))

    def sigma_ratio_v3(self, k_Mpc: float) -> float:
        """σ_BPR/σ_ΛCDM with zone-boundary enhancement.

        σ_V3(k) = σ_V1(k) × [1 + (1/3) × f_ZB(k)]

        The 1/3 = 1/d amplitude is the boundary fraction of gravitational
        degrees of freedom in d=3 spatial dimensions (no free parameters).
        """
        base = self.sigma_ratio(k_Mpc)
        return base * (1.0 + (1.0 / 3.0) * self.zone_boundary_factor(k_Mpc))

    @property
    def sigma8_v3(self) -> float:
        """σ₈ with zone-boundary enhancement.

        k_sigma8 = 0.2 ≪ k_★ = 3.14  →  f_ZB(0.2) ≈ 0
        → sigma_ratio_v3(0.2) ≈ sigma_ratio(0.2)  →  σ₈_V3 ≈ σ₈_V2.
        """
        k_sigma8 = 0.2
        sigma_r  = self.sigma_ratio_v3(k_sigma8)
        f_growth = self.boundary_growth_suppression_v2(z_form=0.0)
        G_ratio  = self.G_eff_ratio(k_sigma8)
        return _SIGMA8_PLANCK * sigma_r * f_growth * math.sqrt(G_ratio)

    @property
    def S8_v3(self) -> float:
        """BPR V3 S₈ = σ₈_V3 √(Ωm / 0.3)."""
        return self.sigma8_v3 * math.sqrt(_OMEGA_M / 0.3)

    def uv_luminosity_function_v3(self, M_UV: float, z: float) -> float:
        """log₁₀(φ) with V2 MOND gate and zone-boundary enhancement.

        V2 (Universal Phase Transition Taxonomy):
            z > z_PT → MOND δ_c=1.33, dissipation suspended
            z ≤ z_PT → Newtonian δ_c=1.686, dissipation from z_PT

        Substrate Zone-Boundary Enhancement (V3 addition):
            spectral correction uses sigma_ratio_v3 instead of sigma_ratio
            → 3×log₁₀(σ_V3/σ_V1) additional dex at k_halo > k_★
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

        # PS ratio from epoch-gated MOND (V2 logic)
        if delta_c_eff < delta_c_newton:
            log_ratio = (math.log10(nu_e / nu_n)
                         + (nu_n ** 2 - nu_e ** 2) / (2.0 * math.log(10.0)))
            log_ratio = max(-4.0, min(4.0, log_ratio))
        else:
            log_ratio = 0.0

        # Zone-boundary enhanced spectral correction (V3 change vs V2)
        rho_m0 = 2.775e11 * _OMEGA_M * (_H0_PLANCK / 100.0) ** 2
        R      = (3.0 * M_halo / (4.0 * math.pi * rho_m0)) ** (1.0 / 3.0)
        k_halo = 2.0 * math.pi / max(R, 0.01)
        delta_ns_corr = 3.0 * math.log10(self.sigma_ratio_v3(k_halo))

        # Boundary dissipation (only active at z < z_PT, same as V2)
        f_growth    = self.boundary_growth_suppression_v2(z_form=z)
        dissip_corr = 3.0 * math.log10(max(f_growth, 1e-10))

        return log_phi_lcdm + log_ratio + delta_ns_corr + dissip_corr


# ═══════════════════════════════════════════════════════════════════════════
#  BPR V4 — IMPEDANCE-WEIGHTED COLLAPSE THRESHOLD
# ═══════════════════════════════════════════════════════════════════════════

# SI constants for virial computation (module-level to avoid per-call lookup)
_G_SI   = 6.674e-11        # m³/(kg s²)
_MSUN   = 1.989e30         # kg
_MPC_SI = 3.086e22         # m


class BPRCosmologyV4(BPRCosmologyV2):
    """BPR V4 — Universal Phase Transition Taxonomy + Impedance-Weighted Collapse Threshold.

    Replaces the hard MOND / Newtonian step in V2 with a continuous,
    mass-dependent collapse threshold derived from the Vacuum Impedance
    Mismatch interpolation function μ(a/a₀).

    DERIVATION
    ----------
    Vacuum Impedance Mismatch gives MOND with scale a₀ ≈ 1.18×10⁻¹⁰ m/s².
    In standard MOND, μ(a/a₀) interpolates between two regimes:
        μ → 0  (a ≪ a₀):  deep MOND  → gravity ∝ 1/r
        μ → 1  (a ≫ a₀):  Newtonian  → gravity ∝ 1/r²

    For spherical collapse, the same interpolation determines the effective
    overdensity threshold δ_c.  For a virialized halo with mass M at
    redshift z, the characteristic internal acceleration is:

        a_vir(M, z) = G M / R_vir(M, z)²

    where R_vir = (3M / (4π × 200 ρ_crit(z)))^{1/3}.

    The impedance interpolation then gives the collapse threshold:

        δ_c(M, z) = δ_c^{MOND} + (δ_c^{Newton} − δ_c^{MOND}) × μ(a_vir/a₀)
                  = 1.330 + 0.356 × μ(a_vir/a₀)

    using μ(x) = x/√(1+x²) (standard form 2 from Famaey & McGaugh 2012).

    TRANSITION MASS M★(z)
    ----------------------
    At a_vir(M★) = a₀, setting to zero:
        G M★^{1/3} × (800π ρ_crit(z)/3)^{2/3} = a₀
        M★(z) = (a₀/G)³ / (800π ρ_crit(z)/3)²

    Key values (from Vacuum Impedance Mismatch a₀):
        M★(z=9)  = 5.4×10¹¹ M☉  — both z=9 halos below M★ → MOND-like
        M★(z=10) = 3.1×10¹¹ M☉  — bright z=10 halos near M★ → transition
        M★(z=12) = 1.1×10¹¹ M☉  — z=12 bright halos above M★ → Newtonian
        M★(z=16) = 2.3×10¹⁰ M☉  — z=16 halos well above M★ → Newtonian

    PHYSICAL PICTURE
    ----------------
    At higher z the universe is denser → R_vir smaller → a_vir larger →
    crosses a₀ at lower halo mass.  Galaxy formation at z=12–16 operates
    in the Newtonian regime (a_vir > a₀), reducing the runaway overshoot
    that plagues V2 at those redshifts.  At z=9 halos remain below M★
    and still receive a partial MOND boost (δ_c ≈ 1.53, vs 1.33 in V2).

    ZERO FREE PARAMETERS
    --------------------
    a₀ from Vacuum Impedance Mismatch (p, c, H₀ only); μ is fixed
    by theory; z_PT from V2; 200×ρ_crit is the standard virial convention.
    """

    # ── virial acceleration ────────────────────────────────────────────────

    def _virial_acceleration(self, M_halo_Msun: float, z: float) -> float:
        """Characteristic gravitational acceleration of a virialized halo [m/s²].

        a_vir = G M / R_vir²,   R_vir = (3M / (4π × 200 ρ_crit(z)))^{1/3}

        Uses ΛCDM E(z)² = Ωm(1+z)³ + ΩΛ for ρ_crit(z).
        """
        M_si  = M_halo_Msun * _MSUN
        H0_si = _H0_PLANCK * 1e3 / _MPC_SI
        om, ol = _OMEGA_M, 1.0 - _OMEGA_M
        a_scale = 1.0 / (1.0 + z)
        E2    = om / a_scale ** 3 + ol
        rho_c = 3.0 * (H0_si * math.sqrt(E2)) ** 2 / (8.0 * math.pi * _G_SI)
        R_vir = (3.0 * M_si / (4.0 * math.pi * 200.0 * rho_c)) ** (1.0 / 3.0)
        return _G_SI * M_si / R_vir ** 2

    def m_star(self, z: float) -> float:
        """Transition mass M★(z) [M☉] where a_vir = a₀.

        M★(z) = (a₀/G)³ / (800π ρ_crit(z)/3)²

        Above M★, halos are Newtonian; below M★, they are MOND-like.
        M★ decreases with z (higher z → denser halos → Newtonian at lower M).
        """
        H0_si = _H0_PLANCK * 1e3 / _MPC_SI
        om, ol = _OMEGA_M, 1.0 - _OMEGA_M
        a_scale = 1.0 / (1.0 + z)
        E2    = om / a_scale ** 3 + ol
        rho_c = 3.0 * (H0_si * math.sqrt(E2)) ** 2 / (8.0 * math.pi * _G_SI)
        rho_vir_f = 800.0 * math.pi * rho_c / 3.0
        return (self.mond_a0 / _G_SI) ** 3 / rho_vir_f ** 2 / _MSUN

    # ── continuous collapse threshold ──────────────────────────────────────

    def delta_c_v4(self, M_halo_Msun: float, z: float) -> float:
        """Mass-dependent collapse threshold from Vacuum Impedance Mismatch.

        z < z_PT  →  1.686  (Newtonian, same as V2)
        z > z_PT  →  1.330 + 0.356 × μ(a_vir/a₀)

        μ(x) = x/√(1+x²) is the MOND interpolation function
        (Famaey & McGaugh 2012, form 2).
        """
        delta_c_newton = 1.686
        if z <= self.z_pt:
            return delta_c_newton

        a_vir = self._virial_acceleration(M_halo_Msun, z)
        x     = a_vir / self.mond_a0
        mu    = x / math.sqrt(1.0 + x * x)

        delta_c_mond = 1.33
        return delta_c_mond + (delta_c_newton - delta_c_mond) * mu

    # ── σ₈ / S₈ (unchanged from V2 — z=0 is always Newtonian) ─────────────

    @property
    def sigma8_v4(self) -> float:
        """σ₈ for V4: identical to V2 (collapse threshold irrelevant at z=0)."""
        return self.sigma8_v2

    @property
    def S8_v4(self) -> float:
        """BPR V4 S₈ = σ₈_V4 √(Ωm / 0.3)."""
        return self.sigma8_v4 * math.sqrt(_OMEGA_M / 0.3)

    # ── UV luminosity function ─────────────────────────────────────────────

    def uv_luminosity_function_v4(self, M_UV: float, z: float) -> float:
        """log₁₀(φ) with impedance-weighted continuous δ_c(M, z).

        z < z_PT → Newtonian (same as V2)
        z > z_PT → δ_c = 1.33 + 0.356 × μ(a_vir/a₀)

        This replaces V2's hard step with a smooth mass-dependent threshold:
        • Low-mass / low-z halos (a_vir ≪ a₀): δ_c → 1.33 (deep MOND boost)
        • High-mass / high-z halos (a_vir ≫ a₀): δ_c → 1.686 (Newtonian)
        """
        log_phi_lcdm = self._lcdm.uv_luminosity_function(M_UV, z)

        M_halo  = 1e11 * 10.0 ** (-0.4 * (M_UV + 21.0))
        sigma_z = self._lcdm.sigma_M_at_z(M_halo, z)
        sigma_z = max(sigma_z, 1e-6)

        delta_c_newton = 1.686
        delta_c_eff    = self.delta_c_v4(M_halo, z)

        nu_n = delta_c_newton / sigma_z
        nu_e = delta_c_eff   / sigma_z

        # PS ratio from continuous MOND interpolation
        if delta_c_eff < delta_c_newton:
            log_ratio = (math.log10(nu_e / nu_n)
                         + (nu_n ** 2 - nu_e ** 2) / (2.0 * math.log(10.0)))
            log_ratio = max(-4.0, min(4.0, log_ratio))
        else:
            log_ratio = 0.0

        # Δn_s spectral correction (same as V2)
        rho_m0 = 2.775e11 * _OMEGA_M * (_H0_PLANCK / 100.0) ** 2
        R      = (3.0 * M_halo / (4.0 * math.pi * rho_m0)) ** (1.0 / 3.0)
        k_halo = 2.0 * math.pi / max(R, 0.01)
        delta_ns_corr = 3.0 * math.log10(self.sigma_ratio(k_halo))

        # Boundary dissipation (only active at z < z_PT, same as V2)
        f_growth    = self.boundary_growth_suppression_v2(z_form=z)
        dissip_corr = 3.0 * math.log10(max(f_growth, 1e-10))

        return log_phi_lcdm + log_ratio + delta_ns_corr + dissip_corr


# ═══════════════════════════════════════════════════════════════════════════
#  V5 — Impedance-Screened MOND Collapse (Vacuum Impedance Mismatch + V4)
# ═══════════════════════════════════════════════════════════════════════════

class BPRCosmologyV5(BPRCosmologyV4):
    """BPR V5 — Universal Phase Transition Taxonomy + Impedance-Screened MOND Collapse.

    Extends V4 by adding a topological impedance screening factor that suppresses
    the MOND boost for halos above the impedance crossover mass M_imp(z).

    PHYSICAL MECHANISM
    ------------------
    DM solitons aggregate at halo boundaries.  The collective boundary winding
    grows as the SQUARE of the mass ratio to the MOND transition mass M★(z):

        W_halo = (M / M★(z))²

    (Quadratic scaling reflects coherent winding-pair accumulation at the halo
    boundary, analogous to BKT Cooper pairing in BPR's Class B topology.)

    When W_halo > W_c = p^{1/5}, the boundary impedance Z_halo > √2 Z₀ screens
    the cosmological MOND phonon field, recovering Newtonian collapse.

    SCREENING FACTOR
    ----------------
    Mirrors TopologicalImpedance.em_coupling(W):

        g_screen(M, z) = 1 / (1 + (M/M★(z))⁴ / W_c²)

    CROSSOVER MASS
    --------------
    g_screen = 1/2 when (M/M★)⁴ = W_c²:
        M_imp(z) = W_c^{1/2} × M★(z)

    Key values  (W_c ≈ 10.09, M★ from `m_star()`):
        M_imp(z=9)  ≈  1.6×10¹² M☉
        M_imp(z=10) ≈  9.5×10¹¹ M☉  ≈ 10¹² M☉
        M_imp(z=12) ≈  3.2×10¹¹ M☉

    V5 COLLAPSE THRESHOLD
    ----------------------
    z < z_PT → 1.686 (Newtonian, same as V2/V4)
    z > z_PT → δ_c = 1.330 + 0.356 × μ(a_vir/a₀) × g_screen(M, z)

    EFFECT ON JWST UV LF
    --------------------
    Screening is concentrated at M_UV < −22.5 (the bright-end overshoot):
        g_screen(M_UV=−23, z=10) ≈ 0.34  →  MOND boost reduced 66%
        g_screen(M_UV=−21.5, z=10) ≈ 0.99 →  low-mass unaffected

    ZERO FREE PARAMETERS
    --------------------
    W_c = p^{1/5} (from derived_critical_winding); M★(z) from m_star() (V4).
    No new parameters beyond those in V4.
    """

    @property
    def _W_c(self) -> float:
        """Critical winding W_c = p^{1/5} (Vacuum Impedance Mismatch substrate scale)."""
        from .impedance import derived_critical_winding
        return derived_critical_winding(self.p)

    def _g_screen(self, M_halo_Msun: float, z: float) -> float:
        """Topological impedance screening factor g_screen(M, z).

        g_screen = 1 / (1 + (M/M★(z))⁴ / W_c²)

        Equals 1 (no screening) for M << M_imp; equals 1/2 at M = M_imp.
        """
        M_star = self.m_star(z)
        u = M_halo_Msun / max(M_star, 1e6)
        W_c = self._W_c
        return 1.0 / (1.0 + u ** 4 / W_c ** 2)

    def m_imp(self, z: float) -> float:
        """Impedance crossover mass M_imp(z) = W_c^{1/2} × M★(z) [M☉].

        Above M_imp, g_screen < 1/2 — the halo is impedance-screened.
        """
        return self._W_c ** 0.5 * self.m_star(z)

    def delta_c_v5(self, M_halo_Msun: float, z: float) -> float:
        """Impedance-screened MOND collapse threshold δ_c(M, z).

        z ≤ z_PT → 1.686  (Newtonian, same as V4)
        z > z_PT → 1.330 + 0.356 × μ(a_vir/a₀) × g_screen(M, z)
        """
        if z <= self.z_pt:
            return 1.686
        a_vir = self._virial_acceleration(M_halo_Msun, z)
        x = a_vir / self.mond_a0
        mu = x / math.sqrt(1.0 + x * x)
        g = self._g_screen(M_halo_Msun, z)
        return 1.330 + 0.356 * mu * g

    # ── S8 unchanged from V4 (z=0 is always Newtonian) ─────────────────────

    @property
    def S8_v5(self) -> float:
        """S8 = σ₈ √(Ωm/0.3) — identical to V4 (z=0 is Newtonian)."""
        return self.S8_v4

    @property
    def sigma8_v5(self) -> float:
        """σ₈ — identical to V4."""
        return self.sigma8_v4

    # ── UV luminosity function ───────────────────────────────────────────────

    def uv_luminosity_function_v5(self, M_UV: float, z: float) -> float:
        """log₁₀(φ) [Mpc⁻³ mag⁻¹] with impedance-screened continuous δ_c(M, z).

        Uses V5 threshold:
            z < z_PT → Newtonian (same as V4)
            z > z_PT → δ_c = 1.330 + 0.356 × μ(a_vir/a₀) × g_screen(M, z)
        """
        log_phi_lcdm = self._lcdm.uv_luminosity_function(M_UV, z)

        M_halo  = 1e11 * 10.0 ** (-0.4 * (M_UV + 21.0))
        sigma_z = self._lcdm.sigma_M_at_z(M_halo, z)
        sigma_z = max(sigma_z, 1e-6)

        delta_c_newton = 1.686
        delta_c_eff    = self.delta_c_v5(M_halo, z)

        nu_n = delta_c_newton / sigma_z
        nu_e = delta_c_eff    / sigma_z

        if delta_c_eff < delta_c_newton:
            log_ratio = (math.log10(nu_e / nu_n)
                         + (nu_n ** 2 - nu_e ** 2) / (2.0 * math.log(10.0)))
            log_ratio = max(-4.0, min(4.0, log_ratio))
        else:
            log_ratio = 0.0

        # Δn_s and dissipation corrections (same as V4)
        rho_m0 = 2.775e11 * _OMEGA_M * (_H0_PLANCK / 100.0) ** 2
        R = (3.0 * M_halo / (4.0 * math.pi * rho_m0)) ** (1.0 / 3.0)
        k_halo = 2.0 * math.pi / max(R, 0.01)
        delta_ns_corr = 3.0 * math.log10(self.sigma_ratio(k_halo))
        f_growth      = self.boundary_growth_suppression_v2(z_form=z)
        dissip_corr   = 3.0 * math.log10(max(f_growth, 1e-10))

        return log_phi_lcdm + log_ratio + delta_ns_corr + dissip_corr


# ═══════════════════════════════════════════════════════════════════════════
#  V6 — Corrected Impedance-Screened MOND Collapse
# ═══════════════════════════════════════════════════════════════════════════

class BPRCosmologyV6(BPRCosmologyV5):
    """BPR V6 — Corrected Impedance-Screened MOND Collapse.

    Fixes the formula error in V5.

    V5 BUG
    ------
    V5 used:
        δ_c = 1.330 + 0.356 × μ × g_screen

    When g_screen → 0 with μ ≈ 1 (a Newtonian halo above M_imp), this drives
    δ_c → 1.330 — the deep-MOND floor — for a halo that is physically
    Newtonian.  At z=16, where M★(z) is tiny and all halos satisfy M >> M★,
    g_screen ≈ 0.2 and μ ≈ 0.85, giving δ_c_V5 ≈ 1.39 vs V4's 1.63 — a
    sub-Newtonian collapse threshold for a nearly-Newtonian halo.

    THE FIX
    -------
    Screen the MOND *boost* (deviation from Newtonian), not the full
    interpolation:

        δ_c_V6 = 1.686 − 0.356 × (1 − μ) × g_screen

    Derivation:
        V4 formula:  δ_c = 1.686 − 0.356 × (1 − μ)   [rewrite of 1.330 + 0.356μ]
        MOND boost:  Δ = 0.356 × (1 − μ)              [reduction from Newtonian]
        Screening:   Δ_screened = Δ × g_screen
        → δ_c = 1.686 − 0.356 × (1 − μ) × g_screen

    Behaviour at extremes:
        μ → 1  (Newtonian):  δ_c = 1.686   for any g_screen  ✓
        μ → 0, g=1 (MOND, no screen):  δ_c = 1.330  ✓
        μ → 0, g=0 (MOND, screened):   δ_c = 1.686  (Newtonian restored)  ✓

    EQUIVALENCE
    -----------
    V6 ≡ V5 only when g_screen = 1 (no screening).
    They differ by  0.356 × (1 − g_screen) > 0 when g_screen < 1:
        V6 always gives δ_c ≥ V5's δ_c (same or higher threshold).

    EFFECT ON JWST BINS
    -------------------
    z=9:   g_screen ≈ 1 → V6 ≡ V5 ≡ V4  (screening doesn't engage)
    z=10:  g_screen ≈ 0.996 → V6 ≈ V5 ≈ V4  (tiny effect)
    z=12:  g_screen ≈ 0.81, μ ≈ 0.79 → V6 slightly above V4 (screens small boost)
    z=16:  g_screen ≈ 0.21, μ ≈ 0.85 → V6 ≈ V4  (halo near-Newtonian, tiny boost)

    The z=16 overshoot from V5 is fully corrected.
    """

    def delta_c_v6(self, M_halo_Msun: float, z: float) -> float:
        """Corrected impedance-screened MOND collapse threshold δ_c(M, z).

        z ≤ z_PT → 1.686  (Newtonian, same as V4/V5)
        z > z_PT → 1.686 − 0.356 × (1 − μ(a_vir/a₀)) × g_screen(M, z)

        Screens the MOND boost (deviation from Newtonian), not the full
        interpolation.  δ_c is always ≥ 1.330 and ≤ 1.686.
        """
        if z <= self.z_pt:
            return 1.686
        a_vir = self._virial_acceleration(M_halo_Msun, z)
        x = a_vir / self.mond_a0
        mu = x / math.sqrt(1.0 + x * x)
        g = self._g_screen(M_halo_Msun, z)
        return 1.686 - 0.356 * (1.0 - mu) * g

    @property
    def S8_v6(self) -> float:
        """S8 — identical to V4/V5 (z=0 is always Newtonian)."""
        return self.S8_v4

    @property
    def sigma8_v6(self) -> float:
        """σ₈ — identical to V4/V5."""
        return self.sigma8_v4

    def uv_luminosity_function_v6(self, M_UV: float, z: float) -> float:
        """log₁₀(φ) [Mpc⁻³ mag⁻¹] with corrected impedance-screened δ_c(M, z).

        Uses V6 threshold:
            z < z_PT → Newtonian (same as V4/V5)
            z > z_PT → δ_c = 1.686 − 0.356 × (1 − μ) × g_screen
        """
        log_phi_lcdm = self._lcdm.uv_luminosity_function(M_UV, z)

        M_halo  = 1e11 * 10.0 ** (-0.4 * (M_UV + 21.0))
        sigma_z = self._lcdm.sigma_M_at_z(M_halo, z)
        sigma_z = max(sigma_z, 1e-6)

        delta_c_newton = 1.686
        delta_c_eff    = self.delta_c_v6(M_halo, z)

        nu_n = delta_c_newton / sigma_z
        nu_e = delta_c_eff    / sigma_z

        if delta_c_eff < delta_c_newton:
            log_ratio = (math.log10(nu_e / nu_n)
                         + (nu_n ** 2 - nu_e ** 2) / (2.0 * math.log(10.0)))
            log_ratio = max(-4.0, min(4.0, log_ratio))
        else:
            log_ratio = 0.0

        rho_m0 = 2.775e11 * _OMEGA_M * (_H0_PLANCK / 100.0) ** 2
        R = (3.0 * M_halo / (4.0 * math.pi * rho_m0)) ** (1.0 / 3.0)
        k_halo = 2.0 * math.pi / max(R, 0.01)
        delta_ns_corr = 3.0 * math.log10(self.sigma_ratio(k_halo))
        f_growth      = self.boundary_growth_suppression_v2(z_form=z)
        dissip_corr   = 3.0 * math.log10(max(f_growth, 1e-10))

        return log_phi_lcdm + log_ratio + delta_ns_corr + dissip_corr


# ═══════════════════════════════════════════════════════════════════════════
#  V7 — MOND-Enhanced Baryonic Retention (f_star channel)
# ═══════════════════════════════════════════════════════════════════════════

class BPRCosmologyV7(BPRCosmologyV6):
    """BPR V7 — MOND-Enhanced Star Formation Efficiency (f_star channel).

    Addresses the z=9 underprediction through a physically independent channel:
    enhanced baryonic retention in MOND-regime halos from boundary impedance
    mismatch, distinct from the collapse-threshold δ_c mechanism in V2–V6.

    DERIVATION
    ----------
    In Vacuum Impedance Mismatch, the halo boundary has impedance:

        Z_halo = Z₀ / μ(a_vir/a₀)

    Feedback-driven gas outflows cross from the halo (Z_halo) to the IGM
    (Z_IGM ≈ Z₀).  At an impedance step, the reflection coefficient for
    electromagnetic / phonon energy is:

        R(μ) = ((Z_halo − Z₀) / (Z_halo + Z₀))²
              = ((1 − μ) / (1 + μ))²

    Gas that is "reflected" at the boundary is retained and forms stars.
    Gas that is transmitted escapes as a feedback-driven outflow.

    The baryonic retention enhancement over Newtonian:

        η(μ) = 1 + R(μ) = 1 + ((1 − μ)/(1 + μ))²

    Boundary values:
        μ = 1  (Newtonian):   R = 0,  η = 1  (no enhancement)  ✓
        μ → 0  (deep MOND):   R → 1,  η → 2  (retention doubles)  ✓

    COUPLING TO UV LUMINOSITY FUNCTION
    -----------------------------------
    Enhanced f_star means the same M_halo produces more stellar mass and thus
    a brighter galaxy.  Equivalently, a galaxy observed at M_UV in BPR comes
    from a LESS MASSIVE (more abundant) halo than in ΛCDM.

    The effective M_UV shift (galaxy is brighter by this amount):

        ΔM_UV = 2.5 × log₁₀(η(μ))   [positive → dimmer in ΛCDM sense]

    For a fixed observed M_UV, the BPR galaxy corresponds to M_UV + ΔM_UV in
    ΛCDM (dimmer → more common).  The f_star correction to log φ:

        Δ log φ_fstar = log φ_ΛCDM(M_UV + ΔM_UV, z) − log φ_ΛCDM(M_UV, z)

    This is always positive (dimmer M_UV → more abundant) and naturally
    captures the local slope of the PS mass function without approximation.

    ZERO FREE PARAMETERS
    --------------------
    μ from Vacuum Impedance Mismatch (existing V4 method); η from the
    standard impedance reflection formula.  No tuning.

    EXPECTED MAGNITUDE
    ------------------
    At z=9, M_UV=−22 (μ ≈ 0.61):  η ≈ 1.060,  ΔM_UV ≈ +0.063 mag
    Δ log φ ≈ 0.1–0.2 dex  (ΛCDM slope × ΔM_UV / 2.5)

    HONEST CAVEAT
    -------------
    The reflection formula limits η ≤ 2, so ΔM_UV ≤ 0.75 mag and Δ log φ
    is bounded below the 1.2 dex needed to close the z=9 gap.  The mechanism
    is correct physics but quantitatively insufficient alone — establishing
    this precisely is the scientific result.  The z=9/z=10 coupling (same
    halo mass, similar μ) means any larger enhancement also boosts z=10.

    ONLY ACTIVE AT z > z_PT (MOND epoch, same gate as V2–V6).
    """

    def _fstar_retention(self, M_halo_Msun: float, z: float) -> float:
        """Baryonic retention enhancement η(μ) from boundary impedance mismatch.

        η(μ) = 1 + ((1−μ)/(1+μ))²

        Returns 1.0 at z ≤ z_PT (Newtonian epoch, no enhancement).
        """
        if z <= self.z_pt:
            return 1.0
        a_vir = self._virial_acceleration(M_halo_Msun, z)
        x = a_vir / self.mond_a0
        mu = x / math.sqrt(1.0 + x * x)
        r = (1.0 - mu) / (1.0 + mu)
        return 1.0 + r * r

    def uv_luminosity_function_v7(self, M_UV: float, z: float) -> float:
        """log₁₀(φ) with V6 collapse threshold + MOND-enhanced f_star retention.

        Adds the f_star correction on top of V6's δ_c mechanism:

            Δ log φ_fstar = log φ_ΛCDM(M_UV + ΔM_UV, z) − log φ_ΛCDM(M_UV, z)
            ΔM_UV = 2.5 × log₁₀(η(μ))  (brightening from enhanced retention)
        """
        log_phi_v6 = self.uv_luminosity_function_v6(M_UV, z)

        if z <= self.z_pt:
            return log_phi_v6

        M_halo = 1e11 * 10.0 ** (-0.4 * (M_UV + 21.0))
        eta    = self._fstar_retention(M_halo, z)
        if abs(eta - 1.0) < 1e-10:
            return log_phi_v6

        delta_MUV = 2.5 * math.log10(eta)          # positive → dimmer M_UV
        M_UV_shifted = M_UV + delta_MUV             # shifted ΛCDM effective M_UV

        log_phi_lcdm_shifted = self._lcdm.uv_luminosity_function(M_UV_shifted, z)
        log_phi_lcdm_base    = self._lcdm.uv_luminosity_function(M_UV, z)

        fstar_corr = log_phi_lcdm_shifted - log_phi_lcdm_base   # > 0
        fstar_corr = max(0.0, min(4.0, fstar_corr))              # physical cap

        return log_phi_v6 + fstar_corr

    @property
    def S8_v7(self) -> float:
        """S8 — identical to V6 (f_star affects only high-z UV LF, not 8 Mpc clustering)."""
        return self.S8_v6

    @property
    def sigma8_v7(self) -> float:
        """σ₈ — identical to V6."""
        return self.sigma8_v6


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
    bpr_v3 = BPRCosmologyV3(p=p)
    bpr_v4 = BPRCosmologyV4(p=p)
    bpr_v5 = BPRCosmologyV5(p=p)
    bpr_v6 = BPRCosmologyV6(p=p)
    bpr_v7 = BPRCosmologyV7(p=p)
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
    S8_bpr_v3 = bpr_v3.S8_v3
    S8_bpr_v4 = bpr_v4.S8_v4
    S8_bpr_v5 = bpr_v5.S8_v5
    S8_bpr_v6 = bpr_v6.S8_v6
    S8_bpr_v7 = bpr_v7.S8_v7
    S8_tension_sigma = (_S8_PLANCK - _S8_WL_OBS) / _S8_WL_ERR
    raw_fraction    = (_S8_PLANCK - S8_bpr)    / (_S8_PLANCK - _S8_WL_OBS)
    raw_fraction_v2 = (_S8_PLANCK - S8_bpr_v2) / (_S8_PLANCK - _S8_WL_OBS)
    raw_fraction_v3 = (_S8_PLANCK - S8_bpr_v3) / (_S8_PLANCK - _S8_WL_OBS)
    raw_fraction_v4 = (_S8_PLANCK - S8_bpr_v4) / (_S8_PLANCK - _S8_WL_OBS)
    raw_fraction_v5 = (_S8_PLANCK - S8_bpr_v5) / (_S8_PLANCK - _S8_WL_OBS)
    raw_fraction_v6 = (_S8_PLANCK - S8_bpr_v6) / (_S8_PLANCK - _S8_WL_OBS)
    raw_fraction_v7 = (_S8_PLANCK - S8_bpr_v7) / (_S8_PLANCK - _S8_WL_OBS)
    results["s8_tension"] = {
        "S8_planck":            _S8_PLANCK,
        "S8_observed_WL":       _S8_WL_OBS,
        "S8_bpr":               S8_bpr,
        "sigma8_bpr":           bpr.sigma8_bpr,
        "S8_bpr_v2":            S8_bpr_v2,
        "sigma8_bpr_v2":        bpr_v2.sigma8_v2,
        "S8_bpr_v3":            S8_bpr_v3,
        "sigma8_bpr_v3":        bpr_v3.sigma8_v3,
        "S8_bpr_v4":            S8_bpr_v4,
        "sigma8_bpr_v4":        bpr_v4.sigma8_v4,
        "S8_bpr_v5":            S8_bpr_v5,
        "sigma8_bpr_v5":        bpr_v5.sigma8_v5,
        "S8_bpr_v6":            S8_bpr_v6,
        "sigma8_bpr_v6":        bpr_v6.sigma8_v6,
        "S8_bpr_v7":            S8_bpr_v7,
        "sigma8_bpr_v7":        bpr_v7.sigma8_v7,
        "tension_sigma_lcdm":   S8_tension_sigma,
        "tension_sigma_bpr":    abs(S8_bpr    - _S8_WL_OBS) / _S8_WL_ERR,
        "tension_sigma_bpr_v2": abs(S8_bpr_v2 - _S8_WL_OBS) / _S8_WL_ERR,
        "tension_sigma_bpr_v3": abs(S8_bpr_v3 - _S8_WL_OBS) / _S8_WL_ERR,
        "tension_sigma_bpr_v4": abs(S8_bpr_v4 - _S8_WL_OBS) / _S8_WL_ERR,
        "tension_sigma_bpr_v5": abs(S8_bpr_v5 - _S8_WL_OBS) / _S8_WL_ERR,
        "tension_sigma_bpr_v6": abs(S8_bpr_v6 - _S8_WL_OBS) / _S8_WL_ERR,
        "tension_sigma_bpr_v7": abs(S8_bpr_v7 - _S8_WL_OBS) / _S8_WL_ERR,
        "fraction_explained":   raw_fraction,
        "fraction_explained_v2": raw_fraction_v2,
        "fraction_explained_v3": raw_fraction_v3,
        "fraction_explained_v4": raw_fraction_v4,
        "fraction_explained_v5": raw_fraction_v5,
        "fraction_explained_v6": raw_fraction_v6,
        "fraction_explained_v7": raw_fraction_v7,
        "overshoots":    S8_bpr    < _S8_WL_OBS,
        "overshoots_v2": S8_bpr_v2 < _S8_WL_OBS,
        "overshoots_v3": S8_bpr_v3 < _S8_WL_OBS,
        "overshoots_v4": S8_bpr_v4 < _S8_WL_OBS,
        "overshoots_v5": S8_bpr_v5 < _S8_WL_OBS,
        "overshoots_v6": S8_bpr_v6 < _S8_WL_OBS,
        "overshoots_v7": S8_bpr_v7 < _S8_WL_OBS,
    }

    # ── 3. JWST UV LF ─────────────────────────────────────────────────────
    uv_comparisons = []
    for pt in JWST_UV_LF:
        log_phi_lcdm = lcdm.uv_luminosity_function(pt.M_UV, pt.z)
        log_phi_bpr  = bpr.uv_luminosity_function(pt.M_UV, pt.z)
        log_phi_mond = bpr.uv_luminosity_function_mond(pt.M_UV, pt.z)
        log_phi_v2   = bpr_v2.uv_luminosity_function_v2(pt.M_UV, pt.z)
        log_phi_v3   = bpr_v3.uv_luminosity_function_v3(pt.M_UV, pt.z)
        log_phi_v4   = bpr_v4.uv_luminosity_function_v4(pt.M_UV, pt.z)
        log_phi_v5   = bpr_v5.uv_luminosity_function_v5(pt.M_UV, pt.z)
        log_phi_v6   = bpr_v6.uv_luminosity_function_v6(pt.M_UV, pt.z)
        log_phi_v7   = bpr_v7.uv_luminosity_function_v7(pt.M_UV, pt.z)
        gap_lcdm     = pt.log_phi - log_phi_lcdm
        gap_bpr      = pt.log_phi - log_phi_bpr
        gap_mond     = pt.log_phi - log_phi_mond
        gap_v2       = pt.log_phi - log_phi_v2
        gap_v3       = pt.log_phi - log_phi_v3
        gap_v4       = pt.log_phi - log_phi_v4
        gap_v5       = pt.log_phi - log_phi_v5
        gap_v6       = pt.log_phi - log_phi_v6
        gap_v7       = pt.log_phi - log_phi_v7
        bpr_closes   = (gap_lcdm - gap_bpr)  / abs(gap_lcdm) if gap_lcdm != 0 else 0
        mond_closes  = (gap_lcdm - gap_mond) / abs(gap_lcdm) if gap_lcdm != 0 else 0
        v2_closes    = (gap_lcdm - gap_v2)   / abs(gap_lcdm) if gap_lcdm != 0 else 0
        v3_closes    = (gap_lcdm - gap_v3)   / abs(gap_lcdm) if gap_lcdm != 0 else 0
        v4_closes    = (gap_lcdm - gap_v4)   / abs(gap_lcdm) if gap_lcdm != 0 else 0
        v5_closes    = (gap_lcdm - gap_v5)   / abs(gap_lcdm) if gap_lcdm != 0 else 0
        v6_closes    = (gap_lcdm - gap_v6)   / abs(gap_lcdm) if gap_lcdm != 0 else 0
        v7_closes    = (gap_lcdm - gap_v7)   / abs(gap_lcdm) if gap_lcdm != 0 else 0

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
            "log_phi_v3":  log_phi_v3,
            "log_phi_v4":  log_phi_v4,
            "log_phi_v5":  log_phi_v5,
            "log_phi_v6":  log_phi_v6,
            "log_phi_v7":  log_phi_v7,
            "gap_lcdm_dex": gap_lcdm,
            "gap_bpr_dex":  gap_bpr,
            "gap_mond_dex": gap_mond,
            "gap_v2_dex":   gap_v2,
            "gap_v3_dex":   gap_v3,
            "gap_v4_dex":   gap_v4,
            "gap_v5_dex":   gap_v5,
            "gap_v6_dex":   gap_v6,
            "gap_v7_dex":   gap_v7,
            "bpr_fraction_closed":  bpr_closes,
            "mond_fraction_closed": mond_closes,
            "v2_fraction_closed":   v2_closes,
            "v3_fraction_closed":   v3_closes,
            "v4_fraction_closed":   v4_closes,
            "v5_fraction_closed":   v5_closes,
            "v6_fraction_closed":   v6_closes,
            "v7_fraction_closed":   v7_closes,
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
        "k_star_Mpc":   bpr_v3.k_star,
        "m_star_v4_z9":  bpr_v4.m_star(9.0),
        "m_star_v4_z10": bpr_v4.m_star(10.0),
        "m_star_v4_z12": bpr_v4.m_star(12.0),
        "W_c_v5":        bpr_v5._W_c,
        "m_imp_v5_z9":   bpr_v5.m_imp(9.0),
        "m_imp_v5_z10":  bpr_v5.m_imp(10.0),
        "m_imp_v5_z12":  bpr_v5.m_imp(12.0),
        "m_imp_v6_z9":   bpr_v6.m_imp(9.0),
        "m_imp_v6_z10":  bpr_v6.m_imp(10.0),
        "m_imp_v6_z12":  bpr_v6.m_imp(12.0),
        # V7 f_star retention: η(μ) at representative halo masses
        "fstar_eta_v7_z9":  bpr_v7._fstar_retention(2.5e11, 9.0),
        "fstar_eta_v7_z10": bpr_v7._fstar_retention(2.5e11, 10.0),
        "fstar_eta_v7_z12": bpr_v7._fstar_retention(1.0e11, 12.0),
        "fstar_eta_v7_z16": bpr_v7._fstar_retention(5.0e10, 16.0),
    }

    return results
