"""
Experimental Data Registry
===========================

Structured database of measured physical quantities with uncertainties
and citations.  Used by the benchmark runner to compare BPR predictions
against published results from PDG, Planck, CODATA, and dedicated
experiments.

Every entry carries:
    value       – central measured value (SI or conventional units)
    uncertainty – 1σ symmetric uncertainty (0.0 when exact or for bounds)
    unit        – human-readable unit string
    source      – short citation
    year        – publication year
    is_upper_bound / is_lower_bound – for experimental limits

Usage
-----
>>> from bpr.experimental_data import MEASUREMENTS, get_measurement
>>> m = get_measurement("P5.7_theta13_deg")
>>> print(f"{m.value} ± {m.uncertainty} {m.unit}  [{m.source}]")
8.54 ± 0.15 deg  [PDG 2024]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import math


@dataclass(frozen=True)
class Measurement:
    """A single experimental measurement or bound."""
    prediction_id: str          # matches key in predictions dict
    name: str                   # human-readable description
    value: float                # central value (or bound value)
    uncertainty: float          # 1σ symmetric; 0 for bounds / exact
    unit: str
    source: str                 # short citation
    year: int
    is_upper_bound: bool = False
    is_lower_bound: bool = False
    is_exact: bool = False      # fundamental definition (e.g. dimensions = 3)
    bpr_status: str = "DERIVED" # DERIVED | FRAMEWORK | SUSPICIOUS | CONSISTENT | CONJECTURAL

    @property
    def has_uncertainty(self) -> bool:
        return self.uncertainty > 0 and not self.is_upper_bound and not self.is_lower_bound

    def sigma_deviation(self, predicted: float) -> Optional[float]:
        """Number of σ the prediction deviates from measurement.
        Returns None if no uncertainty or if this is a bound."""
        if self.is_upper_bound or self.is_lower_bound:
            return None
        if self.uncertainty <= 0:
            return None
        return abs(predicted - self.value) / self.uncertainty

    def satisfies_bound(self, predicted: float) -> Optional[bool]:
        """For bounds, check if predicted value satisfies the constraint."""
        if self.is_upper_bound:
            return predicted <= self.value
        if self.is_lower_bound:
            return predicted >= self.value
        return None

    def relative_deviation(self, predicted: float) -> Optional[float]:
        """Fractional deviation |pred - obs| / |obs|.  None if obs == 0."""
        if self.value == 0:
            return None
        return abs(predicted - self.value) / abs(self.value)


# ═══════════════════════════════════════════════════════════════════════
#  MEASUREMENT REGISTRY
# ═══════════════════════════════════════════════════════════════════════
#  Sources:
#    PDG 2024  = Particle Data Group, Phys. Rev. D 110, 030001 (2024)
#    Planck18  = Planck Collaboration, A&A 641, A6 (2020)
#    CODATA18  = CODATA 2018 recommended values
#    Super-K   = Super-Kamiokande Collaboration
#    LHAASO    = LHAASO Collaboration, PRL 128, 051102 (2022)
#    Fermi-LAT = Fermi-LAT Collaboration
#    NICER     = Riley+ (2021), ApJL 918, L27
#    SPARC     = Lelli+ (2017), ApJ 836, 152
# ═══════════════════════════════════════════════════════════════════════

_REGISTRY: list[Measurement] = [
    # ─── Boundary-Mediated Neutrino Dynamics──────────────────────────────────
    Measurement(
        prediction_id="P5.5_theta12_deg",
        name="Neutrino mixing angle θ₁₂",
        value=33.41, uncertainty=0.8, unit="deg",
        source="PDG 2024", year=2024,
        bpr_status="DERIVED",  # sin²θ₁₂ = 1/3 - 1/(3.5×ln(p))
    ),
    Measurement(
        prediction_id="P5.6_theta23_deg",
        name="Neutrino mixing angle θ₂₃",
        value=49.0, uncertainty=1.3, unit="deg",
        source="PDG 2024 (NO)", year=2024,
        bpr_status="DERIVED",  # 1/2 + Δm²₂₁/Δm²₃₁×1.35 + charged-lepton
    ),
    Measurement(
        prediction_id="P5.7_theta13_deg",
        name="Neutrino mixing angle θ₁₃",
        value=8.54, uncertainty=0.15, unit="deg",
        source="PDG 2024", year=2024,
        bpr_status="DERIVED",
    ),
    Measurement(
        prediction_id="P5.2_sum_masses_eV",
        name="Sum of neutrino masses Σm_ν",
        value=0.12, uncertainty=0.0, unit="eV",
        source="Planck18 + BAO", year=2020,
        is_upper_bound=True,
        bpr_status="DERIVED",
    ),
    Measurement(
        prediction_id="P5.8_delta_m21_sq_eV2",
        name="Solar mass splitting Δm²₂₁",
        value=7.53e-5, uncertainty=0.18e-5, unit="eV²",
        source="PDG 2024", year=2024,
        bpr_status="DERIVED",
    ),
    Measurement(
        prediction_id="P5.9_delta_m32_sq_eV2",
        name="Atmospheric mass splitting |Δm²₃₂|",
        value=2.453e-3, uncertainty=0.033e-3, unit="eV²",
        source="PDG 2024 (NO)", year=2024,
        bpr_status="DERIVED",
    ),
    Measurement(
        prediction_id="P5.10_number_of_generations",
        name="Number of neutrino generations",
        value=3, uncertainty=0.0, unit="",
        source="LEP (Z width)", year=2006,
        is_exact=True,
        bpr_status="DERIVED",
    ),

    # ─── Vacuum Impedance Mismatch──────────────────────────
    Measurement(
        prediction_id="P2.2_MOND_a0",
        name="MOND acceleration scale a₀",
        value=1.2e-10, uncertainty=0.2e-10, unit="m/s²",
        source="SPARC (Lelli+ 2017)", year=2017,
        bpr_status="DERIVED",  # a₀ = c H₀/(2π) × (1 + z/(4 ln p))
    ),
    Measurement(
        prediction_id="P2.7_DM_sigma_over_m_cm2_g",
        name="DM self-interaction σ/m",
        value=0.6, uncertainty=0.0, unit="cm²/g",
        source="Bullet Cluster (Markevitch+ 2004)", year=2004,
        is_upper_bound=True,
        bpr_status="DERIVED",
    ),
    Measurement(
        prediction_id="P2.15_proton_lifetime_years",
        name="Proton lifetime τ_p (p → e⁺π⁰)",
        value=2.4e34, uncertainty=0.0, unit="yr",
        source="Super-K (2020)", year=2020,
        is_lower_bound=True,
        bpr_status="DERIVED",
    ),

    # ─── BPR Cosmology & Early Universe────────────────────────────────────────
    Measurement(
        prediction_id="P11.2_spectral_index",
        name="Scalar spectral index n_s",
        value=0.9649, uncertainty=0.0042, unit="",
        source="Planck18", year=2020,
        bpr_status="DERIVED",  # n_s = 1 - 2/N, N = p^(1/3)×(1+1/d) from boundary
    ),
    Measurement(
        prediction_id="P11.3_tensor_to_scalar_r",
        name="Tensor-to-scalar ratio r",
        value=0.044, uncertainty=0.0, unit="",
        source="BICEP/Keck 2021", year=2021,
        is_upper_bound=True,
        bpr_status="DERIVED",  # r = 12/N², N from p^(1/3)
    ),
    Measurement(
        prediction_id="P11.7_baryon_asymmetry_eta",
        name="Baryon-to-photon ratio η",
        value=6.12e-10, uncertainty=0.04e-10, unit="",
        source="Planck18 (BBN)", year=2020,
        bpr_status="DERIVED",  # J from CKM; kappa_sph from boundary winding
    ),
    Measurement(
        prediction_id="P11.15_DM_relic_Omega_h2",
        name="Dark matter relic density Ω_DM h²",
        value=0.120, uncertainty=0.001, unit="",
        source="Planck18", year=2020,
        bpr_status="DERIVED",  # Thermal freeze-out from BPR coupling/mass (no longer hardcoded)
    ),
    Measurement(
        prediction_id="P11.14_delta_Neff",
        name="Extra effective neutrino species ΔN_eff",
        value=0.2, uncertainty=0.0, unit="",
        source="Planck18", year=2020,
        is_upper_bound=True,
        bpr_status="DERIVED",
    ),

    # ─── Universal Phase Transition Taxonomy────────────────────────────────
    Measurement(
        prediction_id="P4.7_Tc_niobium_K",
        name="Superconducting Tc (Niobium)",
        value=9.25, uncertainty=0.02, unit="K",
        source="ASM Handbook", year=1990,
        bpr_status="DERIVED",  # N0V from BPR + z² + Eliashberg vertex correction
    ),
    Measurement(
        prediction_id="P4.9_Tc_MgB2_K",
        name="Superconducting Tc (MgB₂)",
        value=39.0, uncertainty=0.5, unit="K",
        source="Nagamatsu+ (2001) Nature 410, 63", year=2001,
        bpr_status="DERIVED",  # N0V from E_F_eff = 2×(E_σ+E_π)
    ),

    # ─── QCD & Flavor Physics────────────────────────────────────
    Measurement(
        prediction_id="P12.2_m_u_MeV",
        name="Up quark mass m_u",
        value=2.16, uncertainty=0.49, unit="MeV",
        source="PDG 2024 (MS-bar, 2 GeV)", year=2024,
        bpr_status="DERIVED",  # From S² boundary modes l=(283,24,1), anchored to m_t
    ),
    Measurement(
        prediction_id="P12.3_m_d_MeV",
        name="Down quark mass m_d",
        value=4.67, uncertainty=0.48, unit="MeV",
        source="PDG 2024 (MS-bar, 2 GeV)", year=2024,
        bpr_status="DERIVED",  # b = -W_c×(1−1/(4z)) from boundary
    ),
    Measurement(
        prediction_id="P12.4_m_s_MeV",
        name="Strange quark mass m_s",
        value=93.4, uncertainty=8.6, unit="MeV",
        source="PDG 2024 (MS-bar, 2 GeV)", year=2024,
        bpr_status="DERIVED",  # From spectrum with derived m_b
    ),
    Measurement(
        prediction_id="P12.5_m_c_MeV",
        name="Charm quark mass m_c",
        value=1270.0, uncertainty=20.0, unit="MeV",
        source="PDG 2024 (MS-bar)", year=2024,
        bpr_status="DERIVED",  # From S² boundary modes l=(283,24,1), anchored to m_t
    ),
    Measurement(
        prediction_id="P12.6_m_b_MeV",
        name="Bottom quark mass m_b",
        value=4180.0, uncertainty=30.0, unit="MeV",
        source="PDG 2024 (MS-bar)", year=2024,
        bpr_status="DERIVED",  # m_b = m_t×(E_b/c_t)×(2+1/(3 ln p))
    ),
    Measurement(
        prediction_id="P12.7_m_t_MeV",
        name="Top quark mass m_t",
        value=172690.0, uncertainty=300.0, unit="MeV",
        source="PDG 2024 (direct)", year=2024,
        bpr_status="DERIVED",  # m_t = v_EW/√2 from boundary (v_EW derived)
    ),
    Measurement(
        prediction_id="P12.8_CKM_theta12_deg",
        name="CKM angle θ₁₂ (Cabibbo)",
        value=12.96, uncertainty=0.03, unit="deg",
        source="PDG 2024", year=2024,
        bpr_status="DERIVED",  # Gatto–Sartori–Tonin: sin(θ_C) = √(m_d/m_s)
    ),
    Measurement(
        prediction_id="P12.9_CKM_theta23_deg",
        name="CKM angle θ₂₃",
        value=2.38, uncertainty=0.06, unit="deg",
        source="PDG 2024", year=2024,
        bpr_status="DERIVED",  # |V_cb| = √(m_s/m_b) / √(ln(p) + z/3)
    ),
    Measurement(
        prediction_id="P12.10_CKM_theta13_deg",
        name="CKM angle θ₁₃",
        value=0.209, uncertainty=0.005, unit="deg",
        source="PDG 2024", year=2024,
        bpr_status="DERIVED",  # |V_ub| = √(m_u/m_t)
    ),
    Measurement(
        prediction_id="P12.11_CKM_Jarlskog",
        name="Jarlskog invariant J",
        value=3.08e-5, uncertainty=0.15e-5, unit="",
        source="PDG 2024", year=2024,
        bpr_status="DERIVED",  # From derived θ₂₃, θ₁₃, δ = π/2 − 1/√(z+1)
    ),
    Measurement(
        prediction_id="P12.13_proton_mass_GeV",
        name="Proton mass",
        value=0.938272, uncertainty=0.000001, unit="GeV",
        source="CODATA 2018", year=2018,
        bpr_status="CONSISTENT",  # Standard QCD: m_p ≈ 3Λ_QCD, not a BPR prediction
    ),
    Measurement(
        prediction_id="P12.14_pion_mass_MeV",
        name="Pion mass (π⁰)",
        value=134.977, uncertainty=0.005, unit="MeV",
        source="PDG 2024", year=2024,
        bpr_status="DERIVED",  # Condensate √(2/3) + NLO δ_π=6.2%
    ),

    # ─── Gauge Unification & Hierarchy──────────────────────────────
    Measurement(
        prediction_id="P17.8_proton_lifetime_GUT_years",
        name="Proton lifetime (GUT prediction)",
        value=2.4e34, uncertainty=0.0, unit="yr",
        source="Super-K (2020)", year=2020,
        is_lower_bound=True,
        bpr_status="DERIVED",
    ),
    Measurement(
        prediction_id="P17.13_v_EW_GeV",
        name="Electroweak scale (Higgs VEV)",
        value=246.0, uncertainty=0.5, unit="GeV",
        source="PDG 2024 (from GF)", year=2024,
        bpr_status="DERIVED",  # v = Λ_QCD × p^(1/3) × (ln(p) + z − 2)
    ),

    # ─── Charged Lepton Masses───────────────────────────────
    Measurement(
        prediction_id="P18.1_m_electron_MeV",
        name="Electron mass",
        value=0.51099895, uncertainty=0.00000002, unit="MeV",
        source="CODATA 2018", year=2018,
        bpr_status="DERIVED",  # From S² boundary modes l=(59,14,1), anchored to m_τ; 0.11% off
    ),
    Measurement(
        prediction_id="P18.2_m_muon_MeV",
        name="Muon mass",
        value=105.6583755, uncertainty=0.0000023, unit="MeV",
        source="CODATA 2018", year=2018,
        bpr_status="DERIVED",  # From S² boundary modes l=(59,14,1), anchored to m_τ; 5.3% off
    ),
    Measurement(
        prediction_id="P18.3_m_tau_MeV",
        name="Tau mass",
        value=1776.86, uncertainty=0.12, unit="MeV",
        source="PDG 2024", year=2024,
        bpr_status="DERIVED",  # m_τ = v_EW × α (EM coupling from boundary)
    ),
    Measurement(
        prediction_id="P18.4_koide_parameter",
        name="Koide parameter Q",
        value=2.0/3.0, uncertainty=0.0,
        unit="", source="Koide (1981), exact relation",
        year=1981, is_exact=True,
        bpr_status="DERIVED",  # Emerges from l² spectrum (Q ≈ 0.672, 0.75% from exact 2/3)
    ),

    # ─── Nuclear Physics from Boundary Shell─────────────────────────────────
    Measurement(
        prediction_id="P19.7_B_per_A_Fe56_MeV",
        name="Binding energy per nucleon ⁵⁶Fe",
        value=8.790, uncertainty=0.001, unit="MeV",
        source="AME2020", year=2020,
        bpr_status="DERIVED",  # BW + BPR shell; a_S from n_sat
    ),
    Measurement(
        prediction_id="P19.8_B_per_A_He4_MeV",
        name="Binding energy per nucleon ⁴He",
        value=7.074, uncertainty=0.001, unit="MeV",
        source="AME2020", year=2020,
        bpr_status="DERIVED",  # alpha-clustering = a_S×4^(2/3)/24
    ),
    Measurement(
        prediction_id="P19.9_saturation_density_fm3",
        name="Nuclear saturation density",
        value=0.16, uncertainty=0.01, unit="fm⁻³",
        source="Nuclear Physics A", year=2010,
        bpr_status="DERIVED",  # r_ch=1.25 fm from boundary mode packing
    ),
    Measurement(
        prediction_id="P19.10_NS_max_mass_solar",
        name="Neutron star maximum mass",
        value=2.08, uncertainty=0.07, unit="M_☉",
        source="Fonseca+ (2021) PSR J0740+6620", year=2021,
        bpr_status="DERIVED",  # TOV + boundary pressure κ/ξ³
    ),
    Measurement(
        prediction_id="P19.11_NS_radius_km",
        name="Neutron star radius (1.4 M_☉)",
        value=12.35, uncertainty=0.75, unit="km",
        source="NICER (Riley+ 2021)", year=2021,
        bpr_status="DERIVED",  # R ∝ n_sat^(-1/3) from boundary-derived n_sat
    ),

    # ─── Quantum Gravity Phenomenology─────────────────────────────────
    Measurement(
        prediction_id="P20.7_LI_delta_c_over_c",
        name="Lorentz invariance violation |δc/c|",
        value=6.0e-21, uncertainty=0.0, unit="",
        source="Fermi-LAT GRB 090510", year=2009,
        is_upper_bound=True,
        bpr_status="DERIVED",
    ),
    Measurement(
        prediction_id="P20.4_GUP_beta",
        name="GUP parameter β",
        value=4.0e4, uncertainty=0.0, unit="",
        source="Various (see PDG QG review)", year=2024,
        is_upper_bound=True,
        bpr_status="DERIVED",
    ),
    Measurement(
        prediction_id="P20.1_LIV_xi1",
        name="Linear LIV parameter ξ₁",
        value=0.0, uncertainty=0.0, unit="",
        source="LHAASO (2022)", year=2022,
        is_exact=True,
        bpr_status="DERIVED",
    ),

    # ─── Emergent Spacetime & Holography─────────────────────────────
    Measurement(
        prediction_id="P13.3_spatial_dimensions",
        name="Number of spatial dimensions",
        value=3, uncertainty=0.0, unit="",
        source="Observation", year=2000,
        is_exact=True,
        bpr_status="DERIVED",
    ),
    Measurement(
        prediction_id="P13.4_time_dimensions",
        name="Number of time dimensions",
        value=1, uncertainty=0.0, unit="",
        source="Observation", year=2000,
        is_exact=True,
        bpr_status="DERIVED",
    ),
    Measurement(
        prediction_id="P13.5_total_dimensions",
        name="Total spacetime dimensions",
        value=4, uncertainty=0.0, unit="",
        source="Observation", year=2000,
        is_exact=True,
        bpr_status="DERIVED",
    ),
    Measurement(
        prediction_id="P13.8_planck_length_emergent",
        name="Planck length",
        value=1.616255e-35, uncertainty=0.000018e-35, unit="m",
        source="CODATA 2018", year=2018,
        bpr_status="OPEN",
    ),

    # ─── Topological Condensed Matter──────────────────────────────
    Measurement(
        prediction_id="P14.2_hall_resistance_nu1_Ohm",
        name="von Klitzing constant R_K",
        value=25812.80745, uncertainty=0.0, unit="Ω",
        source="CODATA 2018 (exact since 2019 SI)", year=2019,
        is_exact=True,
        bpr_status="CONSISTENT",
    ),
    Measurement(
        prediction_id="P14.12_conductance_quantum_S",
        name="Conductance quantum G₀ = 2e²/h",
        value=7.748091729e-5, uncertainty=0.0, unit="S",
        source="CODATA 2018 (exact)", year=2019,
        is_exact=True,
        bpr_status="CONSISTENT",
    ),

    # ─── Quantum Foundations──────────────────────────────
    Measurement(
        prediction_id="P16.7_bell_bpr_bound",
        name="Tsirelson bound 2√2",
        value=2.0 * math.sqrt(2.0), uncertainty=0.0, unit="",
        source="Tsirelson (1980), exact", year=1980,
        is_exact=True,
        bpr_status="DERIVED",
    ),

    # ─── Gravitational Wave Phenomenology──────────────────────────────────────────────
    Measurement(
        prediction_id="P7.1_vGW_equals_c",
        name="GW speed deviation |v_GW/c - 1|",
        value=7.0e-16, uncertainty=0.0, unit="",
        source="GW170817 (LIGO/Virgo)", year=2017,
        is_upper_bound=True,
        bpr_status="CONSISTENT",
    ),
]


# ═══════════════════════════════════════════════════════════════════════
#  Lookup helpers
# ═══════════════════════════════════════════════════════════════════════

MEASUREMENTS: Dict[str, Measurement] = {m.prediction_id: m for m in _REGISTRY}


def get_measurement(prediction_id: str) -> Optional[Measurement]:
    """Look up an experimental measurement by prediction ID."""
    return MEASUREMENTS.get(prediction_id)


def list_measurement_ids() -> list[str]:
    """All prediction IDs that have experimental data."""
    return sorted(MEASUREMENTS.keys())


def summary_counts() -> Dict[str, int]:
    """Count measurements by BPR status category."""
    counts: Dict[str, int] = {}
    for m in _REGISTRY:
        counts[m.bpr_status] = counts.get(m.bpr_status, 0) + 1
    return counts
