"""What BPR-6D predicts: the Green-Schwarz QCD axion and the family number.

See doc/derivations/predictions_2026-09-26.md. Standard inputs:
- QCD axion mass m_a = 5.70 micro-eV (1e12 GeV / f_a) (Grilli di Cortona et al., JHEP 01 (2016) 034);
- misalignment abundance Omega_a h^2 ~ 0.12 theta_i^2 (f_a / 9e11 GeV)^(7/6) (standard post-inflationary-free
  estimate, O(1) uncertain);
- compactification band 1/r = 2 g4 M_Pl / m for g4 in [0.02, 0.4] (flux_compactification section 4).
"""

import numpy as np

MODEL_ID = "bpr6d-predictions-v1"
REDUCED_PLANCK_GEV = 2.435e18

LIMITATIONS = [
    "The axion decay constant is estimated only to order of magnitude from the compactification band; the 2-form's 6D coupling is not fixed.",
    "The Green-Schwarz axion is generic to 6D Green-Schwarz and string models, not unique to BPR-6D.",
    "Axion mass and abundance use standard QCD and misalignment formulas with O(1) uncertainties.",
    "n_gen in 3Z holds for the minimal non-supersymmetric completion only (rounds 4 and 6).",
]


def stueckelberg_axions(k_b, k_a, f_b, f_a):
    """Two axions b, a with Stueckelberg couplings to A_F: (d b + k_b A)^2 f_b^2 and (d a + k_a A)^2 f_a^2.

    In canonical fields (theta = f b), A eats the direction proportional to (k_b f_b, k_a f_a); the physical axion
    is the orthogonal unit vector. Only b couples to Spin(10) (hence QCD) instantons, through Y_g = 3 lambda_V + ...
    Returns the physical axion's component along b.
    """
    eaten = np.array([k_b * f_b, k_a * f_a], dtype=float)
    eaten /= np.linalg.norm(eaten)
    physical = np.array([-eaten[1], eaten[0]])
    return {"eaten": eaten.tolist(), "physical": physical.tolist(), "qcd_component": float(physical[0])}


def gs_stueckelberg_charges(flux=1):
    """Shift charges from the round-4 couplings: Y_e = 6 x^2 (b via the Chern-Simons term) and the 9 x^2 term of
    Y_g (a via the BF term); both are proportional to the flux (factor 2 from the cross term)."""
    return {"k_b": 2 * 6 * flux, "k_a": 2 * 9 * flux, "qcd_coefficient_of_b": 3}


def axion_mass_ev(fa_gev):
    return 5.70e-6 * (1e12 / fa_gev)


def misalignment_theta_for_dm(fa_gev, omega_target=0.12):
    """Initial angle giving the observed dark matter: Omega h^2 = 0.12 theta^2 (f_a / 9e11)^(7/6)."""
    return float(np.sqrt(omega_target / (0.12 * (fa_gev / 9e11) ** (7 / 6))))


def compactification_band(flux=1, g_range=(0.02, 0.4), parent_charge=3):
    """1/r = 2 g4 M_Pl / m_parent, with m_parent = parent_charge * flux in the parent-charge-1 units of the flux note."""
    m_parent = parent_charge * flux
    return [2 * g * REDUCED_PLANCK_GEV / m_parent for g in g_range]


def axion_band(loop_factor_range=(1 / (8 * np.pi ** 2), 1.0)):
    """f_a between (1/r)/(8 pi^2) and 1/r across the compactification band (order-of-magnitude)."""
    lo_r, hi_r = compactification_band()
    fa = (lo_r * loop_factor_range[0], hi_r * loop_factor_range[1])
    return {"fa_gev": fa, "ma_ev": (axion_mass_ev(fa[1]), axion_mass_ev(fa[0])),
            "theta_i_for_dm": (misalignment_theta_for_dm(fa[1]), misalignment_theta_for_dm(fa[0]))}


def family_number_statements():
    """n_gen = q |m| with 3 | q (minimal completion): allowed 3, 6, 9, ...; a fourth (or fifth) chiral family is excluded."""
    allowed = [3 * k for k in range(1, 5)]
    return {"allowed": allowed, "excludes": [n for n in range(1, 13) if n not in allowed],
            "observed": 3, "consistent": 3 in allowed}


def demonstration_report():
    charges = gs_stueckelberg_charges()
    ratios = [0.01, 0.1, 1.0, 10.0, 100.0]
    comps = [stueckelberg_axions(charges["k_b"], charges["k_a"], 1.0, r)["qcd_component"] for r in ratios]
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "status": "qcd_axion_near_compactification_scale",
        "empirical_validation": False,
        "stueckelberg_charges": charges,
        "physical_axion_qcd_component_vs_fa_over_fb": dict(zip([str(r) for r in ratios], comps)),
        "compactification_band_gev": compactification_band(),
        "axion_band": axion_band(),
        "family_number": family_number_statements(),
        "limitations": list(LIMITATIONS),
    }
