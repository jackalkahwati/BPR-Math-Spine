"""What BPR-6D predicts: the Green-Schwarz axion, its Peccei-Quinn structure, and the family number.

See doc/derivations/predictions_2026-09-26.md. Standard inputs:
- QCD axion mass m_a = 5.70 micro-eV (1e12 GeV / f) and topological susceptibility chi^(1/4) = 75.5 MeV
  (Grilli di Cortona et al., JHEP 01 (2016) 034);
- pre-inflationary misalignment Omega_a h^2 ~ 0.12 theta_i^2 (f / 9e11 GeV)^(7/6), with exponent 3/2 above
  f ~ 1.5e17 GeV where oscillations start after the QCD crossover (O(1) uncertain);
- Planck isocurvature bound beta_iso < 0.038 and P_R = 2.1e-9;
- black-hole superradiance: 6e-13 < m_a < 2e-11 eV disfavoured (Arvanitaki-Baryakhtar-Huang 2015);
- compactification band 1/r = 2 g4 M_Pl / m for g4 in [0.02, 0.4] (flux_compactification section 4).
"""

from fractions import Fraction

import numpy as np
import sympy as sp

try:
    from .green_schwarz_quantization import hyperbolic_solution, minimal_fields
except ImportError:  # loaded as a top-level module by the demo script
    from green_schwarz_quantization import hyperbolic_solution, minimal_fields

MODEL_ID = "bpr6d-predictions-v2"
REDUCED_PLANCK_GEV = 2.435e18
HIGGS_VEV_GEV = 246.22
CHI_QCD_QUARTER_GEV = 0.0755
EV_TO_HZ = 2.418e14

LIMITATIONS = [
    "The Higgs sector is supplied: with it, U(1)_F acts as a Peccei-Quinn symmetry, and a viable (invisible) QCD axion "
    "needs a supplied F-charged Standard-Model-singlet order parameter; without one the axion is an excluded PQWW axion.",
    "The decay constants scale as kappa^(+-1/2) with the unfixed 2-form coupling kappa; the band assumes kappa ~ O(1) "
    "and no lighter F-charged vev.",
    "Axion quality is unresolved: gauge-invariant non-QCD potentials for the theta-bar direction (wrapped Euclidean "
    "strings dressed by charged fields, small Spin(10) instantons, gravitational instantons) are not computed.",
    "The Green-Schwarz axion is generic to 6D Green-Schwarz and string models, not unique to BPR-6D.",
    "Axion mass, abundance and cosmological bounds use standard formulas with O(1) uncertainties.",
    "n_gen in 3Z holds for the minimal non-supersymmetric completion in class C only (rounds 4 and 6).",
]

# Standard-Model content of one 16: (name, colour dimension, SU(3) index T, hypercharge).
SM16 = [("Q", 3, Fraction(1, 2), Fraction(1, 6)), ("Q", 3, Fraction(1, 2), Fraction(1, 6)),
        ("u^c", 3, Fraction(1, 2), Fraction(-2, 3)), ("d^c", 3, Fraction(1, 2), Fraction(1, 3)),
        ("L", 1, 0, Fraction(-1, 2)), ("L", 1, 0, Fraction(-1, 2)), ("e^c", 1, 0, Fraction(1)),
        ("nu^c", 1, 0, Fraction(0))]


def gs_stueckelberg_charges(flux=1, parent_charge=3):
    """Shift charges read off the round-4 factorization I8 = Y_e Y_g.

    b = int_{S^2} B shifts through the Chern-Simons term of Y_e (cross term 2 x_bg x), a = dual of B_mu nu through
    the BF term from the x^2 part of Y_g; only b multiplies the Spin(10) density, with the lambda_V coefficient.
    """
    sol = hyperbolic_solution(minimal_fields(parent_charge))
    lamV, x2 = sp.symbols("lambda_V x2")
    Y_e, Y_g = sp.sympify(sol["Y_e"]), sp.sympify(sol["Y_g"])
    return {"k_b": int(2 * Y_e.coeff(x2) * flux), "k_a": int(2 * Y_g.coeff(x2) * flux),
            "qcd_coefficient_of_b": int(Y_g.coeff(lamV))}


def f_su3_anomaly(families=3, f_charge=3):
    """U(1)_F-SU(3)^2 anomaly sum_i q_i T(r_i) of the 4D chiral families (Weyl fermions of one 16 each)."""
    return families * f_charge * sum(T for _, _, T, _ in SM16)


def theta_bar_structure(families=3, f_charge=3, higgs_f_charge=-6):
    """Gauge-invariant theta-bar = theta_0 + c_b b + c_u arg H_u + c_d arg H_d (+ arg det Y); a does not enter.

    c_b follows from the anomaly: b must cancel the U(1)_F shift 2 A lambda of theta, so c_b k_b = 2 A. The Higgs
    phases enter through arg det M_q, one per quark flavour. Returns coefficients, F-charges and hypercharges.
    """
    k = gs_stueckelberg_charges()
    A = f_su3_anomaly(families, f_charge)
    c_b = Fraction(2 * A, k["k_b"])
    coeffs = {"b": c_b, "a": Fraction(0), "arg_Hu": Fraction(families), "arg_Hd": Fraction(families)}
    f_charges = {"b": k["k_b"], "a": k["k_a"], "arg_Hu": higgs_f_charge, "arg_Hd": higgs_f_charge}
    hypercharges = {"b": 0, "a": 0, "arg_Hu": Fraction(1, 2), "arg_Hd": Fraction(-1, 2)}
    return {"coefficients": coeffs, "f_charges": f_charges, "hypercharges": hypercharges,
            "anomaly": A, "gs_coefficient_matches_anomaly": c_b == k["qcd_coefficient_of_b"],
            "f_invariant": sum(coeffs[x] * f_charges[x] for x in coeffs) == 0,
            "y_invariant": sum(coeffs[x] * hypercharges[x] for x in coeffs) == 0}


def gauge_invariant_phases(with_singlet_charge=None):
    """Nullspace of the (U(1)_F, U(1)_Y) charge matrix on the phases (b, a, arg H_u, arg H_d[, arg S]).

    Without S the physical phases are spanned by theta-bar = (3, 0, 3, 3) and zeta = 3b - 2a: two directions,
    so zeta is a massless Goldstone with no QCD coupling unless something else lifts it.
    """
    th = theta_bar_structure()
    names = ["b", "a", "arg_Hu", "arg_Hd"]
    F = [th["f_charges"][n] for n in names]
    Y = [th["hypercharges"][n] for n in names]
    if with_singlet_charge is not None:
        names.append("arg_S")
        F.append(with_singlet_charge)
        Y.append(0)
    null = sp.Matrix([F, Y]).nullspace()
    return {"fields": names, "basis": [[str(x) for x in v] for v in null], "dimension": len(null)}


def qcd_decay_constant(entries):
    """1/f^2 = sum_i c_i^2 / f_i^2 for theta-bar = sum_i c_i phi_i / f_i with canonical phases phi_i.

    theta-bar is gauge invariant, so its gradient is already orthogonal to every eaten direction.
    """
    return float(sum(c ** 2 / f ** 2 for c, f in entries) ** -0.5)


def pqww_decay_constant(f_b=1e16, tan_beta=1.0, v=HIGGS_VEV_GEV):
    """No F-charged vev beyond b and the doublets: f = (9/f_b^2 + 9/v_u^2 + 9/v_d^2)^(-1/2) ~ v sin2beta / 6."""
    beta = np.arctan(tan_beta)
    return qcd_decay_constant([(3, f_b), (3, v * np.sin(beta)), (3, v * np.cos(beta))])


def dfsz_like_decay_constant(f_b, f_s, c_s):
    """Doublet phases locked to a heavy F-charged singlet S: f = (9/f_b^2 + c_S^2/f_S^2)^(-1/2)."""
    return qcd_decay_constant([(3, f_b), (c_s, f_s)])


def axion_mass_ev(f_gev):
    return 5.70e-6 * (1e12 / f_gev)


def misalignment_theta_for_dm(f_gev, omega_target=0.12, f_crossover=1.5e17):
    """Pre-inflationary initial angle for the observed dark matter.

    Omega h^2 = 0.12 theta^2 (f / 9e11)^(7/6) below f_crossover; above it oscillations start after the QCD
    crossover and Omega grows as f^(3/2), matched continuously.
    """
    if f_gev <= f_crossover:
        omega_at_theta1 = 0.12 * (f_gev / 9e11) ** (7 / 6)
    else:
        omega_at_theta1 = 0.12 * (f_crossover / 9e11) ** (7 / 6) * (f_gev / f_crossover) ** 1.5
    return float(np.sqrt(omega_target / omega_at_theta1))


def isocurvature_bounds(f_gev, beta_max=0.038, P_R=2.1e-9):
    """If the (pre-inflationary) axion is all the dark matter: H_I < pi f theta_i sqrt(P_S,max) and the implied r."""
    theta = misalignment_theta_for_dm(f_gev)
    P_S = beta_max / (1 - beta_max) * P_R
    H_max = np.pi * f_gev * theta * np.sqrt(P_S)
    r_max = 2 * H_max ** 2 / (np.pi ** 2 * REDUCED_PLANCK_GEV ** 2 * P_R)
    return {"H_inflation_max_gev": float(H_max), "tensor_to_scalar_max": float(r_max)}


def superradiance_excluded_f(m_range_ev=(6e-13, 2e-11)):
    """Decay constants disfavoured by black-hole superradiance (Arvanitaki-Baryakhtar-Huang 2015)."""
    return (5.70e-6 * 1e12 / m_range_ev[1], 5.70e-6 * 1e12 / m_range_ev[0])


def quality_required_action(M_gev=1e17, precision=1e-10):
    """Instanton action S with e^{-S} M^4 < precision * chi_QCD, for a potential along the theta-bar direction."""
    return float(np.log(M_gev ** 4 / (precision * CHI_QCD_QUARTER_GEV ** 4)))


def wrapped_string_action(rM_range=(1.0, 3.0)):
    """Euclidean string wrapped on S^2 with tension T ~ M^2: S = 4 pi (r M)^2 (order of magnitude)."""
    return [float(4 * np.pi * x ** 2) for x in rM_range]


def compactification_band(flux=1, g_range=(0.02, 0.4), parent_charge=3):
    """1/r = 2 g4 M_Pl / m_parent, with m_parent = parent_charge * flux in the parent-charge-1 units of the flux note."""
    m_parent = parent_charge * flux
    return [2 * g * REDUCED_PLANCK_GEV / m_parent for g in g_range]


def axion_band(loop_factor_range=(1 / (8 * np.pi ** 2), 1.0)):
    """Order-of-magnitude upper band for f: f_b/3 between (1/r)/(8 pi^2) and 1/r, if kappa ~ O(1).

    A lighter F-charged singlet vev lowers f (dfsz_like_decay_constant).
    """
    lo_r, hi_r = compactification_band()
    f = (lo_r * loop_factor_range[0], hi_r * loop_factor_range[1])
    m = (axion_mass_ev(f[1]), axion_mass_ev(f[0]))
    return {"f_gev": f, "ma_ev": m, "frequency_hz": (m[0] * EV_TO_HZ, m[1] * EV_TO_HZ),
            "theta_i_for_dm": (misalignment_theta_for_dm(f[1]), misalignment_theta_for_dm(f[0])),
            "isocurvature_if_dm": [isocurvature_bounds(x) for x in f],
            "superradiance_excluded_f": superradiance_excluded_f()}


def family_number_statements():
    """n_gen = q |m| with 3 | q (minimal completion in class C): allowed 3, 6, 9, ...; a fourth (or fifth) chiral
    family is excluded. Outside C, e.g. 16_+(1) + 16_-(2), the result does not hold."""
    allowed = [3 * k for k in range(1, 5)]
    return {"allowed": allowed, "excludes": [n for n in range(1, 13) if n not in allowed],
            "observed": 3, "consistent": 3 in allowed, "class": "C (no extra massless Spin(10) matter)"}


def demonstration_report():
    theta = theta_bar_structure()
    return {
        "schema_version": 2,
        "model_id": MODEL_ID,
        "status": "qcd_axion_conditional_on_supplied_higgs_sector",
        "empirical_validation": False,
        "stueckelberg_charges": gs_stueckelberg_charges(),
        "theta_bar": {"coefficients": {k: str(v) for k, v in theta["coefficients"].items()},
                      "anomaly": str(theta["anomaly"]),
                      "gs_coefficient_matches_anomaly": theta["gs_coefficient_matches_anomaly"],
                      "f_invariant": theta["f_invariant"], "y_invariant": theta["y_invariant"]},
        "gauge_invariant_phases": gauge_invariant_phases(),
        "pqww_decay_constant_gev": pqww_decay_constant(),
        "dfsz_like_example_gev": dfsz_like_decay_constant(1e16, 1e12, 6),
        "quality": {"required_action": quality_required_action(), "wrapped_string_estimate": wrapped_string_action()},
        "compactification_band_gev": compactification_band(),
        "axion_band": axion_band(),
        "family_number": family_number_statements(),
        "limitations": list(LIMITATIONS),
    }
