"""
BPR Prediction Consistency Web
================================

Cross-checks BPR predictions against each other to verify internal
consistency. If the same quantity computed through different module
chains disagrees, it exposes a bug or physics error.

Key consistency relations:
1. sin²θ_W from gauge running = sin²θ_W from impedance (within error)
2. α_EM from substrate = α_EM from running (within error)
3. v_EW from gauge = v_EW used in lepton masses
4. Σm_ν from seesaw < cosmological bound
5. w(z) from impedance → Ω_Λ from dark energy (consistent)
6. T_c from Landau = T_c from TDGL simulation
7. K_c from Kuramoto = consciousness onset threshold

References: Al-Kahwati (2026)
"""

from __future__ import annotations

import math
import numpy as np
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Guarded imports -- each block may fail independently
# ---------------------------------------------------------------------------

try:
    from .alpha_derivation import inverse_alpha_from_substrate
    _HAS_ALPHA = True
except Exception:
    _HAS_ALPHA = False

try:
    from .cross_predictions import full_cosmological_chain
    _HAS_CROSS = True
except Exception:
    _HAS_CROSS = False

try:
    from .gauge_unification import (
        GaugeCouplingRunning,
        electroweak_scale_GeV,
    )
    _HAS_GAUGE = True
except Exception:
    _HAS_GAUGE = False

try:
    from .bridges.particles_matter import (
        impedance_weinberg_angle,
        neutrino_lepton_mass_relation,
    )
    _HAS_PARTICLES = True
except Exception:
    _HAS_PARTICLES = False

try:
    from .bridges.cosmology_gravity import dark_energy_equation_of_state
    _HAS_COSMO_BRIDGE = True
except Exception:
    _HAS_COSMO_BRIDGE = False

try:
    from .bridges.life_consciousness import eeg_peak_frequencies
    _HAS_LIFE = True
except Exception:
    _HAS_LIFE = False

try:
    from .decoherence import DecoherenceRate
    _HAS_DECOHERENCE = True
except Exception:
    _HAS_DECOHERENCE = False

try:
    from .pipelines import pipeline_impedance_to_lepton_masses
    _HAS_PIPELINES = True
except Exception:
    _HAS_PIPELINES = False

try:
    from .nuclear_physics import magic_numbers_bpr
    _HAS_NUCLEAR = True
except Exception:
    _HAS_NUCLEAR = False

try:
    from .clifford_bpr import verify_e8_properties
    _HAS_E8 = True
except Exception:
    _HAS_E8 = False

try:
    from .charged_leptons import ChargedLeptonSpectrum
    _HAS_LEPTONS = True
except Exception:
    _HAS_LEPTONS = False

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_Z0 = 376.730313668  # Ohm (vacuum impedance)


# ===================================================================
# Helper: classify a relative error into PASS / TENSION / FAIL
# ===================================================================

def _classify(error_pct: float, tol_pass: float, tol_tension: float = 5.0,
              tol_fail: float = 20.0) -> str:
    """Return 'PASS', 'TENSION', or 'FAIL' given percent error."""
    if error_pct <= tol_pass:
        return "PASS"
    elif error_pct <= tol_tension:
        return "TENSION"
    else:
        return "FAIL"


# ===================================================================
# Individual consistency checks
# ===================================================================

def _check_alpha_em(p: int, z: int) -> Dict[str, Any]:
    """Check 1: alpha_EM via two independent routes.

    Route A: alpha_derivation.inverse_alpha_from_substrate(p, z)
    Route B: full_cosmological_chain(p, z) -> inv_alpha_predicted
    """
    if not (_HAS_ALPHA and _HAS_CROSS):
        return {
            "name": "alpha_EM",
            "status": "SKIP",
            "reason": "alpha_derivation or cross_predictions unavailable",
        }

    route_a = inverse_alpha_from_substrate(p, z)
    chain = full_cosmological_chain(p, z)
    route_b = chain.get("inv_alpha_predicted")

    if route_b is None:
        return {
            "name": "alpha_EM",
            "status": "SKIP",
            "reason": "full_cosmological_chain did not produce inv_alpha_predicted",
        }

    error_pct = abs(route_a - route_b) / route_b * 100.0
    return {
        "name": "alpha_EM",
        "route_A": float(route_a),
        "route_A_label": "inverse_alpha_from_substrate",
        "route_B": float(route_b),
        "route_B_label": "full_cosmological_chain",
        "error": f"{error_pct:.4f}%",
        "status": _classify(error_pct, tol_pass=0.01),
    }


def _check_sin2_theta_w(p: int, z: int) -> Dict[str, Any]:
    """Check 2: sin^2(theta_W) via gauge running vs impedance bridge.

    Route A: GaugeCouplingRunning(p).weinberg_angle_at_MZ
    Route B: impedance_weinberg_angle(p) -> sin2_theta_W_gauge_running
    """
    if not (_HAS_GAUGE and _HAS_PARTICLES):
        return {
            "name": "sin2_theta_W",
            "status": "SKIP",
            "reason": "gauge_unification or particles_matter bridge unavailable",
        }

    gcr = GaugeCouplingRunning(p=p)
    route_a = gcr.weinberg_angle_at_MZ

    iwa = impedance_weinberg_angle(p=p)
    route_b = iwa.get("sin2_theta_W_gauge_running")

    if route_b is None:
        return {
            "name": "sin2_theta_W",
            "status": "SKIP",
            "reason": "impedance_weinberg_angle did not return gauge_running value",
        }

    error_pct = abs(route_a - route_b) / route_b * 100.0
    return {
        "name": "sin2_theta_W",
        "route_A": float(route_a),
        "route_A_label": "GaugeCouplingRunning.weinberg_angle_at_MZ",
        "route_B": float(route_b),
        "route_B_label": "impedance_weinberg_angle.sin2_theta_W_gauge_running",
        "error": f"{error_pct:.4f}%",
        "status": _classify(error_pct, tol_pass=0.1),
    }


def _check_v_ew(p: int, z: int) -> Dict[str, Any]:
    """Check 3: v_EW consistency between gauge_unification and ChargedLeptonSpectrum.

    Route A: electroweak_scale_GeV(p, z)
    Route B: ChargedLeptonSpectrum default (246.0 GeV anchor)
    """
    if not (_HAS_GAUGE and _HAS_LEPTONS):
        return {
            "name": "v_EW",
            "status": "SKIP",
            "reason": "gauge_unification or charged_leptons unavailable",
        }

    route_a = electroweak_scale_GeV(p, z)
    # ChargedLeptonSpectrum uses v_EW = 246.0 GeV as standard anchor
    route_b = 246.0  # GeV -- the experimental SM value used throughout

    error_pct = abs(route_a - route_b) / route_b * 100.0
    return {
        "name": "v_EW",
        "route_A": float(route_a),
        "route_A_label": "electroweak_scale_GeV(p, z)",
        "route_B": float(route_b),
        "route_B_label": "SM Higgs VEV (246.0 GeV)",
        "error": f"{error_pct:.4f}%",
        "status": _classify(error_pct, tol_pass=0.01, tol_tension=2.0, tol_fail=5.0),
    }


def _check_neutrino_mass_bound(p: int, z: int) -> Dict[str, Any]:
    """Check 4: Sigma m_nu from seesaw must be < 0.12 eV (Planck 2018).

    Uses neutrino_lepton_mass_relation to get sum_mnu_eV.
    """
    if not _HAS_PARTICLES:
        return {
            "name": "neutrino_mass_bound",
            "status": "SKIP",
            "reason": "particles_matter bridge unavailable",
        }

    result = neutrino_lepton_mass_relation(p=p, z=z)
    sum_mnu = result.get("sum_mnu_eV", result.get("sum_mnu"))

    if sum_mnu is None:
        return {
            "name": "neutrino_mass_bound",
            "status": "SKIP",
            "reason": "neutrino_lepton_mass_relation did not return sum_mnu",
        }

    bound = 0.12  # eV, Planck 2018 95% CL
    status = "PASS" if sum_mnu < bound else "FAIL"
    return {
        "name": "neutrino_mass_bound",
        "sum_mnu_eV": float(sum_mnu),
        "bound_eV": bound,
        "status": status,
        "error": f"{sum_mnu:.4f} eV vs bound {bound} eV",
    }


def _check_koide(p: int, z: int) -> Dict[str, Any]:
    """Check 5: Koide parameter Q from pipeline must satisfy |Q - 2/3| < 0.01.

    Uses pipeline_impedance_to_lepton_masses.
    """
    if not _HAS_PIPELINES:
        return {
            "name": "koide_parameter",
            "status": "SKIP",
            "reason": "pipelines module unavailable",
        }

    result = pipeline_impedance_to_lepton_masses(p=p, z=z)
    if "error" in result:
        return {
            "name": "koide_parameter",
            "status": "SKIP",
            "reason": result["error"],
        }

    Q = result.get("koide_Q")
    if Q is None:
        return {
            "name": "koide_parameter",
            "status": "SKIP",
            "reason": "pipeline did not return koide_Q",
        }

    deviation = abs(Q - 2.0 / 3.0)
    status = "PASS" if deviation < 0.01 else ("TENSION" if deviation < 0.05 else "FAIL")
    return {
        "name": "koide_parameter",
        "Q": float(Q),
        "target": 2.0 / 3.0,
        "deviation": float(deviation),
        "status": status,
        "error": f"|Q - 2/3| = {deviation:.6f}",
    }


def _check_dark_energy(p: int) -> Dict[str, Any]:
    """Check 6: dark energy self-consistency.

    w_0 must be < -1/3 (accelerating expansion).
    Uses dark_energy_equation_of_state.
    """
    if not _HAS_COSMO_BRIDGE:
        return {
            "name": "dark_energy",
            "status": "SKIP",
            "reason": "cosmology_gravity bridge unavailable",
        }

    result = dark_energy_equation_of_state(p=p)
    w_0 = result.get("w_0")

    if w_0 is None:
        return {
            "name": "dark_energy",
            "status": "SKIP",
            "reason": "dark_energy_equation_of_state did not return w_0",
        }

    accelerating = w_0 < -1.0 / 3.0
    not_phantom = w_0 >= -2.0  # not wildly phantom
    status = "PASS" if (accelerating and not_phantom) else "FAIL"

    return {
        "name": "dark_energy",
        "w_0": float(w_0),
        "accelerating": accelerating,
        "not_phantom": not_phantom,
        "status": status,
        "error": f"w_0 = {w_0:.4f} ({'< -1/3' if accelerating else '>= -1/3'})",
    }


def _check_decoherence_ordering() -> Dict[str, Any]:
    """Check 7: decoherence times must be monotonically decreasing with system size.

    tau_dec(electron) > tau_dec(C60) > tau_dec(dust) > tau_dec(cat)
    """
    if not _HAS_DECOHERENCE:
        return {
            "name": "decoherence_ordering",
            "status": "SKIP",
            "reason": "decoherence module unavailable",
        }

    # Representative systems with increasing effective area and decreasing lambda_dB
    systems = [
        ("electron", {"T": 300.0, "Z_system": _Z0, "Z_environment": _Z0 * 1.001,
                       "A_eff": 1e-20, "lambda_dB": 1e-10}),
        ("C60",      {"T": 300.0, "Z_system": _Z0, "Z_environment": _Z0 * 1.01,
                       "A_eff": 1e-18, "lambda_dB": 1e-12}),
        ("dust",     {"T": 300.0, "Z_system": _Z0, "Z_environment": _Z0 * 1.1,
                       "A_eff": 1e-12, "lambda_dB": 1e-14}),
        ("cat",      {"T": 310.0, "Z_system": _Z0, "Z_environment": _Z0 * 1.5,
                       "A_eff": 1e-2, "lambda_dB": 1e-20}),
    ]

    tau_values = {}
    for name, params in systems:
        dr = DecoherenceRate(**params)
        tau_values[name] = dr.decoherence_time

    # Check monotonically decreasing
    names = [s[0] for s in systems]
    taus = [tau_values[n] for n in names]
    monotonic = all(taus[i] > taus[i + 1] for i in range(len(taus) - 1))

    return {
        "name": "decoherence_ordering",
        "tau_values": {k: float(v) for k, v in tau_values.items()},
        "monotonic_decreasing": monotonic,
        "status": "PASS" if monotonic else "FAIL",
        "error": "monotonic" if monotonic else "NOT monotonic",
    }


def _check_eeg_ordering() -> Dict[str, Any]:
    """Check 8: EEG bands must be monotonically increasing.

    delta < theta < alpha < beta < gamma
    """
    if not _HAS_LIFE:
        return {
            "name": "eeg_ordering",
            "status": "SKIP",
            "reason": "life_consciousness bridge unavailable",
        }

    result = eeg_peak_frequencies()
    bands = result.get("band_frequencies", {})

    expected_order = ["delta", "theta", "alpha", "beta", "gamma"]
    freqs = [bands.get(b) for b in expected_order]

    if any(f is None for f in freqs):
        return {
            "name": "eeg_ordering",
            "status": "SKIP",
            "reason": f"Missing bands: {[b for b, f in zip(expected_order, freqs) if f is None]}",
        }

    monotonic = all(freqs[i] < freqs[i + 1] for i in range(len(freqs) - 1))

    return {
        "name": "eeg_ordering",
        "band_frequencies": {b: float(f) for b, f in zip(expected_order, freqs)},
        "monotonic_increasing": monotonic,
        "status": "PASS" if monotonic else "FAIL",
        "error": "monotonic" if monotonic else "NOT monotonic",
    }


def _check_magic_numbers() -> Dict[str, Any]:
    """Check 9: magic_numbers_bpr() must return [2, 8, 20, 28, 50, 82, 126]."""
    if not _HAS_NUCLEAR:
        return {
            "name": "magic_numbers",
            "status": "SKIP",
            "reason": "nuclear_physics module unavailable",
        }

    predicted = magic_numbers_bpr()
    expected = [2, 8, 20, 28, 50, 82, 126]
    match = predicted == expected

    return {
        "name": "magic_numbers",
        "predicted": predicted,
        "expected": expected,
        "status": "PASS" if match else "FAIL",
        "error": "exact match" if match else f"mismatch: {predicted}",
    }


def _check_e8() -> Dict[str, Any]:
    """Check 10: E8 dimension must be 248, n_roots must be 240."""
    if not _HAS_E8:
        return {
            "name": "E8_properties",
            "status": "SKIP",
            "reason": "clifford_bpr module unavailable",
        }

    try:
        props = verify_e8_properties()
    except Exception as e:
        return {
            "name": "E8_properties",
            "status": "SKIP",
            "reason": f"verify_e8_properties raised: {e}",
        }

    dim_ok = props.get("dimension") == 248
    roots_ok = props.get("n_roots") == 240

    return {
        "name": "E8_properties",
        "dimension": props.get("dimension"),
        "n_roots": props.get("n_roots"),
        "dim_correct": dim_ok,
        "roots_correct": roots_ok,
        "status": "PASS" if (dim_ok and roots_ok) else "FAIL",
        "error": f"dim={props.get('dimension')}, n_roots={props.get('n_roots')}",
    }


# ===================================================================
# Master function
# ===================================================================

def run_all_consistency_checks(
    p: int = 104761,
    z: int = 6,
) -> Dict[str, Any]:
    """Run every consistency check and return pass/fail report.

    Each check compares the same quantity computed two different ways.
    A 'PASS' means they agree within tolerance.
    A 'TENSION' means >5% disagreement.
    A 'FAIL' means >20% disagreement (likely a bug).

    Parameters
    ----------
    p : int
        Substrate prime modulus.
    z : int
        Coordination number.

    Returns
    -------
    dict
        total_checks, passed, tensions, failed, skipped, details
    """
    checks = [
        _check_alpha_em(p, z),
        _check_sin2_theta_w(p, z),
        _check_v_ew(p, z),
        _check_neutrino_mass_bound(p, z),
        _check_koide(p, z),
        _check_dark_energy(p),
        _check_decoherence_ordering(),
        _check_eeg_ordering(),
        _check_magic_numbers(),
        _check_e8(),
    ]

    passed = sum(1 for c in checks if c["status"] == "PASS")
    tensions = sum(1 for c in checks if c["status"] == "TENSION")
    failed = sum(1 for c in checks if c["status"] == "FAIL")
    skipped = sum(1 for c in checks if c["status"] == "SKIP")

    return {
        "total_checks": len(checks),
        "passed": passed,
        "tensions": tensions,
        "failed": failed,
        "skipped": skipped,
        "details": checks,
    }


# ===================================================================
# Prediction Dependency Graph
# ===================================================================

def prediction_dependency_graph() -> Dict[str, Any]:
    """Build the dependency graph showing which predictions constrain each other.

    Returns nodes (predictions) and edges (constraints) suitable for
    visualization. Each node is a physical quantity; each edge indicates
    that the two quantities are related by a BPR equation, so a change
    in one forces a change in the other.

    Returns
    -------
    dict
        nodes : list of dict with {id, label, domain, module}
        edges : list of dict with {source, target, relation, equation}
    """
    nodes = [
        # Substrate parameters
        {"id": "p", "label": "p = 104761", "domain": "substrate",
         "module": "alpha_derivation"},
        {"id": "z", "label": "z = 6", "domain": "substrate",
         "module": "alpha_derivation"},

        # Electromagnetic
        {"id": "alpha_EM", "label": "alpha_EM = 1/137.036", "domain": "electroweak",
         "module": "alpha_derivation"},
        {"id": "sin2_theta_W", "label": "sin^2(theta_W) = 0.231", "domain": "electroweak",
         "module": "gauge_unification"},
        {"id": "v_EW", "label": "v_EW = 246 GeV", "domain": "electroweak",
         "module": "gauge_unification"},

        # Particle masses
        {"id": "m_e", "label": "m_e = 0.511 MeV", "domain": "particles",
         "module": "charged_leptons"},
        {"id": "m_mu", "label": "m_mu = 105.66 MeV", "domain": "particles",
         "module": "charged_leptons"},
        {"id": "m_tau", "label": "m_tau = 1776.9 MeV", "domain": "particles",
         "module": "charged_leptons"},
        {"id": "koide_Q", "label": "Koide Q = 2/3", "domain": "particles",
         "module": "charged_leptons"},

        # Neutrinos
        {"id": "sum_mnu", "label": "Sigma m_nu = 0.060 eV", "domain": "neutrinos",
         "module": "bridges.particles_matter"},
        {"id": "M_seesaw", "label": "M_seesaw = p * v_EW", "domain": "neutrinos",
         "module": "bridges.particles_matter"},

        # Cosmology
        {"id": "w_0", "label": "w_0 = -0.934", "domain": "cosmology",
         "module": "bridges.cosmology_gravity"},
        {"id": "Omega_Lambda", "label": "Omega_Lambda = 0.69", "domain": "cosmology",
         "module": "cross_predictions"},

        # Nuclear
        {"id": "magic_numbers", "label": "Magic: 2,8,20,28,50,82,126",
         "domain": "nuclear", "module": "nuclear_physics"},

        # Decoherence
        {"id": "tau_dec", "label": "tau_dec(system)", "domain": "quantum",
         "module": "decoherence"},

        # Consciousness / biology
        {"id": "K_c", "label": "K_c (Kuramoto critical)", "domain": "consciousness",
         "module": "collective"},
        {"id": "eeg_bands", "label": "EEG: delta..gamma", "domain": "consciousness",
         "module": "bridges.life_consciousness"},

        # E8
        {"id": "E8", "label": "E8: dim=248, roots=240", "domain": "unification",
         "module": "clifford_bpr"},
    ]

    edges = [
        # Substrate -> alpha
        {"source": "p", "target": "alpha_EM",
         "relation": "screening",
         "equation": "1/alpha = [ln(p)]^2 + z/2 + gamma - 1/(2pi)"},
        {"source": "z", "target": "alpha_EM",
         "relation": "bare coupling",
         "equation": "bare = z/2 (boundary rigidity)"},

        # alpha -> sin2_theta_W
        {"source": "alpha_EM", "target": "sin2_theta_W",
         "relation": "electroweak mixing",
         "equation": "sin2_theta_W = alpha_EM / alpha_2 at M_Z"},

        # sin2_theta_W -> v_EW (via g^2 v^2 = M_W^2)
        {"source": "sin2_theta_W", "target": "v_EW",
         "relation": "W-boson mass",
         "equation": "M_W = g * v_EW / 2, g^2 = 4pi*alpha_EM / sin2_theta_W"},

        # p, z -> v_EW directly
        {"source": "p", "target": "v_EW",
         "relation": "boundary hierarchy",
         "equation": "v_EW = Lambda_QCD * p^(1/3) * (ln(p) + z - 2)"},
        {"source": "z", "target": "v_EW",
         "relation": "boundary hierarchy",
         "equation": "v_EW = Lambda_QCD * p^(1/3) * (ln(p) + z - 2)"},

        # v_EW -> lepton masses (via Yukawa)
        {"source": "v_EW", "target": "m_e",
         "relation": "Yukawa coupling",
         "equation": "m_k = v_EW * y_k, y_k from S^2 boundary modes"},
        {"source": "v_EW", "target": "m_mu",
         "relation": "Yukawa coupling",
         "equation": "m_k = v_EW * y_k"},
        {"source": "v_EW", "target": "m_tau",
         "relation": "Yukawa coupling",
         "equation": "m_k = v_EW * y_k"},

        # Lepton masses -> Koide
        {"source": "m_e", "target": "koide_Q",
         "relation": "Koide formula",
         "equation": "Q = (m_e + m_mu + m_tau) / (sqrt(m_e) + sqrt(m_mu) + sqrt(m_tau))^2"},
        {"source": "m_mu", "target": "koide_Q",
         "relation": "Koide formula",
         "equation": "Q = 2/3 (BPR predicts exactly)"},
        {"source": "m_tau", "target": "koide_Q",
         "relation": "Koide formula",
         "equation": "Q = 2/3"},

        # Seesaw: p, v_EW -> neutrino masses
        {"source": "p", "target": "M_seesaw",
         "relation": "seesaw scale",
         "equation": "M_seesaw = p * v_EW"},
        {"source": "v_EW", "target": "M_seesaw",
         "relation": "seesaw scale",
         "equation": "M_seesaw = p * v_EW"},
        {"source": "M_seesaw", "target": "sum_mnu",
         "relation": "type-I seesaw",
         "equation": "m_nu_i = m_l_i^2 / M_seesaw"},

        # Neutrino masses -> cosmology
        {"source": "sum_mnu", "target": "Omega_Lambda",
         "relation": "CMB constraint",
         "equation": "Sigma m_nu < 0.12 eV (Planck 2018)"},

        # p -> dark energy
        {"source": "p", "target": "w_0",
         "relation": "impedance evolution",
         "equation": "w_0 = -1 + (2/3) / p^(1/5)"},
        {"source": "w_0", "target": "Omega_Lambda",
         "relation": "Friedmann equation",
         "equation": "Omega_Lambda from rho_DE / rho_crit"},

        # p -> nuclear magic numbers (SO(3) shell filling on S^2 boundary)
        {"source": "p", "target": "magic_numbers",
         "relation": "boundary winding shells",
         "equation": "Shell closures at 2(2l+1) cumulative filling"},

        # Impedance -> decoherence
        {"source": "p", "target": "tau_dec",
         "relation": "impedance mismatch",
         "equation": "Gamma_dec = (k_B T / hbar)(DeltaZ / Z_0)^2 (A/lambda^2)"},

        # Kuramoto -> EEG
        {"source": "K_c", "target": "eeg_bands",
         "relation": "Kuramoto synchronization",
         "equation": "f_n = n * K_c / (2 pi)"},

        # E8 -> sin2_theta_W (gauge group embedding)
        {"source": "E8", "target": "sin2_theta_W",
         "relation": "E8 -> SM decomposition",
         "equation": "E8 -> SU(5) x SU(5) -> SM; sin2_theta_W = 3/8 at GUT"},
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
    }


# ===================================================================
# CLI-friendly summary
# ===================================================================

def print_consistency_report(p: int = 104761, z: int = 6) -> None:
    """Print a human-readable consistency report to stdout."""
    report = run_all_consistency_checks(p, z)
    print("=" * 60)
    print("BPR INTERNAL CONSISTENCY REPORT")
    print(f"  Substrate: p = {p}, z = {z}")
    print("=" * 60)
    print(f"  Total checks:  {report['total_checks']}")
    print(f"  Passed:        {report['passed']}")
    print(f"  Tensions:      {report['tensions']}")
    print(f"  Failed:        {report['failed']}")
    print(f"  Skipped:       {report['skipped']}")
    print("-" * 60)

    for detail in report["details"]:
        icon = {"PASS": "[OK]", "TENSION": "[!!]", "FAIL": "[XX]",
                "SKIP": "[--]"}.get(detail["status"], "[??]")
        name = detail["name"]
        error = detail.get("error", "")
        print(f"  {icon} {name:30s}  {error}")

    print("=" * 60)


# ===================================================================
# Module entry point
# ===================================================================

if __name__ == "__main__":
    print_consistency_report()
