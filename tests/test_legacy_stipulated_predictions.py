"""Tripwires for legacy predictions withdrawn or relabelled in 2026-10.

Found in the 2026-10 review of unmerged branches (claude/bpr-ecdlp-research-SWQnW,
claude/audit-bpr-parameters-6fSrQ); recorded in doc/CLOSED_AND_DEPRECATED.md.
These tests pin the labels and the corrected arithmetic, not any claim of agreement with experiment:
- Bridges 11 and 12 (BH quasi-normal-mode shift, prime-periodic ringdown) are stipulated deviations, with no
  detector claim and no "horizon winding entropy" justification;
- the CKM phase delta_CP = pi/2 - 1/sqrt(z+1) is an ANSATZ, and its 1.196 fallback is an outdated input;
- bpr.quantum_gravity_pheno is legacy and superseded, with the factor-2 (energy shift) and 3/2 (time delay)
  corrections;
- P20.7 is no longer a flagship of the research loop.
"""

import inspect
import json
import math
import re
from pathlib import Path

import pytest

from bpr.bridges import cosmology_gravity as cg

REPO_ROOT = Path(__file__).resolve().parents[1]
P = 104761

FORBIDDEN_QNM_WORDING = [
    r"cosmic\s+explorer",
    r"einstein\s+telescope",
    r"\bCE\s*/\s*ET\b",
    r"\bLISA\b",
    r"\b3G\s+detector",
    r"testable",
    r"winding\s+entropy",
    r"horizon\s+entropy",
    r"from\s+the\s+horizon",
]


def _string_values(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _string_values(value)


def _assert_no_forbidden(text, where):
    for pattern in FORBIDDEN_QNM_WORDING:
        assert not re.search(pattern, text, flags=re.IGNORECASE), f"{pattern!r} found in {where}"


# ---------------------------------------------------------------------------
# Bridges 11 and 12
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M_solar", [0.5, 1.0, 10.0, 30.0, 62.0])
def test_bridge11_status_is_stipulated_and_shift_is_the_written_coefficient(M_solar):
    r = cg.bh_quasinormal_modes(M_solar=M_solar, p=P)
    assert r["status"] == "stipulated_deviation"
    assert "withdrawn" in r["limitation"].lower()
    # Mass-independent stipulated shift ln(p)/(4 pi p) = 8.78e-6 at p = 104761
    assert r["fractional_shift"] == pytest.approx(math.log(P) / (4 * math.pi * P), rel=1e-9)
    assert r["fractional_shift"] == pytest.approx(8.7807e-6, rel=1e-4)
    for text in _string_values(r):
        _assert_no_forbidden(text, f"bh_quasinormal_modes(M_solar={M_solar}) output")


def test_bridge11_detectable_by_agrees_with_docstring():
    # The nominal 0.1 Hz threshold is met only below ~1.06 M_sun at p = 104761.
    light = cg.bh_quasinormal_modes(M_solar=1.0, p=P)
    assert light["exceeds_nominal_resolution"] is True
    assert light["nominal_threshold_mass_solar"] == pytest.approx(1.06, abs=0.005)
    assert light["detectable_by"].startswith("None claimed")
    for M_solar in (1.07, 10.0, 30.0):
        r = cg.bh_quasinormal_modes(M_solar=M_solar, p=P)
        assert r["exceeds_nominal_resolution"] is False
        assert r["detectable_by"] == "Not detectable with current or planned detectors"
    doc = cg.bh_quasinormal_modes.__doc__
    assert "LIMITATION (2026-10)" in doc
    assert "Not detectable with current or planned detectors" in doc
    assert "1.06 M_sun" in doc


def test_bridge12_status_is_stipulated():
    r = cg.bh_ringdown_prime_periodic(M_solar=10.0, p=P, n_samples=64)
    assert r["status"] == "stipulated_deviation"
    assert "withdrawn" in r["limitation"].lower()
    assert r["epsilon_modulation"] == pytest.approx(1.0 / P)
    assert r["f_modulation_Hz"] == pytest.approx(1.77e38, rel=1e-2)
    assert "LIMITATION (2026-10)" in cg.bh_ringdown_prime_periodic.__doc__
    for text in _string_values(r):
        _assert_no_forbidden(text, "bh_ringdown_prime_periodic output")


@pytest.mark.parametrize("func", [cg.bh_quasinormal_modes, cg.bh_ringdown_prime_periodic])
def test_qnm_bridges_source_has_no_detector_or_horizon_entropy_claims(func):
    _assert_no_forbidden(inspect.getsource(func), f"source of {func.__name__}")


# ---------------------------------------------------------------------------
# delta_CP
# ---------------------------------------------------------------------------

def test_delta_cp_is_labelled_ansatz_not_derived():
    from bpr.qcd_flavor import CKMMatrix

    doc = CKMMatrix.__doc__
    assert "δ_CP: ANSATZ" in doc
    assert "δ_CP: DERIVED" not in doc
    source = inspect.getsource(CKMMatrix)
    assert "δ_CP: DERIVED" not in source
    assert "outdated experimental input" in source
    assert "PDG 2024: 65.5°–72.2°)" not in source


def test_delta_cp_values_unchanged_by_relabelling():
    from bpr.qcd_flavor import CKMMatrix

    ansatz = CKMMatrix(p=P).mixing_angles()["delta_CP_deg"]
    assert ansatz == pytest.approx(math.degrees(math.pi / 2 - 1 / math.sqrt(7)), rel=1e-12)
    # Fallback when p is None: the outdated experimental input 1.196 rad, not a BPR value
    fallback = CKMMatrix().mixing_angles()["delta_CP_deg"]
    assert fallback == pytest.approx(math.degrees(1.196), rel=1e-12)


# ---------------------------------------------------------------------------
# quantum_gravity_pheno (legacy LIV)
# ---------------------------------------------------------------------------

def test_quantum_gravity_pheno_is_marked_superseded():
    import bpr.quantum_gravity_pheno as qg

    assert "SUPERSEDED (2026-10)" in qg.__doc__
    assert "SUPERSEDED" in qg.ModifiedDispersion.__doc__
    assert "SUPERSEDED" in qg.GeneralizedUncertainty.__doc__
    assert "SUPERSEDED" in qg.LorentzInvariance.__doc__
    li = qg.LorentzInvariance(p=P)
    assert li.fractional_speed_variation == pytest.approx(math.exp(-P ** (1 / 3)), rel=1e-12)
    assert li.fractional_speed_variation == pytest.approx(3.3641e-21, rel=1e-4)


def _toy_dispersion_coefficients(xi=1e-3, M=1.0, k=0.1):
    """Solve E^2 = k^2 + xi E^4 / M^2 numerically (toy units); return the first-order coefficients
    a in E = k (1 + a xi (k/M)^2) and b in v = dE/dk = 1 + b xi (k/M)^2."""

    def energy(kk):
        # small root of xi E^4 / M^2 - E^2 + kk^2 = 0
        x = (1 - math.sqrt(1 - 4 * xi * kk ** 2 / M ** 2)) / (2 * xi / M ** 2)
        return math.sqrt(x)

    h = 1e-5
    E = energy(k)
    v = (energy(k + h) - energy(k - h)) / (2 * h)
    scale = xi * (k / M) ** 2
    return (E / k - 1) / scale, (v - 1) / scale


def test_dispersion_first_order_coefficients_are_one_half_and_three_halves():
    a, b = _toy_dispersion_coefficients()
    assert a == pytest.approx(0.5, rel=1e-3)
    assert b == pytest.approx(1.5, rel=1e-3)


def test_energy_correction_and_grb_delay_use_corrected_factors():
    import bpr.quantum_gravity_pheno as qg

    md = qg.ModifiedDispersion(p=104729)
    E_GeV, D_Mpc = 1000.0, 1000.0
    assert md.energy_correction(E_GeV) == pytest.approx(
        0.5 * md.xi_2 * (E_GeV / qg._E_PLANCK_GEV) ** 2, rel=1e-12)
    E_J = E_GeV * 1.602e-10
    L_m = D_Mpc * 3.0857e22
    M_c2 = qg._M_PL_KG * qg._C ** 2
    assert md.grb_time_delay(E_GeV, D_Mpc) == pytest.approx(
        1.5 * md.xi_2 * (E_J / M_c2) ** 2 * L_m / qg._C, rel=1e-12)
    # Legacy P20.3 value at 1 TeV, 1 Gpc after the 3/2 correction (was 3.30e-21 s with 1/2)
    assert md.grb_time_delay(E_GeV, D_Mpc) == pytest.approx(9.888e-21, rel=1e-3)


# ---------------------------------------------------------------------------
# Research-loop policy
# ---------------------------------------------------------------------------

def test_p20_7_is_not_a_flagship_and_boundary_snapshot_matches_policy():
    from bpr.research.policy import DEFAULT_AUTONOMY_POLICY

    flagships = list(DEFAULT_AUTONOMY_POLICY.flagship_prediction_ids)
    assert "P20.7_LI_delta_c_over_c" not in flagships
    snapshot = json.loads((REPO_ROOT / "data" / "research" / "evaluator_boundary.json").read_text())
    assert snapshot["flagship_prediction_ids"] == flagships
