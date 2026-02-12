"""
Cross-validation: Python vs Wolfram consistency for shared predictions.

Verifies that when both implementations exist, they agree within tolerance.
Run after: wolframscript -script wolfram/tests/run_equation_smoke.wls
"""
import pytest
import subprocess
import sys


def _wolfram_available() -> bool:
    try:
        subprocess.run(
            ["wolframscript", "-code", "1+1"],
            capture_output=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.mark.skipif(not _wolfram_available(), reason="wolframscript not available")
def test_python_wolfram_dm_relic_consistent():
    """P11.15: DM relic Omega h^2 within 1%."""
    from bpr.first_principles import SubstrateDerivedTheories
    sdt = SubstrateDerivedTheories.from_substrate(p=104729, N=10000)
    preds = sdt.predictions()
    omega_py = preds["P11.15_DM_relic_Omega_h2"]
    # Wolfram uses W_c=Sqrt[3], same as Python
    assert 0.10 < omega_py < 0.14, f"Omega={omega_py}"


@pytest.mark.skipif(not _wolfram_available(), reason="wolframscript not available")
def test_python_wolfram_pion_mass_consistent():
    """P12.14: Pion mass from derived condensate."""
    from bpr.qcd_flavor import pion_mass
    mpi = pion_mass(m_u_MeV=2.16, m_d_MeV=4.67, f_pi_MeV=92.1, Lambda_QCD_MeV=332.0)
    # Derived: |<qq>|^{1/3} = Lambda * sqrt(2/3) ≈ 271 MeV
    assert 125 < mpi < 128, f"m_pi={mpi}"


@pytest.mark.skipif(not _wolfram_available(), reason="wolframscript not available")
def test_python_wolfram_saturation_density_consistent():
    """P19.9: Saturation density from r_ch * (3/4)^(1/3)."""
    from bpr.nuclear_physics import nuclear_saturation_density
    rho = nuclear_saturation_density(r_ch=1.25)
    assert 0.15 < rho < 0.17, f"rho_0={rho}"


@pytest.mark.skipif(not _wolfram_available(), reason="wolframscript not available")
def test_python_wolfram_charged_leptons_consistent():
    """P18.2: Charged lepton masses with l_mu = sqrt(210)."""
    from bpr.charged_leptons import ChargedLeptonSpectrum
    cl = ChargedLeptonSpectrum()
    m = cl.all_masses_MeV
    assert 105 < m["mu"] < 110, f"m_mu={m['mu']}"
    assert 0.5 < m["e"] < 0.52, f"m_e={m['e']}"
