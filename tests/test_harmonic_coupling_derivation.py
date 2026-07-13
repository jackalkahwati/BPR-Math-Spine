"""Lock in the K_ij derivation: symmetry fixes the structure, but it's frustrated.

Guards the two halves of the honest result: (a) translation invariance forces a
circulant, Gauss-sum-balanced coupling (real, derived), and (b) that derived
coupling does NOT produce coherence on its own — coherence is a condensate
phenomenon. No test asserts the substrate structure "works," so a later edit
can't spin the frustrated result into a success.
"""
import numpy as np
import pytest

from bpr.harmonic_coupling_derivation import (
    legendre_circulant,
    circulant_eigenvalues,
    spectral_balance,
    coherence_comparison,
    report,
)


def test_coupling_is_circulant():
    """K_ij depends only on (i−j) mod p — the translation-invariance result."""
    p = 101
    K = legendre_circulant(p)
    # every descending diagonal is constant ⇒ circulant
    for shift in (1, 5, 37):
        vals = [K[i, (i + shift) % p] for i in range(p)]
        assert np.allclose(vals, vals[0])


def test_spectrum_is_balanced_at_sqrt_p():
    """p ≡ 1 mod 4: eigenvalues are ±√p, split exactly half positive/half negative
    (Gauss sum). This is the frustration, computed not asserted."""
    for p in (101, 229, 401):
        bal = spectral_balance(p)
        assert bal["p_mod_4"] == 1
        assert bal["lambda_scale_vs_sqrt_p"] == pytest.approx(1.0, abs=1e-6)
        assert bal["frac_positive"] == pytest.approx(0.5, abs=0.02)
        assert bal["balanced"] is True


def test_eigenvalues_magnitude():
    """Non-zero eigenvalues sit at magnitude √p."""
    p = 101
    ev = circulant_eigenvalues(p, g=1.0)
    nz = ev[np.abs(ev) > 1e-9]
    assert np.allclose(np.abs(nz), np.sqrt(p), atol=1e-6)


def test_meanfield_coheres_substrate_does_not():
    """KEY RESULT: at matched top eigenvalue, the uniform condensate coheres but
    the derived (frustrated) substrate coupling does not."""
    c = coherence_comparison(p=101)
    assert c["r_meanfield"] > 0.8           # condensate → coherent
    assert c["r_substrate"] < 0.2           # frustrated → incoherent
    # and the substrate coupling is genuinely much weaker at producing coherence
    assert c["r_meanfield"] > 4 * c["r_substrate"]


def test_report_states_derived_structure_and_frustration():
    """HONESTY GUARD: report must credit the derived structure AND state that it
    does not deliver coherence by itself (condensate does)."""
    txt = report(p=101)
    assert "circulant" in txt
    assert "frustrated" in txt.lower()
    assert "does NOT produce global coherence" in txt or \
           "does not produce global coherence" in txt.lower()
    assert "condensate" in txt.lower()
