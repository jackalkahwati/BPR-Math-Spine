import numpy as np


def test_resonance_zero_loader_seed():
    from bpr.resonance import load_riemann_zeros, RIEMANN_ZEROS

    z = load_riemann_zeros(5)
    assert z.shape == (5,)
    assert np.allclose(z, RIEMANN_ZEROS[:5])


def test_resonance_pair_correlation_runs():
    from bpr.resonance import RiemannZeroVerification

    v = RiemannZeroVerification(n_zeros=20)
    ok, info = v.verify_pair_correlation(max_r=2.0, bins=20, sigma_threshold=10.0)
    # This is a smoke test (statistics are noisy at n=20); just ensure it runs and returns expected keys.
    assert "chi_sq_per_dof" in info
    assert np.isfinite(info["chi_sq_per_dof"])


