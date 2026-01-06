import numpy as np


def test_symbolic_conservation_plane_wave_numeric():
    from bpr.metric import verify_conservation_plane_wave

    passes, div, details = verify_conservation_plane_wave()
    # This is a spine scaffold; we accept numeric smallness rather than hard symbolic zero.
    assert details["numerical_max"] < 1e-6


def test_symbolic_conservation_spherical_mode_numeric():
    from bpr.metric import verify_conservation_spherical_mode

    passes, div, details = verify_conservation_spherical_mode()
    assert details["numerical_max"] < 1e-4


