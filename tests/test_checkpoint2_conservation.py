"""Checkpoint 2: covariant conservation nabla_mu T^munu = 0 at linear order.

Runnable Python/SymPy verification mirroring wolfram/checkpoint2_conservation.wls.
"""
import pytest


def test_conservation_plane_wave_linear_order():
    """The plane-wave verifier returns True for linear-order conservation."""
    from bpr.metric import verify_conservation_plane_wave
    result = verify_conservation_plane_wave()
    holds = result[0] if isinstance(result, tuple) else result
    assert holds is True


def test_conservation_spherical_mode_linear_order():
    from bpr.metric import verify_conservation_spherical_mode
    result = verify_conservation_spherical_mode()
    holds = result[0] if isinstance(result, tuple) else result
    assert holds is True
