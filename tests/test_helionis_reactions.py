"""Tests for Helionis reaction accounting and screening physics."""

import pytest

from helionis.reactions import D_D_AVERAGE, D_HE3, D_T


def test_d_he3_energy_is_charged_particle_dominant():
    """D-He3 primary products should carry essentially all energy as charged particles."""
    assert D_HE3.q_mev == pytest.approx(18.353, rel=1e-3)
    assert D_HE3.neutron_fraction == pytest.approx(0.0)
    assert D_HE3.charged_fraction == pytest.approx(1.0)


def test_d_t_energy_is_neutron_dominant():
    """D-T is useful as a baseline because most output energy is carried by neutrons."""
    assert D_T.q_mev == pytest.approx(17.589, rel=1e-3)
    assert D_T.neutron_fraction == pytest.approx(14.1 / 17.589, rel=1e-3)
    assert D_T.charged_fraction < 0.21


def test_d_d_average_accounts_for_side_neutron_branch():
    """A 50/50 D-D branch average should keep the side-neutron burden visible."""
    assert D_D_AVERAGE.q_mev == pytest.approx((4.033 + 3.269) / 2.0)
    assert D_D_AVERAGE.neutron_mev == pytest.approx(2.45 / 2.0)
    assert 0.3 < D_D_AVERAGE.neutron_fraction < 0.4
