"""Tests for LunarFire v0.6 mirror/nozzle loss-cone and collector model."""

from math import sqrt

import pytest

from helionis.mirror_nozzle_v06 import (
    D_HE3_ALPHA_ENERGY_MEV,
    D_HE3_PROTON_ENERGY_MEV,
    MirrorNozzleV06Assumptions,
    charged_product_collector_efficiency,
    loss_cone_fraction,
    run_mirror_nozzle_v06_sweep,
)
from helionis.mirror_nozzle_v06_reporting import (
    write_mirror_nozzle_v06_csv,
    write_mirror_nozzle_v06_markdown,
)


def test_loss_cone_fraction_decreases_with_mirror_ratio():
    """Higher mirror ratio should reduce the isotropic loss-cone fraction."""
    low_ratio = loss_cone_fraction(2.5)
    high_ratio = loss_cone_fraction(8.0)

    assert 0.0 < high_ratio < low_ratio < 1.0
    assert loss_cone_fraction(4.0) == pytest.approx(1.0 - sqrt(0.75))
    assert loss_cone_fraction(1.0e9) == pytest.approx(1.0 / (2.0e9))


def test_charged_product_collector_efficiency_prefers_staged_voltage_match():
    """Separate alpha/proton voltage stages should beat a badly mismatched pair."""
    matched = charged_product_collector_efficiency(
        alpha_collector_voltage_kv=(D_HE3_ALPHA_ENERGY_MEV / 2.0) * 1000.0,
        proton_collector_voltage_kv=D_HE3_PROTON_ENERGY_MEV * 1000.0,
    )
    mismatched = charged_product_collector_efficiency(
        alpha_collector_voltage_kv=500.0,
        proton_collector_voltage_kv=2500.0,
    )

    assert matched > mismatched
    assert matched <= MirrorNozzleV06Assumptions().max_direct_conversion_efficiency


def test_mirror_nozzle_v06_sweep_returns_sorted_candidates():
    """The v0.6 sweep should rank feasible mirror/nozzle points by plant-net."""
    rows = run_mirror_nozzle_v06_sweep(target_screening_net_mw=50.0, limit=6)

    assert len(rows) == 6
    assert rows == sorted(rows, key=lambda row: row.plant_net_power_mw, reverse=True)
    assert all(row.loss_cone_fraction > 0.0 for row in rows)
    assert all(row.plug_field_t <= row.max_plug_field_t for row in rows)
    for row in rows:
        assert row.plant_net_power_mw == pytest.approx(
            row.screening_net_power_mw - row.engineering_load_mw
        )
        assert row.effective_magnet_mass_proxy_tonnes == pytest.approx(
            row.magnet_mass_proxy_tonnes + row.plug_coil_mass_proxy_tonnes
        )
        assert row.collector_area_m2 == pytest.approx(
            row.charged_power_mw / row.assumed_collector_power_density_mw_m2
        )


def test_v06_closure_is_not_better_than_unpenalized_collector_case():
    """More detailed collector staging should not silently exceed the cap."""
    assumptions = MirrorNozzleV06Assumptions(max_direct_conversion_efficiency=0.80)
    rows = run_mirror_nozzle_v06_sweep(
        target_screening_net_mw=50.0,
        assumptions=assumptions,
        limit=4,
    )

    assert rows
    assert all(row.direct_conversion_efficiency <= 0.80 for row in rows)


def test_mirror_nozzle_v06_assumptions_validate_inputs():
    """Public v0.6 assumptions should reject invalid values early."""
    with pytest.raises(ValueError, match="loss_cone_transport_scale"):
        MirrorNozzleV06Assumptions(loss_cone_transport_scale=-1.0)

    with pytest.raises(ValueError, match="finite number"):
        MirrorNozzleV06Assumptions(beta=None)

    with pytest.raises(ValueError, match="finite number"):
        charged_product_collector_efficiency(
            alpha_collector_voltage_kv=None,
            proton_collector_voltage_kv=15000.0,
        )


def test_mirror_nozzle_v06_reporting_writes_outputs(tmp_path):
    """Reports should include loss-cone and staged collector fields."""
    rows = run_mirror_nozzle_v06_sweep(target_screening_net_mw=50.0, limit=4)
    csv_path = tmp_path / "mirror_nozzle_v06.csv"
    markdown_path = tmp_path / "mirror_nozzle_v06.md"

    write_mirror_nozzle_v06_csv(rows, csv_path)
    write_mirror_nozzle_v06_markdown(rows, markdown_path)

    assert "loss_cone_fraction" in csv_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "LunarFire v0.6 Mirror/Nozzle Output" in markdown
    assert "loss cone" in markdown.lower()
    assert "proton collector" in markdown.lower()
