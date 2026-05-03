"""Tests for LunarFire v0.5 mirror/nozzle model."""

import pytest

from helionis.mirror_nozzle import (
    MirrorNozzleAssumptions,
    direct_conversion_efficiency_for_collector,
    mirror_end_loss_multiplier,
    run_mirror_nozzle_sweep,
)
from helionis.mirror_nozzle_reporting import (
    write_mirror_nozzle_csv,
    write_mirror_nozzle_markdown,
)


def test_mirror_end_loss_decreases_with_mirror_ratio():
    """Higher mirror ratio should reduce the explicit end-loss term."""
    low_ratio = mirror_end_loss_multiplier(4.0)
    high_ratio = mirror_end_loss_multiplier(12.0)

    assert high_ratio < low_ratio
    assert high_ratio > 0.0


def test_collector_efficiency_has_voltage_match_peak():
    """Collector voltage should matter for charged-particle conversion."""
    near_target = direct_conversion_efficiency_for_collector(1500.0)
    far_low = direct_conversion_efficiency_for_collector(250.0)
    far_high = direct_conversion_efficiency_for_collector(4000.0)

    assert near_target > far_low
    assert near_target > far_high
    assert near_target <= 0.88


def test_mirror_nozzle_sweep_returns_sorted_feasible_candidates():
    """The default sweep should rank feasible mirror/nozzle points by plant-net."""
    rows = run_mirror_nozzle_sweep(target_screening_net_mw=50.0, limit=8)

    assert len(rows) == 8
    assert rows == sorted(rows, key=lambda row: row.plant_net_power_mw, reverse=True)
    assert all(row.plug_field_t <= row.max_plug_field_t for row in rows)
    assert all(row.collector_voltage_kv > 0.0 for row in rows)


def test_mirror_nozzle_sweep_respects_plug_field_limit():
    """A low plug-field cap should filter all otherwise plausible rows."""
    assumptions = MirrorNozzleAssumptions(max_plug_field_t=2.0)

    rows = run_mirror_nozzle_sweep(
        target_screening_net_mw=50.0,
        assumptions=assumptions,
    )

    assert rows == []


def test_plug_coil_mass_penalty_reduces_plant_net():
    """Charging plug-field mass should reduce the apparent plant-net margin."""
    no_penalty = MirrorNozzleAssumptions(plug_coil_mass_coefficient_tonnes_per_t2=0.0)
    high_penalty = MirrorNozzleAssumptions(
        plug_coil_mass_coefficient_tonnes_per_t2=0.40
    )

    optimistic = run_mirror_nozzle_sweep(
        target_screening_net_mw=50.0,
        assumptions=no_penalty,
        limit=1,
    )[0]
    burdened = run_mirror_nozzle_sweep(
        target_screening_net_mw=50.0,
        assumptions=high_penalty,
        limit=1,
    )[0]

    assert burdened.plant_net_power_mw < optimistic.plant_net_power_mw
    assert burdened.effective_magnet_mass_proxy_tonnes > burdened.magnet_mass_proxy_tonnes


def test_mirror_nozzle_assumptions_validate_inputs():
    """Public mirror/nozzle assumptions should reject invalid inputs early."""
    with pytest.raises(ValueError, match="collector_voltage_target_kv"):
        MirrorNozzleAssumptions(collector_voltage_target_kv=0.0)

    with pytest.raises(ValueError, match="max_plug_field_t"):
        MirrorNozzleAssumptions(max_plug_field_t=-1.0)

    with pytest.raises(ValueError, match="finite number"):
        MirrorNozzleAssumptions(beta=None)


def test_mirror_nozzle_sweep_validates_limit():
    """Limit should be a positive integer when provided."""
    with pytest.raises(ValueError, match="integer"):
        run_mirror_nozzle_sweep(limit=1.5)

    with pytest.raises(ValueError, match="positive"):
        run_mirror_nozzle_sweep(limit=0)


def test_mirror_nozzle_reporting_writes_outputs(tmp_path):
    """CSV and Markdown outputs should include mirror-specific fields."""
    rows = run_mirror_nozzle_sweep(target_screening_net_mw=50.0, limit=4)
    csv_path = tmp_path / "mirror_nozzle.csv"
    markdown_path = tmp_path / "mirror_nozzle.md"

    write_mirror_nozzle_csv(rows, csv_path)
    write_mirror_nozzle_markdown(rows, markdown_path)

    assert "mirror_ratio" in csv_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "LunarFire v0.5 Mirror/Nozzle Output" in markdown
    assert "collector voltage" in markdown.lower()
