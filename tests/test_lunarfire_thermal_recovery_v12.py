"""Tests for v1.2 thermal-packaging recovery analysis."""

import pytest

from helionis.thermal_recovery_v12 import (
    ThermalRecoveryV12Assumptions,
    evaluate_thermal_recovery,
    run_thermal_recovery_v12,
)
from helionis.thermal_recovery_v12_reporting import (
    write_thermal_recovery_v12_csv,
    write_thermal_recovery_v12_markdown,
)
from helionis.cad_envelope_v11 import run_cad_envelope_v11


def test_thermal_recovery_improves_span_and_plant_net_when_direct_recovery_added():
    """Direct conversion recovery should reduce heat and improve plant-net."""
    base = run_cad_envelope_v11(target_screening_net_mw=50.0, limit=1)[0]
    recovered = evaluate_thermal_recovery(
        base,
        ThermalRecoveryV12Assumptions(
            direct_heat_recovery_fraction=0.02,
            radiator_temperature_k=1200.0,
            topology_packing_factor=2.0,
        ),
    )

    assert recovered.recovered_electric_power_mw > 0.0
    assert recovered.plant_net_power_mw > base.source_plant_net_power_mw
    assert recovered.adjusted_radiator_area_m2 < base.radiator_area_m2
    assert recovered.adjusted_wing_span_each_m < base.radiator_wing_span_each_m


def test_thermal_recovery_can_find_cad_ready_recipe():
    """The sweep should identify a closing thermal-packaging recipe if present."""
    rows = run_thermal_recovery_v12(target_screening_net_mw=50.0, limit=12)

    assert rows
    assert rows == sorted(
        rows,
        key=lambda row: (
            not row.cad_ready,
            row.recovery_aggressiveness,
            -row.cad_readiness_score,
        ),
    )
    assert any(row.cad_ready for row in rows)
    assert rows[0].direct_heat_recovery_fraction == pytest.approx(0.01)
    assert rows[0].radiator_temperature_k == pytest.approx(1000.0)
    assert rows[0].topology_packing_factor == pytest.approx(1.0)


def test_thermal_recovery_temperature_only_does_not_close_power():
    """Hotter radiators alone improve packaging but should not fix plant-net."""
    base = run_cad_envelope_v11(target_screening_net_mw=50.0, limit=1)[0]
    thermal_only = evaluate_thermal_recovery(
        base,
        ThermalRecoveryV12Assumptions(
            direct_heat_recovery_fraction=0.0,
            radiator_temperature_k=1400.0,
            topology_packing_factor=4.0,
        ),
    )

    assert thermal_only.adjusted_wing_span_each_m < base.radiator_wing_span_each_m
    assert thermal_only.plant_net_power_mw == pytest.approx(
        base.source_plant_net_power_mw
    )
    assert thermal_only.closes_engineering_net is False


def test_thermal_recovery_rejects_invalid_assumptions():
    """Recovery assumptions should validate early."""
    with pytest.raises(ValueError, match="radiator_temperature_k"):
        ThermalRecoveryV12Assumptions(radiator_temperature_k=0.0)
    with pytest.raises(ValueError, match="direct_heat_recovery_fraction"):
        ThermalRecoveryV12Assumptions(direct_heat_recovery_fraction=1.2)
    with pytest.raises(ValueError, match="direct_heat_recovery_fraction"):
        ThermalRecoveryV12Assumptions(direct_heat_recovery_fraction=1.0)


def test_thermal_recovery_does_not_clear_control_or_radius_blockers():
    """Thermal recovery can only clear power and radiator blockers."""
    blocked = next(
        row
        for row in run_cad_envelope_v11(target_screening_net_mw=50.0, limit=None)
        if "control row is not controllable" in row.blockers
    )
    recovered = evaluate_thermal_recovery(
        blocked,
        ThermalRecoveryV12Assumptions(
            direct_heat_recovery_fraction=0.04,
            radiator_temperature_k=1400.0,
            topology_packing_factor=4.0,
        ),
    )

    assert recovered.cad_ready is False
    assert "control row is not controllable" in recovered.blockers
    assert "radiator wing span remains too large" not in recovered.blockers


def test_thermal_recovery_reporting_writes_outputs(tmp_path):
    """Reports should summarize whether thermal recovery finds CAD-ready rows."""
    rows = run_thermal_recovery_v12(target_screening_net_mw=50.0, limit=6)
    full_rows = run_thermal_recovery_v12(target_screening_net_mw=50.0, limit=None)
    csv_path = tmp_path / "thermal_recovery.csv"
    markdown_path = tmp_path / "thermal_recovery.md"

    write_thermal_recovery_v12_csv(rows, csv_path)
    write_thermal_recovery_v12_markdown(rows, markdown_path, summary_results=full_rows)

    assert "adjusted_wing_span_each_m" in csv_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Thermal Packaging Recovery" in markdown
    assert "direct conversion recovery" in markdown
    assert "Minimum-assumption" in markdown
    best_full_score = max(row.cad_readiness_score for row in full_rows if row.cad_ready)
    assert f"{best_full_score:.3f}" in markdown
