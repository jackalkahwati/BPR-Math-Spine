"""Tests for v1.3 collector/nozzle heat-channel proof point."""

from dataclasses import replace

import pytest

from helionis.cad_envelope_v11 import run_cad_envelope_v11
from helionis.collector_nozzle_thermal_v13 import (
    CollectorNozzleThermalV13Assumptions,
    heat_channel_split,
    run_collector_nozzle_thermal_v13,
    evaluate_collector_nozzle_thermal,
)
from helionis.collector_nozzle_thermal_v13_reporting import (
    write_collector_nozzle_thermal_v13_csv,
    write_collector_nozzle_thermal_v13_markdown,
)


def test_heat_channel_split_conserves_rejected_heat():
    """The channel split should preserve the total heat budget."""
    split = heat_channel_split(
        rejected_heat_mw=700.0,
        assumptions=CollectorNozzleThermalV13Assumptions(),
    )

    assert split.total_heat_mw == pytest.approx(700.0)
    assert (
        split.bremsstrahlung_core_mw
        + split.transport_wall_mw
        + split.collector_waste_mw
        + split.nozzle_waste_mw
        + split.power_conditioning_heat_mw
        + split.unallocated_heat_mw
    ) == pytest.approx(split.total_heat_mw)
    assert split.recoverable_heat_mw < split.total_heat_mw
    assert split.unrecoverable_core_heat_mw > split.recoverable_heat_mw
    assert split.bremsstrahlung_core_mw > 0.0


def test_collector_nozzle_capture_is_channel_limited():
    """Recovered electric power should be bounded by recoverable channels."""
    envelope = run_cad_envelope_v11(target_screening_net_mw=50.0, limit=1)[0]
    row = evaluate_collector_nozzle_thermal(
        envelope,
        CollectorNozzleThermalV13Assumptions(
            collector_capture_efficiency=0.20,
            nozzle_capture_efficiency=0.10,
            radiator_temperature_k=1000.0,
        ),
    )

    assert row.recovered_electric_power_mw <= row.recoverable_channel_heat_mw
    assert row.recovered_electric_power_mw > 0.0
    assert row.adjusted_rejected_heat_mw < row.inferred_rejected_heat_mw
    assert row.adjusted_rejected_heat_mw == pytest.approx(
        row.bremsstrahlung_core_mw
        + row.transport_wall_mw
        + row.unallocated_heat_mw
        + row.collector_waste_mw * (1.0 - row.collector_capture_efficiency)
        + row.nozzle_waste_mw * (1.0 - row.nozzle_capture_efficiency)
        + row.power_conditioning_heat_mw
        * (1.0 - row.power_conditioning_capture_efficiency)
    )
    assert row.bremsstrahlung_core_mw > row.collector_waste_mw


def test_higher_collector_capture_does_not_lower_readiness_score():
    """More channel capture should increase recovered power and not reduce score."""
    envelope = run_cad_envelope_v11(target_screening_net_mw=50.0, limit=1)[0]
    weak = evaluate_collector_nozzle_thermal(
        envelope,
        CollectorNozzleThermalV13Assumptions(
            collector_capture_efficiency=0.06,
            radiator_temperature_k=1000.0,
        ),
    )
    stronger = evaluate_collector_nozzle_thermal(
        envelope,
        CollectorNozzleThermalV13Assumptions(
            collector_capture_efficiency=0.12,
            radiator_temperature_k=1000.0,
        ),
    )

    assert stronger.recovered_electric_power_mw > weak.recovered_electric_power_mw
    assert stronger.cad_readiness_score >= weak.cad_readiness_score


def test_temperature_only_cannot_close_plant_net():
    """Hot radiators alone cannot solve a negative plant-net row."""
    envelope = run_cad_envelope_v11(target_screening_net_mw=50.0, limit=1)[0]
    row = evaluate_collector_nozzle_thermal(
        envelope,
        CollectorNozzleThermalV13Assumptions(
            collector_capture_efficiency=0.0,
            nozzle_capture_efficiency=0.0,
            radiator_temperature_k=1400.0,
            topology_packing_factor=4.0,
        ),
    )

    assert row.plant_net_power_mw == pytest.approx(envelope.source_plant_net_power_mw)
    assert row.closes_engineering_net is False


def test_collector_nozzle_sweep_finds_channel_specific_cad_ready_case():
    """The default sweep should find a CAD-ready channel-specific recipe."""
    rows = run_collector_nozzle_thermal_v13(target_screening_net_mw=50.0, limit=12)

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
    assert rows[0].collector_capture_efficiency > 0.0
    assert rows[0].radiator_temperature_k >= 1000.0


def test_collector_nozzle_does_not_clear_control_blockers():
    """Collector/nozzle thermal fixes cannot clear upstream control failures."""
    blocked = next(
        row
        for row in run_cad_envelope_v11(target_screening_net_mw=50.0, limit=None)
        if "control row is not controllable" in row.blockers
    )
    recovered = evaluate_collector_nozzle_thermal(
        blocked,
        CollectorNozzleThermalV13Assumptions(
            collector_capture_efficiency=0.20,
            nozzle_capture_efficiency=0.10,
            radiator_temperature_k=1400.0,
            topology_packing_factor=4.0,
        ),
    )

    assert recovered.cad_ready is False
    assert "control row is not controllable" in recovered.blockers


def test_collector_nozzle_requires_modulus_control_source():
    """A row without Modulus control provenance cannot become CAD-ready."""
    envelope = run_cad_envelope_v11(target_screening_net_mw=50.0, limit=1)[0]
    invalid_source = replace(envelope, uses_modulus_control=False)
    recovered = evaluate_collector_nozzle_thermal(
        invalid_source,
        CollectorNozzleThermalV13Assumptions(
            collector_capture_efficiency=0.20,
            radiator_temperature_k=1400.0,
            topology_packing_factor=4.0,
        ),
    )

    assert recovered.cad_ready is False
    assert "source row does not use Modulus control" in recovered.blockers


def test_collector_nozzle_rejects_invalid_assumptions():
    """Invalid heat splits and capture efficiencies should fail early."""
    with pytest.raises(ValueError, match="collector_heat_fraction"):
        CollectorNozzleThermalV13Assumptions(collector_heat_fraction=-0.1)
    with pytest.raises(ValueError, match="heat channel fractions"):
        CollectorNozzleThermalV13Assumptions(
            bremsstrahlung_heat_fraction=0.7,
            transport_heat_fraction=0.3,
            collector_heat_fraction=0.2,
        )
    with pytest.raises(ValueError, match="collector_capture_efficiency"):
        CollectorNozzleThermalV13Assumptions(collector_capture_efficiency=1.0)
    with pytest.raises(ValueError, match="radiator_temperature_k"):
        CollectorNozzleThermalV13Assumptions(radiator_temperature_k=100_000.0)
    with pytest.raises(ValueError, match="collector_capture_efficiencies"):
        run_collector_nozzle_thermal_v13(collector_capture_efficiencies=(0.31,))


def test_collector_nozzle_reporting_writes_outputs(tmp_path):
    """Reports should describe channel-level collector/nozzle recovery."""
    rows = run_collector_nozzle_thermal_v13(target_screening_net_mw=50.0, limit=6)
    full_rows = run_collector_nozzle_thermal_v13(
        target_screening_net_mw=50.0, limit=None
    )
    csv_path = tmp_path / "collector_nozzle.csv"
    markdown_path = tmp_path / "collector_nozzle.md"

    write_collector_nozzle_thermal_v13_csv(rows, csv_path)
    write_collector_nozzle_thermal_v13_markdown(
        rows,
        markdown_path,
        summary_results=full_rows,
    )

    assert "collector_waste_mw" in csv_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Collector/Nozzle Thermal Architecture" in markdown
    assert "bremsstrahlung" in markdown
    assert "recoverable channel" in markdown
    assert "heuristic-aggressiveness" in markdown
