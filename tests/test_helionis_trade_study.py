"""Regression tests for the Helionis one-week trade-study MVP."""

import pytest

from helionis.architecture import Scenario, evaluate_scenario, run_trade_study
from helionis.plasma import estimate_reactivity
from helionis.reporting import write_csv, write_markdown_summary
from helionis.reactions import D_HE3, D_T


def test_reactivity_estimate_increases_across_d_he3_operating_window():
    """The screening model should reflect D-He3's high-temperature bias."""
    low = estimate_reactivity(D_HE3.key, 50.0)
    high = estimate_reactivity(D_HE3.key, 100.0)

    assert high > low
    assert high == pytest.approx(2.5e-22, rel=0.25)


def test_d_he3_scenario_has_lower_neutron_fraction_than_dt_reference():
    """D-He3 scenarios should expose lower neutron burden, not hide net-power risk."""
    dhe3 = Scenario(
        name="test_dhe3",
        reaction_key=D_HE3.key,
        temperature_kev=100.0,
        ion_density_m3=3.0e20,
        confinement_s=5.0,
        plasma_volume_m3=100.0,
        direct_conversion_efficiency=0.65,
        thermal_conversion_efficiency=0.38,
        dd_side_reaction_fraction=0.05,
    )
    dt = Scenario(
        name="test_dt",
        reaction_key=D_T.key,
        temperature_kev=15.0,
        ion_density_m3=1.5e20,
        confinement_s=1.0,
        plasma_volume_m3=100.0,
        direct_conversion_efficiency=0.0,
        thermal_conversion_efficiency=0.38,
    )

    dhe3_result = evaluate_scenario(dhe3)
    dt_result = evaluate_scenario(dt)

    assert dhe3_result.neutron_fraction_of_fusion_power < 0.02
    assert dt_result.neutron_fraction_of_fusion_power > 0.75
    assert dhe3_result.shielding_mass_proxy_tonnes < dt_result.shielding_mass_proxy_tonnes
    assert dt_result.useful_power_mw > dt_result.neutron_power_mw * 0.38


def test_trade_study_rows_are_stable_and_reportable():
    """Default trade-study rows should include the investor-facing fields."""
    rows = run_trade_study()
    names = {row.name for row in rows}

    assert "lunar_infrastructure_dhe3" in names
    assert "dt_reference_power_block" in names
    assert all(row.model_label == "order_of_magnitude_trade_study" for row in rows)
    assert all(row.required_volume_for_target_m3 > 0 for row in rows)


def test_out_of_range_temperature_is_warned_not_hidden():
    """Out-of-window reactivity estimates should be visible in result warnings."""
    scenario = Scenario(
        name="too_cold_dhe3",
        reaction_key=D_HE3.key,
        temperature_kev=20.0,
        ion_density_m3=1.0e20,
        confinement_s=1.0,
        plasma_volume_m3=10.0,
        direct_conversion_efficiency=0.60,
        thermal_conversion_efficiency=0.35,
    )

    result = evaluate_scenario(scenario)

    assert "reactivity estimate clamped" in result.warnings


def test_scenario_rejects_invalid_side_reaction_fraction():
    """Side-reaction assumptions are bounded because they drive neutron burden."""
    with pytest.raises(ValueError, match="dd_side_reaction_fraction"):
        Scenario(
            name="invalid",
            reaction_key=D_HE3.key,
            temperature_kev=100.0,
            ion_density_m3=1.0e20,
            confinement_s=1.0,
            plasma_volume_m3=10.0,
            direct_conversion_efficiency=0.60,
            thermal_conversion_efficiency=0.35,
            dd_side_reaction_fraction=-0.1,
        )


def test_reporting_writes_csv_and_markdown(tmp_path):
    """The week-one deliverable should write reviewable report files."""
    results = run_trade_study()
    csv_path = tmp_path / "trade_study.csv"
    markdown_path = tmp_path / "trade_study.md"

    write_csv(results, csv_path)
    write_markdown_summary(results, markdown_path)

    assert "lunar_infrastructure_dhe3" in csv_path.read_text(encoding="utf-8")
    assert "Helionis D-He3 Trade Study Output" in markdown_path.read_text(
        encoding="utf-8"
    )
