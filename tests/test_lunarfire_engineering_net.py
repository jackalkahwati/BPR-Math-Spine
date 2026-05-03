"""Tests for the LunarFire v0.2 engineering net-power budget."""

import pytest

from helionis.engineering_net import (
    EngineeringAssumptions,
    evaluate_engineering_net,
    run_engineering_net_budget,
)
from helionis.engineering_reporting import (
    write_engineering_csv,
    write_engineering_markdown,
)


def test_default_engineering_budget_does_not_close_plant_net():
    """The v0.1 screening-net point should miss after engineering loads."""
    result = evaluate_engineering_net()

    assert result.screening_net_power_mw == pytest.approx(10.0, rel=1e-6)
    assert result.engineering_load_mw > result.screening_net_power_mw
    assert result.plant_net_power_mw < 0.0
    assert result.closes_engineering_net is False
    assert "does not close plant-net power" in result.warnings


def test_engineering_load_budget_adds_up():
    """Plant-net power should equal screening-net minus explicit load terms."""
    result = evaluate_engineering_net()
    load_sum = (
        result.current_drive_mw
        + result.formation_average_mw
        + result.cryogenic_wallplug_mw
        + result.power_conditioning_loss_mw
        + result.thermal_rejection_parasitic_mw
        + result.fixed_balance_of_plant_mw
    )

    assert result.engineering_load_mw == pytest.approx(load_sum)
    assert result.plant_net_power_mw == pytest.approx(
        result.screening_net_power_mw - load_sum
    )
    assert result.radiator_area_m2 > 0.0
    assert result.conversion_waste_mw == pytest.approx(
        result.gross_fusion_mw - result.useful_power_mw
    )
    assert result.rejected_heat_mw > (
        result.bremsstrahlung_loss_mw + result.conversion_waste_mw
    )


def test_engineering_assumptions_validate_ranges():
    """Engineering assumptions should reject invalid public inputs."""
    with pytest.raises(ValueError, match="power_conditioning_loss_fraction"):
        EngineeringAssumptions(power_conditioning_loss_fraction=1.2)
    with pytest.raises(ValueError, match="radiator_temperature_k"):
        EngineeringAssumptions(radiator_temperature_k=0.0)
    with pytest.raises(ValueError, match="radiator_emissivity"):
        EngineeringAssumptions(radiator_emissivity=0.0)


def test_engineering_budget_limit_validation():
    """Engineering budget runner should reject surprising limit values."""
    with pytest.raises(ValueError, match="limit"):
        run_engineering_net_budget(limit=0)


def test_engineering_reporting_writes_outputs(tmp_path):
    """Engineering net-power reports should expose plant-net caveats."""
    results = run_engineering_net_budget(limit=3)
    csv_path = tmp_path / "engineering.csv"
    markdown_path = tmp_path / "engineering.md"

    write_engineering_csv(results, csv_path)
    write_engineering_markdown(results, markdown_path)

    assert "plant_net_power_mw" in csv_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "LunarFire v0.2 Engineering Net-Power Output" in markdown
    assert "Plant net power" in markdown
    assert "Fusion conversion waste heat" in markdown
    assert "Negative plant-net power" in markdown
