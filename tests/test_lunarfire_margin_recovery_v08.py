"""Tests for LunarFire v0.8 mirror/nozzle margin recovery."""

import pytest

from helionis.margin_recovery_v08 import (
    MarginRecoveryV08Assumptions,
    minimum_recovery_recipe,
    recovery_aggressiveness_score,
    run_margin_recovery_v08_sweep,
)
from helionis.margin_recovery_v08_reporting import (
    write_margin_recovery_v08_csv,
    write_margin_recovery_v08_markdown,
)


def test_recovery_aggressiveness_increases_with_more_optimistic_assumptions():
    """The recipe score should penalize harder-to-believe improvements."""
    conservative = MarginRecoveryV08Assumptions(
        pitch_angle_scattering_s=80.0,
        mirror_stabilization_factor=1.0,
        direct_conversion_cap=0.86,
        plug_coil_mass_coefficient_tonnes_per_t2=0.20,
    )
    aggressive = MarginRecoveryV08Assumptions(
        pitch_angle_scattering_s=320.0,
        mirror_stabilization_factor=4.0,
        direct_conversion_cap=0.92,
        plug_coil_mass_coefficient_tonnes_per_t2=0.05,
    )

    assert recovery_aggressiveness_score(aggressive) > recovery_aggressiveness_score(
        conservative
    )


def test_margin_recovery_sweep_returns_sorted_rows_with_closure():
    """Default v0.8 sweep should identify at least one plant-net closing recipe."""
    rows = run_margin_recovery_v08_sweep(target_screening_net_mw=50.0, limit=12)

    assert rows
    assert any(row.closes_engineering_net for row in rows)
    assert rows == sorted(
        rows,
        key=lambda row: (
            not row.closes_engineering_net,
            row.aggressiveness_score,
            -row.plant_net_power_mw,
        ),
    )


def test_minimum_recovery_recipe_returns_lowest_aggressive_closing_case():
    """The selected recipe should be the least aggressive closing row."""
    rows = run_margin_recovery_v08_sweep(target_screening_net_mw=50.0, limit=None)
    recipe = minimum_recovery_recipe(target_screening_net_mw=50.0)
    closing_rows = [row for row in rows if row.closes_engineering_net]

    assert recipe is not None
    assert recipe.closes_engineering_net
    assert recipe.direct_conversion_cap == pytest.approx(0.88)
    assert recipe.collector_match_bonus == pytest.approx(0.30)
    assert recipe.is_direct_conversion_cap_limited
    assert recipe.aggressiveness_score == pytest.approx(
        min(row.aggressiveness_score for row in closing_rows)
    )


def test_margin_recovery_assumptions_validate_inputs():
    """Public v0.8 assumptions should reject invalid recovery knobs."""
    with pytest.raises(ValueError, match="direct_conversion_cap"):
        MarginRecoveryV08Assumptions(direct_conversion_cap=1.2)

    with pytest.raises(ValueError, match="finite number"):
        MarginRecoveryV08Assumptions(pitch_angle_scattering_s=None)


def test_margin_recovery_reporting_writes_outputs(tmp_path):
    """Reports should include the minimum recovery recipe and caveats."""
    rows = run_margin_recovery_v08_sweep(target_screening_net_mw=50.0, limit=8)
    csv_path = tmp_path / "margin_recovery_v08.csv"
    markdown_path = tmp_path / "margin_recovery_v08.md"

    write_margin_recovery_v08_csv(rows, csv_path)
    write_margin_recovery_v08_markdown(rows, markdown_path)

    assert "aggressiveness_score" in csv_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "LunarFire v0.8 Margin Recovery Output" in markdown
    assert "minimum recovery recipe" in markdown.lower()
