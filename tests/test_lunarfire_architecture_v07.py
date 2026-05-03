"""Tests for LunarFire v0.7 same-assumption architecture comparison."""

import pytest

from helionis.architecture_comparison_v07 import (
    ArchitectureV07Assumptions,
    mirror_leakage_transport_multiplier,
    run_architecture_v07_comparison,
)
from helionis.architecture_v07_reporting import (
    write_architecture_v07_csv,
    write_architecture_v07_markdown,
)


def test_mirror_leakage_responds_to_mirror_ratio_and_scattering():
    """Mirror leakage should improve with higher ratio and slower scattering."""
    low_ratio = mirror_leakage_transport_multiplier(
        mirror_ratio=3.0,
        confinement_s=20.0,
        pitch_angle_scattering_s=80.0,
    )
    high_ratio = mirror_leakage_transport_multiplier(
        mirror_ratio=6.0,
        confinement_s=20.0,
        pitch_angle_scattering_s=80.0,
    )
    slow_scattering = mirror_leakage_transport_multiplier(
        mirror_ratio=3.0,
        confinement_s=20.0,
        pitch_angle_scattering_s=160.0,
    )

    assert high_ratio < low_ratio
    assert slow_scattering < low_ratio
    assert low_ratio > 0.0


def test_architecture_v07_comparison_returns_frc_and_mirror():
    """The same-assumption comparison should include both candidate families."""
    rows = run_architecture_v07_comparison(target_screening_net_mw=50.0)
    families = {row.family for row in rows}

    assert families == {"frc", "mirror_nozzle"}
    assert rows == sorted(rows, key=lambda row: row.plant_net_power_mw, reverse=True)
    assert all(row.target_screening_net_mw == 50.0 for row in rows)
    assert all(row.thermal_conversion_efficiency == pytest.approx(0.35) for row in rows)


def test_architecture_v07_rows_keep_accounting_invariants():
    """Plant-net and magnet accounting should stay internally consistent."""
    rows = run_architecture_v07_comparison(target_screening_net_mw=50.0)

    for row in rows:
        assert row.plant_net_power_mw == pytest.approx(
            row.screening_net_power_mw - row.engineering_load_mw
        )
        assert row.effective_magnet_mass_proxy_tonnes == pytest.approx(
            row.magnet_mass_proxy_tonnes + row.plug_coil_mass_proxy_tonnes
        )
        if row.family == "mirror_nozzle":
            assert row.collector_nozzle_load_mw > 0.0
            assert "collector/nozzle auxiliary load included" in row.warnings


def test_architecture_v07_assumptions_validate_inputs():
    """Public v0.7 assumptions should reject invalid values early."""
    with pytest.raises(ValueError, match="pitch_angle_scattering_s"):
        ArchitectureV07Assumptions(pitch_angle_scattering_s=0.0)

    with pytest.raises(ValueError, match="finite number"):
        ArchitectureV07Assumptions(beta_frc=None)

    with pytest.raises(ValueError, match="max_direct_conversion_efficiency"):
        ArchitectureV07Assumptions(
            frc_direct_conversion_efficiency=0.80,
            max_direct_conversion_efficiency=0.70,
        )


def test_architecture_v07_reporting_writes_outputs(tmp_path):
    """Reports should explicitly be same-assumption architecture comparisons."""
    rows = run_architecture_v07_comparison(target_screening_net_mw=50.0)
    csv_path = tmp_path / "architecture_v07.csv"
    markdown_path = tmp_path / "architecture_v07.md"

    write_architecture_v07_csv(rows, csv_path)
    write_architecture_v07_markdown(rows, markdown_path)

    assert "family" in csv_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "LunarFire v0.7 Shared-Grid Architecture Output" in markdown
    assert "shared-grid" in markdown.lower()
    assert "architecture-specific" in markdown.lower()
    assert "mirror_nozzle" in markdown
    assert "frc" in markdown
