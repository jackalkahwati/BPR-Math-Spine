"""Tests for LunarFire v0.4 plant-net geometry re-score."""

import pytest

from helionis.geometry import GeometryFamily
from helionis.geometry_engineering import (
    GeometryEngineeringProfile,
    run_geometry_engineering_rescore,
)
from helionis.geometry_engineering_reporting import (
    write_geometry_engineering_csv,
    write_geometry_engineering_markdown,
)


def test_geometry_engineering_rescore_returns_all_default_geometries():
    """The default re-score should compare FRC, mirror, and spherical torus."""
    rows = run_geometry_engineering_rescore(target_screening_net_mw=50.0)
    families = {row.family for row in rows}

    assert families == {
        GeometryFamily.FRC.value,
        GeometryFamily.MIRROR.value,
        GeometryFamily.SPHERICAL_TORUS.value,
    }
    assert rows == sorted(
        rows,
        key=lambda row: (row.feasible_screening_candidate, row.plant_net_power_mw),
        reverse=True,
    )
    assert rows[-1].feasible_screening_candidate is False
    assert any(row.feasible_screening_candidate is False for row in rows)


def test_geometry_engineering_rows_preserve_caveats():
    """Plant-net geometry comparison should keep caveats in every row."""
    rows = run_geometry_engineering_rescore(target_screening_net_mw=50.0)

    assert all(
        ("plant-net includes" in row.warnings)
        or ("no feasible screening-net candidate" in row.warnings)
        for row in rows
    )
    assert all(
        row.gross_fusion_mw > row.screening_net_power_mw
        for row in rows
        if row.feasible_screening_candidate
    )
    assert any(row.rejection_summary for row in rows if not row.feasible_screening_candidate)


def test_geometry_profile_rejects_invalid_beta():
    """Public geometry profile inputs should fail early."""
    with pytest.raises(ValueError, match="beta"):
        GeometryEngineeringProfile(
            family=GeometryFamily.FRC,
            beta=0.0,
            aspect_ratio=6.0,
            shape_factor=0.85,
            direct_conversion_efficiency=0.78,
            thermal_conversion_efficiency=0.38,
            transport_loss_multiplier=0.15,
            z_eff=1.2,
            dd_side_reaction_fraction=0.03,
            current_drive_fraction_of_gross_fusion=0.02,
            direct_conversion_access=0.95,
            stability_confidence=0.55,
            engineering_simplicity=0.78,
        )


def test_geometry_engineering_reporting_writes_outputs(tmp_path):
    """Geometry re-score reports should state the best current geometry."""
    rows = run_geometry_engineering_rescore(target_screening_net_mw=50.0)
    csv_path = tmp_path / "geometry_engineering.csv"
    markdown_path = tmp_path / "geometry_engineering.md"

    write_geometry_engineering_csv(rows, csv_path)
    write_geometry_engineering_markdown(rows, markdown_path)

    assert "plant_net_power_mw" in csv_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "LunarFire v0.4 Plant-Net Geometry Re-Score Output" in markdown
    assert "Best current geometry" in markdown
