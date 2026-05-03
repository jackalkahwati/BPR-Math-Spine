"""Tests for Helionis zero-shot geometry downselect math."""

import pytest

from helionis.architecture import DEFAULT_SCENARIOS
from helionis.geometry import (
    GeometryCandidate,
    GeometryFamily,
    build_geometry_candidates,
    evaluate_geometry_candidate,
    magnetic_field_for_beta,
    plasma_pressure,
    rank_geometry_candidates,
    run_geometry_downselect,
)
from helionis.fields import axisymmetric_frc_field_map
from helionis.geometry_reporting import write_geometry_csv, write_geometry_markdown


def test_pressure_and_beta_field_are_consistent():
    """The 0D field solver should recover the requested beta relationship."""
    scenario = DEFAULT_SCENARIOS[0]
    pressure = plasma_pressure(scenario)
    field_t = magnetic_field_for_beta(pressure, beta=0.85)

    result = evaluate_geometry_candidate(
        build_geometry_candidates(scenario)[0],
        scenario,
    )

    assert pressure > 0
    assert field_t > 0
    assert result.beta == pytest.approx(0.85)
    assert result.required_field_t == pytest.approx(field_t)


def test_frc_ranks_first_for_default_dhe3_scenarios():
    """Zero-shot scoring should prefer FRC for Helionis' compact D-He3 wedge."""
    dhe3_scenarios = [
        scenario for scenario in DEFAULT_SCENARIOS if scenario.reaction_key == "d_he3"
    ]

    for scenario in dhe3_scenarios:
        ranked = rank_geometry_candidates(scenario)
        assert ranked[0].family == GeometryFamily.FRC.value
        assert ranked[0].total_score > ranked[-1].total_score


def test_geometry_candidates_share_scenario_volume_scale():
    """Candidate geometries should preserve scenario volume for fair comparison."""
    scenario = DEFAULT_SCENARIOS[1]
    candidates = build_geometry_candidates(scenario)
    volumes = [candidate.volume_m3 for candidate in candidates]

    assert {candidate.family for candidate in candidates} == {
        GeometryFamily.FRC,
        GeometryFamily.MIRROR,
        GeometryFamily.SPHERICAL_TORUS,
    }
    for volume in volumes:
        assert volume == pytest.approx(scenario.plasma_volume_m3, rel=1e-12)


def test_magnetic_mass_penalty_affects_geometry_score():
    """An otherwise similar high-field candidate should be penalized."""
    scenario = DEFAULT_SCENARIOS[0]
    efficient_field = GeometryCandidate(
        name="efficient_field",
        family=GeometryFamily.FRC,
        volume_m3=scenario.plasma_volume_m3,
        major_radius_m=5.0,
        minor_radius_m=1.7,
        length_m=10.0,
        beta_target=0.85,
        shape_factor=0.85,
        compactness_weight=0.85,
        direct_conversion_access=0.95,
        stability_confidence=0.55,
        engineering_simplicity=0.78,
    )
    high_field = GeometryCandidate(
        name="high_field",
        family=GeometryFamily.FRC,
        volume_m3=scenario.plasma_volume_m3,
        major_radius_m=5.0,
        minor_radius_m=1.7,
        length_m=10.0,
        beta_target=0.20,
        shape_factor=0.85,
        compactness_weight=0.85,
        direct_conversion_access=0.95,
        stability_confidence=0.55,
        engineering_simplicity=0.78,
    )

    efficient_result = evaluate_geometry_candidate(efficient_field, scenario)
    high_field_result = evaluate_geometry_candidate(high_field, scenario)

    assert high_field_result.required_field_t > efficient_result.required_field_t
    assert high_field_result.magnet_mass_proxy_tonnes > (
        efficient_result.magnet_mass_proxy_tonnes
    )
    assert high_field_result.total_score < efficient_result.total_score


def test_spherical_torus_uses_package_envelope_not_circumference():
    """The torus package length should be the outer diameter for scoring."""
    scenario = DEFAULT_SCENARIOS[0]
    torus = [
        candidate
        for candidate in build_geometry_candidates(scenario)
        if candidate.family == GeometryFamily.SPHERICAL_TORUS
    ][0]
    result = evaluate_geometry_candidate(torus, scenario)

    assert torus.length_m == pytest.approx(
        2.0 * (torus.major_radius_m + torus.minor_radius_m)
    )
    assert result.surface_area_m2 > 4.0 * 3.14159**2 * torus.major_radius_m


def test_invalid_geometry_candidate_inputs_are_rejected():
    """Public geometry inputs should fail early with clear validation."""
    with pytest.raises(ValueError, match="shape_factor"):
        GeometryCandidate(
            name="bad_shape",
            family=GeometryFamily.FRC,
            volume_m3=10.0,
            major_radius_m=2.0,
            minor_radius_m=1.0,
            length_m=4.0,
            beta_target=0.8,
            shape_factor=0.0,
            compactness_weight=0.5,
            direct_conversion_access=0.5,
            stability_confidence=0.5,
            engineering_simplicity=0.5,
        )


def test_downselect_filters_to_dhe3_scenarios_and_reports(tmp_path):
    """Geometry outputs should focus on D-He3 and write shareable files."""
    rows = run_geometry_downselect(DEFAULT_SCENARIOS)
    csv_path = tmp_path / "geometry.csv"
    markdown_path = tmp_path / "geometry.md"

    write_geometry_csv(rows, csv_path)
    write_geometry_markdown(rows, markdown_path)

    assert len(rows) == 12
    assert all("dt_reference" not in row.scenario_name for row in rows)
    assert "frc_linear_plasmoid" in csv_path.read_text(encoding="utf-8")
    assert "Helionis Geometry Downselect Output" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_axisymmetric_frc_field_map_shapes_and_midplane_sign():
    """The 2D FRC prototype should return finite field arrays on an R-Z grid."""
    field_map = axisymmetric_frc_field_map(
        separatrix_radius_m=1.5,
        length_m=8.0,
        axial_field_t=5.0,
        radial_points=32,
        axial_points=48,
    )

    assert field_map.r_grid.shape == (48, 32)
    assert field_map.z_grid.shape == (48, 32)
    assert field_map.b_z_t.shape == (48, 32)
    assert field_map.flux_webers.shape == (48, 32)
    assert field_map.b_z_t[24, 0] < 0
    assert field_map.b_z_t[24, -1] > 0
