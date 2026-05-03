"""Tests for the LunarFire 10 MW FRC reference-design sweep."""

import pytest

from helionis.reference_design import (
    ReferenceDesignTarget,
    best_reference_design,
    solve_reference_design,
)
from helionis.reference_reporting import write_reference_csv, write_reference_markdown


def test_reference_solver_finds_10mw_frc_candidate():
    """The default sweep should close a 10 MW screening-net FRC candidate."""
    best = best_reference_design()

    assert best.screening_net_power_mw == pytest.approx(10.0, rel=1e-6)
    assert best.separatrix_radius_m <= 3.0
    assert best.length_m <= 20.0
    assert best.required_field_t <= 12.0
    assert best.neutron_wall_load_mw_m2 <= 0.10
    assert best.bremsstrahlung_loss_mw > best.screening_net_power_mw
    assert "optimistic transport/direct-conversion" in best.warnings


def test_reference_results_are_sorted_by_design_objective():
    """Candidate list should be ranked by compact design objective."""
    results = solve_reference_design(limit=5)
    objectives = [result.objective_score for result in results]

    assert len(results) == 5
    assert objectives == sorted(objectives)


def test_reference_target_rejects_invalid_power():
    """Reference target validation should reject nonsensical design goals."""
    with pytest.raises(ValueError, match="target_screening_net_mw"):
        ReferenceDesignTarget(target_screening_net_mw=0.0)


def test_reference_solver_rejects_invalid_limit_and_sweep_values():
    """Solver inputs should fail early instead of silently returning bad slices."""
    with pytest.raises(ValueError, match="limit"):
        solve_reference_design(limit=0)
    with pytest.raises(ValueError, match="temperatures_kev"):
        solve_reference_design(temperatures_kev=(0.0,))


def test_reference_target_rejects_invalid_constraints():
    """Maximum design constraints should be positive."""
    with pytest.raises(ValueError, match="max_field_t"):
        ReferenceDesignTarget(max_field_t=0.0)


def test_reference_reporting_writes_csv_and_markdown(tmp_path):
    """The 10 MW design sweep should produce shareable output files."""
    results = solve_reference_design(limit=3)
    csv_path = tmp_path / "reference.csv"
    markdown_path = tmp_path / "reference.md"

    write_reference_csv(results, csv_path)
    write_reference_markdown(results, markdown_path)

    assert "lunarfire_frc_10mw_v0_1" in csv_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "LunarFire 10 MW FRC Reference Design Output" in markdown
    assert "Screening net power" in markdown
    assert "Bremsstrahlung loss" in markdown
    assert "excludes recirculating power" in markdown
