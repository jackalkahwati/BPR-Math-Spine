"""Tests for the Modulus Fusion real-time control twin."""

from fractions import Fraction

import pytest

from helionis.bpr_coupled_v09 import run_bpr_coupled_v09_sweep
from helionis.modulus_fusion_control import (
    ModulusFusionControlAssumptions,
    evaluate_modulus_fusion_control,
    exact_state_update,
    numerical_drift_after_exact_updates,
    run_modulus_fusion_control_twin,
)
from helionis.modulus_fusion_reporting import (
    write_modulus_fusion_control_csv,
    write_modulus_fusion_control_markdown,
)


def test_exact_state_update_accumulates_zero_numerical_drift():
    """Exact update arithmetic should not add roundoff drift."""
    drift = numerical_drift_after_exact_updates(
        initial_state=Fraction(1, 3),
        command_delta=Fraction(1, 10_000),
        disturbance_delta=Fraction(-1, 20_000),
        steps=10_000,
    )

    assert drift == Fraction(0, 1)


def test_exact_state_update_is_fraction_exact():
    """The primitive update should preserve exact fractional state."""
    updated = exact_state_update(
        state=Fraction(1, 7),
        command_delta=Fraction(2, 7),
        disturbance_delta=Fraction(-1, 14),
    )

    assert updated == Fraction(5, 14)


def test_exact_state_update_rejects_non_fraction_inputs():
    """Exact update primitives should not silently accept floats."""
    with pytest.raises(TypeError, match="Fraction"):
        exact_state_update(0.1, Fraction(1, 10), Fraction(0, 1))


def test_control_evaluator_uses_v09_geometry_and_zero_numerical_drift():
    """The v1.0 evaluator should consume a v0.9 geometry directly."""
    v09_best = run_bpr_coupled_v09_sweep(target_screening_net_mw=50.0, limit=1)[0]
    result = evaluate_modulus_fusion_control(v09_best)

    assert result.source_plant_net_power_mw == v09_best.plant_net_power_mw
    assert result.uses_bpr_coupled_geometry is True
    assert result.numerical_drift_fraction == 0.0
    assert result.drift_claim == "zero numerical drift in deterministic control math"


def test_control_twin_limit_applies_after_control_ranking():
    """Output limiting should not pre-filter by plant-net ranking."""
    full = run_modulus_fusion_control_twin(target_screening_net_mw=50.0, limit=None)
    limited = run_modulus_fusion_control_twin(target_screening_net_mw=50.0, limit=1)

    assert limited[0] == full[0]


def test_control_twin_coil_command_scales_with_update_period():
    """Slower cadence should increase per-update correction and lower score."""
    fast = run_modulus_fusion_control_twin(
        target_screening_net_mw=50.0,
        assumptions=ModulusFusionControlAssumptions(update_period_ms=0.5),
        limit=1,
    )[0]
    slow = run_modulus_fusion_control_twin(
        target_screening_net_mw=50.0,
        assumptions=ModulusFusionControlAssumptions(update_period_ms=4.0),
        limit=1,
    )[0]

    assert slow.required_field_correction_t > fast.required_field_correction_t
    assert slow.controllability_score < fast.controllability_score


def test_control_twin_flags_uncontrollable_actuator_limit():
    """Actuator limits should affect the controllability classification."""
    result = run_modulus_fusion_control_twin(
        target_screening_net_mw=50.0,
        assumptions=ModulusFusionControlAssumptions(
            update_period_ms=4.0,
            actuator_slew_t_per_ms=0.005,
        ),
        limit=1,
    )[0]

    assert result.controllable is False
    assert result.coil_command_fraction > 1.0


def test_modulus_control_rejects_invalid_assumptions():
    """Invalid timing assumptions should fail early."""
    with pytest.raises(ValueError, match="update_period_ms"):
        ModulusFusionControlAssumptions(update_period_ms=0.0)
    with pytest.raises(ValueError, match="sensor_noise_fraction"):
        ModulusFusionControlAssumptions(sensor_noise_fraction=2.0)
    with pytest.raises(ValueError, match="exact_math_enabled"):
        ModulusFusionControlAssumptions(exact_math_enabled="yes")


def test_control_twin_reports_finite_drift_when_exact_math_disabled():
    """Non-exact mode should not claim zero numerical drift."""
    result = run_modulus_fusion_control_twin(
        target_screening_net_mw=50.0,
        assumptions=ModulusFusionControlAssumptions(exact_math_enabled=False),
        limit=1,
    )[0]

    assert result.numerical_drift_fraction > 0.0
    assert "floating-point" in result.drift_claim


def test_modulus_control_reporting_writes_outputs(tmp_path):
    """CSV and Markdown reports should preserve zero-drift phrasing."""
    rows = run_modulus_fusion_control_twin(target_screening_net_mw=50.0, limit=3)
    csv_path = tmp_path / "modulus_fusion_control.csv"
    markdown_path = tmp_path / "modulus_fusion_control.md"

    write_modulus_fusion_control_csv(rows, csv_path)
    write_modulus_fusion_control_markdown(rows, markdown_path)

    assert "numerical_drift_fraction" in csv_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Modulus Fusion Control Twin" in markdown
    assert "zero numerical drift" in markdown
    assert "not zero plasma motion" in markdown
