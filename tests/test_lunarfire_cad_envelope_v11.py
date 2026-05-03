"""Tests for the v1.1 control-constrained CAD envelope."""

from dataclasses import replace

import pytest

from helionis.cad_envelope_v11 import (
    CADEnvelopeV11Assumptions,
    evaluate_cad_envelope,
    run_cad_envelope_v11,
)
from helionis.cad_envelope_v11_reporting import (
    write_cad_envelope_v11_csv,
    write_cad_envelope_v11_markdown,
)
from helionis.modulus_fusion_control import run_modulus_fusion_control_twin


def test_cad_envelope_uses_control_constrained_geometry():
    """The CAD envelope should consume Modulus Fusion control rows."""
    control = run_modulus_fusion_control_twin(target_screening_net_mw=50.0, limit=1)[0]
    envelope = evaluate_cad_envelope(control)

    assert envelope.uses_modulus_control is True
    assert envelope.source_controllability_score == control.controllability_score
    assert envelope.plasma_radius_m == control.separatrix_radius_m
    assert envelope.machine_length_m > control.plasma_length_m
    assert envelope.outer_radius_m > envelope.plasma_radius_m


def test_cad_envelope_collector_margin_expands_area():
    """Collector envelope should include explicit CAD margin."""
    control = run_modulus_fusion_control_twin(target_screening_net_mw=50.0, limit=1)[0]
    envelope = evaluate_cad_envelope(
        control,
        assumptions=CADEnvelopeV11Assumptions(collector_area_margin=0.25),
    )

    assert envelope.collector_surface_area_m2 == pytest.approx(
        control.collector_area_m2 * 1.25
    )


def test_cad_envelope_flags_packaging_constraint_failure():
    """Tight packaging constraints should mark a row as not CAD ready."""
    control = run_modulus_fusion_control_twin(target_screening_net_mw=50.0, limit=1)[0]
    envelope = evaluate_cad_envelope(
        control,
        assumptions=CADEnvelopeV11Assumptions(max_machine_length_m=5.0),
    )

    assert envelope.cad_ready is False
    assert "machine length exceeds" in envelope.warnings


def test_cad_envelope_requires_engineering_net_closure_for_readiness():
    """Relaxed packaging should not make a negative-net plant CAD-ready."""
    control = run_modulus_fusion_control_twin(target_screening_net_mw=50.0, limit=1)[0]
    envelope = evaluate_cad_envelope(
        control,
        assumptions=CADEnvelopeV11Assumptions(
            max_machine_length_m=200.0,
            max_outer_radius_m=100.0,
            max_radiator_area_m2=100_000.0,
            max_radiator_wing_span_each_m=2_000.0,
        ),
    )

    assert control.source_plant_net_power_mw < 0.0
    assert envelope.cad_ready is False
    assert "source plant is net-negative" in envelope.blockers


def test_cad_envelope_rejects_invalid_control_rows():
    """Handcrafted invalid control rows should not score as CAD-ready."""
    control = run_modulus_fusion_control_twin(target_screening_net_mw=50.0, limit=1)[0]
    invalid = replace(control, controllability_score=1.2)

    with pytest.raises(ValueError, match="controllability_score"):
        evaluate_cad_envelope(invalid)

    inconsistent = replace(control, source_closes_engineering_net=True)
    with pytest.raises(ValueError, match="inconsistent"):
        evaluate_cad_envelope(inconsistent)


def test_run_cad_envelope_v11_returns_ranked_rows():
    """v1.1 should rank rows by CAD readiness and score."""
    rows = run_cad_envelope_v11(target_screening_net_mw=50.0, limit=5)

    assert rows
    assert rows == sorted(
        rows,
        key=lambda row: (row.cad_ready, row.cad_readiness_score),
        reverse=True,
    )
    assert all(row.machine_length_m > 0.0 for row in rows)


def test_cad_envelope_rejects_invalid_assumptions():
    """CAD assumptions should fail early for invalid inputs."""
    with pytest.raises(ValueError, match="collector_area_margin"):
        CADEnvelopeV11Assumptions(collector_area_margin=-0.1)


def test_cad_envelope_reporting_writes_outputs(tmp_path):
    """Reports should describe the parametric CAD envelope."""
    rows = run_cad_envelope_v11(target_screening_net_mw=50.0, limit=3)
    csv_path = tmp_path / "cad_envelope.csv"
    markdown_path = tmp_path / "cad_envelope.md"

    write_cad_envelope_v11_csv(rows, csv_path)
    write_cad_envelope_v11_markdown(rows, markdown_path)

    assert "machine_length_m" in csv_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Parametric CAD Envelope" in markdown
    assert "control-constrained" in markdown
    assert "Readiness blockers" in markdown
    assert "radiator wing span exceeds CAD envelope limit" in markdown
