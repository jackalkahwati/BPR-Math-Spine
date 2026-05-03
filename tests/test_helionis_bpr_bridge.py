"""Tests for connecting Helionis screening to BPR Math Spine primitives."""

import pytest

from helionis.bpr_bridge import (
    BPRBridgeInputs,
    bpr_bridge_factors,
    resonance_alignment_factor,
    topological_impedance_match,
)
from helionis.bpr_coupled_v09 import run_bpr_coupled_v09_sweep
from helionis.bpr_coupled_v09_reporting import (
    write_bpr_coupled_v09_csv,
    write_bpr_coupled_v09_markdown,
)


def test_topological_impedance_match_uses_bpr_impedance_behavior():
    """Higher effective winding should reduce the BPR impedance match."""
    low = topological_impedance_match(effective_winding=2.0)
    high = topological_impedance_match(effective_winding=10.0)

    assert 0.0 < high < low <= 1.0


def test_resonance_alignment_factor_is_bounded():
    """BPR resonance alignment should produce a bounded dimensionless factor."""
    factor = resonance_alignment_factor(radius_m=0.8, length_m=8.0)

    assert 0.0 < factor <= 1.0


def test_resonance_alignment_factor_validates_zero_count():
    """Invalid zero-table requests should fail with a clear error."""
    with pytest.raises(ValueError, match="n_zeros"):
        resonance_alignment_factor(radius_m=0.8, length_m=8.0, n_zeros=0)


def test_bpr_bridge_factors_report_source_modules():
    """The bridge should make BPR provenance explicit in each result."""
    factors = bpr_bridge_factors(
        BPRBridgeInputs(
            mirror_ratio=5.0,
            radius_m=0.8,
            length_m=8.0,
            base_transport_multiplier=0.14,
            base_direct_conversion_efficiency=0.84,
        )
    )

    assert factors.transport_multiplier <= 1.0
    assert factors.direct_conversion_multiplier >= 1.0
    assert "bpr.impedance.TopologicalImpedance" in factors.source_modules
    assert "bpr.resonance.load_riemann_zeros" in factors.source_modules


def test_bpr_coupled_v09_sweep_returns_bpr_provenance():
    """v0.9 rows should expose BPR factors and remain sorted by plant-net."""
    rows = run_bpr_coupled_v09_sweep(target_screening_net_mw=50.0, limit=6)

    assert rows
    assert rows == sorted(rows, key=lambda row: row.plant_net_power_mw, reverse=True)
    assert all(row.uses_bpr_math for row in rows)
    assert all(row.bpr_transport_multiplier <= 1.0 for row in rows)
    assert len({row.mirror_aspect_ratio for row in rows}) > 1


def test_bpr_coupled_v09_resonance_changes_with_aspect_ratio():
    """Aspect-ratio changes should flow through the BPR resonance factor."""
    rows = run_bpr_coupled_v09_sweep(
        target_screening_net_mw=50.0,
        mirror_aspect_ratios=(8.0, 12.0),
        limit=None,
    )

    alignments = {row.mirror_aspect_ratio: row.bpr_resonance_alignment for row in rows}

    assert alignments[8.0] != alignments[12.0]


def test_bpr_bridge_rejects_invalid_inputs():
    """Public bridge inputs should fail early on invalid values."""
    with pytest.raises(ValueError, match="mirror_ratio"):
        BPRBridgeInputs(
            mirror_ratio=1.0,
            radius_m=0.8,
            length_m=8.0,
            base_transport_multiplier=0.14,
            base_direct_conversion_efficiency=0.84,
        )


def test_bpr_coupled_v09_reporting_writes_outputs(tmp_path):
    """Reports should state that BPR math is being used."""
    rows = run_bpr_coupled_v09_sweep(target_screening_net_mw=50.0, limit=4)
    csv_path = tmp_path / "bpr_coupled_v09.csv"
    markdown_path = tmp_path / "bpr_coupled_v09.md"

    write_bpr_coupled_v09_csv(rows, csv_path)
    write_bpr_coupled_v09_markdown(rows, markdown_path)

    csv_lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert "bpr_resonance_alignment" in csv_lines[0]
    assert csv_lines[1].split(",")[1] == str(rows[0].plant_net_power_mw)
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "LunarFire v0.9 BPR-Coupled Output" in markdown
    assert "bpr.impedance" in markdown
