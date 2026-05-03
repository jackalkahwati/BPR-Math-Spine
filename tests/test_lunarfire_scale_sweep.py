"""Tests for the LunarFire v0.3 scale sweep."""

import pytest

from helionis.scale_reporting import write_scale_csv, write_scale_markdown
from helionis.scale_sweep import minimum_viable_scale, run_scale_sweep


def test_scale_sweep_preserves_target_order_and_10mw_fails():
    """The scale sweep should show 10 MW remains below plant-net closure."""
    rows = run_scale_sweep(targets_mw=(10.0, 25.0, 50.0))

    assert [row.target_screening_net_mw for row in rows] == [10.0, 25.0, 50.0]
    assert rows[0].plant_net_power_mw < 0.0
    assert rows[0].closes_engineering_net is False


def test_default_scale_sweep_covers_expected_targets_and_none_close():
    """The default contract should cover all product-scale targets explicitly."""
    rows = run_scale_sweep()

    assert [row.target_screening_net_mw for row in rows] == [
        10.0,
        25.0,
        50.0,
        100.0,
        250.0,
    ]
    assert all(row.closes_engineering_net is False for row in rows)


def test_minimum_viable_scale_reports_none_when_no_target_closes():
    """The current assumptions should not force a positive crossing."""
    result = minimum_viable_scale(targets_mw=(10.0, 25.0, 50.0, 100.0, 250.0))

    assert result is None


def test_scale_sweep_rejects_invalid_targets():
    """Scale inputs should reject nonsensical target lists."""
    with pytest.raises(ValueError, match="targets_mw"):
        run_scale_sweep(targets_mw=(0.0,))
    with pytest.raises(ValueError, match="targets_mw"):
        run_scale_sweep(targets_mw=())


def test_scale_reporting_writes_outputs(tmp_path):
    """Scale reports should include the minimum viable scale summary."""
    rows = run_scale_sweep(targets_mw=(100.0, 10.0, 50.0, 25.0))
    csv_path = tmp_path / "scale.csv"
    markdown_path = tmp_path / "scale.md"

    write_scale_csv(rows, csv_path)
    write_scale_markdown(rows, markdown_path)

    assert "target_screening_net_mw" in csv_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "LunarFire v0.3 Minimum Viable Scale Output" in markdown
    assert "No tested target closes" in markdown
    assert markdown.index("| 10 |") < markdown.index("| 100 |")
