#!/usr/bin/env python3
"""Generate LunarFire v1.2 thermal packaging recovery outputs.

Usage:
    python3 scripts/lunarfire_thermal_recovery_v12.py --output-dir data/helionis
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helionis.thermal_recovery_v12 import run_thermal_recovery_v12
from helionis.thermal_recovery_v12_reporting import (
    plot_thermal_recovery_v12,
    write_thermal_recovery_v12_csv,
    write_thermal_recovery_v12_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LunarFire v1.2 thermal packaging recovery outputs."
    )
    parser.add_argument(
        "--output-dir",
        default="data/helionis",
        help="Directory for CSV, Markdown, and plot outputs.",
    )
    parser.add_argument(
        "--target-mw",
        type=float,
        default=50.0,
        help="Screening-net target power in MW.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=24,
        help="Number of top-ranked rows to write.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip PNG plot generation.",
    )
    args = parser.parse_args()
    if not math.isfinite(args.target_mw) or args.target_mw <= 0:
        parser.error("--target-mw must be positive and finite")
    if args.limit <= 0:
        parser.error("--limit must be positive")

    summary_results = run_thermal_recovery_v12(
        target_screening_net_mw=args.target_mw,
        limit=None,
    )
    results = run_thermal_recovery_v12(
        target_screening_net_mw=args.target_mw,
        limit=args.limit,
    )
    if not results or not summary_results:
        parser.error("no thermal recovery rows found")

    output_dir = Path(args.output_dir)
    csv_path = output_dir / "lunarfire_thermal_recovery_v1_2.csv"
    markdown_path = output_dir / "lunarfire_thermal_recovery_v1_2.md"
    plot_path = output_dir / "lunarfire_thermal_recovery_v1_2.png"

    write_thermal_recovery_v12_csv(results, csv_path)
    write_thermal_recovery_v12_markdown(
        results,
        markdown_path,
        summary_results=summary_results,
    )
    if not args.no_plot:
        plot_thermal_recovery_v12(results, plot_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    if not args.no_plot:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
