#!/usr/bin/env python3
"""Generate the LunarFire / Modulus Fusion v1.0 control-twin screen.

Usage:
    python3 scripts/lunarfire_modulus_fusion_v10.py --output-dir data/helionis
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helionis.modulus_fusion_control import (
    ModulusFusionControlAssumptions,
    run_modulus_fusion_control_twin,
)
from helionis.modulus_fusion_reporting import (
    plot_modulus_fusion_control,
    write_modulus_fusion_control_csv,
    write_modulus_fusion_control_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LunarFire / Modulus Fusion v1.0 control-twin outputs."
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
        "--update-period-ms",
        type=float,
        default=1.0,
        help="Control-loop update period in milliseconds.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=12,
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
    if not math.isfinite(args.update_period_ms) or args.update_period_ms <= 0:
        parser.error("--update-period-ms must be positive and finite")
    if args.limit <= 0:
        parser.error("--limit must be positive")

    assumptions = ModulusFusionControlAssumptions(
        update_period_ms=args.update_period_ms,
    )
    results = run_modulus_fusion_control_twin(
        target_screening_net_mw=args.target_mw,
        assumptions=assumptions,
        limit=args.limit,
    )
    if not results:
        parser.error("no Modulus Fusion control rows found")

    output_dir = Path(args.output_dir)
    csv_path = output_dir / "lunarfire_modulus_fusion_v1_0.csv"
    markdown_path = output_dir / "lunarfire_modulus_fusion_v1_0.md"
    plot_path = output_dir / "lunarfire_modulus_fusion_v1_0.png"

    write_modulus_fusion_control_csv(results, csv_path)
    write_modulus_fusion_control_markdown(results, markdown_path)
    if not args.no_plot:
        plot_modulus_fusion_control(results, plot_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    if not args.no_plot:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
