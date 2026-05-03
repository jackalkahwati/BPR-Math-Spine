#!/usr/bin/env python3
"""Generate the LunarFire v0.6 mirror/nozzle sweep.

Usage:
    python3 scripts/lunarfire_mirror_nozzle_v06.py --output-dir data/helionis
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helionis.mirror_nozzle_v06 import run_mirror_nozzle_v06_sweep
from helionis.mirror_nozzle_v06_reporting import (
    plot_mirror_nozzle_v06,
    write_mirror_nozzle_v06_csv,
    write_mirror_nozzle_v06_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LunarFire v0.6 mirror/nozzle sweep outputs."
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
        default=12,
        help="Number of top-ranked rows to write.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip PNG plot generation.",
    )
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.target_mw <= 0:
        parser.error("--target-mw must be positive")

    output_dir = Path(args.output_dir)
    results = run_mirror_nozzle_v06_sweep(
        target_screening_net_mw=args.target_mw,
        limit=args.limit,
    )
    if not results:
        parser.error("no feasible mirror/nozzle v0.6 candidates found")

    csv_path = output_dir / "lunarfire_mirror_nozzle_v0_6.csv"
    markdown_path = output_dir / "lunarfire_mirror_nozzle_v0_6.md"
    plot_path = output_dir / "lunarfire_mirror_nozzle_v0_6.png"

    write_mirror_nozzle_v06_csv(results, csv_path)
    write_mirror_nozzle_v06_markdown(results, markdown_path)
    if not args.no_plot:
        plot_mirror_nozzle_v06(results, plot_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    if not args.no_plot:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
