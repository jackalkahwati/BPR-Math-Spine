#!/usr/bin/env python3
"""Generate LunarFire v1.1 control-constrained CAD envelope outputs.

Usage:
    python3 scripts/lunarfire_cad_envelope_v11.py --output-dir data/helionis
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helionis.cad_envelope_v11 import run_cad_envelope_v11
from helionis.cad_envelope_v11_reporting import (
    plot_cad_envelope_v11,
    write_cad_envelope_v11_csv,
    write_cad_envelope_v11_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LunarFire v1.1 CAD envelope outputs."
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
    if not math.isfinite(args.target_mw) or args.target_mw <= 0:
        parser.error("--target-mw must be positive and finite")
    if args.limit <= 0:
        parser.error("--limit must be positive")

    results = run_cad_envelope_v11(
        target_screening_net_mw=args.target_mw,
        limit=args.limit,
    )
    if not results:
        parser.error("no CAD envelope rows found")

    output_dir = Path(args.output_dir)
    csv_path = output_dir / "lunarfire_cad_envelope_v1_1.csv"
    markdown_path = output_dir / "lunarfire_cad_envelope_v1_1.md"
    plot_path = output_dir / "lunarfire_cad_envelope_v1_1.png"

    write_cad_envelope_v11_csv(results, csv_path)
    write_cad_envelope_v11_markdown(results, markdown_path)
    if not args.no_plot:
        plot_cad_envelope_v11(results, plot_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    if not args.no_plot:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
