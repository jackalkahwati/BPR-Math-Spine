#!/usr/bin/env python3
"""Run the Helionis one-week D-He3 architecture trade study.

Usage:
    python scripts/helionis_trade_study.py --output-dir data/helionis
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helionis.architecture import run_trade_study
from helionis.reporting import plot_trade_study, write_csv, write_markdown_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Helionis D-He3 trade-study outputs."
    )
    parser.add_argument(
        "--output-dir",
        default="data/helionis",
        help="Directory for CSV, Markdown, and plot outputs.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip PNG plot generation.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results = run_trade_study()

    csv_path = output_dir / "helionis_trade_study.csv"
    markdown_path = output_dir / "helionis_trade_study.md"
    plot_path = output_dir / "helionis_trade_study.png"

    write_csv(results, csv_path)
    write_markdown_summary(results, markdown_path)
    if not args.no_plot:
        plot_trade_study(results, plot_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    if not args.no_plot:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
