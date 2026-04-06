"""
BPR Solution Space Visualization
==================================

Generates comprehensive maps of the BPR parameter landscape:
  1. Core-score heatmap (p × z), full range
  2. Contour plot with Pareto frontier overlay
  3. Z profile: best-p core score per z (bar chart)
  4. P profile: score vs p at z=6 (line plot with viable band)
  5. Per-observable error bands (1/alpha, v_EW, n_s, delta_CP)
  6. Pareto frontier scatter (inv_alpha err vs v_EW err)
  7. Solution families: viable regions at 1% and 5% thresholds
  8. Sensitivity: gradient map near (104729, 6)

Author: Claude Code audit, 2026-04-06
"""

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "analysis" / "results" / "solution_space"
PLOTS_DIR = REPO_ROOT / "analysis" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_P = 104729
BASELINE_Z = 6

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> List[Dict]:
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = int(v)
                except ValueError:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plot 1: Core-score heatmap (log scale)
# ---------------------------------------------------------------------------

def plot_heatmap(rows: List[Dict], output: Path):
    """Heatmap of core score across (p, z) space."""
    ps = sorted(set(r["p"] for r in rows))
    zs = sorted(set(r["z"] for r in rows))

    # Build grid
    score_grid = np.full((len(zs), len(ps)), np.nan)
    p_idx = {p: i for i, p in enumerate(ps)}
    z_idx = {z: i for i, z in enumerate(zs)}

    for r in rows:
        pi, zi = p_idx[r["p"]], z_idx[r["z"]]
        score_grid[zi, pi] = r["score_core"]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Log-scale colormap
    vmin, vmax = 0.001, 0.30
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn_r  # red=bad, green=good
    im = ax.imshow(score_grid, aspect="auto", origin="lower",
                   norm=norm, cmap=cmap,
                   extent=[min(ps), max(ps), min(zs) - 0.5, max(zs) + 0.5])

    cbar = fig.colorbar(im, ax=ax, label="Core Score (RMS fractional error)", pad=0.02)
    cbar.ax.axhline(y=0.01, color="blue", lw=2, ls="--", alpha=0.8)
    cbar.ax.text(1.5, 0.01, "1%", transform=cbar.ax.transData, color="blue",
                 fontsize=8, va="center")

    # Mark baseline
    ax.axvline(x=BASELINE_P, color="white", lw=1.5, ls="--", alpha=0.9)
    ax.axhline(y=BASELINE_Z, color="white", lw=1.5, ls="--", alpha=0.9)
    ax.plot(BASELINE_P, BASELINE_Z, "w*", ms=18, zorder=10, label=f"Baseline (p={BASELINE_P}, z={BASELINE_Z})")

    ax.set_xlabel("Prime p", fontsize=12)
    ax.set_ylabel("Coordination number z", fontsize=12)
    ax.set_title("BPR Core Score Heatmap: RMS(1/α error, v_EW error)", fontsize=13, fontweight="bold")
    ax.set_yticks(zs)
    ax.legend(loc="upper left", fontsize=10)

    ax.text(0.02, 0.95, "Provenance: inv_alpha_0 = core-derived | v_EW = bridge formula",
            transform=ax.transAxes, fontsize=8, color="white", alpha=0.9,
            va="top", bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))

    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output.name}")


# ---------------------------------------------------------------------------
# Plot 2: Z profile (best score per z) with observable decomposition
# ---------------------------------------------------------------------------

def plot_z_profile(z_profiles: List[Dict], rows: List[Dict], output: Path):
    """Bar chart of best core score per z, with per-observable components."""
    zs = [r["z"] for r in z_profiles]
    best_scores = [r["best_score_core"] for r in z_profiles]
    dcp_sigmas = [r["delta_cp_sigma"] for r in z_profiles]

    # Get alpha error and vEW error for best p at each z
    best_p_per_z = {r["z"]: r["best_p"] for r in z_profiles}
    alpha_errs, vew_errs = [], []
    for z in zs:
        bp = best_p_per_z[z]
        pt = next((r for r in rows if r["p"] == bp and r["z"] == z), None)
        if pt:
            alpha_errs.append(pt["inv_alpha_0_frac_err"] * 100)
            vew_errs.append(pt["v_EW_frac_err"] * 100)
        else:
            alpha_errs.append(np.nan)
            vew_errs.append(np.nan)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    colors = ["#d62728" if z == BASELINE_Z else "#1f77b4" for z in zs]

    # Top: core score
    bars = ax1.bar(zs, [s * 100 for s in best_scores], color=colors)
    ax1.axhline(1.0, color="green", ls="--", lw=1.5, alpha=0.8, label="1% threshold")
    ax1.axhline(5.0, color="orange", ls="--", lw=1.5, alpha=0.8, label="5% threshold")
    ax1.set_ylabel("Best Core Score (%)", fontsize=11)
    ax1.set_title("Best Achievable Core Score per z", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.text(BASELINE_Z, best_scores[zs.index(BASELINE_Z)] * 100 + 0.2,
             f"z=6\n({best_scores[zs.index(BASELINE_Z)]*100:.2f}%)", ha="center", fontsize=8, color="#d62728")

    # Middle: per-observable errors
    x = np.array(zs)
    width = 0.35
    ax2.bar(x - width/2, alpha_errs, width, label="1/α error", color="#ff7f0e", alpha=0.85)
    ax2.bar(x + width/2, vew_errs, width, label="v_EW error", color="#2ca02c", alpha=0.85)
    ax2.axhline(1.0, color="black", ls=":", lw=1, alpha=0.5)
    ax2.set_ylabel("Fractional Error (%)", fontsize=11)
    ax2.set_title("Per-Observable Errors at Best p (Core Observables)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_ylim(bottom=0)

    # Bottom: δ_CP sigma (ansatz observable)
    dcp_colors = ["#d62728" if z == BASELINE_Z else "#9467bd" for z in zs]
    ax3.bar(zs, dcp_sigmas, color=dcp_colors, alpha=0.85)
    ax3.axhline(1.0, color="green", ls="--", lw=1.5, alpha=0.8, label="1σ threshold")
    ax3.axhline(2.0, color="orange", ls="--", lw=1.5, alpha=0.8, label="2σ threshold")
    ax3.set_ylabel("δ_CP Error (σ)", fontsize=11)
    ax3.set_xlabel("Coordination Number z", fontsize=12)
    ax3.set_title("δ_CP Sigma Error per z [ANSATZ — not core]", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.text(0.02, 0.95, "NOTE: δ_CP formula is ansatz (fallback = PDG value). Separate layer.",
             transform=ax3.transAxes, fontsize=8, color="gray", va="top")

    ax3.set_xticks(zs)

    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output.name}")


# ---------------------------------------------------------------------------
# Plot 3: Score vs p at z=6 (line plot with viable band)
# ---------------------------------------------------------------------------

def plot_p_landscape_z6(rows: List[Dict], output: Path):
    """Score vs p at z=6, showing core score, observable components, viable window."""
    z6 = sorted([r for r in rows if r["z"] == 6], key=lambda x: x["p"])
    ps = [r["p"] for r in z6]
    scores = [r["score_core"] * 100 for r in z6]
    alpha_errs = [r["inv_alpha_0_frac_err"] * 100 for r in z6]
    vew_errs = [r["v_EW_frac_err"] * 100 for r in z6]
    ns_errs = [r["n_s_frac_err"] * 100 for r in z6]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Top: core score
    ax1.semilogy(ps, scores, "b-", lw=1.5, alpha=0.8, label="Core Score (RMS)")
    ax1.axhline(1.0, color="green", ls="--", lw=1.5, alpha=0.9, label="1% threshold")
    ax1.axhline(5.0, color="orange", ls="--", lw=1.5, alpha=0.7, label="5% threshold")
    ax1.axvline(BASELINE_P, color="red", ls=":", lw=2, alpha=0.9, label=f"Baseline p={BASELINE_P}")

    # Mark best point
    best_z6 = min(z6, key=lambda r: r["score_core"])
    ax1.axvline(best_z6["p"], color="darkgreen", ls=":", lw=2, alpha=0.9,
                label=f"Best p={best_z6['p']} (score={best_z6['score_core']*100:.2f}%)")

    # Shade viable region
    viable_ps = [r["p"] for r in z6 if r["score_core"] <= 0.01]
    if viable_ps:
        ax1.axvspan(min(viable_ps), max(viable_ps), alpha=0.15, color="green",
                    label=f"Viable ≤1% p=[{min(viable_ps)},{max(viable_ps)}]")

    ax1.set_ylabel("Core Score (%)", fontsize=12)
    ax1.set_title(f"BPR Core Score vs Prime p (z=6 fixed)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.set_ylim(0.1, 50)
    ax1.grid(True, alpha=0.3)

    # Bottom: per-observable breakdown
    ax2.semilogy(ps, alpha_errs, "#ff7f0e", lw=1.5, alpha=0.85, label="1/α error (core)")
    ax2.semilogy(ps, vew_errs, "#2ca02c", lw=1.5, alpha=0.85, label="v_EW error (bridge)")
    ax2.semilogy(ps, ns_errs, "#9467bd", lw=1.5, alpha=0.65, ls="--",
                 label="n_s error (ansatz — NOT core)")
    ax2.axhline(1.0, color="gray", ls=":", lw=1.2, alpha=0.5)
    ax2.axvline(BASELINE_P, color="red", ls=":", lw=2, alpha=0.9)

    ax2.set_xlabel("Prime p", fontsize=12)
    ax2.set_ylabel("Fractional Error (%)", fontsize=12)
    ax2.set_title("Per-Observable Errors vs p at z=6", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax2.text(0.02, 0.05,
             "n_s (dashed purple) is ANSATZ: N=p^(1/3)×4/3 asserted, not derived",
             transform=ax2.transAxes, fontsize=8, color="gray")

    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output.name}")


# ---------------------------------------------------------------------------
# Plot 4: Pareto frontier (1/alpha err vs v_EW err)
# ---------------------------------------------------------------------------

def plot_pareto(rows: List[Dict], pareto: List[Dict], output: Path):
    """Pareto frontier in the (1/alpha err, v_EW err) plane."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by z
    z_all = sorted(set(r["z"] for r in rows))
    cmap = plt.cm.tab10
    z_colors = {z: cmap(i / len(z_all)) for i, z in enumerate(z_all)}

    # Scatter all points (subsampled)
    for z in z_all:
        z_pts = [r for r in rows if r["z"] == z]
        xs = [r["inv_alpha_0_frac_err"] * 100 for r in z_pts]
        ys = [r["v_EW_frac_err"] * 100 for r in z_pts]
        ax.scatter(xs, ys, c=[z_colors[z]], s=4, alpha=0.25, rasterized=True)

    # Pareto front (only core 2-obs pareto)
    core_obj = ["inv_alpha_0_frac_err", "v_EW_frac_err"]
    # Sort by alpha error for line
    pf_sorted = sorted(pareto, key=lambda r: r["inv_alpha_0_frac_err"])
    pf_xs = [r["inv_alpha_0_frac_err"] * 100 for r in pf_sorted]
    pf_ys = [r["v_EW_frac_err"] * 100 for r in pf_sorted]
    ax.step(pf_xs, pf_ys, "k-", lw=2, alpha=0.7, label="Pareto front (4-obj)", where="post")

    # Mark baseline
    bl = next((r for r in rows if r["p"] == BASELINE_P and r["z"] == BASELINE_Z), None)
    if bl:
        ax.scatter([bl["inv_alpha_0_frac_err"] * 100], [bl["v_EW_frac_err"] * 100],
                   c="red", s=250, marker="*", zorder=10, label=f"Baseline ({BASELINE_P}, z=6)")

    # Best core point
    best = min(rows, key=lambda r: r["score_core"])
    ax.scatter([best["inv_alpha_0_frac_err"] * 100], [best["v_EW_frac_err"] * 100],
               c="gold", s=200, marker="D", zorder=10, edgecolors="k",
               label=f"Best core (p={best['p']}, z={best['z']})")

    # Add threshold box
    ax.axvline(1.0, color="green", ls="--", lw=1.2, alpha=0.6)
    ax.axhline(1.0, color="green", ls="--", lw=1.2, alpha=0.6)
    rect = Rectangle((0, 0), 1.0, 1.0, linewidth=2, edgecolor="green",
                      facecolor="green", alpha=0.05, label="1% box (both observables)")
    ax.add_patch(rect)

    # Legend for z colors
    legend_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=z_colors[z],
                              markersize=8, label=f"z={z}") for z in z_all]
    legend_handles += [
        Line2D([0], [0], c="red", marker="*", ms=12, ls="None", label=f"Baseline ({BASELINE_P}, 6)"),
        Line2D([0], [0], c="gold", marker="D", ms=10, ls="None",
               markeredgecolor="k", label=f"Best core (p={best['p']}, z={best['z']})"),
        Line2D([0], [0], c="k", lw=2, label="Pareto front"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right", ncol=2)

    ax.set_xlabel("1/α Fractional Error (%)", fontsize=12)
    ax.set_ylabel("v_EW Fractional Error (%)", fontsize=12)
    ax.set_title("Pareto Frontier: 1/α vs v_EW Error\n(two core observables)", fontsize=13, fontweight="bold")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output.name}")


# ---------------------------------------------------------------------------
# Plot 5: Viable region map (solution families)
# ---------------------------------------------------------------------------

def plot_solution_families(rows: List[Dict], clusters: Dict, output: Path):
    """Visual map of viable solution families at different thresholds."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    thresholds = [("5pct_core", 0.05, "orange", "≤5% core"), ("1pct_core", 0.01, "green", "≤1% core")]

    ps_all = sorted(set(r["p"] for r in rows))
    zs_all = sorted(set(r["z"] for r in rows))

    for ax, (key, thresh, color, label) in zip(axes, thresholds):
        viable = {(r["p"], r["z"]) for r in rows if r["score_core"] <= thresh}
        not_viable = {(r["p"], r["z"]) for r in rows} - viable

        # Scatter non-viable gray
        ax.scatter([p for p, z in not_viable], [z for p, z in not_viable],
                   c="lightgray", s=6, alpha=0.4, rasterized=True, label="Not viable")

        # Scatter viable colored
        if viable:
            ax.scatter([p for p, z in viable], [z for p, z in viable],
                       c=color, s=20, alpha=0.8, zorder=5, label=f"Viable ({label})")

        # Mark baseline
        ax.axvline(BASELINE_P, color="red", ls=":", lw=2, zorder=10)
        ax.plot(BASELINE_P, BASELINE_Z, "r*", ms=18, zorder=11,
                label=f"Baseline ({BASELINE_P}, z=6)")

        ax.set_ylabel("Coordination Number z", fontsize=11)
        ax.set_yticks(zs_all)
        ax.legend(fontsize=9, loc="upper left")
        ax.set_title(f"Solution Families — {label} threshold", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2)

        # Annotate count
        ax.text(0.98, 0.05, f"{len(viable)} viable points",
                transform=ax.transAxes, fontsize=10, ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))

    axes[-1].set_xlabel("Prime p", fontsize=12)
    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output.name}")


# ---------------------------------------------------------------------------
# Plot 6: Sensitivity map (local gradient near baseline)
# ---------------------------------------------------------------------------

def plot_sensitivity(rows: List[Dict], output: Path):
    """Score gradient near (104729, 6) — p in tight window, z=[5,6,7]."""
    focus = [r for r in rows if abs(r["p"] - BASELINE_P) <= 2000 and 5 <= r["z"] <= 7]
    focus.sort(key=lambda r: (r["z"], r["p"]))

    fig, ax = plt.subplots(figsize=(12, 6))

    linestyles = {5: "--", 6: "-", 7: ":"}
    colors = {5: "#ff7f0e", 6: "#1f77b4", 7: "#2ca02c"}
    for z in [5, 6, 7]:
        pts = sorted([r for r in focus if r["z"] == z], key=lambda r: r["p"])
        if not pts:
            continue
        ps = [r["p"] for r in pts]
        ss = [r["score_core"] * 100 for r in pts]
        ax.plot(ps, ss, color=colors[z], ls=linestyles[z], lw=2,
                label=f"z={z}", marker=".", ms=5, alpha=0.85)

    ax.axvline(BASELINE_P, color="red", ls=":", lw=2, alpha=0.9, label=f"Baseline p={BASELINE_P}")
    ax.axhline(1.0, color="green", ls="--", lw=1.5, alpha=0.8, label="1% threshold")

    ax.set_xlabel("Prime p (window: ±2000 around baseline)", fontsize=12)
    ax.set_ylabel("Core Score (%)", fontsize=12)
    ax.set_title(f"Sensitivity Map: Score Near Baseline (p≈{BASELINE_P}, z=5,6,7)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Mark best z=6 point in window
    z6_pts = [r for r in focus if r["z"] == 6]
    if z6_pts:
        best_local = min(z6_pts, key=lambda r: r["score_core"])
        ax.axvline(best_local["p"], color="darkblue", ls="-.", lw=1.5, alpha=0.8,
                   label=f"Best z=6 (p={best_local['p']}, {best_local['score_core']*100:.2f}%)")
        ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output.name}")


# ---------------------------------------------------------------------------
# Plot 7: Observable independence panel
# ---------------------------------------------------------------------------

def plot_observable_panel(rows: List[Dict], output: Path):
    """4-panel plot of each observable's error landscape at z=6."""
    z6 = sorted([r for r in rows if r["z"] == 6], key=lambda x: x["p"])
    ps = [r["p"] for r in z6]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    panels = [
        ("inv_alpha_0_frac_err", "1/α Error", "#ff7f0e", "core-derived",
         "Formula: ln(p)² + z/2 + γ - 1/(2π)"),
        ("v_EW_frac_err", "v_EW Error", "#2ca02c", "bridge formula",
         "Formula: Λ_QCD × p^(1/3) × (ln(p) + z − 2)"),
        ("n_s_frac_err", "n_s Error", "#9467bd", "ANSATZ",
         "Formula: 1−2/N, N=p^(1/3)×4/3 [N derivation absent]"),
        ("delta_cp_frac_err", "δ_CP Error", "#d62728", "ANSATZ",
         "Formula: π/2 − 1/√(z+1) [fallback=PDG value; z-fixed so flat in p]"),
    ]

    for ax, (key, title, color, status, note) in zip(axes.flat, panels):
        vals = [r[key] * 100 for r in z6]
        ax.plot(ps, vals, color=color, lw=1.5, alpha=0.85)
        ax.axvline(BASELINE_P, color="red", ls=":", lw=2, label=f"p={BASELINE_P}")
        ax.axhline(1.0, color="green", ls="--", lw=1.2, alpha=0.7, label="1%")

        # Best point
        best_idx = vals.index(min(vals))
        ax.axvline(ps[best_idx], color="darkblue", ls="-.", lw=1.5,
                   label=f"Best p={ps[best_idx]} ({min(vals):.2f}%)")

        status_color = "red" if "ANSATZ" in status else "steelblue"
        ax.set_title(f"{title}\n[{status}]", fontsize=11, fontweight="bold", color=status_color)
        ax.set_xlabel("Prime p", fontsize=10)
        ax.set_ylabel("Fractional Error (%)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=7,
                va="top", color="gray", wrap=True)

    fig.suptitle("Per-Observable Error Landscape at z=6\n(Ansatz observables labeled RED)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    rows = load_csv(DATA_DIR / "all_points.csv")
    z_profiles = load_csv(DATA_DIR / "z_profiles.csv")
    pareto = load_csv(DATA_DIR / "pareto_front.csv")
    clusters = load_json(DATA_DIR / "clusters.json")

    print(f"Loaded {len(rows)} data points")
    print("Generating plots...")

    plot_heatmap(rows, PLOTS_DIR / "solution_space_heatmap.png")
    plot_z_profile(z_profiles, rows, PLOTS_DIR / "solution_space_z_profile.png")
    plot_p_landscape_z6(rows, PLOTS_DIR / "solution_space_p_landscape_z6.png")
    plot_pareto(rows, pareto, PLOTS_DIR / "solution_space_pareto.png")
    plot_solution_families(rows, clusters, PLOTS_DIR / "solution_space_families.png")
    plot_sensitivity(rows, PLOTS_DIR / "solution_space_sensitivity.png")
    plot_observable_panel(rows, PLOTS_DIR / "solution_space_observables.png")

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
