"""
BPR Corridor Visualization — 4 plots
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
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "analysis" / "results" / "corridor"
PLOT_DIR  = REPO_ROOT / "analysis" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_P  = 104729
CORRIDOR_LO = 103935
CORRIDOR_HI = 110634

# ── Load data ────────────────────────────────────────────────────────────────
def load_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                try:    parsed[k] = int(v)
                except:
                    try: parsed[k] = float(v)
                    except: parsed[k] = v
            rows.append(parsed)
    return rows

rows    = load_csv(DATA_DIR / "ranked.csv")       # sorted by score
all_pts = load_csv(DATA_DIR / "all_primes.csv")   # sorted by p
all_pts.sort(key=lambda r: r["p"])

with open(DATA_DIR / "corridor_summary.json") as f:
    summary = json.load(f)

RMS_P   = summary["rms_optimal"]["p"]
ALPHA_P = summary["alpha_optimal"]["p"]
VEW_P   = summary["vEW_optimal"]["p"]
ROB_P   = summary["robustness_optimal"]["p"]

ps      = [r["p"] for r in all_pts]
scores  = [r["core_score_pct"] for r in all_pts]
ae_vals = [r["inv_alpha_err_ppm"] for r in all_pts]
ve_vals = [r["v_EW_err_pct"] for r in all_pts]
rob_vals = [r["robustness"] for r in all_pts]

def get(p):
    return next(r for r in all_pts if r["p"] == p)

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Score across the corridor with annotations
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(ps, scores, color="#1f77b4", lw=1.2, alpha=0.7, label="Core score (%)")

# Shade regions
ax.axhspan(0, 0.5, alpha=0.08, color="green", label="< 0.5% (elite)")
ax.axhspan(0.5, 1.0, alpha=0.05, color="lime", label="0.5–1.0% (viable)")

# Mark special primes
specials = [
    (ALPHA_P, "alpha-opt\np=%d" % ALPHA_P, "orange", "^", 12),
    (VEW_P,   "vEW-opt\np=%d"   % VEW_P,   "green",  "s", 10),
    (RMS_P,   "RMS-opt\np=%d"   % RMS_P,   "gold",   "D", 12),
    (ROB_P,   "robust-opt\np=%d" % ROB_P,  "purple", "P", 12),
    (BASELINE_P, "BASELINE\np=%d" % BASELINE_P, "red", "*", 18),
]
for p_mark, lbl, col, mk, ms in specials:
    r = get(p_mark)
    ax.scatter([p_mark], [r["core_score_pct"]], c=col, s=ms**2 if mk=="*" else ms*20,
               marker=mk, zorder=10, edgecolors="k" if col!="red" else "darkred",
               linewidths=1.0)
    ax.annotate(lbl, (p_mark, r["core_score_pct"]),
                xytext=(0, 14 + (30 if p_mark==BASELINE_P else 0)),
                textcoords="offset points", ha="center", fontsize=7.5,
                color=col, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.8))

ax.set_xlim(CORRIDOR_LO - 100, CORRIDOR_HI + 100)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Prime p", fontsize=12)
ax.set_ylabel("Core Score (%)", fontsize=12)
ax.set_title("BPR Viable Prime Corridor — Core Score Landscape  [z=6]",
             fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.25)

# Score statistics annotation
ax.text(0.98, 0.95,
        "580 viable primes\nScore range: %.4f–%.4f%%\nMean: %.4f%%  Std: %.4f%%" % (
            min(scores), max(scores), np.mean(scores), np.std(scores)),
        transform=ax.transAxes, fontsize=9, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

# Custom legend
handles = [
    Line2D([0],[0], marker="*", color="w", markerfacecolor="red",
           markeredgecolor="darkred", ms=14, label="Baseline p=104729"),
    Line2D([0],[0], marker="^", color="w", markerfacecolor="orange",
           markeredgecolor="k", ms=10, label="alpha-optimal p=%d" % ALPHA_P),
    Line2D([0],[0], marker="D", color="w", markerfacecolor="gold",
           markeredgecolor="k", ms=10, label="RMS-optimal p=%d" % RMS_P),
    Line2D([0],[0], marker="s", color="w", markerfacecolor="green",
           markeredgecolor="k", ms=10, label="vEW-optimal p=%d" % VEW_P),
    Line2D([0],[0], marker="P", color="w", markerfacecolor="purple",
           markeredgecolor="k", ms=10, label="robust-optimal p=%d" % ROB_P),
]
ax.legend(handles=handles, fontsize=8.5, loc="upper right")

plt.tight_layout()
fig.savefig(PLOT_DIR / "corridor_score.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved corridor_score.png")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: 1/alpha vs v_EW tradeoff (Pareto view inside corridor)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

# Colour by p position within corridor (blue=low-p, red=high-p)
p_arr = np.array(ps)
norm  = mcolors.Normalize(vmin=p_arr.min(), vmax=p_arr.max())
cmap  = plt.cm.RdYlBu_r
colors_sc = cmap(norm(p_arr))

sc = ax.scatter(ae_vals, ve_vals, c=p_arr, cmap=cmap,
                s=12, alpha=0.65, rasterized=True)
cbar = fig.colorbar(sc, ax=ax, label="Prime p", pad=0.02)

# Mark specials
for p_mark, lbl, col, mk, ms in specials:
    r = get(p_mark)
    ax.scatter([r["inv_alpha_err_ppm"]], [r["v_EW_err_pct"]],
               c=col, s=ms**2 if mk=="*" else ms*25,
               marker=mk, zorder=10, edgecolors="k",
               linewidths=1.2, label=lbl.replace("\n"," "))

# Mark Pareto frontier within corridor
ae_arr = np.array(ae_vals)
ve_arr = np.array(ve_vals)
# Simple Pareto: sort by alpha error, keep points where vEW is non-dominated
order = np.argsort(ae_arr)
pareto_ps, pareto_ae, pareto_ve = [], [], []
best_ve = np.inf
for i in order:
    if ve_arr[i] < best_ve:
        pareto_ps.append(ps[i])
        pareto_ae.append(ae_arr[i])
        pareto_ve.append(ve_arr[i])
        best_ve = ve_arr[i]
ax.step(pareto_ae, pareto_ve, "k-", lw=2.0, alpha=0.7,
        where="post", label="Pareto frontier (corridor)", zorder=8)

# Threshold lines
ax.axvline(x=100, color="gray", ls=":", lw=1, alpha=0.5)
ax.axhline(y=0.1, color="gray", ls=":", lw=1, alpha=0.5)

ax.set_xlabel("1/α Error (ppm)", fontsize=12)
ax.set_ylabel("v_EW Error (%)", fontsize=12)
ax.set_title("1/α vs v_EW Tradeoff — Viable Corridor\n(colour = prime p, blue=low, red=high)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.25)

plt.tight_layout()
fig.savefig(PLOT_DIR / "corridor_tradeoff.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved corridor_tradeoff.png")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Robustness score vs p
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

# Top: robustness (mean score in ±200 window)
ax1.plot(ps, rob_vals, color="purple", lw=1.3, alpha=0.8, label="Robustness (mean score ±200p window)")
ax1.axvline(ROB_P,   color="purple", ls="--", lw=1.5, alpha=0.8,
            label="Most robust p=%d (%.4f%%)" % (ROB_P, get(ROB_P)["robustness"]))
ax1.axvline(BASELINE_P, color="red", ls=":", lw=2, alpha=0.9,
            label="Baseline p=%d (%.4f%%)" % (BASELINE_P, get(BASELINE_P)["robustness"]))
ax1.set_ylabel("Robustness Score\n(mean core score, ±200p)", fontsize=11)
ax1.set_title("Corridor Robustness: How Stable is Each Prime?", fontsize=13, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.25)

# Bottom: local curvature
curv = [r["curvature"] * 1e6 for r in all_pts]  # scale to readable units
ax2.plot(ps, curv, color="#9467bd", lw=1, alpha=0.7)
ax2.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
ax2.axvline(BASELINE_P, color="red", ls=":", lw=2, alpha=0.9)
ax2.axvline(RMS_P, color="gold", ls="--", lw=1.5, alpha=0.8)
ax2.set_xlabel("Prime p", fontsize=12)
ax2.set_ylabel("Local Curvature\n(×10⁻⁶ per p²)", fontsize=11)
ax2.set_title("Score Curvature — Near-Zero = Flat Family; Positive = Local Well",
              fontsize=12, fontweight="bold")
ax2.grid(True, alpha=0.25)

plt.tight_layout()
fig.savefig(PLOT_DIR / "corridor_robustness.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved corridor_robustness.png")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Classification overview
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))

# Background colour bands
ax.axvspan(CORRIDOR_LO, CORRIDOR_LO + 0.05*(CORRIDOR_HI-CORRIDOR_LO),
           alpha=0.12, color="red",   label="Fragile edge zone (5%)")
ax.axvspan(CORRIDOR_HI - 0.05*(CORRIDOR_HI-CORRIDOR_LO), CORRIDOR_HI,
           alpha=0.12, color="red")

# Per-prime classification strip
class_colors = {
    "alpha-optimal":   "#ff7f0e",
    "vEW-optimal":     "#2ca02c",
    "RMS-optimal":     "#FFD700",
    "robust":          "#9467bd",
    "robust-interior": "#17becf",
    "fragile-edge":    "#d62728",
    "mid-range":       "#aec7e8",
}

for r in all_pts:
    c_str = r["classification"]
    # Primary colour: first category
    primary = c_str.split("|")[0]
    col = class_colors.get(primary, "#aec7e8")
    ax.axvline(r["p"], color=col, lw=1.0, alpha=0.6, ymin=0.1, ymax=0.9)

# Overlay score line (thin, secondary axis)
ax2 = ax.twinx()
ax2.plot(ps, scores, "k-", lw=1.5, alpha=0.4, label="Core score")
ax2.set_ylabel("Core Score (%)", fontsize=10, color="gray")
ax2.tick_params(axis="y", colors="gray")
ax2.set_ylim(0, 1.05)

# Mark specials
for p_mark, lbl, col, mk, ms in specials:
    r = get(p_mark)
    ax.axvline(p_mark, color=col, lw=2.5, alpha=1.0, zorder=10)
    ax.text(p_mark, 0.92, lbl.split("\n")[0], rotation=90,
            transform=ax.get_xaxis_transform(),
            fontsize=7, color=col, fontweight="bold", va="top", ha="right")

ax.set_xlim(CORRIDOR_LO - 50, CORRIDOR_HI + 50)
ax.set_ylim(0, 1)
ax.set_xlabel("Prime p", fontsize=12)
ax.set_yticks([])
ax.set_title("Classification Map — 580 Viable Primes at z=6",
             fontsize=13, fontweight="bold")

legend_handles = [Line2D([0],[0], color=v, lw=6, alpha=0.7, label=k)
                  for k, v in class_colors.items()]
legend_handles += [
    Line2D([0],[0], marker="*", color="w", markerfacecolor="red",
           markeredgecolor="darkred", ms=12, label="Baseline"),
    Line2D([0],[0], color="k", lw=1.5, alpha=0.4, label="Core score"),
]
ax.legend(handles=legend_handles, fontsize=7.5, loc="lower center",
          ncol=5, bbox_to_anchor=(0.5, -0.22))

plt.tight_layout()
fig.savefig(PLOT_DIR / "corridor_classification.png", dpi=150,
            bbox_inches="tight")
plt.close()
print("  Saved corridor_classification.png")

print("\nAll corridor plots saved.")
