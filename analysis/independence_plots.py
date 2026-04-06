"""Generate plots for the independence audit."""
import sys, csv, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "analysis" / "results"
PLOTS_DIR = REPO_ROOT / "analysis" / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO_ROOT))

def load(name):
    with open(RESULTS_DIR / name) as f:
        return list(csv.DictReader(f))

def f(s):
    try: return float(s)
    except: return float("nan")

ns = load("independence_ns_sweep.csv")
dcp = load("independence_dcp_sweep.csv")
combined = load("independence_combined_sweep.csv")

# -- Plot 1: n_s vs prime --
primes = [int(r["p"]) for r in ns]
ns_vals = [f(r["n_s_pred"]) for r in ns]
ns_errs = [f(r["n_s_signed"]) for r in ns]
EXP_NS = 0.9649
NS_UNC = 0.0042

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.plot(primes, ns_vals, "g-o", ms=4, lw=1.5, label="BPR n_s prediction")
ax1.axhline(EXP_NS, color="black", ls="-", lw=1.2, label=f"Exp: {EXP_NS}")
ax1.axhline(EXP_NS + NS_UNC, color="black", ls="--", lw=0.8, alpha=0.5)
ax1.axhline(EXP_NS - NS_UNC, color="black", ls="--", lw=0.8, alpha=0.5, label="±1σ")
ax1.fill_between(primes, EXP_NS - NS_UNC, EXP_NS + NS_UNC, alpha=0.15, color="green")
ax1.axvline(104729, color="red", ls="--", lw=1.5, alpha=0.8, label="p=104729")
ax1.set_ylabel("Predicted n_s")
ax1.set_title("n_s (Inflationary Spectral Index) vs Substrate Prime\n"
              "Formula: 1 − 3/(2·p^{1/3})  [INDEPENDENT of 1/α — different functional form]")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2.plot(primes, ns_errs, "g-o", ms=4, lw=1.5, label="n_s − Exp")
ax2.axhline(0, color="black", lw=0.8)
ax2.axhline(NS_UNC, color="black", ls="--", lw=0.8, label="+1σ")
ax2.axhline(-NS_UNC, color="black", ls="--", lw=0.8, label="-1σ")
ax2.fill_between(primes, -NS_UNC, NS_UNC, alpha=0.15, color="green")
ax2.axvline(104729, color="red", ls="--", lw=1.5, alpha=0.8)
ax2.set_xlabel("Substrate prime p")
ax2.set_ylabel("Signed error: n_s − 0.9649")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.annotate("n_s-optimal p ≈ 78,046\n(off-chart left)", xy=(104479, 0.003),
             xytext=(104560, 0.001), arrowprops=dict(arrowstyle="->", color="purple"),
             color="purple", fontsize=8)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "independence_ns_vs_prime.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: independence_ns_vs_prime.png")

# -- Plot 2: delta_CP vs z --
z_vals = [int(r["z"]) for r in dcp]
dcp_vals = [f(r["delta_CP_pred_rad"]) for r in dcp]
dcp_errs = [f(r["delta_CP_sigma"]) for r in dcp]
EXP_DCP = 1.196
DCP_UNC = 0.045

fig, ax = plt.subplots(figsize=(9, 5))
colors = ["red" if z == 6 else "steelblue" for z in z_vals]
bars = ax.bar(z_vals, dcp_vals, color=colors, edgecolor="black", linewidth=0.8, width=0.6)
ax.axhline(EXP_DCP, color="black", ls="-", lw=1.5, label=f"Exp δ_CP = {EXP_DCP} rad")
ax.axhline(EXP_DCP + DCP_UNC, color="gray", ls="--", lw=0.8)
ax.axhline(EXP_DCP - DCP_UNC, color="gray", ls="--", lw=0.8, label="±1σ")
ax.fill_between([z_vals[0]-0.5, z_vals[-1]+0.5], EXP_DCP-DCP_UNC, EXP_DCP+DCP_UNC,
                alpha=0.15, color="green")
for z, v, sig in zip(z_vals, dcp_vals, dcp_errs):
    ax.text(z, v + 0.006, f"{sig:.2f}σ", ha="center", fontsize=8,
            color="red" if z == 6 else "black")
ax.set_xlabel("Coordination number z")
ax.set_ylabel("Predicted δ_CP [rad]")
ax.set_title("δ_CP (CKM CP-Phase) vs Coordination Number z\n"
             "Formula: π/2 − 1/√(z+1)  [INDEPENDENT of z/2 in 1/α — different functional form]\n"
             "z=6 gives the best match (0.07σ)")
ax.legend()
ax.set_ylim(0.9, 1.35)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "independence_dcp_vs_z.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: independence_dcp_vs_z.png")

# -- Plot 3: Combined 3-observable score vs prime --
c_primes = [int(r["p"]) for r in combined]
c_scores = [f(r["composite_3obs_score"]) for r in combined]
c_alpha_errs = [f(r["inv_alpha_0_frac_err"]) * 1e6 for r in combined]
c_ns_errs = [f(r["n_s_frac_err"]) * 100 for r in combined]
c_dcp_errs = [f(r["delta_CP_frac_err"]) * 100 for r in combined]

fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)

axes[0].plot(c_primes, c_scores, "purple", marker="o", ms=4, lw=1.5)
axes[0].axvline(104729, color="red", ls="--", lw=1.5, alpha=0.8, label="p=104729")
best_p = c_primes[c_scores.index(min(c_scores))]
axes[0].axvline(best_p, color="green", ls=":", lw=1.5, alpha=0.8, label=f"Best p={best_p}")
axes[0].set_ylabel("Combined 3-obs score\n(lower=better)")
axes[0].set_title("Combined 3-Observable Score (1/α + n_s + δ_CP) vs Prime\n"
                  "With genuinely independent observables, a local minimum appears")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(c_primes, c_alpha_errs, "b-o", ms=4, lw=1.5)
axes[1].axvline(104729, color="red", ls="--", lw=1.5, alpha=0.8)
axes[1].axvline(best_p, color="green", ls=":", lw=1.5, alpha=0.8)
axes[1].set_ylabel("1/α error (ppm)\n← DECREASES as p ↑")
axes[1].grid(True, alpha=0.3)

axes[2].plot(c_primes, c_ns_errs, "g-s", ms=4, lw=1.5)
axes[2].axvline(104729, color="red", ls="--", lw=1.5, alpha=0.8)
axes[2].axvline(best_p, color="green", ls=":", lw=1.5, alpha=0.8)
axes[2].set_ylabel("n_s error (%)\n← INCREASES as p ↑")
axes[2].grid(True, alpha=0.3)

axes[3].plot(c_primes, c_dcp_errs, "r-^", ms=4, lw=1.5)
axes[3].axvline(104729, color="red", ls="--", lw=1.5, alpha=0.8)
axes[3].axvline(best_p, color="green", ls=":", lw=1.5, alpha=0.8)
axes[3].set_ylabel("δ_CP error (%)\n(constant — no p dependence)")
axes[3].set_xlabel("Substrate prime p")
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "independence_combined_score.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: independence_combined_score.png")

# -- Plot 4: Wide landscape showing where n_s and 1/alpha disagree --
# Analytical computation
test_primes_range = range(70000, 120000, 100)
from bpr.alpha_derivation import inverse_alpha_from_substrate
GAMMA = 0.5772156649
EXP_ALPHA = 137.035999084

def ns_pred(p):
    N = p**(1/3) * 4/3
    return 1 - 2/N

def alpha_frac_err(p, z=6):
    inv_a = np.log(p)**2 + z/2 + GAMMA - 1/(2*np.pi)
    return abs(inv_a - EXP_ALPHA) / EXP_ALPHA

p_range = list(test_primes_range)
ns_ferr = [abs(ns_pred(p) - 0.9649)/0.9649 for p in p_range]
alpha_ferr = [alpha_frac_err(p) for p in p_range]
combined_ferr = [np.sqrt((a**2 + n**2)/2) for a, n in zip(alpha_ferr, ns_ferr)]

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axes[0].semilogy(p_range, alpha_ferr, "b-", lw=1.5, label="1/α frac error")
axes[0].semilogy(p_range, ns_ferr, "g-", lw=1.5, label="n_s frac error")
axes[0].semilogy(p_range, combined_ferr, "purple", lw=2, label="Combined RMS")
axes[0].axvline(104729, color="red", ls="--", lw=1.5, alpha=0.8, label="p=104729")
opt_p = p_range[combined_ferr.index(min(combined_ferr))]
axes[0].axvline(opt_p, color="orange", ls=":", lw=1.5, alpha=0.8,
                label=f"Combined optimum p≈{opt_p}")
axes[0].set_ylabel("Fractional error (log scale)")
axes[0].set_title("Wide-range landscape: 1/α vs n_s — Two Observables Disagree\n"
                  "1/α prefers larger p; n_s prefers smaller p → minimum somewhere between")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3, which="both")

axes[1].plot(p_range, combined_ferr, "purple", lw=2)
axes[1].axvline(104729, color="red", ls="--", lw=1.5, alpha=0.8, label="p=104729")
axes[1].axvline(opt_p, color="orange", ls=":", lw=1.5, alpha=0.8,
                label=f"True minimum p≈{opt_p}")
min_val = min(combined_ferr)
axes[1].axhline(min_val, color="orange", ls=":", lw=0.8, alpha=0.5)
axes[1].set_xlabel("Substrate prime p")
axes[1].set_ylabel("Combined RMS error")
axes[1].set_title(f"Combined optimum is at p≈{opt_p}, NOT at p=104729")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "independence_wide_landscape.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: independence_wide_landscape.png")
print(f"\nTrue combined optimum (1/α + n_s): p ≈ {opt_p}")
print(f"Baseline p=104729 is {abs(104729 - opt_p)} units from the true optimum")
