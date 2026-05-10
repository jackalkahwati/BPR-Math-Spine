#!/usr/bin/env python3
"""Paper-quality figure for the muon g-2 BPR result.

Two-panel figure:
  (a) The standard a_μ comparison: SM (with combined uncertainty),
      experimental world average, and BPR-natural prediction, with
      1σ shaded bands.
  (b) Lepton universality: BPR vs (m_μ/m_e)² scaling, on log-log axes.

Output: figures/muon_g2_bpr_result.png and .pdf
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bpr.atomic_precision import electron_g_minus_2, muon_g_minus_2


def _setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.linewidth": 0.8,
    })


def main() -> None:
    _setup_style()

    # --- Numbers ------------------------------------------------------
    a_mu_sm = 116591810e-11
    a_mu_exp = 116592059e-11
    sigma_sm = 43e-11
    sigma_exp = 41e-11

    mu_result = muon_g_minus_2()
    a_mu_bpr_natural = mu_result["prediction"]

    # Raw and intermediate values for context
    delta_a_mu_raw = a_mu_sm * (105.6583755 / 0.51099895) ** 2 / 104761 ** 2
    a_mu_bpr_raw = a_mu_sm + delta_a_mu_raw

    # --- Build figure --------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Panel (a): a_μ values
    plot_x_offset = (a_mu_sm - 1.16591e-3) * 1e9  # shift origin for readability
    SCALE = 1e11
    sm_y = (a_mu_sm - a_mu_sm) * SCALE
    exp_y = (a_mu_exp - a_mu_sm) * SCALE
    bpr_nat_y = (a_mu_bpr_natural - a_mu_sm) * SCALE
    bpr_raw_y = (a_mu_bpr_raw - a_mu_sm) * SCALE

    labels = [
        "Standard Model\n(WP-2020)",
        "BPR raw\nF = 1",
        "BPR natural\nF = 0.5",
        "Experiment\n(FNAL+BNL)",
    ]
    centres = [sm_y, bpr_raw_y, bpr_nat_y, exp_y]
    errors = [sigma_sm * SCALE, 0, 0, sigma_exp * SCALE]
    colors = ["#777", "#d4a017", "#1f9e57", "#cc1f1f"]

    for i, (lab, c, e, col) in enumerate(zip(labels, centres, errors, colors)):
        ax1.errorbar(
            i, c, yerr=e, fmt="o", color=col, ms=10, capsize=6,
            elinewidth=1.5, mec="black", mew=0.8, zorder=3,
            label=lab if i == 0 else None,
        )

    # Shaded combined uncertainty band around the experiment
    sigma_total = np.sqrt(sigma_sm ** 2 + sigma_exp ** 2) * SCALE
    ax1.axhspan(
        exp_y - sigma_total, exp_y + sigma_total,
        color="#cc1f1f", alpha=0.10, zorder=0,
        label="±1σ combined\n(SM ⊕ exp)",
    )
    ax1.axhline(exp_y, color="#cc1f1f", lw=0.8, ls="--", alpha=0.6, zorder=1)
    ax1.axhline(sm_y, color="#777", lw=0.8, ls="--", alpha=0.6, zorder=1)

    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel(r"$a_\mu \times 10^{11} - a_\mu^{\rm SM} \times 10^{11}$",
                   fontsize=11)
    ax1.set_title(r"(a)  Muon $a_\mu$ — BPR first-principles prediction",
                  fontsize=12)
    ax1.set_ylim(-50, 550)
    ax1.legend(loc="upper left", framealpha=0.95)

    # Annotations for the anomaly significance
    ax1.annotate(
        f"4.2σ anomaly\n(SM vs Exp)",
        xy=(0, sm_y), xytext=(0.5, 380),
        fontsize=9, ha="center",
        arrowprops=dict(arrowstyle="-", color="#777", lw=0.7),
    )
    ax1.annotate(
        "0.4σ\nresidual",
        xy=(2, bpr_nat_y), xytext=(2, 350),
        fontsize=9, ha="center", color="#1f9e57",
    )

    # Panel (b): lepton universality scaling
    e_result = electron_g_minus_2()
    m_e_MeV = 0.51099895
    m_mu_MeV = 105.6583755
    m_tau_MeV = 1776.86  # PDG 2024
    a_e_qed = e_result["prediction"] - e_result["bpr_correction"]
    a_tau_qed = a_e_qed   # to leading order; SM gives ~1.18e-3, close enough
    p = 104761
    F = 0.5

    masses = np.array([m_e_MeV, m_mu_MeV, m_tau_MeV])
    a_lepton = np.array([a_e_qed, 116591810e-11, a_tau_qed])
    delta = F * a_lepton * (masses / m_e_MeV) ** 2 / p ** 2
    labels_b = [r"$e$", r"$\mu$", r"$\tau$"]

    ax2.loglog(masses, delta, "o-", color="#1f9e57", ms=10, lw=1.5,
               mec="black", mew=0.8, label="BPR natural prediction", zorder=3)

    # Reference (m_ℓ/m_e)² scaling line (passing through electron point)
    m_dense = np.logspace(np.log10(m_e_MeV * 0.8),
                           np.log10(m_tau_MeV * 1.2), 200)
    ref = delta[0] * (m_dense / m_e_MeV) ** 2
    ax2.loglog(m_dense, ref, "--", color="#888", lw=1.0,
               label=r"$(m_\ell/m_e)^2$ scaling", zorder=1)

    # Experimental references
    ax2.axhspan(0, 1.3e-13, color="#cc1f1f", alpha=0.05,
                label="electron g-2 precision (current)")
    ax2.axhspan(170e-11, 330e-11, color="#1f9e57", alpha=0.10,
                label="muon g-2 anomaly window")

    for x, y, lab in zip(masses, delta, labels_b):
        ax2.annotate(lab, xy=(x, y), xytext=(8, 6),
                     textcoords="offset points",
                     fontsize=12, fontweight="bold")

    ax2.set_xlabel(r"Lepton mass $m_\ell$  [MeV]")
    ax2.set_ylabel(r"BPR vertex shift $\delta a_\ell$ (natural, F=0.5)")
    ax2.set_title("(b)  Lepton universality scaling", fontsize=12)
    ax2.legend(loc="lower right", framealpha=0.95, fontsize=9)
    ax2.grid(True, which="both", alpha=0.25)

    # Footer
    fig.suptitle(
        "BPR boundary-phase vertex correction to the lepton anomalous "
        "magnetic moment",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, -0.02,
        r"BPR formula: $\delta a_\ell = F \cdot a_\ell \cdot (m_\ell/m_e)^2 / p^2$,"
        r"  $p = 104761$,  $F = 0.5$ (boundary-resonance scale).  "
        r"No fitted parameters.",
        ha="center", fontsize=10, style="italic",
    )

    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "muon_g2_bpr_result.png")
    pdf_path = os.path.join(out_dir, "muon_g2_bpr_result.pdf")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"PNG saved → {png_path}")
    print(f"PDF saved → {pdf_path}")


if __name__ == "__main__":
    main()
