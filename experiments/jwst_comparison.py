"""
BPR vs JWST: Honest Comparison
================================

Runs BPR predictions against three JWST-era anomalies and prints a
plain-English verdict on what BPR explains, what it doesn't, and what
new physics would be required.

Usage
-----
    python experiments/jwst_comparison.py
    python experiments/jwst_comparison.py --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bpr.jwst_cosmology import run_jwst_comparison

# ── ANSI colours ──────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def _bar(fraction: float, width: int = 30) -> str:
    """Simple ASCII progress bar for fraction [0, 1]."""
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    colour = GREEN if fraction > 0.5 else (YELLOW if fraction > 0.2 else RED)
    bar = "█" * filled + "░" * (width - filled)
    return f"{colour}{bar}{RESET} {fraction*100:.1f}%"


def print_header(title: str) -> None:
    print(f"\n{BOLD}{BLUE}{'━'*66}")
    print(f"  {title}")
    print(f"{'━'*66}{RESET}")


def run(args: argparse.Namespace) -> None:
    results = run_jwst_comparison()

    if args.json:
        print(json.dumps(results, indent=2))
        return

    # ── 1. Hubble Tension ─────────────────────────────────────────────────
    print_header("ANOMALY 1 — Hubble Tension  (H₀ local vs CMB)")
    ht = results["hubble_tension"]
    print(f"  Planck CMB:   {ht['H0_planck_km_s_Mpc']:.1f} km/s/Mpc")
    print(f"  Local (Riess):{ht['H0_local_km_s_Mpc']:.2f} km/s/Mpc")
    print(f"  Tension:      {ht['tension_sigma']:.1f}σ")
    print()
    print(f"  BPR ΔNeff:    {ht['delta_Neff']:.4f}  (shifts H₀ via sound horizon)")
    print(f"  BPR H₀:       {ht['H0_bpr_km_s_Mpc']:.3f} km/s/Mpc")
    print(f"  BPR explains: {_bar(ht['fraction_explained'])}")
    print()
    print(f"  {YELLOW}Verdict:{RESET} BPR moves H₀ in the right direction but explains only")
    print(f"  {ht['fraction_explained']*100:.1f}% of the tension.  Full resolution would require")
    print(f"  ΔNeff ≈ 0.40 (BPR predicts {ht['delta_Neff']:.3f}).  To match H₀ local,")
    print(f"  BPR would need ~11× more effective radiation than it currently derives.")

    # ── 2. S8 Tension ─────────────────────────────────────────────────────
    print_header("ANOMALY 2 — S8 Tension  (weak lensing vs Planck+ΛCDM)")
    s8 = results["s8_tension"]
    z_pt = results["bpr_params"]["z_pt"]
    print(f"  Planck+ΛCDM:  S₈ = {s8['S8_planck']:.3f},  σ₈ = {0.811:.3f}")
    print(f"  Weak lensing: S₈ = {s8['S8_observed_WL']:.3f} ± 0.020")
    print(f"  Tension:      {s8['tension_sigma_lcdm']:.1f}σ  (Planck over-predicts clustering)")
    print()
    print(f"  {'Model':<22}  {'σ₈':>6}  {'S₈':>6}  {'Residual':>10}  Verdict")
    print(f"  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*30}")
    print(f"  {'Planck+ΛCDM':<22}  {0.811:>6.4f}  {s8['S8_planck']:>6.4f}  "
          f"{'3.3σ above WL':>10}")
    print(f"  {'Weak-lensing obs':<22}  {'0.748':>6}  {s8['S8_observed_WL']:>6.4f}  "
          f"{'target':>10}")
    print()

    # Standard BPR row
    overshoots = s8.get("overshoots", False)
    frac = s8["fraction_explained"]
    if overshoots:
        bpr_verdict = f"{RED}OVERSHOOTS (too much suppression){RESET}"
    elif frac > 0.4:
        bpr_verdict = f"{GREEN}closes {frac*100:.0f}% of tension{RESET}"
    else:
        bpr_verdict = f"{YELLOW}closes {frac*100:.0f}% of tension{RESET}"
    print(f"  {'Standard BPR':<22}  {s8['sigma8_bpr']:>6.4f}  {s8['S8_bpr']:>6.4f}  "
          f"{s8['tension_sigma_bpr']:>8.2f}σ  {bpr_verdict}")

    # V2 row
    frac_v2 = s8["fraction_explained_v2"]
    overshoots_v2 = s8.get("overshoots_v2", False)
    if overshoots_v2:
        v2_verdict = f"{RED}OVERSHOOTS{RESET}"
    elif frac_v2 > 0.4:
        v2_verdict = f"{GREEN}closes {frac_v2*100:.0f}% of tension{RESET}"
    elif frac_v2 > 0.2:
        v2_verdict = f"{YELLOW}closes {frac_v2*100:.0f}% of tension{RESET}"
    else:
        v2_verdict = f"{RED}closes {frac_v2*100:.0f}% of tension{RESET}"
    print(f"  {f'BPR V2 (z_PT={z_pt:.1f})':<22}  {s8['sigma8_bpr_v2']:>6.4f}  "
          f"{s8['S8_bpr_v2']:>6.4f}  {s8['tension_sigma_bpr_v2']:>8.2f}σ  {v2_verdict}")

    # V3 row
    frac_v3 = s8["fraction_explained_v3"]
    overshoots_v3 = s8.get("overshoots_v3", False)
    if overshoots_v3:
        v3_verdict = f"{RED}OVERSHOOTS{RESET}"
    elif frac_v3 > 0.4:
        v3_verdict = f"{GREEN}closes {frac_v3*100:.0f}% of tension{RESET}"
    elif frac_v3 > 0.2:
        v3_verdict = f"{YELLOW}closes {frac_v3*100:.0f}% of tension{RESET}"
    else:
        v3_verdict = f"{RED}closes {frac_v3*100:.0f}% of tension{RESET}"
    k_star = results["bpr_params"]["k_star_Mpc"]
    print(f"  {f'BPR V3 (k_★={k_star:.2f})':<22}  {s8['sigma8_bpr_v3']:>6.4f}  "
          f"{s8['S8_bpr_v3']:>6.4f}  {s8['tension_sigma_bpr_v3']:>8.2f}σ  {v3_verdict}")

    # V4 row
    frac_v4 = s8["fraction_explained_v4"]
    overshoots_v4 = s8.get("overshoots_v4", False)
    if overshoots_v4:
        v4_verdict = f"{RED}OVERSHOOTS{RESET}"
    elif frac_v4 > 0.4:
        v4_verdict = f"{GREEN}closes {frac_v4*100:.0f}% of tension{RESET}"
    elif frac_v4 > 0.2:
        v4_verdict = f"{YELLOW}closes {frac_v4*100:.0f}% of tension{RESET}"
    else:
        v4_verdict = f"{RED}closes {frac_v4*100:.0f}% of tension{RESET}"
    print(f"  {'BPR V4 (impedance μ)':<22}  {s8['sigma8_bpr_v4']:>6.4f}  "
          f"{s8['S8_bpr_v4']:>6.4f}  {s8['tension_sigma_bpr_v4']:>8.2f}σ  {v4_verdict}")

    # V5 row
    frac_v5 = s8["fraction_explained_v5"]
    overshoots_v5 = s8.get("overshoots_v5", False)
    if overshoots_v5:
        v5_verdict = f"{RED}OVERSHOOTS{RESET}"
    elif frac_v5 > 0.4:
        v5_verdict = f"{GREEN}closes {frac_v5*100:.0f}% of tension{RESET}"
    elif frac_v5 > 0.2:
        v5_verdict = f"{YELLOW}closes {frac_v5*100:.0f}% of tension{RESET}"
    else:
        v5_verdict = f"{RED}closes {frac_v5*100:.0f}% of tension{RESET}"
    print(f"  {'BPR V5 (impedance screen)':<22}  {s8['sigma8_bpr_v5']:>6.4f}  "
          f"{s8['S8_bpr_v5']:>6.4f}  {s8['tension_sigma_bpr_v5']:>8.2f}σ  {v5_verdict}")
    print()
    print(f"  {BOLD}Universal Phase Transition Taxonomy derivation of z_PT:{RESET}")
    print(f"    Γ_b(z_PT) = ω_MOND  →  H(z_PT)/p^{{1/3}} = a₀/c")
    print(f"    Solved in ΛCDM:  z_PT = {z_pt:.2f}  (zero free parameters)")
    print(f"    At z < z_PT: Newtonian gravity, dissipation from z_PT only")
    print(f"    At z > z_PT: MOND active (δ_c=1.33), dissipation suspended")
    ms9  = results["bpr_params"]["m_star_v4_z9"]
    ms10 = results["bpr_params"]["m_star_v4_z10"]
    ms12 = results["bpr_params"]["m_star_v4_z12"]
    print(f"  {BOLD}Impedance-Weighted Collapse Threshold (V4) — M★(z):{RESET}")
    print(f"    δ_c(M,z) = 1.33 + 0.356×μ(a_vir/a₀),  μ = x/√(1+x²)")
    print(f"    M★ = (a₀/G)³/(800π ρ_crit(z)/3)²  (zero free parameters)")
    print(f"    M★(z=9)  = {ms9:.2e} M☉  — z=9 halos below M★ → MOND-like")
    print(f"    M★(z=10) = {ms10:.2e} M☉  — z=10 bright halos near M★ → transition")
    print(f"    M★(z=12) = {ms12:.2e} M☉  — z=12 halos above M★ → Newtonian")
    W_c  = results["bpr_params"]["W_c_v5"]
    mi9  = results["bpr_params"]["m_imp_v5_z9"]
    mi10 = results["bpr_params"]["m_imp_v5_z10"]
    mi12 = results["bpr_params"]["m_imp_v5_z12"]
    print(f"  {BOLD}Impedance-Screened MOND (V5) — M_imp(z) = W_c^{{1/2}} × M★:{RESET}")
    print(f"    g_screen(M,z) = 1/(1 + (M/M★)⁴/W_c²),  W_c = p^{{1/5}} = {W_c:.2f}")
    print(f"    M_imp(z=9)  = {mi9:.2e} M☉  — halos above M_imp: impedance-screened")
    print(f"    M_imp(z=10) = {mi10:.2e} M☉  ≈ 10¹² M☉  (bright-end overshoot regime)")
    print(f"    M_imp(z=12) = {mi12:.2e} M☉  — z=12 low-mass halos: unaffected")

    # ── 3. JWST UV Luminosity Function ─────────────────────────────────────
    print_header("ANOMALY 3 — JWST 'Too-Early Galaxies'  (UV luminosity function z=9–16)")
    mond_a0 = results["bpr_params"]["mond_a0_m_s2"]
    k_star  = results["bpr_params"]["k_star_Mpc"]
    print(f"  Vacuum Impedance Mismatch MOND:  a₀ = {mond_a0:.2e} m/s²  (observed: 1.2×10⁻¹⁰, off by "
          f"{abs(mond_a0/1.2e-10 - 1)*100:.1f}%)")
    print(f"  Universal Phase Transition Taxonomy z_PT = {z_pt:.2f}: MOND active at z > z_PT, Newtonian at z < z_PT")
    print(f"  V4 M★(z=9/10/12) = {results['bpr_params']['m_star_v4_z9']:.1e} / "
          f"{results['bpr_params']['m_star_v4_z10']:.1e} / "
          f"{results['bpr_params']['m_star_v4_z12']:.1e} M☉  "
          f"(continuous δ_c via μ(a_vir/a₀))")
    print(f"  V5 M_imp(z=9/10/12) = {results['bpr_params']['m_imp_v5_z9']:.1e} / "
          f"{results['bpr_params']['m_imp_v5_z10']:.1e} / "
          f"{results['bpr_params']['m_imp_v5_z12']:.1e} M☉  "
          f"(impedance screening crossover, W_c={results['bpr_params']['W_c_v5']:.2f})\n")
    print(f"  {'z':>4}  {'M_UV':>6}  {'obs':>7}  {'ΛCDM':>7}  {'V2':>7}  "
          f"{'V3':>7}  {'V4':>7}  {'V5':>7}  {'gap':>6}  {'V2':>6}  {'V3':>6}  {'V4':>6}  {'V5':>6}  source")
    print(f"  {'-'*4}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  "
          f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*15}")

    n_points       = len(results["jwst_uv_lf"])
    total_gap_lcdm = 0.0
    total_gap_bpr  = 0.0
    total_gap_mond = 0.0
    total_gap_v2   = 0.0
    total_gap_v3   = 0.0
    total_gap_v4   = 0.0
    total_gap_v5   = 0.0

    def _frac_col(f: float) -> str:
        c = GREEN if 0.3 < f <= 1.0 else (YELLOW if f > 0 else RED)
        return f"{c}{f*100:>+5.0f}%{RESET}"

    for row in results["jwst_uv_lf"]:
        gap_lcdm = row["gap_lcdm_dex"]
        gap_bpr  = row["gap_bpr_dex"]
        gap_mond = row["gap_mond_dex"]
        gap_v2   = row["gap_v2_dex"]
        gap_v3   = row["gap_v3_dex"]
        gap_v4   = row["gap_v4_dex"]
        gap_v5   = row["gap_v5_dex"]
        total_gap_lcdm += abs(gap_lcdm)
        total_gap_bpr  += abs(gap_bpr)
        total_gap_mond += abs(gap_mond)
        total_gap_v2   += abs(gap_v2)
        total_gap_v3   += abs(gap_v3)
        total_gap_v4   += abs(gap_v4)
        total_gap_v5   += abs(gap_v5)

        print(
            f"  {row['z']:>4.0f}  {row['M_UV']:>6.1f}  "
            f"{row['log_phi_obs']:>7.2f}  "
            f"{row['log_phi_lcdm']:>7.2f}  "
            f"{row['log_phi_v2']:>7.2f}  "
            f"{row['log_phi_v3']:>7.2f}  "
            f"{row['log_phi_v4']:>7.2f}  "
            f"{row['log_phi_v5']:>7.2f}  "
            f"{gap_lcdm:>+6.2f}  "
            f"{_frac_col(row['v2_fraction_closed'])}  "
            f"{_frac_col(row['v3_fraction_closed'])}  "
            f"{_frac_col(row['v4_fraction_closed'])}  "
            f"{_frac_col(row['v5_fraction_closed'])}  "
            f"{row['source']}"
        )

    def _mean_pct(total_miss: float) -> float:
        return (total_gap_lcdm - total_miss) / total_gap_lcdm * 100

    print()
    print(f"  Average ΛCDM gap:  {total_gap_lcdm/n_points:.2f} dex")
    print(f"  V2  (z_PT={z_pt:.1f}):        closes {_mean_pct(total_gap_v2):+.1f}%  "
          f"(gap → {total_gap_v2/n_points:.2f} dex)")
    print(f"  V3  (k_★={k_star:.2f}):       closes {_mean_pct(total_gap_v3):+.1f}%  "
          f"(gap → {total_gap_v3/n_points:.2f} dex)  [V2+ZB, worsens average]")
    print(f"  V4  (μ interp):        closes {_mean_pct(total_gap_v4):+.1f}%  "
          f"(gap → {total_gap_v4/n_points:.2f} dex)")
    v5_delta = _mean_pct(total_gap_v5) - _mean_pct(total_gap_v4)
    v5_color = GREEN if v5_delta > 0 else RED
    print(f"  V5  (impedance screen): closes {_mean_pct(total_gap_v5):+.1f}%  "
          f"(gap → {total_gap_v5/n_points:.2f} dex)  "
          f"[{v5_color}{v5_delta:+.1f}% vs V4{RESET}]")

    # ── Overall verdict ────────────────────────────────────────────────────
    print_header("OVERALL VERDICT")
    s8_overshoot = s8.get("overshoots", False)
    s8_std_str = (f"overshoots by {(frac-1)*100:.0f}%  (σ₈={s8['sigma8_bpr']:.3f} < 0.748)"
                  if s8_overshoot else f"closes {frac*100:.0f}%")
    s8_v2_str = (f"overshoots"
                 if overshoots_v2 else f"closes {frac_v2*100:.0f}%  "
                 f"(σ₈={s8['sigma8_bpr_v2']:.3f}, {s8['tension_sigma_bpr_v2']:.1f}σ residual)")
    frac_v3      = s8["fraction_explained_v3"]
    overshoots_v3 = s8.get("overshoots_v3", False)
    s8_v3_str = (f"overshoots"
                 if overshoots_v3 else f"closes {frac_v3*100:.0f}%  "
                 f"(σ₈={s8['sigma8_bpr_v3']:.3f}, {s8['tension_sigma_bpr_v3']:.1f}σ residual)")
    s8_v4_str = (f"overshoots"
                 if overshoots_v4 else f"closes {frac_v4*100:.0f}%  "
                 f"(σ₈={s8['sigma8_bpr_v4']:.3f}, {s8['tension_sigma_bpr_v4']:.1f}σ residual)")
    bpr_mean  = _mean_pct(total_gap_bpr)
    mond_mean = _mean_pct(total_gap_mond)
    v2_mean   = _mean_pct(total_gap_v2)
    v3_mean   = _mean_pct(total_gap_v3)
    v4_mean   = _mean_pct(total_gap_v4)
    v5_mean   = _mean_pct(total_gap_v5)
    s8_v5_str = (f"overshoots"
                 if overshoots_v5 else f"closes {frac_v5*100:.0f}%  "
                 f"(σ₈={s8['sigma8_bpr_v5']:.3f}, {s8['tension_sigma_bpr_v5']:.1f}σ residual)")
    print(f"""
  {BOLD}MECHANISM COMPARISON — three JWST-era anomalies:{RESET}

  Hubble tension (4.9σ):
    All BPR mechanisms:  closes ~7%  (ΔNeff = {results['bpr_params']['delta_Neff']:.3f}, needs ~0.4)
    Needed: 11× more boundary radiation species.  Unsolved.

  S8 tension (3.3σ):
    Standard BPR:                          {s8_std_str}
    Vacuum Impedance MOND (no z_PT):       WORSENS  (MOND at z=0 boosts 8 Mpc clustering)
    {BOLD}BPR V2 (Phase Transition):{RESET}             {GREEN}{s8_v2_str}{RESET}
    {BOLD}BPR V3 (+ Zone-Boundary):{RESET}              {GREEN}{s8_v3_str}{RESET}
    {BOLD}BPR V4 (impedance μ):{RESET}                  {GREEN}{s8_v4_str}{RESET}
    {BOLD}BPR V5 (impedance screen):{RESET}             {GREEN}{s8_v5_str}{RESET}
    → V5 S8 = V4 S8 (screening only at z > z_PT bright end, z=0 unchanged)

  JWST UV LF (z=9–16):
    Standard BPR:                          closes {bpr_mean:+.0f}%
    Vacuum Impedance MOND (no z_PT):      {YELLOW}closes {mond_mean:+.0f}%{RESET}  (worsens S8)
    BPR V2 (Phase Transition):            {YELLOW}closes {v2_mean:+.0f}%{RESET}  (S8 preserved)
    BPR V3 (+ Zone-Boundary):            {YELLOW}closes {v3_mean:+.0f}%{RESET}  (worsens average vs V2)
    {BOLD}BPR V4 (impedance μ):{RESET}                 \
{GREEN if v4_mean > v2_mean else YELLOW}closes {v4_mean:+.0f}%{RESET}  \
({GREEN if v4_mean > v2_mean else RED}{v4_mean - v2_mean:+.0f}% vs V2{RESET}, S8 preserved)
    {BOLD}BPR V5 (impedance screen):{RESET}            \
{GREEN if v5_mean > v4_mean else YELLOW}closes {v5_mean:+.0f}%{RESET}  \
({GREEN if v5_mean > v4_mean else RED}{v5_mean - v4_mean:+.0f}% vs V4{RESET}, S8 preserved)

  {BOLD}V5 Impedance-Screened MOND — derivation:{RESET}
  δ_c(M,z) = 1.33 + 0.356×μ(a_vir/a₀)×g_screen(M,z)
  g_screen  = 1 / (1 + (M/M★)⁴/W_c²),  W_c = p^{{1/5}} = {results['bpr_params']['W_c_v5']:.2f}
  M_imp(z)  = W_c^{{1/2}}×M★(z)  (crossover mass, g_screen=1/2):
    z=9:  M_imp={results['bpr_params']['m_imp_v5_z9']:.2e} M☉
    z=10: M_imp={results['bpr_params']['m_imp_v5_z10']:.2e} M☉  ← targets bright-end overshoot
    z=12: M_imp={results['bpr_params']['m_imp_v5_z12']:.2e} M☉

  {BOLD}Best current state (V5):{RESET}
  • Hubble:     {s8['tension_sigma_bpr_v5']:.1f}σ residual  (7% closed — unsolved)
  • S8 tension: {GREEN}closes {frac_v5*100:.0f}%{RESET}  ({s8['tension_sigma_bpr_v5']:.1f}σ residual)
  • JWST UV LF: closes {v5_mean:+.0f}% on average  \
({GREEN if v5_mean > v4_mean else RED}{v5_mean - v4_mean:+.0f}% vs V4{RESET})

  {BOLD}Remaining gap:{RESET}
  The JWST z=9 undershoots and z=10–16 overshoots reflect the PS exponential
  tail's extreme sensitivity to δ_c at high-ν.  The impedance screening (V5)
  adds a second suppression of the bright end, but the semi-analytic PS model
  is not accurate enough for precision comparison — full N-body is required.
  Hubble tension remains untouched by all five mechanisms.
""")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BPR vs JWST: honest comparison of predictions to anomalies"
    )
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
