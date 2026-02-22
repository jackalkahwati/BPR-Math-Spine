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
    print()
    print(f"  {BOLD}Universal Phase Transition Taxonomy derivation of z_PT:{RESET}")
    print(f"    Γ_b(z_PT) = ω_MOND  →  H(z_PT)/p^{{1/3}} = a₀/c")
    print(f"    Solved in ΛCDM:  z_PT = {z_pt:.2f}  (zero free parameters)")
    print(f"    At z < z_PT: Newtonian gravity, dissipation from z_PT only")
    print(f"    At z > z_PT: MOND active (δ_c=1.33), dissipation suspended")
    print(f"  {BOLD}Substrate Zone-Boundary Enhancement derivation of k_★:{RESET}")
    print(f"    k_★ = 2π p^{{2/3}} H₀/c  →  {k_star:.2f} Mpc⁻¹  (zero free parameters)")
    print(f"    f_ZB(k) = logistic step; sharpness p^{{1/3}} ≈ 47")
    print(f"    f_ZB(0.2) ≈ 0  →  σ₈ unchanged;  f_ZB(5) ≈ 1  →  σ enhanced by 4/3")

    # ── 3. JWST UV Luminosity Function ─────────────────────────────────────
    print_header("ANOMALY 3 — JWST 'Too-Early Galaxies'  (UV luminosity function z=9–16)")
    mond_a0 = results["bpr_params"]["mond_a0_m_s2"]
    k_star  = results["bpr_params"]["k_star_Mpc"]
    print(f"  Vacuum Impedance Mismatch MOND:  a₀ = {mond_a0:.2e} m/s²  (observed: 1.2×10⁻¹⁰, off by "
          f"{abs(mond_a0/1.2e-10 - 1)*100:.1f}%)")
    print(f"  Universal Phase Transition Taxonomy z_PT = {z_pt:.2f}: MOND active at z > z_PT, Newtonian at z < z_PT")
    print(f"  Substrate Zone-Boundary Enhancement k_★ = {k_star:.2f} Mpc⁻¹  "
          f"(= 2π p^{{2/3}} H₀/c, galaxy formation band)")
    print(f"  At M~10¹¹ M☉: a_char ≈ 10⁻¹⁴ m/s² ≪ a₀  →  deep MOND  →  δ_c = 1.33\n")
    print(f"  {'z':>4}  {'M_UV':>6}  {'obs':>7}  {'ΛCDM':>7}  {'BPR':>7}  "
          f"{'MOND':>7}  {'V2':>7}  {'V3':>7}  {'gap':>6}  {'BPR':>6}  {'MOND':>6}  {'V2':>6}  {'V3':>6}  source")
    print(f"  {'-'*4}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  "
          f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*15}")

    n_points       = len(results["jwst_uv_lf"])
    total_gap_lcdm = 0.0
    total_gap_bpr  = 0.0
    total_gap_mond = 0.0
    total_gap_v2   = 0.0
    total_gap_v3   = 0.0

    def _frac_col(f: float) -> str:
        c = GREEN if 0.3 < f <= 1.0 else (YELLOW if f > 0 else RED)
        return f"{c}{f*100:>+5.0f}%{RESET}"

    for row in results["jwst_uv_lf"]:
        gap_lcdm = row["gap_lcdm_dex"]
        gap_bpr  = row["gap_bpr_dex"]
        gap_mond = row["gap_mond_dex"]
        gap_v2   = row["gap_v2_dex"]
        gap_v3   = row["gap_v3_dex"]
        total_gap_lcdm += abs(gap_lcdm)
        total_gap_bpr  += abs(gap_bpr)
        total_gap_mond += abs(gap_mond)
        total_gap_v2   += abs(gap_v2)
        total_gap_v3   += abs(gap_v3)

        print(
            f"  {row['z']:>4.0f}  {row['M_UV']:>6.1f}  "
            f"{row['log_phi_obs']:>7.2f}  "
            f"{row['log_phi_lcdm']:>7.2f}  "
            f"{row['log_phi_bpr']:>7.2f}  "
            f"{row['log_phi_mond']:>7.2f}  "
            f"{row['log_phi_v2']:>7.2f}  "
            f"{row['log_phi_v3']:>7.2f}  "
            f"{gap_lcdm:>+6.2f}  "
            f"{_frac_col(row['bpr_fraction_closed'])}  "
            f"{_frac_col(row['mond_fraction_closed'])}  "
            f"{_frac_col(row['v2_fraction_closed'])}  "
            f"{_frac_col(row['v3_fraction_closed'])}  "
            f"{row['source']}"
        )

    def _mean_pct(total_miss: float) -> float:
        return (total_gap_lcdm - total_miss) / total_gap_lcdm * 100

    print()
    print(f"  Average ΛCDM gap:  {total_gap_lcdm/n_points:.2f} dex")
    print(f"  Standard BPR:      closes {_mean_pct(total_gap_bpr):+.1f}%  "
          f"(gap → {total_gap_bpr/n_points:.2f} dex)")
    print(f"  MOND (no z_PT):    closes {_mean_pct(total_gap_mond):+.1f}%  "
          f"(gap → {total_gap_mond/n_points:.2f} dex)")
    print(f"  V2  (z_PT={z_pt:.1f}):   closes {_mean_pct(total_gap_v2):+.1f}%  "
          f"(gap → {total_gap_v2/n_points:.2f} dex)")
    print(f"  V3  (k_★={k_star:.2f}):  closes {_mean_pct(total_gap_v3):+.1f}%  "
          f"(gap → {total_gap_v3/n_points:.2f} dex)")

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
    bpr_mean  = _mean_pct(total_gap_bpr)
    mond_mean = _mean_pct(total_gap_mond)
    v2_mean   = _mean_pct(total_gap_v2)
    v3_mean   = _mean_pct(total_gap_v3)
    print(f"""
  {BOLD}MECHANISM COMPARISON — three JWST-era anomalies:{RESET}

  Hubble tension (4.9σ):
    Standard BPR:                          closes  ~7%   (ΔNeff = {results['bpr_params']['delta_Neff']:.3f}, needs ~0.4)
    Vacuum Impedance Mismatch MOND:        same   ~7%   (MOND does not shift sound horizon)
    BPR V2 (Phase Transition):             same   ~7%   (transition does not affect ΔNeff)
    BPR V3 (+ Zone-Boundary):             same   ~7%   (k_★ affects matter only, not ΔNeff)
    Needed: 11× more boundary radiation species

  S8 tension (3.3σ):
    Standard BPR:                          {s8_std_str}
    Vacuum Impedance (no z_PT):            WORSENS  (MOND at z=0 boosts 8 Mpc clustering)
    {BOLD}BPR V2 (Phase Transition):{RESET}             {GREEN}{s8_v2_str}{RESET}
    {BOLD}BPR V3 (+ Zone-Boundary):{RESET}              {GREEN}{s8_v3_str}{RESET}
    → k_★ = {k_star:.2f} Mpc⁻¹ ≫ k_σ₈ = 0.2 Mpc⁻¹  →  S8 scale unaffected by zone boundary
    → σ₈_V3 ≈ σ₈_V2  (zone-boundary does not degrade the S8 solution)

  JWST UV LF (z=9–16):
    Standard BPR:                          closes {bpr_mean:+.0f}%   (dissipation suppresses early structure)
    Vacuum Impedance (no z_PT):           {YELLOW}closes {mond_mean:+.0f}%{RESET}  (δ_c=1.33, but worsens S8)
    BPR V2 (Phase Transition):            {YELLOW}closes {v2_mean:+.0f}%{RESET}  (MOND at z>{z_pt:.1f}, S8 preserved)
    {BOLD}BPR V3 (+ Zone-Boundary):{RESET}             {YELLOW}closes {v3_mean:+.0f}%{RESET}  (V2 + k_★ scale cutoff)
    CAVEAT: z=10 bright-end overshoot persists — PS exponential tail
    is sensitive to Δδ_c at high ν; exact amplitude requires N-body.

  {BOLD}Substrate Zone-Boundary Enhancement — derivation:{RESET}
  p^{{2/3}} = {int(round(results['bpr_params']['p']**(2/3)))} → R_H/p^{{2/3}} = 2.00 Mpc → k_★ = 2π p^{{2/3}} H₀/c = {k_star:.2f} Mpc⁻¹
  Enhancement: σ_eff(k) = σ_V1(k)×[1 + (1/3)×f_ZB(k)]  (1/3 = 1/d, d=3)
  Logistic sharpness p^{{1/3}} ≈ 47  →  f_ZB(0.2)≈0,  f_ZB(5)≈1  (zero params)

  {BOLD}Combined V3 — what it achieves:{RESET}
  • Hubble:     unchanged (~7%)
  • S8 tension: 3.3σ → {s8['tension_sigma_bpr_v3']:.1f}σ  ({GREEN}closes {frac_v3*100:.0f}%{RESET})  [same as V2; k_★ preserves σ₈]
  • JWST UV LF: closes {v3_mean:+.0f}% on average  \
({RED if v3_mean < v2_mean else GREEN}{v3_mean - v2_mean:+.0f}% vs V2{RESET})

  {BOLD}Honest diagnosis of V3 JWST result:{RESET}
  V3 WORSENS the average JWST fit relative to V2 because V2 already overshoots
  at z=10 (by ~1 dex from MOND δ_c=1.33) and the zone-boundary boost amplifies
  those points further.  V3 correctly helps z=9 where V2 undershoots, but the
  net effect on total |gap| is negative because z=10 dominates.
  Root cause: MOND at z>z_PT is uniformly applied at all halo masses, making
  it too strong at the bright end (M_UV < -22).

  {BOLD}Remaining gap to a full solution:{RESET}
  A mass-dependent collapse threshold δ_c(M) that reverts toward 1.686 for
  M > 10¹² M☉ halos would selectively suppress the z=10 bright-end overshoot
  while preserving the z=12–16 enhancement.  The zone-boundary k_★ derivation
  stands as a genuine BPR prediction; combining it with a mass-dependent
  MOND onset is the next derivation target.
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
