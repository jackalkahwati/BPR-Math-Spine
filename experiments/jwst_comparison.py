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
    print(f"  Planck+ΛCDM:  S₈ = {s8['S8_planck']:.3f},  σ₈ = {0.811:.3f}")
    print(f"  Weak lensing: S₈ = {s8['S8_observed_WL']:.3f} ± 0.020")
    print(f"  Tension:      {s8['tension_sigma_lcdm']:.1f}σ  (Planck over-predicts clustering)")
    print()
    print(f"  BPR σ₈:       {s8['sigma8_bpr']:.4f}")
    print(f"  BPR S₈:       {s8['S8_bpr']:.4f}")
    print(f"  Residual:     {s8['tension_sigma_bpr']:.2f}σ from WL observation")
    print()
    frac = s8["fraction_explained"]
    overshoots = s8.get("overshoots", False)
    if overshoots:
        pct_suppressed = (1.0 - s8["sigma8_bpr"] / 0.811) * 100.0
        verdict = f"{RED}BPR OVERSHOOTS: σ₈ = {s8['sigma8_bpr']:.4f} < WL obs ≈ 0.748.{RESET}"
        print(f"  Verdict: {verdict}")
        print(f"  BPR boundary dissipation suppresses late-time growth by {pct_suppressed:.1f}%,")
        print(f"  which is {frac*100 - 100:.0f}% more suppression than needed.  BPR passes through")
        print(f"  the WL target and ends up on the wrong side of the data.")
    elif frac > 0.8:
        verdict = f"{GREEN}BPR substantially resolves the S8 tension.{RESET}"
        print(f"  Verdict: {verdict}")
    elif frac > 0.3:
        verdict = f"{YELLOW}BPR partially reduces the S8 tension.{RESET}"
        print(f"  Verdict: {verdict}")
    else:
        verdict = f"{RED}BPR barely moves the S8 tension.{RESET}"
        print(f"  Verdict: {verdict}")

    # ── 3. JWST UV Luminosity Function ─────────────────────────────────────
    print_header("ANOMALY 3 — JWST 'Too-Early Galaxies'  (UV luminosity function z=9–16)")
    mond_a0 = results["bpr_params"]["mond_a0_m_s2"]
    print(f"  Theory II MOND:  a₀ = {mond_a0:.2e} m/s²  (observed: 1.2×10⁻¹⁰, off by "
          f"{abs(mond_a0/1.2e-10 - 1)*100:.1f}%)")
    print(f"  At M~10¹¹ M☉: a_char ≈ 10⁻¹⁴ m/s² ≪ a₀  →  deep MOND regime → δ_c = 1.33\n")
    print(f"  {'z':>4}  {'M_UV':>6}  {'obs':>7}  {'ΛCDM':>7}  {'BPR':>7}  {'MOND':>7}  "
          f"{'ΛCDM gap':>9}  {'BPR':>6}  {'MOND':>6}  source")
    print(f"  {'-'*4}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  "
          f"{'-'*9}  {'-'*6}  {'-'*6}  {'-'*18}")

    n_points       = len(results["jwst_uv_lf"])
    total_gap_lcdm = 0.0
    total_gap_bpr  = 0.0
    total_gap_mond = 0.0

    for row in results["jwst_uv_lf"]:
        gap_lcdm = row["gap_lcdm_dex"]
        gap_bpr  = row["gap_bpr_dex"]
        gap_mond = row["gap_mond_dex"]
        total_gap_lcdm += abs(gap_lcdm)
        total_gap_bpr  += abs(gap_bpr)
        total_gap_mond += abs(gap_mond)

        frac_bpr  = row["bpr_fraction_closed"]
        frac_mond = row["mond_fraction_closed"]

        def _frac_col(f: float) -> str:
            c = GREEN if 0.3 < f <= 1.0 else (YELLOW if f > 0 else RED)
            return f"{c}{f*100:>+5.0f}%{RESET}"

        print(
            f"  {row['z']:>4.0f}  {row['M_UV']:>6.1f}  "
            f"{row['log_phi_obs']:>7.2f}  "
            f"{row['log_phi_lcdm']:>7.2f}  "
            f"{row['log_phi_bpr']:>7.2f}  "
            f"{row['log_phi_mond']:>7.2f}  "
            f"{gap_lcdm:>+9.2f}  "
            f"{_frac_col(frac_bpr)}  "
            f"{_frac_col(frac_mond)}  "
            f"{row['source']}"
        )

    def _mean_pct(total_miss: float) -> float:
        return (total_gap_lcdm - total_miss) / total_gap_lcdm * 100

    print()
    print(f"  Average ΛCDM gap: {total_gap_lcdm/n_points:.2f} dex")
    print(f"  BPR  closes {_mean_pct(total_gap_bpr):+.1f}% on average  "
          f"(gap → {total_gap_bpr/n_points:.2f} dex)")
    print(f"  MOND closes {_mean_pct(total_gap_mond):+.1f}% on average  "
          f"(gap → {total_gap_mond/n_points:.2f} dex)")
    print()

    # What would be needed?
    print(f"  {BOLD}What BPR would need to explain JWST:{RESET}")
    # Use median point as representative
    mid = results["jwst_uv_lf"][3]  # z=10, M=-20.5
    print(f"  Reference point: z={mid['z']:.0f}, M_UV={mid['M_UV']:.1f}")
    print(f"    σ enhancement needed:  {mid['sigma_ratio_needed']:.2f}×  "
          f"(BPR provides {(1.0 + 0.0076):.4f}×)")
    print(f"    n_s needed:            {mid['n_s_needed_to_explain']:.4f}  "
          f"(BPR predicts {results['bpr_params']['n_s_bpr']:.4f})")
    print(f"    n_s gap:               Δn_s ≈ {mid['n_s_needed_to_explain'] - 0.9649:.4f}  "
          f"(BPR gives Δn_s = {results['bpr_params']['delta_n_s']:.4f})")
    print()
    print(f"  {YELLOW}Verdict:{RESET} BPR's current n_s = {results['bpr_params']['n_s_bpr']:.4f}")
    print(f"  provides only +0.76% more small-scale power.  JWST requires")
    print(f"  ~{mid['sigma_ratio_needed']:.1f}× more σ on galaxy scales.  This is a genuine")
    print(f"  shortfall of {mid['sigma_ratio_needed'] / 1.0076:.1f}× — BPR does not resolve")
    print(f"  the JWST early-galaxy problem with its current mechanism.")

    # ── Overall verdict ────────────────────────────────────────────────────
    print_header("OVERALL VERDICT")
    s8_overshoot = s8.get("overshoots", False)
    s8_status = (f"overshoots by {(s8['fraction_explained']-1)*100:.0f}% "
                 f"(σ₈ = {s8['sigma8_bpr']:.3f} < WL obs ≈ 0.748)"
                 if s8_overshoot else f"closes {s8['fraction_explained']*100:.0f}%")
    mond_mean = _mean_pct(total_gap_mond)
    print(f"""
  {BOLD}MECHANISM COMPARISON — three JWST-era anomalies:{RESET}

  Hubble tension (4.9σ):
    Standard BPR:   closes  ~7%   (ΔNeff = {results['bpr_params']['delta_Neff']:.3f}, needs ~0.4)
    BPR + Theory II: same   ~7%   (MOND does not affect ΔNeff)
    Needed: 11× more boundary radiation species

  S8 tension (3.3σ):
    Standard BPR:   {s8_status}
    BPR + Theory II: WORSE  (MOND reduces δ_c universally → more clustering at 8 Mpc)
    Fundamental problem: S8 and JWST PULL IN OPPOSITE DIRECTIONS.

  JWST UV LF (z=9–16):
    Standard BPR:   closes {_mean_pct(total_gap_bpr):+.0f}%  (boundary dissipation suppresses early structure)
    BPR + Theory II:{YELLOW} closes {mond_mean:+.0f}%{RESET} (MOND δ_c=1.33 boosts halo abundance)
    BUT: MOND at z=16 OVER-predicts by ~1.4 dex (too strong at high z)
    AND: MOND also boosts σ₈-scale clustering → S8 becomes worse

  {BOLD}The fundamental tension BPR cannot yet resolve:{RESET}
  S8 (less clustering needed) ←→ JWST (more clustering needed)
  These are contradictory requirements at different epochs.
  No single "more/less structure" switch can fix both.

  {BOLD}What each theory contributes:{RESET}
  Theory II  (MOND/Impedance): a₀ from BPR is correct to 1.5%, confirms MOND regime
             at ALL galaxy scales.  Closes ~{mond_mean:.0f}% of JWST gap but
             over-predicts at z>13 and worsens S8.
  Theory XI  (Inflation): n_s = {results['bpr_params']['n_s_bpr']:.4f} is derived, not tuned.  Correct direction.
  Theory I   (Memory): running spectral index not yet computed — could help at small k.

  {BOLD}Path forward:{RESET}
  Need a mechanism that is ACTIVE at high z (z=5–16) but DORMANT at low z (z<2).
  Theory IV (Phase Transitions) suggests a boundary phase transition at z ~ 2–5
  could switch MOND-like enhancement ON for early galaxy formation while
  deactivating it before the epoch measured by S8.
  This is the next genuine BPR cosmological extension to derive.
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
