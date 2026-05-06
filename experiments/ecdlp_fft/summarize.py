"""
Aggregate trial CSV into per-(field, control, bits) statistics:
  - hit-rate of top-K bins matching k_in_window
  - chance baseline = top_k / window_size  (uniform-random baseline)
  - exact binomial p-value for hit-rate vs chance
  - BH-FDR-corrected significance across all (field, control, bits)
  - mutual information I(k_in_window ; spectral_features)

Also build amortization-aware comparison vs Pollard rho.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

from analysis import benjamini_hochberg


def load(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def hit_rate_table(rows: list[dict], top_k: int = 5) -> list[dict]:
    """One entry per (bits, field, window, control) with hit count + p-value."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["window"] != "in":
            continue  # only "k inside window" rows have meaningful k_in_window
        key = (int(r["bits"]), r["field"], r["control"])
        buckets[key].append(r)

    out = []
    for (bits, field, control), trials in buckets.items():
        n_trials = len(trials)
        hits = sum(int(t["hit_topk"]) for t in trials)
        m = int(trials[0]["window_size"])
        chance = top_k / m
        if n_trials > 0:
            test = binomtest(hits, n_trials, p=chance, alternative="greater")
            pval = float(test.pvalue)
        else:
            pval = 1.0
        out.append({
            "bits": bits,
            "field": field,
            "control": control,
            "trials": n_trials,
            "hits": hits,
            "hit_rate": hits / n_trials if n_trials else 0.0,
            "chance": chance,
            "pval": pval,
        })
    return out


def amortization_table(rows: list[dict]) -> list[dict]:
    """Compare field-build cost vs sqrt(n) (rho baseline)."""
    out = []
    for r in rows:
        if r["control"] != "none" or r["window"] != "in":
            continue
        if r["field_cost_ops"] in (None, "", "None"):
            continue
        ops = int(r["field_cost_ops"])
        rho = int(r["rho_ops"])
        out.append({
            "bits": int(r["bits"]),
            "field": r["field"],
            "n": int(r["n"]),
            "field_cost_ops": ops,
            "rho_ops": rho,
            "ratio_field_to_rho": ops / max(1, rho),
            "sqrt_n": int(r["sqrt_n"]),
        })
    return out


def mutual_information_field_vs_k(rows: list[dict]) -> list[dict]:
    """For each (bits, field, control) compute MI between

      a feature = peak_sharpness  (continuous)
      and  k_in_window / window_size in [0,1]   (continuous)

    using histogram MI estimator, and similarly for spectral_entropy.
    Reported as I_peak, I_entropy. Only over window=='in' rows.
    """
    from analysis import mutual_information_continuous
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["window"] != "in":
            continue
        if r["k_in_window"] in ("-1", "", "None"):
            continue
        key = (int(r["bits"]), r["field"], r["control"])
        buckets[key].append(r)
    out = []
    for (bits, field, control), trials in buckets.items():
        if len(trials) < 8:
            continue
        ks = np.array([int(t["k_in_window"]) / int(t["window_size"])
                       for t in trials])
        sharp = np.array([float(t["peak_sharpness"]) for t in trials])
        ent = np.array([float(t["spectral_entropy"]) for t in trials])
        mi_p = mutual_information_continuous(ks, sharp, bins=8)
        mi_e = mutual_information_continuous(ks, ent, bins=8)
        out.append({
            "bits": bits, "field": field, "control": control,
            "n_trials": len(trials),
            "MI_k_vs_peak_sharpness": mi_p,
            "MI_k_vs_spectral_entropy": mi_e,
        })
    return out


def main(csv_in: Path, outdir: Path):
    rows = load(csv_in)
    print(f"loaded {len(rows)} rows from {csv_in}")
    outdir.mkdir(parents=True, exist_ok=True)

    hr = hit_rate_table(rows, top_k=5)
    pvals = np.array([h["pval"] for h in hr])
    sig_mask = benjamini_hochberg(pvals, alpha=0.05)
    for h, s in zip(hr, sig_mask):
        h["bh_significant"] = bool(s)

    with (outdir / "hit_rate.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hr[0].keys()))
        w.writeheader()
        w.writerows(hr)

    am = amortization_table(rows)
    if am:
        with (outdir / "amortization.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(am[0].keys()))
            w.writeheader()
            w.writerows(am)

    mi = mutual_information_field_vs_k(rows)
    if mi:
        with (outdir / "mutual_information.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(mi[0].keys()))
            w.writeheader()
            w.writerows(mi)

    print("\n=== HIT RATE TABLE (window=in) ===")
    print(f"{'bits':>4}  {'field':28s} {'ctrl':9s} {'trials':>6} "
          f"{'hits':>5} {'rate':>6} {'chance':>6} {'pval':>10} {'BH':>4}")
    for h in sorted(hr, key=lambda x: (x["bits"], x["field"], x["control"])):
        print(f"{h['bits']:>4}  {h['field']:28s} {h['control']:9s} "
              f"{h['trials']:>6d} {h['hits']:>5d} {h['hit_rate']:>6.3f} "
              f"{h['chance']:>6.3f} {h['pval']:>10.4f} "
              f"{'YES' if h['bh_significant'] else '   ':>4}")

    print("\n=== MI BY (bits, field, control) ===")
    print(f"{'bits':>4}  {'field':28s} {'ctrl':9s} {'I_peak':>10} {'I_ent':>10}")
    for m in sorted(mi, key=lambda x: (x["bits"], x["field"], x["control"])):
        print(f"{m['bits']:>4}  {m['field']:28s} {m['control']:9s} "
              f"{m['MI_k_vs_peak_sharpness']:>10.4f} "
              f"{m['MI_k_vs_spectral_entropy']:>10.4f}")

    print("\n=== AMORTIZATION (field cost vs sqrt(n) rho baseline) ===")
    by_bits_field: dict[tuple, list[float]] = defaultdict(list)
    for a in am:
        by_bits_field[(a["bits"], a["field"])].append(a["ratio_field_to_rho"])
    print(f"{'bits':>4}  {'field':28s} {'mean ratio':>12} {'n':>4}")
    for (bits, field), ratios in sorted(by_bits_field.items()):
        print(f"{bits:>4}  {field:28s} {np.mean(ratios):>12.3f} {len(ratios):>4d}")

    return hr, am, mi


if __name__ == "__main__":
    import sys
    from pathlib import Path
    csv_in = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/results.csv")
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data")
    main(csv_in, outdir)
