"""
Aggregate extended_features.csv with FDR control.

Two complementary tests:

  (a) PER-FEATURE marginal test: for each (bits, field, feature),
      Welch's t-test of feature distribution under control='none' vs
      control='qsub'. If feature distribution is the same under
      true-Q and Q-substitution, that feature is Q-blind.

  (b) JOINT distinguisher tests: for each (bits, field), pool features
      across instances and run MMD + cross-validated classifier AUC
      between control='none' and control='qsub'.

BH-FDR is applied across all (bits, field, feature) marginal tests
and separately across all (bits, field) joint tests.
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind

from analysis import benjamini_hochberg
from distinguishers import mmd2_with_pvalue, classifier_auc

META_COLS = {"bits", "instance_seed", "key_seed", "field", "control",
             "n", "k", "k_in_window", "window_size"}


def load(path: Path) -> tuple[list[str], list[dict]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return [], []
    feat_cols = [c for c in rows[0].keys() if c not in META_COLS]
    return feat_cols, rows


def matrix_for(rows, bits, field, control, feat_cols):
    sel = [r for r in rows if int(r["bits"]) == bits
           and r["field"] == field and r["control"] == control]
    M = np.zeros((len(sel), len(feat_cols)))
    for i, r in enumerate(sel):
        for j, c in enumerate(feat_cols):
            try:
                M[i, j] = float(r[c])
            except (ValueError, TypeError):
                M[i, j] = 0.0
    return M


def main(csv_in: Path, outdir: Path):
    feat_cols, rows = load(csv_in)
    print(f"loaded {len(rows)} rows, {len(feat_cols)} features")
    bits_set = sorted({int(r["bits"]) for r in rows})
    fields = sorted({r["field"] for r in rows})
    print(f"bits: {bits_set}")
    print(f"fields: {fields}")

    # ----- (a) per-feature marginal tests, none vs qsub -----
    marginal = []
    for bits in bits_set:
        for field in fields:
            M_real = matrix_for(rows, bits, field, "none", feat_cols)
            M_qsub = matrix_for(rows, bits, field, "qsub", feat_cols)
            M_shuf = matrix_for(rows, bits, field, "shuffle", feat_cols)
            if len(M_real) < 4 or len(M_qsub) < 4:
                continue
            for j, c in enumerate(feat_cols):
                a, b = M_real[:, j], M_qsub[:, j]
                if a.std() < 1e-12 and b.std() < 1e-12:
                    pval = 1.0
                else:
                    try:
                        pval = float(ttest_ind(a, b, equal_var=False).pvalue)
                        if not np.isfinite(pval):
                            pval = 1.0
                    except Exception:
                        pval = 1.0
                marginal.append({
                    "bits": bits, "field": field, "feature": c,
                    "mean_real": float(a.mean()),
                    "mean_qsub": float(b.mean()),
                    "std_real": float(a.std()),
                    "std_qsub": float(b.std()),
                    "pval_real_vs_qsub": pval,
                })
    pvals = np.array([m["pval_real_vs_qsub"] for m in marginal])
    sig = benjamini_hochberg(pvals, alpha=0.05)
    for m, s in zip(marginal, sig):
        m["bh_significant"] = bool(s)

    with (outdir / "extended_marginal.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(marginal[0].keys()))
        w.writeheader()
        w.writerows(marginal)

    n_marginal = len(marginal)
    n_raw_sig = sum(1 for m in marginal if m["pval_real_vs_qsub"] < 0.05)
    n_bh_sig = sum(1 for m in marginal if m["bh_significant"])
    print(f"\n=== MARGINAL TESTS (real vs qsub per feature) ===")
    print(f"total tests: {n_marginal}")
    print(f"raw p<0.05:  {n_raw_sig}  (chance: {n_marginal*0.05:.1f})")
    print(f"BH-FDR sig:  {n_bh_sig}")

    # ----- (b) joint MMD + classifier per (bits, field) -----
    joint = []
    for bits in bits_set:
        for field in fields:
            M_real = matrix_for(rows, bits, field, "none", feat_cols)
            M_qsub = matrix_for(rows, bits, field, "qsub", feat_cols)
            if len(M_real) < 8 or len(M_qsub) < 8:
                continue
            mmd = mmd2_with_pvalue(M_real, M_qsub, n_permutations=200, seed=bits)
            clf = classifier_auc(M_real, M_qsub, n_permutations=50, seed=bits)
            joint.append({
                "bits": bits, "field": field,
                "n_real": len(M_real), "n_qsub": len(M_qsub),
                "mmd2": mmd["mmd2"], "mmd_pval": mmd["pval"], "sigma": mmd["sigma"],
                "auc": clf["auc"], "auc_pval": clf["pval"],
            })
    pvals_mmd = np.array([j["mmd_pval"] for j in joint])
    pvals_auc = np.array([j["auc_pval"] for j in joint])
    sig_mmd = benjamini_hochberg(pvals_mmd, alpha=0.05)
    sig_auc = benjamini_hochberg(pvals_auc, alpha=0.05)
    for j, sm, sa in zip(joint, sig_mmd, sig_auc):
        j["mmd_bh"] = bool(sm)
        j["auc_bh"] = bool(sa)

    with (outdir / "extended_joint.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(joint[0].keys()))
        w.writeheader()
        w.writerows(joint)

    print(f"\n=== JOINT DISTINGUISHER TESTS (real vs qsub, all features) ===")
    print(f"{'bits':>4}  {'field':28s} {'mmd2':>10} {'p_mmd':>7} "
          f"{'auc':>6} {'p_auc':>7} {'mmd_BH':>7} {'auc_BH':>7}")
    for j in sorted(joint, key=lambda x: (x["bits"], x["field"])):
        print(f"{j['bits']:>4}  {j['field']:28s} {j['mmd2']:>10.4e} "
              f"{j['mmd_pval']:>7.3f} {j['auc']:>6.3f} {j['auc_pval']:>7.3f} "
              f"{'YES' if j['mmd_bh'] else '   ':>7} "
              f"{'YES' if j['auc_bh'] else '   ':>7}")

    n_mmd_bh = sum(1 for j in joint if j["mmd_bh"])
    n_auc_bh = sum(1 for j in joint if j["auc_bh"])
    n_mmd_raw = sum(1 for j in joint if j["mmd_pval"] < 0.05)
    n_auc_raw = sum(1 for j in joint if j["auc_pval"] < 0.05)
    print(f"\nMMD: raw p<0.05: {n_mmd_raw}/{len(joint)}, BH: {n_mmd_bh}/{len(joint)}")
    print(f"AUC: raw p<0.05: {n_auc_raw}/{len(joint)}, BH: {n_auc_bh}/{len(joint)}")

    # ----- top 10 strongest features overall -----
    print("\n=== TOP 10 SMALLEST RAW p-values (per-feature marginal) ===")
    top = sorted(marginal, key=lambda x: x["pval_real_vs_qsub"])[:10]
    for m in top:
        print(f"  bits={m['bits']:>2} {m['field']:28s} "
              f"{m['feature']:24s} p={m['pval_real_vs_qsub']:.4g} "
              f"BH={'YES' if m['bh_significant'] else 'no'}")

    # By-feature-family BH-significant count
    print("\n=== BH-SIGNIFICANT MARGINALS BY FEATURE-FAMILY PREFIX ===")
    fam_sig = defaultdict(int)
    fam_total = defaultdict(int)
    for m in marginal:
        prefix = m["feature"].split("_", 1)[0]
        fam_total[prefix] += 1
        if m["bh_significant"]:
            fam_sig[prefix] += 1
    for prefix in sorted(fam_total):
        print(f"  {prefix:>4}: {fam_sig[prefix]:>3} / {fam_total[prefix]:>3} "
              f"BH-significant")

    return marginal, joint


if __name__ == "__main__":
    import sys
    csv_in = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/extended_features.csv")
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data")
    main(csv_in, outdir)
