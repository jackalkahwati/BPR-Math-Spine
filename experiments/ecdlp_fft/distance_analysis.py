"""
Stronger test than top-K hit rate: does the FFT-predicted top peak land
*closer to k* than uniform random would?

For each trial with window=='in', let p1 = top peak index (excluding DC).
Compute distance d = min(|p1 - k_in_window|, m - |p1 - k_in_window|).
Under the null, E[d] = m/4. We test whether mean(d) < m/4.
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def main(csv_in: Path):
    with csv_in.open() as f:
        rows = list(csv.DictReader(f))

    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r["window"] != "in":
            continue
        if r["k_in_window"] in ("-1", "", "None"):
            continue
        m = int(r["window_size"])
        k_iw = int(r["k_in_window"])
        top = json.loads(r["top_idx"])
        if not top:
            continue
        p1 = int(top[0])
        d = abs(p1 - k_iw)
        d = min(d, m - d)
        norm_d = d / m  # in [0, 0.5]
        key = (int(r["bits"]), r["field"], r["control"])
        buckets[key].append(norm_d)

    print(f"{'bits':>4}  {'field':28s} {'ctrl':9s} {'n':>4} "
          f"{'mean d/m':>10} {'null':>6} {'t':>7} {'pval':>8}")
    flagged = []
    for (bits, field, control), ds in sorted(buckets.items()):
        ds = np.array(ds)
        mean = ds.mean()
        # Two-sided t-test against null mean = 0.25 (uniform on [0, 0.5])
        # Variance of uniform on [0, 0.5] = (0.5)^2 / 12 = 0.0208
        # We use one-sided: are predicted peaks closer than chance?
        t = (mean - 0.25) / (ds.std(ddof=1) / np.sqrt(len(ds))) if len(ds) > 1 else 0.0
        pval = stats.ttest_1samp(ds, 0.25, alternative="less").pvalue if len(ds) > 1 else 1.0
        if pval < 0.05:
            flagged.append((bits, field, control, mean, pval))
        print(f"{bits:>4}  {field:28s} {control:9s} {len(ds):>4d} "
              f"{mean:>10.4f} {0.25:>6.3f} {t:>7.2f} {pval:>8.4f}")

    if flagged:
        print(f"\n{len(flagged)} buckets had peak-distance significantly below chance (uncorrected p<0.05):")
        for f in flagged:
            print(" ", f)
    else:
        print("\nNo bucket showed peak-distance significantly below chance.")


if __name__ == "__main__":
    import sys
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/results.csv"))
