"""
Extended experiment driver: applies the full transform library to each
(instance, field, control) combination.

Records per-trial scalar features in a wide CSV. Then runs:
  - per-feature per-bucket comparison (real vs qsub) with t-test
  - MMD and classifier AUC distinguishers per (bits, field) using all
    features stacked

Outputs:
  data/extended_features.csv     -- raw per-trial features
  data/extended_buckets.csv      -- per-(bits, field, transform_family) tests
  data/extended_distinguishers.csv -- MMD and classifier results
  REPORT_EXTENDED.md             -- written by separate writeup step
"""
from __future__ import annotations
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from curve import Curve, Point, scalar_mult, add
from curve_gen import make_toy_instance, ToyInstance
from solvers import bsgs, pollard_rho
from phase_fields import FIELDS, BPRParams
from run_experiment import random_curve_point_of_order, alternate_generator
from transforms import all_features
from distinguishers import mmd2_with_pvalue, classifier_auc


def _walk_xs(E: Curve, G: Point, w0: int, m: int) -> np.ndarray:
    """Return integer x-coords of [w0*G ... (w0+m-1)*G]. Uses E.p as
    sentinel for the point at infinity."""
    xs = np.zeros(m, dtype=np.int64)
    R = scalar_mult(w0, G, E)
    for i in range(m):
        xs[i] = E.p if R.is_inf else R.x
        if i < m - 1:
            R = add(R, G, E)
    return xs


def run_one_trial(inst: ToyInstance, key_seed: int, *,
                  window_size: int = 256,
                  bpr: BPRParams = BPRParams()) -> list[dict]:
    rng = random.Random(f"key-{inst.bits}-{inst.seed}-{key_seed}")
    k = rng.randrange(1, inst.n)
    Q = scalar_mult(k, inst.G, inst.E)

    m = min(window_size, inst.n)
    w0_in = max(0, min(k - m // 2, inst.n - m))
    if k < w0_in or k >= w0_in + m:
        w0_in = max(0, min(k - m // 4, inst.n - m))

    R = random_curve_point_of_order(inst, key_seed)
    Gprime = alternate_generator(inst, key_seed)

    # Precompute integer x-coordinate sequences once per (G, w0_in):
    xs_G_in = _walk_xs(inst.E, inst.G, w0_in, m)
    xs_Gp_in = _walk_xs(inst.E, Gprime, w0_in, m)

    rows = []
    for field_name, field_fn in FIELDS.items():
        # We compute the field 3 times: with Q (true), with R (qsub),
        # with G' (gsub). We also include a shuffle and randphase control.
        phi_true = field_fn(inst.E, inst.G, Q, inst.n, w0_in, m, bpr)
        phi_qsub = field_fn(inst.E, inst.G, R, inst.n, w0_in, m, bpr)
        phi_gsub = field_fn(inst.E, Gprime, Q, inst.n, w0_in, m, bpr)
        # shuffle and randphase
        rng2 = np.random.default_rng(hash((field_name, w0_in, key_seed)) & 0xFFFFFFFF)
        phi_shuf = phi_true.copy()
        rng2.shuffle(phi_shuf)
        phi_rand = np.exp(2j * np.pi * rng2.random(m))

        for control, phi, xs in [
            ("none", phi_true, xs_G_in),
            ("qsub", phi_qsub, xs_G_in),
            ("gsub", phi_gsub, xs_Gp_in),
            ("shuffle", phi_shuf, xs_G_in),
            ("randphase", phi_rand, xs_G_in),
        ]:
            feats = all_features(phi, xs=xs, p=inst.E.p)
            row = {
                "bits": inst.bits, "instance_seed": inst.seed,
                "key_seed": key_seed, "field": field_name,
                "control": control,
                "n": inst.n, "k": k,
                "k_in_window": k - w0_in if (w0_in <= k < w0_in + m) else -1,
                "window_size": m,
            }
            row.update(feats)
            rows.append(row)
    return rows


def run(*, bits_list, n_instances, n_keys_per_instance, window_size,
        outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "extended_features.csv"
    rows_total = []
    t_start = time.time()
    for bits in bits_list:
        for inst_seed in range(n_instances):
            t = time.time()
            inst = make_toy_instance(bits=bits, seed=inst_seed)
            for key_seed in range(n_keys_per_instance):
                rows = run_one_trial(inst, key_seed, window_size=window_size)
                rows_total.extend(rows)
            print(f"[{time.time()-t_start:6.1f}s] bits={bits} inst={inst_seed} "
                  f"n={inst.n} ({time.time()-t:.1f}s)")
    if rows_total:
        fieldnames = list(rows_total[0].keys())
        # Make sure all rows have the same keys
        all_keys = set()
        for r in rows_total:
            all_keys.update(r.keys())
        fieldnames = sorted(all_keys, key=lambda k: (
            0 if k in ["bits","instance_seed","key_seed","field","control",
                       "n","k","k_in_window","window_size"] else 1, k))
        for r in rows_total:
            for k in fieldnames:
                r.setdefault(k, 0.0)
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_total)
        print(f"wrote {len(rows_total)} rows to {csv_path}")
    return rows_total


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, nargs="+", default=[12, 14, 16, 18])
    ap.add_argument("--instances", type=int, default=30)
    ap.add_argument("--keys-per-instance", type=int, default=1)
    ap.add_argument("--window-size", type=int, default=256)
    ap.add_argument("--outdir", type=Path, default=HERE / "data")
    args = ap.parse_args()
    run(bits_list=args.bits, n_instances=args.instances,
        n_keys_per_instance=args.keys_per_instance,
        window_size=args.window_size, outdir=args.outdir)
