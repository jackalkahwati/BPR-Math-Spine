"""
Main experiment driver.

For each (bit-width, instance-seed):
  - Build toy curve, generator G, secret k, public Q = k*G
  - Choose a window [w0, w0+m) on Z/nZ. We test BOTH:
      window_in: w0 <= k < w0+m  (the "spectrum can find k inside" question)
      window_out: w0 fixed away from k (the "spectrum tells us where k is" question)
  - For each phase-field variant:
      - Build phi(W) on the window
      - Run shuffled, Q-sub, G-sub, random-phase controls
      - FFT and extract features
      - Score whether top-K bins point to k
  - Run BSGS and Pollard rho baselines for comparison

Outputs CSV of trial results to data/results.csv.
"""
from __future__ import annotations
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

# add cwd
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from curve import Curve, Point, scalar_mult, lift_x
from curve_gen import make_toy_instance, ToyInstance
from solvers import bsgs, pollard_rho
from phase_fields import FIELDS, BPRParams, field_cost
from analysis import (
    fft_features, hits_target, shuffle_phi, random_phase_control,
    mutual_information_continuous, benjamini_hochberg, predicted_k_bins,
)


def random_curve_point_of_order(inst: ToyInstance, seed: int) -> Point:
    """Return a uniformly random non-identity point in <G>: pick a random
    scalar r in [1, n) and return r*G. (DDH-equivalent random point.)"""
    rng = random.Random(f"rand_pt-{inst.bits}-{inst.seed}-{seed}")
    r = rng.randrange(1, inst.n)
    return scalar_mult(r, inst.G, inst.E)


def alternate_generator(inst: ToyInstance, seed: int) -> Point:
    """Return G' = c*G for random c coprime to n. <G'> = <G> still."""
    rng = random.Random(f"alt_g-{inst.bits}-{inst.seed}-{seed}")
    while True:
        c = rng.randrange(2, inst.n)
        if math.gcd(c, inst.n) == 1:
            return scalar_mult(c, inst.G, inst.E)


def run_one_trial(inst: ToyInstance, key_seed: int, *,
                  window_size: int = 256,
                  top_k: int = 5,
                  bpr: BPRParams = BPRParams()) -> list[dict]:
    """Returns list of trial result dicts (one per field x control combo)."""
    rng = random.Random(f"key-{inst.bits}-{inst.seed}-{key_seed}")
    k = rng.randrange(1, inst.n)
    Q = scalar_mult(k, inst.G, inst.E)

    m = min(window_size, inst.n)

    # Three window placements:
    # (a) window contains k
    # (b) window does not contain k
    w0_in = max(0, k - m // 2)
    w0_in = min(w0_in, inst.n - m)
    # ensure k in window
    if k < w0_in or k >= w0_in + m:
        w0_in = max(0, min(k - m // 4, inst.n - m))
    w0_out = (w0_in + inst.n // 2) % inst.n
    if w0_out + m > inst.n:
        w0_out = max(0, inst.n - m - 1)

    # Compute baselines once per instance (not per field)
    inst.E.counter.reset()
    try:
        bsgs_k = bsgs(inst.E, inst.G, Q, inst.n)
        bsgs_ops = inst.E.counter.total
        assert bsgs_k == k, f"bsgs returned {bsgs_k} != {k}"
    except Exception as e:
        bsgs_ops = -1
    inst.E.counter.reset()
    try:
        rho_k = pollard_rho(inst.E, inst.G, Q, inst.n, seed=key_seed)
        rho_ops = inst.E.counter.total
        assert rho_k == k
    except Exception as e:
        rho_ops = -1

    rows = []

    # Q-substitution control: replace Q with random R = r*G (uniformly random
    # in <G>). This R has its own discrete log r' which we don't tell the field.
    R = random_curve_point_of_order(inst, key_seed)
    # G-substitution control: G' = c*G. Q under G' coords has a different
    # discrete log (k' = k * c^{-1} mod n). Field built with G'.
    Gprime = alternate_generator(inst, key_seed)

    for field_name, field_fn in FIELDS.items():
        for window_label, w0 in [("in", w0_in), ("out", w0_out)]:
            k_in_window = k - w0 if (w0 <= k < w0 + m) else -1

            # ---- main field with true Q ----
            inst.E.counter.reset()
            phi_main = field_fn(inst.E, inst.G, Q, inst.n, w0, m, bpr)
            ops_main = inst.E.counter.total

            # ---- Q-sub: same field, but Q -> R (random point) ----
            inst.E.counter.reset()
            phi_qsub = field_fn(inst.E, inst.G, R, inst.n, w0, m, bpr)

            # ---- G-sub: build with G', Q stays (so secret in G' coord is k') ----
            inst.E.counter.reset()
            phi_gsub = field_fn(inst.E, Gprime, Q, inst.n, w0, m, bpr)

            # ---- shuffle ----
            phi_shuf = shuffle_phi(phi_main, seed=hash((field_name, w0, key_seed)) & 0xFFFFFFFF)

            # ---- random-phase ----
            phi_rand = random_phase_control(m, seed=hash((field_name, w0, key_seed, "r")) & 0xFFFFFFFF)

            for ctrl_label, phi in [
                ("none", phi_main),
                ("qsub", phi_qsub),
                ("gsub", phi_gsub),
                ("shuffle", phi_shuf),
                ("randphase", phi_rand),
            ]:
                feats = fft_features(phi, top_k=top_k)
                hit = (k_in_window >= 0
                       and k_in_window in predicted_k_bins(feats, m, top_k))
                rows.append({
                    "bits": inst.bits,
                    "instance_seed": inst.seed,
                    "key_seed": key_seed,
                    "field": field_name,
                    "window": window_label,
                    "control": ctrl_label,
                    "n": inst.n,
                    "k": k,
                    "k_in_window": k_in_window,
                    "window_size": m,
                    "spectral_entropy": feats["spectral_entropy"],
                    "peak_sharpness": feats["peak_sharpness"],
                    "top_idx": json.dumps([int(x) for x in feats["top_idx"]]),
                    "top_amp": json.dumps([float(x) for x in feats["top_amp"]]),
                    "field_cost_ops": ops_main if ctrl_label == "none" else None,
                    "field_cost_model": field_cost(field_name, m, w0),
                    "bsgs_ops": bsgs_ops,
                    "rho_ops": rho_ops,
                    "sqrt_n": int(math.isqrt(inst.n)),
                    "hit_topk": int(hit),
                })
    return rows


def run(*, bits_list, n_instances, n_keys_per_instance, window_size,
        outdir: Path, bpr: BPRParams = BPRParams()):
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "results.csv"
    fieldnames = None
    rows_total = []
    t_start = time.time()
    for bits in bits_list:
        for inst_seed in range(n_instances):
            t = time.time()
            inst = make_toy_instance(bits=bits, seed=inst_seed)
            for key_seed in range(n_keys_per_instance):
                rows = run_one_trial(inst, key_seed, window_size=window_size,
                                     bpr=bpr)
                rows_total.extend(rows)
            print(f"[{time.time()-t_start:6.1f}s] bits={bits} inst={inst_seed} "
                  f"n={inst.n} keys={n_keys_per_instance} ({time.time()-t:.1f}s)")
    if rows_total:
        fieldnames = list(rows_total[0].keys())
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
