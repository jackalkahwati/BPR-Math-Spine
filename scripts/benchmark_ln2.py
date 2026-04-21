"""
Controlled Benchmark: Recovery of [ln(x)]²
==========================================

Gate experiment before full BPR.  All parameters locked; change nothing
between depth runs so comparisons are apples-to-apples.

SUCCESS CRITERIA (defined before running):
  - exact_recovery : snapped_loss < 1e-6           (symbolic tree recovered)
  - close_approx   : 1e-6 ≤ snapped_loss < 1e-2   (numerically close but not exact)
  - diverged       : snapped_loss ≥ 1e-2            (optimizer failed)

DEPTH LADDER: 4, 5, 6, 7
  Hypothesis: [ln(x)]² = compose(x², ln(x)).  ln(x) needs depth 3;
  x² needs additional depth; minimum master-formula depth is unknown
  and is the primary question this benchmark answers.

Run:
    python scripts/benchmark_ln2.py

Expected runtime: ~5–20 min depending on hardware.
"""

import sys
import time

sys.path.insert(0, "/home/user/BPR-Math-Spine")

import numpy as np
from bpr.eml_regression import fit, FitResult, target_ln_sq, target_ln, target_exp

# ─────────────────────────────────────────────────────────────────────────────
# Locked benchmark parameters
# Depth is fixed at 6: continuous loss 4.6e-07 at depth 6 confirms
# representational capacity; snap quality is now the only bottleneck.
# ─────────────────────────────────────────────────────────────────────────────
BENCHMARK = dict(
    target_fn    = target_ln_sq,
    x_range      = (0.5, 5.0),   # ln(x)∈[-0.69, 1.61], [ln(x)]²∈[0, 2.59]
    n_points     = 64,
    n_steps      = 5000,
    n_restarts   = 20,
    lr           = 0.02,
    seed         = 42,
    normalize_y  = True,
    depth        = 6,
)

# Hardening configurations to compare (last param = label)
HARDEN_CONFIGS = [
    dict(harden_frac=0.0,  min_temp=1.0,  entropy_coeff=0.0,  label="no hardening (baseline)"),
    dict(harden_frac=0.3,  min_temp=0.1,  entropy_coeff=0.0,  label="temp anneal only"),
    dict(harden_frac=0.3,  min_temp=0.1,  entropy_coeff=0.1,  label="temp anneal + entropy"),
    dict(harden_frac=0.3,  min_temp=0.05, entropy_coeff=0.2,  label="aggressive hardening"),
]

# Success thresholds (fixed before seeing results)
EXACT_THRESHOLD  = 1e-6
CLOSE_THRESHOLD  = 1e-2

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def classify(snapped_loss: float) -> str:
    if snapped_loss < EXACT_THRESHOLD:
        return "EXACT_RECOVERY"
    if snapped_loss < CLOSE_THRESHOLD:
        return "CLOSE_APPROX"
    return "DIVERGED"


def report(depth: int, result: FitResult, elapsed: float) -> str:
    n_params = 5 * 2**depth - 6
    cat = classify(result.snapped_loss)
    lines = [
        f"\n{'═'*64}",
        f"  [ln(x)]²  depth={depth}  n_params={n_params}  restarts={result.n_restarts_tried}",
        f"{'─'*64}",
        f"  continuous_loss = {result.final_loss:.4e}",
        f"  snapped_loss    = {result.snapped_loss:.4e}",
        f"  outcome         = {cat}",
        f"  recovered_expr  = {result.recovered_expr!r}",
        f"  elapsed         = {elapsed:.1f}s",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Control run: confirm ln(x) still works (sanity check, should succeed fast)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*64)
print("  CONTROL: ln(x) at depth 3  [must pass — known EML tree exists]")
print("═"*64)
t0 = time.time()
ctrl = fit(
    target_fn   = target_ln,
    depth       = 3,
    x_range     = (0.1, 5.0),
    n_points    = 64,
    n_steps     = 5000,
    n_restarts  = 20,
    lr          = 0.05,
    seed        = 42,
    normalize_y = False,  # ln(x) already O(1) on this range
)
elapsed_ctrl = time.time() - t0
ctrl_ok = ctrl.snapped_loss < EXACT_THRESHOLD
print(f"  snapped_loss = {ctrl.snapped_loss:.4e}  |  {'PASS' if ctrl_ok else 'FAIL'}")
print(f"  recovered    = {ctrl.recovered_expr!r}")
print(f"  elapsed      = {elapsed_ctrl:.1f}s")
if not ctrl_ok:
    print("\n  *** CONTROL FAILED — do not interpret hardening results ***")


# ─────────────────────────────────────────────────────────────────────────────
# Snap hardening comparison at depth 6
# Depth is fixed: continuous loss 4.6e-07 at depth 6 confirms capacity.
# The only question now: which hardening config produces clean snap?
# ─────────────────────────────────────────────────────────────────────────────

depth = BENCHMARK.pop("depth")  # remove from BENCHMARK so we can pass explicitly
n_params = 5 * 2**depth - 6

print(f"\n\n{'═'*64}")
print(f"  SNAP HARDENING COMPARISON  [ln(x)]²  depth={depth}  n_params={n_params}")
print(f"  x_range={BENCHMARK['x_range']}  n_points={BENCHMARK['n_points']}")
print(f"  n_steps={BENCHMARK['n_steps']}  n_restarts={BENCHMARK['n_restarts']}")
print(f"  success if snapped_loss < {EXACT_THRESHOLD:.0e}")
print("═"*64)

results: dict[str, tuple[FitResult, float]] = {}

for cfg in HARDEN_CONFIGS:
    label = cfg.pop("label")
    print(f"\n  → {label}  starting...", flush=True)
    t0 = time.time()
    r = fit(depth=depth, **BENCHMARK, **cfg)
    elapsed = time.time() - t0
    results[label] = (r, elapsed)
    print(report(depth, r, elapsed), flush=True)
    cfg["label"] = label  # restore for summary

    if r.success:
        print(f"\n  *** EXACT RECOVERY with '{label}' ***")
        break


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n\n{'═'*64}")
print("  SUMMARY")
print(f"{'─'*64}")
print(f"  {'config':<28}  {'cont_loss':>10}  {'snap_loss':>10}  {'outcome':<16}  {'expr'}")
print(f"{'─'*64}")
for label, (r, _) in results.items():
    cat = classify(r.snapped_loss)
    expr_str = repr(r.recovered_expr) if r.recovered_expr else "None"
    print(
        f"  {label:<28}  {r.final_loss:>10.3e}"
        f"  {r.snapped_loss:>10.3e}  {cat:<16}  {expr_str}"
    )
print(f"{'═'*64}")

# Diagnosis
first_exact = next((l for l, (r, _) in results.items() if r.snapped_loss < EXACT_THRESHOLD), None)
first_close = next((l for l, (r, _) in results.items() if r.snapped_loss < CLOSE_THRESHOLD), None)
all_diverged = all(r.snapped_loss >= CLOSE_THRESHOLD for r, _ in results.values())

print("\n  DIAGNOSIS")
print(f"{'─'*64}")
if first_exact is not None:
    print(f"  Exact recovery with: '{first_exact}'")
    print(f"  [ln(x)]² snap quality solved. Next: promote to permanent benchmark,")
    print(f"  then move to smallest BPR subterm.")
elif first_close is not None:
    print(f"  Best config '{first_close}' gets close but not exact snap.")
    print(f"  Hardening helps but logits still ambiguous at snap time.")
    print(f"  Try: more restarts, stronger entropy_coeff, or two-stage lr.")
elif all_diverged:
    print(f"  All hardening configs still diverge at snap.")
    print(f"  Continuous loss is fine — problem is purely snap sharpness.")
    print(f"  Consider: staged training (freeze lower layers), Gumbel-softmax,")
    print(f"  or iterative snap-and-finetune.")
print(f"{'═'*64}")
