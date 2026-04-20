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
# Locked benchmark parameters — do not change between depth runs
# ─────────────────────────────────────────────────────────────────────────────
BENCHMARK = dict(
    target_fn   = target_ln_sq,
    x_range     = (0.5, 5.0),   # ln(x)∈[-0.69, 1.61], [ln(x)]²∈[0, 2.59]
    n_points    = 64,
    n_steps     = 5000,
    n_restarts  = 20,
    lr          = 0.02,
    seed        = 42,
    normalize_y = True,
)

DEPTH_LADDER = [4, 5, 6, 7]

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
    print("\n  *** CONTROL FAILED — do not interpret depth-ladder results ***")


# ─────────────────────────────────────────────────────────────────────────────
# Depth ladder for [ln(x)]²
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n\n{'═'*64}")
print("  DEPTH LADDER: [ln(x)]²")
print(f"  x_range={BENCHMARK['x_range']}  n_points={BENCHMARK['n_points']}")
print(f"  n_steps={BENCHMARK['n_steps']}  n_restarts={BENCHMARK['n_restarts']}")
print(f"  lr={BENCHMARK['lr']}  seed={BENCHMARK['seed']}  normalize_y={BENCHMARK['normalize_y']}")
print(f"  success if snapped_loss < {EXACT_THRESHOLD:.0e}")
print("═"*64)

results: dict[int, tuple[FitResult, float]] = {}

for depth in DEPTH_LADDER:
    print(f"\n  → depth={depth}  n_params={5*2**depth-6}  starting...", flush=True)
    t0 = time.time()
    r = fit(depth=depth, **BENCHMARK)
    elapsed = time.time() - t0
    results[depth] = (r, elapsed)
    print(report(depth, r, elapsed), flush=True)

    if r.success:
        print(f"\n  *** EXACT RECOVERY at depth={depth} ***  stopping ladder.")
        break


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n\n{'═'*64}")
print("  SUMMARY")
print(f"{'─'*64}")
print(f"  {'depth':>5}  {'n_params':>8}  {'cont_loss':>10}  {'snap_loss':>10}  {'outcome':<16}  {'expr'}")
print(f"{'─'*64}")
for depth, (r, _) in results.items():
    n_params = 5 * 2**depth - 6
    cat = classify(r.snapped_loss)
    expr_str = repr(r.recovered_expr) if r.recovered_expr else "None"
    print(
        f"  {depth:>5}  {n_params:>8}  {r.final_loss:>10.3e}"
        f"  {r.snapped_loss:>10.3e}  {cat:<16}  {expr_str}"
    )
print(f"{'═'*64}")

# Diagnosis
first_exact = next((d for d, (r, _) in results.items() if r.snapped_loss < EXACT_THRESHOLD), None)
first_close = next((d for d, (r, _) in results.items() if r.snapped_loss < CLOSE_THRESHOLD), None)
all_diverged = all(r.snapped_loss >= CLOSE_THRESHOLD for r, _ in results.values())

print("\n  DIAGNOSIS")
print(f"{'─'*64}")
if first_exact is not None:
    print(f"  Exact recovery achieved at depth={first_exact}.")
    print(f"  [ln(x)]² is representable and recoverable at depth {first_exact}.")
    print(f"  Next step: BPR subterm regression.")
elif first_close is not None:
    print(f"  Close approximation (not exact) first seen at depth={first_close}.")
    print(f"  Optimizer converges continuously but snap does not hit exact tree.")
    print(f"  Possible causes: depth insufficient, more restarts needed,")
    print(f"    or logits not sharp enough at snap time.")
elif all_diverged:
    print(f"  All depths DIVERGED (snapped_loss ≥ {CLOSE_THRESHOLD:.0e}).")
    print(f"  Causes: depth too shallow, overflow, or target outside EML reach.")
    print(f"  Check: continuous_loss trend across depths for capacity signal.")
print(f"{'═'*64}")
