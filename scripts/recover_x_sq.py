"""
Recover x² as an EML symbolic tree.
====================================

Motivation: [ln(x)]² = x² ∘ ln(x).  We already have ln(x) at depth 3.
If we can recover x² independently, we know its EML tree structure and
can warm-start the depth-6 [ln(x)]² search (ln(x) locked at the bottom,
x²-structure at the top).  This splits the 2^442 search into two pieces.

Depth ladder 3→6.  Same locked params as benchmark_ln2.py.
Success criterion: snapped_loss < 1e-6 (exact symbolic recovery).

Run:
    python scripts/recover_x_sq.py
"""

import sys, time
sys.path.insert(0, "/home/user/BPR-Math-Spine")

import numpy as np
from bpr.eml_regression import fit, target_x_sq, target_ln

EXACT = 1e-6

BASE = dict(
    target_fn   = target_x_sq,
    x_range     = (0.5, 3.0),   # x²∈[0.25, 9]; ln(x)∈[-0.7, 1.1] → safe
    n_points    = 64,
    n_steps     = 3000,
    n_restarts  = 5,
    lr          = 0.02,
    seed        = 42,
    normalize_y = True,
)

# ── Control ──────────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("  CONTROL: ln(x) at depth 3")
t0 = time.time()
ctrl = fit(target_fn=target_ln, depth=3, x_range=(0.1,5.0),
           n_points=64, n_steps=3000, n_restarts=5, lr=0.05, seed=42)
print(f"  snapped_loss={ctrl.snapped_loss:.3e}  "
      f"{'PASS' if ctrl.snapped_loss < EXACT else 'FAIL'}  {ctrl.recovered_expr!r}"
      f"  ({time.time()-t0:.0f}s)")

# ── x² depth ladder ──────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print("  x²  DEPTH LADDER  (use_ste=True throughout)")
print(f"  {BASE}")
print("═"*60)

for depth in [3, 4, 5, 6]:
    n_params = 5 * 2**depth - 6
    print(f"\n  → depth={depth}  n_params={n_params}  ...", flush=True)
    t0 = time.time()
    r = fit(depth=depth, use_ste=True, **BASE)
    elapsed = time.time() - t0
    cat = "EXACT" if r.snapped_loss < EXACT else ("CLOSE" if r.snapped_loss < 1e-2 else "MISS")
    print(f"  cont={r.final_loss:.3e}  snap={r.snapped_loss:.3e}  "
          f"{cat}  {r.recovered_expr!r}  ({elapsed:.0f}s)")
    if r.success:
        print(f"\n  *** RECOVERED x² at depth={depth}: {r.recovered_expr!r} ***")
        print(f"  This tree warm-starts the [ln(x)]² depth-{depth+3} search.")
        break

print(f"\n{'═'*60}")
