"""
EML Symbolic Regression on BPR Targets
=======================================

Attempts to recover BPR sub-expressions from numerical data using
gradient-based EML tree optimisation (Odrzywolek 2026, §4.3).

Targets (in order of difficulty):
  1. ln(x)            depth 3   K=7    ~25% success/restart  [known answer]
  2. [ln(p)]²         depth 4   K≈35   unknown success rate  [BPR screening term]
  3. 1/α(p) - const   depth 5+  K>50   unlikely              [full BPR formula]

Run:
    python scripts/run_eml_bpr_regression.py
"""

import sys
import time

sys.path.insert(0, "/home/user/BPR-Math-Spine")

import numpy as np
from bpr.eml_regression import (
    fit, FitResult,
    target_ln, target_bpr_screening, target_bpr_alpha,
    EMLMasterFormula,
)
from bpr.constants import P_DEFAULT


def _report(label: str, result: FitResult, elapsed: float) -> None:
    print(f"\n{'═'*62}")
    print(f"  {label}")
    print(f"{'─'*62}")
    print(f"  depth          = {result.depth}")
    print(f"  n_params       = {5 * 2**result.depth - 6}")
    print(f"  restarts       = {result.n_restarts_tried}")
    print(f"  final_loss     = {result.final_loss:.4e}")
    print(f"  snapped_loss   = {result.snapped_loss:.4e}")
    print(f"  success        = {result.success}  (threshold: 1e-6)")
    print(f"  recovered_expr = {result.recovered_expr!r}")
    print(f"  elapsed        = {elapsed:.1f}s")


# ── Target 1: ln(x)  ─────────────────────────────────────────────────────────
# Known answer: eml(1, eml(eml(1, x), 1))   depth 3, K=7
print("\nTarget 1: ln(x)  [known EML tree exists at depth 3]")
t0 = time.time()
r1 = fit(
    target_fn=target_ln,
    depth=3,
    x_range=(0.1, 5.0),
    n_points=64,
    n_steps=4000,
    n_restarts=20,
    lr=0.05,
    seed=42,
)
_report("ln(x) — depth 3, 20 restarts", r1, time.time() - t0)

# ── Target 2: [ln(p)]²  ──────────────────────────────────────────────────────
# The dominant BPR fine-structure term.  Function of p, not x.
# Direct search EML K≈35 (x² needs K≈19, composed with ln depth-3 tree).
# Previous attempt used x_range up to P_DEFAULT*2 ≈ 209k which caused exp()
# saturation inside EML (loss ~2.5e43 = e^50 squared).  Fix: narrow range so
# ln(x) stays ≤ ln(100) ≈ 4.6, and z-score targets so outputs are O(1).
print("\n\nTarget 2: [ln(p)]²  [dominant BPR 1/α term, ~97% of value]")
for depth, restarts, steps, lr_ in [(4, 20, 6000, 0.02), (5, 10, 8000, 0.01)]:
    t0 = time.time()
    r = fit(
        target_fn=target_bpr_screening,
        depth=depth,
        x_range=(2.0, 100.0),        # ln(100)²≈21; keeps EML inputs finite
        n_points=64,
        n_steps=steps,
        n_restarts=restarts,
        lr=lr_,
        seed=7,
        normalize_y=True,            # z-score targets → O(1) outputs during training
    )
    _report(f"[ln(p)]²  — depth {depth}, {restarts} restarts", r, time.time() - t0)
    if r.success:
        print(f"  *** SUCCESS at depth {depth} ***")
        break

# ── Target 3: full 1/α(p)  ───────────────────────────────────────────────────
# Full BPR formula: [ln(p)]² + 3 + γ − 1/(2π)
# Expected K > 150 — recovery at depth ≤ 5 is very unlikely.
# We run at depth 4 to see how close the fit gets continuously.
print("\n\nTarget 3: 1/α(p) — full BPR formula  [K>150, continuous fit only]")
t0 = time.time()
r3 = fit(
    target_fn=target_bpr_alpha,
    depth=4,
    x_range=(100.0, 1000.0),        # narrowed from (1000, 3*P_DEFAULT) to avoid overflow
    n_points=64,
    n_steps=5000,
    n_restarts=5,
    lr=0.02,
    seed=13,
    normalize_y=True,
)
_report("1/α(p) — depth 4, 5 restarts", r3, time.time() - t0)
print(f"\n  BPR value at p={P_DEFAULT}: {target_bpr_alpha(np.array([float(P_DEFAULT)]))[0]:.6f}")
print(f"  Formula: [ln(p)]² + 3 + γ − 1/(2π) ≈ 137.031")
print(f"  (Full symbolic recovery needs K>150; depth 4 gives a partial fit)")

print(f"\n{'═'*62}")
print("  Summary")
print(f"{'─'*62}")
print(f"  ln(x)    depth 3  success={r1.success}  loss={r1.snapped_loss:.2e}")
print(f"  [ln(p)]² depth 4  success={r.success}   loss={r.snapped_loss:.2e}")
print(f"  1/α(p)   depth 4  success={r3.success}  loss={r3.snapped_loss:.2e}")
print(f"{'═'*62}")
