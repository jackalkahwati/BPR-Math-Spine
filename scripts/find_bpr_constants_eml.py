"""
Exhaustive constant-EML search for BPR formula constants.

Searches for EML trees whose leaves are ALL Const(1) — no variable.
Targets: 3 (= z/2), γ, 1/(2π), and their combinations.

The search is much faster than the function search because:
  - only one leaf value (1.0), so many expressions collapse to same constant
  - fingerprinting collapses the space quickly

Run:
    python scripts/find_bpr_constants_eml.py
"""

import sys, time
import numpy as np

sys.path.insert(0, "/home/user/BPR-Math-Spine")

# ── Targets ───────────────────────────────────────────────────────────────────
_GAMMA  = 0.5772156649015329
_TARGETS = {
    "3 (z/2)":         3.0,
    "γ":               _GAMMA,
    "1/(2π)":          1.0 / (2.0 * np.pi),
    "-1/(2π)":        -1.0 / (2.0 * np.pi),
    "C (tail sum)":    3.0 + _GAMMA - 1.0 / (2.0 * np.pi),
    "e":               np.e,          # sanity check — should find at K=3
    "0":               0.0,           # sanity check — should find at K=7
}

K_MAX = 35
ATOL  = 1e-7
RTOL  = 1e-7

# ── Numerically stable EML (constant version) ─────────────────────────────────

def safe_eml(a: float, b: float) -> float:
    exp_part = np.exp(np.clip(a, -60.0, 60.0))
    if b == 0:
        return np.inf
    log_arg = abs(b) + 1e-300
    return exp_part - np.log(log_arg)


def is_finite(v: float) -> bool:
    return np.isfinite(v)


def fingerprint(v: float) -> int:
    return round(v * 1e7)


# ── Bottom-up enumeration of constant-only EML trees ─────────────────────────

DB: dict[int, list] = {1: [(1.0, '1')]}

print("EML constant search  (K_MAX=%d)" % K_MAX)
print("Targets:", list(_TARGETS.keys()))
print()

found: dict[str, tuple[int, str]] = {}
t_start = time.time()

for k in range(3, K_MAX + 1, 2):
    t0 = time.time()
    seen: set = set()
    exprs_k: list = []
    n_candidates = 0

    for k_l in range(1, k - 1, 2):
        k_r = k - 1 - k_l
        if k_r < 1 or k_r not in DB or k_l not in DB:
            continue
        for (val_l, name_l) in DB[k_l]:
            for (val_r, name_r) in DB[k_r]:
                n_candidates += 1
                try:
                    result = safe_eml(val_l, val_r)
                except Exception:
                    continue
                if not is_finite(result):
                    continue
                fp = fingerprint(result)
                if fp in seen:
                    continue
                seen.add(fp)
                name = f'eml({name_l},{name_r})'
                exprs_k.append((result, name))
                for label, target in _TARGETS.items():
                    if label not in found:
                        if abs(result - target) < ATOL or (target != 0 and abs((result - target)/target) < RTOL):
                            found[label] = (k, name, result)
                            print(f"  *** FOUND {label!r} at K={k}: {name}  [{result:.10f}]")

    DB[k] = exprs_k
    elapsed = time.time() - t0
    print(f"K={k:2d}: {n_candidates:>8,} candidates → {len(exprs_k):>6,} unique finite"
          f" | {elapsed:.1f}s  total={time.time()-t_start:.0f}s  "
          f"found={len(found)}/{len(_TARGETS)}", flush=True)

    if len(found) == len(_TARGETS):
        print("\nAll targets found — stopping early.")
        break

print()
print("=" * 60)
for label, (k, name, val) in found.items():
    print(f"  {label:20s}  K={k:2d}  val={val:.10f}  {name}")
missing = [t for t in _TARGETS if t not in found]
if missing:
    print(f"\n  NOT FOUND (up to K={K_MAX}): {missing}")
print("=" * 60)
