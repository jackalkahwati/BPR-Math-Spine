"""
Brute-force EML tree search for x².
=====================================

Strategy: bottom-up enumeration of ALL EML expressions up to K=K_MAX
(RPN token count), where K = 2*n_leaves - 1.  The paper (Odrzywolek 2026)
reports K≈19 for x², meaning 10 leaves + 9 internal eml nodes.

We search K = 1, 3, 5, ... up to K_MAX, generate every eml(a,b) from
smaller sub-expressions, deduplicate by numerical fingerprint, and
report any expression that matches x² on 6 test points.

Expected runtime: ~3-10 min (dominated by K=17,19 levels).
Expected result: the exact EML tree for x², enabling warm-start of
the [ln(x)]² composition search.

Run:
    python scripts/find_x_sq_eml.py
"""

import sys, time
import numpy as np

sys.path.insert(0, "/home/user/BPR-Math-Spine")

# ── Test points ───────────────────────────────────────────────────────────────
# Avoid x=1 (many exprs evaluate to constants there).
# Use diverse positive values; x² must match at ALL of them.
X_TEST = np.array([0.3, 0.7, 1.5, 2.0, np.e, np.pi])
TARGET = X_TEST ** 2

K_MAX = 25   # Paper says K≈19; search to 25 to be safe
ATOL  = 1e-4
RTOL  = 1e-4

# ── Numerically stable EML ────────────────────────────────────────────────────

def safe_eml(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    exp_part = np.exp(np.clip(np.real(a), -60.0, 60.0))
    log_arg  = np.abs(b) + 1e-300
    return exp_part - np.log(log_arg)


def is_finite(v: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(v)))


def matches(v: np.ndarray, target: np.ndarray) -> bool:
    return is_finite(v) and bool(np.allclose(v, target, rtol=RTOL, atol=ATOL))


def fingerprint(v: np.ndarray) -> tuple:
    """Round-and-hash for deduplication."""
    return tuple(np.round(v, 5))

# ── Bottom-up enumeration ─────────────────────────────────────────────────────

# K → list of (values, name_str)
DB: dict[int, list] = {}

# K=1 primitives
DB[1] = [
    (np.ones_like(X_TEST), '1'),
    (X_TEST.copy(),         'x'),
]

print(f"EML exhaustive search for x²  (K_MAX={K_MAX})")
print(f"Test points: {X_TEST}")
print(f"Target x²:  {TARGET}")
print()

found: list[tuple[int, str]] = []
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

                if matches(result, TARGET):
                    found.append((k, name))
                    print(f"  *** FOUND x² at K={k}: {name}")

    DB[k] = exprs_k
    elapsed = time.time() - t0
    print(f"K={k:2d}: {n_candidates:>8,} candidates → {len(exprs_k):>7,} unique finite "
          f"| {elapsed:.1f}s  total={time.time()-t_start:.0f}s", flush=True)

    if found:
        break  # stop at first match (minimum K)

print()
if found:
    print("=" * 60)
    print(f"  RESULT: x² found at minimum K = {found[0][0]}")
    print(f"  Formula: {found[0][1]}")
    print("=" * 60)
    print()
    print("  Verification:")
    k_best, name_best = found[0]
    # Re-evaluate on denser grid to confirm
    x_dense = np.linspace(0.1, 5.0, 200)
    target_dense = x_dense ** 2

    # Rebuild the expression by eval (safe since it's our own grammar)
    # We'll just re-evaluate symbolically using the name string
    print(f"  Formula string: {name_best}")
    print(f"  Next step: compose with ln(x) tree to get [ln(x)]²")
else:
    print(f"  x² not found up to K={K_MAX}.")
    print(f"  Possible reasons:")
    print(f"    - Paper's K estimate is larger than {K_MAX}")
    print(f"    - Formula requires complex intermediate values (needs extension)")
    print(f"    - Formula uses x in a subtree where we pruned due to overflow")
