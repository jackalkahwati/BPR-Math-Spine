"""
Search for addition-like functions in EML.

Tests whether `x + c` (for several constants c) is expressible
in the EML grammar  S → 1 | x | eml(S, S).

If x+1 is NOT found by K=K_MAX, that establishes that addition of
independent EML terms is outside the finite EML closure — meaning the
BPR formula 1/α = [ln(p)]² + C cannot be expressed as a single EML
tree, and each term must be represented separately.

Also tests x+e, x+2, x-1 (forms of shift and subtraction).

Run:
    python scripts/find_addition_eml.py
"""

import sys, time
import numpy as np

sys.path.insert(0, "/home/user/BPR-Math-Spine")

X_TEST = np.array([0.3, 0.7, 1.5, 2.0, np.e, np.pi])

TARGETS = {
    "x+1":    X_TEST + 1.0,
    "x+e":    X_TEST + np.e,
    "x+2":    X_TEST + 2.0,
    "x-1":    X_TEST - 1.0,
    "x·2":    X_TEST * 2.0,    # sanity: 2x = exp(ln(x)+ln(x)) — might be hard
    "x/2":    X_TEST / 2.0,
    "x²+1":   X_TEST**2 + 1.0,
    "1/x":    1.0 / X_TEST,    # reciprocal — should be findable
}

K_MAX = 23
ATOL  = 1e-4
RTOL  = 1e-4

def safe_eml(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    exp_part = np.exp(np.clip(np.real(a), -60.0, 60.0))
    log_arg  = np.abs(b) + 1e-300
    return exp_part - np.log(log_arg)


def is_finite(v: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(v)))


def matches(v: np.ndarray, target: np.ndarray) -> bool:
    return is_finite(v) and bool(np.allclose(v, target, rtol=RTOL, atol=ATOL))


def fingerprint(v: np.ndarray) -> tuple:
    return tuple(np.round(v, 5))


DB: dict[int, list] = {
    1: [
        (np.ones_like(X_TEST), '1'),
        (X_TEST.copy(),         'x'),
    ]
}

found: dict[str, tuple[int, str]] = {}
t_start = time.time()

print("EML function search: addition-like functions  (K_MAX=%d)" % K_MAX)
print("Targets:", list(TARGETS.keys()))
print()

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
                for label, target in TARGETS.items():
                    if label not in found and matches(result, target):
                        found[label] = (k, name)
                        print(f"  *** FOUND {label!r} at K={k}: {name}")

    DB[k] = exprs_k
    elapsed = time.time() - t0
    print(f"K={k:2d}: {n_candidates:>8,} candidates → {len(exprs_k):>7,} unique finite"
          f" | {elapsed:.1f}s  total={time.time()-t_start:.0f}s"
          f"  found={len(found)}/{len(TARGETS)}", flush=True)

print()
print("=" * 60)
for label, (k, name) in found.items():
    print(f"  FOUND  {label:10s}  K={k:2d}  {name}")
missing = [t for t in TARGETS if t not in found]
if missing:
    print(f"\n  NOT FOUND (up to K={K_MAX}): {missing}")
    print()
    print("  If x+c is missing: addition of independent EML terms is outside")
    print("  the finite EML closure.  Each BPR term must be composed separately.")
    print("  If 1/x, x·2 are found but x+1 is not: EML closure = mult/div,")
    print("  not addition.  This defines the scope of the enumeration roadmap.")
print("=" * 60)
