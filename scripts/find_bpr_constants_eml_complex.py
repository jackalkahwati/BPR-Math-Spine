"""
Exhaustive constant-EML search using TRUE COMPLEX arithmetic.

The safe_eml search (log|b|) can find spurious matches when intermediate
values go negative.  This script uses the actual complex EML operator
from bpr/eml.py and only reports results where the final value is REAL
(|imaginary part| < 1e-8).

Run:
    python scripts/find_bpr_constants_eml_complex.py
"""

import sys, time
import numpy as np

sys.path.insert(0, "/home/user/BPR-Math-Spine")
from bpr.eml import eml

_GAMMA = 0.5772156649015329
_TARGETS = {
    "1 (sanity)":      1.0,
    "e (sanity)":      np.e,
    "0 (sanity)":      0.0,
    "2":               2.0,
    "3 (z/2)":         3.0,
    "4":               4.0,
    "e-1":             np.e - 1,
    "2e-1":            2*np.e - 1,
    "γ":               _GAMMA,
    "1/(2π)":          1/(2*np.pi),
    "-1/(2π)":        -1/(2*np.pi),
}

K_MAX = 29
ATOL  = 1e-8    # tighter than safe_eml search

def safe_complex_eml(a: complex, b: complex) -> complex | None:
    """True complex EML, return None if result is not finite."""
    try:
        result = eml(a, b)
        if not np.isfinite(result.real) or not np.isfinite(result.imag):
            return None
        if abs(result.real) > 1e10:
            return None  # too large to be useful
        return result
    except Exception:
        return None


def is_real(v: complex) -> bool:
    return abs(v.imag) < 1e-8


def fingerprint(v: complex) -> tuple:
    """Round real part; only fingerprint real-valued results."""
    return round(v.real * 1e6), round(abs(v.imag) * 1e6)


# K=1: just the constant 1 (real)
DB: dict[int, list] = {1: [(complex(1.0), '1')]}

found: dict[str, tuple] = {}
t_start = time.time()

print(f"Complex EML constant search (K_MAX={K_MAX})")
print("Targets:", list(_TARGETS.keys()))
print()

for k in range(3, K_MAX + 1, 2):
    t0 = time.time()
    seen: set = set()
    exprs_k: list = []
    n_candidates = 0
    n_real = 0

    for k_l in range(1, k - 1, 2):
        k_r = k - 1 - k_l
        if k_r < 1 or k_r not in DB or k_l not in DB:
            continue
        for (val_l, name_l) in DB[k_l]:
            for (val_r, name_r) in DB[k_r]:
                n_candidates += 1
                result = safe_complex_eml(val_l, val_r)
                if result is None:
                    continue
                fp = fingerprint(result)
                if fp in seen:
                    continue
                seen.add(fp)
                name = f'eml({name_l},{name_r})'
                exprs_k.append((result, name))
                if is_real(result):
                    n_real += 1
                    rv = result.real
                    for label, target in _TARGETS.items():
                        if label not in found:
                            if abs(rv - target) < ATOL:
                                found[label] = (k, name, rv)
                                print(f"  *** FOUND {label!r} at K={k}: {name}  [{rv:.10f}]")

    DB[k] = exprs_k
    elapsed = time.time() - t0
    print(f"K={k:2d}: {n_candidates:>8,} cands → {len(exprs_k):>6,} finite"
          f" ({n_real:>6,} real) | {elapsed:.1f}s total={time.time()-t_start:.0f}s"
          f"  found={len(found)}/{len(_TARGETS)}", flush=True)

    if len(found) == len(_TARGETS):
        print("\nAll targets found — stopping early.")
        break

print()
print("=" * 60)
for label, (k, name, val) in sorted(found.items(), key=lambda x: x[1][0]):
    print(f"  {label:15s}  K={k:2d}  val={val:.10f}")
    print(f"    {name}")
missing = [t for t in _TARGETS if t not in found]
if missing:
    print(f"\n  NOT found (up to K={K_MAX}): {missing}")
print("=" * 60)
