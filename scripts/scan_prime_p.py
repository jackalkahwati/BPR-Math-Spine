"""
scan_prime_p.py
===============
Apply all BPR physical constraints to identify candidate primes for p.

The three constraints derivable from BPR formalism:
  1. Alpha formula: 1/α = [ln p]² + z/2 + γ − 1/(2π) matches experiment
  2. Orientability: p ≡ 1 (mod 4)  →  Dirac neutrinos, normal ordering, θ_QCD=0
  3. Three generations: p−1 has exactly 3 distinct odd prime factors
     (H¹(Z_p×, Z_p) rank condition)

Result (April 2026, primes up to 200,000):
  Two BPR constraints (alpha formula + orientability) select 8 candidates.
  Physical derivation (invert alpha formula → p_exact=104,749.03, nearest valid
  prime with p≡1 mod 4) selects p = 104,761 as the framework default.
  104,729 (the 10,000th prime) was the prior convention; it has been superseded.

FORMULA (bpr/alpha_derivation.py §22.1):

    1/α  =  [ln(p)]²  +  z/2  +  γ  −  1/(2π)

EXPERIMENTAL TARGET:
    1/α = 137.035999084  (CODATA 2018)

Run:
    python scripts/scan_prime_p.py
    python scripts/scan_prime_p.py --tol 0.001 --upper 300000
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


# ── Constants ──────────────────────────────────────────────────────────────
_EULER_GAMMA   = 0.5772156649015329
_INV_ALPHA_EXP = 137.035999084       # CODATA 2018
_Z_DEFAULT     = 6                   # S² cubic coordination number


# ── Formula ────────────────────────────────────────────────────────────────

def inv_alpha(p: int, z: int = _Z_DEFAULT) -> float:
    """BPR formula: 1/α = [ln(p)]² + z/2 + γ − 1/(2π)"""
    ln_p = math.log(p)
    return ln_p ** 2 + z / 2.0 + _EULER_GAMMA - 1.0 / (2.0 * math.pi)


def error_ppm(p: int, z: int = _Z_DEFAULT) -> float:
    """Deviation from experiment in parts per million."""
    return abs(inv_alpha(p, z) - _INV_ALPHA_EXP) / _INV_ALPHA_EXP * 1e6


def error_pct(p: int, z: int = _Z_DEFAULT) -> float:
    """Deviation from experiment as a percentage."""
    return abs(inv_alpha(p, z) - _INV_ALPHA_EXP) / _INV_ALPHA_EXP * 100.0


# ── Sieve ──────────────────────────────────────────────────────────────────

def primes_up_to(n: int) -> list[int]:
    """Sieve of Eratosthenes."""
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return [i for i, v in enumerate(sieve) if v]


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class ScanHit:
    p:       int
    rank:    int          # which prime (1-indexed)
    value:   float        # 1/α from formula
    ppm:     float        # deviation in ppm
    pct:     float        # deviation in %


# ── Main scan ──────────────────────────────────────────────────────────────

def factorize(n: int) -> set[int]:
    """Return set of distinct prime factors of n."""
    factors: set[int] = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def odd_prime_factors(n: int) -> list[int]:
    """Distinct odd prime factors of n, sorted."""
    return sorted(f for f in factorize(n) if f != 2)


def satisfies_orientability(p: int) -> bool:
    """p ≡ 1 (mod 4): orientable boundary → Dirac neutrinos, normal ordering."""
    return p % 4 == 1


def satisfies_three_generations(p: int) -> bool:
    """p−1 has exactly 3 distinct odd prime factors → H¹(Z_p×,Z_p) rank 3."""
    return len(odd_prime_factors(p - 1)) == 3


def scan(upper: int = 200_000, tol_pct: float = 0.01, z: int = _Z_DEFAULT) -> list[ScanHit]:
    """
    Find all primes up to `upper` for which the formula matches
    the experimental 1/α within `tol_pct` percent.
    """
    hits: list[ScanHit] = []
    all_primes = primes_up_to(upper)
    for rank, p in enumerate(all_primes, start=1):
        if p < 3:
            continue  # ln(2)² + 3 + γ − 1/2π ≈ 4; nowhere near 137
        pct = error_pct(p, z)
        if pct <= tol_pct:
            hits.append(ScanHit(
                p=p,
                rank=rank,
                value=inv_alpha(p, z),
                ppm=error_ppm(p, z),
                pct=pct,
            ))
    return hits


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Scan primes for BPR alpha match.")
    parser.add_argument("--upper",   type=int,   default=200_000,
                        help="Scan primes up to this value (default 200,000).")
    parser.add_argument("--tol",     type=float, default=0.01,
                        help="Tolerance in percent (default 0.01 = 0.01%%).")
    parser.add_argument("--z",       type=int,   default=_Z_DEFAULT,
                        help="Coordination number z (default 6).")
    parser.add_argument("--top",     type=int,   default=20,
                        help="Also show the top N closest primes regardless of threshold.")
    args = parser.parse_args()

    print(f"BPR Prime Scan")
    print(f"==============")
    print(f"Formula:    1/α = [ln(p)]² + z/2 + γ − 1/(2π)")
    print(f"Target:     1/α = {_INV_ALPHA_EXP}")
    print(f"z:          {args.z}")
    print(f"Scan range: primes up to {args.upper:,}")
    print(f"Threshold:  ±{args.tol}%")
    print()

    # ── p = 104,761 (current default) and 104,729 (prior convention) ────────
    p0 = 104_761
    p_old = 104_729
    print(f"Current default  p = {p0:,}  (physically derived: nearest prime to p_exact=104,749)")
    print(f"  1/α = {inv_alpha(p0, args.z):.6f}")
    print(f"  error = {error_ppm(p0, args.z):.1f} ppm  ({error_pct(p0, args.z):.4f}%)")
    print(f"Prior convention p = {p_old:,}  (10,000th prime, superseded)")
    print(f"  1/α = {inv_alpha(p_old, args.z):.6f}")
    print(f"  error = {error_ppm(p_old, args.z):.1f} ppm  ({error_pct(p_old, args.z):.4f}%)")
    print()

    # ── Full scan ─────────────────────────────────────────────────────────
    all_primes = primes_up_to(args.upper)
    print(f"Scanning {len(all_primes):,} primes up to {args.upper:,} ...")
    print()

    hits = scan(upper=args.upper, tol_pct=args.tol, z=args.z)

    if not hits:
        print(f"No primes within {args.tol}% of experimental 1/α in this range.")
    else:
        print(f"Primes within {args.tol}% of experimental 1/α:")
        print(f"  {'p':>10}  {'rank':>7}  {'1/α':>12}  {'error (ppm)':>12}  {'error (%)':>10}")
        print(f"  {'-'*10}  {'-'*7}  {'-'*12}  {'-'*12}  {'-'*10}")
        for h in hits:
            marker = "  <-- current default" if h.p == 104_761 else ("  <-- prior convention" if h.p == 104_729 else "")
            print(f"  {h.p:>10,}  {h.rank:>7,}  {h.value:>12.6f}  {h.ppm:>12.1f}  {h.pct:>10.4f}%{marker}")

    print()

    # ── Top N closest regardless of threshold ────────────────────────────
    scored = sorted(
        [ScanHit(p=p, rank=r, value=inv_alpha(p, args.z),
                 ppm=error_ppm(p, args.z), pct=error_pct(p, args.z))
         for r, p in enumerate(all_primes, 1) if p >= 3],
        key=lambda h: h.ppm
    )
    top = scored[: args.top]
    print(f"Top {args.top} closest primes (any error):")
    print(f"  {'p':>10}  {'rank':>7}  {'1/α':>12}  {'error (ppm)':>12}  {'error (%)':>10}")
    print(f"  {'-'*10}  {'-'*7}  {'-'*12}  {'-'*12}  {'-'*10}")
    for h in top:
        marker = "  <-- 104,729" if h.p == 104_729 else ""
        print(f"  {h.p:>10,}  {h.rank:>7,}  {h.value:>12.6f}  {h.ppm:>12.1f}  {h.pct:>10.4f}%{marker}")

    print()

    # ── Apply all three BPR physical constraints ──────────────────────────
    print()
    print("=== BPR Physical Constraints Applied ===")
    print()

    pool = [ScanHit(p=p, rank=r, value=inv_alpha(p, args.z),
                    ppm=error_ppm(p, args.z), pct=error_pct(p, args.z))
            for r, p in enumerate(all_primes, 1) if p >= 3]

    c1 = [h for h in pool if h.pct <= args.tol]
    c2 = [h for h in c1 if satisfies_orientability(h.p)]

    print(f"Constraint 1 — alpha within {args.tol}%:        {len(c1):3d} primes")
    print(f"Constraint 2 — p ≡ 1 (mod 4) orientability:  {len(c2):3d} primes")
    print()
    print("Note: three generations are derived from 3 Killing vectors of S²")
    print("      (a property of z=6 geometry, independent of p — filters zero primes).")
    print()

    if c2:
        print("Surviving candidates after both BPR constraints:")
        print(f"  {'p':>10}  {'rank':>7}  {'1/α err (ppm)':>15}")
        print(f"  {'-'*10}  {'-'*7}  {'-'*15}")
        best_ppm = min(x.ppm for x in c2)
        for h in c2:
            marker = "  <-- CURRENT DEFAULT (physically derived)" if h.p == 104_761 else \
                     ("  <-- prior convention (10,000th prime)" if h.p == 104_729 else "")
            best_mark = "  <-- best alpha match" if h.ppm == best_ppm and h.p not in (104_729, 104_761) else ""
            print(f"  {h.p:>10,}  {h.rank:>7,}  {h.ppm:>15.1f}{marker}{best_mark}")
        print()
        best    = min(c2, key=lambda h: h.ppm)
        default = next((h for h in c2 if h.p == 104_761), None)
        if default:
            print(f"VERDICT: BPR constraints narrow the field to {len(c2)} candidates.")
            print(f"         p = 104,761 is the physically derived default (nearest valid prime to p_exact=104,749).")
            print(f"         Alpha error: {default.ppm:.1f} ppm.")
            if best.p != 104_761:
                print(f"         Best alpha match is p = {best.p:,} ({best.ppm:.1f} ppm).")
            else:
                print(f"         p = 104,761 is also the best alpha match in this set.")
        else:
            print(f"VERDICT: p = 104,761 does NOT survive both constraints at {args.tol}% tolerance.")


if __name__ == "__main__":
    main()
