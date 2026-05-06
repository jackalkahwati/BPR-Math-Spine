"""
ECDLP solvers with group-operation counters.
Brute force, Baby-Step Giant-Step, Pollard's rho.
"""
from __future__ import annotations
import math
import random

from curve import Curve, Point, add, neg, scalar_mult


def brute_force(E: Curve, G: Point, Q: Point, n: int) -> int:
    """O(n) sequential search. Returns k such that k*G = Q."""
    E.counter.reset()
    R = Point.infinity()
    for k in range(n):
        if R == Q or (R.is_inf and Q.is_inf):
            return k
        R = add(R, G, E)
    raise RuntimeError("brute force failed")


def bsgs(E: Curve, G: Point, Q: Point, n: int) -> int:
    """Baby-step giant-step: O(sqrt n) time, O(sqrt n) memory."""
    E.counter.reset()
    m = int(math.isqrt(n)) + 1
    # Baby steps: store {j*G : j in [0,m)}
    table = {}
    R = Point.infinity()
    for j in range(m):
        key = (R.x, R.y, R.is_inf)
        table[key] = j
        R = add(R, G, E)
    # Compute -m*G
    mG = scalar_mult(m, G, E)
    neg_mG = neg(mG, E)
    # Giant steps: Q + i*(-m*G)
    S = Q
    for i in range(m + 1):
        key = (S.x, S.y, S.is_inf)
        if key in table:
            j = table[key]
            k = (i * m + j) % n
            if k != 0:
                return k
        S = add(S, neg_mG, E)
    raise RuntimeError("BSGS failed")


def pollard_rho(E: Curve, G: Point, Q: Point, n: int, *,
                seed: int = 0, max_iter_factor: int = 64) -> int:
    """Pollard's rho with 3-partition iteration. Floyd cycle detection.

    Standard partition: f(R, a, b) = R + R0/R1/R2 with three branches.
    """
    rng = random.Random(seed)

    # Build three random multipliers M_i = u_i*G + v_i*Q
    triples = []
    for _ in range(3):
        u = rng.randrange(1, n)
        v = rng.randrange(1, n)
        M = add(scalar_mult(u, G, E), scalar_mult(v, Q, E), E)
        triples.append((u, v, M))

    def step(R: Point, a: int, b: int) -> tuple[Point, int, int]:
        # Partition by R.x mod 3 (independent of k)
        idx = 0 if R.is_inf else (R.x % 3)
        u, v, M = triples[idx]
        return add(R, M, E), (a + u) % n, (b + v) % n

    # Reset counter AFTER setup
    E.counter.reset()

    a0 = rng.randrange(n)
    b0 = rng.randrange(n)
    R0 = add(scalar_mult(a0, G, E), scalar_mult(b0, Q, E), E)
    a, b = a0, b0
    A, B = a0, b0
    R, S = R0, R0

    max_iter = max_iter_factor * int(math.isqrt(n) + 1)
    for _ in range(max_iter):
        R, a, b = step(R, a, b)
        S, A, B = step(S, A, B)
        S, A, B = step(S, A, B)
        if (R.x, R.y, R.is_inf) == (S.x, S.y, S.is_inf):
            db = (b - B) % n
            if db == 0:
                # Failure; restart with fresh seed
                return pollard_rho(E, G, Q, n, seed=seed + 1,
                                   max_iter_factor=max_iter_factor)
            try:
                from curve import modinv
                k = ((A - a) * modinv(db, n)) % n
            except ZeroDivisionError:
                return pollard_rho(E, G, Q, n, seed=seed + 1,
                                   max_iter_factor=max_iter_factor)
            # verify
            if (scalar_mult(k, G, E).x, scalar_mult(k, G, E).y) == (Q.x, Q.y):
                return k
            return pollard_rho(E, G, Q, n, seed=seed + 1,
                               max_iter_factor=max_iter_factor)
    raise RuntimeError(f"rho exceeded max iterations (n={n})")
