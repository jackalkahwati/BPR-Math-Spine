"""
Toy curve generation. Random Weierstrass curves over F_p with prime
(or near-prime) point order. Seeded for reproducibility.
"""
from __future__ import annotations
import random
from dataclasses import dataclass

from curve import Curve, Point, lift_x, scalar_mult, curve_order_enum, add


def _is_probable_prime(n: int, k: int = 20) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    rng = random.Random(12345)
    for _ in range(k):
        a = rng.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime(n: int) -> int:
    if n < 2:
        return 2
    n |= 1
    while not _is_probable_prime(n):
        n += 2
    return n


def find_point_order(P: Point, E: Curve, N: int) -> int:
    """Order of P given that it divides N. Factor N and probe."""
    factors = {}
    m = N
    d = 2
    while d * d <= m:
        while m % d == 0:
            factors[d] = factors.get(d, 0) + 1
            m //= d
        d += 1
    if m > 1:
        factors[m] = factors.get(m, 0) + 1
    order = N
    for q in factors:
        while order % q == 0:
            test = order // q
            if scalar_mult(test, P, E).is_inf:
                order = test
            else:
                break
    return order


@dataclass
class ToyInstance:
    E: Curve
    G: Point
    n: int  # prime order of G
    bits: int
    seed: int


def make_toy_instance(bits: int, seed: int, *, weak: str | None = None) -> ToyInstance:
    """Generate a toy curve with a generator G of large prime order n.

    weak=None       : random-Weierstrass, near-prime order target
    weak='anomalous': try to construct |E|=p (Smart's attack applies)
    """
    rng = random.Random(f"{bits}-{seed}-{weak}")

    # Pick prime p of given bit width
    p = next_prime(rng.randrange(1 << (bits - 1), 1 << bits))

    for attempt in range(2000):
        a = rng.randrange(p)
        b = rng.randrange(p)
        try:
            E = Curve(a, b, p)
        except ValueError:
            continue

        N = curve_order_enum(E)
        if weak == "anomalous":
            if N != p:
                continue
        # Find largest prime factor of N as candidate generator order
        m = N
        largest_prime = 1
        d = 2
        while d * d <= m:
            while m % d == 0:
                largest_prime = max(largest_prime, d)
                m //= d
            d += 1
        if m > 1:
            largest_prime = max(largest_prime, m)
        if largest_prime < (1 << (bits - 2)):
            continue
        # Pick a random point and project to subgroup of order largest_prime
        cofactor = N // largest_prime
        for _ in range(50):
            x = rng.randrange(p)
            P = lift_x(x, E)
            if P is None:
                continue
            G = scalar_mult(cofactor, P, E)
            if not G.is_inf:
                actual = find_point_order(G, E, largest_prime)
                if actual == largest_prime:
                    E.counter.reset()
                    return ToyInstance(E=E, G=G, n=largest_prime,
                                       bits=bits, seed=seed)
    raise RuntimeError(f"failed to build toy instance bits={bits} seed={seed}")
