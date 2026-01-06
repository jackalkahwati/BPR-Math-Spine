"""
RPST Substrate - Eq 0a/0b scaffolding

Defines a prime-modulus substrate state on Z_p and basic arithmetic utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def legendre_symbol(a: int, p: int) -> int:
    """
    Compute the Legendre symbol (a/p) for odd prime p.

    Returns:
        0 if a ≡ 0 (mod p)
        1 if a is a quadratic residue mod p
        -1 otherwise
    """
    a = int(a) % int(p)
    p = int(p)
    if p < 2:
        raise ValueError("p must be >= 2")
    if a == 0:
        return 0
    # Euler's criterion: a^((p-1)/2) mod p is 1 or p-1 for prime p.
    ls = pow(a, (p - 1) // 2, p)
    if ls == 1:
        return 1
    if ls == p - 1:
        return -1
    # If p isn't prime, Euler criterion can return other values.
    raise ValueError("Legendre symbol undefined (p may not be prime).")


class PrimeField:
    """Arithmetic in the finite field Z_p (prime modulus)."""

    def __init__(self, p: int):
        self.p = int(p)
        if self.p <= 1:
            raise ValueError("p must be > 1")

    def mod(self, x):
        return np.mod(x, self.p)

    def add(self, a, b):
        return self.mod(np.asarray(a) + np.asarray(b))

    def sub(self, a, b):
        return self.mod(np.asarray(a) - np.asarray(b))

    def mul(self, a, b):
        return self.mod(np.asarray(a) * np.asarray(b))


@dataclass(frozen=True)
class SubstrateState:
    """
    RPST phase-space state on Z_p.

    q : positions in Z_p
    pi: conjugate momenta in Z_p
    p : prime modulus
    """

    q: np.ndarray
    pi: np.ndarray
    p: int
    t: int = 0

    def __post_init__(self):
        object.__setattr__(self, "p", int(self.p))
        q = np.asarray(self.q, dtype=int) % self.p
        pi = np.asarray(self.pi, dtype=int) % self.p
        if q.shape != pi.shape:
            raise ValueError("q and pi must have the same shape")
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "pi", pi)

    @property
    def n(self) -> int:
        return int(self.q.size)

    def with_time(self, t: int) -> "SubstrateState":
        return SubstrateState(q=self.q, pi=self.pi, p=self.p, t=int(t))


