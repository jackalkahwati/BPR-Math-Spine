"""
Toy elliptic curve over F_p with group-operation counter.

For research use on synthetic toy curves only. Not for production.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


def egcd(a: int, b: int) -> tuple[int, int, int]:
    if b == 0:
        return a, 1, 0
    g, x1, y1 = egcd(b, a % b)
    return g, y1, x1 - (a // b) * y1


def modinv(a: int, p: int) -> int:
    g, x, _ = egcd(a % p, p)
    if g != 1:
        raise ZeroDivisionError(f"no inverse for {a} mod {p}")
    return x % p


def is_qr(a: int, p: int) -> bool:
    a %= p
    if a == 0:
        return True
    return pow(a, (p - 1) // 2, p) == 1


def modsqrt(a: int, p: int) -> int:
    a %= p
    if a == 0:
        return 0
    if not is_qr(a, p):
        raise ValueError(f"{a} is not a QR mod {p}")
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)
    # Tonelli-Shanks
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while is_qr(z, p):
        z += 1
    m = s
    c = pow(z, q, p)
    t = pow(a, q, p)
    r = pow(a, (q + 1) // 2, p)
    while t != 1:
        i = 0
        temp = t
        while temp != 1:
            temp = (temp * temp) % p
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        m = i
        c = (b * b) % p
        t = (t * c) % p
        r = (r * b) % p
    return r


@dataclass
class OpCounter:
    adds: int = 0
    doubles: int = 0
    negs: int = 0

    @property
    def total(self) -> int:
        return self.adds + self.doubles

    def reset(self) -> None:
        self.adds = 0
        self.doubles = 0
        self.negs = 0


class Curve:
    """Short Weierstrass curve y^2 = x^3 + a*x + b over F_p."""

    def __init__(self, a: int, b: int, p: int):
        self.a = a % p
        self.b = b % p
        self.p = p
        disc = (-16 * (4 * pow(a, 3, p) + 27 * pow(b, 2, p))) % p
        if disc == 0:
            raise ValueError("singular curve")
        self.counter = OpCounter()

    def __repr__(self) -> str:
        return f"Curve(y^2 = x^3 + {self.a}*x + {self.b} mod {self.p})"

    def contains(self, P: "Point") -> bool:
        if P.is_inf:
            return True
        return (P.y * P.y - P.x * P.x * P.x - self.a * P.x - self.b) % self.p == 0


@dataclass(frozen=True)
class Point:
    x: int
    y: int
    is_inf: bool = False

    @staticmethod
    def infinity() -> "Point":
        return Point(0, 0, is_inf=True)


def neg(P: Point, E: Curve) -> Point:
    if P.is_inf:
        return P
    E.counter.negs += 1
    return Point(P.x, (-P.y) % E.p)


def add(P: Point, Q: Point, E: Curve) -> Point:
    """Group law on E. Increments counter. Doubles counted separately."""
    p = E.p
    if P.is_inf:
        return Q
    if Q.is_inf:
        return P
    if P.x == Q.x:
        if (P.y + Q.y) % p == 0:
            return Point.infinity()
        # P == Q -> double
        E.counter.doubles += 1
        s_num = (3 * P.x * P.x + E.a) % p
        s_den = modinv((2 * P.y) % p, p)
        s = (s_num * s_den) % p
    else:
        E.counter.adds += 1
        s = ((Q.y - P.y) % p) * modinv((Q.x - P.x) % p, p) % p
    x3 = (s * s - P.x - Q.x) % p
    y3 = (s * (P.x - x3) - P.y) % p
    return Point(x3, y3)


def scalar_mult(k: int, P: Point, E: Curve) -> Point:
    """Double-and-add. Counts each underlying group op."""
    if k == 0 or P.is_inf:
        return Point.infinity()
    if k < 0:
        return scalar_mult(-k, neg(P, E), E)
    R = Point.infinity()
    Q = P
    while k > 0:
        if k & 1:
            R = add(R, Q, E)
        Q = add(Q, Q, E)
        k >>= 1
    return R


def lift_x(x: int, E: Curve) -> Optional[Point]:
    """Return a point with given x (lower y); None if x not on curve."""
    rhs = (pow(x, 3, E.p) + E.a * x + E.b) % E.p
    if not is_qr(rhs, E.p):
        return None
    y = modsqrt(rhs, E.p)
    return Point(x, y)


def curve_order_enum(E: Curve) -> int:
    """Count |E(F_p)| by enumerating x. Cheap only for small p (<= ~2^20)."""
    n = 1  # point at infinity
    for x in range(E.p):
        rhs = (pow(x, 3, E.p) + E.a * x + E.b) % E.p
        if rhs == 0:
            n += 1
        elif is_qr(rhs, E.p):
            n += 2
    return n
