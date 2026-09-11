"""Conditional mathematical checks for the repaired flavor foundations.

These functions do not derive physical families or flavor masses. See
doc/derivations/{generations_from_CFT,color_bundle_index}.md.
"""
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from typing import Tuple


def _integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def compact_boson_weights(
    momentum: int, winding: int, radius_squared: Fraction = Fraction(3)
) -> Tuple[Fraction, Fraction]:
    """(h, hbar) at alpha-prime=1: p_L,p_R=m/R +/- nR, h=p_L²/4.

    The radius is a specified input. The untwisted integer charge lattice
    has conformal spin h-hbar=momentum*winding, never half-integral.
    """
    m = _integer(momentum, "momentum")
    n = _integer(winding, "winding")
    r2 = Fraction(radius_squared)
    if r2 <= 0:
        raise ValueError("radius_squared must be positive")
    diagonal = (m * m / r2 + n * n * r2) / 4
    cross = Fraction(m * n, 2)
    return diagonal + cross, diagonal - cross


@dataclass(frozen=True)
class DiracZeroModes:
    """Chiral dimensions for a line twist of the spin Dirac operator on S²."""
    positive: int
    negative: int

    @property
    def index(self) -> int:
        return self.positive - self.negative


def sphere_line_zero_modes(degree: int) -> DiracZeroModes:
    """Count H⁰(O(q-1)) and H¹(O(q-1)) on CP¹ for specified degree q.

    Uses the spin structure K^(1/2)=O(-1) and the line bundle O(q).
    The count is not a prediction of q or of spacetime generations.
    """
    q = _integer(degree, "degree")
    return DiracZeroModes(max(q, 0), max(-q, 0))


def su_bundle_twisted_index(rank: int, line_degree: int = 0) -> int:
    """Index of D_(E tensor O(q)) on S², E an ordinary SU(rank) bundle.

    c₁(E)=0; c₁(E tensor O(q))=rank*q. Index counts net components,
    including color multiplicity, not copies of a color representation.
    It does not count possible additional pairs of opposite-chirality modes.
    """
    r = _integer(rank, "rank")
    q = _integer(line_degree, "line_degree")
    if r < 2:
        raise ValueError("rank must be at least 2 for this SU(rank) API")
    return r * q
