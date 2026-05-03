"""Fusion reaction definitions for Helionis trade studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from helionis.constants import MEV_TO_J


@dataclass(frozen=True)
class Reaction:
    """Energy partition for a fusion reaction branch or averaged branch set."""

    key: str
    name: str
    fuels: Tuple[str, ...]
    q_mev: float
    charged_mev: float
    neutron_mev: float
    same_species: bool = False
    notes: str = ""

    @property
    def q_j(self) -> float:
        """Total energy per reaction in joules."""
        return self.q_mev * MEV_TO_J

    @property
    def charged_j(self) -> float:
        """Charged-particle energy per reaction in joules."""
        return self.charged_mev * MEV_TO_J

    @property
    def neutron_j(self) -> float:
        """Neutron energy per reaction in joules."""
        return self.neutron_mev * MEV_TO_J

    @property
    def charged_fraction(self) -> float:
        """Fraction of reaction energy carried by charged particles."""
        return self.charged_mev / self.q_mev

    @property
    def neutron_fraction(self) -> float:
        """Fraction of reaction energy carried by neutrons."""
        return self.neutron_mev / self.q_mev


D_HE3 = Reaction(
    key="d_he3",
    name="D-He3",
    fuels=("D", "He3"),
    q_mev=18.353,
    charged_mev=18.353,
    neutron_mev=0.0,
    notes="Primary aneutronic branch: D + He3 -> alpha + proton.",
)

D_T = Reaction(
    key="d_t",
    name="D-T",
    fuels=("D", "T"),
    q_mev=17.589,
    charged_mev=3.5,
    neutron_mev=14.1,
    notes="Mainstream baseline: high 14.1 MeV neutron energy fraction.",
)

D_D_AVERAGE = Reaction(
    key="d_d_average",
    name="D-D average",
    fuels=("D", "D"),
    q_mev=(4.033 + 3.269) / 2.0,
    charged_mev=((4.033) + (3.269 - 2.45)) / 2.0,
    neutron_mev=2.45 / 2.0,
    same_species=True,
    notes="50/50 average of D(d,p)T and D(d,n)He3 side branches.",
)

REACTIONS = {
    D_HE3.key: D_HE3,
    D_T.key: D_T,
    D_D_AVERAGE.key: D_D_AVERAGE,
}


def get_reaction(key: str) -> Reaction:
    """Return a known reaction by key."""
    try:
        return REACTIONS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(REACTIONS))
        raise ValueError(f"Unknown reaction '{key}'. Valid reactions: {valid}") from exc
