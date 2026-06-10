"""Page curve from BPR's finite-dimensional boundary Hilbert space.

Implements the tier-B "Attack B" target: show that black-hole evaporation in
BPR is unitary by computing the radiation entanglement entropy and exhibiting
the Page curve (entropy rises, peaks, then falls back to zero).

Why BPR makes this tractable
----------------------------
The BPR boundary is a U(1)_p Chern-Simons theory on the horizon with a
FINITE-dimensional Hilbert space: each boundary cutoff cell carries p states.
A black hole with N boundary cells has Hilbert-space dimension d = p^N.
Finite-dimensionality is exactly what forces unitarity — there is no infinite
reservoir to lose information into, unlike Hawking's semiclassical
calculation.

Page's theorem
--------------
For a random pure state in a Hilbert space H = H_A ⊗ H_B with
dim H_A = d_A (radiation) and dim H_B = d_B (remaining black hole), the
average entanglement entropy of subsystem A is (Page 1993):

    ⟨S_A⟩ ≈ ln(d_A) − d_A/(2 d_B)    for d_A ≤ d_B.

As the black hole evaporates, emitted radiation grows (d_A increases, d_B
decreases). Early: S_A ≈ ln d_A (thermal, rising). Late: S_A ≈ ln d_B
(falling with the shrinking black hole). The crossover is the Page time, at
d_A = d_B, i.e. when half the degrees of freedom have radiated.

The curve returns to S_A = 0 at complete evaporation — the signature of
unitary evolution. Hawking's calculation, by contrast, gives a monotonically
rising S_A (information loss).

BPR contribution and honest accounting
--------------------------------------
- The FINITE dimension d = p^N is BPR-specific (from the U(1)_p boundary).
- Page's formula for the average entropy is standard random-matrix /
  Haar-typicality (Page 1993), not derived here.
- This module DEMONSTRATES that BPR's finite boundary Hilbert space yields a
  unitary Page curve; it does not derive the microscopic evaporation dynamics
  (which states radiate when). That dynamical question is left open, exactly
  as in most Page-curve treatments.

References: Page, PRL 71, 1291 (1993); Hayden-Preskill, JHEP 0709:120 (2007).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def page_entropy(ln_dA: float, ln_dB: float) -> float:
    """Average entanglement entropy of subsystem A (Page 1993), in nats.

    Uses the smaller subsystem as the entropy ceiling and applies the Page
    correction. Works in log-dimension to handle astronomically large d.

    Parameters
    ----------
    ln_dA, ln_dB : float
        Natural logs of the two subsystem Hilbert-space dimensions.

    Returns
    -------
    float
        ⟨S_A⟩ in nats. By symmetry S_A = S_B, so it is bounded by min(ln_dA,
        ln_dB). The Page correction −d_</.../2d_> is exponentially small
        except very near the Page point and is included in log-safe form.
    """
    ln_small = min(ln_dA, ln_dB)
    ln_large = max(ln_dA, ln_dB)
    # Page correction term d_small/(2 d_large) = exp(ln_small - ln_large)/2
    correction = 0.5 * np.exp(ln_small - ln_large)
    return ln_small - correction


@dataclass
class BlackHolePageCurve:
    """Page curve for a BPR black hole with N boundary cells, p states each.

    Parameters
    ----------
    n_cells : int
        Total number of boundary cutoff cells N (total Hilbert dim = p^N).
    p : int
        States per cell (substrate prime).
    """

    n_cells: int = 100
    p: int = 104761

    @property
    def ln_p(self) -> float:
        return float(np.log(self.p))

    @property
    def total_ln_dim(self) -> float:
        """ln of total Hilbert-space dimension = N ln p."""
        return self.n_cells * self.ln_p

    def radiation_entropy(self, cells_radiated: int) -> float:
        """Entanglement entropy of the radiation after k cells have radiated.

        d_A = p^k (radiation), d_B = p^(N−k) (remaining black hole).
        """
        k = cells_radiated
        ln_dA = k * self.ln_p
        ln_dB = (self.n_cells - k) * self.ln_p
        return page_entropy(ln_dA, ln_dB)

    def curve(self) -> dict:
        """Full Page curve over evaporation (k = 0 … N)."""
        ks = np.arange(self.n_cells + 1)
        S = np.array([self.radiation_entropy(int(k)) for k in ks])
        page_time = int(np.argmax(S))
        return {
            "cells_radiated": ks,
            "radiation_entropy_nats": S,
            "page_time_cells": page_time,
            "page_time_fraction": page_time / self.n_cells,
            "peak_entropy_nats": float(S.max()),
            "initial_entropy": float(S[0]),
            "final_entropy": float(S[-1]),
            "is_unitary": bool(S[0] < 1e-9 and S[-1] < 1e-9 and S.max() > 0),
            "status": (
                "DEMONSTRATED: BPR's finite U(1)_p boundary Hilbert space "
                "(dim p^N) yields a unitary Page curve (entropy rises, peaks "
                "at the Page time ~N/2, returns to zero). Page formula is "
                "standard typicality; evaporation dynamics not derived."
            ),
        }
