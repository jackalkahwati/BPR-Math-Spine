"""
Theory IX: Bioelectric Substrate Coupling
==========================================

Extends BPR into biology: morphogenetic fields as boundary phase patterns,
cellular memory as local winding, aging as boundary phase decoherence,
and evolution as topological configuration-space exploration.

Key objects
-----------
* ``MorphogeneticField``    – φ_morph satisfying bio-BPR equation
* ``CellularWinding``       – local W encoding differentiation state
* ``AgingModel``            – coherence decay τ_coh(age)
* ``EvolutionTopology``     – speciation as winding-sector bifurcation

References: Al-Kahwati (2026), *Ten Adjacent Theories*, §11
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# §11.2  Morphogenetic phase field
# ---------------------------------------------------------------------------

@dataclass
class MorphogeneticField:
    """Morphogenetic field φ_morph satisfying the bio-BPR equation:

        κ_bio ∇² φ_morph(x) = Σ_cells χ_cell δ(x − x_cell) + V'(φ_morph)

    where:
        κ_bio  = biological boundary stiffness (gap-junction coupling)
        χ_cell = cellular source charge = α_bio V_mem
        V      = developmental potential landscape
    """
    kappa_bio: float = 1.0        # gap-junction coupling strength
    alpha_bio: float = 1.0        # bioelectric-to-phase coupling
    n_cells: int = 100
    V_mem: Optional[np.ndarray] = None       # membrane potentials (mV)
    positions: Optional[np.ndarray] = None   # cell positions (n_cells, dim)

    def cell_sources(self) -> np.ndarray:
        """χ_cell = α_bio × V_mem for each cell."""
        if self.V_mem is None:
            return self.alpha_bio * np.zeros(self.n_cells)
        return self.alpha_bio * self.V_mem

    def solve_1d(self, L: float = 1.0, n_grid: int = 256) -> np.ndarray:
        """Solve the morphogenetic BPR equation on a 1-D domain [0, L].

        Uses spectral method (FFT) with periodic boundary conditions.
        """
        dx = L / n_grid
        x = np.linspace(0, L, n_grid, endpoint=False)

        # Build source on grid
        source = np.zeros(n_grid)
        chi = self.cell_sources()
        if self.positions is not None:
            for i in range(min(self.n_cells, len(chi))):
                idx = int(self.positions[i, 0] / dx) % n_grid
                source[idx] += chi[i]
        else:
            # Uniform distribution
            cell_indices = np.linspace(0, n_grid - 1, self.n_cells, dtype=int)
            for i, idx in enumerate(cell_indices):
                source[idx] += chi[i]

        # Spectral Laplacian solve: κ k² φ̂ = ŝource
        k = np.fft.fftfreq(n_grid, d=dx)
        k2 = (2.0 * np.pi * k) ** 2
        source_hat = np.fft.fft(source)
        phi_hat = np.zeros(n_grid, dtype=complex)
        for i in range(n_grid):
            if k2[i] > 1e-12:
                phi_hat[i] = source_hat[i] / (self.kappa_bio * k2[i])
        return np.real(np.fft.ifft(phi_hat))


# ---------------------------------------------------------------------------
# §11.3  Cellular winding and differentiation
# ---------------------------------------------------------------------------

@dataclass
class CellularWinding:
    """Cellular winding number W_cell encoding differentiation state.

    Conservation under division:
        W_parent = W_daughter1 + W_daughter2

    Cancer: W_cancer ≠ W_normal for the tissue context.
    """
    W: float = 1.0     # winding number
    cell_type: str = "normal"

    def divide(self, split_ratio: float = 0.5) -> tuple["CellularWinding", "CellularWinding"]:
        """Symmetric or asymmetric division conserving total winding."""
        W1 = self.W * split_ratio
        W2 = self.W * (1.0 - split_ratio)
        return (
            CellularWinding(W=W1, cell_type=self.cell_type),
            CellularWinding(W=W2, cell_type=self.cell_type),
        )

    def is_aberrant(self, W_normal: float, tol: float = 0.1) -> bool:
        """Check if winding deviates from tissue-normal value (cancer indicator)."""
        return abs(self.W - W_normal) > tol

    @staticmethod
    def bioelectric_reset(cell: "CellularWinding", W_target: float) -> "CellularWinding":
        """Topological repair: reset winding to normal tissue value.

        Models bioelectric normalisation of cancer cells
        (experimentally demonstrated by Levin's group).
        """
        return CellularWinding(W=W_target, cell_type=cell.cell_type)


# ---------------------------------------------------------------------------
# §11.4  Aging as boundary phase decoherence
# ---------------------------------------------------------------------------

@dataclass
class AgingModel:
    """Biological aging as accumulated boundary phase decoherence.

    τ_coh(age) = τ₀ exp(−age / τ_aging)

    As coherence decreases, morphogenetic precision degrades.

    Parameters
    ----------
    tau_0 : float
        Initial coherence time at birth (s or years, consistent units).
    tau_aging : float
        Characteristic aging timescale (same units).
    """
    tau_0: float = 100.0     # years
    tau_aging: float = 30.0  # years

    def coherence_time(self, age: float | np.ndarray) -> float | np.ndarray:
        """Boundary coherence time at a given biological age."""
        return self.tau_0 * np.exp(-np.asarray(age) / self.tau_aging)

    def morphogenetic_precision(self, age: float | np.ndarray) -> float | np.ndarray:
        """Relative morphogenetic field precision (1.0 at birth)."""
        return self.coherence_time(age) / self.tau_0

    def intervention_effect(self, age: float, enhancement_factor: float) -> float:
        """Effect of coherence-enhancing intervention.

        Improved gap-junction coupling / bioelectric normalisation →
        effective coherence time increased by *enhancement_factor*.
        """
        return self.coherence_time(age) * enhancement_factor


# ---------------------------------------------------------------------------
# §11.5  Evolution as topological configuration-space exploration
# ---------------------------------------------------------------------------

@dataclass
class EvolutionTopology:
    """Evolution as exploration of viable winding configuration space.

    Speciation = topological bifurcation (winding sector splits).
    Convergent evolution = different genome regions mapping to same W
    (topological equivalence).
    """
    winding_configs: Optional[np.ndarray] = None  # (n_species, n_traits)

    def speciation_distance(self, species_a: int, species_b: int) -> float:
        """Topological distance between two species in winding space."""
        if self.winding_configs is None:
            raise ValueError("Winding configuration matrix not set.")
        return float(np.linalg.norm(
            self.winding_configs[species_a] - self.winding_configs[species_b]
        ))

    def is_convergent(self, species_a: int, species_b: int,
                      tol: float = 0.1) -> bool:
        """Check if two species are topologically equivalent (convergent evolution)."""
        return self.speciation_distance(species_a, species_b) < tol

    @staticmethod
    def bifurcation_from_ancestor(W_ancestor: np.ndarray,
                                   delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Model speciation as a symmetric bifurcation in winding space."""
        return W_ancestor + delta, W_ancestor - delta


# ---------------------------------------------------------------------------
# §11.6  Aging reversal biomarkers  (Prediction 17)
# ---------------------------------------------------------------------------

@dataclass
class AgingReversalPrediction:
    """Quantitative predictions for aging reversal interventions.

    Any intervention that boosts gap-junction coupling (κ_bio) or reduces
    impedance mismatch (ΔZ) should manifest as:

    1. Improved wound-healing rate  ∝  morphogenetic_precision
    2. Increased tissue-pattern fidelity  ∝  coherence_time
    3. Enhanced regeneration capacity  ∝  κ_bio_effective

    Parameters
    ----------
    age : float        – biological age (years)
    tau_0 : float      – initial coherence time
    tau_aging : float   – aging timescale
    enhancement : float – fold-increase in gap-junction coupling (≥ 1)
    """
    age: float = 60.0
    tau_0: float = 100.0
    tau_aging: float = 30.0
    enhancement: float = 2.0

    @property
    def baseline_coherence(self) -> float:
        """Coherence without intervention."""
        return self.tau_0 * np.exp(-self.age / self.tau_aging)

    @property
    def enhanced_coherence(self) -> float:
        """Coherence with intervention."""
        return self.baseline_coherence * self.enhancement

    @property
    def effective_rejuvenation_years(self) -> float:
        """How many years of aging the intervention reverses.

        Solve: τ₀ exp(−age_eff / τ_aging) = enhanced_coherence
        → age_eff = −τ_aging ln(enhanced_coherence / τ₀)
        → rejuvenation = age − age_eff
        """
        ratio = self.enhanced_coherence / self.tau_0
        if ratio <= 0 or ratio > 1:
            return 0.0
        age_eff = -self.tau_aging * np.log(ratio)
        return max(0.0, self.age - age_eff)

    @property
    def wound_healing_improvement(self) -> float:
        """Fold-improvement in wound healing rate."""
        return self.enhanced_coherence / self.baseline_coherence

    @property
    def regeneration_index(self) -> float:
        """Regeneration capacity index (0–1 scale, 1 = embryonic)."""
        return min(1.0, self.enhanced_coherence / self.tau_0)


# ---------------------------------------------------------------------------
# §11.7  Convergent evolution distance  (Prediction 18)
# ---------------------------------------------------------------------------

def convergent_evolution_similarity(W_species_a: np.ndarray,
                                     W_species_b: np.ndarray) -> float:
    """Topological similarity between two species.

    Phenotypic similarity should correlate with this number more
    strongly than with genome similarity.

    similarity = 1 − ||W_a − W_b|| / (||W_a|| + ||W_b||)

    Returns 1.0 for identical winding configs, 0.0 for maximally different.
    """
    W_a = np.asarray(W_species_a, dtype=float)
    W_b = np.asarray(W_species_b, dtype=float)
    diff = np.linalg.norm(W_a - W_b)
    total = np.linalg.norm(W_a) + np.linalg.norm(W_b)
    if total < 1e-30:
        return 1.0
    return 1.0 - diff / total
