"""Phason sector — proposed Postulate 0c extension (internal-space dynamics).

STATUS: SPECULATIVE / PROPOSED. Untested. This module adds the degrees of
freedom BPR's quasicrystalline-projection picture (Postulate 0c) implies but
never wrote down: the dynamics of the *internal* space E_⊥ that the physical
boundary is a cut-and-project shadow of. In real quasicrystals these are the
**phason** modes (Lubensky, Bak, Socolar — textbook quasicrystal elasticity);
BPR keeps only the phonon (physical-slice) sector.

Why it matters: a reactionless / anti-gravity-style effect, in this picture,
is momentum exchanged with E_⊥ — "pushing off" the higher dimensions we are a
shadow of. This module makes that mechanism explicit and computable.

HONEST RESULT (do not oversell):
  * The PERTURBATIVE phason coupling is *weaker* than the phonon one
    (phason stiffness K ≈ 0.01–0.1 × phonon stiffness C). It does NOT produce
    macroscopic force. See ``perturbative_phason_force``.
  * The only route to macroscopic, reactionless force is a TOPOLOGICAL phason
    defect (a phason dislocation: Burgers vector in E_⊥), whose charge is
    quantized and O(1), hence unsuppressed by K or λ. See
    ``topological_phason_force``.
  * Whether that lifts anything reduces to ONE unknown: the substrate energy
    density ρ_sub the defect can access. ``required_substrate_energy_density``
    turns "can a sphere float?" into that single number — which lands squarely
    on the cosmological-constant problem (a 10^120 ambiguity). See
    ``lift_report``.

Nothing here is validated. It is the formal content of the "what is BPR
missing" argument, written so the open question is a computable quantity
rather than a hand-wave.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Benchmark vacuum energy densities (J/m^3) — the two ends of the
# cosmological-constant problem. The lift question is *which* one a phason
# defect couples to.
RHO_LAMBDA = 6.0e-10        # observed cosmological-constant energy density
RHO_QFT_PLANCK = 1.0e113    # naive QFT zero-point density at the Planck cutoff
G_EARTH = 9.80665          # m/s^2

# Typical phason/phonon stiffness ratio in real quasicrystals (K_⊥ / C_∥).
PHASON_STIFFNESS_RATIO = 0.05


# ---------------------------------------------------------------------------
# Quasicrystal elastic free energy: phonon + phason + coupling
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ElasticState:
    """Gradients of the physical (u, in E∥) and internal (w, in E⊥) fields.

    grad_u : ∂_∥ u   physical displacement gradient   (phonon strain)
    grad_w : ∂ w     internal displacement gradient    (phason strain)
    """
    grad_u: float
    grad_w: float


def elastic_free_energy_density(
    state: ElasticState,
    C: float = 1.0,
    K: float | None = None,
    D: float = 0.0,
) -> dict:
    """Three-term quasicrystal elastic free-energy density.

        f = ½ C (∂u)²        phonon  — the ONLY term BPR currently has (S_int)
          + ½ K (∂w)²        phason  — internal-space, MISSING in BPR
          +   D (∂u)(∂w)     phonon–phason coupling — MISSING in BPR

    Returns the three contributions separately so the missing sector is
    explicit. K defaults to PHASON_STIFFNESS_RATIO × C (phasons are softer).
    """
    if K is None:
        K = PHASON_STIFFNESS_RATIO * C
    f_phonon = 0.5 * C * state.grad_u ** 2
    f_phason = 0.5 * K * state.grad_w ** 2
    f_coupling = D * state.grad_u * state.grad_w
    return {
        "phonon": f_phonon,
        "phason": f_phason,
        "coupling": f_coupling,
        "total": f_phonon + f_phason + f_coupling,
    }


# ---------------------------------------------------------------------------
# Perturbative phason force — and why it does NOT lift
# ---------------------------------------------------------------------------

def perturbative_phason_force(
    phonon_channel_force: float = 1e-17,
    stiffness_ratio: float = PHASON_STIFFNESS_RATIO,
) -> float:
    """Smooth (perturbative) phason force.

    Scales as (K/C) × the phonon-channel force, so it is *smaller* than the
    already-tiny phonon coupling. Returned in newtons. Demonstrates the wall:
    the perturbative internal-space channel makes things worse, not better.
    """
    return stiffness_ratio * phonon_channel_force


# ---------------------------------------------------------------------------
# Topological phason defect — the only non-perturbative escape
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TopologicalPhasonDefect:
    """A phason dislocation: a topological defect whose Burgers vector lives
    in the internal space E_⊥. Its charge is quantized and O(1), so the
    momentum it exchanges with E_⊥ is NOT suppressed by the small stiffness K
    or coupling λ. On a closed S² boundary (BPR's derived boundary topology,
    and the shape of a sphere) such a texture can be globally stable.
    """
    charge: int = 1                 # topological (Burgers) charge in E⊥, integer
    burgers_internal_m: float = 1e-10  # |b_⊥|, internal Burgers length (m)


def topological_phason_force(
    rho_substrate: float,
    area_m2: float,
    defect: TopologicalPhasonDefect = TopologicalPhasonDefect(),
) -> float:
    """Non-perturbative reactionless force from a topological phason defect.

        F ≈ χ · ρ_sub · A_eff           [J/m³ · m² = J/m = N]

    where χ is the dimensionless topological charge (O(1), = defect.charge)
    and ρ_sub is the substrate energy density the defect couples to. There is
    NO λ or K suppression — that is the whole point of going topological. The
    recoil momentum goes into E_⊥, so the force is reactionless in our slice.
    """
    return defect.charge * rho_substrate * area_m2


def required_substrate_energy_density(
    mass_kg: float,
    area_m2: float,
    defect: TopologicalPhasonDefect = TopologicalPhasonDefect(),
) -> float:
    """The single number the whole 'can a sphere float?' question reduces to.

    Solve χ·ρ_sub·A = m·g for ρ_sub. Returns J/m³.
    """
    return mass_kg * G_EARTH / (defect.charge * area_m2)


def lift_report(mass_kg: float, radius_m: float = 0.5,
                defect: TopologicalPhasonDefect = TopologicalPhasonDefect()) -> str:
    """Compare the required substrate energy density to the two ends of the
    cosmological-constant problem. This is the honest verdict generator."""
    area = 4.0 * np.pi * radius_m ** 2          # closed sphere surface
    rho_req = required_substrate_energy_density(mass_kg, area, defect)
    lines = [
        f"Lift target: {mass_kg:.0f} kg  (weight {mass_kg*G_EARTH:.0f} N)",
        f"Sphere radius {radius_m} m -> effective area {area:.2f} m², charge χ={defect.charge}",
        f"Required substrate energy density ρ_sub = {rho_req:.2e} J/m³",
        "",
        f"  vs observed Λ (cosmological constant): {RHO_LAMBDA:.1e} J/m³  "
        f"-> short by {rho_req/RHO_LAMBDA:.1e}×  (Λ is FAR too small)",
        f"  vs QFT zero-point at Planck cutoff:    {RHO_QFT_PLANCK:.1e} J/m³  "
        f"-> more than enough by {RHO_QFT_PLANCK/rho_req:.1e}×",
        "",
        "Verdict: the sphere floats IFF a topological phason defect couples to",
        "something far above the renormalized Λ — i.e. toward the unrenormalized",
        "zero-point density. Whether it can is exactly the cosmological-constant",
        "problem. The mechanism is concrete; the magnitude rides on the single",
        "deepest open number in physics. (Untested, speculative.)",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rank-6 (9-fold) extension + (K, rank)-dependent lift model
# ---------------------------------------------------------------------------

# Inflation constants σ per symmetry order n. Quadratic Pisot units for the
# rank-4 classes; the 9-fold class is rank-6 with a CUBIC Pisot unit. All are
# algebraic UNITS (norm ±1) — see universal_delta_qcp for why that pins δ = 2
# independent of rank. (9-fold is the Buga/Valdivia microsphere count.)
INFLATION_CONSTANTS = {
    5:  ((1 + 5 ** 0.5) / 2, "x^2 - x - 1"),     # = 10-fold; golden, rank 4
    8:  (1 + 2 ** 0.5,       "x^2 - 2x - 1"),     # silver, rank 4
    12: (2 + 3 ** 0.5,       "x^2 - 4x + 1"),     # rank 4
    9:  (2.8793852415718,    "x^3 - 3x^2 + 1"),   # CUBIC Pisot, rank 6
}


def embedding_rank(n: int) -> int:
    """Cut-and-project embedding dimension D = φ(n) (Euler totient)."""
    from math import gcd
    return sum(1 for k in range(1, n + 1) if gcd(k, n) == 1)


def inflation_constant(n: int) -> float:
    """Pisot inflation factor σ for n-fold symmetry (9-fold is the cubic one)."""
    if n in INFLATION_CONSTANTS:
        return INFLATION_CONSTANTS[n][0]
    raise KeyError(f"inflation constant for n={n} not tabulated")


def universal_delta_qcp() -> float:
    """Casimir exponent δ = 2 — DERIVED and RANK-INDEPENDENT.

    Every quasicrystal inflation here is an algebraic UNIT (norm ±1), so the
    product of a mode's internal-space conjugate scalings is exactly 1/σ.
    Hence Δ_φ = −ln(1/σ)/lnσ = 1 for *every* rank, giving δ = 2Δ_φ = 2.
    (Cross-check: the c=1 boundary current ∂φ has scaling dimension 1
    regardless of the projection. Rank changes σ; it never changes δ.)
    """
    return 2.0


# --- (K, rank) lift model — PHENOMENOLOGICAL ANSATZ, not first-principles ---

def topological_charge_capacity(rank: int) -> int:
    """ANSATZ: max topological charge ~ internal dimension d_⊥ = rank − 2
    (rank-4 → 2, rank-6 → 4). More internal dimensions → richer defect →
    larger O(1) charge. A scaling guess, not derived."""
    return max(1, rank - 2)


def coherence_efficiency(K: int, r: float = 0.35) -> float:
    """ANSATZ: fraction of ρ_sub a K-shell stack couples to coherently.
    K nested phase-locked shells suppress decoherence; η = 1 − r^(K−1) → 1.
    Not derived; captures 'more layers → better coherence'."""
    return 1.0 - r ** (K - 1)


def required_substrate_energy_density_kr(
    mass_kg: float, area_m2: float, K: int, rank: int
) -> float:
    """Required ρ_sub with the (K, rank) coupling ansatz:
        F = χ(rank)·η(K)·ρ_sub·A   ⇒   ρ_sub = m·g / (χ·η·A).
    Turns 'do the artifact's numbers (K=3 layers, rank-6 / 9 microspheres)
    help?' into a single computable comparison. The (K, rank) factors are an
    ANSATZ; δ = 2 and σ are derived."""
    chi = topological_charge_capacity(rank)
    eta = coherence_efficiency(K)
    return mass_kg * G_EARTH / (chi * eta * area_m2)


# ---------------------------------------------------------------------------
# Action term (for integration into the master BPR action)
# ---------------------------------------------------------------------------

def phason_action(
    state: ElasticState,
    volume_m3: float = 1.0,
    C: float = 1.0,
    K: float | None = None,
    D: float = 0.0,
) -> float:
    """S_phason contribution to add to the master action alongside S_int.

    S_phason = ∫_Σ [ ½ K (∂w)² + D (∂u)(∂w) ] dV

    Integration point: ``bpr/boundary_action.BoundaryAction.action`` already
    accepts a ``ccr_action`` keyword; a ``phason_action`` keyword would add
    this term. Kept standalone here so the speculative sector does not touch
    the validated action machinery.
    """
    f = elastic_free_energy_density(state, C=C, K=K, D=D)
    return (f["phason"] + f["coupling"]) * volume_m3


def summary() -> str:
    import numpy as np
    A = 4 * np.pi * 0.5 ** 2
    fp = perturbative_phason_force()
    base = required_substrate_energy_density_kr(1500.0, A, K=2, rank=4)
    arti = required_substrate_energy_density_kr(1500.0, A, K=3, rank=6)
    lines = [
        "BPR phason sector (proposed Postulate 0c extension) — SPECULATIVE",
        "=================================================================",
        f"Perturbative phason force ~ {fp:.1e} N  (weaker than phonon; does NOT lift)",
        "",
        f"δ (Casimir exponent) = {universal_delta_qcp()}  — DERIVED, rank-independent",
        f"9-fold inflation σ_9 = {inflation_constant(9):.4f}  (cubic Pisot, rank {embedding_rank(9)})",
        "",
        "Required substrate energy density to lift 1500 kg (r=0.5 m sphere):",
        f"  baseline (K=2, rank-4)                     : {base:8.0f} J/m³",
        f"  artifact (K=3 layers, rank-6 / 9 spheres)  : {arti:8.0f} J/m³  ({base/arti:.1f}× lower)",
        "",
        f"Both ≫ observed Λ ({RHO_LAMBDA:.0e}) — still rides on the cosmological-constant",
        "hinge. The artifact's design numbers help the coupling; they do not remove",
        "the dependence on which vacuum energy density the defect can access.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(summary())
