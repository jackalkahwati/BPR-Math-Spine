# Emergent U(1) gauge structure from link bosons (proposed amendment)

2026-09-25. Status: new conditional mathematics with an implementation
(`bpr/link_boson_gauge.py`), independent tests
(`tests/test_link_boson_gauge.py`) and a demo
(`scripts/demo_link_boson_gauge.py`). The independent review in section 10
found no blockers; its repairs are applied. **This is a proposed model
amendment, adopted at the user's request to keep developing the theory. It is
not a consequence of the existing site-boson substrate.** It establishes an
emergent Abelian gauge structure, not the Standard Model gauge group, and it
relies on published numerics for the existence of a photon phase.

## 0. Why

The unification map's gauge layer had no route from any substrate to a gauge
field. For the one-dimensional Bose-ring Hamiltonian, the
[local-symmetry theorem](substrate_local_symmetry_2026-09-13.md) excludes exact
bounded short-interval Gauss operators. That note is explicit that it is not a
general no-go theorem. The standard way around exact local conservation is
energetic: the Gauss law is not an exact symmetry, but an energy cost makes it
hold in the low-energy sector. This note builds that mechanism for bosons and
derives its effective theory exactly to third order in the hopping.

The foundation decision asked for "a genuinely three-dimensional local bosonic
graph Hamiltonian with conserved total number". The model below is in that
class (section 1).

## 1. Model

Let Λ be a finite simple graph, with vertex degrees d_v and integer targets
q_v. Hard-core bosons n_ℓ∈{0,1} live on its links ℓ. The vertex charge is
Q_v=Σ_{ℓ∋v} n_ℓ, and

    H = U Σ_v (Q_v - q_v)²  -  t Σ_v Σ_{ℓ≠ℓ'∋v} b_ℓ† b_ℓ',     U>0.

Total number N=Σq_v/2 is conserved. On every bipartite connected component,
the targets must sum equally over the two sublattices, because each link has
one end on each. The implementation enforces this component by component.

Equivalently, this is hard-core bosons on the line graph L(Λ) with
nearest-neighbour hopping t and nearest-neighbour repulsion V=2U. The
remaining one-body term is a link-dependent potential -2U(q_a+q_b)n_ℓ; it
becomes a chemical potential when the targets are uniform. For Λ the diamond
lattice, L(Λ) is the pyrochlore lattice.

## 2. Exact facts at finite t

(a) **The Gauss law is not exact.** A hop via v from (v,b) to (v,a) changes
Q_a by +1 and Q_b by -1. Consequently [H,Q_a]≠0 whenever a has a neighbour
of degree at least 2 and such a hop exists in the sector. That holds at every
vertex of the lattices considered here. At a vertex whose neighbours all have
degree 1, for example the centre of a star, the commutator vanishes. This is
consistent with, not a counterexample to, the ring theorem.

(b) **The ice manifold is gapped at t=0.** At t=0 the ground space is P_0, the
set of configurations with Q_v=q_v at every vertex. Every other N-particle
configuration has Σ_v(Q_v-q_v)=0 with at least two nonzero terms, so its
energy is at least 2U.

## 3. Second order (any simple graph)

For a 4-cycle □ whose occupations alternate, let F_□ map the pattern that
starts occupied on its first link to the complementary pattern. F_□† is the
reverse map; this is U_□ in section 5. The term -K(F_□+F_□†) contributes -K
once between each pair of complementary configurations.

**Theorem 1.** On the range of P_0,

    H_eff^(2) = -(t²/2U) Σ_v q_v e_v  -  K Σ_{4-cycles □} (F_□ + F_□†),
    K = 2t²/U,   e_v = d_v - q_v.

*Proof.* In a simple graph the two far endpoints of any hop are distinct, so
every hop leaves P_0 and P_0VP_0=0. Every intermediate state is a single ±1
dipole of energy exactly 2U, which gives H^(2)=-(1/2U)P_0V²P_0.
- **Diagonal part.** A hop and its reverse. There are q_v e_v such pairs at
  vertex v.
- **Off-diagonal part.** A second hop that undoes the dipole must pass through
  a common neighbour of its two ends. Unless it is the reverse hop, this closes
  an alternating 4-cycle.
- **Counting.** Each alternating 4-cycle has two pivot pairs and two
  orderings. That is four paths, each with amplitude -t²/2U, giving -2t²/U.

Hard-core bosons on different links commute, so no exchange signs appear. ∎

## 4. Third order (triangle-free graphs)

**Convention.** With P_0VP_0=0, the third-order Schrieffer–Wolff effective
Hamiltonian, PVRVRVP-½{PVR²VP,PVP}, and Bloch's, PVRVRVP-PVR²V·PVP, both
reduce to

    H^(3) = P_0 V R V R V P_0,    R = -Q H_0^{-1} Q.

(des Cloizeaux's agrees as well.) The conventions first differ at fourth
order, and then only by a similarity transformation.

**Theorem 2.** For triangle-free Λ, on P_0:

    H_eff^(3) = -(t³/4U²) Σ_v q_v e_v (d_v-2)
                - Σ_□ δK_□ (F_□+F_□†)
                - K_6 Σ_{6-cycles ⬡} (F_⬡+F_⬡†),

where δK_□ = (3t³/4U²) Σ_{corners c of □} (d_c-2), K_6 = 3t³/U², and F_⬡ is
defined for alternating 6-cycles as in section 3.

*Proof.* A hop through pivot p moves one unit of charge between two neighbours
x and y of p. Three moves that return to P_0 carry zero net charge, so they
form a directed 3-cycle a→b→c→a of charge transfers. Any two of the moves then
compose to a single ±1 dipole, so every intermediate energy is exactly 2U, and
each path contributes -t³/4U². Triangle-freeness keeps each pivot outside
{a,b,c}. Three cases remain:
- **All pivots equal (a star).** Two single-vertex patterns: one boson touring
  two empty links, q e(e-1) sequences; and a push through an occupied link,
  q(q-1)e sequences. Together that is q e(d-2) per vertex, a constant on P_0.
- **Two pivots equal.** The moves span a 4-cycle v-a-w-c, with the third
  charge point b∈N(v)∖{a,c}. That gives d_v-2 choices; b≠w by
  triangle-freeness. These are the "detour" and "push" replacements of one
  plaquette move. Each has three orderings of the single move relative to the
  two-step move. Summing over both pivot pairs gives δK_□.
- **All pivots distinct.** A simple 6-cycle with two pivot triples, each with
  3! orderings, gives K_6. Chords do not contribute, because hops occur only
  between consecutive cycle links. 5-cycles never enter. ∎

## 5. Gauge interpretation on bipartite graphs

Orient every link from sublattice A to sublattice B, and put
E_ℓ = n_ℓ - 1/2 ∈ {±1/2}. The Gauss operator is G_v = ε_v(Q_v - d_v/2), with
ε=±1 on A/B. The constraint Q_v=q_v is Gauss's law with static background
charge ε_v(q_v - d_v/2); when q_v=d_v/2 there is none.

In the spin-½ link language b_ℓ†↔S⁺_ℓ. The loop operators F_□ and F_⬡ are the
plaquette operators U_□=Π S^± of the spin-½ U(1) quantum link model (Horn
1981; Orland–Rohrlich 1990; Chandrasekharan–Wiese 1997). S⁺ is not unitary;
the phase e^{iA} is the rotor-model analogue. So H_eff is a compact U(1)
lattice gauge theory in quantum-link form.

Gauge invariance holds for H_eff at every order by construction: every G_v is
constant on P_0, and each term flips closed alternating loops. The conserved
microscopic operators are the dressed e^S G_v e^{-S}. The perturbative series
is expected to be only asymptotic in the thermodynamic limit.

**Electric charges** are the vertex defects Q_v≠q_v. Each costs U, they move at
first order in t, and they are bosons.

## 6. Rokhsar–Kivelson family

Add a supplied potential λ Σ_□ P_flippable. At λ=K the second-order model is,
up to the constant, 2K Σ_□ |ψ_□^-><ψ_□^-|, where ψ^- is the normalized
antisymmetric combination of the two alternating patterns. The single-square
spectrum is {0,2K}. The model is frustration-free: a zero-energy state must
have equal amplitude on any two configurations related by a flip. Each
flip-connected class therefore has exactly one zero-energy ground state, its
equal-amplitude superposition. The implementation verifies this on every
cluster.

Near the RK point, single-mode and field-theory arguments give a photon whose
speed vanishes as λ→K⁻ (Moessner–Sondhi 2003; Hermele–Fisher–Balents 2004).
Those arguments are not proofs.

## 7. The two three-dimensional lattices

**Cubic lattice.** Here q=3 and d=6.
- The effective couplings are K=2t²/U+12t³/U², K_6=3t³/U², and a constant
  -(9/2)t²/U-9t³/U² per vertex. The independent review verified these on
  cubic tori with n=4 and n=6 (648 links).
- The third-order diagonal term is constant, so no RK-type potential appears
  through third order and λ/K=O(t²/U²). The third-order corrections are
  larger: δK/K≈6t/U and K_6/K≈1.5t/U. The model is near the pure-kinetic
  cubic spin-½ quantum link model, with those corrections.
- Whether that point is in the deconfined Coulomb phase is **not
  established**. The literature points found for this review report Coulomb
  phases for the cubic lattice mainly near the RK point or for larger link
  spin.

**Diamond lattice.** Here q=2 and d=4.
- There are no 4-cycles, so second order is a constant (-2t²/U per vertex).
- Dynamics starts at third order with hexagon ring exchange K_6=3t³/U². This
  matches Hermele–Fisher–Balents' 12J_±³/J_z² under J_z=2U, J_±=t. The
  third-order constant is -2t³/U² per vertex.
- The full model is hard-core bosons on the pyrochlore lattice with
  unfrustrated nearest-neighbour hopping (our -t with t>0, i.e. K_6>0) and
  repulsion V=2U at half filling.
- Published results:
  - **Banerjee, Isakov, Damle, Kim** (PRL 100, 047208, 2008), quantum Monte
    Carlo on this model: a superfluid at small V/t; at large V/t an insulator
    with no broken translation symmetry, whose correlations match the Coulomb
    phase (a U(1) fractionalized Mott insulator).
  - **Shannon, Sikora, Pollmann, Penc, Fulde** (PRL 108, 067204, 2012),
    zero-temperature quantum Monte Carlo on the effective ring-exchange model:
    a quantum-ice liquid described by the Maxwell action of 3+1D quantum
    electrodynamics, including the pure ring-exchange point, with a competing
    ordered state elsewhere.
  - **Kato–Onoda** (PRL 115, 077202, 2015), quantum Monte Carlo on the
    spin-½ XXZ form with ferromagnetic transverse exchange (the same sign
    class): a crossover into a bosonic quantum Coulomb liquid.
- **The diamond version is reported by this literature to realize an emergent
  3+1D compact-QED (Coulomb) phase at large U/t.** Nothing here re-derives
  those Monte Carlo results. The review could not re-check the exact parameter
  ranges in the full texts.

## 8. Consequences for the unification program

1. **A gauge field from bosons, in principle.** An ordinary conserved-number
   bosonic lattice model has a regime whose low-energy theory, according to
   the cited numerics, is compact U(1) gauge theory: a photon plus gapped
   bosonic charges and monopoles. The gauge layer moves from "open" to
   "conditional (U(1) only)".
2. **Photon and superfluid phonon, same bosons.** For the diamond model the
   numerics show a superfluid at small U/t and the Coulomb liquid at large U/t.
   In the superfluid the U(1) number symmetry is broken. Every number-changing
   operator changes some Q_v, so breaking it means condensing charged objects,
   which Higgses the gauge field. That is a heuristic, not a theorem.
   - The superfluid's gapless mode is the Goldstone mode of these link bosons,
     not the Bogoliubov phonon derived for the soft-core site bosons in the
     [condensate note](cubic_condensate_regime_2026-09-25.md).
   - For the cubic line graph, a superfluid at small U/t is plausible but not
     shown.
   - Within this single-species model, photon and superfluid phonon therefore
     occupy different phases. A two-species substrate, e.g. neutral site
     bosons plus link bosons, can have both; that is the multi-species
     light-cone problem of the
     [common-light-cone note](common_light_cone_2026-09-25.md).
3. **Speed of light.** In the Coulomb phase the photon speed is
   c=α·K_6·a/ħ, with α an O(1) number known only numerically; c vanishes at
   the RK point. It is unrelated to other species' speeds. The charges hop at
   O(t), so they are faster than the photon by a factor of order (U/t)².
4. **Still missing.**
   - A non-Abelian group: quantum link models for SU(N) use fermionic rishons,
     and no bosonic substrate derivation of SU(3)×SU(2) exists here.
   - Fermionic charges.
   - Chiral matter.
   - A dynamical metric.

## 9. Exact and numerical controls

`bpr/link_boson_gauge.py` and its tests check four clusters:
- the non-bipartite 3×3 square torus (18 links, sector dimension 48,620);
- an open cube (12 links);
- two open cubes (20 links, 125,970);
- adamantane, a diamond fragment with four hexagons.

The checks are:
- The analytic H^(2) equals the numerical P_0VRVP_0 to ~1e-15 relative on
  every cluster.
- The analytic H^(3) equals P_0VRVRVP_0 on the three triangle-free clusters,
  also against an independent sparse Fock-space oracle for the 12-link ones.
  On the torus, triangles give configuration-dependent third-order diagonal
  terms, as expected; the formula is not claimed there.
- Full exact diagonalization: the second-order error scales as t³ (ratio ≈8
  when t doubles). The third-order error scales as t⁴ (ratio ≈16) on the
  triangle-free clusters, where the analytic H^(3) is claimed.
- A Richardson extrapolation of the second-order spectrum.
- The Gauss-commutator statement, including the star exception.
- Validation of targets per connected component.
- The RK ground states.

These are finite algebraic checks. They do not establish a phase in the
thermodynamic limit.

## 10. Review record

An independent adversarial review (2026-09-25) found **no blockers**.

What the reviewer checked, with code that does not import this module:
- brute-force P_0VRVP_0 and P_0VRVRVP_0 on many graphs, with exact agreement
  to Theorems 1 and 2:
  - complete bipartite graphs K3,3, K3,4 and K4,4;
  - the Petersen graph;
  - honeycomb clusters and mixed-degree graphs;
  - random bipartite and triangle-free graphs;
  - the open 2×2×2 block of cubes;
  - cubic tori n=4 and n=6;
  - a 128-link diamond torus;
- that every returning third-order path has intermediate energies (2U,2U);
- a spectral check on K4,4;
- the Banerjee et al. mapping, V=2U and the hopping sign.

Repairs applied:
- the third-order proof, via directed charge 3-cycles, with the
  effective-Hamiltonian convention stated;
- the physics consequences reworded as phase-diagram and Higgs-heuristic
  statements, with the model distinctions;
- the Shannon et al. description corrected and Kato–Onoda added;
- the third-order couplings in code metadata and the demo;
- the Gauss-commutator statement and the link-dependent potential;
- per-component validation and empty-manifold errors;
- the F_□ definition and quantum-link wording;
- the RK normalization and frustration-free uniqueness;
- the size of the cubic corrections;
- the photon-speed form;
- Richardson input guards and tests;
- the ring-theorem scope.

The literature characterizations were checked by web search for existence and
abstract-level content, not by reading full texts. This record is AI review,
not certification or empirical validation.
