# Emergent U(1) gauge structure from link bosons (proposed amendment)

2026-09-25. Status: new conditional mathematics with an implementation
(`bpr/link_boson_gauge.py`), independent tests
(`tests/test_link_boson_gauge.py`) and a demo
(`scripts/demo_link_boson_gauge.py`). Independent review is recorded in
section 10. **This is a proposed model amendment, adopted at the user's request
to keep developing the theory. It is not a consequence of the existing
site-boson substrate.** It establishes an emergent Abelian gauge structure, not
the Standard Model gauge group, and it relies on published numerics for the
existence of a photon phase.

## 0. Why

The unification map's gauge layer had no route from any substrate to a gauge
field. The ring theorem ([local symmetry](substrate_local_symmetry_2026-09-13.md))
excludes exact bounded short-interval Gauss operators on unrestricted Bose Fock
space. The known way around this is energetic: the Gauss law is not an exact
symmetry, but an energy cost makes it hold in the low-energy sector. This note
builds that mechanism for bosons and derives its effective theory exactly to
third order in the hopping.

The foundation decision asked for "a genuinely three-dimensional local bosonic
graph Hamiltonian with conserved total number". The model below is in that
class. Up to a constant and a chemical potential at fixed N, it is an ordinary
extended Bose–Hubbard model on a line graph (section 1).

## 1. Model

Let Λ be a finite simple graph, with vertex degrees d_v and integer targets
q_v. Hard-core bosons n_ℓ∈{0,1} live on its links ℓ. The vertex charge is
Q_v=Σ_{ℓ∋v} n_ℓ, and

    H = U Σ_v (Q_v - q_v)²  -  t Σ_v Σ_{ℓ≠ℓ'∋v} b_ℓ† b_ℓ',     U>0.

Total number N=Σq_v/2 is conserved. The model has no local occupation cutoff
beyond the hard-core condition and no gauge field put in by hand. On a
bipartite graph every link has one end on each sublattice, so the targets must
sum equally over the two sublattices; the implementation enforces this.

Equivalently, this is hard-core bosons on the line graph L(Λ), with
nearest-neighbour hopping t. With uniform q and d, and at fixed N,
U Σ_v (Q_v-q)² = 2U Σ_{⟨ℓℓ'⟩} n_ℓ n_ℓ' + const, a nearest-neighbour repulsion
V=2U. For Λ the diamond lattice, L(Λ) is the pyrochlore lattice.

## 2. Exact facts at finite t

(a) **The Gauss law is not exact.** A hop via v from ℓ'=(v,b) to ℓ=(v,a)
changes Q_a by +1 and Q_b by -1, so [H,Q_v]≠0 for t≠0. This is consistent
with, not a counterexample to, the ring obstruction.

(b) **The ice manifold is gapped at t=0.** At t=0 the ground space is P_0, the
set of configurations with Q_v=q_v at every vertex. Every other N-particle
configuration has Σ_v(Q_v-q_v)=0 with at least two nonzero terms, so its
energy is at least 2U.

## 3. Second order (any simple graph)

**Theorem 1.** On the range of P_0,

    H_eff^(2) = -(t²/2U) Σ_v q_v e_v  -  K Σ_{4-cycles □} (F_□ + F_□†),
    K = 2t²/U,   e_v = d_v - q_v,

where F_□ flips an alternating 4-cycle (occupied/empty/occupied/empty) to its
complement.

*Proof.* In a simple graph every hop leaves P_0, so P_0VP_0=0. Every
intermediate state has energy exactly 2U, which gives
H^(2)=-(1/2U)P_0V²P_0.
- **Diagonal part.** A hop and its reverse. At vertex v there are q_v e_v such
  pairs, a constant on P_0.
- **Off-diagonal part.** Two distinct hops that restore every charge. The
  second hop must move a boson from a link at a to a link at b. Those links
  share a vertex w≠v, so v-a-w-b is a 4-cycle, alternating in occupation.
- **Counting.** Each alternating 4-cycle has two pivot pairs, {v,w} and {a,b},
  and two time orderings of the hops. That is four paths, each with amplitude
  t²·(-1/2U), giving K=2t²/U.

For hard-core bosons on different links there are no exchange signs. ∎

## 4. Third order (triangle-free graphs)

Assume Λ has no triangles; bipartite graphs, including the cubic and diamond
lattices, qualify.

**Theorem 2.** With P_0VP_0=0, H^(3)=P_0VRVRVP_0, where R=-QH_0^{-1}Q. On P_0,

    H_eff^(3) = -(t³/4U²) Σ_v q_v e_v (d_v-2)
                - Σ_□ δK_□ (F_□+F_□†)
                - K_6 Σ_{6-cycles ⬡} (F_⬡+F_⬡†),

where δK_□ = (3t³/4U²) Σ_{corners c of □} (d_c-2), K_6 = 3t³/U², and F_⬡ flips
an alternating 6-cycle.

*Proof.* Three hops that return to P_0 are pairwise-adjacent link moves. In a
triangle-free graph, three pairwise-adjacent links meet at one vertex.
- **Diagonal part.** There are two single-vertex patterns: one boson touring
  two empty links, q e(e-1) sequences; and a push through an occupied link,
  q(q-1)e sequences. Together they give q e(d-2) per vertex.
- **4-cycle part.** One of the two moves of a pivot pair is replaced by a
  two-step move through one of the d-2 other links at its pivot: a detour
  through an empty link or a push through an occupied one. The single move
  can be placed in three positions in the sequence.
- **6-cycle part.** There are two pivot triples, each with 3! orderings.
- **Energies.** In every case each partial sequence leaves a single ±1 dipole,
  so all intermediate energies are 2U and every path contributes -t³/4U².

Summing the counts gives the stated coefficients. ∎

## 5. Gauge interpretation on bipartite graphs

Orient every link from sublattice A to sublattice B, and put
E_ℓ = n_ℓ - 1/2 ∈ {±1/2}. Then:
- The Gauss operator is G_v = ε_v(Q_v - d_v/2), with ε=±1 on A/B. The
  constraint Q_v=q_v is Gauss's law with static background charge
  ε_v(q_v - d_v/2); when q_v=d_v/2 there is no background charge.
- F_□ and F_⬡ are the lattice Wilson-loop operators (U_□ = Π e^{±iA_ℓ}) of the
  spin-1/2 U(1) quantum link model. So H_eff is a compact U(1) lattice gauge
  theory in quantum-link form (Horn 1981; Orland–Rohrlich 1990;
  Chandrasekharan–Wiese 1997).
- Gauge invariance is exact for H_eff at every order: each term maps P_0 to
  P_0 by flipping closed alternating loops. It is only emergent for H, broken
  at energy scale U.
- **Electric charges** are the vertex defects Q_v≠q_v. Each costs U, they
  move at first order in t, and they are bosons.

## 6. Rokhsar–Kivelson family

Add a supplied potential λ Σ_□ P_flippable. At λ=K the second-order model is
K Σ_□ |ψ_□^-><ψ_□^-|, a sum of positive projectors. Each flip-connected sector
then has exactly one zero-energy ground state, the equal-amplitude
superposition of its configurations. The implementation verifies this on every
cluster. Near the RK point, standard arguments (Moessner–Sondhi 2003;
Hermele–Fisher–Balents 2004) give a photon with speed that vanishes as λ→K⁻.
Those are single-mode and field-theory arguments, not proofs.

## 7. The two three-dimensional lattices

**Cubic lattice.** Here q=3 and d=6.
- The effective couplings are K=2t²/U+12t³/U², K_6=3t³/U², and a constant
  -(9/2)t²/U-9t³/U² per vertex.
- An RK-type potential is not generated at orders two or three, because the
  third-order diagonal term is constant. So λ/K=O(t²/U²), which puts the
  model near the pure-kinetic spin-1/2 cubic quantum link model.
- Whether that point is in the deconfined Coulomb phase is **not established**.
  The literature points found for this review report Coulomb phases for the
  cubic lattice mainly near the RK point, or for larger link spin. The
  pure-kinetic S=1/2 point remains uncertain.

**Diamond lattice.** Here q=2 and d=4.
- There are no 4-cycles, so second order is a constant (-2t²/U per vertex).
- Dynamics starts at third order with hexagon ring exchange K_6=3t³/U². The
  third-order constant is -2t³/U² per vertex.
- This is exactly hard-core bosons on the pyrochlore lattice with
  nearest-neighbour hopping and nearest-neighbour repulsion V=2U. For that
  model, quantum Monte Carlo (Banerjee, Isakov, Damle, Kim, PRL 100, 047208,
  2008) found a superfluid at small V/t, and at large V/t an insulator with no
  broken translation symmetry whose correlations match the Coulomb phase.
  Related work on the XXZ form (Shannon et al. 2012) reports the same U(1)
  liquid with an emergent photon.
- **So the diamond version realizes emergent 3+1D compact QED at large U/t,
  according to that literature.** Nothing here re-derives the Monte Carlo
  result.

## 8. Consequences for the unification program

1. **A gauge field from bosons, in principle.** An ordinary conserved-number
   bosonic lattice model has a regime whose low-energy theory is compact U(1)
   gauge theory, with a photon (diamond lattice, per the Monte Carlo) and
   gapped bosonic charges and monopoles. The gauge layer moves from "open" to
   "conditional (U(1) only)".
2. **Phonon and photon exclude each other in one model.** Small U/t is a
   superfluid: the charges have condensed, the gauge field is Higgsed, and the
   gapless mode is the phonon of the
   [condensate note](cubic_condensate_regime_2026-09-25.md). Large U/t is the
   Coulomb liquid: the photon is gapless and the charges are gapped. The
   acoustic cone and the photon cone come from different phases of the same
   bosons. So the condensate result cannot supply the light cone of a world
   that also contains this photon.
3. **Speed of light.** In the Coulomb phase the photon speed is set by K_6·a
   (diamond). It is unrelated to any other species' speed. The common-cone
   problem ([common light cone](common_light_cone_2026-09-25.md)) applies
   directly.
4. **Still missing.**
   - A non-Abelian group: quantum link models for SU(N) use fermionic rishons,
     and no bosonic substrate derivation of SU(3)×SU(2) exists here.
   - Fermionic charges.
   - Chiral matter.
   - A dynamical metric.

## 9. Exact and numerical controls

`bpr/link_boson_gauge.py` checks the following on four clusters:
- the non-bipartite 3×3 square torus (18 links, sector dimension 48,620);
- an open cube (12 links);
- two open cubes (20 links, 125,970);
- adamantane, a diamond fragment with four hexagons.

The checks are:
- The analytic H^(2) equals the numerical P_0VRVP_0 to ~1e-15 relative on
  every cluster.
- The analytic H^(3) equals the numerical P_0VRVRVP_0 to ~1e-15 relative on
  the three triangle-free clusters. On the torus, triangles give
  configuration-dependent third-order diagonal terms, as expected; the formula
  is not claimed there.
- Full exact diagonalization against the effective models: the second-order
  error scales as t³ (ratio ≈8 when t doubles), and the third-order error as t⁴
  (ratio ≈16).
- [H,Q_v]≠0 at t>0.
- At the RK point, the number of zero-energy states equals the number of
  flip-connected sectors, and the equal superpositions are exact.

These are finite algebraic checks. They do not establish a phase in the
thermodynamic limit.

## 10. Review record

See the end of this file once the independent review is complete.
