# A common light cone for several emergent species

2026-09-25. Status: Bogoliubov-level mathematics with an implementation
(`bpr/common_light_cone.py`), tests (`tests/test_common_light_cone.py`) and a
demo (`scripts/demo_common_light_cone.py`). The independent review in section 8
found no blockers; its repairs are applied. The second boson species is a
proposed extension of the supplied cubic lattice. The radiative and
renormalization-group statements in section 5 are cited from the literature,
not derived here.

## 0. Question

The [condensate note](cubic_condensate_regime_2026-09-25.md) gives one scalar
phonon whose low-energy cone is Lorentz-invariant with speed c_s. A physical
world has many fields, and they must share one limiting speed to high
precision. Does anything in this substrate class force that?

## 1. Two species on the cubic lattice

Take species i=a,b with hopping C_i (κ_i=C_i a²), contact couplings g_ij
(symmetric, g_ab the inter-species density coupling) and uniform condensate
densities ν_i. The Bogoliubov problem is governed by the symmetric Hartree
matrix

    G = N^{1/2} g N^{1/2},   G_ij = g_ij sqrt(ν_i ν_j),   N = diag(ν_a, ν_b).

The asymmetric "physical potential" g_ij ν_j has the same spectrum only up to
similarity. It is wrong in the symmetric-block formulas below; a test shows the
difference at unequal densities. The tests check G against an independent
finite-difference Jacobian of the two-species lattice Gross–Pitaevskii
equations, at densities 0.4 and 2.3 and at 1.5 and 0.2.

**Lemma 1 (Hartree minimizer).** If g is positive semidefinite, i.e.
g_ab² <= g_aa g_bb (miscibility, independent of the densities), the uniform
mixture minimizes the two-species Hartree functional.

*Proof.* Each species' kinetic term is minimized only by a constant profile.
The interaction Σ_x ρ_x^T g ρ_x is convex in the local densities ρ_x, so
Jensen's inequality over sites makes the uniform distribution optimal. Because
the kinetic minimizer is unique, the minimizer is unique even when
det g=0. ∎

**Converse, with a finite-lattice caveat.** If G has an eigenvalue m_-<0, the
uniform mixture is dynamically unstable at momenta where ε(k)<2|m_-|. With
equal hopping this requires ε_*=4C sin²(π/n)<2|m_-|. The instability is
therefore certain only on large enough lattices, in particular in the
infinite-lattice limit. For example, G=[[1,1.05],[1.05,1]] is immiscible yet
has no unstable mode on n=16, while n=20 has six.

## 2. Two-branch spectrum and speeds

**Theorem 1.** Let E(k)=diag(ε_a(k),ε_b(k)), with ε_i(k)=2C_iΣ_j(1-cos k_j).
The BdG block at momentum k has A=E+G and B=G. Its positive frequencies satisfy

    ω(k)² ∈ spec( E(k) (E(k) + 2G) ).

With equal hopping (E=ε·I) this becomes e_± = sqrt(ε(ε+2m_±)), where m_± are
the eigenvalues of G.

*Proof.* For real symmetric A and B, the squared eigenvalues of
σ_z[[A,B],[B,A]] are the eigenvalues of (A-B)(A+B)=E(E+2G). With equal
hopping the orthogonal matrix diagonalizing G decouples the two species at
every k. ∎

**Theorem 2 (speeds).** As k→0, E≈|q|²D with D=diag(κ_a,κ_b), so the phonon
speeds are

    c_±² = eigenvalues of S = 2 D^{1/2} G D^{1/2}.

For a stable mixture, a single nonzero cone c_+=c_->0 holds iff S=c²I. That
is equivalent to

    g_ab = 0   and   κ_a g_aa ν_a = κ_b g_bb ν_b.

*Proof:* a symmetric 2×2 matrix with equal eigenvalues is a multiple of the
identity. ∎

## 3. What symmetry does and does not do

**(i) Coupled species never share a cone in this class.** Within the
U(1)×U(1) density-coupled contact class at Bogoliubov level, any inter-species
coupling g_ab≠0 splits the cones for every choice of the other parameters. No
tuning of κ or g_ii restores a common cone.

At the Z2-symmetric point (κ_a=κ_b=κ, equal G_ii=μ) the speeds are
c_±=sqrt(2κ(μ±G_ab)). Each speed moves by a fraction ±G_ab/(2μ), so
(c_+-c_-)/c≈G_ab/μ.

**(ii) Decoupled species.** A common cone requires g_ab=0 and then
κ_a g_aa ν_a=κ_b g_bb ν_b.
- Decoupling is exact and stable: H=H_a+H_b, with each species separately
  translation invariant. That protection comes from the absence of any
  coupling, not from an internal symmetry. No internal symmetry of soft-core
  bosons makes n_a n_b odd, and particle–hole symmetry is absent in the
  condensate regime.
- The speed condition also involves state data, the densities. Exchange
  symmetry of H protects it only if the state is exchange symmetric: equal
  densities, or equal chemical potentials grand-canonically.
- So the only protected common cone is identical, non-interacting copies in
  symmetric states. Those are separate worlds rather than one.

**(iii) SU(2) point.** With g_aa=g_bb=g_ab and equal hopping, G has rank one
(det G=0) for any density ratio. One branch is linear, with speed²
2κ·tr G. The other is the quadratic type-B Goldstone mode ω=ε exactly at
Bogoliubov level. Enlarging the symmetry removes the second cone rather than
aligning it.

**(iv) Beyond this class.** Laser-induced (Rabi) interconversion between the
components allows the limiting speeds to be tuned equal, with one massless and
one massive mode (Visser–Weinfurtner, PRD 72, 044020, 2005;
Liberati–Visser–Weinfurtner, PRL 96, 151301, 2006). Those papers present this
as a naturalness problem: the tuning is not protected. Beyond Bogoliubov
level, entrainment (Andreev–Bashkin drag) makes D non-diagonal, so
"iff g_ab=0" is a Bogoliubov-level statement.

**Conclusion.** A universal cone for interacting species is not natural in
this substrate class. This is a fine-tuning (naturalness) obstruction, not a
no-go: tuned common cones exist, for example decoupled tuned species or
Rabi-coupled species at the tuned point. It is the known analogue-gravity
bimetricity/monometricity problem, made explicit for the BPR cubic lattice.

## 4. Photon versus phonon

Three models are easy to conflate:
- the [condensate note](cubic_condensate_regime_2026-09-25.md) treats
  soft-core site bosons in mean-field scaling;
- the [emergent-gauge note](emergent_gauge_link_bosons_2026-09-25.md) treats
  hard-core bosons on a line graph;
- in the link model the superfluid (small U/t) has its own phonon, not the one
  derived in the condensate note.

Within the link model:
- the photon exists only in the Coulomb phase at large U/t. For the diamond
  lattice this relies on published quantum Monte Carlo; for the cubic lattice
  it is unestablished;
- in the superfluid phase the charges have condensed, the gauge field is
  Higgsed, and the photon and the superfluid phonon do not coexist.

A photon and a phonon can coexist if they come from different bosons, for
example neutral site bosons plus link bosons. That is exactly the multi-species
problem of section 3, with no mechanism aligning the two speeds.

Inside the Coulomb phase the charges are not on the photon cone either. They
hop at first order in t, with group velocity of order t·a, while the photon
speed is of order K_6·a=3t³a/U². The charges are thus faster than light by a
factor of order (U/t)², and would emit Cherenkov photons.

## 5. Beyond Bogoliubov (literature)

**Radiative transmission.** Suppose the limiting speeds were tuned equal at
some scale. In an interacting theory with a preferred frame and an ultraviolet
cutoff, loop corrections generate dimension-4 Lorentz-violating operators,
i.e. differences in limiting speeds, that are not suppressed by E/Λ (Collins,
Perez, Sudarsky, Urrutia, Vucetich, PRL 93, 191301, 2004). In their Yukawa
example the induced coefficient is bounded below by g²/(6π²) for any
regulator.

**Experimental bounds.** Collider data bound isotropic photon–electron speed
differences at the ~10⁻¹¹ level (Hohensee, Lehnert, Phillips, Walsworth,
PRL 102, 170402, 2009):
- the absence of vacuum Cherenkov radiation from 104.5 GeV LEP electrons
  gives one side;
- the stability of 300 GeV Tevatron photons against decay gives the other.

A 2024 LHC analysis has since tightened one side, and astrophysical bounds are
stronger.

**Renormalization-group convergence.** Interactions do drive limiting speeds
together in the infrared (Chadha–Nielsen, NPB 217, 125, 1983). In weakly
coupled theories the approach is only logarithmic, which is too slow to meet
the bounds without tuning (Anber–Donoghue, PRD 83, 105027, 2011). At strong
coupling the approach can be power-law in the ratio of infrared to ultraviolet
scales (Bednik–Pujolàs–Sibiryakov, JHEP 11 (2013) 064).

**Protecting mechanisms.**
- Supersymmetry pushes the lowest Lorentz-violating operators to dimension 5
  (Groot Nibbelink–Pospelov, PRL 94, 081601, 2005). Supersymmetry breaking
  regenerates lower-dimension operators suppressed by powers of m_soft/M. A
  lattice origin would need emergent supersymmetry.
- Sequestering keeps Lorentz violation in a separate sector coupled to the
  Standard Model only through Planck-suppressed operators (Pospelov–Shang,
  PRD 85, 105001, 2012).

## 6. Consequence for the unification program

This is the sharpest structural obstruction found so far for the architecture
"spacetime and all fields emerge from a lattice substrate":
- the condensate supplies one cone;
- an emergent gauge sector supplies a different cone, in a different phase or
  from different bosons;
- a second matter species supplies another cone unless decoupled and tuned;
- loops and slow renormalization-group running undo tree-level tuning unless a
  mechanism intervenes.

The map records common_limiting_speed as **obstructed, in the naturalness
sense, within this class**. Ways forward, each requiring a new principle:
1. A substrate whose entire low-energy content is the excitation spectrum of
   one strongly coupled relativistic fixed point, with power-law (not
   logarithmic) approach to Lorentz invariance.
2. A protecting symmetry, such as emergent supersymmetry, or sequestering of
   the Lorentz-violating sector behind Planck-suppressed couplings.
3. Lorentz invariance as fundamental, with the substrate inside spacetime
   rather than spacetime emerging from it. This changes the program's central
   claim.

## 7. Controls

`bpr/common_light_cone.py` and its tests check:
- exact 4×4 BdG frequencies against Theorem 1 at every mode of an n=8 lattice
  (~1e-14), with stability agreement asserted;
- a site-basis BdG oracle for the Fourier reduction;
- a finite-difference Gross–Pitaevskii Jacobian at unequal densities, for the
  linearization and the symmetric G;
- Theorem 2 against long-wave numerical slopes, on and off the tuned manifold;
- the Z2 splitting and the rank-one SU(2) branch at unequal densities;
- unstable and degenerate mixtures never reporting a common cone;
- the finite-lattice immiscibility caveat (n=16 stable, n=20 unstable).

## 8. Review record

An independent adversarial review (2026-09-25) found **no blockers**. It
verified:
- Lemma 1 (including by numerical minimization), Theorems 1 and 2, the Z2
  formula and the SU(2) branch;
- the Gross–Pitaevskii linearization at unequal densities.

Repairs applied:
- the explicit symmetric G and density-independent miscibility;
- the finite-lattice caveat on the converse;
- the corrected conclusion that coupled species never share a cone in this
  class;
- the state-symmetry condition for protection;
- the rank-one SU(2) statement;
- the Visser–Weinfurtner / Liberati–Visser–Weinfurtner prior art and the
  Rabi-coupled counterpoint;
- the Tevatron mechanism (photon decay);
- the Chadha–Nielsen / Anber–Donoghue / Bednik–Pujolàs–Sibiryakov
  renormalization-group literature;
- the Collins et al. lower bound;
- the model distinctions in section 4 and the Cherenkov remark;
- the fine-tuning (not no-go) label;
- sequestering;
- code fixes so unstable or degenerate cases never report a cone and
  imaginary speeds are not masked as zero;
- new independent tests.

The literature citations were checked by web search for existence and content
summaries, not by reading full texts. This record is AI review, not
certification or empirical validation.
