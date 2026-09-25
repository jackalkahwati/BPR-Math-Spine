# A common light cone for several emergent species

2026-09-25. Status: Bogoliubov-level mathematics with an implementation
(`bpr/common_light_cone.py`), tests (`tests/test_common_light_cone.py`) and a
demo (`scripts/demo_common_light_cone.py`). Independent review is recorded in
section 8. The second boson species is a proposed extension of the supplied
cubic lattice. The radiative argument in section 5 is cited from the
literature, not derived here.

## 0. Question

The [condensate note](cubic_condensate_regime_2026-09-25.md) gives one scalar
phonon whose low-energy cone is Lorentz-invariant with speed c_s. A physical
world has many fields, and they must share one limiting speed to high
precision. Does anything in this substrate class force that?

## 1. Two species on the cubic lattice

Take species i=a,b with hopping C_i (κ_i=C_i a²) and uniform condensates of
densities ν_i. In the mean-field scaling of the condensate note put
G_ij=μ_ij, the Hartree potential matrix, with μ_ab the inter-species term.

**Lemma 1 (Hartree minimizer).** If G is positive semidefinite (the
miscibility condition μ_ab² <= μ_aa μ_bb), the uniform mixture minimizes the
two-species Hartree functional. The kinetic terms are separately minimized by
uniform profiles, and the interaction Σ_x ρ_x^T G ρ_x is convex in the local
densities; Jensen's inequality over sites does the rest.

If G has a negative eigenvalue, the uniform mixture is dynamically unstable at
long wavelength (Theorem 1 with ε→0). The implementation finds unstable modes
on n=8 for an immiscible example.

## 2. Two-branch spectrum

**Theorem 1.** Let E(k)=diag(ε_a(k),ε_b(k)), with ε_i(k)=2C_iΣ_j(1-cos k_j).
The Bogoliubov–de Gennes block at momentum k has A=E+G and B=G. Its positive
frequencies satisfy

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

A single cone c_+=c_- holds iff S∝I, which is equivalent to

    μ_ab = 0   and   κ_a μ_aa = κ_b μ_bb.

This is a codimension-two condition. *Proof:* a symmetric 2×2 matrix with
equal eigenvalues is a multiple of the identity. ∎

## 3. Symmetry does not make one cone natural

- **(a) Coupled, symmetric species.** With Z2 exchange symmetry (κ_a=κ_b=κ,
  μ_aa=μ_bb=μ) and coupling μ_ab≠0, the speeds are c_±=sqrt(2κ(μ±μ_ab)).
  They split at first order: each speed moves by a fraction ±μ_ab/(2μ), so
  (c_+-c_-)/c ≈ μ_ab/μ with c=sqrt(2κμ). U(1)×U(1)×Z2 allows the
  inter-species density coupling, so nothing forbids μ_ab.
- **(b) SU(2)-symmetric couplings.** With μ_aa=μ_bb=μ_ab and equal densities,
  G has eigenvalues 2μ and 0. One branch is linear, and the other is the
  quadratic type-B Goldstone mode ω=ε. Enlarging the symmetry removes the
  second cone entirely rather than aligning it.
- **(c) Decoupled species.** If μ_ab=0 the full Hamiltonian is a sum of two
  independent systems, so μ_ab stays zero exactly. Equal speeds then hold only
  under the extra tuning κ_aμ_aa=κ_bμ_bb. That is protected only when the two
  species are identical, non-interacting copies, which are separate worlds
  rather than one.

**Conclusion at Bogoliubov level.** In this substrate class, interacting
species share one light cone only under tuning that no symmetry protects.
This is the analogue-gravity "bimetricity" problem made explicit for the BPR
cubic lattice.

## 4. Photon versus phonon

The [emergent-gauge note](emergent_gauge_link_bosons_2026-09-25.md) derives a
compact U(1) gauge sector from link bosons. Its photon exists only in the
Coulomb phase at large U/t. The condensate phonon exists in the superfluid
phase at small U/t, where the charges have condensed and the gauge field is
Higgsed. They do not coexist for the same bosons.

In the Coulomb phase the photon speed scales as K_6·a, and the gapped charges
are nonrelativistic lattice particles. So even the emergent photon and its own
charges do not share a cone.

## 5. Beyond Bogoliubov: radiative transmission (literature)

Suppose the limiting speeds were tuned equal at some scale. In an interacting
theory with a preferred frame and an ultraviolet cutoff, loop corrections
generate dimension-4 Lorentz-violating operators, i.e. differences in
limiting speeds. Their coefficients are of order coupling²/(16π²), and they are
not suppressed by powers of E/Λ (Collins, Perez, Sudarsky, Urrutia, Vucetich,
PRL 93, 191301, 2004). The absence of vacuum Cherenkov radiation at LEP and
the Tevatron already bounds isotropic photon–electron speed differences at
the ~10⁻¹¹ level, and astrophysical bounds are stronger.

A lattice substrate with a preferred frame therefore needs its species speeds
tuned to at least that precision, pair by pair and order by order, unless a
principle enforces equality. Candidate principles discussed in the literature
include:
- supersymmetry, reported to forbid dimension-4 Lorentz-violating operators
  (Groot Nibbelink–Pospelov 2005);
- all low-energy fields arising from a single relativistic critical point.

Neither is realized in this repository.

## 6. Consequence for the unification program

This is the sharpest structural obstruction found so far for the BPR
architecture, "spacetime and all fields emerge from a lattice substrate":

- the condensate supplies one cone;
- the emergent gauge sector supplies a different cone in a different phase;
- a second matter species supplies a third cone unless tuned;
- radiative corrections undo the tuning.

The map records common_limiting_speed as **obstructed within this class at
Bogoliubov level, with a literature radiative argument**. Three ways forward
exist, and each requires a new principle rather than more of the same
construction:
1. Find a substrate whose entire low-energy content is the excitation spectrum
   of ONE relativistic fixed point, and prove its approach to Lorentz
   invariance is fast enough.
2. Supply a symmetry that forbids dimension-4 Lorentz violation.
3. Treat Lorentz invariance as fundamental: put the substrate inside
   spacetime instead of deriving spacetime from it.

The third changes the program's central claim.

## 7. Controls

`bpr/common_light_cone.py` checks:
- exact 4×4 BdG frequencies against Theorem 1 at every mode of an n=8 lattice
  (agreement ~1e-14) for six cases: decoupled identical, decoupled tuned with
  unequal hopping, Z2-coupled, SU(2), generic and immiscible;
- Theorem 2 speeds against long-wave numerical slopes;
- the Z2 splitting formula;
- the quadratic SU(2) branch;
- dynamical instability of the immiscible case.

## 8. Review record

See the end of this file once the independent review is complete.
