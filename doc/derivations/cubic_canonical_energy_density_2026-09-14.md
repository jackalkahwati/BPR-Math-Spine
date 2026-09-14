# Canonical thermodynamic energy density in the dilute interval

2026-09-14. Status: canonical energy-density theorem independently reviewed with
no mathematical blocker; independent saved-document/hash audit completed.
This is an energy-infimum theorem for the specified continuum model, not a
construction of an equilibrium state, a uniform phase or a sound response.

## 1. Statement and established inputs

Retain the same supplied R,kappa>0, finite signed s!=0, contact parameter and
hard exclusion of every triple whose diameter is at most R. Let E(n,L) denote
the bosonic energy infimum of the specified Dirichlet-cube Friedrichs realization.
E(0,L)=0; empty sectors have E=+infinity. The ensemble is zero-temperature
canonical, with the original continuum threshold reference.

The preceding fixed-density note establishes:
- E(n,L)>=-B n, where B=kappa[1_{s>0}/s²+64pi²/R²];
- exact gluing of finitely many admissible box states when their spatial boxes
  have pairwise separation>R, giving an upper bound by the sum of box energies;
- a normalized single-particle Dirichlet sine orbital in side ell has energy
  3pi²kappa/ell², with the requisite form-domain approximation;
- along N_m=m³, L_m=m rho^(-1/3), the explicit upper bound
  E(N_m,L_m)/L_m³<=48pi²kappa rho^(5/3) for0<rho R³<1/8.

The proposed theorem is: for every0<rho<rho_*=1/(8R³), and EVERY sequence
L_j->infinity, N_j integer>=0, N_j/L_j³->rho,

    E(N_j,L_j)/L_j³ -> e(rho),                          (1)

where e is finite, convex and continuous on that open interval, and

    -B rho<=e(rho)<=48pi²kappa rho^(5/3).                (2)

No finite-box minimizer attainment is assumed. No many-particle wavefunction
is spatially cut apart to obtain the lower comparison below.

## 2. A fixed repair cell and eventual admissibility

Fix a packing buffer g=3R/2>R. This g is geometric notation, NOT the bare
contact coupling g_a, and is not a new physical model parameter. Let
h_*=2R, ell_*=R/2, so h_*=ell_*+g. A centered ell_*-cube in an h_*-cell
has margin g/2 on every face. A single particle inside costs

    K=3pi²kappa/ell_*²=12pi²kappa/R².

Any subset of these cells can be occupied, exactly one particle per occupied
inner cube. Different inner cubes are separated by at least g>R; gluing gives
energy at most K times the occupied count. A side-L cube contains floor(L/h_*)³
such cells. For any sequence in(1), this capacity divided by L_j³ tends to
rho_*>rho. Hence eventually N_j<=floor(L_j/h_*)³ and

    -B N_j<=E(N_j,L_j)<=K N_j.                          (3)

Thus eventual nonemptiness is established for arbitrary canonical sequences,
not only a specially rounded exact-density subsequence.

## 3. Define a finite-seed convex envelope before taking any limit

A seed is an admissible pair (n,ell), n>=1, ell>0, with finite E(n,ell).
Associate buffered volume v=(ell+g)³, buffered density d=n/v and buffered
energy density c=E(n,ell)/v. For r>=0 define

    F(r)=inf sum_i w_i c_i,

where the infimum is over finite seed lists and weights w_i>=0 such that
sum_i w_i<=1 and sum_i w_i d_i=r. Unused volume weight is vacuum; the empty
mixture is allowed at r=0. No optimizer or infinite seed distribution is assumed.

Each c_i>=-B d_i, so F(r)>=-Br whenever there is an admissible mixture. The
repair seed (1,ell_*) has density rho_* and energy density K rho_*; mixture
with vacuum gives F(r)<=Kr for0<=r<=rho_*. Therefore F is finite there.
Concatenating near-minimizing finite lists, with weights multiplied by t and
1-t, proves convexity. This operation preserves both total volume weight and
the exact weighted density. Thus F is continuous and locally Lipschitz in
(0,rho_*), by elementary finite convex-function theory. This continuity is
proved before any thermodynamic limit and is not a continuity assumption about E.

## 4. Lower comparison: the large cube itself is a seed

For a canonical sequence in(1), put r_j=N_j/(L_j+g)³. It tends to rho and is
in the open interval eventually. By(3), eventually N_j>=1 and its box has finite
energy. Take that entire box as the single seed with weight1 in the definition
of F. Then

    F(r_j)<=E(N_j,L_j)/(L_j+g)³,
    E(N_j,L_j)/L_j³ >= (1+g/L_j)³ F(r_j).

Continuity of the already defined finite envelope yields

    liminf_j E(N_j,L_j)/L_j³>=F(rho).                   (4)

No lower energy comparison from cutting singular states has been used. The
boundary correction is only the scalar buffered-volume ratio tending to1,
not an unproved surface-energy estimate. Negative values of F cause no problem:
it is finite and continuous near rho, and the multiplying factor tends to1.

## 5. Exact particle-count upper construction

Choose ANY finite mixture at density rho, with cost A=sum_i w_i c_i. Fix a
repair fraction0<t<1. In each large cube of side L_j, split the first coordinate
into slabs of volume fractions alpha_i=(1-t)w_i, one repair slab of fraction t,
and remaining vacuum. Each slab spans the full other two coordinates; zero
weights may be omitted. This is only a trial-state arrangement, not a claimed
macroscopic equilibrium phase.

For seed i let h_i=ell_i+g. Tile its slab by contained h_i-cells and center an
ell_i-seed cube in each. The exact number of seed copies is

    M_(i,j)=floor(alpha_i L_j/h_i) floor(L_j/h_i)².

For fixed finite list and fixed t,
M_(i,j)/L_j³->alpha_i/h_i³. The main-region particle count
P_j=sum_i n_i M_(i,j) therefore obeys P_j/L_j³->(1-t)rho.

The repair slab contains

    C_j=floor(t L_j/h_*) floor(L_j/h_*)²

one-particle cells, with C_j/L_j³->t rho_*. Set the EXACT integer residual
q_j=N_j-P_j. Then q_j/L_j³->t rho. Since0<t rho<t rho_*, eventually
0<=q_j<=C_j. Occupy precisely q_j repair cells. This repairs both floor losses
and arbitrary o(L_j³) canonical count deviations. No particles are removed
from a seed, and no monotonicity of E(n,ell) in n is assumed.

The separation condition holds also BETWEEN slabs with different cell sizes.
Each inner cube is at least g/2 inside its cell, and each cell is contained in
its slab. At an adjacent interface the two inner cubes are each at least g/2
away, giving total gap at least g>R. Unfilled cell remainders only increase it.
The same holds for repair cubes because h_*-ell_*=g. All boxes lie strictly
inside the outer cube with their prescribed margins.

Apply the established finite separated-box gluing lemma:

    E(N_j,L_j)<=sum_i M_(i,j)E(n_i,ell_i)+K q_j.          (5)

Although the number of copies grows with j, each use of the lemma has only
finitely many boxes. Approximate minimizers can be chosen with arbitrarily
small total error for that j before taking the limit; no uniform core-approximation
rate or attained ground states are required.

Divide(5) by L_j³. At fixed mixture and t,

    limsup_j E(N_j,L_j)/L_j³ <=(1-t)A+Kt rho.

Then let t down to0, and finally infimize over finite mixtures at density rho.
This gives

    limsup_j E(N_j,L_j)/L_j³ <=F(rho).                  (6)

The order is essential: finite list and positive reserve first, large box next,
reserve to zero next, and only then optimize the list. No interchange over
unbounded seed sizes or infinite cluster distributions occurs. The strict
condition rho<rho_* is used precisely for repair-capacity slack.

## 6. Identification, convexity and scope

Equations(4),(6) prove(1) with e(rho)=F(rho), independent of the prescribed
canonical sequence. Convexity and continuity come from section3. Stability
gives the lower bound in(2). For its sharper upper bound, apply sequence
independence to the previously established exact-density separated-orbital
sequence, then pass its bound48pi²kappa rho^(5/3) to the limit.

The identity e=F is a variational characterization, NOT an evaluated equation
of state. It neither identifies an optimal seed nor asserts that a finite-seed
mixture attains the thermodynamic infimum. A convex energy density may have
affine segments or nondifferentiable points; the theorem establishes neither
strict curvature nor differentiability, and does not select a morphology.

The theorem concerns continuum Dirichlet cubes at supplied density in the
OPEN interval(0,rho_*). There is no claim at its endpoint, for other boundary
conditions or shapes, or for a joint lattice-spacing/thermodynamic limit.
No extra particle interaction, chemical potential or background phase has been
introduced by the geometric packing convention.

An actual thermodynamic state remains unconstructed. In particular, this does
not prove convergence of approximate minimizers in any local-observable topology,
density retention, stationarity, homogeneity, extremality or connected-correlation
decay. It establishes neither microscopic cluster preference nor macroscopic
phase separation; s>0 pair binding remains compatible with a molecular fluid.
Compressibility, phase stiffness, density response and acoustic poles require
additional state and regularity/dynamical results. No empirical validation or
TOE claim follows from an energy-infimum limit.

## 7. Sources and preservation

Protected sources, relative to doc/derivations/:
- `cubic_fixed_density_phase_gate_2026-09-14.md`: specified finite-box realization,
  extensive stability, single-orbital and exact-density trials, separated-box
  gluing proved using intrinsic singular cores and bosonic assignments.
- `cubic_finite_particle_continuum_2026-09-14.md`: intrinsic singular domains,
  extensive lower bound and exact one-particle identification.

The finite envelope and exact particle-repair construction are the new proof
steps. No generic regular-potential thermodynamic theorem is silently applied
to the singular contact domains. Prior notes' earlier unresolved thermodynamic
status is historical, not retrospectively edited.

Baseline694tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-canonical-energy-density-4nw019d_/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored files
and .git are excluded. Only this note is added. No scientific imports, tests,
numerical evaluations, existing-model changes or publication are authorized.
## 8. Independent mathematical review

Reviewer a9e50e05622e282a4 cleared the theorem with no blocker or necessary repair,
checking the noncircular finite envelope and continuity, large-box seed lower
comparison, cross-slab gaps, exact canonical residual counts, signs with negative
seed energies, order of limits, finite approximate-minimizer errors and singular-
domain bosonic gluing.

Clearance covers a finite, sequence-independent, convex and continuous canonical
energy-infimum limit on continuum Dirichlet cubes for0<rho<1/(8R³), with the
stated bounds. It does not establish a thermodynamic state, phase selection,
differentiability, response or acoustics. The reviewer used direct reads only.
Original proof-draft-v1.md and its hash are preserved; no mathematical constant
or physical hypothesis changed. Independent auditor a59df7e9b62ed5eb4 confirmed
all694baseline files unchanged, exactly one added note and695current files matching
final-hashes.json. The mathematical body is unchanged from the submitted draft;
the note matched pre-audit-note.md. This is artifact/scope clearance, not mathematical
recertification or proof of absence of unrecorded execution. The audited final-hashes.json
is retained; closing-hashes.json binds final audit-status edits.
