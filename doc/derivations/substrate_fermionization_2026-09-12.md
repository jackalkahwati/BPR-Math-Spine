# A conditional fermionic dictionary for the bosonic ring

**2026-09-12. One-dimensional hard-core mapping and finite-repulsion certificates,
not a derivation of physical fermions.**

The [nonlinear-response calculation](substrate_nonlinear_response_2026-09-12.md)
is classical. Here we test a quantum statistics bridge rather than identifying
classical Fourier modes with the independently stipulated flavor fermions.
The construction uses the existing bosonic hopping and onsite interaction;
no pairing term, background charge, chemical potential or new coupling is added.

## 1. What is inherited, and what is extended

`bpr/zp_selection_principle.py:_small_hilbert` implements three bosonic sites
with local occupations 0,1,2 and Hamiltonian

\[
H=-C\sum_{x=0}^{L-1}(a_x^\dagger a_{x+1}+a_{x+1}^\dagger a_x)
+\frac g2\sum_x n_x(n_x-1),\qquad N=\sum_x n_x,
\tag{1}
\]

with periodic sites and L=3. We **extend that quantum prescription** to a small
L-site ring, L>=3, C>0, g>=0, at fixed integer 0<=N<=L. It is not claimed that
classical DNLS uniquely derives this quantization. The normal ordering n(n-1)
is inherited, not replaced by n-squared. The unshifted free dispersion is
-2C cos(k); the shifted classical response convention differs by 2CN in a
fixed-number quantum block.

The complete fixed-N basis consists of nonnegative integer tuples summing to N:

\[
\dim\mathcal H_N=\binom{L+N-1}{N}.
\]

There is no additional local occupancy cutoff. The historical three-site toy
is complete only in N<=2. Its N=3 sector has dimension seven, whereas the
complete sector has dimension ten, including (3,0,0) and permutations. These
models must not be silently equated at N>=3. Finite local annihilation matrices
also do not obey exact canonical bosonic commutators on their top cutoff state.

Site count L and neutrality modulus q are independent. The old
`global_shift_invariance(n=101,p=104761)` already distinguishes them.
Here q is an integer >=2; primality is unnecessary for the counting identities.

## 2. Hard-core compression and an exact string dictionary

Write H=gD+V with \(D=\frac12\sum_x n_x(n_x-1)\). Let P project onto
no-double-occupancy configurations, Q=I-P. Then

\[
P=\ker D,\quad d_P=\binom LN,\quad D_Q\ge I_Q,
\qquad A=PHP=PVP.
\]

The compressed hopping is exact as a matrix, but P need not be invariant under
full finite-g evolution. In the full binary occupation space introduce b_x,
which lowers 1 to 0. Operators at different sites commute; onsite
\(b_x^2=0\), \(\{b_x,b_x^\dagger\}=I\). Define

\[
Z_x=1-2n_x,\qquad c_x=\left(\prod_{y<x}Z_y\right)b_x.
\tag{2}
\]

The strings imply \(\{c_x,c_y\}=0\) and
\(\{c_x,c_y^\dagger\}=\delta_{xy}\). Also
\(c_x^\dagger c_x=n_x\). These CAR identities must be tested on the full binary
space: an annihilator leaves a fixed-N block, so restricting it there first
would invalidate the test.

Equation (2) is nonlocal in the bosonic site variables. For example, c_x for
x>y depends on Z_y and does not commute with all operators on site y. This is
not a local bosonic operator acquiring fermionic exchange statistics in three
spatial dimensions. It is an exact one-dimensional change of variables.

### Ring boundary conditions are essential

Interior hopping obeys \(b_x^\dagger b_{x+1}=c_x^\dagger c_{x+1}\). The ring
boundary link contains the parity string. In a fixed-N sector its hopping
coefficient is multiplied by

\[
\tau_N=(-1)^{N-1}:
\quad A=-C\left[\sum_{x=0}^{L-2}(c_x^\dagger c_{x+1}+\mathrm{h.c.})
+\tau_N(c_{L-1}^\dagger c_0+\mathrm{h.c.})\right].\tag{3}
\]

Thus odd N uses periodic fermions and even N antiperiodic fermions. At N=0 the
boundary sign is immaterial because there is no hopping. Allowed one-particle
momenta are 2pi m/L for tau=+1 and (2m+1)pi/L for tau=-1; fixed-N free-fermion
energies are sums of N distinct levels -2C cos(k). One untwisted periodic
Hamiltonian cannot represent every number sector correctly.

Basis order is explicit: the existing three-site bosonic tensor index is
9n0+3n1+n2, whereas the existing CAR implementation uses sum 2^x n_x.
Matrix comparisons require a permutation to common occupation labels.

## 3. A finite-repulsion validity certificate

All norms below are spectral/operator 2-norms, or vector Euclidean norms.
The one-particle ring hopping has norm at most 2C; hence at fixed N

\[
\|V\|\le v=2CN.\tag{4}
\]

Assume Q is nonempty and g>2v. The unperturbed gD has d_P zero eigenvalues,
with all remaining eigenvalues >=g. Weyl's inequality applied to gD+V gives
**exactly d_P low eigenvalues in [-v,v]**, with all remaining eigenvalues
at least g-v>v. This is a conservative sufficient separation condition, not
a necessary phase boundary. Failing it does not prove the mapping useless.

Define B=PHQ and K=QHQ. For real |E|<=v,

\[
K-E\ge(g-v-|E|)I_Q\ge(g-2v)I_Q.
\]

A low eigenvector psi satisfies

\[
Q\psi=-(K-E)^{-1}B^\dagger P\psi,
\qquad \boxed{\|Q\psi\|\le\frac{\|B\|}{g-2v}\|P\psi\|}.\tag{5}
\]

Its retained part solves \(S(E)P\psi=EP\psi\), where

\[
S(E)=A-B(K-E)^{-1}B^\dagger,
\qquad \boxed{\|S(E)-A\|\le\frac{\|B\|^2}{g-2v}}.\tag{6}
\]

This Schur operator is energy dependent. It is not an independently specified
autonomous Hamiltonian, an all-time approximation or a Schrieffer-Wolff unitary
transformation. Equations (5) and (6) concern low eigenstates and the resolvent.

The second-order virtual correction supplied by the *same* Hamiltonian is

\[
\Delta H_2=-B(gD_Q)^{-1}B^\dagger,\qquad H_2=A+\Delta H_2.\tag{7}
\]

Since K=gD_Q+QVQ, the resolvent identity gives

\[
\|(K-E)^{-1}-(gD_Q)^{-1}\|
\le\frac{v+|E|}{g(g-v-|E|)}.
\]

Consequently

\[
\boxed{\|S(E)-H_2\|\le\frac{2v\|B\|^2}{g(g-2v)}}
\quad(|E|\le v).\tag{8}
\]

This certifies the Schur operator approximation, not ordered H2 eigenvalue
errors without an additional theorem. Numerical eigenvalue differences can be
reported separately as diagnostics. The bounds are not rigorous floating-point
roundoff guarantees. For cancellation-resistant evaluation, the exact remainder
can be computed directly as

\[
S(E)-H_2=B(gD_Q)^{-1}(QVQ-EI_Q)(K-E)^{-1}B^\dagger,
\]

rather than subtracting two nearly equal corrections. This is the same
resolvent identity used in the proof, not an additional approximation.

Vacuum and N=1 have Q empty and exact compression at every g. They must not
produce zero-denominator placeholders. If Q is nonempty and g=0, (7) is
undefined; if 0<g<=2v it is algebraically defined but not certified by (8).

## 4. Explicit virtual processes, not new adjustable interactions

For L=5,N=2, use tuples in site order 0,...,4. The adjacent pair |11000>
can hop into |20000> or |02000>. Each process has matrix element -sqrt(2)C
and each intermediate state has D=1. Therefore

\[
\langle11000|\Delta H_2|11000\rangle=-4C^2/g.\tag{9}
\]

The transition |11000> to |01100> has one common excluded intermediate
state, |02000>, giving

\[
\langle01100|\Delta H_2|11000\rangle=-2C^2/g.\tag{10}
\]

Paths through retained states belong to repeated dynamics under A, not to
this excluded-state correction. The diagonal pair term and correlated transfer
show why finite-repulsion fermionic variables are generally **interacting**:
the strict hard-core free-fermion result does not extend unchanged to finite g.

For L=N=3, P has only |111> and A=0, but hopping connects it to six distinct
doublon-hole states. Each has D=1 and amplitude -sqrt(2)C, so

\[
\langle111|\Delta H_2|111\rangle=-12C^2/g.\tag{11}
\]

A trivial hard-core block therefore does not imply absent finite-g virtual
physics. The old occupancy-two cutoff contains these particular intermediate
states, but omits triple occupancy from the exact full finite-g spectrum.

## 5. Neutrality is not supplied by the string

The inherited neutrality assumption declares exp(2pi i N/q)=1 on physical
states. It is not implied by number conservation or by CAR. Under that
assumption, strict hard-core sector dimensions are

\[
d_N^{\mathrm{neutral}}=\mathbf1_{N\equiv0\pmod q}\binom LN,
\qquad0\le N\le L.\tag{12}
\]

- If L=q, only N=0,L survive, each dimension one. Fixed-number projected
  dynamics and density are then trivial.
- If q>L, only vacuum survives.
- If L>q, intermediate neutral sectors may exist. Choosing such a geometry
  is an additional physical choice, not a consequence of this calculation.

The strings commute with total N, so
\([N,c_x^\dagger]=c_x^\dagger\). A Jordan-Wigner fermion still carries charge
one and fails the stipulated neutrality condition for q>=2. Neutral bilinears
are distinct operators. Neither neutral eligibility nor sector counting proves
binding, dynamical confinement, composite statistics or a selected population.
At finite repulsion, doublons and larger N sectors exist; (12) is not a counting
theorem for the full unrestricted bosonic Hilbert space.

## 6. Missing physics and legacy claim boundary

This construction does **not** derive spacetime spin, relativistic chirality,
a three-dimensional exchange law, the monopole source geometry, three families,
physical masses, mixing or gravity. The internal CAR flavor model remains a
separate ansatz. A controlled one-dimensional dictionary is a useful test of
conventions and limits, not completion of the microscopic statistics bridge.

The legacy `bpr/topological_matter.py:185–240` does not furnish that missing
bridge: `AnyonStatistics` assigns a phenomenological exchange phase pi W/p,
but its particle classifier calls W=p bosonic despite phase pi, and calls
W/p=1/2 fermionic despite phase pi/2. Its scalar braiding matrices
also do not establish the advertised non-Abelian behavior. Those inconsistencies
are recorded here, not silently imported as evidence or repaired in this pass.
No historical closed particle sector is reopened by the current calculation.

## 7. Frozen demonstration and verification

The executable demonstration freezes:

- L=5,N=2,C=1,g=40 with q=5. The mathematical hard-core comparison is
  nonneutral under the chosen q. Here v=4 and g-2v=32>0.
- The same L,N,C at g=0.7: the sufficient strong-repulsion certificate fails;
  this is labeled uncertified, not disproved.
- L=3,N=q=3,C=1,g=40: a neutral one-dimensional hard-core block with
  finite-g correction -12/40=-0.3 and a complete ten-state bosonic sector.

These are illustrative mathematical inputs, not measured parameters or selected
physical sectors. No fitting or parameter scan is performed. Physical masses
and mixing remain null.

- `bpr/substrate_fermionization.py`: bounded fixed-number matrices, exact string
  dictionary, neutrality counting and finite-repulsion certificates.
- `tests/test_substrate_fermionization.py`: independent tensor and signed-bit
  oracles, Schur checks, coefficient witnesses and numerical regressions.
- `python3 scripts/demo_substrate_fermionization.py`: stdout demonstration;
  add `--json` for strict structured output.

The implementation caps site count at 12, the complete fixed-N dimension at
512, and the full binary space used for CAR tests at 256. These are resource
limits, not physical cutoffs, and are checked before allocation. Certificate status
must distinguish structural exact cases, insufficient analytic separation, and
numerically unresolved positive margins. Small C²/g corrections are reported
separately from rounded total matrices or eigenvalues. Arithmetic resolution
is not empirical evidence or a substitute for an analytic approximation bound.
The combined numerical report can reject an extreme scale even when the analytic
certificate remains valid: a full-matrix eigensolver must resolve both the low
hopping and virtual-correction scales against its much larger doublon energies.
Such rejection is not failure of the hard-core theorem or permission to present
unresolved eigenvalues as low-energy predictions. The low-level analytic
certificate and separately evaluated virtual correction remain distinct from
that combined eigensystem report. For nonempty Q, the report rejects when its
heuristic eigensolver scale 64 eps ||H|| reaches 1e-6 times the smaller of
v and ||Delta H2|| (where the correction is defined). This conservative
resolution gate is not a rigorous eigensolver-error theorem.

Independent mathematical review found no substantive errors. Full binary CAR,
charge commutators, interior hopping and parity-twisted fixed-N matrices gave
zero residual in independent finite constructions. For L=5, distinct-momentum
energy sums matched every hard-core sector within 1.34e-15. Virtual coefficient
witnesses agreed within 5.56e-17 at C=1,g=40; independent neutrality counting
agreed exactly.

For the frozen L=5,N=2 case, the largest low-state leakage ratio was about
0.061019 against a bound 0.088388. The Schur shift norm was at most 0.213792
against 0.25, and its second-order remainder at most 0.015194 against 0.05.
For L=N=3, the corresponding observed/bound pairs were 0.092960/0.123718,
0.321957/0.428571 and 0.021957/0.128571. Excluded-component reconstruction
residuals were below 3.0e-16 and Schur eigen-equation residuals below 1.63e-14.
These independent-review maxima use the low eigenvalues as E. The stdout
report's separate Schur diagnostic uses E=0, so its remainder need not equal
these maxima. In particular, the L=5,N=2 remainder at E=0 vanishes because
QVQ=0 there, but the general energy-dependent remainder does not vanish.
These are numerical checks of proven bounds, not empirical validation.

The complete L=N=3 ground energy was approximately -0.3219567198, versus
-0.3215302247 for the old occupancy-two cutoff, confirming that equality of
second-order witnesses does not make their exact spectra identical.

The prior bounded regression selection passed **623 tests**, with one intentional
phase-scan deselection and the existing unregistered `slow` marker warning.
Eight algebra-check groups and text/strict-JSON formats of all five previous
demos also passed, invoked from an empty temporary directory without output files.
The new focused suite passed **122 tests with warnings treated as errors**.
The new demo passed text and strict JSON checks from an empty temporary
directory, without stderr or output files. Implementation review identified JSON
scalar conversion and extreme-scale numerical-reporting defects. The fixes
normalize accepted NumPy scalars, reject unresolved full eigensystem reports,
and use the inverse-resolvent identity for small remainder values. Independent
rechecking against an exact-radical, 100-digit oracle recovered the nonzero
L=N=3,E=0 remainders -3.6e-31, -3.6e-35 and -3.6e-39 at g=1e16,1e18,1e20,
with relative errors at most 3.46e-16. Those scales now reject the combined
full-eigensystem report while retaining the separate analytic certificate.
The new identity also agreed with independent moderate-scale subtraction,
including common parameter scales 1e-100 and 1e100, within 6.75e-15 relative
error. Structural zeros and strict JSON with NumPy scalar inputs were checked.
No substantive findings remained after independent rechecking with warnings
as errors.

The final combined bounded selection passed **745 tests**, with one intentional
phase-scan deselection and the existing marker warning. No full-repository test
or empirical-validation claim is made.
No existing quantum or classical module is modified by this calculation.

## Neutral finite-repulsion follow-up

The [neutral density-response calculation](substrate_neutral_response_2026-09-12.md)
examines the complete N=L=q sector rather than restricting it to the single
hard-core state. Its one-doublon/one-hole compression has nontrivial dynamics,
and virtual ground-state admixture couples to a density measurement. Excited-band
and total-weight bounds do not establish physical fermions, a selected vacuum,
or individually certified spectral lines; the frozen total-weight bound does
not resolve the leading signal. The Hamiltonian above is unchanged.

Reproduce the focused checks and demonstration:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m pytest -q -W error tests/test_substrate_fermionization.py
python3 scripts/demo_substrate_fermionization.py
python3 scripts/demo_substrate_fermionization.py --json
```
