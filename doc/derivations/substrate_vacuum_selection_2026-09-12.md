# Neutral population selection in the conditional quantum ring

**2026-09-12. Energetic number-sector selection, not a physical vacuum or a
particle-preparation mechanism.**

The [neutral response calculation](substrate_neutral_response_2026-09-12.md)
chose N=L=q. We now compare that population with all neutral sectors, rather
than treating the previous numerical restriction N<=L as physics.

## 1. Fixed Hamiltonian and scope

Keep the inherited unshifted quantum prescription

\[
H=gD+V,\qquad D=\tfrac12\sum_x n_x(n_x-1),\qquad
V=-C\sum_x(a_x^\dagger a_{x+1}+a_{x+1}^\dagger a_x),
\quad C>0,\ g\ge0.
\]

There are L>=3 periodic sites, and the independently assumed neutrality rule
admits N=kq for integer k>=0, q>=2. Neither q=L nor a chosen population is
universal. No chemical potential, background charge, number-changing term,
or new fitted parameter is introduced. This remains a stipulated quantum
extension of the legacy bosonic model, not a unique derivation from DNLS.

Every fixed-N sector has complete dimension binomial(L+N-1,N). The new
bounded builder admits N>L, with no additional local occupation cutoff.
The old hard-core-oriented builder and its N<=L contract remain unchanged.
The full model has infinitely many number sectors; a numerical matrix cap
cannot select its ground sector.

## 2. An exact lower bound and an infinite-tail exclusion

Write N=La+r with 0<=r<L. A minimum of D has r sites with occupation a+1
and L-r with occupation a, giving

\[
D_{\min}(L,N)=L\frac{a(a-1)}2+ar.
\]

Indeed, transferring a particle from occupation u to v when u>=v+2 reduces
D by u-v-1>0. Iterating gives the balanced configuration.
The one-particle hopping minimum is -2C, so in the N-boson sector

\[
\boxed{E_0(N)\ge\ell_N=gD_{\min}(L,N)-2CN}.
\]

Cauchy-Schwarz also implies D>= (N²/L-N)/2. Therefore for g>0,

\[
E_0(N)\ge\frac g{2L}N^2-\left(\frac g2+2C\right)N,
\qquad N>T:=L(1+4C/g)\ \Longrightarrow\ E_0(N)>0.
\]

The vacuum N=0 is neutral and has exactly zero energy. Thus every neutral
sector above T is excluded from global minimization. **Equality N=T is not
excluded by this argument.** The preliminary neutral list is

\[
N=kq,\qquad 0\le k\le\lfloor T/q\rfloor.
\]

The lower bound diverges with N. Each sector is finite dimensional, hence
for finite L and g>0 the global neutral minimum is attained in finitely many
sectors. This is a mathematical tail theorem, not a finite Fock cutoff.

At g=0, the uniform-mode condensate attains E0(N)=-2CN. Along N=kq this
tends to minus infinity. The unchanged free all-neutral model is
**unbounded below**, despite every fixed-N matrix having a ground state.

## 3. Trial upper bounds and exact comparisons

Two normalized trial states give

\[
u_N^{\rm balanced}=gD_{\min}(L,N),\qquad
u_N^{\rm uniform}=-2CN+\frac g{2L}N(N-1).
\]

The first is an occupation state with zero hopping expectation. The second
has all N bosons in the uniform one-particle mode; its occupation amplitudes
are sqrt(N!/(L^N product_x n_x!)). These are variational upper bounds, not
new interactions or fitted spectra.

At N=L, let Omega=|1,...,1> and let p be one normalized adjacent doublon-hole
configuration. The two-state compression is

\[
\begin{pmatrix}0&-\sqrt2 C\\-\sqrt2 C&g\end{pmatrix}.
\]

For s=C/(g+2C), the normalized trial (Omega+sqrt(2)s p)/sqrt(1+2s²) gives

\[
\boxed{u_L^{\rm pair}=-\frac{2C^2(g+4C)}{(g+2C)^2+2C^2}<0}.
\]

This rational expression suffices for the proof and avoids a numerical
subtraction between nearby square roots.

Take U to be the best trial upper bound from the **neutral** preliminary
sectors only. Exclude sector N precisely when ell_N>U. Equality remains a
candidate. One survivor certifies the minimizing number sector; multiple
survivors do not establish which wins. If unit filling is not neutral,
its trial cannot enter the neutral incumbent U.

All signs, cutoffs and comparisons are evaluated using exact rational values
of the accepted binary64 C and g. This does not assert that measured physical
parameters are exact. In particular, binary64 0.7 is not exactly 7/10.
Approximate display values never participate in a certificate.

## 4. Unit filling is selected at sufficient repulsion

For q=L, every nonzero competing neutral population is N=kL, k>=2. Then

\[
\ell_{kL}=Lk\left[\frac g2(k-1)-2C\right]\ge0
\quad\text{if }g\ge4C.
\]

The empty vacuum has zero energy, whereas the pair trial gives E0(L)<0.
Consequently:

**For every finite L>=3, q=L, C>0 and finite g>=4C, every global neutral
ground state belongs to N=L.** The equality g=4C is included. This theorem
is independent of all site, matrix and enumeration resource caps.

It selects a population, not the product state Omega: V Omega is nonzero,
so Omega itself is not an eigenstate. This is not a claim about observed
particles, a spacetime vacuum, or a thermodynamic limit. Below g=4C, failure
of this sufficient theorem does not itself disprove selection.

## 5. Frozen controls: selected, excluded and unresolved

Use the same illustrative parameters as the preceding calculation.

- **L=q=5,C=1,g=40:** the theorem certifies N=5 over all neutral populations,
  including those above the previous N<=L computational restriction.
- **L=q=5,C=1,g=0.7:** the tail retains 0,5,10,15,20,25,30. The uniform
  N=15 trial has upper bound approximately -15.3, below the N=5 lower
  bound -10. Unit filling is therefore **analytically excluded**. Exact
  lower-bound pruning leaves N=10,15,20; these bounds do not select a winner.
  Their dimensions are 1001,3876,10626, all above the existing dense cap512.
  They remain unresolved, not absent. This does not invalidate within-sector
  neutral-response calculations, but changes their ground-population meaning.
- **g=0, L=q=5,C=1:** the all-neutral energy is unbounded below, with no
  finite minimizing population.
- **L=3,q=4,C=1,g=40:** T=3.3<q; only vacuum survives and is selected.
- A separate **L=3,N=6,C=1,g=0.7** matrix witness has dimension28 and checks
  complete above-unit-filling assembly. It is not itself a global selection
  statement or neutral for every possible q.

No parameter scan or tuning is used to improve these outcomes.

## 6. Energy convention changes cross-sector physics

Within one sector, adding 2CN changes only the overall time-evolution phase.
Across sectors the operator 2C Nhat is not a global constant. In fact,

\[
H'=H+2C\hat N=gD+C\sum_x(a_x-a_{x+1})^\dagger(a_x-a_{x+1})\ge0.
\]

At fixed N, the connected-ring hopping Laplacian has a one-dimensional
kernel: the uniform-mode N-boson condensate. For N>=2 it has
<D>=N(N-1)/(2L)>0 and hence is not in ker D. Two nonnegative operators
can have zero sum expectation only on their common kernel. Thus for g>0
and q>=2 every nonzero neutral sector has strictly positive ground energy:
**the shifted model uniquely selects the empty vacuum.** At g=0 all neutral
uniform condensates tie at zero.

This is a diagnostic of the convention's consequence, not a replacement of H
chosen to obtain a preferred conclusion. Replacing n(n-1) by n² likewise
introduces a number-dependent energy term and is not innocuous here.

## 7. Selection does not prepare a population

Every hopping term preserves total number, and D is a function of occupations.
Thus [H,Nhat]=0, and each sector projector commutes with H. The probability
in each number sector is constant under isolated unitary evolution.
A state prepared in N=0 cannot relax into a preferred N=L sector under this
Hamiltonian. A reservoir, number-changing process or initial-population
postulate would be an additional physical input; none is supplied here.

## 8. Computational and scientific limits

Analytic bounds do not inherit the numerical site cap. Preliminary sector
lists are capped at256 before enumeration; an oversized list is explicitly
resource-unresolved unless an independent theorem settles selection.
No truncated list is presented as exhaustive. Numerical matrices retain
max12 sites and dimension512, with at most four attempted numerical
sector diagnostics per report. Site limits, dimension limits, exhausted
budgets and unresolved numerical arithmetic are distinct from analytic
exclusion. Empty hard-core blocks at N>L are structural matrix blocks, not
an empty full sector or a ground-state certificate.

Exact rational reports preserve numerator/denominator strings. Float summaries
that overflow, underflow to zero, or become subnormal are unavailable rather
than false zeros. Normal-binary64 input and matrix arithmetic restrictions
follow the existing model. Scaled eigensystems provide diagnostics only;
rounded eigenvalue orderings never establish an analytic global winner.
The numerical eigensystem uses H/max(abs(H)) and a heuristic tolerance
128 eps times the matrix dimension times max(1,max(abs(scaled eigenvalues))).
An unresolved ground energy near zero is rejected; levels within that
scale of the ground are labeled an unresolved numerical near-tie, not an
exact multiplicity theorem. These checks do not certify floating-point error.

This strengthens the conditional model analysis, not its empirical status.
Physical fermions, relativistic chirality, selected geometry, observed masses
and mixing, unified gauge/gravity dynamics and a demonstrated TOE remain
underived. The historical failed benchmark is not rerun or reinterpreted.

## 9. Verification

Independent mathematical review found no defects in the bounds, strict tail,
strong selection theorem, shifted convention or conservation claims. Exact
standard-library Fraction calculations reproduced the weak-case incumbent
-68905074298768593/4503599627370496 and surviving populations10,15,20.
Independent implementation review found no actionable defects. Additional
bounded checks covered extreme normal binary64 inputs, 5,001-digit integers,
reduced candidate caps, equality retention, strict JSON and numerical failures
leaving analytic certificates unchanged.

The first focused suite passed105 tests with warnings as errors, including
independent occupation/tensor oracles, trial vectors, all-number completeness,
old-API preservation, exact cutoff/candidate boundaries and number conservation.
The expanded focused suite passed **113 tests with warnings as errors**.
All eight algebra-check groups passed. All eight demos passed text and
strict JSON from an empty temporary directory without stderr or output files.
The final combined bounded regression passed **937 tests**, with one
intentional phase-scan deselection and the existing unregistered `slow` marker
warning. No full-repository test or empirical-validation claim is made.
Previous computational modules remain unchanged. This task is local and
uncommitted on `science/neutral-vacuum-selection`; no remote operations,
benchmark runs, or instruction-file changes were performed.
The separate L=3,N=6 numerical diagnostic gave energy approximately
-8.705936056256364 and within-sector gap3.9855780600379074; these are numerical
checks, not certified cross-sector comparisons or physical particle energies.

Reproduction commands:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m pytest -q -W error tests/test_substrate_vacuum_selection.py
python3 scripts/demo_substrate_vacuum_selection.py
python3 scripts/demo_substrate_vacuum_selection.py --json
```
