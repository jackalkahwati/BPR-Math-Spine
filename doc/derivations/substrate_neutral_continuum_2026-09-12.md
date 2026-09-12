# Large-ring neutral response: a continuum in the leading compression

**2026-09-12. A spectral limit of a conditional model, not a spatial continuum
limit, physical particle spectrum, or completed theory of everything.**

The [finite-ring response calculation](substrate_neutral_response_2026-09-12.md)
produced discrete doublon–hole lines. Here we determine their large-ring limit
without constructing an exponentially large bosonic Fock matrix. The result is
an absolutely continuous spectral measure for the leading compression. It does
not exclude an isolated excitation in the exact interacting Hamiltonian.

## 1. Unchanged Hamiltonian and energy reference

Keep the stipulated quantum extension

\[
H=gD+V,\qquad D=\tfrac12\sum_xn_x(n_x-1),\qquad
V=-C\sum_x(a_x^\dagger a_{x+1}+a_{x+1}^\dagger a_x),
\quad C>0,\ g\ge0.
\]

The sector is N=L=q on a periodic L>=3 ring. The neutral modulus and quantum
prescription remain assumptions. No new interaction, chemical potential,
background charge, physical scale, or parameter fit is introduced.
Let Omega=|1,...,1>. The D=1 subspace consists of one doublon and one hole.
Its exact compression is A1=gI+P1 V P1; it is not an invariant subspace of H.

Throughout this note, E is a **compression energy relative to the unperturbed
product-state energy zero**, not a certified excitation frequency E_n-E_0 of
the full interacting system. For numerical work retain s=(E-g)/C directly;
subtracting a large rounded g from a rounded energy would lose bandwidth.

## 2. Momentum block and endpoint source

For k=2pi m/L and r=1,...,L-1 use the existing convention

\[
|k,r\rangle=L^{-1/2}\sum_h e^{-ikh}|h+r,h\rangle,
\qquad T|k,r\rangle=e^{ik}|k,r\rangle.
\]

The lower off-diagonal of A1(k) is -C(2+e^{-ik}). The relative chain is open:
collision hops leave P1, rather than wrapping between r=1 and r=L-1.
Put

\[
a(k)=|2+e^{-ik}|=\sqrt{5+4\cos k},\quad t=Ca(k),\quad
\theta=\arg(2+e^{-ik}).
\]

The diagonal phase transformation with entries exp(i(r-1)theta) reduces the
block to a real open chain with hopping -t. Its eigenvectors and energies are

\[
u_j(r)=\sqrt{2/L}\,e^{i(r-1)\theta}\sin(\pi jr/L),\qquad
E_j=g-2t\cos(\pi j/L),\quad j=1,\ldots,L-1.
\]

For g>0 define w=-V Omega/g and the density observable
rho_k=L^(-1/2) sum_x exp(-ikx)(n_x-1). Every adjacent doublon–hole configuration
in w has coefficient sqrt(2)C/g. Projecting rho_k w onto the momentum block
leaves only two endpoints:

\[
z_1=\frac{\sqrt2 C}{g}(e^{-ik}-1),\qquad
z_{L-1}=\frac{\sqrt2 C}{g}(e^{ik}-1).
\]

They are distinct even for L=3. Thus

\[
W=\|z\|^2=16(C/g)^2\sin^2(k/2).
\]

At exactly m=0 modulo L, the density operator is structurally zero in N=L.
For g>0 its leading measure is zero and cannot be normalized. At g=0, w and
the virtual-source construction are undefined, including its use for m=0;
this does not remove the separate structural identity rho_0=0.

## 3. Exact finite-line weights, including interference

For nonzero m and g>0, set phi=k-(L-2)theta and beta=-cos(phi). Taking overlaps
with the sine eigenvectors gives

\[
w_j=\frac2L\sin^2(\pi j/L)
\left|z_1+(-1)^{j+1}e^{-i(L-2)\theta}z_{L-1}\right|^2,
\]

and, because z_last=-exp(ik)z_1,

\[
p_j=\frac{w_j}{W}=\frac2L\sin^2(\pi j/L)
[1+\beta(-1)^{j+1}].
\]

Completeness gives sum_j p_j=1. Since |beta|<=1,

\[
0\le p_j\le4/L,\qquad w_j\le4W/L.
\]

For numerical evaluation use (4/L)sin²(pi j/L) times sin²(phi/2) for odd j
and cos²(phi/2) for even j. This avoids subtracting nearly equal numbers;
it does not authorize clipping weights or renormalizing their sum.
At k=pi, allowed only at even L, theta=0 and even-j weights vanish
structurally. Generic tiny numerical weights are not certified dark lines.

A small-size counterexample to premature symmetry is L=3,m=1 or2:

\[
(E_1,E_2)=(g-\sqrt3C,g+\sqrt3C),\quad
(p_1,p_2)=((2+\sqrt3)/4,(2-\sqrt3)/4),
\]

so sum_j p_j(E_j-g)=-3C/2. Finite-size interference cannot simply be dropped.

## 4. Weak limit with a quantitative bound

Let mu_(L,k)=sum_j p_j delta_(E_j). Define the probability measure

\[
d\sigma_k(E)=\frac{\sqrt{4t^2-(E-g)^2}}{2\pi t^2}
\mathbf1_{|E-g|\le2t}\,dE.
\]

Under E=g-2t cos(u), this becomes (2/pi)sin²(u)du on [0,pi]. It is a
semicircle rather than the inverse-square-root density of the unweighted
chain levels: the endpoint source supplies the sin² factor.

For every real K-Lipschitz function f,

\[
\boxed{|\mu_{L,k}(f)-\sigma_k(f)|\le32tK/(3L).}
\]

Here Lipschitz continuity need only hold on the common spectral support.
To prove the bound, subtract f(g), which integrates equally in both
probability measures, and set

\[
F(u)=\sin^2u\,[f(g-2t\cos u)-f(g)].
\]

This absolutely continuous function vanishes at both endpoints. Its total
variation satisfies

\[
\operatorname{Var}(F)\le K\int_0^\pi
[4t\sin u\cos^2u+2t\sin^3u]du=16tK/3.
\]

For h=pi/L and u_j=jh, the composite trapezoidal error in
(2/L)sum_j F(u_j) is at most Var(F)/L. The interference sum obeys

\[
\sum_{j=1}^{L-1}(-1)^{j+1}F(u_j)
=\tfrac12\sum_{j=1}^L(-1)^{j+1}[F(u_j)-F(u_{j-1})].
\]

Its absolute value is at most Var(F)/2; multiplying by 2beta/L gives another
Var(F)/L. These estimates prove the bound uniformly in the endpoint phase.

To compare k_L with a target k, couple the two semicircles by the same u.
Then their f expectations differ by at most 2K|t(k_L)-t(k)|. The reverse
triangle inequality gives |a(k_L)-a(k)|<=|exp(-ik_L)-exp(-ik)|<=d_circle(k_L,k).
Consequently, at fixed C,g,

\[
\boxed{|\mu_{L,k_L}(f)-\sigma_k(f)|\le
K[32t(k_L)/(3L)+2C\,d_{\mathbb T}(k_L,k)].}
\]

This proves weak convergence for any sequence of nonzero allowed modes with
k_L tending to k. All supports lie in [g-6C,g+6C]; uniform piecewise-linear
approximation extends the result to continuous test functions. The limit is
absolutely continuous and has no atom. A vanishing maximum finite-line weight
alone would not prove that fact: many vanishing lines could otherwise collect
at one energy. The weak-limit theorem supplies the missing argument.

In s=(E-g)/C coordinates replace t by a and set the center to zero. The
normalized density is sqrt(4a²-s²)/(2pi a²) on |s|<=2a; dE=C ds. The bound
implemented in these coordinates is K[32a(k_L)/(3L)+2d_circle(k_L,k)].

## 5. Independent moment and limiting-sequence checks

An endpoint-to-endpoint walk on the L-1 vertex chain first exists at length
L-2. A diagonal endpoint walk first detects the missing far half-line vertex
at length 2(L-1). Thus the full source measure matches semicircle polynomial
moments for every degree d<L-2. Centered odd moments vanish; even moments are
Cat_n t^(2n), where Cat_n=binomial(2n,n)/(n+1). At the first excluded degree,

\[
\mu_{L,k}((E-g)^{L-2})-\sigma_k((E-g)^{L-2})
=\beta(-t)^{L-2}.
\]

The strict degree inequality matters, especially for the L=3 mean above.
Useful independent expectations in s coordinates are E[s]=0, E[s²]=a² and
E[|s|]=8a/(3pi). Numerical residuals against these identities are diagnostics,
not empirical validation or roundoff certificates.

For fixed nonzero integer m, fixed C/g and L tending to infinity,

\[
W_L\sim16\pi^2m^2(C/g)^2/L^2.
\]

The absolute measure tends to zero while its normalized shape tends to the
k=0 semicircle with a(0)=3. This is not normalization of the exactly zero
mode. For fixed limiting nonzero lattice momentum, W instead approaches a
positive value at fixed C/g. If g grows at fixed C, energies translate to
infinity; use centered coordinates rather than claiming a tight limit on
the original energy axis. These are large-volume lattice limits, not a
limit of physical lattice spacing to zero.

## 6. What this does not establish for the full Hamiltonian

The previous sufficient excitation separation uses v=2CL and requires
**g>4CL**. At C=1,g=40 it holds only for L<10 and fails at equality L=10.
The full numerical adapter remains capped to L=3,...,6. The analytic path
formulas here avoid that allocation, but do not extend the full-model proof.

The separate [population theorem](substrate_vacuum_selection_2026-09-12.md)
requires q=L and g>=4C. Selecting N=L energetically does not establish an
excitation bound, select the product state Omega, or prepare that population
under number-conserving evolution. Varying q=L along the sequence is also
an explicitly chosen family of neutrality constraints.

Existing density estimates control total weight, not matched frequencies,
individual residues, or confinement to the first excited band. Even choosing
a growing g_L to maintain separation does not turn those estimates into a
spectral-measure convergence theorem. Uniform local or spectral estimates for
the full Hamiltonian remain needed. The compression result therefore does
not rule out finite-g bound states, other particles, or other emergent mechanisms.

The lower compression threshold has expansion

\[
g-2t(k)=g-6C+(2C/3)k^2+O(Ck^4).
\]

It is not an identified particle pole. A massive relativistic dispersion also
has a quadratic small-momentum expansion, so this neither establishes nor
disproves relativity. No common limiting speed, Lorentz-covariant interactions,
physical fermions, selected geometry, masses or mixing are derived. The legacy
bridge that inserts Planck spacing and a light speed is a different construction
and is neither reused nor repaired here. BPR's empirical/TOE status is unchanged.

## 7. Bounded implementation and verification

The new analytic module uses O(L) arrays with a pre-allocation cap L<=4096;
scalar threshold audits do not inherit this array cap. Existing computational
modules and full-Fock caps remain unchanged. Numerical evaluation follows the
normal-binary64 input policy. Signed momentum aliases, stable half-angle source
amplitudes, centered energies, and separately reported raw-weight availability
prevent underflow or bandwidth loss from masquerading as physical zero.
Threshold comparisons use exact Fractions of accepted binary64 C,g values.
Other floating evaluations do not include rigorous roundoff bounds.

Frozen illustrations use L=5,C=1,g=40,m=1; L=16,64,256 at k=pi/2; the same
three sizes with m=1; and zero-mode/free-source controls. No scan, tuning,
physical calibration, or large full-Hamiltonian diagonalization is performed.
The fixed k=pi/2 examples have W=0.005 and a=sqrt(5). Their full-Hamiltonian
separation condition is unavailable, notwithstanding the compression theorem.
The demo writes only text or strict JSON to stdout.

Independent mathematical review confirmed the source phases, interference,
strict moment thresholds and the constant 32tK/(3L) in the weak-limit proof.
Direct small-system source/probability comparisons in that review differed by
less than 4e-16; these are numerical diagnostics, not rigorous roundoff bounds.
The first focused implementation suite passed **78 tests with warnings as
errors**. The existing bounded regression separately passed **937 tests**, with
one intentional phase-scan deselection and the existing unregistered `slow`
marker warning. All eight algebra-check groups and all eight prior demos passed;
the new demo passed text/strict JSON from an empty directory in the focused suite.
The first expansion passed **85 tests**. Independent implementation review
then found cancellation in a scalar circular-distance calculation for extremely
large L near the -pi/pi seam: a nonzero distance could round to zero. The helper
now computes that difference using exact rational representations of the input
angles before returning to floating point. Regression tests cover both seams.
This fixes numerical evaluation of the displayed bound, not its analytic proof;
angle conversion and transcendental functions still lack certified roundoff.
Scalar overflow validation was also standardized to the inherited ValueError
contract. The post-fix focused suite passed **89 tests with warnings as errors**;
the implementation reviewer independently reproduced the corrected seam bound
and reported no remaining findings. Additional review probes covered exact
audits through L=10^5000, strict JSON, allocation caps and Python 3.8 grammar.
The final post-fix combined bounded regression passed **1,026 tests**, with one
intentional phase-scan deselection and the same existing marker warning. All
nine demos passed text and strict JSON from an empty directory without stderr
or emitted files; all eight algebra-check groups passed again. These are
conditional mathematical and implementation checks, not a full-repository test
or empirical validation. This task changed only the four new files and two
documentation links; previous computational modules stayed unchanged. No
commits, pushes, merge attempts, remote operations or instruction edits were
performed during this task.

Illustrative computed maximum normalized line weights at fixed k=pi/2 were
approximately 0.145059, 0.0454533 and 0.0156177 for L=16,64,256. The analytic
upper bounds are respectively 0.25, 0.0625 and 0.015625. At fixed m=1 the total
weights were approximately 3.80602e-4, 2.40764e-5 and 1.50591e-6. These are
checks of the conditional formulas, not observed particle intensities. No
monotonic finite-size error or empirical agreement is inferred.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m pytest -q -W error tests/test_substrate_neutral_continuum.py
python3 scripts/demo_substrate_neutral_continuum.py
python3 scripts/demo_substrate_neutral_continuum.py --json
```
