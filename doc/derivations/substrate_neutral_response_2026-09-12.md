# Neutral excitations and density response of the finite bosonic ring

**2026-09-12. A conditional finite-system calculation, not physical fermion
emergence or a completed theory of everything.**

## 1. Inherited model and chosen sector

Use the complete quantum Hamiltonian from the
[fermionization diagnostic](substrate_fermionization_2026-09-12.md), unchanged:

\[
H=gD+V,\quad D=\frac12\sum_x n_x(n_x-1),\quad
V=-C\sum_x(a_x^\dagger a_{x+1}+a_{x+1}^\dagger a_x),
\quad C>0,\ g\ge0.
\]

We choose N=L and neutrality modulus q=L. The inherited rule N=0 modulo q
then admits **every** state in this fixed-number space, not just the single
hard-core configuration. This is a chosen population and geometry, not a
vacuum-selection theorem. The quantum prescription itself remains a declared
extension of the legacy three-site model, not uniquely derived from classical
DNLS. No background charge, chemical potential, interaction, or fitted scale
is introduced.

The dimension is binomial(2L-1,L), without an additional local occupation
cutoff. The existing resource cap permits L=3,4,5,6, with dimensions
10,35,126,462. In particular, triple occupancy is retained at L=3.

## 2. One doublon and one hole

The unique D=0 state is Omega=|1,...,1>. Define P0=|Omega><Omega| and
P1=1[D=1]. Every P1 configuration has one site d with occupation two,
one site h!=d with occupation zero, and unit occupation elsewhere:

\[
\dim P_1=L(L-1),\qquad A_1=P_1HP_1=gI+P_1VP_1.
\]

In normalized occupation states |d,h>, a doublon hops to a neighboring
non-hole site with coefficient -2C; a hole hops to a neighboring
non-doublon site with coefficient -C. These factors are the bosonic square
roots, not new parameters. Collision transitions leave P1. For example,
a neighboring doublon and hole annihilate into Omega with coefficient
-sqrt(2)C. Coupling to higher D is also present. Therefore P1 is not an
invariant finite-g subspace.

Choose translation T|d,h>=|d+1,h+1>, with sites modulo L. Let

\[
|k,r\rangle=L^{-1/2}\sum_h e^{-ikh}|h+r,h\rangle,
\quad k=2\pi m/L,\quad r=1,\ldots,L-1.
\]

Then T has eigenvalue exp(ik), and

\[
A_1(k)|r\rangle=g|r\rangle
-C(2+e^{-ik})|r+1\rangle-C(2+e^{ik})|r-1\rangle.
\]

Terms outside 1,...,L-1 are omitted. This is an **open** relative-coordinate
chain, not a periodic chain allowing the defects to pass through each other.
Removing the constant hopping phase by a diagonal unitary gives

\[
\boxed{\lambda_{m,j}=g-2C\sqrt{5+4\cos k}\cos(\pi j/L)},
\quad j=1,\ldots,L-1.
\]

These are exact eigenvalues of the compression. They are not individually
matched exact excitations, nor a dispersion relation for observed particles.
The hopping describes constrained defects, not two independent fermions.

## 3. An isolated excited-cluster certificate

The one-particle hopping norm is at most 2C. Second quantization at fixed
N=L gives ||V||<=v=2CL. If g>2v, Weyl's inequality separates the full spectrum
into exactly one eigenvalue in [-v,v], exactly L(L-1) in [g-v,g+v], and
all others >=2g-v. At L=3 the D=2 manifold is absent, but this lower bound
remains valid and conservative.

Let Q1=I-P1, B1=P1HQ1, K1=Q1HQ1, delta=g-2v. The unperturbed K1 has one
zero eigenvalue and all others at least 2g. Its hopping perturbation has norm
at most v. Hence for E in [g-v,g+v],

\[
\operatorname{dist}(E,\sigma(K_1))\ge\delta,
\qquad S_1(E)=A_1-B_1(K_1-E)^{-1}B_1^\dagger,
\]
\[
\boxed{\|S_1(E)-A_1\|\le\|B_1\|^2/\delta},\qquad
\boxed{\|Q_1\psi\|\le(\|B_1\|/\delta)\|P_1\psi\|}.
\]

The latter follows by solving the excluded component of H psi=E psi.
K1-E is **indefinite**: K1 Omega=0 because V Omega lies in P1. Neither the
positive ground-complement proof nor a negative-semidefinite sign for this
Schur correction applies. Evaluate the correction separately from A1 to
avoid subtractive cancellation.

These are energy-dependent resolvent and eigenstate bounds. They are not
an autonomous exact Hamiltonian, all-time dynamics certificate, or sorted
individual-eigenvalue error theorem. At g<=2v the sufficient certificate is
unavailable; that does not disprove the existence of excitations.

## 4. A density measurement couples through virtual admixture

Define the diagonal density Fourier observable

\[
\rho_m=L^{-1/2}\sum_x e^{-ikx}(n_x-1).
\]

Subtracting one defines a density fluctuation, not background charge in H.
At fixed N=L, rho0=0 structurally. Every rho_m annihilates Omega. An
undressed hard-core calculation would therefore miss finite-g density response.
For nonzero m, ||rho_m||=sqrt L: the triangle inequality gives the upper
bound and putting all L particles at one site attains it.

For g>0 the leading virtual component is w=-V Omega/g. It has 2L distinct
adjacent doublon-hole configurations, each with coefficient sqrt(2)C/g,
including on the three-site ring. On |d,h>, rho_m has diagonal element
(exp(-ikd)-exp(-ikh))/sqrt L. Consequently

\[
\boxed{W_m^{(2)}=\|\rho_m w\|^2
=8(C/g)^2(1-\cos k)=16(C/g)^2\sin^2(\pi m/L)}.
\]

The last expression avoids small-angle subtraction. At g=0, w and this
perturbative weight are undefined; the exact finite-system response is not.

### Total-weight error bound

For g>2v, write the normalized ground vector G=alpha Omega+chi, with
alpha positive and chi orthogonal to Omega. In this paragraph only,
Q0=I-P0 and K0=Q0HQ0. Define

\[
b=\|Q_0V\Omega\|=2C\sqrt L,\quad r=b/\delta.
\]

Ground-complement elimination gives
chi=-alpha(K0-E0)^(-1)Q0V Omega and ||chi||<=r alpha. Thus
alpha>=(1+r²)^(-1/2), and 1-alpha<=r²/2. Moreover D V Omega=V Omega,
so (gD_Q0)^(-1)Q0V Omega=Q0V Omega/g. Using |E0|<=v and the resolvent
identity,

\[
\|(K_0-E_0)^{-1}-(gD_{Q_0})^{-1}\|
\le\frac{2v}{g\delta}.
\]

Splitting the difference into normalization and resolvent terms proves

\[
\|\chi-w\|\le\eta=\frac{r^2}{2}\frac bg+\frac{2vb}{g\delta}.
\]

Since rho_m Omega=0, let d_m=||rho_m|| and apply the difference-of-squares
inequality to rho_m chi and rho_m w:

\[
\boxed{\left|\|\rho_mG\|^2-W_m^{(2)}\right|
\le d_m\eta\left(2\sqrt{W_m^{(2)}}+d_m\eta\right)}.
\]

This bounds the **total** density weight, not individual spectral-line
intensities, matched frequencies, or the fraction concentrated in the first
excited band. No rigorous floating-point roundoff bound is included.

## 5. Exact finite-system spectral response and operator algebra

For full eigenpairs, the numerical density spectral measure is

\[
S_m(\omega)=\sum_n|\langle n|\rho_m|G\rangle|^2
\delta(\omega-(E_n-E_0)).
\]

Sum weights over degenerate eigenspaces: individual vectors inside those
spaces are arbitrary. A finite numerical tolerance identifies unresolved
energy groups; record the tolerance and energy spread, not an assertion of
exact equality based only on proximity. Sum rules are

\[
\sum_n W_n=\|\rho_mG\|^2,\qquad
\sum_n(E_n-E_0)W_n=(\rho_mG)^\dagger(H-E_0)(\rho_mG).
\]

The fixed-number hopping graph is connected and its off-diagonal elements
are nonpositive for C>0. Perron-Frobenius gives a unique positive ground
vector, which is translation invariant. Since T rho_m T†=exp(ik)rho_m,
the elastic expectation vanishes at nonzero m. Numerical elastic residuals
must still be reported. A density-dark state may lie below the first visible
line, so a density-visible gap is not necessarily the smallest neutral gap.

A leading approximate measure may use A1(k) with source P1 rho_m w, taking
the zeroth-order ground energy as zero. Its total weight is W_m^(2). This
does not individually certify its line weights or frequencies. In the weak
control, omit approximate spectral predictions: negative compression values
relative to zero would not represent negative exact excitation gaps.

Number-preserving bilinears satisfy, on the complete fixed-number space,

\[
[a_d^\dagger a_h,a_u^\dagger a_v]
=\delta_{hu}a_d^\dagger a_v-\delta_{dv}a_u^\dagger a_h.
\]

They preserve the chosen neutrality sector, but they do not obey canonical
fermion anticommutation relations. For example, with X=a0†a1,
X²|0,2,L-2,0,...>=2|2,0,L-2,0,...> is nonzero. This is an explicit
non-CAR witness, not a determination of a hypothetical composite's exchange
statistics. Compression to P1 need not preserve the bilinear commutator algebra.

## 6. Frozen controls and scientific limits

Freeze L=N=q=5, C=1, m=1, at g=40 and g=0.7, with structural m=0 controls.
These are mathematical examples, not measured parameter values. At g=40,
W_m^(2) is about 0.00345492 while the conservative total-weight error bound
is about 0.0957881. The theorem is valid but **does not resolve the leading
signal** at this frozen example. Do not tune g to conceal that limitation.
The g=0.7 example has no sufficient separation certificate.

This work addresses nontrivial neutral dynamics inside a stipulated bosonic
ring. It does not derive physical fermions, relativistic chirality, selected
vacuum population, three families, observed masses or mixing, confinement,
unified gauge/gravity dynamics, or empirical evidence for BPR. The earlier
withdrawn TOE claim and failed historical particle benchmark remain unchanged.

### All-population follow-up

The [neutral population-selection calculation](substrate_vacuum_selection_2026-09-12.md)
compares all N=0 modulo q sectors, not just N<=L. In the same unshifted model,
q=L and g>=4C energetically select N=L, but do not prepare that population.
The frozen weak case instead excludes N=L through a higher-population trial;
its remaining global candidates are unresolved. The within-sector response
above remains a conditional calculation. A number-dependent energy shift
changes cross-sector selection and must not be silently introduced.

## 7. Reproduction and verification

Independent mathematical review found no defects in the stated derivations.
Independent occupation-basis calculations for L=3 through 6 reproduced the
momentum blocks and compression spectra, with maximum spectral discrepancy
6.3e-15. Density selection, the leading weight and the density operator norm
were independently checked at every momentum in those sizes.

For the frozen strong control, independent diagonalization gave cluster
populations 1,20,105, exact density weight 0.003491421563, leading weight
0.003454915028, and absolute difference 0.000036506535. The conservative
bound was 0.095788056680. The observed ground-vector remainder was
0.011019617934, compared with eta=0.114598483847. These numerical agreements
are not roundoff certificates or empirical evidence.

The previous bounded regression passed 745 tests, with its intentional
phase-scan deselection and existing unregistered `slow` marker warning.
The eight-group algebra checker and all six previous stdout demos passed
text/strict-JSON checks from an empty temporary directory with no output files.
The initial new focused run had 65 passes and four assertion failures caused
by few-ULP tolerances and a rounded test reference; corrected independent
assertions passed all 69 tests with warnings as errors. Additional report
contract tests passed. Implementation review found one additional API-domain
issue: the bilinear diagnostic's two-particle witness constructed an invalid
occupation for N=0 or N=1 inherited models. It now retains valid commutator
diagnostics and reports that particular witness unavailable in those sectors.
The chosen unit-filling calculations were unaffected. Regression tests cover
N=0,1,2,3 explicitly. Independent rechecking confirmed the fix, valid
commutator diagnostics and strict JSON, with no remaining actionable findings.
The final new demo passed text and strict JSON from an empty directory, with
no stderr or output files. Strong/weak nonzero-mode exact density weights were
0.0034914215631285314 and 0.7369743978785371; both zero-mode controls were
structurally zero. The final focused suite passed **79 tests with warnings
as errors**. These validate conditional model identities and numerical
contracts, not nature's agreement with BPR. The final combined bounded
selection passed **824 tests**, with the same one intentional phase-scan
deselection and existing unregistered `slow` marker warning. No full-repository
test or empirical-validation claim is made. All work remains local; no commit,
push, benchmark run or change to a previous computational module was performed.

Numerical reports use normal binary64 intermediates, rejecting nonfinite,
subnormal or unresolved arithmetic rather than assigning physical significance
to numerical zeros. The full eigensystem uses a scaled Hamiltonian. Its heuristic
resolution proxy is 64 eps times the full dimension, required to be less than
1e-6 times C/max(abs(H)), and for nonzero momentum less than 1e-6 times
min(1,C/g) (with the g=0 response scale set to one). These conservative screens
are not rigorous eigensolver error estimates. Low-level analytic certificates
remain separate from whether a numerical report resolves its requested scales.
Spectral groups use absolute tolerance 128 eps times the number of levels times
max(abs(energies)); the whole group span must fit that tolerance. Actual energy
spreads are reported, and first moments retain actual level energies rather
than replacing them with the group's representative energy. No density clipping,
line-weight fitting, or empirical calibration is performed.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m pytest -q -W error tests/test_substrate_neutral_response.py
python3 scripts/demo_substrate_neutral_response.py
python3 scripts/demo_substrate_neutral_response.py --json
```
