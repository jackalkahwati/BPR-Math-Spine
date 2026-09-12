# Endogenous monopole sources: conditional selection and an alignment obstruction

**2026-09-12. Status: a new fixed-occupation classical mean-field model, not a BPR-derived vacuum or physical flavor prediction.**

The [externally forced prototype](toe_constructive_extension_2026-09-12.md)
solves for scalar profiles given arbitrary sources. Its nine real low harmonics
span all Hermitian 3×3 overlap matrices, so that construction does not explain
flavor. Here we replace prescribed angular sources by densities of varied
positive occupation matrices and classify the global minima of one specified
action. No empirical mass or mixing targets enter the calculation.

The result is restrictive: the model selects coherent nonuniform densities and
split overlap eigenvalues, but an attractive inter-sector coupling aligns the
two sectors and gives identity absolute mixing. At zero coupling, their relative
orientation is undetermined. This is progress in constraining sources, **not a
successful explanation of observed flavor**.

## 1. Added assumptions and conventions

Retain the round sphere of radius R, q=3 monopole sections, the assumed 4D
chiral EFT, and its overlap prescription from `bpr/chiral_flavor_prototype.py`.
The sphere, flux, EFT field content and physical scales are not derived here.
Introduce two collective variables

\[
\rho_u,\rho_d\succeq0,\qquad \operatorname{Tr}\rho_u=
\operatorname{Tr}\rho_d=1.
\]

These are **normalized mean-field occupations**, not established Lorentz-scalar
SM condensates. A trace-one occupation is not a statement that three fermion
families are filled. With three states, total fermion occupation three and the
Pauli bound \(0\preceq\rho\preceq I\) would instead force \(\rho=I\).
Neither the existence nor the fixed magnitude of the occupation is derived.

The implemented sections are

\[
\psi_k=\sqrt{\frac{3\binom2k}{4\pi R^2}}
\cos^{2-k}(\theta/2)\sin^k(\theta/2)e^{ik\phi},\quad k=0,1,2.
\]

Their common patch phase cancels below. Define

\[
T(\Omega)_{ij}=\psi_i^*\psi_j,\qquad
n_\rho(\Omega)=\operatorname{Tr}(\rho T(\Omega)).
\]

For \(\rho=vv^\dagger\), this is \(|\sum_i\psi_iv_i|^2\), **not**
\(\psi^\dagger\rho\psi\) with the same coefficient matrix. Consequently

\[
\int dA\,n_\rho=1,\qquad
\int dA\,h n_\rho=\operatorname{Tr}[\rho P(h)],\quad
P(h)_{ij}=\int dA\,\psi_i^*h\psi_j.
\]

With the existing real harmonics normalized by \(\int d\Omega Y_A Y_B=\delta_{AB}\),

\[
T_A=P(Y_A),\qquad n_A=\frac{\operatorname{Tr}(\rho T_A)}{R^2},
\qquad n_\rho=\sum_A n_A Y_A.
\]

For q=3 the products of sections have only \(\ell=0,1,2\), so this is an
exact density expansion, not a low-mode truncation of a larger density.
The distinction between area and solid-angle coefficients accounts for R⁻².
A coherent occupation pointing at \((\theta_n,\phi_n)\) uses
\(v=\psi(\theta_n,\phi_n)^*/\|\psi(\theta_n,\phi_n)\|\).
Omitting the conjugation reflects the azimuth under the implemented convention.

### Controls and parameter freedom

The binomial theorem gives, for every positive q,

\[
\sum_{k=0}^{q-1}|\psi_k|^2=\frac{q}{4\pi R^2}.
\]

Equal normalized filling \(\rho=I/q\) therefore gives constant density and a
scalar overlap proportional to identity. Degenerate singular values make
mixing **undefined**, not a prediction of a particular mixing matrix.

Positivity alone does not select a density. The fixed-trace PSD q=3 domain has
eight real local directions about I/3, and the overlap map is injective on
those directions. It is a constrained region, not all signed sources; it still
contains continuous freedom until an action and a state-selection rule are added.

## 2. One specified angular-source-free action

Set \(f_a=h_a-h_0\), and stipulate

\[
E[\rho,h]=\int dA\left[
\frac12\sum_{a=u,d}\left(\kappa|\nabla f_a|^2+\mu^2 f_a^2\right)
-\eta f_uf_d-\sum_{a=u,d}g_a(h_0+f_a)n_{\rho_a}\right],
\]

with

\[
R>0,\quad\kappa\ge0,\quad\mu^2>\eta\ge0,\quad
 g_u,g_d>0,\quad h_0\ge0.
\]

“Source-free” here means no prescribed angular source coefficients. The
occupation constraint, couplings, baseline and action itself are **new inputs**.
The scalar fields are not a complete matter dynamics. The restriction on h₀
will ensure positive selected Yukawa eigenvalues; boundedness alone does not
require that sign.

For a degree-\(\ell\) harmonic define

\[
d_\ell=\mu^2+\kappa\ell(\ell+1)/R^2,\qquad
D_\ell=\begin{pmatrix}d_\ell&-\eta\\-\eta&d_\ell\end{pmatrix},\qquad
j_A=\begin{pmatrix}g_un_{u,A}\\g_dn_{d,A}\end{pmatrix}.
\]

The unique scalar response is \(f_A=D_{\ell_A}^{-1}j_A\). The scalar Hessian
blocks \(R^2D_\ell\) are strictly positive. All unsourced higher modes minimize
to zero. Eliminating the scalar fields gives

\[
E_{\mathrm{eff}}(\rho_u,\rho_d)=-h_0(g_u+g_d)
-\frac{R^2}{2}\sum_A j_A^T D_{\ell_A}^{-1}j_A. \tag{1}
\]

This is bounded below on the compact product of occupation domains and attains
its minimum. It is **not jointly convex** in matter and scalar variables:
scalar exchange contributes a negative occupation quadratic form. Removing
the trace constraint without another stabilizing term permits unbounded
occupation amplitude, even though the scalar Hessian is positive.

## 3. Complete global-minimum classification

Write

\[
a_\ell=\frac{d_\ell}{d_\ell^2-\eta^2},\qquad
b_\ell=\frac{\eta}{d_\ell^2-\eta^2}.
\]

### Purity

The reward (minus the occupation-dependent part of (1)) is strictly convex
on joint traceless perturbations: every inverse D is positive definite,
both g are positive, and the nine overlap operators span the Hermitian
matrices. A nonextreme point is a strict convex combination of distinct
occupation pairs and therefore cannot maximize this reward. Thus **every**
global-minimizing pair has both occupations pure.

### Pure spin-one multipoles

For a pure state set \(s^2=|\langle\mathbf J\rangle|^2\in[0,1]\).
The degree powers are

\[
p_0=\frac1{4\pi R^4},\qquad
p_1=\sum_{\ell_A=1}n_A^2=\frac{3s^2}{16\pi R^4},\qquad
p_2=\sum_{\ell_A=2}n_A^2=\frac{4-3s^2}{80\pi R^4}. \tag{2}
\]

One way to derive (2) is to decompose the density matrix into orthogonal
irreducible scalar, vector and quadrupole parts. Their Hilbert–Schmidt squared
norms are \(1/3,s^2/2,2/3-s^2/2\), respectively: the vector part is
\(\langle\mathbf J\rangle\cdot\mathbf J/2\), since
\(\operatorname{Tr}(J_iJ_j)=2\delta_{ij}\), and purity supplies the remainder.
The overlap map multiplies these norms by
\(3/(4\pi R^4),3/(8\pi R^4),3/(40\pi R^4)\), as follows by evaluating a
normalized harmonic operator in each irreducible subspace. This yields (2).
The y-generator may acquire a sign under the section convention; s² is
unchanged.

For positive weights w₁,w₂, the weighted nonconstant power increases with s²
when \(5w_1>w_2\). The a weights are positive and nonincreasing in degree;
for eta>0 the b weights are too. The inequality is strict even at kappa=0,
where weights coincide. Thus every self term is maximal at s²=1: a coherent
spin-one state. Weighted Cauchy–Schwarz bounds the cross term by the product
of the b-weighted norms. Identical coherent occupations simultaneously
saturate all bounds. For eta>0 equality in the cross-term bound requires
identical density coefficient vectors, hence identical occupations.

The complete minimizing family is therefore

\[
\begin{cases}
\rho_u=\rho_d=\rho(n),\quad n\in S^2,&\eta>0,\\
\rho_u=\rho(n_u),\ \rho_d=\rho(n_d),\quad n_u,n_d\in S^2,&\eta=0.
\end{cases} \tag{3}
\]

With \(c=(1/(4\pi),3/(16\pi),1/(80\pi))\), the exact bound is

\[
\boxed{E_{\min}=-h_0(g_u+g_d)-\frac1{2R^2}
\sum_{\ell=0}^2c_\ell\left[a_\ell(g_u^2+g_d^2)+2b_\ell g_ug_d\right].} \tag{4}
\]

The exact certificate is \(E_{\mathrm{eff}}-E_{\min}\), not an optimizer convergence
flag. Numerically, subtracting total energies can erase a small selection gap
beneath the common baseline and monopole reward. A stable evaluation must
cancel those occupation-independent terms analytically and compare the
nonconstant multipole rewards, with a numerical resolution estimate. A
residual consistent with zero at that resolution is not by itself proof of
membership in the minimizing family; the analytic classification (3) supplies
that proof for the constructed representatives. The common orientation in the coupled model is a symmetry orbit, not a
new observable flavor parameter. In the uncoupled model the relative angle
is a genuine flat direction of this specified action, not a prediction.

## 4. Selected overlaps and the no-mixing result

Choose the common north-axis representative **after** deriving (3). Then

\[
\rho=\operatorname{diag}(1,0,0),\qquad
n=\frac1{R^2}\left[\frac1{4\pi}+\frac{3z}{8\pi}+\frac{P_2(z)}{8\pi}\right].
\]

Define \(w_{a,\ell}=a_\ell g_a+b_\ell g_{\bar a}\). The selected profile is
\(h_a=A_a+B_az+C_aP_2(z)\), with

\[
A_a=h_0+\frac{w_{a,0}}{4\pi R^2},\quad
B_a=\frac{3w_{a,1}}{8\pi R^2},\quad
C_a=\frac{w_{a,2}}{8\pi R^2}.
\]

The actual overlap integral yields

\[
P(h_a)=\operatorname{diag}\left(
A_a+B_a/2+C_a/10,\ A_a-C_a/5,\ A_a-B_a/2+C_a/10\right). \tag{5}
\]

Monotonic w gives \(B_a\ge3C_a>0\) and
\(A_a-h_0\ge2B_a/3\). Both adjacent gaps \(B_a/2\pm3C_a/10\) are positive,
and the lowest eigenvalue obeys

\[
\lambda_{\min}\ge h_0+B_a/6+C_a/10>0.
\]

For positive effective Yukawa couplings, singular values therefore have the
same ordering in both sectors. The sectors share their left eigenvectors:
**\(|V|=I\) and the Jarlskog invariant is zero** throughout the stated
eta>0 domain, not merely at the demonstration point. Equal source couplings
produce equal profiles; equal effective Yukawa couplings then also produce
equal spectra. These restrictions rule out this model as a complete flavor
explanation without any numerical comparison to measured targets.

At eta=0, coherent axes can differ at no energy cost. Their relative mixing is
a spin-one rotation with an undetermined relative angle. Each representative
has well-defined singular vectors but the action does not select the mixing
magnitudes. The rotation family can be rephased to a real small Wigner matrix,
so its CP-odd quartet is zero. This is special to coherent axial minima:
**real scalar profiles in general need not conserve CP** in this background,
as the externally forced prototype demonstrates.

## 5. Classical selection is not quantum vacuum selection

Rotational averaging of the coherent projector gives I/3. Its density is
constant and its direct mean-field response is isotropic. However, evaluating
the nonlinear eliminated action on this averaged occupation is not the same
as averaging the energies of correlated, individually minimizing
occupation–field configurations. The former loses the correlations that
lower the energy. Neither calculation establishes the exact quantum ground
state. On a finite system, symmetry averaging may restore rotational
invariance; this classical orbit is not proof of spontaneous quantum symmetry
breaking.

## 6. Executable checks and frozen demonstration

Implementation: `bpr/flavor_source_selection.py`.
Tests: `tests/test_flavor_source_selection.py`.
Stdout-only demonstration:

```bash
python3 scripts/demo_flavor_source_selection.py
python3 scripts/demo_flavor_source_selection.py --json
```

Frozen inputs, chosen before calculating outputs:
R=kappa=mu2=g_u=g_d=y_eff_u=y_eff_d=1, h0=2, eta=1/4.
The exact aligned profile has
\(A=2+1/(3\pi), B=3/(22\pi), C=1/(54\pi)\).
Both sectors have ascending dimensionless singular values
\[
\left(2+\frac{793}{2970\pi},\quad
2+\frac{89}{270\pi},\quad
2+\frac{599}{1485\pi}\right)
\simeq(2.0849898114,2.1049243699,2.1283957049).
\]
The analytic minimum is \(-4-599/(1485\pi)\simeq-4.1283957049\),
whereas equal filling gives \(-4-1/(3\pi)\simeq-4.1061032954\).
These numbers follow from the frozen toy inputs, not physical calibration.

Controls are equal filling, eta=0 with different representative relative
orientations, and rotationally averaged occupation. A representative matrix
must not be labeled a dynamically predicted mixing matrix in the flat case.

Verification covers direct density quadrature, complex transpose convention,
radius factors, multipole identities, scalar stationarity, the exact global
bound, positive ordered overlaps and the distinction between degeneracy and
unselected orientation. Finite trial states are regression checks of the
analytic theorem, not an exhaustive numerical proof.

Independent proof review found no substantive mathematical error in this
stated domain. Separate quadrature checks at R=1.7 confirmed the density
convention, multipole powers and normalization for four deterministic pure
states, including complex ones. The axial overlap formula agreed within
2.3e-15. This review checked the theorem and existing overlap conventions,
not the new implementation. A subsequent independent implementation review
found and verified fixes for nonfinite inherited diagnostics and cancellation
in the numerical energy certificate. The latter now resolves an equal-filled
gap of 3.0504697425946596e-17 at kappa=1e15, even when direct total-energy
subtraction rounds to zero. No substantive findings remained in the reviewed
scope after these fixes.

The new source-selection suite has 82 tests. Together with the 71 existing
flavor-prototype tests, **153 tests passed**, including the new demo's text and
strict JSON modes, complex density conventions, scalar-action quadrature,
global-bound checks and the numerical regressions. The final combined gauge,
flavor, foundation and historical regression selection passed **338 tests**,
with one phase-scan test deliberately deselected and the existing unregistered
`slow` marker warning. No full-repository test claim is made.

The numerical implementation has a narrower domain than the exact theorem:
it rejects nonfinite intermediate calculations, unresolved energy scales,
unresolved analytic spectral gaps and nonfinite inherited mixing diagnostics.
These are conservative float64 restrictions, not claims that the mathematical
model ceases to exist at those parameters. In particular a representable final
energy may still be rejected if its intermediate quadratic reward overflows.
No eigenvalue clipping, occupation renormalization or logarithmic floor repairs
such inputs silently.

The pre-existing bounded regression selection was rerun: 256 tests passed,
with one phase-scan test deliberately deselected and one existing unregistered
`slow` marker warning. All eight independent algebra-check groups and both
output modes of the original gauge/flavor demo also passed. The full
repository suite was not run.

No benchmark targets, production Monte Carlo, fitting, remote computation or
physical unit matching are part of this pass. Previous gauge and externally
forced flavor examples remain unchanged.

## Operator follow-up (separate model)

The [quantum source diagnostic](quantum_flavor_sources_2026-09-12.md) lifts
the overlap algebra into a newly stipulated six-mode fermionic Hamiltonian.
Expectations of density squares differ from squared mean densities; the
1+1 quantum ground space is rank five at positive exchange and rank nine at
zero exchange. This does not invalidate the classical theorem above, but
prevents interpreting its minimizing orbit as an established unique quantum
vacuum. Operator ordering and the missing microscopic matching inputs are
reported explicitly. No physical mixing prediction follows from selecting
an arbitrary vector in the degenerate ground space.

## 7. What remains open

The action, occupation constraint and coupling signs were stipulated. They
have not been derived from the BPR substrate or from a consistent
higher-dimensional theory. Flux selection, radius stabilization, mirror
removal, the physical scalar source operator and the quantum state remain
open. Adding arbitrary occupation textures would simply reintroduce the
source freedom removed here.

A subsequent mechanism would have to justify an interaction that selects a
non-aligned relative state, with its coefficients independently fixed, while
respecting the EFT symmetries and boundedness. This pass does **not** add such
an interaction to manufacture mixing. Its completed scientific claim is the
conditional selection theorem and its alignment obstruction, not a TOE.
