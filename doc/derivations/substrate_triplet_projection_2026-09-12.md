# From the frozen ring to three modes: constructive projection and a source-map obstruction

**2026-09-12. Scope: the existing classical DNLS ring and direct local scalar sources.**

The [quantum-source diagnostic](quantum_flavor_sources_2026-09-12.md) identified
an absent substrate-to-monopole map. Here we test a concrete candidate rather
than postulate another three-state Hamiltonian. A Fourier projection of the
existing ring supplies an exact restricted energy and a finite-time classical
error bound. It does **not** supply the full monopole source map: the direct
local source image has five real directions, whereas the monopole image has
nine. This is a scoped obstruction, not a theorem excluding every emergent
construction.

## 1. The actual starting model

The existing `bpr/condensate_regime.py:95–110` implements

\[
H[\psi]=-C\sum_x(\psi_x^*\psi_{x+1}+\mathrm{c.c.})
+\frac g2\sum_x|\psi_x|^4,
\qquad N=\sum_x|\psi_x|^2.
\]

`bpr/condensate_mechanism.py:51–78` implements its C=1 real-time normalization:

\[
i\dot\psi_x=-C(\psi_{x+1}+\psi_{x-1})+g|\psi_x|^2\psi_x.
\]

Use an integer ring size p>=5, C>0 and g>=0. These ring identities do not need
p to be prime. N is conserved, but its value is an initial condition, not a
selected physical population. The established classical fixed-N uniform
minimum is unchanged. No quantum-vacuum conclusion is imported from it.

The three-site quantum example in `bpr/zp_selection_principle.py:103–122`
uses truncated **bosonic sites**, not monopole modes. Its existence motivates
checking operator conventions, not replacing these sites with canonical
fermions. No new interaction or physical scale is introduced below.

## 2. An exact free three-mode window

Set K={-1,0,1} in that order, and define

\[
U_{xk}=\frac1{\sqrt p}e^{2\pi i xk/p},\qquad U^\dagger U=I_3,
\qquad P=UU^\dagger.
\]

This is an isometry into the ring's one-body vector space. The projector P
commutes with the free Hamiltonian; U intertwines the restricted and full free
dynamics. The interacting flow does not automatically preserve this window. Projectors can be applied as U(U-dagger v), without a dense p-by-p
matrix.

For the site shift `(T psi)_x=psi_(x+1)`, its restriction is
`diag(exp(-2pi i/p),1,exp(2pi i/p))`; site reflection exchanges the outer
modes. The window is a singlet plus a reflection-related doublet under the
ring symmetry. Site translation and common phase rotation are different
actions. A three-dimensional space alone does not establish continuous
rotational symmetry or an irreducible degenerate spin-one spectrum.

The unshifted free energies are E_k=-2C cos(2pi k/p). Measured from E_0,

\[
\epsilon_k=4C\sin^2(\pi k/p),\quad
W=\epsilon_1=4C\sin^2(\pi/p),
\]
\[
\Delta=\epsilon_2-\epsilon_1
=4C\sin(3\pi/p)\sin(\pi/p)>0,
\qquad \frac\Delta W=1+2\cos(2\pi/p)\longrightarrow3.
\]

Thus the rank-three free window really exists and is isolated at every finite
p>=5. Its energies are (W,0,W), not a degenerate triplet. The absolute gap
vanishes as 12 pi-squared C/p-squared; its separation relative to its internal
bandwidth does not diverge. Neither fact alone rules out a weak-interaction
projection, but neither derives a monopole kinetic operator.

## 3. The exact restricted interaction

Write z=(a,b,c), psi=Uz, and N=z-dagger z. Discrete Fourier orthogonality gives

\[
\boxed{
H_G(z)=-2CN+W(|a|^2+|c|^2)
+\frac g{2p}\left[N^2+2|ba^*+cb^*|^2+2|ca^*|^2\right].}
\]

This is **exactly H[Uz]**, for every p>=5. For example, the density has only
Fourier differences 0,+/-1,+/-2, so Parseval gives the three displayed terms.
The cubic projection obeys

\[
i\dot z_k=E_k z_k+
\frac gp\sum_{i,j,l\in K\,:\,i-j+l\equiv k\ (\mathrm{mod}\ p)}
z_i z_j^*z_l.
\]

The same equation follows by differentiating H_G with respect to z-star.
It conserves N and H_G. This Galerkin dynamics is a constructive restriction
of an existing Hamiltonian, not a newly chosen flavor interaction. It is not
generally the exact full-ring motion from the same initial state.

### Explicit nonlinear leakage

The omitted vector field is

\[
r(z)=g(I-P)(|Uz|^2Uz).
\]

Its discarded Fourier amplitudes are g/p times

\[
A_3=c^2a^*,\quad A_2=c^2b^*+2bca^*,\quad
A_{-2}=a^2b^*+2bac^*,\quad A_{-3}=a^2c^*.
\]

Aggregate residues modulo p **before** taking squared magnitudes:

- p=5: A_3 joins A_-2, and A_-3 joins A_2.
- p=6: A_3 and A_-3 join one another.
- p>=7: all four discarded residues are distinct.

There are no aliases back into K for p>=5. Pure Fourier-mode fields have
zero leakage. As a strict witness, a=0 and b=c=sqrt(N/2) give only the
omitted k=2 amplitude

\[
\|r\|_2=\frac{gN^{3/2}}{2\sqrt2\,p}>0\quad(g,N>0).
\]

This proves nonclosure of the full three-mode subspace without a simulation
or a parameter scan. Particular invariant states do not make every state in
the window invariant.

## 4. A genuine, limited finite-time control statement

Let psi(t) solve full DNLS and phi(t)=Uz(t) solve the lifted Galerkin equation,
with identical initial state in the window and common norm-squared N. The
free hopping propagator is unitary. On the norm ball of radius sqrt(N), the
map F(v)=|v|^2 v satisfies

\[
\|F(v)-F(w)\|_2\le3N\|v-w\|_2.
\]

This follows from the real Jacobian of the componentwise cubic, whose norm
is at most 3|v_x|^2, integrated along the segment. Both trajectories remain
in that ball. The three-mode pointwise bound is

\[
\|Uz\|_\infty^2\le\frac{3N}{p},\qquad
\|r(z)\|_2\le\frac{3gN^{3/2}}p.
\]

Duhamel's formula and Gronwall therefore give, for N>0,

\[
\boxed{\frac{\|\psi(t)-\phi(t)\|_2}{\sqrt N}
\le\min\left\{2,\frac{e^{3gN|t|}-1}{p}\right\}.}
\]

The cap follows from the triangle inequality. At g=0 the free window is exact;
at N=0 both trajectories vanish and a relative error is undefined, while the
absolute error is zero. The implementation must distinguish these cases.

For a requested relative tolerance 0<tau<2, a sufficient finite-time condition
is 3gN|t|<=log(1+p tau). This is conservative and can rapidly become vacuous.
It is a classical, finite-time, same-initial-state result, not a quantum or
many-body effective-theory theorem. At fixed N the 1/p factor helps; at fixed
site density N/p the exponential also changes, so there is no automatic
large-p control at physical density. The separate scale gN/(p Delta) is only
an interaction-to-gap diagnostic. At fixed N and C,g it grows proportionally
to p, even as the fixed-time bound improves proportionally to 1/p. These
statements are compatible: the latter uses spatial delocalization, not
spectral isolation. The error proof above does not rely on a one-particle gap
being a many-body nonresonance bound.

## 5. The local source map does not match the monopole map

The natural real local scalar source in the ring is multiplication by f_x;
its bosonic density coupling would be sum_x f_x n_x. Projecting it gives

\[
T_f=U^\dagger\operatorname{diag}(f)U
=\begin{pmatrix}
\alpha&\beta&\gamma\\
\beta^*&\alpha&\beta\\
\gamma^*&\beta^*&\alpha
\end{pmatrix},
\]
\[
\alpha=\frac1p\sum_xf_x,\qquad
\beta=\frac1p\sum_xf_xe^{2\pi i x/p},\qquad
\gamma=\frac1p\sum_xf_xe^{4\pi i x/p}.
\]

The five real source functions 1, cos(theta), sin(theta), cos(2theta),
sin(2theta) independently control these entries for p>=5. The exact linear
real image is thus the **five-dimensional Hermitian Toeplitz subspace**.
For a triplet occupation rho, the density convention remains tr(rho T_f),
including complex off-diagonal entries.

In contrast, the nine real l<=2 harmonics in the existing q=3 monopole
`overlap_matrix` span all Hermitian 3-by-3 matrices. For example,
`P(z)=diag(1/2,0,-1/2)` has nonconstant diagonal and cannot be produced by
this ring map in the displayed basis. The two diagonal-difference directions
and two adjacent-off-diagonal-difference directions are missing. Unitary
basis changes preserve image dimension. Consequently no common unitary
identification can turn this direct linear local source map into the **full**
nine-dimensional monopole map.

This does not say every monopole source is inaccessible, or that restricting
the target to a smaller family is impossible. It does say that a mere
three-state isometry does not reproduce the complete source geometry assumed
by the earlier prototype.

### Crucial limitation: the generated algebra is larger

The compressed cosine and sine sources generate the shift
S=E_12+E_23 (up to the documented sine convention), and its adjoint. Then
S-squared=E_13, I-S-dagger S=E_11, and I-S S-dagger=E_33. Multiplication by
S and S-dagger produces the remaining matrix units. Their complex associative
algebra is therefore **all M3(C)**, of complex dimension nine, not a
five-dimensional algebra. Complex scalar combinations are essential to this
statement. Real-coefficient products of the real-source matrices retain an
antiunitary symmetry (outer-mode reflection followed by conjugation); their
real algebra is conjugate to M3(R), not the eighteen-real-dimensional M3(C).

These products are not single projected local scalar sources:

\[
T_fT_h=U^\dagger\operatorname{diag}(f)P\operatorname{diag}(h)U
\ne U^\dagger\operatorname{diag}(fh)U
\quad\text{in general}.
\]

Thus the linear rank obstruction does not rule out nonlocal operations,
composites, or interaction-dressed effective operators. Those would require a
separate derivation, including their coefficients and validity scale, rather
than being silently substituted here.

Even multiplication itself need not preserve the window. Its closure residual
is diag(f)U-U T_f. For real f and p>=5 it vanishes exactly only for constant
f: applying f to the k=0 column first restricts its Fourier support to K;
applying it to both outer columns then eliminates the +/-1 components.
Numerically compute this residual directly, not by subtracting squared norms
of nearly equal large quantities. Arithmetic consistency with zero is not a
new proof of exact closure.

## 6. What has and has not been matched

| Question | Result of this pass |
|---|---|
| Rank-three vector-space projection | Explicit Fourier isometry |
| Free dynamics | Exact invariant singlet-plus-doublet window |
| Interaction | Exact Galerkin energy; generic nonclosure; conservative finite-time error bound |
| Direct real local source map | Five directions, not the complete nine-direction monopole map |
| Generated complex matrix algebra | Full M3(C); no universal operator-algebra obstruction |
| Statistics | A canonical bosonic mode transformation preserves CCR, not CAR |
| Physical chiral source | Not derived; neutral number density is not automatically a Lorentz-scalar Yukawa bilinear |
| Flux, families and physical scales | Not selected |

The bosonic statement is an algebraic one: for canonical site operators,
b_k=sum_x U_xk-star a_x obeys [b_k,b_l-dagger]=delta_kl. It does not use the
finite-occupation cutoff toy to assert exact CCR, and it does not exclude
fermionic emergence through a separately justified nonlocal/composite
mechanism. It only excludes identifying this linear bosonic transform with
the independently stipulated CAR construction.

Likewise, ring Fourier momentum is not the global phase charge Q. If the
additional Z_p neutrality assumption is imposed, a bare Q=1 excitation is
not itself a physical state for p>=5. A neutral number-conserving bilinear is
a different object; neutrality alone does not identify it with a monopole
section or a chiral Standard Model scalar source.

The scalar l=1 space on a round sphere and the monopole zero-mode triplet can
both carry spin-one rotational representations without having identical
multiplication maps or bundles. The q=3 Dirac twist is the line bundle
O(3), whose positive zero-mode components lie in K^(1/2) tensor O(3)=O(2).
This notation is not the orthogonal group O(3). Neither bundle is derived by
the ring Fourier projection.

## 7. Executable scope and verification

- `bpr/substrate_triplet_projection.py`: analytic projection and source diagnostics.
- `tests/test_substrate_triplet_projection.py`: independent finite-ring, Fourier,
  target-overlap and numerical regression checks.
- `python3 scripts/demo_substrate_triplet_projection.py`: frozen stdout report;
  add `--json` for full structured output.

The demonstration freezes p=7, C=1, g=0.5 and chosen norm N=1 before evaluating
single-mode and multimode controls. At these inputs, W=0.7530203962825329
and Delta=1.6920214716300959. The uniform single-mode energy is -55/28,
while the equal-weight adjacent-mode energy is -2+W/2+3/56, approximately
-1.569918373287305. The adjacent-mode leakage is 1/(28 sqrt(2)), approximately
0.025253813613805267, and the relative finite-time error bound at t=0.1 is
approximately 0.02311917753261188. These are illustrative inputs and outputs, not
physical calibration. No empirical mass or mixing targets are used. Physical
masses and mixing remain null.

Numerical diagnostics separate positive shifted free scales and energy terms
from rounded totals. A large unshifted baseline can hide a small correction
without making the exact spectrum degenerate. Likewise a leakage residual
consistent with zero at arithmetic resolution is not a proof of closure;
structural zero controls and unresolved cancellations have separate statuses.
Finite-time bounds report the trivial norm cap as potentially vacuous and
retain null relative error for the vacuum. Nonfinite or unresolved intermediate
computations are rejected rather than silently treated as exact zero. In
particular, a nonzero cubic monomial lost before coupling rescaling must be
rejected even if its final mathematical contribution would be representable.
Extended NumPy precision is not assumed to exceed float64 on every platform.
These are conservative restrictions of floating-point computation, not new
restrictions on the analytic theorems.

Independent mathematical review verified the projection, quartic, leakage
aliases and gap against direct finite-ring arrays at p=5,6,7,11 within
4.5e-16. All nine target overlaps agreed with independent sphere quadrature
within 2.3e-15. The finite-time proof, local-source closure theorem and complex
algebra distinction were also checked. A separate implementation review found
numerical overflow/underflow defects and verified their fixes: the uncapped
huge-p error bound now remains finite, free g=0 controls skip irrelevant cubic
operations, tiny leakage uses scale-safe norms, mixed-scale monomial underflow
is explicitly rejected, and Gram ranks are stable under uniform rescaling.
Final independent checks ran with warnings treated as errors and left no
outstanding findings in the reviewed scope.

The new suite passed **96 tests with warnings treated as errors**. The final
combined selection, including the previous 390 tests and 25 relevant
condensate/neutrality tests, passed **511 tests**, with one intentional
phase-scan deselection and the existing unregistered `slow` marker warning.
Eight independent algebra-check groups and text/strict-JSON modes of the
three previous demos also passed. The new demo passed both formats from
outside the repository. No full-repository test or empirical-validation claim
is made.

The completed scientific target is an actual restricted Hamiltonian, a scoped
source-map obstruction, and a finite-time classical control statement. It is
not completion of the missing microscopic flavor bridge. No new interaction
is added to overcome the obstruction in this pass.

### Interaction-response follow-up

The [finite-time nonlinear response calculation](substrate_nonlinear_response_2026-09-12.md)
integrates the actual cubic forcing, including retained and discarded modes.
It supplies a first-order local-density correction with explicit second-order
remainder bounds. Retained and omitted pieces cancel at the initial density
slope; an omitted contribution alone is not the complete response. The result
is state dependent and does not replace this note's direct linear source-map
obstruction with a universal dressed matrix or a physical flavor prediction.
