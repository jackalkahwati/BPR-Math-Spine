# Interaction-generated density response of the existing ring

**2026-09-12. Scope: classical finite-time DNLS response, not physical flavor matching.**

The [three-mode projection](substrate_triplet_projection_2026-09-12.md) gives an
exact restricted Hamiltonian but discards part of the nonlinear vector field.
This follow-up integrates that forcing and computes its contribution to a local
density measurement. Both retained and omitted corrections are necessary.
There is no added source potential, chemical potential, interaction or statistics
prescription. The initial state and measured observable remain supplied inputs.

## 1. Dynamics and conventions

Keep the defocusing ring with integer p>=5, C>0 and g>=0:

\[
i\dot\psi_x=-C(\psi_{x+1}+\psi_{x-1})+g|\psi_x|^2\psi_x,
\qquad N=\|\psi\|_2^2.
\]

For this calculation use a global rotating frame with free operator
\(h=-C(T+T^\dagger)+2CI\). If psi solves the original equation,
\(e^{-2iCt}\psi\) solves the shifted equation. This changes no density.
Normalized Fourier vectors and shifted energies are

\[
u_{xk}=p^{-1/2}e^{2\pi i xk/p},\qquad
\epsilon_k=4C\sin^2(\pi k/p).
\]

Initially \(\psi(0)=Uz\), where U contains modes K=(-1,0,1) in that order.
Let P=UU-dagger, Q=I-P and \(v(t)=e^{-iht}Uz\). Norm N is conserved, not
selected. With hbar=1, C and gN have inverse-time units. Time can be negative.

## 2. Exact first derivative in interaction strength

Duhamel's formula in the shifted frame is

\[
\psi(t)=v(t)-ig\int_0^t e^{-ih(t-s)}F(\psi(s))\,ds,
\qquad F(w)=|w|^2w.
\]

Define the derivative at g=0, not a new interacting field equation:

\[
\psi_1(t)=-i\int_0^t e^{-ih(t-s)}F(v(s))\,ds,
\qquad \psi(t)=v(t)+g\psi_1(t)+R(t).
\]

The normalized full Fourier coefficients are

\[
\boxed{\widehat{\psi}_{1,k}(t)=
-\frac{i}{p}e^{-i\epsilon_k t}
\sum_{\substack{i,j,l\in K\\i-j+l\equiv k\pmod p}}
z_i z_j^*z_l\,I_t(\epsilon_i-\epsilon_j+\epsilon_l-\epsilon_k),}
\tag{1}
\]

\[
I_t(\Omega)=\int_0^t e^{-i\Omega s}\,ds
=t\,e^{-i\Omega t/2}\operatorname{sinc}(\Omega t/2),\qquad
\operatorname{sinc}(x)=\sin(x)/x,\quad\operatorname{sinc}(0)=1.
\]

There are only 27 input monomials. The factor 1/p follows from three factors
of 1/sqrt(p) and the output Fourier transform. NumPy's sinc uses sin(pi x)/(pi x),
so its argument is Omega*t/(2*pi). There is no inverse-detuning pole.

**Integrate each monomial before aggregating output residues.** At p=5,
k=3 aliases to -2 and k=-3 aliases to 2, but aliased monomials can have
different detunings. At p=6 the two outer outputs alias to the same residue.
Summing instantaneous leakage first and applying a single time kernel is
therefore not a valid general method. No output aliases into K at p>=5.

Put W=epsilon1, E=epsilon2. Retained detunings are 0 and +/-2W. Omitted
ones are 2W-E, -E and W-epsilon3. All omitted detunings are negative for
p>=5. Useful cancellation-free identities are

\[
2W-E=-2W\cos(2\pi/p),\qquad
W-\epsilon_3=-4C\sin(4\pi/p)\sin(2\pi/p).
\]

Exact resonances follow from channel identities, not rounding small detunings
to zero. The kernel remains continuous near resonance, and resonant retained
terms grow linearly in t. No long-time perturbative accuracy is inferred.

Because P commutes with h, P psi1 is exactly the first derivative of the
lifted Galerkin solution. Q psi1 is the leading field contribution that this
restriction misses. At t=0, \(\dot\psi_1=-iF(Uz)\); projecting Q reproduces
the previous leakage vector divided by g, with the factor -i.

## 3. Finite-time error certificates

Let T=|t|. Both the exact and free trajectories have norm sqrt(N), and

\[
\|F(w)\|_2\le N^{3/2},\qquad
\|F(w)-F(y)\|_2\le3N\|w-y\|_2
\]

on that norm ball. The latter follows from the componentwise real Jacobian
and a segment integral through the convex ball. Unitarity in Duhamel's formula
gives, for \(\delta=\psi-v\),

\[
\|\delta(t)\|_2\le gN^{3/2}T,
\]

\[
\|R(t)\|_2\le g\int_0^T 3N\,gN^{3/2}s\,ds
=\boxed{\tfrac32g^2N^{5/2}T^2}.\tag{2}
\]

The relative bound for N>0 is \(\tfrac32(gNT)^2\). For N=0 the absolute
error is zero and relative error is undefined. At g=0 or t=0 the approximation
is exact. These are conservative finite-time inequalities, not spectral-gap,
adiabatic or quantum many-body theorems. They need no nonresonance assumption.

For a real site weight f, write M=diag(f). It is an observable, **not an
added Hamiltonian source**. The first-order observable approximation is

\[
O^{(1)}_f=v^\dagger Mv+gL_f,
\qquad L_f=2\operatorname{Re}(v^\dagger M\psi_1)=L_P+L_Q.
\tag{3}
\]

The exact remainder identity is

\[
O_f-O^{(1)}_f=2\operatorname{Re}(v^\dagger MR)+\delta^\dagger M\delta.
\]

Consequently

\[
\boxed{|O_f-O^{(1)}_f|\le4\|f\|_\infty g^2N^3T^2}.\tag{4}
\]

The same proof applies to Galerkin motion: insert the contraction P into the
nonlinearity and use its conserved norm. Thus the difference of **exact full
and exact Galerkin observables** is g L_Q with remainder bounded by
\(8\|f\|_\infty g^2N^3T^2\), the sum of two trajectory bounds. Equation (4)
alone is not a bound for that difference.

Do not evaluate the full square of v+g psi1 and call it (3): that adds the
quadratic term \(g^2\psi_1^\dagger M\psi_1\). Nor should the approximate field
be renormalized or negative approximate densities clipped. In particular,
\(2\operatorname{Re}(v^\dagger\psi_1)=0\), but the truncated field has norm
squared \(N+g^2\|\psi_1\|^2\), not N. Constant f has zero first-order response.

## 4. Explicit retained-versus-omitted witness

Choose the real initial triplet and measurement

\[
z=(0,s,s),\quad s=\sqrt{N/2},\qquad f_x=\sin(4\pi x/p).
\]

Define A=N^2/(4p), E=epsilon2, W=epsilon1, D=2W-E. Only k=2 is omitted
for this initial state. Equation (1) and discrete Fourier orthogonality give

\[
\boxed{L_Q=A\,t\,\operatorname{sinc}(Dt/2)
\cos((E+2W)t/2),\qquad
L_P=-A\frac{\sin(2Wt)}{2W}.}\tag{5}
\]

At short time L_Q=At+O(t^3), whereas L_P=-At+O(t^3). Their initial slopes
cancel. The onsite cubic changes phase rather than density instantaneously:
\(2\operatorname{Re}[\psi_x^*(-i|\psi_x|^2\psi_x)]=0\) at every site.
The complete response begins at cubic order in time:

\[
\boxed{L_f=-\frac{AE(E+2W)}6t^3+O(t^5).}\tag{6}
\]

The omitted piece is a genuine contribution missing from Galerkin, but it is
not the full response. A pure Fourier mode has only a nonlinear common phase,
so its density response vanishes and Q psi1=0.

The demonstration freezes p=7, C=1, g=0.5, chosen N=1 and t=0.1. The
coarse error bounds exceed the response signals at these inputs. The nonzero
derivative in g is an analytic result; its finite-g sign is **not certified by
these bounds** at the frozen point. Numerical integration can cross-check the
expansion without turning solver tolerances into rigorous global-error bounds.
This does not mean that no finite-g sign can be certified: for the
interacting-minus-free observable correction, at fixed nonzero L_f and T,
a sufficient condition is
\(0<g<|L_f|/(4\|f\|_\infty N^3T^2)\). For the full-versus-Galerkin difference,
replace L_f by L_Q and 4 by 8. These are mathematical small-coupling conditions,
not selected physical couplings or a reason to tune the frozen example.

## 5. Why this is not a completed flavor source map

L_f is quartic in z; under z -> a z it scales as |a|^4. A universal fixed
three-by-three quadratic source matrix instead scales as |a|^2. This response
therefore cannot simply be relabeled as such a matrix. A state-specific
representation would not derive universal couplings or select that state.

Site conjugation is a time-reversal symmetry: it relates t to -t. A static
antiunitary restriction on source matrices is not a fixed-time rank theorem
for this nonlinear response. Conversely, a nonzero dynamic correction does
not remove the earlier direct local-source rank-five obstruction. Its scope
remains the linear compression U-dagger diag(f) U.

| Missing input | Status after this calculation |
|---|---|
| Interaction-induced discarded-mode response | Derived to first order with finite-time remainder bounds |
| Initial state, norm and measurement | Specified inputs, not dynamically selected |
| Universal monopole source geometry | Not matched |
| Fermions and physical chirality | Not derived |
| Physical couplings, masses and mixing | Not predicted |
| Gravity and unified particle dynamics | Not supplied by this classical ring calculation |

A next microscopic test would need to address an unresolved category rather
than merely compute another adjustable response. For example, one can ask
whether a specified low-energy composite sector of the existing bosonic
substrate supports the required statistics and local operators. That requires
an explicit construction, locality and exchange tests, and a controlled limit;
the present linear Fourier transform and classical response supply none of
those by themselves. It is a research question, not an existence claim.

## 6. Executable scope and verification

- `bpr/substrate_nonlinear_response.py`: analytic finite-time response and bounds.
- `tests/test_substrate_nonlinear_response.py`: independent quadrature, short-time
  integration, symmetry, alias, numerical and regression checks.
- `python3 scripts/demo_substrate_nonlinear_response.py`: frozen stdout report;
  add `--json` for strict structured output.

The numerical domain is narrower than the mathematical domain. The implementation
uses normal binary64 intermediates and rejects subnormal or nonfinite inputs,
operation underflow, unresolved phases and ring sizes above 2^53. It does not
assume extended NumPy precision is wider on the host platform. Such conservative
rejections are numerical limitations, not failure of the analytic theorem.
Arithmetic warning scales are diagnostics, not rigorous roundoff certificates;
the analytic truncation bounds do not include arithmetic or solver error.
Raw arrays are not guaranteed accurate merely because evaluation returned:
consumers must also inspect arithmetic resolution, especially at long times.
Structural zeros remain distinct from unresolved cancellation. Physical masses
and mixing remain null.

Independent mathematical review found no errors in the stated identities or
bounds. Site-space quadrature at p=5,6,7,11,19, including negative times, agreed
with (1) within 2.82e-15 in field norm. The first-order norm response residual
was at most 1.13e-16; the retained and omitted witness terms separately agreed
within 7.64e-16. Independent site-space Taylor recurrence reproduced the cubic
coefficient in (6) within 8.33e-17. These checks did not import the new response
implementation.

At the frozen inputs, g L_f is approximately -2.8633311642e-5 and g L_Q is
1.7503381527e-3. The respective analytic observable error bounds are about
0.00974928 and 0.01949856, using the discrete maximum of the sine weight.
Neither signal's finite-g sign is resolved by its bound.

The unchanged bounded regression selection passed **511 tests**, with one
intentional phase-scan deselection and one existing unregistered `slow` marker
warning. Eight algebra-check groups and text/strict-JSON modes of all four
previous demos passed; demos were invoked from outside the repository.
The final new focused suite passed **112 tests with warnings treated as errors**,
including independent site-space quadrature, full and Galerkin ODE checks,
aliases, symmetries, structural controls and numerical-domain rejection.
The final combined selection passed **623 tests**, with the same intentional
deselection and existing marker warning. The new demo also passed text and
strict JSON checks from an empty temporary directory without writing files.

Independent implementation review identified and verified a fix for a long-time
phase-rounding defect: a magnitude-only warning could label a wrong-sign,
cancellation-sensitive response as resolved. The warning now includes phase
magnitude and uncancelled, unoscillated channel scales. The p=7, z=(1,2,3),
f=cos(2pi x/p), t=417198883.39213 reproduction is marked unresolved with null
resolved value; the raw binary64 derivative differs from an independent
60-digit evaluation by about 42.02. The deliberately conservative warning is
about 2.01e6, not a claim to have corrected that raw value. Two additional
long-time checks also remained unresolved, while the frozen short-time
witness retained its numerical nonzero statuses. Independent rechecking with
warnings treated as errors left no substantive findings in the reviewed scope.
No full-repository test, fitting, benchmark target access, production simulation,
remote execution or empirical validation is claimed.

Reproduce the focused checks and frozen demonstration locally:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m pytest -q -W error tests/test_substrate_nonlinear_response.py
python3 scripts/demo_substrate_nonlinear_response.py
python3 scripts/demo_substrate_nonlinear_response.py --json
```

### Quantum statistics follow-up

The [conditional fermionization diagnostic](substrate_fermionization_2026-09-12.md)
examines a declared extension of the existing bosonic quantum Hamiltonian,
not a quantization uniquely derived from this classical response. It supplies
an exact hard-core string dictionary and finite-repulsion Schur bounds, while
retaining parity-dependent ring boundaries and the assumed neutrality rule.
One-dimensional fermionic variables are not physical chiral matter, and their
strings do not remove charge. This follow-up leaves the current classical
calculation unchanged.
