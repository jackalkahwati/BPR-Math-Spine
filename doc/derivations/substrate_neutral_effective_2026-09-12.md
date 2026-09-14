# Consistently dressed neutral response

## Frozen scope and controls

2026-09-12. Module 1 of the six-module campaign. This note freezes the calculation before implementation; verification results will be appended after execution.

Use the unchanged stipulated complete Bose ring H=gD+C T at N=L=q, C>0, g>0. The complete occupation basis retains all virtual occupations. Full calculations remain L=3..6 under the inherited dimension-512 cap. Zero coupling g=0 makes this perturbation undefined, not the exact response zero. The frozen illustrative cases are L=5,C=1,g=40 and .7 at m=1 and m=0. Additional order checks use L=3,4 and lambda=1/40,1/80,1/160, without fitted coefficients or physical calibration.

## Transformation convention

Let lambda=C/g and let pinching P_D retain equal-D matrix elements. Write Td=P_D(T), To=T-Td. Choose U=exp(S), Htilde=U H U†, S=lambda S1+lambda² S2. The generators are zero within equal-D blocks:

S1_ab=T_ab/(D_a-D_b) for D_a!=D_b.

Then [S1,D]=-To. Define F=[S1,Td]+[S1,To]/2, K=P_D(F), and S2_ab=F_ab/(D_a-D_b) for unequal D. Both generators are anti-Hermitian, [S2,D]=-(F-K), and

Htilde/g = D+lambda Td+lambda² K+O(lambda³).

In particular, the D=1 correction is

K1=P1 T Q1 (1-D_Q1)^(-1) Q1 T P1.

The D=0 intermediate contribution has positive denominator. All reachable higher-D channels have negative denominators; the sum is not assumed negative semidefinite. The unique D=0 state Omega has K0=-||T Omega||²=-4L, so E0[2]=-4LC²/g. Approximate neutral gaps use H1[2]-E0[2], not H1[2] alone. Subtracting this within-sector reference does not alter cross-number population comparisons.

## Observable and perturbative order

For rho_m=L^(-1/2) sum exp(-ikx)(n_x-1), define

rho1=[S1,rho],
rho2=[S2,rho]+[S1,[S1,rho]]/2.

The transformed observable through second order is rho+lambda rho1+lambda² rho2. For fixed finite L in the isolated ground regime, U G=Omega+O(lambda³); Omega is not the exact transformed interacting ground. Its transition source through second order is therefore z=lambda rho1 Omega+lambda² rho2 Omega. The observable-conjugation remainder below does not by itself bound the source error on the true transformed ground; that also requires a ground-vector remainder. Projecting to P1 recovers the inherited first-order endpoint source. Retain other D blocks of the second-order source as diagnostics rather than asserting all density response is in P1.

At exactly m=0 the full fixed-N density operator and every source coefficient vanish. For other modes rho is not Hermitian in general; use conjugate transposes correctly.

The polynomial source norm is lambda²||z1||²+2lambda³ Re(z1†z2) through cubic order. The positive squared truncated source additionally contains lambda⁴||z2||², but misses interference with the uncomputed z3. It is NOT a complete fourth-order prediction. Diagonalizing the truncated Hamiltonian and using its lambda-dependent eigenvectors is a partially resummed effective-model diagnostic, not a strict line-weight Taylor polynomial.

## Finite remainder construction to verify

For any anti-Hermitian S, unitary conjugation is norm preserving. Taylor's integral remainder gives

||exp(S) A exp(-S)-A-[S,A]-[S,[S,A]]/2|| <= (4/3)||S||³||A||.

The first-order version has remainder at most 2||S||²||A||. Apply the second-order expansion to D and first-order expansion to lambda T. In addition to these remainders, retain bounds on the omitted explicit lambda³ and lambda⁴ cross terms in S=lambda S1+lambda² S2. With s1=||S1||,s2=||S2||,d=||D||,t=||T||,s=lambda*s1+lambda²*s2, a sufficient dimensionless Hamiltonian remainder is

RH=(4/3)s³d+2lambda*s²t+4lambda³s1*s2*d+2lambda⁴s2²d+2lambda³s2*t.

For the observable, a sufficient norm remainder is

Rrho=||rho||[(4/3)s³+4lambda³s1*s2+2lambda⁴s2²].

To see the coefficients, use ||[A,B]||<=2||A||||B||. The omitted half-double-commutator cross terms with D contribute at most 4lambda³s1s2d+2lambda⁴s2²d. The omitted lambda[S,T] term is lambda³[S2,T], bounded by 2lambda³s2t. The third-order Taylor remainder for conjugation by S has coefficient (2s)³/3!=4s³/3 because all conjugations in its integral remainder are unitary. The same cross-term argument applied to rho proves Rrho. No exponential in s is required.

These are conservative exact-arithmetic inequalities, not certified floating-point calculations. They remain algebraically valid outside the useful perturbative regime but can be too large to distinguish any spectrum. Global sorted eigenvalues of the full block-diagonal truncated matrix differ from the full H by at most g*RH by Weyl's inequality; this does not independently identify a chosen excited block or certify line weights. If an effective D=0 ground and a D=1 cluster are separated from all other effective blocks by more than the relevant remainder intervals, that identification can be made explicitly. Otherwise report the cluster assignment unavailable. Dynamics after the same U differ in operator norm by at most |time|*g*RH (hbar=1), capped by the trivial bound 2.

### Ground-vector and full-source bound

Let A=H[2]/g, e0=lambda²K0, and Delta0=min spec(Q0 A Q0)-e0. Require Delta0>2RH. Weyl then identifies a unique exact transformed ground with energy e satisfying |e-e0|<=RH. Writing psi=UG=alpha Omega+chi with alpha>=0 and chi orthogonal to Omega, the equation Q0(A-e)chi=-Q0 E psi, ||E||<=RH, gives

||chi|| <= b=RH/(Delta0-RH),
||UG-Omega|| <= sqrt(2)*b.

Since U rho U† has norm ||rho|| even at non-Hermitian nonzero momentum,

||U rho G-z|| <= Rrho + ||rho||*sqrt(2)*RH/(Delta0-RH).

This is a full transformed-source norm certificate including vacuum error. It is unavailable if its ground-isolation hypothesis fails. It is not an individual-residue bound. The phase convention on G matters in the vector comparison but cancels in all spectral weights. At zero momentum the source bound is structurally zero when the ground gate is available; rho0=0 itself does not need that gate.

A Hamiltonian remainder alone gives no individual-residue bound near degeneracies. Compare grouped spectral measures and sum rules numerically; do not label individual weight agreement certified. Apparent isolated finite-ring lines do not establish a thermodynamic bound particle, even when numerical energy error is small.

## Acceptance gates

Independent tests must check generator anti-Hermiticity, BCH cancellations, signed virtual channels, ground coefficient, translation covariance, inherited leading source, aliases and opposite momenta, exact zero modes, source weight orders, full sorted spectral residuals, unitary dynamics/remainder, and strong/weak availability. Compare complete small-ring spectra and density sum rules with independent tensor or occupation oracles. Resource caps and nonfinite/subnormal arithmetic must not silently generate physical zeros. Demos write text or strict JSON only to stdout from arbitrary working directories.

## Independent analytic witnesses

A direct Rayleigh–Schrödinger vacuum expansion provides a check independent of observable commutators. Write G=Omega+lambda p1+lambda² p2+O(lambda³), with p1=-T Omega and

p2=D_Q0^(-1) Q0 T² Omega -2L Omega.

Then p1+S1 Omega=0 and p2+S1 p1+S2 Omega+S1² Omega/2=0. Consequently the second source coefficient also equals rho p2+S1 rho p1. This tests both ground normalization and excited-state dressing.

For L=3,m=1, the exact source coefficients obey ||z1||²=12 and 2Re(z1†z2)=72. For L=4,m=1 these values are 8 and 0. The D=3 component of z2 has norm sqrt(3/2) and sqrt(2/3), respectively. Thus the second-order source is not wholly confined to D=1, and cubic weight corrections do not vanish on every ring. Higher-band absolute weight first appears at order lambda^4; calculating its leading coefficient does not supply all fourth-order weight in the first band.

The D=1 second-order coefficient is indefinite already on L=3, with extreme eigenvalues -3 and 9; on L=4 the extrema are -11 and 1. Any implementation enforcing an everywhere negative correction would discard the virtual vacuum channel incorrectly.

## Independent mathematical review

Independent review verified the SW signs, both norm-remainder constants, vacuum expansion, source coefficients, and the full-source bound including ground-vector error. It required the explicit clarification that Omega is transformed ground only through second order; that clarification and the additional ground/source certificate are included above. The reviewer independently constructed complete occupation matrices without importing the new implementation and reproduced the analytic witnesses.

At the frozen L=5,C=1,g=40 point, independent numerical evaluation gave RH approximately 0.15573551, actual dimensionless Hamiltonian remainder 0.00140928, effective ground gap 0.88591118, and nearest effective D=1/other spectral separation 0.80064948. Both exceed 2RH. At g=.7, RH is approximately 1.1450e6 and block identification is unavailable. These are numerical checks without certified roundoff, not empirical observations; the substantial conservative error bars must be retained. The inherited neutral/fermionization/population/continuum regression passed 403 tests with warnings as errors. All eight independent algebra-check groups passed. The first new focused suite passed 41 tests with warnings as errors, including the stdout demo in text and strict JSON from an empty temporary directory. Expanded report/dynamics checks passed 51 tests. Independent implementation review then found two avoidable extreme-scale errors: forming 2/energy_bound before a capped dynamics comparison could underflow, and multiplying a matrix by the reciprocal of a large spectral scale could underflow even when direct division was representable. Both were repaired: exact integer-ratio product comparison handles saturation, and direct guarded matrix division handles eigensystem scaling. Twelve targeted report/dynamics/extreme-scale checks passed, including a full strict-JSON report at C=1e306,g=4e307. The analytic inequalities are unchanged. The full post-fix focused suite passed 53 tests with warnings as errors. Independent implementation rechecking confirmed both scale fixes, full strict-JSON high-scale reporting, source-block weight partition, bounded-span degeneracy grouping, exact/effective sum rules and isolated P1 gap enclosures, with no remaining verified implementation defects. The pre-fix combined run passed 1,077 tests with one intentional phase-anchor deselection and the existing unregistered slow-marker warning; the full post-fix combined run passed **1,079 tests**, with the same one intentional deselection and existing warning. All ten scientific demos passed both text and strict JSON from an empty temporary directory with no stderr or output files. Python 3.8 grammar parsing and compilation checks passed.

A separate reviewer probe at L=3,C=1,g=2e6 found the numerical sorted spectral difference (~1.86e-9) larger than the analytic truncation bound (~2.38e-10), because eigensolver roundoff is not included in that bound. The report must not claim that every computed residual is mathematically enclosed at all accepted floating-point inputs. Analytic approximation bounds and numerical error remain separate throughout.

The frozen strong-case implementation reports exact total density weight 0.0034914215631285314 versus squared-truncated weight 0.0035059179689863433. The observable norm bound is about 0.03038297 and the full-source bound about 0.7048493: the latter is much larger than the source itself. The global sorted energy error was approximately 0.02899485 against bound 6.22942031. In the frozen weak case, the squared-truncated weight is about 555.0861 versus exact weight 0.7369744, with no cluster/source certificate. These results illustrate failure of the truncation outside strong coupling, not an improved weak-coupling prediction. Neither individual residues nor a bound particle are certified.

During test development, a proposed ratio check on the omitted fourth-order total-weight residual was rejected: independent exact diagonalization at g=40,80,160 gave successive ratios approximately 25.75 and 22.33 rather than a generic factor of 16. Such a ratio assumes a nonzero dominant quartic coefficient and a sufficiently asymptotic regime, neither established for these controls. The test now uses the proved full-source norm bound, the difference-of-squares inequality and the explicit partial quartic term instead. Controls were not changed, and no coefficient or physical parameter was fitted.

## Scientific boundary

This calculation repairs the internal perturbative consistency of a stipulated neutral Bose model. It does not derive physical fermions, a particle pole, observed masses/mixing, spacetime, gravity or empirical support for BPR. Frozen controls cannot be adjusted retrospectively to conceal uninformative error bounds. Verification status is recorded above; publication is tracked in the [campaign ledger](scientific_campaign_2026-09-12.md).

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m pytest -q -W error tests/test_substrate_neutral_effective.py
python3 scripts/demo_substrate_neutral_effective.py
python3 scripts/demo_substrate_neutral_effective.py --json
```

All supported full-matrix cases use the complete finite occupation basis. The Python implementation does not make a rigorous roundoff claim, run a physical benchmark or write demonstration output files.
