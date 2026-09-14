# Endpoint charge space at the three-particle exclusion boundary

2026-09-13. Status: charge-level conclusions independently cleared under the stated
ambient finite-energy hypothesis; independent saved-document/hash audit cleared.
This note concerns charges of physically supported finite-local-energy states.
It does not prove equality of whole-state minimal/maximal domains or lattice
selection of the Friedrichs realization.

## 1. Geometry, norms and theorem scope

Retain the same R,kappa>0 and finite s!=0, the three-particle allowed region
Omega_R={max_ij|xi-xj|>R}, and the fixed matching localization chi_e used in the
continuum construction. For a selected pair use r=xi-xj, c=(xi+xj)/2 and y=xk.
Introduce tangential coordinates

    v=c-y, w=(2c+y)/3,
    c=w+v/3, y=w-2v/3,
    D={(v,w):|v|>R}, d(v)=|v|-R.

The absolute Jacobian of this tangential transformation is1. Its fixed linear
distortion gives equivalent ambient Sobolev H^(1/2) norms. We use Euclidean
product (v,w) coordinates to compute the endpoint weight; estimates transfer
to (c,y) with fixed equivalence constants. Contact has r=0 and permitted
spectator separation |v|>R.

Write H^(1/2)(D) for the intrinsic L² plus Gagliardo seminorm space on D, not
a presumed zero-extension space. The claims to prove are

    closure_{H^(1/2)(R^6)} C_c^infinity(D)
      ={Q in H^(1/2)(R^6): Q=0 a.e. on D^c},            (1)

and, for intrinsic q in H^(1/2)(D),

    E0 q in H^(1/2)(R^6)
      iff integral_D |q(v,w)|²/d(v) dv dw <infinity.    (2)

Here E0 is zero extension. For fixed R the norm equivalence is

    ||E0 q||²_(H1/2(R6)) asymp_R
      ||q||²_(H1/2(D))+integral_D |q|²/d.               (3)

These are supported-space and weighted-integrability results, not classical
traces at the critical Sobolev exponent1/2.

## 2. Exact cross-boundary seminorm and weight

Use the unnormalized Gagliardo seminorm
[f]²=integral integral |f(z)-f(z')|²/|z-z'|^7 dz dz' in dimension6.
Splitting D and D^c gives the intrinsic seminorm plus twice the cross term.
The COM coordinate of the complement is unrestricted, and

    integral_R3 dh/(a²+|h|²)^(7/2)=8pi/(15a^4).

For example radial substitution h=a t reduces the integral to
4pi a^(-4) integral_0^infinity t²/(1+t²)^(7/2)dt;
the last integral is2/15 by t=tan(theta). Therefore exactly

    [E0q]²=[q]²_D+(16pi/15)integral_D |q(v,w)|² K_R(v) dv dw,
    K_R(v)=integral_{|v'|<=R}|v-v'|^(-4) dv'.           (4)

Let rho=|v|>R. Angular integration gives

    K_R(rho)=4pi integral_0^R t²/(rho²-t²)² dt
      =2pi R/(rho²-R²)-(pi/rho)log[(rho+R)/(rho-R)].     (5)

It follows that d K_R(R+d)->pi as d down to0. The singular endpoint power is
thus precisely d^(-1), not a guessed trace exponent.

Bounds avoiding reliance on the antiderivative suffice for(2)–(3). Since the
integration points have separation at least d,

    K_R(v)<=integral_{|h|>=d}|h|^(-4)dh=4pi/d.

For0<d<=R, the ball of radius d/2 centered at (R-d/2)v/|v| is contained in
{|v'|<=R}. Its points have distance at most2d from v. Its volume is pi d³/6,
so

    K_R(v)>=pi/(96d),        0<d<=R.                    (6)

At d>R, the weighted integral with1/d is bounded by R^(-1)||q||²_2. Consequently
(4)–(6), with this L² contribution, prove(2)–(3). K_R is NOT globally comparable
to1/d: at infinity it decays like4pi R³/(3rho^4). The L² term is essential to the
global equivalence. No compactness of the unbounded COM direction was assumed;
that coordinate was integrated exactly.

## 3. Supported ambient charge equals zero-extension closure

The right-hand side of(1) is closed in ambient H^(1/2), since convergence there
implies L² convergence and preserves vanishing on D^c. It contains C_c^infinity(D).
For the reverse inclusion take supported Q in H^(1/2)(R^6). For t>1 define

    Q_t(v,w)=Q(v/t,w).

Its support satisfies |v|>=tR almost everywhere, giving a margin(t-1)R from
the original boundary. Its Fourier norm obeys, up to the same fixed Fourier
normalization on both sides,

    ||Q_t||²_(H1/2)
      =t³ integral (1+t^(-2)|xi|²+|eta|²)^(1/2)
                       |Qhat(xi,eta)|² dxi deta.

The dilation maps are uniformly bounded for1<=t<=2. Strong continuity as t->1
follows first for Schwartz functions and then by their density and uniform
boundedness on H^(1/2). Hence Q_t->Q in ambient H^(1/2).

At fixed t>1 multiply by smooth large-radius cutoffs. The products converge in
H^(1/2) and do not reduce the boundary margin. This standard cutoff fact can
be seen from uniform H^0 and H^1 multiplier bounds, interpolation to H^(1/2),
and convergence on smooth compact functions followed by density. Mollify each
compact product at radius less than half its remaining margin. The result is
in C_c^infinity(D), and mollification converges in ambient H^(1/2). A diagonal
choice as t down to1 proves(1). This is not an appeal to a critical boundary
trace theorem and does not assert intrinsic H^(1/2) convergence alone is enough.

## 4. Extract the charge from a physical zero extension

Let U=tilde(chi_e u) belong to the ambient pair-form domain V_e. Assume U is
physically supported in closure(Omega_R), as a zero extension from the allowed
space. Membership in the minimal Friedrichs closure is NOT assumed. Thus this
argument also applies to a state in the prospective maximal local-energy class.

The established momentum-dependent decomposition, for lambda>E_s, is

    U=phi+G_M Q,
    phi in H¹(R^9), Q in H^(1/2)(R^6),
    (G_M Q)^hat(r,p,k)=exp(-m(p,k)|r|)Qhat(p,k)/(4pi|r|),
    m(p,k)=sqrt((lambda+kappa|p|²/2+kappa|k|²)/(2kappa)). (7)

The full H¹ assertion for this REGULAR PART follows from its weighted integral
in the previous continuum form formula. It is not a claim about a fixed-mu
regular part or a global H¹ assertion for the singular whole U. M is the
positive self-adjoint tangential Fourier multiplier with this symbol.

Choose nonnegative eta in C_c^infinity(B_1(0)), not identically zero, and set

    S_epsilon U(z)=epsilon^(-2)integral eta(r/epsilon)U(r,z) dr,
    C_eta=integral eta(rho)/(4pi|rho|) drho>0.

We claim S_epsilon U->C_eta Q in tangential L². For phi use Cauchy-Schwarz:

    ||S_epsilon phi||_2
      <= C epsilon^(1/2)||phi/|r|||_2
      <= C' epsilon^(1/2)||grad_r phi||_2 ->0.           (8)

The last step is the normal3D Hardy inequality, applied at each tangential
coordinate and integrated. The epsilon power follows by integrating
|eta(r/epsilon)|²|r|² over r, which is O(epsilon^5), before the prefactor
epsilon^(-2).

For the singular part the substitution r=epsilon rho yields exactly

    S_epsilon G_M Q
      =integral eta(rho)/(4pi|rho|) exp(-epsilon|rho|M)Q drho.

Strong continuity and contractivity of the positive-M semigroup on L², followed
by Bochner dominated convergence against this integrable nonnegative weight,
give the asserted limit

    S_epsilon U -> C_eta Q in L²(R^6).                 (9)

This identifies charge without assuming the Green operator preserves tangential
support; in general it does not.

For fixed delta in(0,R), on |v|<=R-delta and sufficiently small
|r|<min(R,2delta), all distances |r|,|v+r/2|,|v-r/2| are at most R, with strict
inequality in the nondegenerate interior. This tube is forbidden. Zero extension
makes S_epsilon U vanish there for sufficiently small epsilon. The L² limit(9)
forces Q to vanish there too. Exhausting |v|<R by delta down to0 and using the
measure-zero boundary sphere proves Q=0 a.e. on D^c.

By(1)–(3), every such physically supported ambient finite-energy localized state
has charge

    Q in closure_{H1/2(R6)} C_c^infinity(D),
    integral_D |Q|²/d <= C_(R,lambda,s,kappa)
                        (a_e[U]+lambda||U||²).         (10)

The weighted charge estimate from(7) controls its ambient H^(1/2) norm; fixed
linear coordinate change only modifies the comparison constant. This uses the
same fixed global localization as the corrected interior-contact note, not
merely compact-interior charge convergence.

## 5. What this removes and the remaining whole-state gap

Equations(1)–(10), if independently cleared, show that ambient supported charges
and zero-extended closure charges coincide in this exterior-sphere product
geometry. The charge condition previously necessary for the minimal domain is
automatic for physically supported finite-localized-energy states. Therefore
that charge condition alone cannot distinguish the minimal Friedrichs domain
from the prospective maximal whole-state comparison domain.

No claim is made that every such charge is realized by a minimal-domain state,
or that charge approximation approximates a whole singular state. For
U=phi+G_MQ, the Green term is tangentially nonlocal; its forbidden-region values
must be canceled by phi. Approximating the two parts independently can destroy
physical support or increase energy at the curved boundary.

The remaining task is joint approximation in the FULL interacting form norm.
A simple boundary-cutoff route would require proving admissibility and controlling
its derivative cost, such as

    epsilon^(-2) integral_{0<dist(X,boundary Omega_R)<C epsilon}|u(X)|² dX,

or finding a justified alternative not relying on that cutoff. Charge inequality
(10) alone does not prove this bound for the whole state and cannot control
arbitrary zero-charge regular components. This is a limitation of the present
proof, not a counterexample to whole-state density. Minimal/maximal form-domain
equality, the N=3 lattice liminf/recovery theorem and selection of the specified
Friedrichs boundary realization remain unproved. Physical assumptions and the
hard constraint have not been changed.

## 6. Sources and preservation

Protected sources, relative to doc/derivations/:
- `cubic_three_particle_continuum_2026-09-13.md`: pair-plus-spectator weighted form,
  fixed matching localization and minimal Friedrichs construction.
- `cubic_three_particle_interior_contact_2026-09-13.md`: corrected localized-charge
  closure statement and distinction between interior and boundary estimates.
This note adds a new charge-level boundary lemma, not a retrospective claim
that the earlier notes proved whole-state density.

Baseline675tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-three-particle-endpoint-charge-_2yzrt8s/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored files
and .git are excluded. Only this proof note is added. No scientific imports,
tests, numerical evaluations or model changes are authorized.

## 7. Independent mathematical review

Reviewer a605452345c881721 cleared conclusions(1)–(10) with no blocking error
and no repair required. They checked the cross-boundary constant, radial kernel,
critical supported-space density, normal averaging/Hardy estimate, semigroup
charge extraction, physical support and weighted energy bound. The hypothesis
U=tilde(chi_e u) in ambient V_e is essential: merely interior-local finite
energy is not promoted to that global zero-extended finite-energy condition.

Clearance does not include charge realizability, whole-state boundary density,
minimal/maximal whole-state domain equality, lattice convergence or Friedrichs
selection. The reviewer used direct reads only. The submitted proof-draft-v1.md
and hash remain preserved; no mathematical hypothesis or constant was changed.
Independent auditor a1802ff05b31a5de6 confirmed all675baseline files unchanged,
exactly one note added and676total, with current inventory matching
final-hashes.json. Sections1–5 are byte-identical to the submitted draft; only
status/review text changed. The ambient-V_e hypothesis and all whole-state/lattice
exclusions remain explicit. This is saved-document/hash clearance, not mathematical
recertification or evidence of absence of unrecorded execution. The audited
final-hashes.json is retained; closing-hashes.json binds final audit-status edits.
