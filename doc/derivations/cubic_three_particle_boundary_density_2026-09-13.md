# Whole-state boundary density for the N=3 contact candidate

2026-09-13. Status: the specified N=3 global form-domain equality independently
cleared; independent saved-document/hash audit completed with no blockers.
This step identifies two specified continuum FORM domains. It does not prove
hard-lattice convergence or a maximal operator-domain trace characterization.

## 1. The two domains and their comparison norm

Retain supplied R,kappa>0 and finite s!=0, Omega_R={diam(X)>R} in R^9, the
specified initial singular domain C_s and Friedrichs form q^D. Identify the
allowed L² space with its zero extension into L²(R^9). Let chi_0 and chi_e be
the fixed smooth matching partition from the continuum construction, with pair
cutoffs at r0=R/8,r1=R/4. At N=3 only the empty component and three pair
components survive on Omega_R.

Let a_0 be the ambient free form and a_e the ambient pair-plus-spectator form
at the ORIGINAL scattering length s. Fix lambda>4E_s and lambda>0, where
E_s=2kappa*1_{s>0}/s². Set V_0=H¹(R^9), V_e=D((A_e+lambda)^(1/2)). Define

    V_max={U in L²(R^9): U=0 a.e. on Omega_R^c,
                         chi_alpha U in V_alpha for all alpha},
    ||U||_max²=sum_alpha(a_alpha[chi_alpha U]
                                  +lambda||chi_alpha U||²).       (1)

This is a GLOBAL physical zero-extension condition with finite localized
ambient energies. Interior-local finite energy alone is not this domain.
Let V_min=D(q^D), the already specified closure of C_s in its shifted energy
norm. The target is

    V_min=V_max,                                                    (2)
    q^D[U]=sum_alpha a_alpha[chi_alpha U]-integral W_loc|U|².         (3)

All estimates are at N=3, with the fixed physical R,s,kappa. Cutoffs, lambda
and state dilations below are auxiliary proof tools, not physical modifications.

Since a_alpha>=-E_s and the partition squares sum to1,
||U||_max²>=(lambda-E_s)||U||². A Cauchy sequence in(1) has an L² limit U
supported in closure(Omega_R). Each localized sequence converges in its complete
ambient V_alpha, and its L² limit is chi_alpha U. Thus V_max is complete.

The form q_loc defined by the right side of(3) is closed and semibounded: if
W_*=||W_loc||_infinity, then

    q_loc[U]+(lambda+W_*)||U||²
      =||U||_max²+integral(W_*-W_loc)|U|²

is a positive complete norm equivalent to(1). The established singular IMS
identity says q_loc agrees with the initial form on C_s. Therefore V_min is
its closed subspace obtained by taking that core closure, and V_min is already
contained in V_max. We prove the reverse density below, rather than assuming it.

## 2. Whole ambient form vectors under physical dilation

For pair coordinates r=xi-xj, z=(c,y), define the unitary dilation

    (D_t U)(r,z)=t^(-9/2)U(r/t,z/t),      1<=t<=2.

Use the established ambient decomposition U=phi+G_M Q. Its regular part is in
H¹(R^9) and Q is in H^(1/2)(R^6); their product norm is equivalent to the
shifted V_e norm. In tangential Fourier variables xi=(p,k),

    m(xi)²=mu²+b(xi), mu²=lambda/(2kappa),
    b(xi)=|p|²/4+|k|²/2.

Direct scaling of the1/|r| coefficient gives

    Q_t(z)=t^(-7/2)Q(z/t), Qhat_t(xi)=t^(5/2)Qhat(t xi),
    ||Q_t||²_2=t^(-1)||Q||²_2.

The singular term transforms exactly as D_t G_M Q=G_(M_t) Q_t, with
m_t(xi)=t^(-1)m(t xi)=sqrt(mu²/t²+b(xi)). Returning to the ORIGINAL fixed M,

    D_t U=phi_t+G_M Q_t,
    phi_t=D_t phi+(G_(M_t)-G_M)Q_t.                     (4)

Omitting the last correction would change the representation parameter rather
than prove fixed-form-domain convergence.

In full Fourier variables (ell,xi), the correction is

    (2pi)^(-3/2) mu²(1-t^(-2)) Qhat_t(xi)
      /[(|ell|²+m_t²)(|ell|²+m²)].

For1<=t<=2 the two squared masses are uniformly comparable to1+|xi|².
Consequently its H¹ norm squared is bounded by C|t-1|² times

    integral |Qhat_t(xi)|² integral_R3
      (1+|ell|²+|xi|²)/[(|ell|²+m_t²)²(|ell|²+m²)²] d ell dxi.

Scaling ell by sqrt(1+|xi|²) bounds the inner integral by
C(1+|xi|²)^(-3/2). Hence the correction tends to0 in H¹, with bound
C|t-1|||Q_t||_2. Ordinary dilation is strongly continuous on H¹, and the
charge dilation is uniformly bounded/strongly continuous on H^(1/2). These
facts follow first on Schwartz functions by Fourier change of variables and
then by density and uniform norm bounds. Thus

    D_t:V_e->V_e is uniformly bounded for1<=t<=2,
    ||D_tU-U||_(V_e)->0 as t down to1.                  (5)

The free V_0 case is ordinary H¹ dilation. All statements concern the WHOLE
ambient vector, not only its charge.

For additional clarification, evaluating the same form at fixed s yields

    a_(e,s)[D_t U]=t^(-2)a_(e,s)[U]
      +(2kappa/(4pi s))(t^(-2)-t^(-1))||Q||²_2.         (6)

To derive this identity temporarily use the auxiliary shift lambda/t². The
regular shifted energy scales by t^(-2), while the charge coefficient becomes
m/t-1/s and the charge measure contributes t^(-1). Subtracting the shifted
L² norm gives(6). Our choice lambda>4E_s keeps all these auxiliary forms
positive for1<=t<=2. Equation(6) does not change the physical s: dilation need
not preserve the operator-domain boundary condition, but it preserves the fixed-s
FORM domain with precisely this controlled energy change.

## 3. Fixed localization after dilation

The fixed localization must still be checked: (5) alone is not a V_max statement.
Use global smooth representatives
f_e(X)=sin(theta(|xi-xj|)) and g_0(X)=product_e cos(theta(|xi-xj|)).
On Omega_R, chi_e=f_e and chi_0=g_0. Indeed if one pair distance<R/4,
allowedness forces both remaining distances>3R/4; otherwise the triangle
inequality would make all three distances<R. This also implies that the active
pair component is the only one.

For these global representatives put
Delta_(alpha,t)(X)=chi_alpha(tX)-chi_alpha(X). Then, for zero-extended U,

    chi_alpha D_t U=D_t(chi_alpha U+Delta_(alpha,t)U).  (7)

On the physical support of U, a nonzero mismatch requires a pair distance in
[r0/2,r1] when1<=t<=2. The other two distances are then>3R/4. Thus every
mismatch is uniformly away from ALL contact surfaces. Choose a fixed bounded
smooth Z that vanishes near all contacts and equals1 on these mismatch regions.
Then Delta_(alpha,t)U=Delta_(alpha,t)ZU.

Using the ambient multiplier estimates from the corrected interior-contact
lemma and the partition identity,

    ZU=sum_beta Z chi_beta (chi_beta U) in H¹(R^9),
    ||ZU||_H1<=C||U||_max.                              (8)

For pair terms the multiplier vanishes near that pair's contact, giving zero
charge and ordinary H¹ control; the free term is already H¹. Normal derivatives
of a singular vector are never asserted to exist globally.

The fixed smooth radial cutoff structure gives
||Delta_(alpha,t)||_(W1,infinity)<=C|t-1|. For each individual pair factor its
variation is confined to bounded pair distances and its first derivative has
a uniformly bounded derivative in t there; finite products give the same
estimate for g_0. This does not require compact center-of-mass support. Hence

    ||Delta_(alpha,t)U||_H1<=C|t-1|||U||_max.             (9)

Apply (5) to chi_alpha U and ordinary H¹ dilation to the mismatch in(7).
Because outward dilation preserves physical allowed support, we conclude

    D_t U in V_max,
    ||D_tU-U||_max->0 as t down to1.                    (10)

## 4. A boundary margin for the whole wavefunction

If U is supported in closure(Omega_R), then D_t U is supported, in the
essential-support sense, in {diam(X)>=tR}. The diameter function is sqrt(2)-
Lipschitz in R^9: each pair distance changes by at most sqrt(2) times the full
configuration displacement, and taking a finite maximum retains this bound.
Therefore

    dist(supp D_tU,Omega_R^c)>=(t-1)R/sqrt(2).           (11)

This margin is for the whole state, including the cancellation of regular and
nonlocal Green parts. It is not a claim that those parts separately satisfy
the physical support constraint. The operator still acts on the ORIGINAL
Omega_R; the expanded support gap is a property of approximating vectors, not
a changed hard radius.

## 5. Margin states belong to the minimal closure

Fix t>1 and W=D_tU. Put v_alpha=chi_alpha W in its corresponding ambient domain.
The norm partition yields W=chi_0 v_0+sum_e f_e v_e. Compactify using an ordinary
smooth large-radius cutoff eta_L(X) for the free part and tangential cutoffs
b_L(z_e) for pair e, equal1 on expanding tangential balls:

    W_L=chi_0 eta_L v_0+sum_e f_e b_L v_e.               (12)

Each pair term is compact because f_e bounds |r_e|<r1 and b_L bounds its
six tangential coordinates; the free term is compact by eta_L. Multiplication
does not reduce the physical boundary margin in(11).

Tangential multipliers b_L have uniformly bounded derivatives and are flat in
the normal coordinate. The ambient multiplier estimate is uniform in L, and
convergence on compact ambient cores gives b_L v_e->v_e in V_e. Similarly
eta_L v_0->v_0 in H¹. The corresponding assembly maps are bounded into V_max:
for physically supported V in V_e, f_e V has own-pair component f_e²V, free
component cos(theta_e)sin(theta_e)V supported away from contact, and other-pair
components zero by matching geometry. These are bounded in their stated norms.
For the free assembly, H¹ embeds continuously in each V_e with zero charge.
It follows that

    ||W_L-W||_max->0.                                   (13)

A compact pair term w=f_e b_L v_e lies in V_e with the boundary margin(11).
Choose a smooth radial a0(r_e), equal1 near0, supported at a radius sufficiently
smaller than both R and (t-1)R. Split w=a0w+(1-a0)w. The second term is H¹,
compactly supported away from all contacts and inside Omega_R. Interior
mollification approximates it by C_c^infinity(Omega_R minus Sigma), hence it
belongs to V_min.

For the first term write v=c-y. Since diam(X)>=tR on its support and |r_e|
is small, diam=max(|r_e|,|v+r_e/2|,|v-r_e/2|) implies

    |v|>=tR-|r_e|/2>R+positive margin.

Choose the support radius small enough that this margin also admits a slightly
larger compact pair tube wholly in Omega_R and disjoint from other collision
surfaces. The corrected compact-interior ambient-form density lemma applies:
a0w is approximable in its ambient form norm by the ORIGINAL fixed-s singular
core C_s. Bounded assembly transfers this to the equivalent global energy norm.
Thus a0w belongs to V_min. The compact free term in(12) is already separated
from contacts by chi_0 and is approximable by regular interior functions too.
Therefore every W_L belongs to V_min.

V_min is closed in V_max by section1. Equations(13) and(10) imply first
D_tU in V_min and then U in V_min as t down to1. This proves(2), and equality
of the closed forms gives(3). For bosons, dilation commutes with permutations
and finite permutation averaging of the approximants preserves energy-norm
convergence and C_s. The domain equality therefore holds on that subspace too.

## 6. Exact scope and next missing lattice estimates

If independently cleared, this proves that the specified minimal Friedrichs
FORM domain equals the GLOBAL physically zero-extended finite-localized-energy
comparison domain. It is a whole-state density theorem, not a deduction from
the earlier charge-only endpoint condition. The proof uses the same physical
R and s throughout. No extra force, boundary parameter or discarded states
have been introduced.

The theorem does not identify a maximal OPERATOR domain at the contact/exclusion
intersection or prove uniqueness among every possible self-adjoint boundary
extension. It gives no N>=4 result. Most importantly, lattice convergence is
still not established. A lattice proof must show that bounded discrete renormalized
energies converge into this ambient localized-energy domain with the needed
liminf bound, and must construct singular recovery sequences. Two-particle
operator convergence and the uniform lattice stability estimate are ingredients,
not substitutes for those steps. No wave-operator, empirical-validation or
TOE conclusion follows.

## 7. Sources and preservation

Protected sources, relative to doc/derivations/:
- `cubic_three_particle_continuum_2026-09-13.md`: fixed matching forms, weighted
  decomposition, intrinsic singular C_s and prescribed Friedrichs closure.
- `cubic_three_particle_interior_contact_2026-09-13.md`: corrected ambient
  multiplier/core transfer and compact-interior form density.
- `cubic_three_particle_endpoint_charge_2026-09-13.md`: charge-level boundary
  result, explicitly insufficient on its own for whole-state approximation.
This is a new proposed whole-state theorem; prior notes are not retrospectively
claimed to contain it.

Baseline676tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-three-particle-boundary-density-7z2tnalq/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored files
and .git are excluded. Only this proof note is added. No scientific imports,
tests, numerical evaluations or model edits are authorized.

## 8. Independent mathematical review

Reviewer a8011f79f965f5b3c cleared the entire stated global form-domain equality
with no blocking mathematical gap or repair required. They checked fixed-s
dilation and Green correction, fixed-localization mismatch control, V_max
completeness, compactification/assembly and noncircular reverse density using
the corrected interior lemma. The clearance includes the bosonic restriction
and is limited to the specified N=3 zero-extended FORM domains.

No maximal operator-domain characterization, uniqueness among other boundary
extensions or lattice convergence is cleared. The reviewer used direct reads
only, with no execution or delegated agents. The submitted proof-draft-v1.md
and hash are preserved externally; no mathematical constants or hypotheses
changed after submission. Independent auditor adcf9be7d20f55e8b confirmed all676
baseline files unchanged, exactly one new note and677current files matching
final-hashes.json. Mathematical sections1–6 are byte-identical to the draft;
only status/review reporting changed, and the note matched pre-audit-note.md.
This is saved-document/hash clearance, not mathematical recertification or
proof of absence of unrecorded execution. The audited final-hashes.json is
retained; closing-hashes.json binds the final audit-status edits.
