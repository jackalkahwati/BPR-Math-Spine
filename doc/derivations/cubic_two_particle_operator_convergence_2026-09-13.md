# Two-particle operator convergence for the tuned cubic model

2026-09-13. Status: independent mathematical review cleared the stated two-particle
operator-convergence results; final independent saved-document/hash audit cleared.
This is the next bounded proof-only step under the user's continuous-work request.
No three-particle boundary identification, empirical validation or TOE result is
claimed. All earlier files and their limitations remain unchanged.

## 1. Operators, conventions and theorem targets

Fix supplied kappa>0 and finite real s!=0. Let B=[-pi,pi)^3,
omega(p)=4 sum_j(1-cos p_j), and W=integral_B dp/[(2pi)^3 omega(p)].
The previous proof establishes0<W<infinity. Retain precisely

    g_a^(-1)=-a²W/kappa+a³/(8pi kappa s),                (1)

for sufficiently small a>0 that g_a is finite (if s>0, take a<8pi Ws).
On ordered lattice two-particle wavefunctions define the threshold-referenced

    H_a^(2)=(kappa/a²)(12I-A_1-A_2)+g_a 1_{n1=n2}.      (2)

This is not a newly shifted many-body model. The declared two-particle energy
reference and the contact factor g_a, not2g_a, are the earlier conventions.
At each fixed a this is a bounded self-adjoint operator. Bosons form its symmetric
reducing subspace.

The continuum target is the self-adjoint tensor sum

    H_s^(2)=-(kappa/2)Delta_c+h_s,
    c=(x1+x2)/2, r=x1-x2,

where h_s=2kappa t_s is the point interaction constructed in the previous
continuum note. For mu>max(0,1/s), put G_mu(r)=exp(-mu|r|)/(4pi|r|). Its domain is

    u=phi+qG_mu, phi in H²,
    phi(0)=(mu-1/s)q/(4pi),
    (h_s+2kappa mu²)u=2kappa(-Delta+mu²)phi.             (3)

There is no new extension parameter. Our resolvents are (z-H)^(-1), Im z!=0,
with inner product antilinear in the first argument. Continuum Fourier transforms
here have no prefactor and use momentum measure dk/(2pi)^3; thus G_mu transforms
to(|k|²+mu²)^(-1). This differs only in representation from the earlier unitary-
prefactor convention and carries no change to the physical operator.

Targets under explicit embeddings below:
- generalized norm-resolvent convergence of each fixed physical total-momentum K
  fiber, including K=0;
- generalized strong-resolvent convergence of the full two-particle operator and
  its bosonic restriction.
No norm-uniform conclusion over unbounded K or real spectral boundary values is
claimed. The calculation starts in infinite spatial volume; no finite-torus
zero-mode expression or exchange of outgoing/spacing limits is used.

## 2. Exact fibers and embeddings

Set r=n1-n2,y=n2, an exact lattice-coordinate bijection. Fourier transform y:
phi_P(r)=sum_y exp(-iP dot y)psi(y+r,y), P in B. Bosonic exchange is
phi_P(r)=exp(iP dot r)phi_P(-r). Gauge by chi_P(r)=exp(-iP dot r/2)phi_P(r);
exchange becomes ordinary evenness. Subsequent relative Fourier transformation
defines a unitary map U into L²(BxB,dP dq/(2pi)^6):

    (Upsi)(P,q)=sum_{r,y} exp[-iP dot(y+r/2)] exp(-iq dot r)psi(y+r,y).

Unitarity follows as a composition of two lattice Fourier transforms and a
fiberwise unimodular gauge. Values at zone seams have measure zero; no globally
periodic gauge is assumed. In the gauged relative fiber,

    h_a(P)=12kappa/a² I
      -(2kappa/a²)sum_j cos(P_j/2)(S_j+S_j*)+g_a|0><0|.  (4)

For the relative lattice Fourier transform F_d define

    (J_a v)(k)=a^(3/2)1_{B/a}(k)(F_d v)(ak).

It is an isometry into L²(R³,dk/(2pi)^3), with range projection pi_a=1_{B/a}.
For the full operator define

    (E_a psi)(K,k)=a³1_{B/a}(K)1_{B/a}(k)(Upsi)(aK,ak). (5)

Its range projection Q_a=1_{B/a}(K)1_{B/a}(k) tends strongly to I. Scaling each
three-dimensional measure by a^(-3) verifies all amplitude factors.

The individual quasimomenta are p1=P/2+q, p2=P/2-q modulo2pi. For each reduced
total P, translating the individual p1 torus by-P/2 gives the centered q chart.
This is an almost-everywhere bijection, not an eightfold momentum cover. At fixed
physical K,k and small a no wrapping occurs. In continuum coordinates both
(x1,x2)<->(c,r) and their dual momentum transformation have absolute Jacobian1.
On the lattice c/a=y+r/2: center half-integer parity is tied to r and is neither
dropped nor independently summed. The exact initial (r,y) bijection accounts
for it. Bosons map under E_a to functions even in k; B/a is reflection invariant
up to measure-zero seams. No even-parity normalization factor is inserted.

For fixed K and sufficiently small a, P=aK is in B. The embedded free symbol is

    E_(a,K)(k)=(4kappa/a²)sum_j[1-cos(aK_j/2)cos(ak_j)]
      =epsilon_a(K)+(4kappa/a²)sum_j cos(aK_j/2)(1-cos ak_j),
    epsilon_a(K)=(4kappa/a²)sum_j(1-cos(aK_j/2)).         (6)

On compact k sets this converges uniformly to epsilon(K)+2kappa|k|²,
epsilon(K)=kappa|K|²/2. The original12kappa/a² diagonal is retained; no fiber
threshold is reset to hide center-of-mass kinetic energy.

## 3. Free resolvent norm and Green-vector convergence

Fix K and nonreal z. For small a all cos(aK_j/2)>=1/2. On B/a, the elementary
inequality1-cos t>=2t²/pi² gives

    E_(a,0)(k)>=8kappa|k|²/pi²,
    E_(a,K)(k)>=4kappa|k|²/pi².

For small |k| use |z-E|>=|Im z|; for large |k| use the quadratic lower bound
and |z-E|>=E-|z|. Together these give a constant independent of small a with

    1_{B/a}(k)/|z-E_(a,K)(k)|<=C/(1+|k|²).             (7)

Compact uniform convergence from(6), followed by uniformly small large-k tails
from(7), establishes multiplier OPERATOR-NORM convergence

    M_[1_{B/a}/(z-E_(a,K))]
      -> M_[1/(z-epsilon(K)-2kappa|k|²)].                (8)

The complement of B/a is handled by the same continuum tail bound.

Put v_(a,K,z)=1_{B/a}/(z-E_(a,K)), and
v_(K,z)=1/(z-epsilon(K)-2kappa|k|²). Their squared bound from(7) is integrable
in dimension3. Dominated convergence gives

    ||v_(a,K,z)-v_(K,z)||_2 ->0.                        (9)

The lattice source vector scales as
J_a(z-h_(0,a)(aK))^(-1)|0>=a^(3/2)v_(a,K,z).
This vector estimate is extra information absent from scalar scattering alone.

## 4. Off-axis renormalized denominator, including moving fibers

Let G_(a,K)(z)=<0|(z-h_(0,a)(aK))^(-1)|0>. Then

    G_(a,K)(z)/a³=integral_{B/a} dk/(2pi)^3 /(z-E_(a,K)(k)),
    integral_{B/a} dk/[(2pi)^3 E_(a,0)(k)]=W/(kappa a).

The difference of free energies obeys

    d_(a,K)(k)=E_(a,K)(k)-E_(a,0)(k)
      =(4kappa/a²)sum_j[1-cos(aK_j/2)]cos(ak_j),
    |d_(a,K)(k)|<=kappa|K|²/2.                          (10)

The bound holds over the entire band, not just at low k. At each fixed k the
difference tends to epsilon(K). Subtract the divergent integral exactly:

    L_(a,K)(z)=G_(a,K)(z)/a³+W/(kappa a)
      =integral_{B/a} [1/(z-E_(a,K))+1/E_(a,0)]dk/(2pi)^3.

The combined integrand equals
(z-d_(a,K))/[E_(a,0)(z-E_(a,K))]. Equations(7),(10) dominate its absolute value
by C/[|k|²(1+|k|²)], integrable at both zero and infinity in3D. Hence

    L_(a,K)(z) -> integral_R³ [1/(w-2kappa|k|²)+1/(2kappa|k|²)]dk/(2pi)^3
                =chi/(8pi kappa),
    w=z-epsilon(K), chi=sqrt(-w/(2kappa)), Re chi>0.     (11)

For real positive chi the radial integral is
chi²/(4pi²kappa) integral_0^infinity dt/(t²+chi²)=chi/(8pi kappa).
Both sides are holomorphic for Re chi>0; locally uniform integrable bounds justify
extension there. This is a convergent OFF-AXIS integral identity, not an unproved
outgoing boundary continuation. Im z!=0 implies chi is not real.

With(1), the scaled interaction denominator is

    D_(a,K)(z)=a^(-3)[g_a^(-1)-G_(a,K)(z)]
      =1/(8pi kappa s)-L_(a,K)(z)
      ->(s^(-1)-chi)/(8pi kappa).                       (12)

Its limit is nonzero for nonreal z. Finite-a nonvanishing follows separately from
Im G_(a,K)(z)=-(Im z)||(z-h_(0,a)(aK))^(-1)delta_0||² and real g_a. Thus

    T_(a,K)(z)=D_(a,K)(z)^(-1)
      ->T_s(w)=8pi kappa/[s^(-1)-sqrt(-w/(2kappa))].      (13)

No quadrature, W approximation or new coupling counterterm appears.

## 5. Rank-one norm limit and the continuum operator identity

The exact embedded resolvent is

    J_a(z-h_a(aK))^(-1)J_a*
      =M_[v_(a,K,z)]
        +T_(a,K)(z)|v_(a,K,z)><v_(a,K,conj(z))|.         (14)

The second vector is evaluated at conj(z) because the two free resolvents in
the rank-one identity are BOTH at z, not adjoints of each other. With the declared
inner product, this bra yields the correct analytic kernel. Each external
source contributes a^(3/2), producing the a³tau normalization in(13).

From(9), both vectors converge in L²; their norms remain bounded. Combined with
scalar convergence(13), the elementary rank-one norm inequality proves norm
convergence of the second term. Add(8) for the free term. The limiting expression
is precisely the momentum-space resolvent of epsilon(K)+h_s in(3): at
w=-2kappa mu², away from its pole, v_w=-G_mu/(2kappa) and
T_s(w)=-8pi kappa/(mu-1/s). The correction is therefore
-[1/(2kappa c_mu)]|G_mu><G_mu|, c_mu=(mu-1/s)/(4pi), matching the negative of
(h_s+2kappa mu²)^(-1)'s correction. The explicit domain equation(3), or its
resolvent identity continued off the real spectrum, identifies the same formula
for every nonreal w. Therefore, for each fixed physical K and Im z!=0,

    ||J_a(z-h_a(aK))^(-1)J_a*
          -(z-epsilon(K)-h_s)^(-1)|| ->0.              (15)

This is GENERALIZED norm-resolvent convergence: the embedded lattice resolvent
is zero on its band complement and is not itself a resolvent on the whole
continuum space. Even relative functions are a reducing subspace on both sides,
so(15) also holds for bosons, including K=0. For s>0 the limiting fiber pole
z=epsilon(K)-2kappa/s² is real and excluded. No real-boundary or uniform-near-pole
claim follows from(15).

## 6. Full two-particle generalized strong-resolvent limit

Under E_a in(5), the full embedded resolvent is decomposable in K with fibers
S_a(K,z)=J_a(z-h_a(aK))^(-1)J_a* for K in B/a and0 otherwise. Every fixed K
belongs to B/a eventually, and(15) gives convergence in fiber operator norm.
Self-adjointness gives the uniform bound

    ||S_a(K,z)||<=1/|Im z|,
    ||(z-epsilon(K)-h_s)^(-1)||<=1/|Im z|.

For any full-space vector f(K,.) these dominate the squared difference by
4||f(K,.)||²/|Im z|². Direct-integral dominated convergence proves

    E_a(z-H_a^(2))^(-1)E_a* -> (z-H_s^(2))^(-1)
       strongly.                                      (16)

Measurability follows from the explicit fiber multipliers and rank-one formula;
zone-boundary changes occur on null sets. Bosonic reduction commutes with the
embedding and operators, so the same statement holds there. No convergence in
norm uniformly over all total K is inferred from this argument.

If conventional common-space terminology is desired, an auxiliary ZERO completion
of the transported lattice operator on ran Q_a, direct-summed with0 on ran(I-Q_a),
is self-adjoint on the common space. Its resolvent differs from the left side
of(16) by z^(-1)(I-Q_a), which vanishes strongly. Thus these comparison completions
converge in the ordinary strong-resolvent sense. This completion does not change
any lattice operator; it only defines action on missing comparison-space modes.
Its complementary term does NOT vanish in operator norm, so it is not used to
claim ordinary norm-resolvent convergence in(15).

## 7. Scope and next hypothesis

The proposed theorem supplies full two-particle operator convergence under the
specified Fourier embedding, rather than merely scalar scattering. It retains
the supplied kappa,s and the same tuned cubic onsite model. It does not repair
unrestricted attractive many-body instability or itself define a three-particle
exclusion-boundary realization.

Remaining gates include the N=3 contact-domain identification, compatible
embeddings at the exclusion boundary, singular-state liminf and recovery,
and proof that the hard lattice selects the previously constructed Friedrichs
candidate. Nothing here excludes a different boundary realization or justifies
exchanging penalty, spacing and outgoing-scattering limits. Arbitrary-N operator
convergence, wave-operator convergence, empirical validity and TOE completion
remain unproved. A separate bounded step must address the next missing lemma.

## 8. Sources, preservation and review

Protected references (all unchanged):
- `cubic_two_body_scattering_2026-09-13.md:11-74,218-275`: lattice pair/fiber
  normalization, resolvent and tuning. Its scalar result does not already prove(15).
- `cubic_three_particle_continuum_2026-09-13.md:19-124`: relative point-interaction
  domain, normalization and tensor interpretation. Its N=3 boundary selection
  is not identified by this two-particle theorem.
Paths are relative to doc/derivations/. The embedding, Green-vector convergence,
moving-fiber subtraction and dominated-convergence arguments are given above,
not attributed to a nonexistent earlier theorem. No new physics is supplied.

Baseline673tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-two-particle-operator-convergence-kx4eerwt/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored files
and .git are outside scope. Only this note may be added in this step. No scientific
imports, tests, matrices or numerical evaluations are authorized.

## 9. Independent review record

Reviewer a31a83679aceec45c cleared the complete proof with no mathematical blockers,
checking exact fibers and embeddings, Brillouin-zone chart multiplicity, bosonic
parity, kinetic operator-norm tails, Green-vector L² convergence, global moving-
fiber subtraction, off-axis denominator, rank-one conjugation and target domain,
and the full direct-integral strong limit/zero completion. The clearance is
restricted to fixed-K generalized norm-resolvent convergence and full two-particle
generalized strong-resolvent convergence for nonreal z. No real-boundary,
three-particle, arbitrary-N or empirical statement is cleared.

The reviewer used direct file reads only. The original submitted proof-draft-v1.md
and its hash are preserved externally; no mathematical hypothesis or constant
changed after submission. Independent auditor a54bb5e84f99736cd confirmed
all673prior tracked/nonignored files unchanged, exactly one addition and674total,
with every current hash matching final-hashes.json and this note matching the
pre-audit snapshot. The mathematical/source body is unchanged from the submitted
draft; only status/provenance changed. This was a saved-document/hash audit,
not mathematical recertification or evidence of absence of unrecorded execution.
The audited final-hashes.json is retained; closing-hashes.json binds the final
audit-status edits.
The next separately reviewed hypothesis concerns N=3 boundary identification,
not a retroactive expansion of this theorem.
