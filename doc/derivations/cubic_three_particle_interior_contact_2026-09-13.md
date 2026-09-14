# Interior contact identification for the three-particle continuum candidate

2026-09-13. Status: interior lemma and specified localized-charge closure independently
cleared after a preserved correction; independent saved-document/hash audit cleared.
This bounded step concerns interior contact traces of the specified Friedrichs
candidate, not its hard-boundary domain identification or lattice convergence.

## 1. Precise target and inherited objects

Retain R,kappa>0, finite real s!=0, Omega_R={max_ij|xi-xj|>R}, and the specified
operator H^D_(R,s), form q^D and initial singular domain C_s from
`cubic_three_particle_continuum_2026-09-13.md`. C_s is a form core by construction;
it is not assumed to be a graph core for H^D. All previous files remain protected.

Fix a pair and write r=xi-xj, z=(c,y), where c=(xi+xj)/2 and y is the spectator.
Let D={z:|c-y|>R}. The ambient pair-plus-spectator operator and form domain are

    A=h_s+H_tan, H_tan=-(kappa/2)Delta_c-kappa Delta_y,
    V=D((A+lambda)^(1/2)), lambda>E_s,
    E_s=2kappa*1_{s>0}/s².

The free differential expression away from collision is
T=-2kappa Delta_r-(kappa/2)Delta_c-kappa Delta_y. Choose O compactly contained
in D and a tube U={|r|<rho,z in O}, with rho<R/8 sufficiently small that its
closure is inside Omega_R and avoids other collision surfaces. (One may choose
O open with compact closure.) Let zeta in C_c^infinity(U) be real and independent
of r for |r|<b, for some b>0. Such cutoffs have a uniform normal-flat region.

The candidate interior lemma is: for every u in D(H^D), its localized zero
extension w=tilde(zeta u) belongs to D(A), with

    A w=tilde(zeta H^D u+[T,zeta]u).                    (1)

Its contact expansion has coefficients A_zeta in H^(3/2)(R^6) and B_zeta in
H^(1/2)(R^6), satisfying in H^(1/2) as r->0

    w(r,.)-A_zeta/|r| -> B_zeta=-A_zeta/s.              (2)

This is an interior local assertion with constants depending on the cutoff and
its distance from the hard boundary. It asserts no classical boundary trace at
|c-y|=R and no uniform estimate as that distance tends to zero.

## 2. Ambient form core retaining the singular charge

Fix mu>max(0,1/s) and a smooth compact radial cutoff rho0=1 near zero. Define
G_mu=exp(-mu|r|)/(4pi|r|), c_mu=(mu-1/s)/(4pi), and
g(r)=rho0(r)(G_mu(r)+c_mu). The relative pair form has domain phi+qG_mu with
phi in H¹ and shifted norm

    2kappa(||grad phi||²+mu²||phi||²+c_mu|q|²).

The class span{g,C_c^infinity(R³ minus{0})} is a relative form core. Indeed any
phi+qG_mu equals qg+[phi+q(G_mu-g)], and the bracket is in H¹. Regular smooth
functions avoiding0 are H¹-dense: after smoothing, removing an epsilon ball
costs O(epsilon) in squared gradient norm for a bounded smooth function. The
singular term qg is NOT removed. Its regular remainder is H² with value c_mu,
so g is also an operator-domain vector for the relative contact operator.

Finite tensor products of this relative core with C_c^infinity(R^6) form an
ambient V-core K. This follows from the positive tensor-sum form construction:
spectral energy cutoffs, density of finite tensor sums on bounded rectangles,
and then removal of the cutoffs in form norm. The functions of K belong to D(A).
For multiplier arguments enlarge K only by regular functions in
C_c^infinity(R^9 minus{r=0}) and smooth compact charge generators whose germs
are exactly eta(z)(G_mu(r)+c_mu) near contact, with normal cutoffs equal1 there.
These additional functions satisfy the operator boundary condition and belong
to D(A); the enlarged class remains a form core. Arbitrary compact smooth
regular functions with nonzero trace at r=0 are NOT included: their zero charge
would require zero regular trace in the operator domain. Products by admissible
multipliers split into an admitted generator with the multiplied charge and a
smooth compact remainder vanishing in a neighborhood of contact. This is a
literal class inclusion, not an unsupported graph-core approximation.

No claim is made that arbitrary ambient form vectors have a fixed-mu decomposition
with a globally H¹ regular part in all nine variables. The previous momentum-
dependent decomposition supplies the actual completed form domain.

## 3. Interior multiplier and derivative estimates

Take a real smooth bounded multiplier f with bounded first and second derivatives,
independent of r in a uniform neighborhood of0. On compact singular core vectors,
f preserves the contact relation: near0 it multiplies the charge by f(0,z),
and its normal derivatives vanish. Subtract a compact radial generator with
that charge; the remainder is smooth away from collision and has no contact
singularity. Thus the product lies in D(A). For noncompact f this reasoning
still applies to the compact support of each core vector.

Rescale f by a constant so |f|<=q<1, and put j=sqrt(1-f²). Both are admissible.
The operator commutator identity on these singular core vectors yields

    a[fv]+a[jv]=a[v]+integral Gamma(f,j)|v|²,
    Gamma=2kappa(|grad_r f|²+|grad_r j|²)
          +(kappa/2)(|grad_c f|²+|grad_c j|²)
          +kappa(|grad_y f|²+|grad_y j|²).              (3)

Derivatives occur away from normal singularities, or tangentially on smooth
charges; this is not an unrenormalized divergent gradient identity. Positivity
of a+lambda and bounded Gamma prove multiplier boundedness on V. Ambient form
core density extends this estimate to every v in V. Undo the harmless rescaling.

Because h_s>=-E_s and H_tan is nonnegative, the tensor form gives

    (kappa/2)||grad_c v||²+kappa||grad_y v||²
      <=a[v]+E_s||v||².                                (4)

Thus tangential derivatives of the WHOLE v are L², even though its form charge
has only H^(1/2) regularity. If a cutoff vanishes near contact, its product with
v has zero charge and ordinary H¹ regularity. For precision this last conclusion
also follows from (3) by approximating with core vectors: products have zero
charge, so their contact form equals free energy, and V-Cauchy convergence implies
H¹ convergence. This controls normal derivatives locally away from r=0.

Consequently for an admissible compact zeta,

    [T,zeta]v=-4kappa grad_r zeta dot grad_r v
      -kappa grad_c zeta dot grad_c v
      -2kappa grad_y zeta dot grad_y v+(Tzeta)v          (5)

is L², bounded by a local V norm. No global L² normal derivative of a singular
state is asserted. Normal cutoff derivatives occur only away from contact.

## 4. Transfer into the prescribed Friedrichs closure

For v in V choose v_n in K converging in V. If zeta is supported in U as in
section1, then zeta v_n belongs to C_s. To see this, each multiplied singular
term has smooth compact charge zeta(0,z)eta(z) supported in D. Choose a radial
generator tube sufficiently small to fit inside U. Its subtraction leaves a
smooth compact regular function vanishing near contact; both parts are admitted
in C_s. The use of a fixed smaller tube is possible because the charge support
has a positive margin from the hard boundary.

On these products the global pairing is the ambient one:

    q^D[zeta v_n]=a[zeta v_n].

This follows from their common off-contact action and matching contact relation;
all support is in the chosen pair patch. The multiplier estimate makes the
sequence Cauchy in the global shifted form norm. Its L² limit is zeta v, so

    zeta v in D(q^D), q^D[zeta v]=a[zeta v].             (6)

If v already has compact support in U, choose such a cutoff equal1 near its
support. This proves local compact-interior form density, without assuming
anything about the global maximal finite-local-energy domain.

Conversely the defining matching partition for q^D has chi_e=1 throughout U.
The continuous localization inclusion proved for q^D gives tilde(chi_e u) in V
for u in D(q^D). Multiplication then implies tilde(zeta u) in V. Polarization
of (6) and approximation by the respective cores transfer localized pairings
as well as norms. No graph-core density is used.

## 5. Operator localization by form tests

Take u in D(H^D), and choose a real admissible xi supported in U equal1 on a
neighborhood of supp zeta. Put w0=tilde(xi u) in V. For every v in V, (6) supplies
the admissible global form test zeta v. Locality gives

    a(zeta v,w0)=q^D(zeta v,u)=<zeta v,H^D u>.           (7)

Establish locality first on compact ambient and global core vectors, where xi
is constant on the test support and the differential/contact expressions agree.
The bounded localization maps then extend the equality in both form norms.
The ambient commutator identity, extended using (5), is

    a(v,zeta w0)=a(zeta v,w0)+<v,[T,zeta]w0>.

Here inner products are antilinear in the first argument. Combining with(7)
gives the L² representation of this pairing against every v in V, proving(1)
by the closed-form representation theorem. Derivatives of xi vanish where
zeta or its derivatives contribute. Thus the right side is precisely the one
in(1), not an expression depending on the auxiliary xi.

## 6. Sobolev interpretation of the interior contact condition

Let w in D(A), F=(A+lambda)w, and let M be the tangential Fourier multiplier
m(p,k)=sqrt((lambda+kappa|p|²/2+kappa|k|²)/(2kappa)). Fiberwise the relative
operator formula gives

    w=phi+G_M q, phi=(T+lambda)^(-1)F in H²(R^9),
    tau phi=(M-1/s)q/(4pi),                            (8)

where (G_M q)^hat(r,p,k)=exp(-m|r|)qhat/(4pi|r|), and tau is the codimension3
trace at r=0. Ellipticity of the free T gives the stated full H² regularity.
The trace theorem gives tau phi in H^(1/2)(R^6). Since lambda>E_s, M-1/s is
strictly positive and comparable to a first-order elliptic multiplier; hence
q in H^(3/2)(R^6).

The codimension3 trace estimate and continuity of translations on H² imply
phi(r,.)->tau phi in H^(1/2) as r->0. This can be seen by applying the uniform
trace map to the normal translate of phi and subtracting phi. For t=|r|,

    w(r,.)-q/(4pi t)=phi(r,.)+[(exp(-tM)-1)/t]q/(4pi).

The multiplier has absolute value at most m. The H^(3/2) regularity of q supplies
an H^(1/2) dominated-convergence majorant, so the second term tends to -Mq/(4pi).
Equation(8) then gives B=-q/(4pi s). With A=q/(4pi), this proves(2).

Applying to w=tilde(zeta u) gives the contact relation for general operator-domain
vectors on every compact interior patch, not merely the original core. Uniqueness
of the singular coefficient makes the localized coefficients compatible on
overlaps. Thus the charge belongs locally to H^(3/2)(D). No classical charge
trace at the exclusion boundary or maximal operator-domain equality follows.

## 7. Remaining hard-boundary selection and lattice gap

This lemma, if cleared, removes the INTERIOR contact-identification gap only.
The global Friedrichs domain remains the closure of compact singular states in
the interacting energy norm. It is included in the space of states with finite
zero-extended localized ambient energy; equality with that entire comparison
class has not been proved.

Specify the GLOBAL localized charge using the fixed matching partition from
the continuum construction: Q_e(u)=Q(tilde(chi_e u)), where Q extracts the unique
ambient momentum-dependent decomposition charge. For a defining sequence
u_n in C_s converging to u in the Friedrichs form norm, subtract the pair-e
generators from tilde(chi_e u_n). The remainder is smooth compactly supported
away from that collision and therefore has zero ambient charge. A generator
with tangential charge eta has decomposition charge exactly eta: fiberwise,
rho(G_mu+c_mu)-G_m belongs to H¹ in the relative variable, and uniqueness of
the relative singular decomposition identifies the coefficient. It follows that
Q_e(u_n) lies in C_c^infinity(D), D={|c-y|>R}.

The inherited continuous localization map and positive weighted charge estimate
give Q_e(u_n)->Q_e(u) in ambient H^(1/2)(R^6). Thus THIS fixed localized charge
belongs to the closure of C_c^infinity(D) in the ZERO-EXTENDED ambient norm.
Interior patch convergence alone would not imply this global closure statement.
The momentum-dependent Green operator need not preserve tangential support;
charge support above is justified by uniqueness of the coefficient, not support
preservation by that Green operator.

This is not an ordinary classical boundary trace at the critical exponent1/2.
Charge approximation alone does not approximate a whole singular state: its
regular and momentum-dependent Green parts must jointly respect the curved
exclusion boundary.

The missing lattice liminf statement, under specified compatible N=3 embeddings,
is: weak convergence u_a->u with uniformly bounded shifted lattice energies
implies u in D(q^D) and q^D[u]<=liminf q_a[u_a]. An estimate proving only finite
localized ambient energy cannot replace membership in the specified minimal
closure without an additional density theorem. The two-particle operator limit
and lattice stability do not establish that boundary theorem.

A sufficient new lemma would approximate every prospective boundary limit by
states supported away from the hard boundary in the full interacting form norm,
including charge-endpoint contributions. Derivatives of boundary cutoffs grow
as their layer shrinks; the interior estimates above depend on cutoff derivatives
and are NOT uniform enough to prove this boundary statement. Singular recovery
sequences and proof that the hard lattice selects this Friedrichs boundary
realization remain separate obligations. No new assumption or alternative
boundary realization is adopted to conceal the gap.

## 8. Provenance and proof-only verification

Sources, unchanged, relative to doc/derivations/:
- `cubic_three_particle_continuum_2026-09-13.md`: specified C_s, relative/contact
  forms, tensor domain, local inclusion and Friedrichs closure.
- `cubic_two_particle_operator_convergence_2026-09-13.md`: independently cleared
  two-particle generalized resolvent convergence; not a hard-boundary theorem.
The present interior-density/localization/contact trace arguments are separate
proposed lemmas, not already established by those sources.

Baseline674tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-three-particle-interior-contact-ihy9m8ox/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored files
and .git are excluded. This note is the sole new repository file. No scientific
imports, tests, numerical evaluations or model edits are authorized.

## 9. Independent review and correction record

Reviewer a613d9c5303de9b58 initially withheld clearance: the enlarged ambient
operator-test class incorrectly admitted arbitrary smooth regular functions,
whose nonzero contact trace can violate the zero-charge operator condition.
The original proof-draft-v1.md remains preserved externally. The repair restricts
regular additions to functions supported away from contact and singular additions
to exactly admitted germs. No physical hypothesis or claimed target was changed.
The fixed-global-localized charge and its support/closure justification were
also made explicit, without attributing support preservation to the Green operator.
The repaired submission is saved as proof-repaired-v2.md with its own hash.

On direct re-review, the same independent reviewer granted scoped clearance
with no remaining blocker: the interior localization, general-domain contact
identification and specified localized-charge closure statement are cleared.
Hard-boundary density, maximal-domain equality, boundary-uniform estimates and
lattice convergence remain unproved. No execution or empirical verification
was part of review. Independent auditor a65b8d2643520d219 confirmed all674baseline
files unchanged, exactly one new note and675total, matching final-hashes.json.
Both preserved versions match their recorded hashes. The initial failure and
narrow repair are retained honestly; v2-to-current changes are status/provenance
only. This is documentary/preservation clearance, not mathematical recertification
or evidence of absence of unrecorded execution. The audited final-hashes.json
is retained; closing-hashes.json binds these final audit-status edits.
