# Hard-lattice convergence at every fixed finite particle number

2026-09-14. Status: convergence synthesis at each fixed finite N independently
cleared; independent saved-document/hash audit completed.
This is a conditional nonrelativistic model theorem, not a thermodynamic limit,
empirical identification or a theory of everything.

## 1. Identical fixed-sector inputs

Fix integer N>=0 BEFORE a down to0. Retain supplied R,kappa>0, finite s!=0 and
 g_a^(-1)=-a²W/kappa+a³/(8pi kappa s). For N>=3, Omega_N imposes diameter>R
on every one of the binomial(N,3) label triples. For N<3 there is no exclusion;
N=0 uses C and zero vacuum energy.

Let H_(N,a)=ell²({n:an in Omega_N}). K_(N,a) is the zero-extension hard
compression of (kappa/a²)(6NI-Adj)+g_a sum_{i<j}1_{n_i=n_j}, with original
kinetic diagonal and forbidden-endpoint costs. Its form is q_(N,a)^hard.
On the common ambient H_N=L²(R^(3N)), let P_N=1_(Omega_N) and iota_N be
inclusion of the allowed continuum space. At N=0 all embeddings/projections
are identity on C.

The target H_N^D is the SPECIFIED Friedrichs operator from the finite-particle
construction, with form q_N^D. The global boundary-domain theorem identifies
its zero-extended domain with U=0 on Omega_N^c and every matching localization
chi_PU in the completed ambient tensor form space. This is global ambient
energy, not merely local finite energy, and it includes any finite matching depth.

Set m_N=floor(N/2), E_lat=8kappa*1_{s>0}/s², L=8pi/R and retain exactly

    M_N=3kappa N(N-1)L², C_N=m_N E_lat+M_N,
    sigma_N=C_N+eta, eta>0,
    b_(N,a)=q_(N,a)^hard+sigma_N||.||²,
    b_N=q_N^D+sigma_N||.||².

The reviewed inputs use the SAME model, form, physical-cell embedding and shift:
(i) I_(N,a)psi_a weakly->U implies b_N[U]<=liminf b_(N,a)[psi_a], with infinity
off the specified form domain;
(ii) every u in D(q_N^D) has exactly hard-supported strong recovery with convergent
b energy and q energy, symmetric when u is bosonic.
Both b forms are eventually >=eta||.||², so K_(N,a),H_N^D>=-C_N at this fixed N.
No unshifted weak lower-limit claim is substituted for(i).

## 2. Physical projections

Use centered physical cells of dimension3N and the amplitude a^(-3N/2). Let
Q_(N,a) be complete cell averaging, m_(N,a) the allowed-node cell mask, and
P_(N,a)=I_(N,a)I_(N,a)*. Then P_(N,a)=m_(N,a)Q_(N,a)=Q_(N,a)m_(N,a).
Complete cell averaging converges strongly to identity on fixed-N L², first
on compact continuous tests and then by density and contraction.

Each particle moves from its node by at most sqrt(3)a/2 within a centered cell.
Every triple diameter changes by at most sqrt(3)a. Except on the finite union
of triple equality sets d_T=R, all allowed decisions stabilize. These equality
sets lie in finitely many pair-distance-R hypersurfaces, null for R>0. Thus
m_(N,a)->P_N almost everywhere and, by dominated convergence,

    ||P_(N,a)f-P_Nf||<=||Q_(N,a)f-f||+||(m_(N,a)-P_N)f|| ->0.        (1)

The limit is P_N, not ambient identity when exclusion exists. Allowed-node
cells can cross the physical boundary at finite a. For N=1,2 the mask is1;
for N=0 all maps are exactly identity.

## 3. Extended generalized Mosco convergence

On common H_N define B_(N,a)(U)=b_(N,a)[I_(N,a)*U] for U in ran I_(N,a),
and +infinity otherwise. Define B_N analogously on iota_N D(q_N^D). These
are closed positive extended-valued forms; their finite domains need not be
dense in ambient H_N.

A finite weak liminf reduces to a subsequence of finite B_(N,a) and input(i).
Input(ii) supplies exact-energy strong recovery on the finite domain. Outside
that domain the recovery upper inequality has right side+infinity, so a
constant ambient sequence suffices. Consequently

    B_(N,a) -> B_N in Mosco sense, for each fixed finite N.          (2)

The general matching-depth and domain work is in the reviewed inputs, not
in an unverified replacement of4byN in a previous theorem.

## 4. Embedded resolvents and comparison completions

For lambda>0 and f in H_N let

    U_a=I_(N,a)(K_(N,a)+sigma_N+lambda)^(-1)I_(N,a)*f,
    U=iota_N(H_N^D+sigma_N+lambda)^(-1)iota_N*f.

They uniquely minimize B_(N,a)(V)+lambda||V||²-2Re<f,V> and its limit.
Coercivity bounds ||U_a|| by ||f||/(eta+lambda). Mosco liminf and recovery of
the unique limiting minimizer show every weak cluster point equals U and the
minimum values converge. For a strong exact-energy recovery V_a of U, the
quadratic-gap identity within the finite-a physical space gives

    F_a(V_a)-F_a(U_a)
      =b_(N,a)[I_(N,a)*(V_a-U_a)]+lambda||V_a-U_a||²
      >=(eta+lambda)||V_a-U_a||².

Hence U_a->U strongly. Fix lambda0>0 and call these embedded positive resolvents
R_a,R. Their spectra lie in [0,(eta+lambda0)^(-1)]. The continuous function
h_z(r)=r/[1-(z+sigma_N+lambda0)r], h_z(0)=0, has no pole there for nonreal z.
Uniform polynomial approximation and convergence of bounded powers yield

    I_(N,a)(K_(N,a)-z)^(-1)I_(N,a)*
      ->iota_N(H_N^D-z)^(-1)iota_N* strongly, Im z!=0.   (3)

The complementary kernels are retained; these are generalized embedded resolvents,
not ordinary resolvents on all H_N when forbidden modes exist.

For comparison only extend transported K_(N,a) by0 on ran(I-P_(N,a)), and
transported H_N^D by0 on ran(I-P_N). These are ordinary self-adjoint ambient
operators tilde K_(N,a), tilde H_N^D. Their resolvents add
-z^(-1)(I-P_(N,a)) and -z^(-1)(I-P_N) to(3). Equation(1) proves ordinary
strong-resolvent convergence of these completions. Both retain the common
FIXED-N lower bound -C_N since C_N>=0. No physical forbidden-region dynamics
or norm-resolvent convergence is asserted by this comparison device.

## 5. Vectorwise compact-time evolution

Let A_a,A be the comparison completions and fix f in H_N. Choose0<=chi_m<=1
in C_c(R), equal1 on expanding spectral intervals. Spectral calculus gives
||(I-chi_m(A))f||->0; strong-resolvent convergence gives
chi_m(A_a)f->chi_m(A)f at fixed m. Therefore

    limsup_a||(I-chi_m(A_a))f||<=||(I-chi_m(A))f||.       (4)

For fixed m,T, exp(-itx)chi_m(x), |t|<=T, is a compact uniform-norm family
in C_0(R). Finite nets give strong convergence uniformly on that family for
the fixed f. Unitarity bounds omitted tails by(4), uniformly in t. Removing m
proves compact-time strong unitary convergence of completions. Subtract their
exact complementary identity terms and apply(1):

    sup_|t|<=T ||[I_(N,a)exp(-itK_(N,a))I_(N,a)*
                     -iota_N exp(-itH_N^D)iota_N*]f|| ->0.           (5)

For heat evolution the common lower bound gives norm at most exp(C_N T) on
0<=t<=T. The compact spectral family exp(-tx)chi_m(x) and the same tails then
give the heat counterpart of(5), uniformly on that time interval. This fixed-N
bound is not uniform as N grows.

Both statements include t=0, when they reduce to P_(N,a)f->P_Nf. The embedded
physical unitary families are not unitary groups on the full ambient space.
The spectral-tail argument is needed because exp(-itx) is not a C_0 function.

For moving allowed initial states I_(N,a)psi_a->iota_Nu strongly, contraction
of embedded unitaries and(5) imply

    sup_|t|<=T ||I_(N,a)exp(-itK_(N,a))psi_a
                         -iota_N exp(-itH_N^D)u|| ->0. (6)

The heat analogue uses exp(C_N T). No finite-energy assumption on u is required.
These are fixed-vector and strongly moving-state conclusions, not operator-norm
convergence over all initial vectors or all time.

## 6. Bosons, small sectors and reference phases

Particle permutations preserve every triple condition, pair contact, cell and
continuum initial-domain prescription. The theorem restricts to bosons. On
the symmetric ambient space projections converge to P_N restricted there;
on the full ordered space bosonic range projections converge to P_N P_sym,
not P_N alone. Individual matching localizations need not be symmetric.

At N=0 the identity embedding and zero operator give sigma_0=eta. At N=1 the
target is free -kappa Delta, M_1=0. At N=2 it is EXACTLY the established full
pair operator with contact g_a, not2g_a, and no hard exclusion. The R-dependent
proof shift does not modify that model. The reviewed construction identifies
N=3 and4 with their earlier initial prescriptions and Friedrichs forms, so the
general result retains, rather than replaces, those sector theorems.

If H_(N,a)^hop omits the threshold diagonal, then

    K_(N,a)=H_(N,a)^hop+6Nkappa/a²,
    exp(-itH_(N,a)^hop)=exp(it6Nkappa/a²)exp(-itK_(N,a)).

The convergence theorem retains the threshold reference and does not drop
this divergent phase. In contrast sigma_N is a common auxiliary shift at
FIXED N; its common physical phase cancels in comparisons. Because sigma_N
depends on N, it is not one scalar shift on full Fock space. Zero completion
of a shifted physical operator is not the same as shifting the entire completed
ambient operator, and those operations are not conflated here.

## 7. Exact achievement and limits

Conditional on independent review of this synthesis and the cited inputs,
(2)–(6) identify at every fixed finite N the hard-lattice limit with the
SPECIFIED Friedrichs realization: generalized Mosco convergence, embedded
strong-resolvent convergence, and vectorwise compact-time unitary/heat evolution.
Boundary selection uses the GLOBAL domain equality and recovery of every form
vector, not only interior contact conditions or two-body scattering.

This is a conditional NONRELATIVISTIC hard-constraint contact model with supplied
R,s,kappa and cubic regularization. It does not derive those parameters, repair
the original ring into a TOE, identify observed particles or supply gravity.
No norm-resolvent or operator-norm dynamics, rate, uniformity in N, joint N,a
limit, thermodynamic convergence, all-time uniform convergence, unbounded-
observable or scattering-completeness statement is established. Maximal operator
traces and uniqueness among all conceivable boundary extensions remain outside
scope. Extensive sector stability is not an N-independent bound on unshifted
full-Fock energy or a uniform approximation estimate.

The next bounded mathematical question is a Fock direct-sum truncation lemma
for embedded nonreal resolvents and possibly compact-time unitaries. Fixed
Fock vectors can be truncated to finitely many sectors; nonreal resolvents
have norm<=1/|Im z| and unitaries norm1 independently of N. That is a separate
argument, not a conclusion already claimed here. The unrestricted full-Fock
heat argument does not follow from exp(C_N T), which grows with N. Nor would
resolvent/unitary convergence alone produce a physical chemical potential,
preferred vacuum or full-Fock semiboundedness.

## 8. Sources and preservation

Protected references, relative to doc/derivations/:
- `cubic_finite_particle_lattice_liminf_2026-09-14.md`: shifted weak lower limit,
  exact cell convention and fixed-N positive bounds.
- `cubic_finite_particle_lattice_recovery_2026-09-14.md`: hard recovery for every
  form vector including bosons and all cross-channel energies.
- `cubic_finite_particle_boundary_density_2026-09-14.md`: global specified domain.
- `cubic_finite_particle_continuum_2026-09-14.md`: intrinsic family, small-sector
  identifications and extensive continuum stability.
- `cubic_four_particle_convergence_2026-09-14.md`: synthesis strategy, with
  every general finite-N hypothesis checked above rather than presumed.

Baseline690tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-finite-particle-convergence-dvvy70k4/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored files
and .git are excluded. Only this note is added. No scientific imports, tests,
numerical evaluations or model edits are authorized.

## 9. Independent mathematical review

Reviewer acbf87201edc9a576 cleared the synthesis at each fixed finite N with
no mathematical blocker or required repair. They checked identical inputs and
shifts, all-triple projection limits, extended Mosco and coercive minimizers,
resolvent signs/completions, spectral-tail compact-time dynamics, small sectors,
bosonic conventions, threshold phases and exact specified-boundary selection
through global domain equality and recovery of every form vector.

Clearance does not extend to uniformity in N, Fock or thermodynamic convergence,
operator-norm dynamics, empirical identification or uniqueness among other
boundary extensions. Direct reads only were used. Original proof-draft-v1.md
and its hash are preserved; no mathematical constant or physical hypothesis
changed. Independent auditor ad2f44a9abd4852f3 confirmed all690baseline files
unchanged, exactly one new note and691current files matching final-hashes.json.
Mathematical sections1–7 are byte-identical to the draft; the note matched its
pre-audit copy. This is artifact clearance, not mathematical recertification
or proof of absence of unrecorded execution. The audited final-hashes.json is
retained; closing-hashes.json binds final audit-status edits.
