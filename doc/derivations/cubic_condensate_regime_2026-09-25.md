# Controlled condensate regime of the supplied cubic Bose lattice

2026-09-25. Status: new conditional mathematics with an implementation
(`bpr/cubic_condensate_regime.py`), independent tests
(`tests/test_cubic_condensate_regime.py`) and a demo
(`scripts/demo_cubic_condensate_regime.py`). An independent adversarial review
is recorded in section 9. This is not empirical validation, not a
thermodynamic-limit theorem and not a completion of BPR.

## 0. Why this calculation, and what it does not touch

The [relativity gate](cubic_relativity_gate_2026-09-14.md) left one collective
route open: a phase/density sound mode. It listed as missing prerequisites a
stationary state with phase rigidity, positive compressibility A and stiffness B,
and a source-normalized pole and residue. The
[equilibrium gate](cubic_equilibrium_transfer_gap_2026-09-14.md) stopped the
tuned continuum contact/hard-exclusion model before any of these.

This note works on a DIFFERENT member of the same supplied family: the lattice
model of [supplied_cubic_bose](supplied_cubic_bose_2026-09-13.md) with its own
admitted parameter domain C>0, g>=0, at fixed lattice spacing. No field, link,
interaction, constraint or chemical potential is added. The equilibrium gate for
the attractive continuum model is not closed by anything below. Repulsive lattice
contact scattering is trivial in the continuum limit
([two-body scattering, eq. 11](cubic_two_body_scattering_2026-09-13.md)), so
nothing here is a statement about that limit.

What is new: (1) an exact vacuum-selection theorem inside every fixed-number
sector; (2) a self-contained proof of complete condensation and of the
Bogoliubov excitation spectrum in the mean-field regime at fixed lattice size;
(3) exact inequalities for the window where the resulting phonon is
Lorentz-invariant; (4) exact lattice stability criteria for current-carrying
condensates, and the identification of their long-wave cone with a Lorentzian
acoustic metric whose signature and ergoregion match those criteria. These
results give the cubic line its first controlled interacting collective regime.
They also show concretely what kinematic "emergent relativity" does and does
not supply.

## 1. Model and notation

For n>=3 let G_n=C_n□C_n□C_n, M=n³ sites, degree 6, unordered simple edges.
On unrestricted bosonic Fock space

    H = (g/2) Σ_x n_x(n_x-1) - C Σ_{xy∈E} (a_x† a_y + a_y† a_x),   C>0, g>=0.

H_N is its restriction to the N-particle sector. Plane waves are
φ_m(x)=M^(-1/2) exp(i k_m·x) with k_m=2πm/n, m∈Z_n³, and b_m=Σ_x conj(φ_m(x)) a_x.
The one-body operator h=-CA has h φ_m=(ε_m-6C)φ_m, where

    ε_m = 2C Σ_j (1-cos k_{m,j}) = 4C Σ_j sin²(π m_j/n),   ε_0=0,
    ε_* := min_{m≠0} ε_m = 4C sin²(π/n) > 0.

The number of excitations is N_+=Σ_{m≠0} b_m†b_m. For the mean-field regime fix
λ>=0 and set g=g_N=λ/(N-1) for N>=2, with μ:=λ/M. Then g_N N/M → μ, so μ is
the mean-field interaction energy per particle.

## 2. Exact sector vacuum (all N, all real g)

**Theorem 1.** For every n>=3, C>0, real g and N>=1:
(i) E_0(N)=min spec H_N is a simple eigenvalue;
(ii) its eigenvector Ω_N has strictly positive coefficients in the occupation
basis, after one overall phase choice;
(iii) Ω_N is invariant under the unitary action of every graph automorphism of
G_n, not only the translations and signed coordinate permutations. For n=4,
C_4³ is the 6-cube, which has more automorphisms; they are included. Hence
Ω_N has total momentum zero and is inversion and point-group invariant;
(iv) Ω_N has strictly positive overlap with every nonzero vector of
nonnegative occupation coefficients, in particular with the pure condensate
(b_0†)^N|0>/sqrt(N!).

*Proof.* In the occupation basis H_N=D-CT. Here D is real diagonal. T has entry
sqrt(ν_y(ν_x+1))>0 between occupations related by one hop along an edge, and
zero otherwise. Because G_n is connected, any N-particle occupation can be
moved to any other by single hops, so the hop graph on occupations is connected
and T is irreducible. For s>=max D, the matrix sI-H_N is nonnegative and
irreducible. Perron–Frobenius gives a simple top eigenvalue s-E_0(N) with a
strictly positive eigenvector. An automorphism π permutes occupation kets and
commutes with H_N. It maps Ω_N to a positive ground vector, which by simplicity
equals c·Ω_N; positivity and unit norm force c=1. Positive vectors have
positive inner products with nonzero nonnegative vectors. ∎

The positivity argument does not use g>=0. The supplied domain is g>=0, which
is all that is used below. Theorem 1 selects the vacuum within a supplied
population; it does not select N. Sector selection was the separate issue
treated for the ring in [vacuum selection](substrate_vacuum_selection_2026-09-12.md).
The gap E_1(N)-E_0(N)>0 is not claimed uniform in N or n.

**Corollary 1 (exact lattice f-sum identity).** Let ρ_q†=Σ_x exp(i q·x) n_x
and T_j=-C Σ_x (a†_{x+e_j}a_x+h.c.). By inversion invariance of Ω_N,

    Σ_ν (E_ν-E_0) |<ν|ρ_q†|Ω_N>|² = Σ_j (1-cos q_j) <Ω_N,(-T_j)Ω_N>.      (F)

The double commutator [ρ_q,[H,ρ_q†]] equals 2Σ_j (1-cos q_j)(-T_j): the
interaction commutes with every n_x. Inversion equates the ρ_q and ρ_q†
spectral sums, which produces the factor 1/2.

## 3. Hartree functional

**Lemma 2.** For λ>=0 and unit u∈C^M let
E_H[u]=<u,hu>+(λ/2)Σ_x|u_x|⁴. Then min E_H=e_H:=-6C+μ/2, attained exactly at
u=e^{iθ}φ_0.

*Proof.* A<=6 with equality only on constants, since G_n is connected and
6-regular. Cauchy–Schwarz gives Σ|u_x|⁴>=(Σ|u_x|²)²/M=1/M, with equality iff
|u_x| is constant. The uniform vector attains both lower bounds simultaneously.
Any minimizer must attain both, hence is constant. ∎

The product state φ_0^{⊗N} has <H_N>=N e_H exactly, because
(g_N/2)N(N-1)/M=Nμ/2. Hence E_0(N)<=N e_H.

## 4. Exact excitation-map identity and complete condensation

Let U_N be the unitary from the N-boson space onto the truncated excitation
Fock space F_+^{<=N} over span{φ_m: m≠0}. It removes the condensate mode,
b_0†b_0 ↦ N-N_+. This is the construction of Lewin–Nam–Serfaty–Solovej
(CPAM 68, 2015), specialized here to a finite one-body space. Sorting the
momentum-conserving interaction by the number of condensate indices, with
nothing dropped, gives

    U_N H_N U_N* = N e_H + 𝔾_N,
    𝔾_N = Σ_{m≠0}(ε_m+μ) n_m
        + (μ/2) Σ_{m≠0} ( b_m† b_{-m}† Θ_N + Θ_N b_{-m} b_m )
        + T_3 + T_4 - (3μ/(2(N-1))) N_+(N_+-1),                           (1)

where:

- Θ_N=sqrt((N-N_+)(N-N_+-1))/(N-1) is a function of N_+ in [0,1];
- T_3=(μ/(N-1)) Σ'_{p,q} ( b_p†b_q† b_{p+q} sqrt(N-N_+) + h.c. ), with p, q
  and p+q all nonzero;
- T_4=(g_N/2)Σ_x (c_x†)²c_x² >= 0, with c_x=Σ_{m≠0}φ_m(x)b_m.

The diagonal coefficient is exact: the n_0(n_0-1) and 4n_0N_+ terms combine to
μN/2+μN_+-(3μ/(2(N-1)))N_+(N_+-1).

**Theorem 3 (coercivity and complete condensation).** For N>=2,

    𝔾_N >= ε_* N_+ - μ(M-1)        on F_+^{<=N}.                          (2)

Consequently every normalized vector with energy <= N e_H+E satisfies
<N_+> <= (E+μ(M-1))/ε_*. In particular Ω_N has
<N_+> <= λ(M-1)/(M ε_*), uniformly in N. Its condensate fraction
<n_0>/N >= 1-λ/(ε_* N) tends to one.

*Proof.* The hopping part of H_N equals -6CN+Σ_{m≠0}ε_m n_m, which is at least
-6CN+ε_* N_+. Occupation numbers satisfy Σ_x n_x² >= N²/M on every occupation
ket. So the interaction is at least (g_N/2)(N²/M-N)=Nμ/2-μN(M-1)/(2(N-1)),
and N/(N-1)<=2. ∎

By Theorem 1 the one-body density matrix of Ω_N commutes with translations, so
it is diagonal in the φ_m. Its top eigenvector is φ_0, with eigenvalue <n_0>.
This is exact finite-volume Bose–Einstein condensation in the mean-field
regime. It is not a thermodynamic phase transition.

**Theorem 4 (Bogoliubov spectrum at fixed lattice size).** Let

    H_B = Σ_{m≠0}(ε_m+μ) n_m + (μ/2) Σ_{m≠0}(b_m† b_{-m}† + b_{-m} b_m)

on the full excitation Fock space F_+. Put
e_m=sqrt(ε_m(ε_m+2μ)) and E_B=-(1/2)Σ_{m≠0}(ε_m+μ-e_m). The spectrum of H_B
is E_B+Σ_{m≠0} ν_m e_m over ν_m∈{0,1,2,...}. For n even, self-conjugate modes
(2k_m≡0) use single-mode squeezing and obey the same formulas. Let
β_1<=β_2<=... be these values with multiplicity, and E_1(N)<=E_2(N)<=... the
eigenvalues of H_N. Then for every fixed j,

    lim_{N→∞} [E_j(N) - N e_H] = β_j,        at fixed n, C, λ.           (3)

In particular E_0(N)-N e_H → E_B, and the excitation energies converge to the
sums Σν_m e_m.

*Proof.* Constants K depend only on n, C, λ and j, and are not optimized.

Step 1 bounds each term of (1) against H_B, for ψ∈F_+^{<=N}:
- (a) Pairs: ±(b_m†b_{-m}†+h.c.) <= n_m+n_{-m}+1 for distinct partners, and
  2n_m+1 for self-conjugate modes.
- (b) Pairing remainder: 0<=Θ_N<=1 and |Θ_N-1|<=(N_++1)/(N-1). Cauchy–Schwarz
  then gives |<P_N-P_∞>| <= μ sqrt(M) <(N_++1)²>/(N-1), where P_N and P_∞ are
  the pairing terms of 𝔾_N and H_B.
- (c) Cubic term:
  |<T_3>| <= 2μ sqrt((M-1)N)/(N-1) <N_+²>^{1/2}<N_+>^{1/2}. This is at most
  4μ sqrt(M/N) <(N_++1)²> and, because N_+<=N, at most 4μ sqrt(M)<N_+>.
- (d) Quartic term: writing T_4 through Σ_r B_r†B_r with B_r=Σ_{k1+k2=r}b_{k2}b_{k1}
  gives 0<=T_4<=μ(M-1)N_+(N_+-1)/(2(N-1)).

Step 2 is localization. Take smooth f,h with f²+h²=1, f=1 on [0,1/2] and f=0
on [1,∞). Put f_L=f(N_+/L) and h_L=h(N_+/L). The pairing and cubic terms move
N_+ by at most two. So the IMS identity
𝔾-f𝔾f-h𝔾h=(1/2)Σ_{φ∈{f,h}}[φ,[φ,𝔾]] has a remainder of operator norm at
most K(1/L+L^{-1/2}N^{-1/2}). The pairing term has size of order L in the
transition region, and the double commutator brings L^{-2}.

Step 3 is the lower bound. On the range of f_L we have N_+<=L. Drop T_4>=0 and
use (b)–(c) to get f𝔾f >= f(H_B-δ_{N,L})f with δ_{N,L}=K(L+1)²/sqrt(N). On the
range of h_L, (2) gives h𝔾h >= (ε_*L/2-μ(M-1))h². Write Π for the projection of
H_B onto its first j-1 eigenvectors. Then H_B >= β_j-(β_j-β_1)Π, and a rank
argument gives

    E_j(N)-N e_H >= min(β_j-δ_{N,L}, ε_*L/2-μ(M-1)) - K(1/L+(LN)^{-1/2}).

Step 4 is the upper bound. Bogoliubov eigenvectors are finite excitations of a
quasi-free state, so they have all moments of N_+. Use f_LΦ_1,...,f_LΦ_j as
trial states. Their Gram matrix is I+O(1/L); H_B satisfies the same IMS bound;
(b)–(d) with T_4 included bound the energy difference by K(L+1)²/sqrt(N).
Min–max gives the matching upper bound.

Choosing L=N^{1/6} makes every error O(N^{-1/6}) and the h-range threshold
diverge. ∎

This is the Lewin–Nam–Serfaty–Solovej / Seiringer (CMP 306, 2011) method with
explicit finite-dimensional constants. It establishes neither the optimal rate
nor uniformity in n; the numerics below suggest the gap error is O(1/N). The
limit is taken at fixed lattice size: it is not the thermodynamic limit and not
the dilute Gross–Pitaevskii regime.

**Residue at Bogoliubov level.** In H_B, ρ_q†Ω maps to
sqrt(N)(u_q-v_q)|one phonon at q>, with (u_q-v_q)²=ε_q/e_q. The one-phonon
density residue per particle is therefore ε_q/e_q, and e_q·(ε_q/e_q)=ε_q
saturates the leading term of (F). This residue statement is at Bogoliubov
level; eigenvector convergence is not proved in this note. The bounded ED
checks below show Z/N approaching ε_q/e_q.

## 5. Acoustic window: exact Lorentz-violation bounds

Restore a spacing a>0. Put κ=Ca² and q=k/a with k_j∈(-π,π]. Then
ε(q)=4CΣ_j sin²(aq_j/2) and e(q)²=ε(ε+2μ). Define

    c_s² := 2μκ,    ξ² := κ/(2μ)   (healing length).

**Theorem 5.** For every lattice momentum q,

    -2μ b(q) <= e(q)² - c_s²|q|² <= κ²|q|⁴,    b(q)=κa²Σ_j q_j⁴/12,         (4)

and hence for q≠0

    -(a²/12) Σ_j q_j⁴/|q|² <= e(q)²/(c_s²|q|²) - 1 <= ξ²|q|²,            (5)
    | e(q)/(c_s|q|) - 1 | <= max(a²/12, ξ²) |q|².                         (6)

*Proof.* e²-c_s²|q|²=ε²-2μ(κ|q|²-ε). The inequalities
0<=κ|q|²-ε<=b(q) are from
[supplied_cubic_bose eq. 5](supplied_cubic_bose_2026-09-13.md). (6) follows from
|x-1|<=|x²-1| for x>=0 and Σq_j⁴<=|q|⁴. ∎

Expanding, e²=c_s²|q|²[1+ξ²|q|²-(a²/12)Σq_j⁴/|q|²]+O(|q|⁶). The isotropic
Bogoliubov term is superluminal. The lattice term is subluminal and
anisotropic: its coefficient ranges over [a²/36, a²/12] across directions. No
single ξ cancels it in every direction, so the leading Lorentz violation is
anisotropic unless ξ>>a. The rest frame of the condensate is a preferred frame.

On a finite lattice the smallest nonzero |q| is 2π/(na). At the lowest phonon,
the bound (6) is at most δ iff n>=π/sqrt(3δ) and μ>=2π²C/(δn²). This is a
sufficient condition; the actual deviation can be smaller. For example, δ=0.01 needs n>=19 and a correspondingly large μ.
Combined with Theorem 4, the low-lying many-body excitations at such q approach
a single linear branch with speed c_s as N→∞ at fixed n. This is the precise
content of "emergent Lorentz invariance" here: one scalar mode, one speed,
violations bounded by (6).

**Link to the relativity gate's conditional Lagrangian.** At Hartree level the
energy density is gρ²a³/2 (A=ga³>0). A twisted condensate φ_p costs Nκ|p|²
(B=2κρ, with superfluid fraction one at this order). Then AB=2κgν→c_s². So
the gate's L_2 with c_s²=AB is realized in this controlled regime, with A and B
obtained rather than assumed. Depletion corrections to B are O(1/N) here and
are not computed.

## 6. Current-carrying condensates, acoustic metric and stability

Lattice units a=1. For p=2πm_p/n the plane wave φ_{m_p} satisfies the Hartree
equation hφ+λ|φ|²φ=(ε(p)-6C+μ)φ exactly, because |φ|² is uniform. So it is a
stationary solution of time-dependent Hartree dynamics. Linearizing about it
couples modes p+k and p-k. The Bogoliubov–de Gennes block is
[[A_k, μ],[μ, B_k]] with A_k=ε(p+k)-ε(p)+μ and B_k=ε(p-k)-ε(p)+μ. Define

    S(k) = (A_k+B_k)/2 - μ = 2C Σ_j cos p_j (1-cos k_j),
    D(k) = (A_k-B_k)/2    = 2C Σ_j sin p_j sin k_j,
    ω_±(k) = D(k) ± sqrt( S(k)(S(k)+2μ) ).                               (7)

This covers the self-conjugate case 2k≡0 as well.

**Theorem 6 (exact finite-lattice criteria, Hartree–Bogoliubov level).**
(a) The condensate at p is linearly (dynamically) unstable if S(k)(S(k)+2μ)<0
for some k≠0. It is linearly stable, with a diagonalizable real spectrum, if
S(k)(S(k)+2μ)>0 for all k≠0. Equality cases are marginal and not classified.
(b) The Hessian of E_H-(ε(p)-6C+μ)||u||² at φ_{m_p} is positive definite on the
complement of φ_{m_p} (energetic or Landau stability) iff for every k≠0:
S(k)>0 and D(k)²<S(k)(S(k)+2μ).
(c) Energetic stability implies dynamical stability. The condensate at p=0 is
energetically stable for all μ>=0.

*Proof.* (7) is the eigenvalue equation of σ_z[[A,μ],[μ,B]]. Positive
definiteness of that block is A>0 and AB>μ². Since AB-μ²=S(S+2μ)-D², and
S<-2μ forces A+B<0, this reduces to (b). ∎

**Theorem 7 (long-wave acoustic metric).** Let v=∇ε(p)=(2C sin p_j) and
W=μ·Hess ε(p)=2μC diag(cos p_j). For k→0, (7) gives the cone
(ω-v·k)²=k^T W k+O(|k|³). Define

    g^{00}=-1,   g^{0i}=-v^i,   g^{ij}=W^{ij}-v^i v^j,

so that g^{μν}k_μk_ν=0 with k_μ=(-ω,k) is exactly the cone. Then:
(i) the Schur complement of g^{00} is W, so g^{μν} has Lorentzian signature iff
W is positive definite, i.e. iff cos p_j>0 for all j. In that case
S(k)=2CΣ_j cos p_j(1-cos k_j)>0 for EVERY k≠0, so the condensate is
dynamically stable at all lattice momenta, not only at long wavelength.
Suppose instead cos p_j<0 for some j. Then S(k e_j)∈(-2μ,0) whenever
2C|cos p_j|(1-cos k)<2μ, which gives a complex BdG pair and dynamical
instability. The lattice resolves such a k whenever
2C|cos p_j|(1-cos(2π/n))<2μ, and for every n large enough at fixed p/(2π)
and μ. Cases with cos p_j=0 have a degenerate cone and are marginal.
**At the resolution where the lattice supplies long-wave modes, the emergent
cone is Lorentzian exactly when the condensate is dynamically stable.**
(ii) For W>0 the covariant metric is g_00=-(1-v^TW^{-1}v), g_0i=-(W^{-1}v)_i,
g_ij=(W^{-1})_ij. Long-wave Landau stability, sup_k (v·k)²/(k^TWk)<1, is
exactly v^TW^{-1}v<1, i.e. g_00<0: no ergoregion. A supercritical current is
an acoustic ergoregion.
(iii) As a→0 at fixed physical p, v→2κp and W→c_s²I. This recovers the
standard acoustic metric of Unruh (1981) and Visser (1998) for a uniform flow,
up to a conformal factor.

Long-wave Landau stability (ii) and exact finite-lattice energetic stability,
Theorem 6(b), are different statements. On a coarse lattice the supercritical
long-wave modes may not exist. For example, n=5, p=(2π/5,0,0), μ=4 has
v^TW^{-1}v≈1.46, an acoustic ergoregion. Yet it satisfies 6(b) at every
resolved k: at the smallest k, D²≈3.27<S(S+2μ)≈3.60. Refining the lattice
exposes the instability. Exact marginal cases also occur: n=8,
p=(π/4,0,0), μ=1 has D²=S(S+2μ)=1 at k=(π/4,0,0), because
(√2-1)(√2+1)=1. The implementation reports these as unresolved rather than
classifying them.

The dynamical-instability condition is the lattice (Wu–Niu type) instability
of condensates in periodic potentials, here derived for the unchanged cubic
model. Only uniform flows are treated. Curved acoustic geometries and horizons
need slowly varying backgrounds and an eikonal or adiabatic theorem that is
not supplied.

## 7. What this establishes and what it does not

| Requirement (foundation decision / relativity gate) | Status after this note |
|---|---|
| Vacuum selection inside a supplied sector | **Exact** (Theorem 1), all N, n, g |
| Population / sector selection | Not addressed; N is supplied |
| Controlled large-system regime | **Mean-field at fixed n** (Theorems 3–4); no thermodynamic limit |
| Positive A, B and sound speed | **Hartree/Bogoliubov level** (sections 4–5) |
| Source-normalized pole and residue | Pole: **rigorous** spectral limit. Residue: Bogoliubov level plus bounded ED |
| Common limiting speed | Trivial: one scalar species. Nothing about other fields |
| Lorentzian cone / metric | **Kinematic, uniform flow** (Theorem 7) |
| Metric dynamics / gravity | **Not supplied.** The acoustic metric follows condensate hydrodynamics, not Einstein equations (Barceló–Liberati–Visser, Living Rev. Rel. 2005/2011) |
| Photons / gauge fields | **Not supplied.** One scalar phonon; no transverse polarizations |
| Chiral fermions | **Not supplied.** Bosonic condensate |

A single scalar sound mode with bounded Lorentz violation is an analogue
spacetime, not physical spacetime. Emergent relativity would require:
- several species sharing one cone, with the second species not present in this
  model;
- tensor dynamics;
- a way around Weinberg–Witten for any emergent massless spin-2 state;
- a mapping of Lorentz-violation scales to real particle data.

Thermodynamic-limit BEC for interacting lattice bosons is proved rigorously
only in special cases. Hard-core bosons at half filling in d>=3 (equivalently
the spin-1/2 XY model) are one such case, via reflection positivity and
infrared bounds (Dyson–Lieb–Simon 1978; Kennedy–Lieb–Shastry 1988;
Aizenman–Lieb–Seiringer–Solovej–Yngvason 2004). That corresponds to the g→∞
member of this family. It is cited as context only; none of it is re-derived
or imported into a claim here.

## 8. Bounded numerical controls

`bpr/cubic_condensate_regime.py` checks the following on n=3, with C=1 and
fixed λ values. There are no fits or tuned tolerances beyond the stated
floating allowances.

- **Theorem 1:** the position-basis ground vector is positive and simple, with
  invariance residuals under translation, inversion and coordinate permutation.
  N<=3, g∈{0,1,5}.
- **Basis cross-check:** momentum-sector and position-basis spectra agree.
- **Theorem 3:** the coercivity bound (2) holds in the tested sectors, and
  depletion stays below its bound.
- **Theorem 4:** ED is run for N=2..5 at λ∈{5,20} and compared with E_B and
  e_(1,0,0). The same runs report the residue Z/N against ε/e and the exact
  identity (F).
- **Theorem 5:** inequalities (4)–(6) are checked for every mode, n=3..16.
- **Theorems 6–7:** formula (7) is compared with direct diagonalization of the
  full 2M×2M BdG matrix. The stability and metric equivalences are checked for
  every p on n=3..8, at several μ.

At λ=20 the gap error E_1-E_0-e_(1,0,0) is -0.491, -0.373, -0.294, -0.239 and
-0.200 for N=2..6, so N×error is roughly -1.2. The residue Z/N is 0.924,
0.888, 0.868 and 0.856 for N=2..5, against ε/e=0.818. These are finite
diagnostics, not proofs of rate.

## 9. Review record

See the end of this file once the independent review is complete.
