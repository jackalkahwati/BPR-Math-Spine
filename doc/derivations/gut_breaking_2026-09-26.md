# Breaking Spin(10) to the Standard Model in BPR-6D: geometric routes are obstructed

2026-09-26. Status: exact weight and root algebra, exhaustive flux scans, and
explicit Wigner rotation matrices. The implementation is
`bpr/gut_breaking.py`, with tests in `tests/test_gut_breaking.py` and a demo
in `scripts/demo_gut_breaking.py`. An independent review (section 5) found no blocker. It
strengthened the flux result with an instability argument, now Lemma 0, and
its repairs are applied.

## 0. Question

BPR-6D has an unbroken Spin(10) in four dimensions. Can the sphere itself
break it to SU(3) × SU(2) × U(1)_Y while keeping three complete families?
There are two geometric mechanisms to check:
- an abelian flux inside Spin(10);
- an orbifold of the sphere.

**Conventions.** The Cartan basis is e₁…e₅, with colour in planes 1–3 and weak
isospin in planes 4–5. The weights of the 16 are (±½)⁵ with an odd number of
minus signs; this set carries Q, u^c, d^c, L, e^c, ν^c with the standard
hypercharges (tested, together with ΣY = ΣY³ = 0). The generators are:
- Y = (e₄+e₅)/2 − (e₁+e₂+e₃)/3;
- X = e₁ + … + e₅;
- X ⊥ Y in the trace form.

## 1. Flux breaking

**Scope.** On S², every Yang–Mills critical point is a constant Cartan flux,
for any metric (Atiyah–Bott). Analysing Cartan fluxes h is therefore
exhaustive for Yang–Mills backgrounds; there are no instantons in two
dimensions. Quantization requires h to be integral with Σh_i even (the D₅
coroot lattice), which is what makes w·h integral on the 10 and the 16.

**Lemma 0 (every Spin(10) flux is unstable).** Every nonzero quantized h has
a root α with |α·h| ≥ 2:
- two nonzero entries give |h_i| + |h_j| ≥ 2;
- a single nonzero entry must be even, so it is at least 2.

The W bosons along such a root see monopole number |n| ≥ 2. By round 3,
Proposition 5b, their lowest level is tachyonic, with m²r² = −|n|/2 (checked
there against Atiyah–Bott and a Yang–Mills Hessian). Since
π₁(Spin(10)) = 0, the flux can relax, and it does.

Two checks:
- An exhaustive scan of the 8,402 nonzero quantized fluxes with |h_i| ≤ 3
  finds none stable.
- The Standard-Model-preserving flux (−2,−2,−2,−3,−3) has max |α·h| = 6.

**Spin(10) flux backgrounds on the round sphere are therefore not vacua.**
This is the primary obstruction. It differs from F-theory, where hypercharge
flux is stable (BPS) on a Kähler surface; S² has no analogue.

Everything below holds even if some unspecified mechanism stabilized the flux.

**Lemma 1 (chirality neutrality).** Take any Cartan flux h in Spin(10), with
the U(1)_F flux m=1 and parent charge 3. A 16 weight w then has index:
- 3 + w·h from the 16₊;
- −w·h from the 16₋, which has the same Spin(10) representation but the
  opposite 6D chirality.

The net count is exactly 3 for every Standard-Model multiplet. Spin(10) flux
would never spoil the family count; it would only add vector-like pairs.

The test computes the anomalies of the full zero-mode spectrum (both 16s, both
chiralities) for random quantized fluxes. All of them vanish: Y³, grav–Y,
SU(3)³, SU(3)²Y, SU(2)²Y and Witten's SU(2) anomaly.

**Theorem 2 (even if stabilized, no flux reaches the Standard Model with a
massless hypercharge).** The round-4 Green–Schwarz class
Y_g = 3λ_V + 9x² − λ_T contains the Spin(10) instanton density. Expanding
around the background gives the 4D BF vector (18m, 3h) on (F_F, F₁…F₅).
This is derived symbolically in `bf_vector` and tested. The Spin(10) direction
that becomes massive (Stückelberg) is therefore h itself.

Two further facts:
- Keeping SU(3) × SU(2)_L unbroken forces h ⊥ the Standard-Model roots, i.e.
  h = (a,a,a,b,b) ∈ span(Y, X).
- The massless Spin(10) U(1) is then h^⊥ within that plane. It equals Y only
  if h ∝ X. But h ∝ X leaves the whole SU(5) unbroken (20 roots).
- The flipped hypercharge Y′ = (⅓,⅓,⅓,½,½), which also lies in the plane,
  behaves the same way. Keeping Y′ massless needs h ∝ (1,1,1,−1,−1), whose
  centralizer is flipped SU(5), again 20 roots.

Hence no flux yields SU(3) × SU(2) × U(1)_Y with a massless hypercharge boson.
An exhaustive scan over integral h with |h_i| ≤ 3 finds fluxes whose
nonabelian centralizer is exactly SU(3) × SU(2) (8 roots). None of them leaves
Y massless.

This is the sphere analogue of the F-theory hypercharge-flux problem. There,
hypercharge flux is harmless only when its class is trivial in the base; on
S² the flux class is the whole second cohomology.

**Why two axions do not help.** The non-chiral 2-form yields two 4D axions:
- ∫_{S²}B, which eats A_F through Y_e = 6x²;
- the dual of B_μν, which eats a combination of A_F and A_h through Y_g.

Together they make both A_F and A_h massive, so the massless Spin(10) U(1)s are
exactly those orthogonal to h.

## 2. Orbifold breaking: S²/(ℤ₂ × ℤ₂)

The two π-rotations of the sphere, about z and about x, carry commuting gauge
twists:
- P_PS realizes Pati–Salam: +1 on (4,2,1) ⊃ {Q, L} and −1 on (4̄,1,2);
- P_SU5 realizes SU(5) × U(1): +1 on the 10 = {Q, u^c, e^c} and −1 on
  5̄ ⊕ 1.

The surviving gauge group is SU(3) × SU(2) × U(1)_Y × U(1)_X, the standard
6D orbifold-GUT result.

**Consistency of the construction.**
- **Flux.** The π-rotations preserve orientation, so the U(1)_F flux ∝ vol is
  invariant. R_x swaps the poles but does not reverse the flux.
- **Lift.** For three families, k = 3 and j = 1 is an integer, so the Klein
  group lifts as itself.
- **U(1)_F phases.** A constant U(1)_F twist is an overall phase on the 16₊,
  absorbed in the intrinsic parity.
- **The twists.** P_PS = exp(iπ(H₄+H₅)) is a genuine involution. P_SU5 is not
  one on spinors: on the 16 it acts as e^{−iπ/4}·(+1 on 10, −1 on 5̄ ⊕ 1).
  Getting ±1 parities needs a compensating fermion-number phase, whose
  discrete anomaly is not checked here.

**Theorem 3 (families are not uniform).** The flux families form a spin-j
multiplet of the sphere's SU(2), with j=1 for three families. The Klein group
{1, R_z(π), R_x(π), R_y(π)} acts on the multiplet with multiplicities:
- (2j+1+3(−1)^j)/4 for the trivial character;
- (2j+1−(−1)^j)/4 for each of the other three.

These are computed from explicit Wigner matrices and agree with the character
formula for j ≤ 10.

A Standard-Model multiplet survives in a given family state only if that
state's character matches the multiplet's twist class (P_PS, P_SU5). This
holds up to the four choices of intrinsic parity, which only permute the
classes. The four twist classes are:
- Q in (+,+);
- u^c and e^c in (−,+);
- d^c and ν^c in (−,−);
- L in (+,−).

Since the four classes receive unequal multiplicities, no choice of j or of
intrinsic parities gives the same number of families for every multiplet. For
j=1, depending on the parities, one or two multiplets get **zero** families
and the others one each. Every such spectrum is anomalous (SU(3)³, Witten or
Y³, as the review checked), unless fixed-point matter is added by hand.

**Theorem 3′ (three families are impossible for any rotation quotient).** A
rotation by 2π/N splits the family triplet by the phases e^{−2πim/N}. So:
- ℤ₂ keeps at most 2 families of any multiplet;
- ℤ_N with N ≥ 3 keeps at most 1;
- ℤ₂ × ℤ₂ keeps at most 1.

This is tested for N ≤ 24. Three families of every multiplet are therefore
impossible for any orbifold of S² by rotations, with any gauge twists. The
review scanned S²/ℤ_N for N ≤ 24 over Standard-Model-preserving Cartan twists
and found no uniform choice, even with one family each.

## 3. Consequences

- The sphere does not break Spin(10) for free:
  - Flux backgrounds are unstable (Lemma 0), and even if stabilized they
    would make hypercharge massive (Theorem 2).
  - Rotation orbifolds cannot keep three families (Theorems 3 and 3′).
- The routes analysed here are obstructed. That leaves:
  - conventional Higgs breaking, with 6D or 4D scalars in, for example, the
    45 or 54 together with the 16 or 126;
  - options not analysed here: other quotients, fixed-point fields, and
    gauge–Higgs components.

  BPR-6D does not supply these fields, so the Spin(10) → Standard Model step
  is recorded as **supplied, not derived**. The net Standard-Model chirality
  survives any Higgs vevs, by 4D anomaly matching.

## 4. Limitations

- The sphere is round. The Cartan-flux analysis is exhaustive for
  Yang–Mills critical points on S².
- Only rotation orbifolds are considered.
- The 4D normalization of the Stückelberg mass is not computed; only the BF
  vector's direction is.
- Localized fixed-point matter and localized anomalies are not included, and
  the discrete anomaly of the compensated P_SU5 is not checked.

## 5. Independent review

An independent review verified, with its own scripts:
- the 16 content and generators;
- the Theorem 2 algebra;
- the sign of Lemma 1 and the D₅ quantization;
- the absence of any group quotient;
- the Stückelberg reduction, with mass map of rank 2 and kernel
  {A_F = 0, tr(hA) = 0}, invariant under electric–magnetic swap;
- the Klein-group characters and the anomalies of every j=1 orbifold spectrum.

Its findings, all repaired:
- **Major.** Lemma 0, instability of every flux, is now the primary
  obstruction.
- **Minor:**
  - Atiyah–Bott exhaustiveness replaces the "instanton" limitation;
  - the flipped-hypercharge case is added;
  - vacuous tests are replaced by anomaly computations and a symbolic BF
    vector;
  - the example flux is quantized, (2,2,2,1,1);
  - orbifold consistency, including the P_SU5 phase, is stated;
  - "one or two multiplets vanish" and Theorem 3′ for all rotation
    quotients;
  - the section 3 scope is narrowed;
  - nits.
