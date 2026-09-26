# Breaking Spin(10) to the Standard Model in BPR-6D: geometric routes are obstructed

2026-09-26. Status: exact weight and root algebra, exhaustive flux scans, and
explicit Wigner rotation matrices. The implementation is
`bpr/gut_breaking.py`, with tests in `tests/test_gut_breaking.py` and a demo
in `scripts/demo_gut_breaking.py`. The independent review is recorded in
section 5.

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

**Lemma 1 (chirality neutrality).** Take any Cartan flux h in Spin(10), with
the U(1)_F flux m=1 and parent charge 3. A 16 weight w then has index:
- 3 + w·h from the 16₊;
- −w·h from the 16₋, which has the same Spin(10) representation but the
  opposite 6D chirality.

The net count is exactly 3 for every Standard-Model multiplet. Spin(10) flux
therefore never spoils the family count; it only adds vector-like pairs.
Tested for random quantized fluxes.

**Theorem 2 (no flux reaches the Standard Model with massless hypercharge).**
The round-4 Green–Schwarz class Y_g = 3λ_V + 9x² − λ_T contains the Spin(10)
instanton density. A flux h in Spin(10) therefore produces a 4D BF coupling
B ∧ tr(⟨F_h⟩F), proportional to 3·tr(h·h) ≠ 0. That coupling makes the
U(1) along h massive (Stückelberg).

Two further facts:
- Keeping SU(3) × SU(2)_L unbroken forces h ⊥ the Standard-Model roots, i.e.
  h = (a,a,a,b,b) ∈ span(Y, X).
- The massless Spin(10) U(1) is then h^⊥ within that plane. It equals Y only
  if h ∝ X. But h ∝ X leaves the whole SU(5) unbroken (20 roots).

Hence no flux yields SU(3) × SU(2) × U(1)_Y with a massless hypercharge boson.
An exhaustive scan over integral h with |h_i| ≤ 3 finds fluxes whose
nonabelian centralizer is exactly SU(3) × SU(2) (8 roots). None of them leaves
Y massless.

This is the sphere version of the known F-theory "hypercharge flux" problem.
There, hypercharge flux is harmless only when its class is trivial in the
base, which is impossible on S², where the flux class is the whole second
cohomology.

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
j=1 one multiplet always gets **zero** families and the others one each. The
resulting 4D spectrum is chiral-anomalous unless fixed-point matter is added
by hand.

## 3. Consequences

- The sphere does not break Spin(10) for free. Flux breaking kills
  hypercharge (Theorem 2), and orbifold breaking destroys the family structure
  (Theorem 3).
- The remaining route is conventional Higgs breaking. That means 6D or 4D
  scalars in, for example, the 45 or 54 together with the 16 or 126, or other
  standard SO(10) breaking chains. Lemma 1 guarantees that the family count
  survives any such background. BPR-6D does not supply these fields, so the
  Spin(10) → Standard Model step is recorded as **supplied, not derived**.
- Lemma 1 is itself useful: in this completion, any later Spin(10) background
  preserves exactly three net families of every Standard-Model multiplet.

## 4. Limitations

- Only abelian fluxes are considered, on the round sphere, and only
  S²/(ℤ₂ × ℤ₂) orbifolds with commuting twists.
- The 4D normalization of the Stückelberg mass is not computed; only its
  nonvanishing is shown.
- Localized fixed-point matter and localized anomalies are not included.
- Non-abelian instanton backgrounds on S² are not considered.

## 5. Independent review

Pending at the time of writing.
