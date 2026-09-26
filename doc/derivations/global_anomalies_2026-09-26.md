# Global anomalies of BPR-6D: none, for both global forms

2026-09-26. Status: exact computations of Adams E₂ pages by GF(2) linear
algebra, plus one reduction argument for the twisted global form. The
implementation is `bpr/global_anomaly_bordism.py`, with tests in
`tests/test_global_anomaly_bordism.py` and a demo in
`scripts/demo_global_anomaly_bordism.py`. An independent adversarial review
(section 9) confirmed the central theorem with separate code and found one
major scope gap, the twisted structure. That gap is now closed in section 7.

## 0. Question

Round 4 ([green_schwarz_quantization_2026-09-26.md](green_schwarz_quantization_2026-09-26.md))
showed that BPR-6D's local anomaly is cancelled by Dirac-quantized
Green–Schwarz couplings when the parent carries charge 3. A theory can still
be inconsistent through a **global** anomaly, which is invisible to the
anomaly polynomial. The 4D example is Witten's SU(2) anomaly. Does BPR-6D
have one?

## 1. Criterion

Take a 6D theory whose anomaly polynomial is cancelled exactly by
Green–Schwarz terms built from integral classes (here I8 = Y_e·Y_g in the even
lattice U). Its anomaly theory is a 7D invertible field theory:
- the exponentiated η-invariant of the fermions (Dai–Freed 1994;
  Witten–Yonekura 2019);
- times the well-defined 7D Green–Schwarz term.

Since I8 − Y_eY_g = 0, the anomaly theory is a bordism invariant, a character
of Ω₇ for the chosen structure (Freed–Hopkins). If that group vanishes, or if
the character is trivial on its generators, there is no global anomaly.

Non-vanishing homotopy groups π_d are neither necessary nor sufficient for a
global anomaly (Davighi–Lohitsiri, JHEP 05 (2021) 267). Lee and Tachikawa
(PTEP 2021, arXiv:2012.11622) observe that properly quantized Green–Schwarz
terms reproduce the older π₆ conditions. Round 4 (quantization) plus round 5
(bordism) is therefore the complete check.

## 2. Reduction to connective real K-theory

- **Degrees below 8.** The Atiyah–Bott–Shapiro map MSpin → ko has a
  7-connected fiber (Anderson–Brown–Peterson: the next summand is Σ⁸ko). So
  Ω_n^Spin(X) ≅ ko_n(X) for n ≤ 7, integrally. Odd torsion cannot appear,
  because H_*(BSpin(10) × ℂP^∞; ℤ) has only 2-torsion.
- **Twisted structures.** For Spin ×_{ℤ₂} Spin(n) structures, meaning a spin
  structure on TM ⊕ V, the bordism groups are ko_* of the Thom spectrum of
  V − n over BSO(n).
- **Splitting.** For the untwisted structure, X₊ = (BSpin(10) × ℂP^∞)₊ splits
  stably as S⁰ ∨ BSpin(10) ∨ ℂP^∞ ∨ (BSpin(10) ∧ ℂP^∞), and ko₇(S⁰) = 0.

## 3. Cohomological input and its checks

- **BSpin(10).** By Quillen (1971), H*(BSpin(10); F₂) is H*(BSO(10)) modulo
  the regular sequence w₂, Sq¹w₂, Sq²Sq¹w₂, Sq⁴Sq²Sq¹w₂, …. Below degree 17
  this is F₂[w₄, w₆, w₇, w₈, w₁₀]; a relation w₇w₁₀ = 0 appears in degree 17.
- **Twisted structures.** The Thom module U·H*(BSO(n)), with Sq¹U = 0 and
  Sq²U = w₂U.
- **ℂP^∞.** F₂[x], with Sq²x = x².
- **Squares and products.** Sq^i on generators comes from the Wu formula,
  Sq^i w_j = Σ_t C(j−i+t−1, t) w_{i−t} w_{j+t}, which is zero for i > j.
  Products use the Cartan formula.

Three independent checks on this input:
1. **Splitting principle.** Sq^i w_j is recomputed from Sq(e) = e + e² on
   formal roots and re-expressed in elementary symmetric polynomials (SymPy),
   w₁ terms included. It matches `wu` for i = 1, 2 and j ≤ 6.
2. **Hand values in BSpin(10):**
   - Sq¹w₄ = 0 and Sq²w₄ = w₆;
   - Sq¹w₆ = w₇ and Sq²w₆ = 0;
   - Sq²w₇ = 0;
   - Sq¹w₈ = 0 and Sq²w₈ = w₁₀.
3. **Adem relations.** Every module satisfies the Adem relations of A(1):
   Sq¹Sq¹ = 0, Sq²Sq² = Sq¹Sq²Sq¹, and (Sq²Sq¹)² = (Sq¹Sq²)².

   This is a necessary check, not a sufficient one. The review showed that
   setting Sq¹w₆ = 0 would still pass it, while producing a false tower in
   stem 7. Checks 1 and 2 are what protect the result.

## 4. Computation

`minimal_resolution` builds a minimal free resolution over
A(1) = ⟨Sq¹, Sq²⟩. A(1) has dimension 8 and an associative multiplication
table. The construction is exact linear algebra over GF(2), degree by degree,
and Ext^{s,t} equals the number of generators of F_s in degree t. Modules are
truncated at degree D. Ext is exact for stems below D, and a test confirms
that D = 12 and 14 agree in stems ≤ 9.

Untwisted E₂ totals for s ≤ 12 (a tower is an h₀-tower, i.e. a ℤ):

| summand | stem 4 | stem 5 | stem 6 | stem 7 | stem 8 |
|---|---|---|---|---|---|
| BSpin(10) | tower (λ_V) | 0 | 0 | **0** | three towers |
| ℂP^∞ | tower (x²) | 0 | two towers | **0** | towers |
| BSpin(10) ∧ ℂP^∞ | 0 | 0 | tower (λ_V x) | **0** | tower |

Stem 7 is empty through s = 20. Above a slope-½ vanishing line, Ext_{A(1)} is
h₀-periodic. The Q₀ (Sq¹) Margolis homology, which locates such towers,
vanishes in every odd degree below D. So stem 7 is empty for all s.

## 5. Validation against known results

The same engine reproduces the following:
- **ko of a point**, with the standard Adams filtrations.
- **Witten's SU(2) anomaly:** Ω₅^Spin(BSU(2)) = ℤ/2.
- **Lee–Tachikawa:** Ω₇^Spin(BSU(2)) = Ω₇^Spin(BSU(3)) = 0.
- **SU(3):** a ℤ in degree 6, which is the 4D cubic anomaly.
- **Spin(10):** no 4D Witten or cubic anomaly (stems 5 and 6 are empty).
- **BU(1):** the known groups.
- **Twisted Spin(10):** Ω₅^{Spin×_{ℤ₂}Spin(10)} = ℤ/2, the w₂w₃ class of
  Wang–Wen–Witten.

The review independently recomputed Tor^{A(1)} with a normalized bar complex,
using its own module code. It matched every chart above in stems 0–10, s ≤ 9.

## 6. Result for the untwisted structure

**Theorem 1.** Ω₇^Spin(B(Spin(10) × U(1))) = 0.

With the round-4 Green–Schwarz couplings, BPR-6D therefore has no global
anomaly on spin manifolds with Spin(10) × U(1) bundles. The review notes why
no extra structure is needed:
- B is dynamical;
- Y_e and Y_g are integral classes;
- the anomaly theory's class lies in Ext(Ω₇, ℤ) = 0.

## 7. The twisted structure Spin ×_{ℤ₂} Spin(10) × U(1)

Every fermion is a Spin(10) spinor and every boson a tensor, so the theory can
also be defined with (−1)^F identified with the center of Spin(10). This is
the only quotient the spectrum allows.

**Proposition 2 (the review's class).** For this structure, E₂ in stem 7 is a
single ℤ/2 at s=0 (computed here and by the review), detected by ∫c₁ w₂(V) w₃(V). A
generator is S² × W with unit flux, where W = SU(3)/SO(3) is the Wu manifold
and V = TW ⊕ ℝ⁵.

**Theorem 3.** The BPR-6D anomaly is trivial on this generator. Hence there
is no global anomaly for the twisted structure either.

*Argument.*
1. **The Green–Schwarz term drops out.** Y̌_e = 6x̌² is pulled back from S²,
   and Ȟ⁴(S²) = 0. So Y̌_e = 0, and the 7D Green–Schwarz term, bilinear in
   Y̌_e and Y̌_g, vanishes. Y_g is integral on twisted backgrounds, because
   3λ_V − λ_T = p₁(V) + (p₁(V) − p₁(T))/2 with w₂(V) = w₂(T).
2. **Product formula.** The bundles are pulled back from the factors, so
   η(S² × W) = ind(D_{S²} ⊗ L^q) · η(D_W ⊗ R).
   - The 16₊ has q=3, and its S² index is 3.
   - The 16₋ has q=0, and its S² index is 0.

   So α = α₁₆(W)³, where α₁₆(W) is the 4D anomaly of one Weyl 16 on W.
3. **Branching.** Under Spin(5) × Spin(5) ⊂ Spin(10), 16 = (4, 4). This is
   checked by restricting weights. On the generator, the second Spin(5)
   bundle is trivial, so the 16 bundle is (4 of the twisted Spin(5)) ⊗ ℂ⁴ and
   α₁₆(W) = α₄(W)⁴.
4. **Order bound.** A 4D Weyl fermion in the 4 of Spin(5) = Sp(2) has no
   perturbative anomaly: the 4 is pseudoreal, so I₆ = 0. Its anomaly α₄ is
   therefore a character of Ω₅^{Spin×_{ℤ₂}Spin(5)}. The computed E₂ page has
   two classes in stem 5, so that group has order at most 4, and α₄⁴ = 1.

Hence α(S² × W) = (α₄(W)⁴)³ = 1. ∎

This matches the expectation that a single 16 of Spin(10) carries no w₂w₃
anomaly (Wang–Wen, arXiv:1809.11171), but it does not depend on that
reference. Steps 2–4 use standard facts: the η product formula, bordism
invariance of perturbatively anomaly-free theories, and E₂ as an upper bound
on group order. No η-invariant is evaluated numerically.

## 8. Status

BPR-6D's anomalies are cancelled at every level tested, for both global
forms:
- local (round 1);
- Dirac-quantized, which forces 3 | parent charge and hence n_gen ∈ 3ℤ
  (round 4);
- global (here).

The bordism groups do not depend on the fermion charges. So the global
conditions add no arithmetic constraint beyond round 4.

Remaining filters toward a UV completion:
- the Morrison–Taylor massless-charge-lattice tension (quantization note,
  section 5);
- an explicit string or F-theory realization;
- the other swampland conditions.

## 9. Independent review

An independent adversarial review:
- rebuilt A(1) from a faithful representation on F₂[x₁..x₆] (all 64 products
  match);
- verified the Wu formula by the splitting principle in 10 variables, and the
  Quillen ideal through Sq⁴w₅;
- recomputed every chart with a separate bar-complex Tor code;
- checked the reduction (ABP integral iso, splitting, truncation) and the
  literature values;
- confirmed that no extra B-field structure is needed.

Its findings, all repaired:
- **Major.** For the twisted structure, Ω₇ contains a ℤ/2. It is resolved by
  Theorem 3.
- **Minor:**
  - the module docstring's degree bound (17, not 32);
  - `wu` for i > j now returns [];
  - the Adem check is no longer overclaimed, and the splitting-principle
    test is added;
  - the ABS reduction is stated integrally;
  - the Davighi–Lohitsiri and Lee–Tachikawa context is added;
  - the vanishing-line wording is fixed;
  - the per-instance Steenrod cache is fixed.

## 10. Limitations

- Theorem 3 is a reduction argument, not a numerical η-invariant. It uses the
  η product formula, bordism invariance of the perturbatively anomaly-free
  Spin(5) 4 theory, and E₂ as an upper bound on group order.
- The Dai–Freed / Witten–Yonekura framework and the standard 7D Green–Schwarz
  term are assumed.
- The untwisted result rests on E₂ = 0 in stem 7, which needs no
  differentials. The twisted order bound uses E₂ as an upper bound.
- Modules are truncated at degree 12–14.
