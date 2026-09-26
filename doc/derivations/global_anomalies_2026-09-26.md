# Global anomalies of BPR-6D: Ω₇^Spin(B(Spin(10)×U(1))) = 0

2026-09-26. Status: exact computation of the E₂ page of an Adams spectral
sequence, by GF(2) linear algebra. The implementation is
`bpr/global_anomaly_bordism.py`, with tests in
`tests/test_global_anomaly_bordism.py` and a demo in
`scripts/demo_global_anomaly_bordism.py`. The conclusion needs no Adams
differentials, because E₂ vanishes in the relevant stem. It covers the
Spin × Spin(10) × U(1) structure. The independent review is recorded in
section 8.

## 0. Question

Round 4 ([green_schwarz_quantization_2026-09-26.md](green_schwarz_quantization_2026-09-26.md))
showed that BPR-6D's local anomaly is cancelled by Dirac-quantized
Green–Schwarz couplings when the parent carries charge 3. A theory can still
be inconsistent through a **global** anomaly, which is invisible to the
anomaly polynomial. The 4D example is Witten's SU(2) anomaly. Does BPR-6D
have one? This is the next condition a consistent quantum completion must
meet.

## 1. Criterion

Take a 6D theory whose anomaly polynomial is cancelled exactly by
Green–Schwarz terms built from integral classes (here I8 = Y_e·Y_g in the even
lattice U). Its anomaly theory is a 7D invertible field theory:
- the exponentiated η-invariant of the fermions (Dai–Freed 1994;
  Witten–Yonekura 2019);
- times the well-defined 7D Green–Schwarz term.

On a closed 7-manifold that bounds an 8-manifold with all structures
extended, its value is exp(2πi∫(I8 − Y_eY_g)) = 1. The anomaly is therefore a
bordism invariant, a homomorphism

    Ω₇^Spin(BG) → U(1),   G = Spin(10) × U(1).

If Ω₇^Spin(BG) = 0, every such 7-manifold bounds, and there is no global
anomaly. See Freed–Hopkins for the general classification. The older π₆
criterion is weaker; it is trivially satisfied here since π₆(Spin(10)) = 0.

## 2. Reduction to connective real K-theory

- **Degrees below 8.** At the prime 2, MSpin ≃ ko ∨ Σ⁸ko ∨ …
  (Anderson–Brown–Peterson 1967). So Ω_n^Spin(X) ≅ ko_n(X) for n ≤ 7 and
  connective X.
- **Odd primes.** H_*(BSpin(10) × ℂP^∞; ℤ) has only 2-torsion. Its homology
  away from 2 is concentrated in even degrees, and Ω_q^Spin ⊗ ℤ[1/2] vanishes
  for q ≢ 0 (mod 4) in this range. So the Atiyah–Hirzebruch spectral sequence
  gives no odd part in degree 7.
- **Splitting.** With X = BSpin(10) × ℂP^∞, X₊ splits stably as
  S⁰ ∨ BSpin(10) ∨ ℂP^∞ ∨ (BSpin(10) ∧ ℂP^∞), and ko₇(S⁰) = 0.

The Adams spectral sequence E₂ = Ext_{A(1)}(H̃*(Y; F₂), F₂) ⇒ ko_*(Y) is
computed for each of the three nontrivial summands Y.

## 3. Cohomological input

- **BSpin(10).** By Quillen (1971), H*(BSpin(10); F₂) is H*(BSO(10); F₂)
  modulo the regular sequence w₂, Sq¹w₂, Sq²Sq¹w₂, Sq⁴Sq²Sq¹w₂, …, which
  kills w₂, w₃, w₅ and w₉ below degree 17. The extra spinor class sits in
  degree 32. Below degree 17 this gives F₂[w₄, w₆, w₇, w₈, w₁₀].
- **Steenrod squares.** Sq¹ and Sq² come from the Wu formula,
  Sq^i w_j = Σ_t C(j−i+t−1, t) w_{i−t} w_{j+t}, followed by projection. By
  hand this gives:
  - Sq¹w₄ = 0 and Sq²w₄ = w₆;
  - Sq¹w₆ = w₇ and Sq²w₆ = 0;
  - Sq²w₇ = w₉ = 0;
  - Sq¹w₈ = 0 and Sq²w₈ = w₁₀.

  The tests check these values.
- **ℂP^∞.** F₂[x], with Sq¹x = 0 and Sq²x = x².
- **Products.** Cartan formula; the smash product uses the diagonal action.

*Independent check of the input.* Every constructed module satisfies the
Adem relations of A(1): Sq¹Sq¹ = 0, Sq²Sq² = Sq¹Sq²Sq¹, and
(Sq²Sq¹)² = (Sq¹Sq²)². This holds for BSO(4), BSO(6), BSO(10), BSpin(10),
ℂP^∞, the smash product, BSU(2) and BSU(3). A wrong Wu coefficient or a wrong
Quillen projection would generically violate them.

## 4. Computation

`minimal_resolution` builds a minimal free resolution over
A(1) = ⟨Sq¹, Sq²⟩. A(1) has dimension 8 and an associative multiplication
table (tested). The construction is exact linear algebra over GF(2), degree by
degree. Since the resolution is minimal, Ext^{s,t} equals the number of
generators of F_s in degree t. Modules are truncated at degree D=14; Ext is
exact for stems below D, and a test confirms that D = 12 and 14 agree in
stems ≤ 9.

E₂ totals for s ≤ 12 in stems 4–8 (a tower means an h₀-tower, i.e. a ℤ
summand):

| summand | stem 4 | stem 5 | stem 6 | stem 7 | stem 8 |
|---|---|---|---|---|---|
| BSpin(10) | tower (λ_V) | 0 | 0 | **0** | three towers |
| ℂP^∞ | tower (x²) | 0 | two towers | **0** | towers |
| BSpin(10) ∧ ℂP^∞ | 0 | 0 | tower (λ_V x) | **0** | tower |

Stem 7 is empty for all s ≤ 20, the range computed in the test. No h₀-tower
can reach stem 7 at higher filtration, because the Q₀ (Sq¹) Margolis
homology vanishes in every odd degree below D (tested). Above the A(1)
vanishing line, only such towers survive.

## 5. Validation against known results

The same engine reproduces the following:
- **ko of a point:** ℤ, ℤ/2, ℤ/2, 0, ℤ, 0, 0, 0, ℤ, with the standard Adams
  filtrations.
- **Witten's SU(2) anomaly:** Ω₅^Spin(BSU(2)) = ℤ/2, as the h₁ class on the
  4-cell.
- **Lee–Tachikawa (PTEP 2021, arXiv:2012.11622):** Ω₇^Spin(BSU(2)) =
  Ω₇^Spin(BSU(3)) = 0.
- **SU(3):** no 4D Witten anomaly (stem 5 is empty), and a ℤ in degree 6,
  which is the 4D cubic anomaly.
- **Spin(10):** no 4D Witten anomaly and no cubic anomaly (stems 5 and 6 are
  empty). This is the familiar statement that SO(10) is 4D-anomaly-free.
- **BU(1):** the odd groups through degree 7 vanish, the degree-2 group is ℤ
  and the degree-6 group is ℤ², as known.

## 6. Result

**Theorem.** Ω₇^Spin(B(Spin(10) × U(1))) = 0.

**Corollary.** BPR-6D, with the round-4 Green–Schwarz couplings, has no
global gauge or gravitational anomaly on spin manifolds with Spin(10) × U(1)
bundles. Its anomalies are therefore cancelled completely at this level:
local (round 1), Dirac-quantized (round 4) and global (here).

The bordism group does not depend on the fermion charges. The global
condition therefore imposes nothing beyond round 4, and the 3 | q constraint
stands as the only arithmetic output of anomaly cancellation.

## 7. Scope

- **Structure.** Spin × Spin(10) × U(1). The spectrum also admits
  Spin ×_{ℤ₂} Spin(10), since all fermions are Spin(10) spinors. That would
  put the theory on some non-spin manifolds, and its bordism group was not
  computed. Defining the theory with the Spin × Spin(10) × U(1) structure is a
  legitimate choice of global form.
- **Green–Schwarz term.** The Green–Schwarz term is the standard 7D term
  for integral Y_e and Y_g in the even lattice U. No quadratic refinement is
  needed.
- **Status toward a UV completion.** BPR-6D now passes:
  - local anomaly cancellation;
  - Dirac quantization;
  - global anomalies.

  The remaining filters are:
  - the Morrison–Taylor massless-charge-lattice tension
    (green_schwarz_quantization, section 5);
  - an explicit string or F-theory realization;
  - the other swampland conditions.

## 8. Independent review

Pending at the time of writing.

## 9. Limitations

- Only the Spin × Spin(10) × U(1) structure is covered.
- The result is an E₂ computation. It is conclusive only because E₂ vanishes
  in stem 7.
- The Dai–Freed / Witten–Yonekura framework is assumed.
- The Green–Schwarz 7D term is assumed standard.
- Modules are truncated at degree 14.
