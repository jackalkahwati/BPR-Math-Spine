# Phase 2: confronting the minimal BPR-6D model with fermion data

2026-09-27. Status: numerical fits with stated inputs and errors. The
implementation is `bpr/fermion_fit.py`, with tests in
`tests/test_fermion_fit.py` and a demo in `scripts/demo_fermion_fit.py`. The
best-fit points are stored in the module and recomputed from scratch by the
tests. **No empirical validation is claimed.** This is the first comparison of
BPR-6D with measured data. The independent review is recorded in section 8.

## 0. Question

Phase 1 fixed a definite model: a single bulk Higgs of each kind (10_H and
126bar_H, both with F = −6), coupled at four branes. Phase 1d pinned the
branes at a regular tetrahedron with a vortex condensate. After symmetry
breaking the model gives the non-supersymmetric SO(10) relations

    M_d = H + F,   M_e = H − 3F,   M_u = r(H + sF),   M_D = r(H − 3sF),
    M_R = w F,     m_ν = −M_D M_R⁻¹ M_Dᵀ (type I),

with H and F complex symmetric. Type II is negligible in this chain,
v_L ~ 10⁻¹¹ eV.

Does this fit the measured quark and lepton masses, the quark mixing (CKM),
and the neutrino data?

## 1. Inputs

**Running to the unification scale.** One-loop Standard-Model running of the
gauge and Yukawa couplings takes the inputs from M_Z up to 2×10¹⁶ GeV. The M_Z
inputs are MS-bar running masses (Xing–Zhang–Zhou 2008, as recalled) and the
PDG CKM parameters. The gauge running reproduces the scales module exactly,
and the standard features are reproduced: no b–τ unification in the SM
(m_b/m_τ = 0.62), |V_cb| growing from 0.042 to 0.048, |V_us| stable (all
tested).

The results at 2×10¹⁶ GeV:

| sector | this work (one loop) | two-loop literature (Bora 2012) |
|---|---|---|
| m_u, m_c, m_t (GeV) | 5.2×10⁻⁴, 0.251, 79.5 | 4.6×10⁻⁴, 0.223, 70.5 |
| m_d, m_s, m_b (GeV) | 1.22×10⁻³, 0.0231, 1.065 | 1.08×10⁻³, 0.0204, 0.932 |
| m_e, m_μ, m_τ (GeV) | 4.76×10⁻⁴, 0.1005, 1.709 | 4.41×10⁻⁴, 0.0931, 1.611 |
| \|V_us\|, \|V_cb\|, \|V_ub\|, J | 0.2250, 0.0477, 0.0042, 4.0×10⁻⁵ | — |

The one-loop values run 5–15% above the two-loop ones. The fit works mostly
with ratios, which differ less.

**Neutrino data.** NuFIT-like values with normal ordering:
Δm²₂₁ = 7.41×10⁻⁵ eV², Δm²₃₁ = 2.511×10⁻³ eV², sin²θ₁₂ = 0.303,
sin²θ₂₃ = 0.451, sin²θ₁₃ = 0.02225. Their running in the SM is neglected.

**Assumed errors at the unification scale:**
- light quarks 30% and m_s 15%;
- heavy quarks 5%;
- charged leptons 1%;
- CKM 1–10%;
- neutrino observables 3–10%.

The χ² is a goodness-of-fit guide, not a likelihood.

## 2. Identical branes give no Yukawas

With identical (A4-symmetric) brane couplings, the spin-3 lowest level of the
bulk Higgs splits as 1 + 3 + 3 (tested). A unique light Higgs must be the
singlet φ, and its ð̄ value vanishes at every vertex.

*Proof.* For the 3-fold rotation g_a about vertex a,
D(g_a)v_a = e^{4πi/3}v_a, where v_a = D(R_a)e_{−2}. Also D(g_a)φ = φ. So
⟨v_a, φ⟩ = e^{4πi/3}⟨v_a, φ⟩, hence ⟨v_a, φ⟩ = 0. ∎

Numerically the coupling is below 10⁻⁸. So identical branes give no fermion
masses, and the brane couplings must break A4 at O(1).

## 3. Fit method

In the basis where M_d is diagonal (a U(3) choice),
M_u = V†·diag(m_u e^{iα})·V* reproduces the up masses and the CKM matrix
exactly. The SO(10) relation M_u = aM_d + bM_e, with a = r(3+s)/4 and
b = r(1−s)/4, then fixes M_e = (M_u − aM_d)/b. The only charged-sector
conditions left are the two lepton mass ratios (tested identities).

There are 18 parameters: mass and CKM pulls, two phases, a, and arg b. The
charged sector then fits **exactly** (χ² ≈ 0) in a few starts (tested). An
earlier generic parametrization had failed only through poor conditioning.

The neutrino observables add four conditions. The fits use multi-start least
squares with basin hopping on four cores.

**The pinned model.** With the branes fixed at the tetrahedron, (H, F) must
also be reachable: some U ∈ U(3) must put both UᵀHU and UᵀFU in W, the span
of the four brane matrices. That is 8 real conditions. They are imposed as a
penalty tightened in stages (10⁻¹ → 10⁻⁴) on (18 + 9) parameters, until the
reachability residual is below 10⁻⁷.

## 4. Results

| model | best χ² | main pulls | status |
|---|---|---|---|
| charged fermions only | ≈ 0 | none | fits exactly |
| free brane positions + neutrinos | **36.2** (≈ 1 degree of freedom) | m_d −4.4σ, m_b +2.3σ, m_s +2.0σ; every neutrino observable within 0.5σ | strained |
| **pinned tetrahedron** + neutrinos | **609** (two searches: 609, 619) | sin²θ₁₃ −17σ, m_s +10σ, CKM δ +10σ, s₂₃ −5σ, m_d −5σ | **strongly disfavoured** |

- **Free positions (generic 10 + 126bar).** Three of four independent searches
  converged to the same χ² = 36.18. The neutrino data fit well, but only by
  pulling the down-quark mass about 4.4σ low (it drops to about 0.3 MeV at the
  unification scale).
- **Pinned tetrahedron.** The generic best fit is **not** reachable with
  tetrahedral branes: its squared distance is 5×10⁻³ (tested). The best
  reachable points found have χ² ≈ 609–619, from two searches with different
  starting distributions. That is not a proof of the global minimum, but the
  failure is large and consistent: θ₁₃ comes out far too small. Whether the
  charged sector alone fits the pinned model is a separate diagnostic, run and
  recorded in section 6.

## 5. Predictions of the free-position fit (conditional)

These come from a strained fit and are not claimed to be unique:
- normal ordering, with m₁ ≈ 6.4 meV, m₂ ≈ 10.7 meV, m₃ ≈ 50.5 meV, and
  Σm_ν ≈ 0.068 eV (below cosmological bounds);
- m_ββ ≈ 0.4 meV, far below current and planned double-beta-decay
  sensitivities;
- sin δ_CP ≈ 0.61; δ_CP was not fitted, since it is poorly measured;
- heavy neutrino masses M_R ≈ (2.9×10⁹, 5.6×10¹¹, 1.0×10¹³) GeV.

**Seesaw scale.** A perturbative 126 Yukawa, |Y126| ≤ √(4π), needs
v_R ≳ 2.9×10¹² GeV. That is about 2800 times the one-loop intermediate scale
M_I ≈ 10⁹ GeV of Phase 1 (tested). This is the Phase 1 seesaw tension, now
quantified. It needs the B−L scale raised by thresholds or light 45 states:
BDM 2012 find M_B−L up to about 10¹⁴ GeV in this Higgs content.

## 6. Verdict

- **The pinned model fails Phase 2.** BPR-6D-M as specified in Phases 1 and
  1d (tetrahedral branes, minimal 10 + 126bar, type-I seesaw) does not fit
  the fermion data in this analysis: best χ² ≈ 609, with θ₁₃ off by about
  17σ. This holds only at leading order. Brane-localized fermion kinetic
  terms and the tension distortion of the zero modes add J=0 pieces that
  enlarge the reachable set, and their size, O(δ) ~ 10%, could matter for
  the small entries.
- **The free-position version is strained but alive.** Its χ² ≈ 36 comes
  mostly from m_d, but it has no stabilized moduli.
- **The two unification-scale scales conflict.** The fit needs
  v_R ≳ 3×10¹² GeV, against M_I ≈ 10⁹ GeV at one loop.

The charged-only diagnostic for the pinned model is reported in the review
record (section 8).

Ways forward, in order of economy:
1. compute the leading J=0 corrections (brane kinetic terms, tension
   distortion) and refit the pinned model;
2. a different stabilized configuration: unequal couplings or tensions move
   the vortices off the regular tetrahedron at O(δ);
3. enlarge the Higgs content (a 120_H). On a brane it couples only with
   c ∈ {−1, 0, 1}, i.e. through ð̄² of a bulk field, which costs ε²;
4. two-loop running, the 3221 stage and thresholds, which may soften the
   generic tension but cannot remove a 17σ θ₁₃ deficit.

## 7. Limitations

- One-loop SM running only: no 3221 stage between M_I and M_GUT, no two-loop
  terms, no thresholds.
- Neutrino running is neglected.
- The errors are assumed.
- The fits are multi-start local optimizations, so better minima may exist.
  The pinned verdict rests on two searches reaching χ² ≈ 609–619.
- The pinned constraint is leading order in the brane kinetic terms and the
  tension distortion.
- The inputs at M_Z are recalled values; the web proxy blocks the journals.

## 8. Independent review

Pending at the time of writing.
