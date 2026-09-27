# Phase 2: confronting the minimal BPR-6D model with fermion data

2026-09-27. Status: numerical fits with stated inputs and errors. The
implementation is `bpr/fermion_fit.py`, with tests in
`tests/test_fermion_fit.py` and a demo in `scripts/demo_fermion_fit.py`. The
best-fit points are stored in the module and recomputed from scratch by the
tests. **No empirical validation is claimed.** This is the first comparison of
BPR-6D with measured data.

The independent review (section 8) found two blockers in the first version,
both repaired here. They reversed its headline numbers:
- The generic construction left out two physical phases. With them, the
  free-position model **fits** (χ² ≈ 2.9), where the first version reported a
  strained χ² ≈ 36.
- The stored pinned point sat slightly off the tetrahedral subspace, so its
  χ² ≈ 609 was not a pinned-model χ². In an exact parametrization the best
  pinned point found has χ² ≈ 124, and the charged sector alone gives 73. The
  failure is in the **charged sector** (m_s about 3.6 times too large), not in
  θ₁₃ as first reported.

## 0. Question

Phase 1 fixed a definite model: a single bulk Higgs of each kind (10_H and
126bar_H, both with F = −6), coupled at four branes. Phase 1d pinned the
branes at a regular tetrahedron with a vortex condensate. After symmetry
breaking the model gives the non-supersymmetric SO(10) relations

    M_d = H + F,   M_e = H − 3F,   M_u = r(H + sF),   M_D = r(H − 3sF),
    M_R = w F,     m_ν = −M_D M_R⁻¹ M_Dᵀ (type I),

with H and F complex symmetric. Type II is negligible in this chain,
v_L ~ 10⁻¹¹ eV. U(1)_F acts as a Peccei–Quinn symmetry, so the complex 10_H
couples without its conjugate.

Does this fit the measured quark and lepton masses, the quark mixing (CKM),
and the neutrino data? And does it still fit with the branes pinned?

## 1. Inputs

**Running to the unification scale.** One-loop Standard-Model running of the
gauge and Yukawa couplings takes the inputs from M_Z up to 2×10¹⁶ GeV. The M_Z
inputs are MS-bar running masses (Xing–Zhang–Zhou 2008) and the PDG CKM
parameters. The gauge running reproduces the scales module exactly, and the
standard features are reproduced (all tested):
- no b–τ unification in the SM (m_b/m_τ = 0.62);
- |V_cb| grows from 0.042 to 0.048;
- |V_us| and the CKM phase δ = 1.144 barely run.

The results at 2×10¹⁶ GeV:

| sector | this work (one loop) | two-loop literature (Bora 2012) |
|---|---|---|
| m_u, m_c, m_t (GeV) | 5.2×10⁻⁴, 0.251, 79.5 | 4.6×10⁻⁴, 0.223, 70.5 |
| m_d, m_s, m_b (GeV) | 1.22×10⁻³, 0.0231, 1.065 | 1.08×10⁻³, 0.0204, 0.932 |
| m_e, m_μ, m_τ (GeV) | 4.76×10⁻⁴, 0.1005, 1.709 | 4.41×10⁻⁴, 0.0931, 1.611 |
| s₁₂, s₂₃, s₁₃, δ | 0.2250, 0.0477, 0.0042, 1.144 | — |

The quark masses run 10–15% above the two-loop values. What matters in the
fit is the down–lepton normalization: m_b/m_τ differs by 7.6% between one
and two loops (review). The roughly 6% lepton offset from Bora is probably a
convention difference, not two-loop running.

**Neutrino data.** NuFIT-like values (recalled), normal ordering:
Δm²₂₁ = 7.41×10⁻⁵ eV², Δm²₃₁ = 2.511×10⁻³ eV², sin²θ₁₂ = 0.303,
sin²θ₂₃ = 0.451, sin²θ₁₃ = 0.02225. Inverted ordering:
Δm²₃₂ = −2.498×10⁻³ eV², sin²θ₂₃ = 0.569, sin²θ₁₃ = 0.02223. Their running
in the SM is neglected.

**Assumed errors at the unification scale:**
- quark masses: 30% for m_u and m_d, 15% for m_s, 5% for the heavy quarks,
  each with an **8% theory error** in quadrature for the one-loop running
  (so 31%, 17% and 9.4%);
- charged leptons 1% (counted once; the first version double-counted them);
- CKM angles 1%, 3% and 5%, and δ ± 0.03 rad about its running value (the
  first version used δ = 1.2 ± 10%);
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

## 3. Parameters and fit method

**Counting.** The model data (H, F, r, s) have 28 real parameters. U(3)
family rotations remove 9. The phase of r is unobservable, since it is an
overall phase of M_u and of m_ν. That leaves **18 physical parameters**,
plus the neutrino scale w, against **17 fitted observables**: 6 quark masses,
4 CKM parameters, 3 lepton masses and 4 neutrino observables (Δm²₂₁/Δm²₃ℓ
and three angles; w absorbs the absolute scale). So there are −1 degrees of
freedom. A small χ² means consistency, not a test, and the free-position
model predicts nothing in the charged sector.

**Generic model (free brane positions).** Work in the basis where
M_d = diag(m_d e^{iβ}), a U(3) choice. Then:
- M_u = V†·diag(m_u e^{iα})·V* reproduces the up masses and the CKM matrix
  exactly;
- the SO(10) relation M_u = aM_d + bM_e, with a = r(3+s)/4 and
  b = r(1−s)/4, gives M_e = (M_u − aM_d)/b, with |b| fixed by m_τ.

The 18 parameters are the pulls of the 6 quark masses, the 4 CKM pulls, two
phases α, two phases β, the complex a, arg b and the pull of m_τ.

The construction's rank modulo U(3) and arg r is **18**, so it is locally
complete (tested). Without the two phases β it is 16, which was the first
version's blocker. Every observable is recomputed from the mass matrices,
and the tests check that this returns the parameters.

**Pinned model (tetrahedral branes).** H and F must lie, in some family
basis, in the span W of the four tetrahedral brane matrices P_a = u_a u_aᵀ.
The fit therefore uses the exact parametrization H = Σ c_a P_a and
F = Σ d_a P_a, with complex c_a and d_a and with r and s. The tests confirm
that projecting onto W changes nothing, and so does the independent
reachability search of Phase 1d.

The first version imposed the constraint as a penalty. It left the matrices
about 5×10⁻⁵ off W, and the light masses and the seesaw are sensitive at
that level.

**What W is.** W is exactly the set of symmetric matrices that are pure
J = 2 and annihilated by the pinning condensate χ itself (tested). Write the
spin-2 part of Y as a quartic polynomial, so that a brane matrix u(z)u(z)ᵀ
evaluates to the χ profile at z. The four branes sit at the zeros of χ, so χ
annihilates all four P_a, and they span its 4-dimensional kernel within
J = 2.

So pinning requires a family basis in which both Yukawa matrices are pure
J = 2 and orthogonal to the tetrahedral condensate. That is 8 real
conditions against the 8 parameters of U(3)/U(1), so the reachable set is
open (rank 24, Phase 1d) but a proper subset. Pinning restricts the domain
of the 18 parameters rather than removing any.

**Searches.**
- **Generic:** three multi-start basin-hopping searches of 15 minutes each,
  giving 90 local minima. Two searches, one from random starts only, reach
  the same best point.
- **Pinned:**
  - the review's four exact pinned points, re-expressed in the canonical
    tetrahedron frame (their χ² is reproduced to machine precision) and
    polished by basin hopping;
  - an independent continuation. Each of the 8 lowest generic minima is
    realized with four free branes, the branes are moved along great circles
    to the nearest regular tetrahedron while refitting, and the endpoint is
    polished by basin hopping.
- Each search is run with and without the neutrino data. Four further
  basin-hopping runs (60 hops each) around the two best pinned points with
  neutrinos give the final value.

## 4. Results

| model | best χ² | pulls | status |
|---|---|---|---|
| charged fermions only | ≈ 0 | none | fits exactly |
| **free positions** + neutrinos, normal ordering | **2.93** (−1 dof) | m_d −1.5σ; all others below 0.6σ | **fits** |
| free positions, m_d error 10% instead of 30% | 8.1 | m_d −1.8σ, sin²θ₂₃ −1.7σ | fits, strained |
| free positions, inverted ordering | 4.4×10⁴ | the neutrino observables | **excluded** |
| **pinned tetrahedron**, charged sector only | **73** | m_s +7.5σ, m_b −3.8σ | **fails** |
| **pinned tetrahedron** + neutrinos | **124** (best found) | m_s +7.5σ, m_u +6.2σ, m_b −3.9σ, m_d −3.0σ; neutrinos within 1σ | **fails** |

**Free positions.** The fit is good. The only pull above 0.6σ is m_d,
about 37% below its central value at the unification scale. This agrees
with the literature: non-supersymmetric 10 + 126bar fits with a type-I
seesaw work for normal ordering only (Joshipura–Patel, PRD 83 (2011)
095002; Ohlsson–Pernow, JHEP 06 (2019) 085). The first version's χ² ≈ 36
should have been a red flag.

**Dependence on the m_d error.** The 30% m_d error follows the input
compilation. Lattice values at low energy are now much more precise, so the
GUT-scale error is dominated by running and thresholds. With a 10% error (plus
the theory error) the refit rebalances to χ² = 8.1, with m_d at −1.8σ and
sin²θ₂₃ at −1.7σ. The review's warning that m_d would exceed 8σ applied to
the fixed point, not to a refit.

**Inverted ordering** is excluded in this model, with best χ² ≈ 4×10⁴. This
agrees with Ohlsson–Pernow.

**Pinned tetrahedron.** Every search fails in the same way:
- **Charged sector:** m_s comes out about 3.6 times too large, while m_d
  and m_b are pulled low. With neutrinos, m_u is also pulled up, about 7
  times too large.
- **Neutrinos:** these observables fit.
- **The charged sector alone fails.** Its χ² is 73 with 18 parameters and 13
  observables. So the failure comes from the restricted domain, not from a
  lack of parameters.
- **Robustness:**
  - *Charged sector only:* 7 of 9 searches end at the same χ² = 73.26 and the
    other two at about 460. This minimum looks global, but that is not
    proved.
  - *With neutrinos:* the 9 searches end at χ² ≈ 129–166, and extended basin
    hopping lowers the best to 123.5. This number is not converged.
  - *Floor:* the charged part of any pinned fit is at least the charged-only
    minimum. If that minimum is global, the full pinned χ² cannot fall below
    73.

The generic best fit is not reachable with tetrahedral branes: its squared
distance from W is above 10⁻⁴ over 15 starts (tested).

## 5. Predictions of the free-position fit (conditional)

The model has −1 degrees of freedom and several minima within Δχ² = 4 of the
best: χ² = 2.93, 3.66 and 6.82. Across these minima the predictions are:

| quantity | range over the minima | stable? |
|---|---|---|
| ordering | normal (inverted excluded) | yes |
| m₃ | 50.2–50.6 meV | yes |
| m₂ | 9.1–11.0 meV | roughly |
| m₁ | 3.0–6.8 meV | no |
| Σm_ν | 0.062–0.068 eV | yes (below cosmological bounds) |
| m_ββ | 0.40–0.43 meV | yes (far below planned double-beta-decay sensitivity) |
| sin δ_CP (lepton) | −0.95 to +0.62 | **no** |
| M_R (GeV) | (0.6–1.2)×10¹⁰, (1.1–5.8)×10¹¹, (1.5–3.8)×10¹² | no |

**Seesaw scale.** A perturbative 126 Yukawa, |Y126| ≤ √(4π), needs
v_R ≳ M_R,max/√(4π) = (0.4–1.1)×10¹² GeV. That is 400–1000 times the
one-loop intermediate scale M_I ≈ 1.05×10⁹ GeV, which is taken from
`model_scales` (tested).

This is the Phase 1 seesaw tension, now quantified. It needs the B−L scale
raised by thresholds or light 45 states. Bertolini–Di Luzio–Malinský 2012
find M_B−L up to about 10¹⁴ GeV in this Higgs content.

## 6. Verdict

- **Free positions: the model fits.** With free brane positions, the minimal
  10 + 126bar sector with a type-I seesaw is consistent with all charged
  fermion and neutrino data. The best χ² is 2.9 with −1 degree of freedom,
  so this is consistency, not a test.
  - Stable consequences, not tests: normal ordering, Σm_ν ≈ 0.06–0.07 eV
    and m_ββ ≈ 0.4 meV.
  - The fit needs v_R ≳ 4×10¹¹ GeV, against M_I ≈ 10⁹ GeV at one loop.
  - The positions are not stabilized.
- **Pinned tetrahedron: the model fails the charged sector.** BPR-6D-M as
  specified in Phases 1 and 1d (tetrahedral branes, minimal 10 + 126bar,
  type-I seesaw) does not fit the data in these searches. The best χ² found is
  124 with neutrinos and 73 for the charged sector alone. In every search m_s
  comes out about 3.6 times too large.
  - This holds at leading order only. Brane-localized fermion kinetic terms
    and the tension distortion of the zero modes add J = 0 pieces that
    enlarge W. At O(δ) ~ 10% they could matter, since the needed changes are
    O(1) factors in light masses.
- **The model's two key structural choices conflict.** Pinning, which Phase
  1d needed for kill check 11, is what spoils the charged-fermion fit.

Ways forward, in order of economy:
1. Compute the leading J = 0 corrections (brane kinetic terms, tension
   distortion) and refit the pinned model. This is decisive, and cheap
   relative to the others.
2. Try a different stabilized configuration. Unequal couplings or tensions
   move the vortices off the regular tetrahedron at O(δ), which changes W
   through χ.
3. Enlarge the Higgs content with a 120_H. On a brane it couples only with
   c ∈ {−1, 0, 1}, i.e. through ð̄² of a bulk field, which costs ε².
4. Two-loop running, the 3221 stage and thresholds shift the targets by
   about 10%. That cannot turn a factor of 4 in m_s into agreement.

## 7. Limitations

- One-loop SM running only: no 3221 stage between M_I and M_GUT, no two-loop
  terms, no thresholds. An 8% theory error on the quark masses stands in for
  them.
- Neutrino running is neglected.
- The errors are assumed, and the free-position verdict depends on the m_d
  error (section 4).
- The fits are multi-start local optimizations, so better minima may exist.
  The pinned verdict rests on 9 searches in each setting. The
  charged-only minimum (73.26) is reproduced by 7 of them; the value with
  neutrinos (123.5) is not converged.
- The pinned constraint is leading order in the brane kinetic terms and the
  tension distortion.
- The inputs at M_Z and the neutrino data are recalled values; the web proxy
  blocks the journals. The two fit papers cited were confirmed by search
  summaries only.

## 8. Independent review

An independent review, with its own code, verified:
- the one-loop RGEs (its own integration in the opposite convention
  reproduces `run_to`);
- the M_Z inputs;
- the SO(10) relations and the construction algebra;
- the CKM and PMNS conventions, the PMNS extraction and the m_ββ formula;
- that the reachability condition is the right one: the choice of uuᵀ
  versus its conjugate and the tetrahedron orientation do not matter;
- the demo's labels.

Its findings, all addressed:
- **Blocker 1.** The generic construction dropped two physical phases (rank
  16 instead of 18). The stored point had a χ² slope of 1204 in a missing
  phase, and a refit gave 7.39 instead of 36.18.
  - The phases are added.
  - Rank 18 is tested, and the rank-16 case is a regression test.
  - Every generic number is redone. With the corrected errors below, the
    best χ² is 2.93.
- **Blocker 2.** The stored pinned point was not exactly reachable: its
  exact projection onto W has χ² = 3155.
  - The pinned model is now parametrized exactly in W. The tests check that
    the projection changes nothing and that the Phase 1d reachability search
    agrees.
  - The review's exact pinned points (χ² = 267 in the old errors) were
    re-expressed and polished. My own continuation search improved on them.
- **Major:**
  - The pinned diagnosis (θ₁₃) was an artifact. The failure is in the
    charged sector, and the charged-only diagnostic is now recorded: the
    review found 109–166 in the old errors; here it is 73.
  - "About 1 degree of freedom" was wrong; it is −1.
  - The predictions changed at the better point. They are now given as
    ranges over the minima within Δχ² = 4, separated into stable and
    unstable.
- **Minor:**
  - δ is now the running value ± 0.03 rad.
  - The lepton errors are counted once.
  - An 8% theory error is added for the running, and the Bora comparison is
    reworded.
  - The dependence on the m_d error is computed: χ² = 8.1 at 10%.
  - Inverted ordering is fitted and excluded.
  - The two literature fits are cited.
  - M_I is taken from `model_scales`.
  - The internal pointers are fixed.
  - The tests now check the parametrization's completeness, local
    minimality (the first version's point would fail), exact membership in
    W, and a pinned-fit positive control on synthetic data.

New in the repair: W is identified as the kernel, within J = 2, of the
pinning condensate itself (section 3; tested).
