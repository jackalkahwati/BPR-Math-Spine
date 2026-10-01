# Phase 2b: can the leading corrections rescue the pinned branes?

2026-10-01. Status: an exact structural reduction plus bounded numerical fits.
The implementation is `bpr/kinetic_corrections.py`, with tests in
`tests/test_kinetic_corrections.py` and a demo in
`scripts/demo_kinetic_corrections.py`. **No empirical validation is claimed.**

## 0. Question

Phase 2 ([phase2_fermion_fit_2026-09-27.md](phase2_fermion_fit_2026-09-27.md))
found a conflict at leading order:
- with free brane positions, the minimal 10 + 126bar model fits the fermion
  data (χ² = 2.9);
- with the branes pinned at the Phase 1d tetrahedron, it fails the charged
  sector (best found χ² = 123.5 with neutrinos and 73.26 without; m_s about
  3.6 times too large).

Phase 2 named the leading corrections as the cheapest decisive check. These
are brane-localized fermion kinetic terms and the distortion of the geometry
by unequal brane tensions. Can corrections of their natural size repair the
pinned fit?

## 1. How the corrections enter (exact)

**The families are holomorphic.** On the flux sphere the three families are
the zero modes

    f_m(z) = √(3/(4πr²)) · (1, √2 z, z²)_m / (1 + |z|²),

which are orthonormal, and Σ_m |f_m|² = 3/(4πr²) everywhere (both tested).
At a brane, ψ(z_a) = √(3/(4πr²)) u_aᵀψ, where u_a = coherent_state(z_a) is the
unit spin-1 coherent state.

A brane couples to the *value* of the zero modes at its position: the m = +1
component in its own frame, which is the J_z rule of Phase 1. The value of a
holomorphic section at a point does not depend on the metric. So the brane
Yukawas stay rank 1, P_a = u_a u_aᵀ, whatever the geometry near the brane.

**Corrections change only the kinetic matrix.** Brane kinetic terms and
metric or flux distortions change the 4D kinetic term to ψ†Kψ, with K
Hermitian and positive. Canonical normalization ψ = K^(−1/2)χ then turns
every Yukawa bilinear ψᵀYψ into χᵀ(AYAᵀ)χ, with A = K̄^(−1/2) and
K̄ = conj(K).

The brane couplings are SO(10) symmetric, so the same A acts on H and F, and
hence on M_u, M_d, M_e, M_D and M_R. The SO(10) relations are kept (tested).

**Brane kinetic terms.** A term (κ_a/M²) ψ̄ iγ·∂ψ at brane a gives

    K̄ = 1 + Σ_a ε_a u_a u_a†,   ε_a = 3κ_a / (4π(rM)²).

A brane kinetic term alone dilutes its own brane's Yukawa:
A P_a Aᵀ = P_a/(1 + ε_a) (tested).

**Equal corrections do nothing.**
- The tetrahedral coherent states form a tight frame:
  Σ_a u_a u_a† = (4/3)·1 (tested). So equal brane kinetic terms only rescale
  every coupling, which the free couplings absorb.
- More generally, the 12 tetrahedral rotations permute the four brane states
  and act irreducibly on the families. By Schur's lemma, any A4-symmetric
  kinetic matrix is a multiple of 1 (tested).
- So equal brane tensions, which preserve A4, leave the pinned fit exactly
  unchanged. **Only brane-to-brane differences matter.**

**Natural sizes.**
- Brane kinetic terms: with κ_a ~ 1 at the cutoff and the Phase 1 control
  window rM = 3–3.9 (for the one-loop M_GUT), ε_a is 0.016–0.027 (tested).
- Tension distortion: the deficit angles are about 0.1. Their differences
  give normalization changes and brane displacements (via the condensate χ)
  of order the deficit spread, i.e. ≲ 0.1. This is an estimate: the 6D
  background with four unequal branes is not solved.

**Three corrections are fitted:**
1. **Brane kinetic terms:** four ε_a, with |ε_a| ≤ bound.
2. **A general kinetic matrix:** K̄ = exp(h) with h traceless Hermitian
   (8 parameters), with each component |t_k| ≤ bound. This covers every
   normalization correction, including the tension distortion.
3. **Brane displacement:** each brane moved along a geodesic by at most the
   bound (radians), with the couplings refitted.

Each is fitted along increasing bounds, each step starting from the previous
optimum plus 10 random restarts, from the Phase 2 pinned best points. Two
seeds are run, both with and without the neutrino data.

## 2. Results

χ²_min at each bound, taken over every run and counting points found at
smaller bounds. The scans fit 13 charged-sector observables, or 17 with the
neutrinos.

**Brane kinetic terms** (bound: max |ε_a|):

| bound | 0 | 0.01 | 0.03 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|---|---|
| charged only | 73.3 | 71.8 | 68.8 | 59.1 | 46.5 | 35.0 | 15.1 |
| with neutrinos | 123.5 | 121.3 | 118.1 | 106.2 | 93.5 | 78.4 | 67.4 |

**General kinetic matrix** (bound: max |t_k| of h = log K̄):

| bound | 0 | 0.02 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 1.0 | 1.5 |
|---|---|---|---|---|---|---|---|---|---|
| charged only | 73.3 | 66.3 | 57.1 | 44.7 | 27.5 | 15.9 | 7.2 | 0.6 | 0.0 |
| with neutrinos | 123.5 | 113.8 | 105.4 | 93.0 | 78.4 | 61.5 | 48.5 | 21.1 | 18.6 |

**Brane displacement** (bound: max geodesic displacement, radians):

| bound | 0 | 0.02 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|---|---|
| charged only | 73.3 | 68.7 | 62.1 | 52.1 | 35.1 | 22.7 | 12.4 |
| with neutrinos | 123.5 | 120.9 | 114.9 | 108.4 | 97.0 | 94.3 | 88.8 |

**All together** (a general kinetic matrix plus displacements; the bound is
a multiple x of the natural sizes, |t_k| ≤ 0.05x and displacement ≤ 0.1x):

| multiple of natural size | 0 | 0.5 | 1 | 2 | 3 | 5 |
|---|---|---|---|---|---|---|
| charged only | 73.3 | 54.9 | 40.6 | 22.1 | 12.5 | 8.0 |
| with neutrinos | 123.5 | 110.3 | 94.2 | 78.5 | 78.5 | 67.2 |

The combined fits with neutrinos were rugged: the first homotopy runs stalled
above the individual scans. They were therefore re-seeded from the
individual optima, so the combined values are never worse than either
correction alone at the same bound.

**Summary:**

| correction | natural size | χ² at natural size (charged / with ν) | bound needed for χ² ≤ number of observables (charged / with ν) |
|---|---|---|---|
| brane kinetic terms | \|ε_a\| ≤ 0.03 | 68.8 / 118.1 | > 0.5, i.e. > 20× natural / not reached |
| general kinetic matrix | \|t_k\| ≤ 0.05 (normalizations ±5%) | 57.1 / 105.4 | 0.5 (10×) / not reached by 1.5 (30×) |
| brane displacement | ≤ 0.1 rad | 52.1 / 108.4 | 0.5 rad (5×) / not reached by 0.5 rad |
| all together | 1× | 40.6 / 94.2 | 3× / not reached by 5× |

**Where the pinned fit fails (a numerical diagnostic, not a theorem).** Work
in the basis where M_d is diagonal. The off-diagonal entries of H and F then
cancel in M_d = H + F but add up in M_e = H − 3F.
- The generic best fit uses a large 2–3 entry, |H₂₃| = |F₂₃| ≈ 0.080 GeV, to
  give the muon its mass without feeding m_s.
- The pinned best points reach only about 0.028 GeV there. They must make
  m_μ from the diagonal F₂₂ instead, which raises m_s.

## 3. Verdict

- **Natural-size corrections do not rescue the pinned model.** All the
  leading corrections together, at their natural sizes, lower χ² from 73 to
  41 for the charged sector alone and from 123 to 94 with neutrinos. Both
  remain strongly excluded.
- **Even the charged sector alone needs large corrections.** It needs at
  least 3× the natural size with everything combined. Singly it needs
  kinetic matrices that change family normalizations by factors of about
  0.6–1.5 (|t_k| ≤ 0.5, 10× natural; the natural bound 0.05 means about
  ±5%), brane displacements of 0.5 rad (5×), or brane kinetic terms beyond
  20×.
- **With neutrinos, nothing tried fits.** No correction in the scanned
  ranges, up to 30× natural for a general kinetic matrix, brings χ² to the
  number of observables.
- **So the Phase 2 conflict survives next-to-leading order.** Pinning the
  branes at the vortex tetrahedron (kill check 11) and fitting the fermion
  data (kill check 5) are incompatible in the minimal model. Repairing that
  means changing the model, not refining it.

These statements rest on bounded local searches (section 4). A better
minimum at a given bound may exist. However, every kind of correction and
every seed gives the same picture.

## 4. Limitations

- **Corrections included:** only the leading ones. Derivative brane kinetic
  terms, SO(10)-breaking (non-universal) brane terms and loop corrections to
  the brane Yukawas are not included. Non-universal terms would add
  separate kinetic matrices for Q, u^c, d^c, L and e^c, which is more
  freedom.
- **Natural sizes are estimates.** The 6D background with four unequal
  branes is not solved, so the tension-induced corrections are bounded by
  their size, not computed.
- **Searches:** these are bounded local optimizations by homotopy from the
  Phase 2 pinned best points, with 10 random restarts per step, two seeds,
  and re-seeding for the combined case.
- **Inputs:** running, errors and inputs are those of Phase 2.

## 5. What is left

Three ways forward:
1. **A different stabilized configuration.** The data favour brane positions
   far from the regular tetrahedron: the Phase 2 continuation started at
   summed squared distances of 1.1–3.4, and this scan needs 0.5 rad even for
   the charged sector. The next well-posed step is to fit families of less
   symmetric configurations, for example a C₃ᵥ family with one brane at a
   pole, D₂ families, or 4 of the 6 zeros of a charge −6 condensate. One then
   asks whether some condensate or brane potential stabilizes a
   configuration that fits.
2. **Enlarge the Higgs sector** with a 120_H. On a brane it couples through
   ð̄² and costs ε².
3. **Give up pinning by the condensate.** Free positions fit (Phase 2), but
   the moduli problem of Phase 1 returns.

## 6. Independent review

Pending at the time of writing.
