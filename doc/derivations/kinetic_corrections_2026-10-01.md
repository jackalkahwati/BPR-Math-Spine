# Phase 2b: do the next-order corrections rescue the pinned branes?

2026-10-01. Status: an exact structural reduction plus bounded numerical fits.
The implementation is `bpr/kinetic_corrections.py`, with tests in
`tests/test_kinetic_corrections.py` and a demo in
`scripts/demo_kinetic_corrections.py`. **No empirical validation is claimed.**

The independent review (section 7) found that the first version omitted a
correction of the same order, the one-derivative brane vertex, and that this
correction does rescue the pinned model. It also found that the general
kinetic-matrix scan with neutrinos had missed fits that exist exactly. Both
are repaired. **The verdict of the first version is reversed.**

## 0. Question

Phase 2 ([phase2_fermion_fit_2026-09-27.md](phase2_fermion_fit_2026-09-27.md))
found a conflict at leading order:
- with free brane positions, the minimal 10 + 126bar model fits the fermion
  data (χ² = 2.9);
- with the branes pinned at the Phase 1d tetrahedron, it fails the charged
  sector. The best χ² found is 120.5 with neutrinos and 73.3 without, with m_s
  about 3.6–3.9 times too large.

Which corrections of the next order, relative size 1/(rM)², can repair the
pinned fit, and how large must they be?

## 1. How the corrections enter (exact)

**The families are holomorphic.** On the flux sphere the three families are
the zero modes

    f_m(z) = √(3/(4πr²)) · (1, √2 z, z²)_m / (1 + |z|²),

which are orthonormal, with Σ_m |f_m|² = 3/(4πr²) everywhere (both tested).
At a brane, ψ(z_a) = √(3/(4πr²)) u_aᵀψ, with u_a = coherent_state(z_a).

At leading order a brane couples to the *value* of the zero modes at its
position (the J_z rule of Phase 1), so its Yukawa matrix is rank 1:
P_a = u_a u_aᵀ.

**Class 1: normalization corrections.** Brane-localized fermion kinetic
terms, with or without derivatives, and the metric and flux distortion from
unequal brane tensions change only the 4D kinetic term, to ψ†Kψ. The value
of a holomorphic section at a point does not depend on the metric, so the
brane vertices are unchanged.

Canonical normalization then turns every Yukawa matrix into AYAᵀ, with
A = K̄^(−1/2) and K̄ = conj(K). The same A acts on H and F, and hence on all
of M_u, M_d, M_e, M_D and M_R (tested through `normalized()`, at a brane with
complex z, which distinguishes K from K̄).

- **Brane kinetic terms.** A term (κ_a/M²) ψ̄ iγ·∂ψ at brane a gives
  K̄ = 1 + Σ_a ε_a u_a u_a†, with ε_a = 3κ_a/(4π(rM)²). Alone, it dilutes
  its own brane's Yukawa: P_a → P_a/(1 + ε_a). It is a special case of a
  general kinetic matrix (tested).
- **Equal corrections do nothing.**
  - The tetrahedral states form a tight frame: Σ_a u_a u_a† = (4/3)·1.
  - The 12 tetrahedral rotations permute the brane states and act
    irreducibly on the families. By Schur's lemma, any A4-symmetric K is a
    multiple of 1 (both tested).
  - So only brane-to-brane differences matter.
- **An O(1) kinetic matrix erases pinning altogether.** Four points in
  general position in CP² are projectively equivalent. Some G ∈ GL(3) maps
  the tetrahedral vectors u(t_a) to multiples of any other four, and
  G = s·V·A splits into a family rotation V and a kinetic factor A.
  - `kinetic_map_to_tetrahedron` uses this to reproduce the Phase 2 generic
    best fit exactly (χ² = 2.9256) with the pinned branes plus a kinetic
    matrix of max|t_k| ≈ 1.0. This is a tested positive control.
  - So the pinning constraint is only as strong as the smallness of these
    corrections.

**Class 2: the one-derivative brane vertex.** By the J_z rule of
[yukawa_mechanisms_2026-09-26.md](yukawa_mechanisms_2026-09-26.md)
(Lemma 0), a fermion bilinear with a ð-derivatives couples to ð̄^(a+1)Φ. The
leading coupling is a = 0. The a = 1 operator 16·(ð16)·(ð̄²Φ) is allowed by
the same rule. At brane a it adds

    c_a [P_a + δ_a S_a],   S_a = u_a w_aᵀ + w_a u_aᵀ,   w_a = D(R_a) e₀,

where w_a is the family direction that a single ð makes nonzero at the brane,
and δ_a is independent for 10_H and 126bar_H.
- S_a is pure J = 2, like P_a, but is **not** annihilated by the condensate
  χ. The vertex therefore relaxes exactly the condensate condition that
  defines the pinned subspace W.
- With δ unbounded, the P_a and S_a span all of J = 2 (rank 5, tested). That
  is the free-position model of Phase 1.

**Class 3: brane displacements.** Unequal tensions also deform the
condensate and move its zeros, and with them the branes.

**Natural sizes, with unit coefficients at the cutoff M:**
- *Brane kinetic terms:* κ_a ~ 1 gives ε_a = 0.016–0.027 for rM = 3–3.9
  (tested).
- *Vertex:* two extra derivatives at the cutoff give δ ~ 1/(rM)² = 0.07–0.11.
  Including the ð eigenvalues on the zero modes and on the l = 3 Higgs level
  (about √2·√10/2 ≈ 2.2) gives up to 0.15–0.25. The vertex lacks the 3/(4π)
  normalization factor of ε, because both of its legs are the same fermion
  bilinear as the leading Yukawa.
- *Tension effects:* a deficit spread of order 0.1 gives normalization
  changes of a few percent and displacements ≲ 0.1 rad. This is an estimate;
  the 6D background with four unequal branes is not solved.
- *Strong coupling:* with strong-coupling naive dimensional analysis at M,
  the coefficients could be up to 8π larger. For brane kinetic terms that
  means ε ≈ 0.4–0.7.

## 2. Results

Each scan fits the pinned model with the correction bounded. It runs along
increasing bounds, with each step starting from the previous optimum plus
random restarts and finishing with a tight polish. Further points come from
re-seeded and reverse-homotopy fits, and from the review's points re-polished
here.

Every entry is the lowest χ² found at that bound or any smaller one. All
entries are **upper bounds** on the true minimum. The charged-only fits use 13
observables, the full fits 17.

**Brane kinetic terms** (bound: max |ε_a|):

| bound | 0 | 0.03 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|---|
| charged only | 73.3 | 68.8 | 59.1 | 46.5 | 35.0 | 15.1 |
| with neutrinos | 120.5 | 117.9 | 106.2 | 93.5 | 78.4 | 67.4 |

**General kinetic matrix** (bound: max |t_k| of h = log K̄; the exact
positive-control point sits at 1.0015):

| bound | 0 | 0.02 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| charged only | 73.3 | 66.1 | 56.7 | 43.9 | 27.5 | 15.9 | 7.2 | 1.5 | 0.0 |
| with neutrinos | 120.5 | 113.8 | 105.4 | 93.0 | 78.4 | 61.5 | 15.6 | 2.9 | 2.9 |

**Brane displacement** (bound: max displacement; a box of side b/√2 per
tangent component, inscribed in the disc):

| bound | 0 | 0.02 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|---|---|
| charged only | 73.3 | 68.7 | 62.1 | 51.9 | 35.1 | 22.7 | 12.4 |
| with neutrinos | 120.5 | 120.5 | 114.9 | 106.3 | 97.0 | 94.3 | 88.8 |

**Kinetic matrix and displacement together** (bound: multiple x of the
natural sizes, |t_k| ≤ 0.05x and displacement ≤ 0.1x):

| x | 0 | 0.5 | 1 | 2 | 3 | 5 |
|---|---|---|---|---|---|---|
| charged only | 73.3 | 54.9 | 40.4 | 22.1 | 12.5 | 8.0 |
| with neutrinos | 120.5 | 110.3 | 91.3 | 78.5 | 78.5 | 21.2 |

**One-derivative vertex** (bound: max |δ_a|, a disc in the complex plane):

| bound | 0 | 0.01 | 0.03 | 0.06 | 0.1 | 0.15 | 0.2 | 0.3 |
|---|---|---|---|---|---|---|---|---|
| charged only | 73.3 | 50.7 | 27.6 | 17.8 | 11.7 | 10.1 | 0.55 | 0.11 |
| with neutrinos | 120.4 | 114.5 | 109.9 | 106.1 | 50.8 | 50.8 | 32.1 | **4.5** |

The with-neutrino landscape is rugged. Repeated values such as 78.5 at 2×
and 3× in the combined scan, or 50.8 at δ ≤ 0.1 and 0.15, are stalled
searches, not plateaus of the true minimum. Several of the lowest points
came from the review's searches and from fits seeded from the charged-only
optima.

**Summary.** The last two columns give the smallest scanned bound with χ²
at most the number of observables; the true threshold lies between it and
the previous scanned bound.

| correction | natural size | χ² at natural size (charged / with ν) | fits charged at | fits with ν at |
|---|---|---|---|---|
| brane kinetic terms | ε ≤ 0.03 | 68.8 / 117.9 | > 0.5 (> 17×) | > 0.5 |
| general kinetic matrix | \|t_k\| ≤ 0.05 | 56.7 / 105.4 | 0.5 (10×) | 0.5 (10×) |
| brane displacement | ≤ 0.1 rad | 51.9 / 106.3 | 0.5 (5×) | > 0.5 |
| kinetic matrix + displacement | 1× | 40.4 / 91.3 | 3× | > 5× (21.2 at 5×) |
| **one-derivative vertex** | **δ ≈ 0.07–0.25** | **11.7 at δ ≤ 0.1** | **≤ 0.1** | **0.3** |

**Where the leading-order pinned fit fails** (a numerical diagnostic, not a
theorem). Work in the basis where M_d is diagonal, so the off-diagonal
entries of H and F cancel in M_d = H + F and add up in M_e = H − 3F.
- The generic best fit has |H₂₃| = |F₂₃| ≈ 0.080 GeV, and its |H₂₂| ≈ 0.013
  GeV partly cancels F₂₂ in M_d.
- The pinned points reach only |H₂₃| ≈ 0.028 GeV.
- They also have |H₂₂| ≈ 0.04–0.08 GeV, adding to F₂₂ instead of cancelling
  it, which raises m_s.

## 3. Verdict

- **Normalization corrections at natural size do not rescue the pinned
  model.** Brane kinetic terms, tension distortions and brane displacements,
  even all together, leave χ² ≳ 40 for the charged sector alone and ≳ 90
  with neutrinos.
  - They fit only at about 3–10× their natural size.
  - An O(1) kinetic matrix erases pinning altogether.
  - With strong-coupling coefficients (up to 8π), brane kinetic terms would
    reach ε ≈ 0.4–0.7, where the charged sector nearly fits.
- **The one-derivative brane vertex, of the same order, does rescue it.**
  - The charged sector fits at |δ| ≤ 0.1, inside the natural range.
  - All the data fit at |δ| ≤ 0.3 (χ² = 4.5, every pull within 2σ). That is
    at or just above the top of the natural range, 0.07–0.25.
- **So the pinned model is not excluded at next-to-leading order.** Kill
  check 5 for the pinned model becomes conditional: it needs derivative
  vertex couplings near the top of their natural size.
- **The rescue costs the predictivity pinning promised.**
  - The vertex adds 16 real parameters.
  - It relaxes exactly the condensate condition that made pinning restrict
    the flavour sector. Unbounded, it gives back the free-position model.
  - So pinned BPR-6D, like free BPR-6D, makes **no flavour prediction**.
    Its stable consequences (normal ordering, Σm_ν ≈ 0.06–0.07 eV,
    m_ββ ≈ 0.4 meV) come from the 10 + 126bar structure, not from the
    branes.

## 4. Limitations

- **Search-based upper bounds.** The scans are bounded local optimizations,
  so every entry is an upper bound, and some with-neutrino rows are
  stalled. None of the review's independent searches found substantially
  lower values for the normalization corrections at natural size.
- **Natural sizes are unit-coefficient estimates.** Strong coupling could
  raise them by up to 8π. The tension effects are estimated, not computed.
- **Not included:**
  - non-universal, SO(10)-breaking brane terms, which would give separate
    kinetic matrices for Q, u^c, d^c, L and e^c;
  - two-derivative vertices, at relative order 1/(rM)⁴;
  - loop corrections.
- **Basis- and shape-dependent bounds.**
  - The general-K bound is a box in a fixed Gell-Mann basis. The natural box
    allows K̄ eigenvalues 0.92–1.12, i.e. ±5% in A and ±10% in K.
  - The displacement bound is a box inscribed in the disc. The review's disc
    fits are slightly lower (50.5 against 51.9 at 0.1, charged).
- **Inputs:** running, errors and inputs are those of Phase 2.

## 5. What is left

1. **Flavour is not predicted, with or without pinning.** Predicting it
   would need the brane couplings (c_a, d_a, δ_a) from a UV completion. That
   points to Phase 3, the string embedding.
2. **Stabilization stands on its own.** Kill check 11 (brane positions)
   stays conditional and metastable (Phase 1d). It is no longer in conflict
   with the fermion data once the derivative vertex is included, so the
   positions no longer need to change.
3. **Optional sharpening:**
   - the vertex coefficients in a concrete UV completion;
   - two-loop running and thresholds, which shift the targets by about 10%;
   - a dedicated global search of the with-neutrino vertex landscape.

## 6. Repairs to Phase 2

The review also found a pinned with-neutrino minimum below Phase 2's stored
one: 121.55 from a random start, and 120.45 after polishing. It is now
Phase 2's stored best pinned point, and the Phase 2 note is updated.

## 7. Independent review

An independent review, with its own implementation, verified:
- the conventions: K̄ = 1 + εuu† and Y → AYAᵀ with A = K̄^(−1/2);
- that the zero-mode profiles are metric-independent, and that the
  universality of A, the tight frame, Schur's lemma and the ε range hold;
- every stored χ², and that the tables match the code.

Its findings and the repairs:
- **Blocker: the omitted one-derivative vertex.**
  - It is implemented, with its J = 2 structure and its span tested.
  - It is scanned, and the verdict is rewritten: the vertex rescues the
    pinned model at the top of its natural size.
- **Major 1: the general-K row with neutrinos was a search failure.**
  - The exact projective construction is implemented and tested as a
    positive control.
  - Reverse homotopies and the review's points give χ² = 15.6 at 0.5 and
    2.9 at 0.75.
- **Major 2: the with-neutrino numbers were not converged.**
  - All fits now end with a tight polish, and the natural-size points are
    re-polished.
  - The values are labelled upper bounds, and the stalled rows are named.
  - The review's lower pinned point is adopted (section 6).
- **Major 3: the natural size rested on an assumption.** The unit-coefficient
  assumption and the strong-coupling range are now stated.
- **Minor:**
  - The thresholds are given as brackets.
  - The box shapes are described as they are.
  - `correction_size` now returns the bounded quantity.
  - The demo labels are fixed.
  - Derivative brane kinetic terms are noted as covered by the general K.
  - The 2–3 diagnostic is reworded (H₂₂, not F₂₂).
  - The old "J = 0 pieces" wording in Phase 1 and Phase 2 is corrected: the
    corrections are a non-unitary congruence plus rank-2 vertex pieces.
- **Tests:**
  - The convention is now tested through `normalized()`.
  - BKTs are tested as a special case of a general K, alongside the positive
    control and the vertex structure.
  - The verdict is tested on recomputed stored points instead of the
    hard-coded table.
  - Local minimality of stored natural points is tested.
