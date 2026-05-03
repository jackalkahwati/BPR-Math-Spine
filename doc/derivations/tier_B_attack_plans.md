# Tier B Gaps — Concrete Attack Plans

> **Status:** April 2026 — these are real open research problems in BPR,
> but they are *attackable*, not intractable. Each has a clearly scoped
> calculation that would close it if executed. Estimated effort per gap:
> 3–12 months of dedicated technical work.

## Gap #3: The 1.5% GUT residual

### Current status

`bpr/gauge_unification.py::forward_threshold_corrections` closes 98.1% of
the gauge-coupling unification gap at 1-loop and 92.0% at 2-loop using
three-channel boundary corrections. The residual 1.5% at 2-loop is
attributed to "superheavy gauge boson thresholds from unknown heavy-state
spectrum" — standard in non-SUSY unification (Weinberg 1980, Georgi 1980).

### Attack: compute the heavy gauge boson spectrum from the CS bulk

**The target.** The BPR UV completion is U(1)_p Chern-Simons theory on S³
with boundary compact boson at R² = z/2. The SM gauge bosons — plus the
"heavy" gauge bosons X, Y with masses near M_GUT — are identified with
specific winding/momentum modes of the boundary CFT.

**What needs to be computed:**

1. **Identify the heavy modes.** The SM gauge bosons W, Z, γ, g correspond
   to specific light modes of the c=1 boson at R² = 3. The heavy X, Y
   bosons of typical GUTs (SU(5), SO(10)) should be higher-momentum modes
   of the same CFT. Step: enumerate the winding-momentum spectrum of the
   c=1 boson at R² = 3 with UV cutoff L_max = √p ≈ 323, and identify
   which modes correspond to which gauge bosons.

2. **Compute their masses.** Each mode (n, w) has mass-squared
   M²_{n,w} = n²/R² + w²R² (standard c=1 formula). With R² = 3:
      M²_{0,1} = 3,  M²_{1,0} = 1/3,  M²_{1,1} = 10/3, ...
   Need to identify which of these sit near M_GUT (dimensionless mass
   ~ p^(1/4)) and include them in the threshold correction.

3. **Integrate them out.** The one-loop threshold correction from heavy
   modes is:
      Δ(1/α_i) = (1/12π) Σ_X b_i^X ln(μ²/M_X²)
   where b_i^X is the SM β-function coefficient for mode X. Summing
   over the heavy modes identified in step 2 gives a calculable
   correction.

4. **Compare to the 1.5% gap.** If the computed heavy-mode correction
   reproduces the 1.5% residual without tuning, the gap is closed. If
   it gives 0.8% or 2.5%, we learn something about the CS spectrum.

**Technical difficulty.** Steps 1–2 require knowing how the SM gauge
group embeds into the c=1 boson's enhanced symmetry structure at R² = 3.
R² = 3 is not a standard self-dual point, so this embedding has to be
worked out. Step 3 is mechanical once the spectrum is known.

**Estimated effort:** 6–9 months for a CFT specialist. The key deliverable
is a mode-spectrum table with each mode tagged by its SM gauge-boson
representation.

**Success criterion.** The heavy-mode threshold correction reproduces the
observed sin²θ_W and α_s(M_Z) to ≤0.5% without tuning.

**What happens if it fails.** If the CFT at R² = 3 does not contain a
GUT-compatible heavy gauge-boson spectrum, BPR would need to either (a)
modify the embedding (allowed), (b) revise R² (fundamental), or (c)
acknowledge that the 1.5% residual is the BPR equivalent of the
traditional "expected non-SUSY threshold uncertainty," without further
decomposition.

---

## Gap #7: Quantum gravity proper

### Current status

BPR derives gravity as *induced* via the Sakharov mechanism: integrating
out the p boundary scalar modes generates an Einstein-Hilbert action with
Newton's constant G_N = 48π²/(pΛ_b²). This gives classical GR at long
distances and predicts M_Pl² = pΛ_b²/(48π²) at 0.2% agreement with
observation.

**What's missing:** Induced gravity is an *effective* description. It
doesn't directly address:

- **UV completion.** What is the quantum theory of gravitons themselves,
  beyond the leading 1-loop induced action?
- **Black hole information.** Does BPR resolve the black hole information
  paradox via boundary non-locality?
- **Singularities.** Do classical GR singularities (Big Bang, black hole
  interiors) get resolved in the substrate?
- **Holographic dictionary.** How exactly do bulk GR observables map to
  boundary CS observables at finite N?

### Attack A: Compute graviton propagator from boundary 2-point function

**May 2026 MVP status.** A leading effective-field-theory scaffold now
exists in `bpr/graviton_propagator.py`, with the derivation note
`doc/derivations/graviton_propagator_from_boundary_tt.md`. It implements
the physical spatial transverse-traceless propagator
`D_ij,kl(k) = P^TT_ij,kl/(M_Pl^2 |k|^2)` using the Sakharov normalization
`M_Pl^2 = p Lambda_b^2/(48π²)`. The first finite-`p` stress-tensor
normalization is now derived from the Hopf/S² mode count:
`D_p/D_GR = p/(floor(sqrt(p))+1)^2`. The known induced `R²` term has also
been checked: it creates the scalaron/Starobinsky scalar sector and gives
zero correction to the physical transverse-traceless spin-2 propagator. The
universal Weyl/Ricci-squared heat-kernel coefficient now gives a spin-2
cutoff coefficient per RG log:
`eta_TT = -[N_S2(p)/(20p)] log(Lambda/mu)`, which is `-0.05010 log(Lambda/mu)`
for `p = 104761`. The RG window is now made concrete by setting
`mu = E_probe` and `Lambda = Lambda_b`; the probe-specific shift is
`delta_TT(E) = eta_TT(E) (E_probe/Lambda_b)^2`.
For gravitational waves and ringdown, `E_probe = hbar 2πf`; a 100 Hz signal
gives `|delta_TT| ~ 1e-78`, so detector-band gravitational waves are not a
realistic observable for this specific curvature-squared correction. The
correction has a hard EFT-envelope maximum at
`E_probe/Lambda_b = exp(-1/2) = 0.6065`, where
`delta_TT,max = -N_S2/(40ep) = -0.00922` for `p = 104761`.
The scalar sector is now separately audited in
`doc/derivations/scalaron_sector_from_boundary_r2.md`: the minimal induced
`R²` coefficient derives the Starobinsky potential shape and keeps
`n_s ≈ 0.968`, `r ≈ 0.003`, but its amplitude is too large by about `8e6`.
Closing the scalar sector therefore requires deriving the winding/anyon-loop
normalization of `alpha_R2`, not just the one-loop heat-kernel term. The old
`1 + W_c/W_bare` estimate gives only `~1.51`, while the observed scalar
amplitude requires `~7.98e6`; `scalaron_normalization_diagnostic(...)` now
records this as an open coefficient-level gap.

**The target.** The effective graviton h_{μν} is the induced fluctuation
of the metric from boundary modes. Its propagator is set by the
2-point function of the boundary energy-momentum tensor:

    ⟨h_{μν}(x) h_{ρσ}(y)⟩ ∝ ⟨T_{μν}(x) T_{ρσ}(y)⟩_bndy

For the c=1 compact boson CFT at R² = 3, ⟨TT⟩ is known. The calculation
is then: project the boundary T_{μν} to the bulk metric perturbation via
the holographic dictionary and compute the leading graviton propagator.

**Deliverable.** A closed-form graviton propagator with:
- Correct tensor structure (spin-2)
- Correct Newton's constant G_N = 48π²/(pΛ_b²)
- Leading finite-`p` stress-tensor normalization from the S² cutoff mode
  count
- Weyl/Ricci-squared spin-2 coefficient per RG log
- Probe-energy RG window helper
- Gravitational-wave frequency map `E_probe = hbar 2πf`
- Near-cutoff maximum `|delta_TT| < 1%` for the derived curvature-squared term
- Scalaron-sector audit: Starobinsky shape derived, amplitude normalization
  still open at the `~8e6` coefficient level
- Winding/anyon normalization diagnostic showing the old `1 + W_c/W_bare`
  factor is insufficient
- Remaining search for near-cutoff systems where the spin-2 correction is not
  crushed by `(E_probe/Lambda_b)^2`

**Remaining effort:** 6–12 months for the exact finite-`p` dictionary. The
c=1 CFT part is standard; mapping real observables to the correct probe scale
and bulk observable from the CS/WZW correlator remains the hard part.

### Attack B: Black hole information via boundary unitarity

**The target.** U(1)_p CS on S³ is a topological theory with a
finite-dimensional Hilbert space of dimension p (one state per anyon).
This is exactly finite-dimensional, so unitarity is manifest. If gravity
is induced from this theory, BPR has a built-in answer to information
preservation: the boundary theory is always unitary, so black hole
evaporation cannot destroy information.

**What's to compute:** Show explicitly that the bulk Hawking evaporation
process in the induced-gravity limit maps to a unitary evolution on the
boundary CS Hilbert space. This is conceptually parallel to the
ER=EPR / Page curve arguments in string theory but should be much more
explicit in a finite-dimensional theory.

**Deliverable.** A Page-curve calculation for a BPR black hole,
demonstrating unitarity and giving a specific entropy-of-entanglement
formula in terms of (p, z, N_BH).

**Effort:** 9–12 months. Requires identifying the BH microstate counting
in CS terms (partial results exist from TQFT literature on BH entropy).

### Attack C: Singularity resolution via substrate discreteness

**The target.** At a Big Bang or BH singularity, GR gives infinite
curvature. If curvature in BPR is bounded by the substrate cutoff
Λ_b ~ 8×10¹⁷ GeV, singularities cannot form — they get "smoothed" at
the substrate scale.

**What's to compute:** Modify the FRW or Schwarzschild metric by
introducing a Λ_b-dependent UV cutoff (analogous to Loop Quantum Cosmology
bounce). Predict the bounce temperature, duration, and any observational
signatures (CMB imprint, primordial GW).

**Deliverable.** A BPR bounce cosmology with numerical predictions for
T_bounce, H(bounce), and the pre-bounce CMB imprint.

**Effort:** 3–6 months. Much of the LQC machinery can be borrowed with
BPR substituting its Λ_b for LQC's critical density.

### Priority ranking

| Attack | Leverage | Effort | Risk |
|---|---|---|---|
| A (graviton propagator) | HIGH — gives quantum gravity predictions | 6–12mo | Medium (holographic dictionary is hard) |
| B (BH information) | MEDIUM — solves an open problem | 9–12mo | High (conceptual depth) |
| C (singularity resolution) | MEDIUM — gives cosmology predictions | 3–6mo | Low (machinery exists) |

**Recommended first target: Attack C.** Shortest path to a testable
prediction. Gives BPR an inflation/bounce story with specific numerical
outputs that CMB-S4 can test.

---

## Overall Tier B assessment

Both Gap #3 and Gap #7 are **real** open problems, but they are
*incremental* research problems, not fundamental obstacles. Each has:

- A concrete calculation that would close it
- Existing technology (CFT, CS theory, LQC) that applies
- A success criterion (percent-level agreement with observation)
- A failure mode (specific revisions to BPR if the calculation fails)

This is the right posture for a theory of everything: not "it's obviously
right, someone will figure out the math" but "here are the calculations,
here's who can do them, here's how long it'll take."

If BPR attracted 2–3 postdocs with CS/CFT expertise for 18 months, both
gaps could plausibly be closed.

---

*April 2026 — concrete attack plans for Gap #3 (GUT residual) and Gap #7
(quantum gravity). Each is a scoped research problem with a clear success
criterion and known technical approach.*
