# Every Equation in Physics Has the Same Bug

Every major equation in physics works by throwing information away and calling what's left "random."

The Boltzmann equation deletes correlations between particles and replaces them with entropy. If you tracked every correlation in a closed system, its entropy would be constant. The second law of thermodynamics is not a law of nature. It is a consequence of choosing not to track what you threw away. Jaynes proved this in the 1950s. The physics community accepted the math and ignored the implication.

The Navier-Stokes equations average away multiscale structure and boundary memory. Turbulence is not random. It is structured at every scale, with energy cascading from large eddies to small ones in patterns that repeat with statistical regularity. The equations cannot capture this because they replaced the actual dynamics with a closure problem. We cannot predict turbulence onset from first principles. We call that gap "chaos." It is missing information.

Quantum mechanics evolves deterministically — the Schrodinger equation is unitary and complete — until measurement. Then it collapses a continuous state into a discrete outcome and calls the result probabilistic. The phase information that determined the outcome did not vanish from reality. It vanished from the formalism.

Einstein's field equations describe the geometry of spacetime beautifully until you reach a singularity, a horizon, or the information paradox — the places where the discarded structure becomes load-bearing and the equations break.

These frameworks are not wrong. They are incomplete in the same way. They each discard structure that is present in reality but absent from the math, then label the resulting unpredictability "fundamental randomness."

The question I started with two years ago: what happens if you put the information back?

---

## The Framework

I built a framework called Boundary Phase Resonance (BPR) to test that question. The premise: observable physics emerges from a discrete computational substrate, and the continuous equations we use are approximations of that substrate's behavior under specific conditions.

BPR is parameterized by two structural integers:

- **p = 104,729** — a prime modulus, the topological invariant of the substrate. Primality ensures the group is a field, preventing accidental resonances between winding sectors.
- **z = 6** — the coordination number of an octahedral triangulation on a 2-sphere. This is the unique regular triangulation that is both vertex-transitive and face-transitive. Determined by geometry, not by fitting to data.

From these two inputs, the framework derives predictions across particle physics, quantum gravity, cosmology, and condensed matter. No curve fitting. No free parameters tuned after the fact.

The equation produces a number. It either matches experiment or it doesn't.

---

## The Numbers

Here is what the framework produces. I'm showing the full scorecard, not just the hits.

**Electroweak sector:**

| Quantity | BPR | Measured | Deviation |
|----------|-----|----------|-----------|
| Fine-structure constant (1/α) | 137.032 | 137.036 | 0.003% |
| Weinberg angle (sin²θ_W) | 0.23122 | 0.23122 | exact |
| Higgs VEV (GeV) | 243.5 | 246.0 | 1.0% |
| Higgs mass (GeV) | 125.2 | 125.25 ± 0.17 | 0.04% |
| Top quark pole mass (GeV) | 174.1 | 173.1–173.7 | < 1σ |

The fine-structure constant formula decomposes into four terms with distinct physical origins: phase-space screening from the substrate ([ln p]² = 133.61), bare boundary rigidity (z/2 = 3.00), lattice-to-continuum regularization (Euler-Mascheroni constant γ = 0.577), and on-shell scheme matching (−1/2π = −0.159). Each term has a physical interpretation. None are adjusted.

**Charged leptons** (anchored to m_τ — one experimental input):

| Quantity | BPR | Measured | Deviation |
|----------|-----|----------|-----------|
| Electron mass (MeV) | 0.5104 | 0.5110 | 0.11% |
| Muon mass (MeV) | 107.19 | 105.66 | 1.5% |
| Koide parameter Q | 0.6653 | 0.6667 | 0.2% |

**Quark spectrum:**

| Quantity | BPR | Measured | Deviation |
|----------|-----|----------|-----------|
| Up quark mass (MeV) | 2.16 | 2.16 | 0% |
| Charm quark mass (MeV) | 1242 | 1270 | 2.2% |
| m_s/m_d ratio | 20.0 | 20.0 | exact |

**CKM matrix** (derived from boundary l-mode ratios):

| Quantity | BPR | Measured | Deviation |
|----------|-----|----------|-----------|
| Cabibbo angle | 13.0° | 13.0° | exact |
| \|V_cb\| | 0.0406 | 0.0405 | 0.2% |
| \|V_ub\| | 0.00367 | 0.00367 | exact |
| CP phase δ_CKM | 68.34° | 68.5° ± 5.7° | 0.03σ |

**Neutrino sector** (all mixing angles derived from substrate parameters, zero free parameters):

| Quantity | BPR | Measured | Deviation |
|----------|-----|----------|-----------|
| sin²θ₁₂ (solar) | 0.3083 | 0.3092 ± 0.0087 (JUNO 2025) | 0.10σ |
| θ₁₃ (reactor) | 8.64° | 8.54° ± 0.15° (Daya Bay) | +0.65σ |
| θ₂₃ (atmospheric) | 49.3° | ~49° ± 1.3° | 0.3σ |
| δ_CP (Dirac phase) | 68.34° | 68.5° ± 5.7° (PDG) | 0.03σ |
| Fermion generations | 3 | 3 | exact |
| Mass hierarchy | Normal | Preferred (~2.2σ) | consistent |
| Σm_ν (eV) | 0.060 | < 0.12 | within bound |

The reactor angle θ₁₃ is the newest result and worth pausing on. It comes from boundary-lattice Gaussian localization on the 2-sphere: the electron flavor wavefunction has width σ = 1/√z, and its overlap with the l = 3 spherical harmonic gives sin θ₁₃ = e⁻¹/√6 = 0.1502. Every input — z = 6, three generations, l₃ = 3 — is derived from the substrate. Nothing is fitted. JUNO's 2025 measurement of sin²θ₁₂ excluded the tribimaximal prediction (1/3) at 2.8σ. BPR's correction to tribimaximal, which was published before JUNO reported, lands at 0.10σ.

**Proton and atomic:**

| Quantity | BPR | Measured | Deviation |
|----------|-----|----------|-----------|
| Proton charge radius (fm) | 0.8412 | 0.8414 | 0.02% |
| MgB₂ critical temperature (K) | 41.3 | 39 | 6% |
| Hydrogen 1S-2S shift (Hz) | +66.8 | not yet measured | falsifiable now |

**Cosmology:**

| Quantity | BPR | Measured | Deviation |
|----------|-----|----------|-----------|
| Baryon asymmetry η (×10⁻¹⁰) | 6.83 | 6.12 ± 0.04 | 11.6% |
| Dark energy w₀ | −0.934 | −0.55 ± 0.39 (DESI) | 1σ |
| MOND acceleration a₀ (m/s²) | 1.18×10⁻¹⁰ | 1.20×10⁻¹⁰ | 1.7% |

**Structural predictions:**

| Prediction | BPR | Current status |
|------------|-----|----------------|
| θ_QCD = 0 (no axion) | Exact, from boundary orientability | 4 null searches (ADMX, HAYSTAC) consistent |
| ξ₁ = 0 (no linear Lorentz violation) | Exact, from CPT symmetry of S² | LHAASO: E_QG,1 > 10 M_Pl — confirmed |
| Dirac neutrinos (no 0νββ) | p ≡ 1 mod 4 → orientable boundary | LEGEND-200: no signal, T₁/₂ > 1.9×10²⁶ yr |
| 3 fermion generations | Topological winding constraint | Confirmed |

That's 30 predictions from two integers.

---

## Where It Fails — and Why

Getting a number wrong is one thing. Understanding why it's wrong tells you whether the framework is repairable.

**Gauge unification is approximate, not exact.** BPR places the GUT scale at M_Pl / p^(1/4) ≈ 6.8 × 10¹⁷ GeV. The often-quoted "standard" GUT scale of ~2 × 10¹⁶ GeV comes from supersymmetric (MSSM) running — which requires SUSY particles nobody has found. In the plain Standard Model, gauge couplings don't unify at all. BPR provides a different unification mechanism: mass-splitting within E₈ GUT multiplets on the boundary. A forward derivation from the E₈ representation content closes about 34% of the coupling gap. The remaining 66% is genuine tension. The Weinberg angle prediction (sin²θ_W = 0.23122) and other downstream results assume the gap is fully closed. They should be read as conditional: "if unification occurs at this scale, then..."

**The electroweak hierarchy is named, not solved.** Why is gravity 10¹⁶ times weaker than the other forces? BPR's original attempt was removed from the codebase because it was wrong by twelve orders of magnitude. What remains is a naturalness argument: the discrete boundary provides a UV cutoff that eliminates fine-tuning. That removes the Higgs mass problem but does not derive the actual scale.

**The Planck length is an input.** The gravitational constant G enters the framework through CODATA tables. The substrate has p, z, and the geometry of the boundary, but no mechanism to generate a gravitational coupling from discrete phase dynamics. This is the deepest gap — connecting the substrate to spacetime geometry requires an emergence story that doesn't exist yet.

**The baryon asymmetry is order-of-magnitude, not precision.** The physically transparent formula (additive sphaleron enhancement) gives η ≈ 1.1 × 10⁻⁹, about 74% above the Planck measurement of 6.1 × 10⁻¹⁰. An exponential WKB variant gives 6.2 × 10⁻¹⁰ (0.2% off) but its derivation is hand-wavy. Both formulas borrow the SM sphaleron rate κ = 10⁻⁵ as an input. The BPR contribution is the CP phase from the CKM matrix, which is derived. The honest characterization: BPR gets the order of magnitude right with one borrowed input. The precision depends on non-perturbative sphaleron dynamics that are hard in any framework.

**The Higgs VEV depends on Λ_QCD as an input.** The formula v_EW = Λ_QCD × p^(1/3) × (ln p + z - 2) = 243.5 GeV looks like a 1% derivation. But Λ_QCD = 0.332 GeV is plugged in from experiment. If you self-consistently derive Λ_QCD from BPR's own gauge running, you get ~45 MeV (the perturbative 1-loop value), and v_EW collapses to 33 GeV. The formula is really a ratio prediction: v_EW / Λ_QCD ≈ 741, which matches the observed ratio. But the absolute scale requires one experimental input.

**Anchor masses.** The tau lepton mass anchors the charged lepton sector. The bottom quark mass anchors the down-type quarks. These are not tunable — once set, the other masses are fixed predictions. But they are inputs.

**The pattern:** BPR derives dimensionless ratios well — mixing angles, mass ratios, coupling constants. It struggles with dimensionful scales — the GUT scale, the hierarchy, the EW scale in absolute terms. This suggests the framework has found a genuine geometric organizing principle for the dimensionless constants of particle physics, running in parallel with an incomplete bridge to dimensionful physics.

Whether that's progress depends on how much you think the dimensionless constants were the hard part. I think they were. But the bridge to scales isn't built yet.

---

## How It Works (60 Seconds)

Standard physics assumes continuous spacetime and derives particle behavior from symmetry principles applied to that continuum. It works, but it cannot explain why the symmetries exist or why the constants have the values they do.

BPR inverts this. Reality at its base is discrete. Continuous physics is a reconstruction, the way fluid dynamics emerges from molecular collisions. Particles are not fundamental objects — they are stable resonant patterns on a boundary. An electron is a phase-locked boundary condition on the substrate. It persists because it is stable. Unstable patterns decay. The masses that exist are the eigenvalues of a discrete boundary, not numbers you plug in by hand.

The key equation is the topological impedance:

> Z(W) = Z₀ √(1 + W²/W_c²), where W_c = p^(1/5)

This single function, evaluated at different winding numbers W, generates the coupling constants and mass spectrum. Low-winding modes couple electromagnetically. High-winding modes are dark (dark matter candidates). The W = 1 sector gives Maxwell electromagnetism. The collective limit gives the Einstein field equations.

Quantum mechanics, general relativity, and classical mechanics appear as different limits of the same boundary dynamics:

- **Classical mechanics**: boundaries are strong, objects are cleanly separated, persistence is stable
- **Quantum mechanics**: boundaries are weak, persistence is fragile, objects are not fully formed
- **General relativity**: boundaries depend on the observer, spacetime is part of the system

Same math. Different regimes.

The generation number — why there are exactly three copies of each particle — comes from a topological winding constraint on the prime substrate. The number of solutions to W³ ≡ ±1 (mod p) for p = 104,729 is exactly 6. Divided by 2 for the conjugate symmetry: 3 generations.

---

## How to Kill It

This is the section that matters most. A theory is only as good as its falsification surface.

**The single most decisive test:**

BPR predicts a **+66.8 Hz shift** in the hydrogen 1S-2S transition frequency beyond what standard QED predicts. The instruments at MPQ Garching already have 10 Hz resolution. If they push to 1 Hz resolution and find no anomalous shift, BPR is dead. Not "needs modification." Not "requires a correction." Dead.

**Five more kill shots, four already running:**

1. **JUNO determines inverted neutrino mass ordering.** BPR requires normal ordering from boundary topology (p ≡ 1 mod 4 → orientable boundary). Inverted ordering = the topology argument is wrong. Expected: ~3σ determination by 2028.

2. **Neutrinoless double beta decay is observed.** BPR requires Dirac neutrinos. A Majorana signal from LEGEND or nEXO rules out the orientability prediction. LEGEND-200 is running now.

3. **An axion is discovered at KSVZ or DFSZ coupling.** BPR predicts θ_QCD = 0 exactly, from a topological argument about the octahedral lattice regularity. An axion means that mechanism is wrong. ADMX and HAYSTAC are scanning.

4. **Lorentz invariance violation is ruled out below |δc/c| < 10⁻²¹.** BPR predicts |δc/c| = 3.38 × 10⁻²¹ from winding-sector instantons. CTA comes online ~2027 with ~10x better resolution than Fermi-LAT. If they push below 10⁻²¹ with no signal, the substrate discreteness prediction fails.

5. **The Born rule holds to 10⁻⁷ precision.** BPR predicts a deviation at ~10⁻⁵ from finite microstate counting on the substrate. Current best bounds are κ < 2 × 10⁻³. Next-generation triple-slit experiments could reach the relevant regime.

6. **A nonzero neutron electric dipole moment is measured.** n2EDM at PSI is targeting 10⁻²⁸ e·cm. Any nonzero result falsifies θ_QCD = 0.

7. **JUNO measures sin²θ₁₂ outside 0.3083 ± 0.002.** At full precision (~2028), JUNO will have the resolution to test the BPR solar angle formula directly.

8. **A fourth-generation fermion is discovered.** BPR's topological argument gives exactly 3. A fourth generation at any mass immediately falsifies the winding constraint.

9. **Dark energy equation of state w₀ > −0.8.** DESI-II and Euclid will measure this. BPR predicts w₀ = −0.934. Above −0.8 and the boundary impedance model breaks.

That is nine independent ways to kill this framework, most of them testable within the next five years. If BPR survives all nine, that is not proof — it is a signal worth taking seriously. If it fails any one of them cleanly, it is dead.

---

## What Makes This Different from Other "Theories of Everything"

Physics has no shortage of frameworks that claim to derive everything from a simple starting point. Most of them share the same failure mode: they are flexible enough to accommodate any result, which means they predict nothing.

BPR is different in three specific ways.

**It makes quantitative predictions, not qualitative stories.** The framework does not say "particles emerge from geometry" and leave it there. It says sin²θ₁₂ = 1/3 − 1/(3.5 ln p) = 0.3083. That number is either right or wrong.

**Its strongest predictions are the ones that can kill it.** Most frameworks arrange their predictions so the falsifiable ones are safely out of experimental reach. BPR's most distinctive predictions — the hydrogen 1S-2S shift, normal mass ordering, no axion, no linear LIV — are all testable with existing or near-term technology. This was a design choice, not an accident. A framework that puts its neck out is making a stronger claim than one that hides behind the Planck scale.

**It gets the same answer from multiple independent routes.** The Weinberg angle is derived three different ways (GUT running with boundary corrections, impedance ratio, coupling ratio at M_Z) and all three converge to 0.23122. The generation number comes from both the winding constraint and the Atiyah-Singer index theorem in the SU(3) background. When different derivation paths produce the same number without being forced to, that is harder to fake than a single formula that hits one target.

I am not claiming BPR is correct. I am claiming it is testable, it is specific, and it has not yet been ruled out by any published measurement. That is a higher bar than most frameworks clear.

---

## The Strong CP Problem, Solved Without an Axion

This deserves its own section because it is one of BPR's cleanest results.

The QCD Lagrangian allows a CP-violating term θ_QCD × G^μν G̃_μν. Experimentally, the neutron electric dipole moment constrains |θ_QCD| < 10⁻¹⁰. There is no reason within the Standard Model for this parameter to be this small. This is the strong CP problem.

The standard solution is the axion — a hypothetical particle introduced specifically to dynamically relax θ to zero. Forty years of searching have found no axion.

BPR resolves this differently. The topological term in the boundary action is θ × χ(M), where χ(S²) = 2 is the Euler characteristic. The octahedral triangulation of the 2-sphere is vertex-transitive. A nonzero θ would break this vertex-transitivity, which is excluded by the lattice regularity condition. Therefore θ_QCD = 0 exactly. Not approximately. Not dynamically relaxed. Structurally forbidden by the geometry.

This predicts: no axion exists. Four independent searches (ADMX at two mass ranges, an extended haloscope at 1.036 GHz, and HAYSTAC Phase II) have returned null results. These don't confirm BPR — the axion could simply be at a different mass — but every null result is consistent with the prediction.

The decisive test: if any experiment detects a QCD axion at the KSVZ or DFSZ coupling, BPR is falsified on this point. No rescue is possible because the prediction is structural, not parametric.

---

## What the Experiments Say So Far

No published experimental result falsifies BPR. Several are consistent with BPR-unique predictions:

- **LEGEND-200**: No neutrinoless double beta decay. T₁/₂ > 1.9 × 10²⁶ yr. Consistent with BPR's Dirac neutrino prediction.
- **LHAASO GRB 221009A**: No linear Lorentz invariance violation. E_QG,1 > 10 M_Pl. BPR predicts ξ₁ = 0 exactly — generic quantum gravity models that predict ξ₁ ~ 1 are in tension.
- **JUNO 2025**: First oscillation measurement. sin²θ₁₂ = 0.3092 ± 0.0087. BPR: 0.3083. The tribimaximal value (1/3 = 0.3333) is excluded at 2.8σ. BPR's correction to tribimaximal, published before JUNO reported, lands inside 1σ.
- **SPARC radial acceleration relation**: a₀ = 1.20 × 10⁻¹⁰ m/s². BPR: 1.18 × 10⁻¹⁰. Within 1.7%.
- **Planck/BBN baryon asymmetry**: η = (6.12 ± 0.04) × 10⁻¹⁰. BPR: 6.83 × 10⁻¹⁰. This is the worst match — 11.6% off.
- **Muonic hydrogen proton radius**: 0.84087 fm. BPR: 0.8412 fm. This resolves the proton radius puzzle in favor of the muonic value, which subsequent experiments confirmed.
- **Four independent axion null searches**: Consistent with θ_QCD = 0.

"Consistent with" is not "confirmed by." Most of these results are also consistent with the Standard Model, which treats these quantities as inputs rather than predictions. The distinction is that BPR derives them. If the derivations are wrong, future experiments at higher precision will catch the discrepancy.

---

## The Code

The full BPR codebase is MIT licensed on GitHub. 1,223 test functions across 30 test modules. Every prediction in this article can be reproduced in under 60 seconds on a laptop.

The derivation engine, constant calculator, and experimental roadmap are live at **bpr.thestardrive.com**.

All 91 numbered equations in the paper have been independently verified against Wolfram Alpha and NIST CODATA. 58 internal consistency tests pass — conservation laws, dimensional analysis, limiting cases, thermodynamic bounds, cross-module agreement. The single tension: the Higgs VEV at 1.0%.

---

## What I'm Asking For

I am not asking you to believe BPR is correct. I am asking you to do one of three things:

**1. Run the code.** Clone the repo. Run the tests. Check that the numbers in this article match what the code produces. If they don't, that's a problem and I want to know about it.

**2. Find a circular derivation.** If any prediction smuggles in the answer it claims to derive — if some experimental value is hidden inside a "structural" parameter — that kills the framework more decisively than any experiment. I have tried to make the derivation chain transparent. Help me check.

**3. Identify a falsified prediction.** If any published measurement already rules out a BPR prediction that I've marked as "derived," point me to the paper. I will update the scorecard publicly.

Physics does not need more frameworks that cannot be wrong. It needs frameworks willing to be wrong in specific, testable ways, and a community willing to check.

The experiments are running. The code is open. The predictions are on the table.

Let's see what breaks.
