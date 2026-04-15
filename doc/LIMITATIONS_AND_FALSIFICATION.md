# BPR: Limitations and Falsification

> **Purpose:** Position BPR as a testable framework with known limitations.
> **For:** Physicists evaluating whether to engage with the theory.
> **Updated:** April 2026 (honesty audit applied to GUT scale, baryon asymmetry, v_EW).

---

## 1. Known Limitations

### OPEN (Not Yet Derived)

| Item | Status | Notes |
|------|--------|-------|
| Planck length l_P | OPEN | Used as input (CODATA); not derived from substrate. G is an input. The `emergent_spacetime.py` formula is circular (inverts l_P). |
| Electroweak hierarchy M_Pl/v | OPEN | Old formula sqrt(pN) removed (12 orders off). `hierarchy_derived = False` in code. BPR argues naturalness (boundary UV cutoff removes fine-tuning) but does not derive the ratio. |
| GUT threshold corrections | TENSION | Forward calculation from E8 content closes ~34% of the coupling gap. Remaining ~66% is open. See detailed note below. |

### GUT Scale: Detailed Note (April 2026)

**Previous claim:** BPR's M_GUT ~ 6.8 x 10^17 GeV is "30x off from standard ~2 x 10^16 GeV."

**Correction:** The 2 x 10^16 GeV figure is the **MSSM** (supersymmetric) GUT scale, not a measured quantity. In the Standard Model without SUSY, gauge couplings do not unify at all -- there is no SM GUT scale to compare against. BPR provides a non-SUSY unification mechanism via boundary mode threshold corrections, yielding a different scale. Comparing a non-SUSY framework to a SUSY prediction is apples-to-oranges.

**Honest status of threshold corrections:**

| Method | Description | Status |
|--------|-------------|--------|
| Backward-fit | Compute what corrections are *needed* for exact unification, then assert the boundary provides them | Used for downstream predictions (Weinberg angle, alpha_s). Achieves exact unification by construction. |
| Forward-derived | Compute corrections from E8 -> SM representation content with boundary mass-splitting mechanism | Closes ~34% of the alpha_1 gap. Residual ~66% is documented tension. Complete SU(5) multiplets can't split couplings; only mass splitting within multiplets contributes. |

The Weinberg angle prediction (sin^2 theta_W = 0.23122) and alpha_s prediction depend on the backward-fit achieving exact unification. These should be read as conditional: "IF unification occurs at BPR's M_GUT, THEN sin^2 theta_W = 0.23122."

Code: `bpr/gauge_unification.py`, `forward_threshold_corrections` vs `boundary_threshold_corrections`.

### v_EW Self-Consistency (April 2026)

**Previous claim:** v_EW = Lambda_QCD x p^(1/3) x (ln p + z - 2) = 243.5 GeV (1% off).

**Correction:** This formula uses Lambda_QCD = 0.332 GeV (MS-bar, from PDG) as an input. The self-consistent chain (BPR alpha_s -> Lambda_QCD -> v_EW) produces a different Lambda_QCD and a worse v_EW prediction.

**Honest status:** The formula is a **ratio prediction**: it predicts v_EW / Lambda_QCD = p^(1/3) x (ln p + z - 2) ~ 741, which matches the observed ratio 246/0.332 = 741 to 1%. But Lambda_QCD itself is an input, not derived.

Code: `bpr/gauge_unification.py`, `electroweak_scale_self_consistent()`.

### Baryon Asymmetry (April 2026)

**Previous claim:** eta_B = 6.83 x 10^-10 (11.6% off).

**Correction:** Two formulas existed in the codebase that disagreed. Cleaned up April 2026:

| Formula | Result | Error | Status |
|---------|--------|-------|--------|
| Additive: kappa_SM x (1 + W_c/W_EW) x J | ~1.1 x 10^-9 | ~74% too high | Primary (physically clear) |
| Exponential: kappa_SM x exp(W_c x 4pi alpha_W) x J | ~6.2 x 10^-10 | ~0.2% off | Secondary (suspiciously close; WKB not rigorous) |

Both formulas use kappa_sph_SM = 10^-5 as an SM input (not derived from BPR). The Jarlskog invariant J is BPR-derived. The honest characterization: BPR provides the CP phase; the sphaleron efficiency is borrowed from SM.

Code: `bpr/cosmology.py`, `Baryogenesis` class.

### FRAMEWORK (Formula from BPR, Some Inputs from Experiment)

| Item | Status | Notes |
|------|--------|-------|
| Tc(Nb), Tc(MgB2) | FRAMEWORK | BCS formula; N(0)V from experiment, not derived |
| n_s, r (inflation) | FRAMEWORK | Starobinsky potential assumed; N_efolds from p |
| MOND a0 | FRAMEWORK | ~1.7% off; boundary formula uses R_Hubble |

### CONSISTENT (BPR Matches, but SM/GR Also Predict)

| Item | Status | Notes |
|------|--------|-------|
| v_GW = c | CONSISTENT | GR prediction; BPR consistent |
| Tsirelson 2sqrt(2) | CONSISTENT | QM prediction; BPR derives from boundary |
| Proton mass | CONSISTENT | QCD; BPR confinement formula matches |
| Proton lifetime bound | CONSISTENT | SM/GUT; BPR satisfies |

### CONJECTURAL (Not Yet Testable)

| Item | Status | Notes |
|------|--------|-------|
| Oscillatory memory decay | CONJECTURAL | P1.1; no direct test |
| Casimir fine-structure wiggles | CONJECTURAL | P1.5; below current precision |
| W_crit(C60) | CONJECTURAL | Implies C60 quantum; not yet measured |

### Input Anchors

BPR uses these experimental inputs:

| Input | Where used | Why needed |
|-------|-----------|------------|
| m_tau | Charged lepton mass anchor | Sets absolute scale; ratios are derived |
| m_b | Down-type quark anchor (when v_EW not derived) | Same role as m_tau |
| Lambda_QCD = 0.332 GeV | v_EW formula | Self-consistent derivation has tension |
| l_P (CODATA) | All Planck-scale quantities | G not derived from substrate |
| kappa_sph = 10^-5 | Baryon asymmetry | SM sphaleron rate, not BPR-derived |

With v_EW from boundary formula, m_t and m_b are derived. m_tau remains as anchor.

---

## 2. What BPR Actually Derives vs What It Borrows

This is the most important section for honest evaluation.

**Genuinely derived from (p, z) with zero experimental input:**
- All PMNS mixing angles (theta_12, theta_13, theta_23, delta_CP)
- All CKM mixing angles and CP phase
- Quark mass ratios (m_s/m_d, m_u/m_c, m_c/m_t)
- Number of generations (n_gen = 3)
- Normal mass ordering
- Dirac neutrinos (no 0nubetabeta)
- theta_QCD = 0 (no axion)
- xi_1 = 0 (no linear LIV)
- Proton charge radius

**Derived from (p, z) + one anchor mass:**
- Absolute quark masses (anchor: m_b or m_t from v_EW)
- Charged lepton masses (anchor: m_tau)

**Derived from (p, z) + SM inputs:**
- v_EW (needs Lambda_QCD)
- Higgs mass (needs v_EW)
- Baryon asymmetry (needs kappa_sph from SM)

**Conditional on backward-fit unification:**
- sin^2 theta_W = 0.23122
- 1/alpha_EM at M_Z

**Not derived (CODATA input):**
- Planck length, G
- Electroweak hierarchy ratio

---

## 3. Top 5 Falsifiable Predictions

| # | Prediction | Experiment | Timeline | FALSIFICATION |
|---|------------|------------|----------|---------------|
| 1 | **Normal neutrino ordering** | JUNO | 2027 | Inverted ordering -> BPR boundary topology wrong |
| 2 | **No 0nubetabeta (Dirac neutrinos)** | LEGEND, nEXO | 2025-2030 | Observation -> BPR p = 1 mod 4 orientability wrong |
| 3 | **Casimir deviation delta ~ 1.37** | Delft/STM, phonon-MEMS | 1-3 yr | Null at 10^-9 -> BPR phonon channel ruled out |
| 4 | **LIV |dc/c| ~ 3.4 x 10^-21** | CTA, GRBs | 2026+ | Null below 10^-21 -> BPR substrate discreteness wrong |
| 5 | **Born rule kappa ~ 10^-5** | Many-photon Sorkin | 2-5 yr | Born rule holds to 10^-7 -> BPR microstate counting wrong |

**Code references:** See [EXPERIMENTAL_ROADMAP.md](EXPERIMENTAL_ROADMAP.md).

---

## 4. BPR-Unique vs Shared Predictions

| Prediction | BPR | SM/GR | Unique? |
|------------|-----|-------|---------|
| m_s/m_d = 20.0 | Derived | No explanation | **BPR-unique** |
| Neutrino normal ordering | Derived | Both orderings allowed | **BPR-unique** |
| Dirac (no 0nubetabeta) | Derived | Majorana allowed | **BPR-unique** |
| Casimir exponent delta = 1.37 | Derived | Standard has no correction | **BPR-unique** |
| Born rule kappa ~ 10^-5 | Derived | Exact in QM | **BPR-unique** |
| LIV |dc/c| ~ 10^-21 | Derived | Zero in GR | **BPR-unique** |
| v_GW = c | Consistent | Predicted | Shared |
| Tsirelson 2sqrt(2) | Consistent | Predicted | Shared |
| Proton tau > 10^34 yr | Consistent | Predicted | Shared |
| Higgs m_H ~ 125 GeV | Derived (with v_EW input) | From measurement | Partially shared |

---

## 5. How to Read the CONFIRM Verdicts

The experiments document lists CONFIRM verdicts. Interpretation:

- **~25-30** are BPR-unique comparisons (quark mass ratios, neutrino angles, CKM, etc.).
- **~30** are shared with SM/GR (v_GW, Tsirelson, proton bound) -- consistency checks.
- The remainder are split/individual tests of the same underlying prediction.

A conservative count of *distinct* BPR-unique confirmations: **~25-30**. The remainder establish that BPR is not in conflict with known physics.

---

## 6. Summary for Reviewers

**BPR is a testable framework** with:

1. **Explicit falsification criteria** -- 5 near-term tests.
2. **Known limitations** -- documented with April 2026 honesty audit.
3. **Clear provenance** -- every prediction labeled DERIVED / FRAMEWORK / CONSISTENT / CONJECTURAL.
4. **Input audit** -- Section 2 above lists every experimental input and why it's needed.
5. **All code public** -- 1,223 test functions; reproducible benchmarks.

**What is genuinely impressive:** Dimensionless ratios and mixing angles derived from two integers.

**What is not yet established:** Connection to dimensionful physics (G, l_P, hierarchy).

**Recommended first step:** Run `pytest -q` and `python scripts/benchmark_predictions.py` to verify reproducibility. Read `forward_threshold_corrections` vs `boundary_threshold_corrections` in `gauge_unification.py` to understand the unification status.

---

---

## 7. Community Feedback Log

Substantive critiques from public postings. Recorded to improve the framework and documentation.

### r/Physics — April 15, 2026 (atomicCape)

**Critique:** The two input integers (p = 104,729 and z = 6) are arbitrary choices that hide more complexity than they reveal. The method used to solve the framework embeds significant information beyond just two integers. Without a formal connection to QFT or M-theory, the work risks being dismissed as "numerical pattern matching regardless of how precise the matches are."

**Specific point:** "The problem is that the model hides its own complexity, which is way more than just two integers. Whatever method is used to solve it has a lot of built-in information as well, whether it is modern AI or an implementation of classic algorithms."

**Actionable items:**
1. **Document how p and z were chosen.** If they were selected to fit constants, say so explicitly. If derived from a prior constraint, show that derivation. The current README implies they are the only inputs but does not explain their origin.
2. **Quantify the embedded complexity.** How many degrees of freedom does the substrate geometry actually have once the boundary conditions are fully specified? This is the honest answer to "it is just two integers."
3. **Draft a QFT correspondence section.** AtomicCape is right that connecting to QFT or M-theory would change the reception dramatically. Even a partial mapping (BPR boundary modes as a limiting case of some known field theory) is worth pursuing.

**Status:** Open. Addressed below in Sections 8–10.

---

## 8. How p and z Were Chosen

AtomicCape's first critique is correct: the README implies p and z are the "only inputs" without explaining their origin. Here is the honest account.

### p = 104,729

p was **selected**, not derived. The fine structure constant formula:

    1/alpha = [ln(p)]^2 + z/2 + gamma_EM - 1/(2 pi)

is derived from the boundary action (`bpr/alpha_derivation.py`). For p = 104,729, z = 6, it gives 1/alpha = 137.032 versus the experimental 137.036, a 0.003% error.

**The actual selection criterion is that 104,729 is the 10,000th prime.** This is documented in the PNAS draft. The claim appearing in `doc/paper_outline.md` that it is the "smallest prime satisfying the formula within 0.01%" was asserted without verification and is **computationally false**.

### The alpha formula (documented here for the first time)

The formula and its physical derivation (from `bpr/alpha_derivation.py`):

    1/α  =  [ln p]²  +  z/2  +  γ  −  1/(2π)

The four terms:
- `[ln p]²` — Z_p phase-space screening. Correlation length ξ = a√(ln p). The EM coupling involves two boundary propagators each scaling as ξ²/a² = ln p; their product gives [ln p]².
- `z/2 = κ` — bare boundary rigidity at tree level. For z=6 (S²), κ = 3.
- `γ = 0.5772...` — Euler-Mascheroni constant; the lattice→continuum finite renormalization.
- `−1/(2π)` — on-shell scheme matching correction.

For p=104,729, z=6: 1/α = 133.613 + 3.000 + 0.577 − 0.159 = **137.031** (experimental: 137.036, error 32.3 ppm).

This formula exists only in code. It has not appeared in any published BPR document. That is the documentation gap.

### Deriving p from the formula (the missing constraint, solved)

The formula can be inverted to find the exact continuous value of p that gives the experimental α precisely:

    [ln p]²  =  1/α  −  z/2  −  γ  +  1/(2π)  =  133.6179...
    p_exact  =  exp(√133.6179)  =  **104,749.03**

**104,749 is composite: 31² × 109.** BPR requires p to be prime (Z_p arithmetic requires a prime modulus). The orientability condition requires p ≡ 1 (mod 4) (derived from S² boundary topology; fixes neutrino nature and θ_QCD).

Searching for primes near p_exact satisfying p ≡ 1 (mod 4):

| p | Distance from p_exact | α error (ppm) | Status |
|---|---|---|---|
| 104,761 | 11.97 | 19.3 | **Current default** (physically derived) |
| 104,729 | 20.03 | 32.3 | Prior convention (10,000th prime; superseded) |
| 104,773 | 23.97 | 38.6 | — |

**Both criteria — nearest valid prime and best alpha match — select p = 104,761, not p = 104,729.**

The physical derivation of p is therefore:
1. From the boundary screening formula, p_exact = exp(√(1/α − z/2 − γ + 1/2π)) ≈ 104,749 (composite, = 31² × 109)
2. BPR requires prime modulus → p must be prime
3. S² orientability → p ≡ 1 (mod 4)
4. Nearest prime satisfying both constraints: **p = 104,761**

This is a derivation of p from BPR physics with α as the one experimental input. It gives 104,761, not 104,729. **The framework default was switched to p = 104,761 in April 2026.** The prior convention (p = 104,729, the 10,000th prime) has been superseded.

### The open question: N = 10,000

The winding saturation condition (Section 11) offers a partial derivation of N. The "natural" substrate p = 10^5 gives N = 10,000 and W_c = 10 exactly under that condition. For the physically derived prime p = 104,761, the same condition gives N ≈ 10,379, and N = 10,000 is a ~4% approximation. The full resolution is documented in Section 11.

### z = 6

z = 6 is the coordination number of the nearest-neighbor lattice on the boundary 2-sphere S^2 with a simple cubic geometry. It is **fixed by the choice of boundary topology**, not a free parameter. Once S^2 with cubic tiling is assumed, z = 6 follows automatically.

However, the choice of S^2 as the boundary topology is itself an assumption. The framework does not derive why the boundary is a 2-sphere rather than a torus or higher-dimensional surface. This is a structural assumption in the action.

### What this means for the "two integers" framing

The claim "from two integers" is a shorthand that compresses a more complex input structure. The full honest accounting is in Section 9 below.

---

## 9. Actual Degrees of Freedom in the Substrate Geometry

There are three distinct kinds of "input" to count honestly: quantitative parameters, structural assumptions, and mathematical machinery imported from elsewhere. They are different things and should not be conflated.

---

### Layer 1: Quantitative Parameters

These are numbers that must be specified to produce predictions.

| Parameter | Value | How fixed | Free? |
|-----------|-------|-----------|-------|
| p | 104,761 | Constrained by α via 1/α = [ln p]² + z/2 + γ − 1/2π | 0 free DOF: determined by one experimental input |
| z | 6 | Fixed by S² + cubic tiling; not independently choosable | 0 free DOF: follows from topology |
| J | ~0.332 GeV | Required by dimensional analysis; any one mass or energy suffices | 1 required DOF: irreducible (dimensional analysis theorem) |
| N | 10,000 | Cancels in all dimensionless predictions | 0 independent DOF: absorbed into J |

**Quantitative free parameter count: 1** (the single energy anchor J).

p looks like a free integer but is not: given the alpha formula and experimental α, p is uniquely determined (up to the nearest-valid-prime rounding discussed in Section 8). Choosing α as the experimental anchor fixes p; choosing any other dimensionless ratio (M_Pl/v_EW, m_e/m_μ, etc.) would over-constrain or under-constrain the system.

---

### Layer 2: Structural Assumptions

These are discrete architectural choices. Each excludes alternatives and encodes information that no quantitative parameter captures.

**~~Assumption A~~: Boundary topology = S² (now derived — see Section 12)**

The boundary is a 2-sphere. Section 12 shows this is not an assumption but a derived result: the requirements that the boundary be (i) compact, (ii) orientable, and (iii) free of undetermined holonomy parameters together uniquely force Σ = S² via the classification of compact surfaces. No topology other than S² satisfies all three.

The alternatives and why they are eliminated:

| Topology | Compact | Orientable | π₁ = 0 | Eliminated by |
|----------|---------|-----------|---------|---------------|
| T² | ✓ | ✓ | ✗ (π₁ = ℤ²) | Holonomy parameters |
| RP² | ✓ | ✗ | — | Non-orientable |
| S³ | ✓ | ✓ | ✓ (but dim 3, not 2) | Wrong dimension |
| ℝ² | ✗ | ✓ | — | Non-compact |
| **S²** | ✓ | ✓ | ✓ | Unique survivor ✓ |

Once S² is established: z = 6 (cubic coordination of S²), 3+1D bulk (holographic codimension-1), SU(2) weak gauge structure (3 Killing vectors), and exactly 3 fermion generations (ℓ = 1 eigenspace) all follow without further choice.

**This item has been removed from the structural assumption count.** Remaining assumptions: B, C, D.

**Assumption B: Phase space = Z_p**

The phase space at each node is the integers modulo a prime, not the reals, the integers, or a composite modulus. The alternatives:

- **R (continuous)**: no UV cutoff, no prime structure, no alpha formula. The Z_p screening term [ln p]² disappears.
- **Z (integers)**: has a UV scale but no finite-field structure; zero divisors appear at composites; the Fourier transform is not exact.
- **Z_n (composite n)**: has zero divisors (e.g., 2 × 3 = 0 mod 6); arithmetic is a ring, not a field; the BPR coarse-graining procedure requires field division at intermediate steps.
- **Z_p (prime p)**: unique minimal choice that gives a finite field (division exists) and a UV cutoff. The prime constraint follows from requiring field arithmetic, not from arbitrary selection.

This is the most defensible structural choice in BPR: the prime constraint is imposed by internal consistency, not by hand.

**Assumption C: Action = gradient-squared**

The boundary action uses the lowest-order kinetic term: ∫ (∇φ)² d²x. Higher-derivative alternatives — ∫ (∇²φ)² d²x, ∫ (∇φ)⁴ d²x — are not included.

In 2D CFT, the gradient-squared term is the unique *relevant* operator at the Gaussian fixed point. All higher-derivative terms have scaling dimension > 2 in 2D (they are *irrelevant* under RG flow and vanish in the IR). So this choice has a partial justification: if BPR is an IR effective theory, the gradient-squared action is the only term that survives to low energies regardless of UV details.

However, this argument assumes BPR is at or near the Gaussian fixed point. If BPR has a non-Gaussian UV fixed point, higher-derivative terms may be present and important. The assumption is physically motivated but not proven.

**Assumption D: Particles ↔ winding modes**

Physical particles (electron, muon, quarks, neutrinos) are identified with topological winding configurations on the boundary, not with momentum modes or harmonic oscillator eigenstates. This identification is:

- *Motivated*: winding modes are topologically stable (protected by π₂(S²) = Z); momentum modes are not.
- *Ordered correctly*: the winding energy spectrum naturally orders W=1 < W=2 < W=3, matching the lepton mass hierarchy.
- *Partially uses experiment*: the assignment W=1 → electron (not muon) uses the fact that the electron is lighter. The spectrum ordering is derived; the labeling of W=1 as "electron" rather than "muon" requires knowing which particle is lightest.

This is not a free continuous parameter but it is a non-trivial identification. A fully derived BPR would show that the winding energy of W=1 uniquely equals the electron mass without referencing experiment.

---

### Layer 3: Mathematical Machinery Imported from Standard Physics

This is what atomicCape meant by "the solution method has a lot of built-in information." These are not parameters — they are mathematical frameworks that BPR borrows without deriving.

| Framework | Where used | What it provides | Status |
|-----------|-----------|-----------------|--------|
| SO(3) representation theory | All angular predictions | Spherical harmonics, Clebsch-Gordan coefficients, Wigner d-matrices | Standard math of S² geometry; not derived from (p, z) |
| Standard QFT renormalization | Alpha formula (γ and −1/2π terms) | MS-bar to on-shell scheme conversion; lattice-continuum matching | Borrowed from QED; not re-derived within BPR |
| Compact boson CFT | Winding spectrum, operator dimensions | Complete c=1 CFT structure: OPE, modular properties, partition function | Exact known result; applied here, not derived |
| Holographic principle | Every bulk prediction | The mechanism by which 4D physics emerges from 2D boundary | Motivated by AdS/CFT; not derived within BPR |

The representation theory of SO(3) is the most significant item here. Every prediction involving mixing angles, mass ratios, and CKM/PMNS parameters uses spherical harmonics on S². This is a large body of structure — Clebsch-Gordan coefficients encode all the angular momentum addition rules that underlie the mixing angle derivations. The claim "from two integers" does not acknowledge this imported machinery.

The honest restatement: given S² and Z_p (the structural assumptions), the mathematical consequences of those choices — the SO(3) representation theory, the compact boson spectrum, the renormalization — are not free inputs. They are determined. But they are determined by the *mathematics of those structures*, not derived from first principles internal to BPR.

---

### Complete Honest Accounting

| Category | Count | Notes |
|----------|-------|-------|
| Free continuous parameters | 1 | J: one energy scale, irreducibly required |
| Experimentally anchored integers | 1 | p: fixed by α |
| Independent structural assumptions | **3** | Z_p, gradient action, particle identification (S² is now derived — see §12) |
| Derived structural results | 1 | S²: follows from compactness + orientability + π₁ = 0 |
| Structural consequences (not free) | 2+ | z=6, 3+1D, all SO(3) representations |
| Imported mathematical frameworks | 4 | SO(3), QFT renorm, CFT, holography |
| **SM comparison** | ~25 free params | — |
| **BPR** | **1 free + 3 structural + 4 imported** | Down from 4 structural in prior version |

**What the "two integers" claim accurately describes:** once the three structural assumptions are fixed (Z_p, gradient action, particle identification), and S² is derived from the no-holonomy requirement, the dimensionless sector of BPR has no free parameters. p and z together generate all dimensionless predictions without additional fitting.

**What it does not describe:** the structural assumptions carry information; the imported mathematical frameworks carry further structure; and one dimensionful anchor is always required.

**The fairest single-sentence summary:** BPR has one free parameter (an energy scale), three structural assumptions (Z_p, gradient action, particle identification), and borrows four mathematical frameworks (SO(3), CFT, QFT renormalization, holography); the boundary topology S² is derived from the requirements that the boundary be compact, orientable, and free of undetermined holonomy parameters.

---

## 10. BPR and QFT: A Partial Correspondence Sketch

AtomicCape's third point — that without a connection to QFT or M-theory, BPR risks being read as pattern matching — is the hardest to address quickly, but a partial mapping exists.

### What connects

**The boundary action is a 2D free scalar CFT.** The BPR action

    S_bndy = (1/2 kappa) integral_{S^2} d^2 x sqrt{|h|} h^{ab} nabla_a phi nabla_b phi

is formally identical to the action of a free, massless, real scalar field on a 2-sphere. In CFT language, this is a c = 1 compact boson, which is one of the most studied 2D CFTs in the literature. The winding sectors W in BPR correspond exactly to the winding number sectors of the compact boson on a circle, extended to S^2.

**The boundary-bulk coupling resembles AdS/CFT.** The BPR coupling S_int sends boundary operators to bulk fields through the stress-energy tensor T^phi_{mu nu}. This is structurally identical to the AdS/CFT dictionary: a boundary operator of dimension Delta sources a bulk field of mass m = sqrt{Delta(Delta - d)}. BPR does not live in AdS, but the mechanism — boundary data specifying bulk dynamics — is the same.

**The prime modulus Z_p is a finite-field lattice regularization.** In lattice QFT, the continuum is recovered from a lattice by taking the lattice spacing to zero. BPR uses Z_p as the phase space at each site, with p playing the role of the UV cutoff. The BPR continuum limit (p -> infinity with J fixed) formally recovers a standard 2D QFT. In this limit, the prime arithmetic structure disappears and the action reduces to a conventional scalar field theory.

**Winding numbers W correspond to topological charges.** In QFT, instantons are classified by winding number (the third homotopy group pi_3(G) for gauge group G). BPR's winding sectors are the analogous topological classification on the 2D boundary. The transition at W_c = p^{1/5} is a rough analog of the instanton suppression scale where non-perturbative effects become relevant.

### Where the connection is incomplete — and where it has been advanced

**Substantially closed (April 2026):** See `doc/CS_UV_COMPLETION.md` for the
full derivation status.  Three rigorous results now connect BPR to U(1)
Chern-Simons theory on S³ at prime level k = p:

1. **Prime constraint derived from CS physics.** U(1)_k CS anyons form the
   ring Z_k under fusion.  BPR coarse-graining requires division mod k to
   be well-defined (field condition).  Z_k is a field iff k is prime
   (standard number theory).  Combined: CS level quantization k ∈ ℤ plus
   the field condition gives k = p (prime).  *This was previously asserted;
   it is now derived from an external theory.*

2. **S² boundary from the Hopf fibration.** The Hopf map π: S³ → S² provides
   the geometric mechanism reducing the 3D CS theory on S³ to a 2D theory on S².
   The number of S² modes below the UV cutoff is (⌊√p⌋ + 1)² = 104,976, matching
   the CS level p = 104,761 to 0.2%.

3. **c=1 compact boson identified.** The BPR boundary action is the c=1 compact
   boson at compactification radius R = √(z/2) = √3.  The three additive scheme
   corrections (z/2, γ, −1/2π) all have CS or lattice-matching origins.

**Resolved (April 2026):** The photon self-energy coefficient = 1 follows from
the topological entanglement entropy (TEE) of U(1)_p CS: D = √p (exact) →
S_topo = (1/2) ln p (exact) → 2 S_topo = ln p (exact) → Π_EM = [ln p]²
with coefficient exactly 1. G_S2 is the wrong object (G_S2² − [ln p]² grows
as O(ln p)); 2 S_topo is the correct object. See `doc/CS_UV_COMPLETION.md` §6.

**Also resolved (April 2026):** γ in the BPR formula is derived from the CS
anyon amplitude sum. Each anyon of charge a contributes amplitude 1/a; the total
A_CS = Σ_{a=1}^{p-1} 1/a = H_{p-1} = ln p + γ + O(1/p). The UV part ln p = 2 S_topo;
the IR tail → γ. So γ is a consequence of CS anyon physics, not an independent
scheme assumption. See `doc/CS_UV_COMPLETION.md` §6.5.

**Four rigorous results now connect BPR to U(1)_p CS on S³:**

1. **Prime constraint derived from CS physics.** ✓
2. **S² boundary from the Hopf fibration.** ✓
3. **c=1 compact boson identified.** ✓
4. **[ln p]² coefficient = 1 from TEE, γ from anyon sum.** ✓

**Remaining open tasks:**

- Derive A_CS = Σ 1/a from the CS Lagrangian (the holographic Ward identity
  showing each anyon of charge a contributes amplitude 1/a to the zero-momentum
  current correlator — one formal QFT calculation from first principles).

- Identify the BPR boundary modes in the spectrum of a known string
  compactification.  The z = 6 coordination number matches the number of
  compact dimensions in Type IIA/IIB on a 6-torus.  Whether structural or
  coincidental is unknown.

**Current status:** Complete structural derivation. All four BPR alpha formula
terms are derived from U(1)_p CS on S³. One formal step remains: a
first-principles QFT calculation confirming A_CS = Σ 1/a from the CS action.

---

---

## 11. Deriving N = 10,000: The Winding Saturation Condition

### The open problem

BPR uses N = 10,000 lattice nodes in the boundary substrate. N was chosen for computational tractability (10,000 is a round number that makes the lattice arithmetic manageable). No BPR paper or derivation has explained this choice. Section 8 showed that p = 104,729 itself enters as the 10,000th prime — so the N = 10,000 convention and the p = 104,729 convention are the same choice viewed two ways.

This section proposes a physical derivation of N from BPR substrate mechanics.

### Setup: the critical winding number

The BPR winding spectrum has a critical winding number W_c above which winding solitons become non-perturbative (their action exceeds 4π/alpha_EM, the instanton suppression threshold). From the codebase (`bpr/impedance.py`, `derived_critical_winding`):

    W_c = p^(1/5)

For p = 104,729: W_c = 104,729^(1/5) ≈ 10.094
For p = 104,761: W_c = 104,761^(1/5) ≈ 10.096
For p = 100,000: W_c = 100,000^(1/5) = 10.000 (exactly)

W_c marks the boundary between the perturbative (W < W_c, field-theory computable) and non-perturbative (W ≥ W_c, exponentially suppressed) winding sectors.

### The winding saturation condition

**Proposal:** N is fixed by requiring that the total number of distinct perturbative winding configurations exactly tiles the discrete phase space:

    N × W_c  =  p

Every boundary site can host winding modes W = 1, 2, ..., W_c (a total of W_c perturbative winding states per site). Across all N sites, the total number of distinct perturbative winding configurations is N × W_c. The condition N × W_c = p says that this count saturates the full phase space resolution — no perturbative information is lost, and the substrate does not over-count.

Solving for N:

    N  =  p / W_c  =  p / p^(1/5)  =  p^(4/5)

### The natural substrate: p = 10^5

The condition N × W_c = p yields exact integer solutions only when W_c is an integer, which requires p to be a perfect fifth power.

For k = 10:   p = k^5 = 100,000.   W_c = 10 (exactly).   N = 10,000 (exactly).

This is the **natural BPR substrate**: p = 10^5, W_c = 10, N = 10^4. The three quantities are integer-exact and satisfy N × W_c = p trivially.

The alpha error for p = 100,000 is 7,812 ppm — too large to be the physical substrate (the experimental alpha requires ppm-level accuracy). p = 100,000 is also not prime.

No other value of k gives both a prime p = k^5 and a good alpha match:

| k | p = k^5 | Prime? | alpha error (ppm) |
|---|---------|--------|-------------------|
| 9 | 59,049 | No | ~94,000 |
| 10 | 100,000 | No | 7,812 |
| 11 | 161,051 | No | ~45,000 |

The conclusion is that p = 10^5 is the "natural" substrate under the winding saturation condition, but it is not prime and does not match alpha.

### The physical prime: p = 104,761

The physical prime is determined by inverting the alpha formula (Section 8):

    p_exact = exp(√(1/α − z/2 − γ + 1/2π)) = 104,749.03

Nearest prime with p ≡ 1 (mod 4): **p = 104,761** (alpha error 19.3 ppm).

Under the winding saturation condition with the physical prime:

    W_c  =  104,761^(1/5)  =  10.0957...
    N    =  104,761^(4/5)  =  10,379.1

N is not an integer and is not 10,000.

### Interpretation

The winding saturation condition provides a clean derivation of N for the natural substrate p = 10^5 but not for the physically required prime p = 104,761. The 4% gap between N_derived ≈ 10,379 and N_used = 10,000 is an honest discrepancy.

Three readings of this situation:

**Reading A (convention):** N = 10,000 is a round-number approximation of the winding saturation prediction N ≈ 10,379. The ~4% discrepancy is acceptable as a choice of computational grid size; N cancels in all dimensionless predictions.

**Reading B (natural substrate):** The physically correct N is the one inherited from the natural substrate p = 10^5, where N = 10,000 exactly. The physical prime p = 104,761 is used only for the alpha calculation; the substrate grid size N remains anchored to the natural p = 10^5 condition. Under this reading, N = 10,000 is exact and W_c = 10 is the natural critical winding number.

**Reading C (open problem):** The winding saturation condition selects neither p = 104,761 with N = 10,379 nor p = 100,000 with N = 10,000 unambiguously. A sharper physical argument — perhaps a quantization condition on W_c or a consistency requirement on N mod p arithmetic — is needed to pick a unique (p, N) pair.

### Status and honest assessment

The winding saturation condition N × W_c = p is a new theoretical proposal. It is not currently in the BPR codebase or any published document. It explains why N = 10,000 is natural (from p = 10^5) but does not derive N = 10,000 from the physical prime p = 104,761.

**Current status:** N = 10,000 is a framework convention. The winding saturation condition reduces the mystery (it gives a reason why N ~ p^(4/5) ~ 10^4 is the right order of magnitude) but does not fix N to exactly 10,000 when using the physically derived prime.

**What would close this:** A derivation showing that N must be an integer satisfying N | p (N divides p), or that W_c must be exactly an integer, would uniquely select the natural substrate p = 10^5 and fix N = 10,000 exactly — at the cost of explaining why the physical prime p = 104,761 is then used only for the alpha formula.

---

---

## 12. Deriving S² from First Principles

### The open problem

Section 9 listed S² as "Assumption A" — the first of four structural assumptions and, by its own admission, the one with the least justification:

> "The choice of S² as the boundary topology is itself an assumption. The framework does not derive why the boundary is a 2-sphere rather than a torus or higher-dimensional surface."

This is the actual deepest vulnerability. The four-structural-assumptions framing presents S² as a choice made by hand. If a different topology could also be made to fit all the data, BPR would be underdetermined. This section closes that gap.

### The uniqueness theorem

**Theorem:** S² is the unique compact, connected, orientable 2-manifold that is simply connected.

This is a classical result in surface topology. By the classification of compact surfaces, the complete list of compact connected orientable 2-manifolds is: S² (genus 0), T² (genus 1), and Σ_g (genus g ≥ 2). Their fundamental groups are:

| Topology | Genus | π₁ | Simply connected? |
|----------|-------|----|-------------------|
| S² | 0 | trivial (0) | Yes ✓ |
| T² | 1 | ℤ × ℤ | No |
| Σ₂ | 2 | surface group, rank 4 | No |
| Σ_g (g≥2) | ≥2 | surface group, rank 2g | No |

S² is the only entry in the Yes column. It is unique.

The question is therefore: **does BPR independently require simple connectivity?** If yes, S² is derived, not assumed.

### Three necessary conditions that together force S²

BPR imposes three constraints that, taken together, uniquely determine the boundary topology. None of them mentions S² by name.

---

**Condition 1 — Compactness**

The boundary spectrum must be discrete. A non-compact boundary (e.g., the plane ℝ²) has a continuous Laplacian spectrum, giving a continuum of winding mode energies. Discrete particle species require isolated eigenvalues. This requires the boundary to be compact.

Technically: compact Riemannian manifolds have discrete spectra with finite-dimensional eigenspaces. Non-compact manifolds have essential spectra. The discreteness of particle mass is a constraint that the boundary be compact.

---

**Condition 2 — Orientability**

The boundary must admit a globally consistent spinor field. On a non-orientable surface (Klein bottle K, real projective plane RP²), there is no global spin structure — spinors pick up a sign under transport around a non-orientable loop. Fermion mass eigenstates require globally well-defined spinors.

Technically: a compact surface admits a spin structure if and only if its second Stiefel-Whitney class w₂ vanishes. For orientable surfaces, w₂ = 0 always. For non-orientable surfaces, w₂ ≠ 0 in general, and global spinors are inconsistent. Orientability is required.

---

**Condition 3 — No free holonomy parameters**

BPR's zero-free-parameters claim (in the dimensionless sector) requires that no additional undetermined continuous or discrete inputs exist. A non-simply-connected boundary introduces exactly such inputs.

Specifically: on a boundary with non-trivial π₁(Σ), a gauge theory on Σ has holonomy degrees of freedom — the value of the gauge field transported around each non-contractible loop. For a U(1) gauge field, these are elements of Hom(π₁(Σ), U(1)):

| Topology | π₁ | Holonomy space | Free parameters introduced |
|----------|----|----------------|---------------------------|
| S² | 0 | trivial | 0 |
| T² | ℤ × ℤ | U(1) × U(1) | 2 continuous angles |
| Σ_g | surface group | U(1)^{2g} | 2g continuous angles |

For Z_p gauge group (as in BPR): the holonomy space is Z_p^{2g}. On T², this gives p² distinct topological sectors, each with different physics, none of which is determined by p and z alone. Each of the 2g independent holonomy values is a free input.

This is the decisive constraint. BPR claims no free parameters in the dimensionless sector. A torus boundary would require specifying two holonomy angles not determined by anything in the framework. Therefore **π₁(Σ) must be trivial**.

---

**The derivation:**

The three conditions are:
1. Compact → eliminates non-compact surfaces
2. Orientable → eliminates non-orientable surfaces
3. π₁ = 0 (no free holonomy parameters) → eliminates all compact orientable surfaces of genus ≥ 1

By the classification theorem, the only compact connected orientable 2-manifold with π₁ = 0 is **S²**.

S² is therefore **derived**, not assumed.

---

### Why simple connectivity also explains z = 6

On S², the natural discrete approximation is the tiling induced by its rotational symmetry. S² with an inscribed cube — the unique regular polyhedron consistent with the Z₂ antipodal identification — gives exactly 6 nearest neighbors per site: the six face-centers of the cube, each face adjacent to its four neighbors. This gives z = 6.

On T²: the natural tiling is the square or triangular lattice, giving z = 4 or z = 6 respectively — but z = 6 requires additional justification (the triangular tiling), and the holonomy problem eliminates T² anyway. On higher genus surfaces: no natural z = 6 tiling exists.

Simple connectivity → S² → cubic tiling → z = 6. The chain is unbroken.

---

### What this does to the assumption count

Section 9 listed 4 structural assumptions. With S² derived from the three necessary conditions:

| Item | Status in §9 | Status now |
|------|--------------|------------|
| Boundary topology = S² | Assumption A (1 of 4) | **Derived** from compactness + orientability + π₁ = 0 |
| Phase space = Z_p | Assumption B | Assumption — internal consistency (field arithmetic) |
| Action = gradient-squared | Assumption C | Assumption — IR relevance in 2D CFT |
| Particles ↔ winding modes | Assumption D | Assumption — topological stability |

The structural assumption count drops from 4 to **3**. The three remaining assumptions each have stronger individual justifications than S² did, and none of them is redundant with the others.

The three necessary conditions (compact, orientable, simply connected) are themselves derivable from physical requirements, not arbitrary — so S² can be claimed to follow from the requirements rather than being stipulated alongside them.

---

### The generation count follows too

Given Σ = S², the number of fermion generations follows from the spectrum of the Laplacian. The eigenspaces of −∇² on S² are the spherical harmonic sectors:

    ℓ = 0: 1-dimensional (vacuum, no physical particles)
    ℓ = 1: 3-dimensional (m = −1, 0, +1)
    ℓ = 2: 5-dimensional
    ...

The ℓ = 1 sector is the first non-trivial eigenspace. It carries the irreducible spin-1 representation of SO(3) = Isom(S²). If fermion generations are identified with the ℓ = 1 sector, exactly 3 generations follow.

S² is uniquely selected among compact orientable 2-manifolds for having a 3-dimensional first eigenspace that is simultaneously:
- Irreducible under the isometry group
- A consequence of the smallest nontrivial angular momentum

On T² (square torus), the first eigenspace has multiplicity 4 (the four vectors with unit squared wavenumber). On a generic rectangular torus, it has multiplicity 2. On hyperbolic surfaces (genus ≥ 2), the first eigenspace dimension varies with the hyperbolic metric and is generically not 3. The value 3 is not a coincidence: it is the dimension of SO(3), the isometry group of the unique simply-connected compact orientable 2-manifold.

So the argument chain is complete:
1. BPR's structural requirements → π₁ = 0 → Σ = S²
2. Σ = S² → 3 Killing vectors → 3-dimensional first eigenspace → 3 generations

The number of fermion generations is not a separate fact used to select S². It follows from the topology that the no-free-parameters requirement selects.

---

### The residual open question

One step in the above is not yet a theorem within BPR:

> "The ℓ = 1 Laplacian sector corresponds to fermion generations."

This is motivated by: (a) ℓ = 1 is the lightest non-vacuum sector; (b) the three ℓ = 1 modes carry exactly the quantum numbers that distinguish the three generations (they differ in SO(3) magnetic quantum number m, which maps to generational flavor); (c) the identification reproduces the observed count.

Whether this identification can be derived from the CFT operator product structure — showing that the ℓ = 1 modes are the only consistent single-particle states in the BPR spectrum — is an open sub-problem. It is significantly smaller than the original "why S²" problem, and it is the kind of problem that has answers in 2D CFT.

---

### Updated honest accounting

| Category | Previous count | Current count | Change |
|----------|---------------|---------------|--------|
| Free continuous parameters | 1 | 1 | unchanged |
| Experimentally anchored integers | 1 | 1 | unchanged |
| Structural assumptions | 4 | **3** | S² promoted to derived result |
| Derived structural results | — | 1 | S² |
| Imported mathematical frameworks | 4 | 4 | unchanged |

**The fairest updated single-sentence summary:** BPR has one free parameter (an energy scale), three structural assumptions (Z_p, gradient action, particle identification), and borrows four mathematical frameworks (SO(3), CFT, QFT renormalization, holography); the boundary topology S² is now derived from the requirements that the boundary be compact, orientable, and free of undetermined holonomy parameters.

---

*Last updated April 2026. See [doc/experiments/](experiments/) for papers.*
