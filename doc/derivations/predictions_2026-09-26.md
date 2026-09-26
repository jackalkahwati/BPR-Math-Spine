# What BPR-6D predicts

2026-09-26. Status: consequences of the derived structure, stated
conditionally. The implementation is `bpr/predictions.py`, with tests in
`tests/test_predictions.py` and a demo in `scripts/demo_predictions.py`. No
prediction has been tested against data here, and **no empirical validation
is claimed**.

An independent review (section 6) found a blocker in the first version: it
left out the U(1)_F charge of the Higgs. Its repairs are applied. The axion
statement is now conditional on the supplied Higgs sector.

## 0. Question

A theory of everything must say something checkable. After ten rounds of
work, what does BPR-6D actually predict, and what could falsify it?

## 1. The Green–Schwarz axion and U(1)_F as a Peccei–Quinn symmetry

**The two axions.** The non-chiral Green–Schwarz 2-form gives two 4D axions.
The shift charges below are parsed from the round-4 factorization
I8 = Y_e·Y_g, with Y_e = 6x² and Y_g = 3λ_V + 9x² − λ_T (tested):
- b = ∫_{S²}B shifts through the Chern–Simons term of Y_e, with charge
  k_b = 12 per flux unit. It is the only field multiplying the Spin(10)
  instanton density, with coefficient 3.
- a, the dual of B_μν, shifts through the BF term from the 9x² part of Y_g,
  with charge k_a = 18.

b is the analogue of the heterotic *model-dependent* axion. a is the analogue
of the model-independent one, and here it does not couple to QCD.

**Anomaly check.** The U(1)_F–SU(3)² anomaly of three 16s of F-charge 3 is
Σ qT = 3 · 3 · 2 = 18. The coefficient of b needed to cancel it is
2·18/k_b = 3, which matches the Green–Schwarz coefficient of λ_V (tested).

**The Higgs is Peccei–Quinn charged.** Any Higgs that gives the families a
Yukawa has F-charge −6 (round 8). The 10_H must then be complex, and both
Higgs doublets H_u and H_d carry F = −6. The conjugate coupling 16·16·10_H*
and the term H_uH_d are forbidden. Perturbatively, U(1)_F therefore acts on
the Higgs doublets exactly as a Peccei–Quinn symmetry. This is the structure
familiar from renormalizable SO(10) models with a complex 10_H (for example
Babu–Mohapatra 1993; Bajc–Melfo–Senjanović–Vissani 2006; cited from
memory).

**The gauge-invariant θ̄** is

    θ̄ = θ₀ + 3b + 3 arg H_u + 3 arg H_d + arg det Y.

It is invariant under U(1)_F (3·12 − 3·6 − 3·6 = 0) and under hypercharge
(tested). **a does not appear.** The phases b, a, arg H_u and arg H_d have
two gauge-invariant combinations (tested as the nullspace of the charge
matrix):
- θ̄, which couples to QCD;
- ζ = 3b − 2a, a Goldstone boson with no QCD coupling. It stays massless
  unless something else lifts it.

Since θ̄ is gauge invariant, its QCD decay constant is simply
1/f² = Σ c_i²/f_i², summed over every F-charged phase.

**Three cases for the supplied Higgs sector.**
1. **Only the Higgs doublets break U(1)_F besides b.** Then
   f ≈ v sin2β/6 ≤ 41 GeV. This is a visible axion of the
   Peccei–Quinn–Weinberg–Wilczek (PQWW) type, excluded long ago. BPR-6D with
   only a 10_H Higgs is therefore not viable.
2. **The H_uH_d term comes from an instanton factor e^{ib}.** This term is
   gauge invariant, and it would supply μ and Bμ. But its potential depends
   on b + arg(H_uH_d) = θ̄/3, so it pins θ̄ at the instanton's phase.
   Strong CP is then unsolved.
3. **A supplied F-charged Standard-Model singlet S with a high vev and a
   link to H_uH_d.** An example is a Kim–Nilles coupling S²H_uH_d/M_* with
   F_S = +6, which gives μ ~ f_S²/M_* and c_S = 6. The doublet phases then
   lock to S, and
   f ≈ (9/f_b² + c_S²/f_S²)^{−1/2} ≈ min(f_b/3, f_S/c_S).
   This is an invisible QCD axion of the DFSZ type (tested for both limits).

**Prediction, stated conditionally.** In BPR-6D, U(1)_F is automatically a
Peccei–Quinn symmetry of any Higgs sector that gives Yukawas. A viable
solution of strong CP needs case 3, which requires an extra supplied field.
In that case the QCD axion has f ≲ f_b/3, and f is lower if S is lighter.

**Order of magnitude of f_b/3.** Take 1/r = 2g₄M_Pl/3 with
g₄ ∈ [0.02, 0.4] (flux note, section 4), and allow
f ∈ [(1/r)/8π², 1/r]. Then:
- f ≈ 4×10¹⁴ – 6×10¹⁷ GeV;
- m_a ≈ 10⁻¹¹ – 10⁻⁸ eV, i.e. frequencies of 2 kHz – 3.4 MHz.

The band is a guess, not a derivation. Reducing (κ/2)H² with an integrally
normalized B gives f_b ∝ √κ/r and f_a ∝ 1/(√κ r), and κ is not fixed. The
band assumes κ ~ O(1). Its top end, g₄ = 0.4, has 1/r near M₆, which is
outside classical control. The heterotic value α_G M_Pl/(2π√2) ≈ 10¹⁶ GeV
lies inside it.

**Axion quality (unresolved).** The axion solves strong CP only if every
non-QCD, gauge-invariant potential along θ̄ is below about 10⁻¹⁰ χ_QCD.
That needs an instanton action S ≳ 190 at M ~ 10¹⁷ GeV (tested). Candidate
sources, none computed here:
- Euclidean strings wrapped on S², dressed by charged fields so that they are
  gauge invariant. They have S ≈ 4πT r² ≈ 10–100 for rM ~ 1–3 and T ~ M².
  The undressed potential cos(3b − 2a + δ) acts on ζ, not on θ̄.
- Small Spin(10) instantons, which are aligned with QCD only up to CP phases
  of the heavy sector.
- Gravitational instantons, through b's λ_T coupling.

**Constraints and tests.**
- **Superradiance.** Black-hole superradiance disfavours
  6×10⁻¹³ < m_a < 2×10⁻¹¹ eV, i.e. f ≳ 2.9×10¹⁷ GeV
  (Arvanitaki–Baryakhtar–Huang 2015). This cuts off the top of the band,
  whether or not the axion is the dark matter.
- **Dark matter.** In the pre-inflationary case (forced for b, which has no
  Peccei–Quinn phase transition), standard misalignment needs
  θ_i ≈ 3×10⁻⁴ – 3×10⁻² across the band. Above f ≈ 1.5×10¹⁷ GeV the
  oscillation starts after the QCD crossover, and the exponent 3/2 replaces
  7/6.
- **Isocurvature.** If this axion is all of the dark matter, Planck's bound
  needs H_I ≲ 3×10⁸ – 6×10⁹ GeV (r ≲ 10⁻¹² – 10⁻⁹) across the band. A
  detection of primordial B-modes would make it a negligible fraction of the
  dark matter.
- **Direct searches.** The photon coupling has E/N = 8/3, as for any
  Spin(10)-complete coupling. Only proposed GUT-scale lumped-element and
  nuclear-spin searches (projected sensitivities) reach this frequency
  range, and only if the axion is the dark matter. DMRadio-m³'s QCD reach
  lies above about 30 MHz.
- **Flux-breaking variant.** If Spin(10) were broken by flux, a would also
  eat A_h and no Green–Schwarz axion would survive. Round 7 shows those
  fluxes are unstable anyway.

**Honest scope.** A Green–Schwarz axion is generic to 6D Green–Schwarz and
string models; it is not unique to BPR-6D. What is specific here is that
U(1)_F makes the Higgs sector Peccei–Quinn charged, which **excludes the
minimal Higgs sector** and forces either an extra singlet or an unsolved
strong-CP problem. No experiment can falsify the axion statement in a
model-independent way.

## 2. The family number (a postdiction with a sharp exclusion)

For the minimal non-supersymmetric completion in class 𝒞, the class with no
extra massless Spin(10) matter (rounds 4 and 6):
- n_gen = q|m| with 3 | q, so n_gen ∈ {3, 6, 9, …};
- the observed three families are consistent;
- a fourth or fifth chiral family is excluded;
- the next allowed value is six.

A fourth sequential chiral family is already excluded by Higgs and
electroweak data, so this is a postdiction. The statement fails outside 𝒞:
16₊(1) ⊕ 16₋(2) evades it (GS note, section 3). It is also lost in
supersymmetric embeddings (round 6).

## 3. Things BPR-6D implies but that are not observable

Each item below sits at the compactification scale 1/r ≈ 10¹⁶–10¹⁸ GeV, or
couples only through the massive U(1)_F:
- Kaluza–Klein modes and the radion, at mass ~1/r;
- states with one third of the matter's U(1)_F charge (round 4);
- vector-like exotics from any Spin(10) background (round 7, Lemma 1);
- the SU(2) family gauge bosons, which must be broken, for example by branes
  (round 8);
- the Goldstone boson ζ, unless it is lifted.

## 4. What BPR-6D does not predict

- Fermion masses and mixings (round 8).
- The Higgs mass (tuned; round 8).
- The Spin(10)-breaking scale, and hence proton decay (round 7).
- The cosmological constant (tuned; round 9).
- The value of g₄, and hence the compactification scale (flux note).
- The axion decay constant, beyond the upper band.

## 5. Limitations

- The Higgs sector, including any singlet S, is supplied.
- The decay-constant band is order of magnitude, and depends on the unfixed
  2-form coupling κ.
- Axion quality is not computed.
- Standard formulas for the QCD axion mass (Grilli di Cortona et al. 2016),
  misalignment, isocurvature and superradiance carry O(1) uncertainties.
- The family-number statement holds for the minimal non-supersymmetric
  completion in class 𝒞 only.
- No comparison with data beyond the family count is made.

## 6. Independent review

An independent review, with its own scripts, verified:
- the reduction of I8 on S², giving k_b = 12 and k_a = 18, and the roles of
  b and a;
- that the reduced 4D anomaly matches three 16s of charge 3;
- that there is no tree-level kinetic mixing;
- the mass relation and the misalignment normalization.

Its findings, all repaired:
- **Blocker.** The F-charged Higgs was omitted. The review found that θ̄
  contains the Higgs phases and not a, that the minimal Higgs gives an
  excluded PQWW axion, and that the earlier "k_a = 0 decouples the axion"
  was an artifact. This is now section 1's case analysis, with tests of
  gauge invariance and of the decay constants.
- **Major:**
  - Axion quality is stated as unresolved.
  - The model-dependent/model-independent labels are corrected.
  - The band's dependence on κ, and the classical-control caveat, are stated.
  - Superradiance and isocurvature checks are added.
- **Minor:**
  - The experiments are narrowed to projected, dark-matter-conditional
    reach, with E/N = 8/3.
  - "Pre-inflationary" misalignment, with the late-oscillation exponent.
  - Charges are now parsed from the factorization, not hard-coded.
  - Notation: f now always means the QCD decay constant.
  - Class 𝒞 is named in the family statement.
