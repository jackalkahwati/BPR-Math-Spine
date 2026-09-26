# What BPR-6D predicts

2026-09-26. Status: consequences of the derived structure, stated
conditionally. The implementation is `bpr/predictions.py`, with tests in
`tests/test_predictions.py` and a demo in `scripts/demo_predictions.py`. No
prediction has been tested against data here, and **no empirical validation
is claimed**.

## 0. Question

A theory of everything must say something checkable. After ten rounds of
work, what does BPR-6D actually predict, and what could falsify it?

## 1. The Green–Schwarz QCD axion (testable in principle)

The non-chiral Green–Schwarz 2-form gives two 4D axions:
- b = ∫_{S²}B, whose shift is gauged by A_F through the Chern–Simons term of
  Y_e = 6x², with charge k_b = 12 per flux unit;
- a, the dual of B_μν, gauged by A_F through the BF term from the 9x² part of
  Y_g, with charge k_a = 18.

The Stückelberg-massive U(1)_F eats the combination along (k_b f_b, k_a f_a).
The orthogonal combination is a physical axion. Only b couples to the Spin(10)
instanton density, with coefficient 3 from Y_g = 3λ_V + 9x² − λ_T. So the
physical axion couples to QCD whenever k_a ≠ 0, whatever the ratio of decay
constants (tested for f_a/f_b from 10⁻³ to 10³). Setting k_a = 0 would
decouple it, which confirms that the coupling comes from the BF term.

**Prediction.** BPR-6D contains a QCD axion that removes the strong-CP θ
parameter. Its decay constant is set by the compactification scale.

**Order of magnitude.** Take 1/r = 2g₄M_Pl/3 with g₄ ∈ [0.02, 0.4] (flux note,
section 4), and allow f_a ∈ [(1/r)/8π², 1/r]:
- f_a ≈ 4×10¹⁴ – 6×10¹⁷ GeV;
- m_a ≈ 5.7 μeV × (10¹²/f_a) ≈ 10⁻¹¹ – 10⁻⁸ eV.

This is the band targeted by nuclear-spin and lumped-element searches such as
CASPEr and DMRadio-type experiments.

**Cosmology.** With standard misalignment, the observed dark-matter density
needs an initial angle θ_i ≈ 4×10⁻⁴ – 3×10⁻². A generic θ_i ~ 1 overproduces
dark matter unless the axion field is diluted or the angle is selected.

**Honest scope.**
- This prediction is generic to 6D Green–Schwarz and string-derived models
  (the "model-independent axion"). It is not unique to BPR-6D.
- The decay constant is not computed precisely, because the 2-form's 6D
  coupling is not fixed.
- Observing a QCD axion at f_a ≪ 10¹⁴ GeV would not strictly falsify BPR-6D,
  since other axions could exist. Finding **no** QCD axion anywhere in the
  band, with strong CP solved some other way, would count against it.

## 2. The family number (a postdiction with a sharp exclusion)

For the minimal non-supersymmetric completion (rounds 4 and 6):
- n_gen = q|m| with 3 | q, so n_gen ∈ {3, 6, 9, …};
- the observed three families are consistent;
- a fourth chiral family, or a fifth, is excluded;
- the next allowed value is six.

This is falsifiable in principle by a fourth sequential chiral family, which
is already strongly disfavoured by Higgs and electroweak data. The statement
is lost in supersymmetric embeddings (round 6).

## 3. Things BPR-6D implies but that are not observable

Each item below sits at the compactification scale 1/r ≈ 10¹⁶–10¹⁸ GeV, or
couples only through the massive U(1)_F:
- Kaluza–Klein modes and the radion, at mass ~1/r;
- states with one third of the matter's U(1)_F charge (round 4);
- vector-like exotics from any Spin(10) background (round 7, Lemma 1);
- the SU(2) family gauge bosons, which must be broken, for example by branes
  (round 8).

## 4. What BPR-6D does not predict

- Fermion masses and mixings (round 8).
- The Higgs mass (tuned; round 8).
- The Spin(10)-breaking scale, and hence proton decay (round 7).
- The cosmological constant (tuned; round 9).
- The value of g₄, and hence the compactification scale (flux note).

## 5. Limitations

- The axion decay constant is an order-of-magnitude band.
- Standard formulas for the QCD axion mass (Grilli di Cortona et al. 2016)
  and for misalignment carry O(1) uncertainties.
- The family-number statement holds for the minimal non-supersymmetric
  completion only.
- No comparison with data beyond the family count is made.
