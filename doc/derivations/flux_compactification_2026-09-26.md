# The BPR-6D flux vacuum: M4 × S² with U(1)_F monopole flux

2026-09-26. Status: exact classical symbolic algebra on a supplied
six-dimensional action. It comes with an implementation
(`bpr/six_dim_flux_vacuum.py`), tests (`tests/test_six_dim_flux_vacuum.py`)
and a demo (`scripts/demo_six_dim_flux_vacuum.py`). The background and the
breathing mode are derived here. Stability against the other fluctuations is
cited from the literature (section 7), not re-derived. Quantum corrections are
not included. The architecture this belongs to is described in
[architecture_decision_2026-09-26.md](architecture_decision_2026-09-26.md).

## 0. Question

BPR-6D (see the decision note) takes six-dimensional Lorentz invariance and
gravity as fundamental, and puts U(1)_F flux m=3 on an internal sphere. The
[chiral completion](chiral_parent_completion_2026-09-25.md) needs that flux
to produce three chiral 16s. Questions:
- Is M4 × S² with that flux a solution?
- Is its size stabilized?
- What four-dimensional scales does it imply?
- Does anything select m=3?

## 1. Action and ansatz

    S = ∫ d⁶x √(−G) [ (M⁴/2) R₆ − Λ − ¼ F_MN F^MN ] + (Spin(10), fermions, Green–Schwarz sector),

where D = ∂ − i e Q A, and Q=1 for the parent 16. The ansatz is:
- flat M4 times a round S² of radius r;
- F_θφ = B r² sin θ, i.e. an orthonormal-frame field B, so F² = 2B².

The Spin(10) field, the fermions and the 2-form vanish in the background. Dirac
quantization for unit charge, e·B·4πr² = 2πm, gives

    B = m / (2 e r²).

## 2. Einstein equations and the solution

**Theorem 1 (vacuum).** The off-diagonal Einstein equations vanish identically.
The 4D block is isotropic, and so is the sphere block. The remaining
equations are

    4D block:     Λ + B²/2 − M⁴/r² = 0,
    sphere block: Λ − B²/2 = 0.

Their solution with quantized flux is

    B = M²/r,   Λ = B²/2 = M⁴/(2r²),   r = m/(2 M² e),   Λ = 2 M⁸ e² / m².

*Proof.* R₆ = 2/r². G_μν = −(1/r²) g_μν on M4 and G_ab = 0 on S². For the
monopole, T_μν = −(B²/2) g_μν and T_ab = +(B²/2) g_ab. `_ricci` computes
curvature by brute force from the metric. The test compares it with these
hand formulas and with Dirac quantization. ∎

This is the Randjbar-Daemi–Salam–Strathdee (RSS) vacuum (1983). A flat 4D
factor requires Λ to take the value above, which is one tuning of the 6D
cosmological constant.

## 3. Breathing mode

Write r = r0 e^ψ(x), and Weyl-rescale the 4D metric by e^{−2ψ} so that
M_Pl² = 4π r0² M⁴ stays fixed (Einstein frame).

**Lemma 2 (potential).** For fixed m and generic Λ,

    V(r) = 4π r0⁴ [ −M⁴/r⁴ + Λ/r² + m²/(8 e² r⁶) ].

The three terms are, in order: internal curvature, the 6D vacuum energy, and
the flux energy. `reduced_potential` integrates the 6D Lagrangian over S² for
the warped product. A test checks it against this hand-reduced form.

**Lemma 3 (kinetic term).** Reducing √(−G)(M⁴/2)R₆ with ψ=ψ(t) gives

    L_kin = −½ K (∂ψ)²,   K = 4 M_Pl².

This is after integrating the ψ'' terms by parts. It agrees with the standard
breathing-mode normalization n(n+2)/2 · M_Pl² for n=2 internal dimensions.

**Theorem 4 (stability).** At the Theorem 1 vacuum:
- V = V′ = 0 and V″(ψ) = 16π M⁴ > 0;
- the radion mass is m_ψ² = V″/K = 1/r0², exactly.

The breathing mode is therefore stable, and it is as heavy as the first
Kaluza–Klein level. It leaves no light modulus and no long-range scalar force.
For m = 1, 2, 3, 5, the tests confirm by grid minimization that:
- the minimum sits at r = m/(2M²e) with V = 0;
- detuning Λ by ±1% makes the minimum de Sitter or anti-de Sitter.

## 4. Four-dimensional scales and classical control

Reducing on S² gives

    M_Pl² = 4π r² M⁴ = π m² / e²,     g4² = e² / (4π r²),

and therefore

    1/r = 2 g4 M_Pl / m,     (1/r)/M = 2 π^{1/4} (g4/m)^{1/2}.

Here g4 is the 4D U(1)_F coupling. Classical 6D gravity is controlled only if
the sphere is larger than the 6D Planck length, 1/r < M. This requires

    g4 < m / (4√π)   (≈ 0.42 for m=3).

In the table, the coupling g4 is an assumption; only the relations above are
derived:

| g4 (assumed) | 1/r (GeV) | M6 (GeV) | (1/r)/M6 |
|---|---|---|---|
| 0.1 | 1.6e17 | 3.3e17 | 0.49 |
| 0.5 | 8.1e17 | 7.5e17 | 1.09 (uncontrolled) |
| 1.0 | 1.6e18 | 1.1e18 | 1.54 (uncontrolled) |

**Consequence.** The internal sphere is within one or two orders of magnitude
of the Planck length. BPR-6D therefore predicts no laboratory-scale boundary
effects. The KK tower, the radion and the SU(2) isometry bosons all sit near
10¹⁷ GeV unless g4 is very small.

## 5. Green–Schwarz sector in the background

The completion factor X is a 2-form with legs only on S². Hence X∧X=0. The
background has S2=0 and p1=0 (flat M4 times a round S²). The Chern–Simons
3-form of the background has legs only on S², so it also vanishes. The 2-form
is therefore unsourced, and there is no tadpole.

Fluctuations are a different matter. The Green–Schwarz coupling B∧X4
together with the background flux gives the 4D U(1)_F gauge boson a
Stückelberg mass. This is the effect recorded in the chiral completion note.
Its value, and whether any axion stays light, are not computed here.

## 6. Does anything select m=3? No.

**Theorem 5 (flux landscape at fixed Λ).** Fix M, e and Λ, and consider all
flux sectors m. Write u=r². Stationary points solve

    2Λ u² − 4M⁴ u + (3/4) m²/e² = 0.

They exist only for m² ≤ (8/3) M⁸e²/Λ. The smaller root is a minimum, and the
larger root is a barrier to decompactification.

Tune Λ to make m₀ flat, Λ = 2M⁸e²/m₀². Then:
- m < m₀ gives anti-de Sitter minima;
- m₀ < m ≤ (2/√3) m₀ gives de Sitter minima;
- larger m has no vacuum.

For m₀ = 3:
- m = 1, 2 are AdS;
- m = 3 is Minkowski;
- m ≥ 4 has no vacuum (4² > 12).

`flux_landscape` computes this. Tests compare it with direct grid
minimization for m₀ = 3 and m₀ = 7.

So "three families" is not selected by the vacuum. Choosing the flux is the
same act as tuning the 6D cosmological constant: whichever flux is flat is
flat because Λ was chosen for it. This answers critical-path item 6 of the
[unification map](unification_map_2026-09-25.md) negatively at the classical
level.

Tunnelling between flux sectors needs magnetically charged branes, which are
absent from BPR-6D as defined. Blanco-Pillado, Schwartz-Perlov and Vilenkin
(2009) and Carroll, Johnson and Randall (2009) study this Einstein–Maxwell
landscape and its transitions.

## 7. Cited, not derived

- Full classical stability of the RSS vacuum against scalar, vector and
  tensor fluctuations was established in the RSS literature (Randjbar-Daemi,
  Salam, Strathdee, Nucl. Phys. B214 (1983) 491). Only the breathing mode is
  re-derived here.
- The massless 4D gauge symmetry includes the SU(2) isometry of the round
  sphere (RSS 1983).
- Salam–Sezgin (Phys. Lett. B147 (1984) 47) is the supersymmetric version with
  gauged U(1)_R. There, flat M4 × S² requires no tuning of Λ. That is a
  candidate next step for removing the tuning. It would change the anomaly
  bookkeeping (the gravitino and gaugini carry R-charge) and would fix the flux
  by supersymmetry, so the family count would have to be redone.

## 8. Independent review

Pending. See the README entry for this date.

## 9. Limitations

- Six-dimensional gravity, U(1)_F and the field content are supplied, not
  derived.
- A flat M4 needs one tuning of Λ, and m=3 is equivalent to that tuning
  (Theorem 5).
- Only the breathing mode is derived; other fluctuations are cited.
- Quantum corrections are not included: the one-loop Casimir energy of all
  fields on the Planck-sized sphere, the Green–Schwarz axion and the
  Stückelberg mass. They can shift the radion potential at the same order as
  its classical terms, because 1/r is close to M.
- The numerical scales depend on an unknown coupling g4 and are illustrative.
- 6D gravity and 6D gauge theory are non-renormalizable. This is an effective
  theory with cutoff of order M; it is not UV complete.
