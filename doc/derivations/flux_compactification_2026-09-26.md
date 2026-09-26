# The BPR-6D flux vacuum: M4 × S² with U(1)_F monopole flux

2026-09-26. Status: exact classical symbolic algebra on a supplied
six-dimensional action. It comes with an implementation
(`bpr/six_dim_flux_vacuum.py`), tests (`tests/test_six_dim_flux_vacuum.py`)
and a demo (`scripts/demo_six_dim_flux_vacuum.py`). The background, the breathing mode and the
4D normalizations are derived here. Stability against the other
Einstein–Maxwell fluctuations is cited from the literature (section 7), not
re-derived. Spin(10), fermion and 2-form fluctuations are not covered by that
citation. Quantum corrections are not included. The independent review in
section 8 found one blocker, three major and seven minor issues; its repairs
are applied. The architecture this belongs to is described in
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

The breathing mode is therefore stable, with its mass at the Kaluza–Klein
scale: the first nonconstant scalar harmonic (l=1) sits at m² = 2/r², a
factor √2 higher. The breathing mode itself is not a light modulus and gives
no long-range scalar force. Whether a Green–Schwarz axion stays light is a
separate question (section 5).
For m = 1, 2, 3, 5, the tests confirm by grid minimization that:
- the minimum sits at r = m/(2M²e) with V = 0;
- detuning Λ by ±1% makes the minimum de Sitter or anti-de Sitter.

## 4. Four-dimensional scales and classical control

Reducing on S² gives

    M_Pl² = 4π r² M⁴ = π m² / e²,     g4² = e² / (4π r²).

Both normalizations are computed, not asserted:
- `planck_normalization` reduces (M⁴/2)√(−G)R₆ for a curved 4D factor
  a(t)²η. It checks R₆ = R₄ + 2/r² and integrates over S², giving Z_g = 4πr².
- `gauge_normalization` reduces −¼√(−G)F² for a 4D field F_tx = E on top of
  the monopole, with a 4D warp Ω. The result is Z = 4πr², independent of Ω.

Together these give

and therefore

    1/r = 2 g4 M_Pl / m,     (1/r)/M = 2 π^{1/4} (g4/m)^{1/2}.

Here g4 is the 4D U(1)_F coupling. The 4D U(1)_F boson is Stückelberg-massive
(section 5), but its coupling is still well-defined.

The relation g4 = m/(2rM_Pl) says that g4 is the unknown radius under another
name. Classical control of 6D gravity needs rM ≫ 1. Where the O(1) threshold
sits depends on convention:

    1/r < M            ⇒  g4 < m/(4√π)  ≈ 0.42 for m=3,
    r > ℓ6 = G6^{1/4}  ⇒  g4 < m/√2     ≈ 2.1  for m=3,

with G6 = 1/(8πM⁴). The two bounds differ by a factor √(8π), and both are
necessary O(1) conditions only. Real control needs g4 well below them.

In the table, g4 is an assumption, and only the relations above are derived.
The ratios near 1 mark the edge of the classical regime, not a sharp
boundary:

| g4 (assumed) | 1/r (GeV) | M6 (GeV) | (1/r)/M6 |
|---|---|---|---|
| 0.1 | 1.6e17 | 3.3e17 | 0.49 |
| 0.5 | 8.1e17 | 7.5e17 | 1.09 |
| 1.0 | 1.6e18 | 1.1e18 | 1.54 |

**Consequence, conditional on g4.** Since r = (m/2g4)ℓ_P, with ℓ_P the
reduced Planck length:
- If g4 is of order 0.02–0.4, the sphere lies within one or two orders of
  magnitude of the Planck length. The KK tower and the radion then sit near
  10¹⁶–10¹⁸ GeV, and BPR-6D gives no laboratory-scale boundary effect.
- A large sphere needs a tiny and unexplained g4. For example, 1/r ≈ 1 TeV
  needs g4 ≈ 10⁻¹⁵. Nothing derived here forbids that. Collider bounds on KK
  excitations of the bulk Spin(10) fields would constrain it.

The SU(2) isometry gauge bosons are **massless at tree level** in this
vacuum, whatever the radius. They are the gauged family symmetry of the
[family note](family_symmetry_from_flux_2026-09-26.md), section 6, and no
breaking mechanism is supplied.

## 5. Green–Schwarz sector in the background

The completion factor X is proportional to the U(1)_F field strength, which
is a 2-form with legs only on S², so X∧X=0. The background has no Spin(10)
field, so S2=0. For flat M4 times a round S², p1=0. The Chern–Simons 3-form
A∧F of the monopole potential also vanishes. The 2-form is therefore
unsourced, and there is no tadpole.

`green_schwarz_background` computes all of these from the fields:
- F∧F from the monopole field;
- tr R∧R from the curvature 2-forms of the product metric;
- A∧F from the potential A_φ = Br²(1−cos θ).

Two controls:
- the same wedge routine finds a nonzero F∧F once a 4D electric field is
  added;
- the curvature 2-forms reproduce the round-sphere Riemann tensor.

Fluctuations are a different matter. The Green–Schwarz coupling B∧X4
together with the background flux gives the 4D U(1)_F gauge boson a
Stückelberg mass. This is the effect recorded in the chiral completion note.
Its value, and whether any axion stays light, are not computed here.

## 6. Does anything select m=3? No.

**Theorem 5 (flux landscape at fixed Λ>0).** Fix M, e and Λ>0, and consider
all flux sectors m. Write u=r². Stationary points solve

    2Λ u² − 4M⁴ u + (3/4) m²/e² = 0.

Minima exist only for m² < (8/3) M⁸e²/Λ. At equality there is an inflection
point, which integer flux never reaches. The smaller root is a minimum, and
the larger root is a barrier to decompactification.

Tune Λ to make m₀ flat, Λ = 2M⁸e²/m₀². Then:
- m < m₀ gives anti-de Sitter minima;
- m₀ < m < (2/√3) m₀ gives de Sitter minima;
- larger m has no vacuum.

For m₀ = 3:
- m = 1, 2 are AdS;
- m = 3 is Minkowski;
- m ≥ 4 has no vacuum (4² > 12).

`flux_landscape` computes this. The tests check it two ways, for m₀ = 3
and m₀ = 7:
- against direct grid minimization of the Einstein-frame potential;
- against a frame-independent oracle suggested by the review: the full 6D
  Einstein equations on dS₄ × S² (flat slicing) and AdS₄ × S² (Poincaré
  patch).

In that oracle the AdS equations are exactly the dS ones continued to
H² = −1/L². Eliminating H² reproduces the stationary-point quadratic, and the
sign of H² at the smaller root matches the vacuum type. The oracle matters
because a grid test at the Minkowski point alone cannot detect a wrong Weyl
factor: stationarity at V=0 is frame-independent.

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
  tensor fluctuations of the Einstein–Maxwell sector was established in the
  RSS literature (Randjbar-Daemi, Salam, Strathdee, Nucl. Phys. B214 (1983)
  491). Only the breathing mode is re-derived here. The citation does not
  cover the Spin(10), fermion and 2-form fluctuations, nor the Green–Schwarz
  mixing.
- The massless 4D gauge symmetry includes the SU(2) isometry of the round
  sphere (RSS 1983).
- Salam–Sezgin (Phys. Lett. B147 (1984) 47) is the supersymmetric version with
  gauged U(1)_R. There, flat M4 × S² requires no tuning of Λ. It is a
  candidate route for removing the tuning, with three costs:
  - The field equations fix the monopole number at ±1, so the family count
    would have to be redone.
  - The theory has a classically flat dilaton–radius direction, so removing
    the tuning brings back a modulus.
  - The anomaly bookkeeping changes, because the gravitino and gaugini carry
    R-charge.

## 8. Independent review

An independent adversarial review re-derived, with its own scripts:
- the Einstein equations;
- the potential;
- the kinetic term, for ψ(t,x) with the correct Lorentz structure;
- V″ and the radion mass;
- the 4D relations;
- Theorem 5, by the full (A)dS₄ × S² solve now used as a test.

It found one blocker, three major and seven minor issues. All are repaired
above:
- **Blocker.** The earlier text said the SU(2) isometry bosons sit near
  10¹⁷ GeV. They are massless at tree level.
- **Major:**
  - "no laboratory-scale boundary effects" was unconditional; it is now
    conditional on g4;
  - the Green–Schwarz test was a tautology; the sources are now computed
    from the fields, with controls;
  - the g4 and M_Pl normalizations were hard-coded; they are now derived by
    reduction.
- **Minor:**
  - the KK comparison;
  - the scope of "no light modulus";
  - the control criterion is O(1) and convention-dependent, and a second
    convention is added;
  - the scope of the RSS stability citation;
  - the Salam–Sezgin flux and modulus;
  - a strict inequality with the Λ>0 assumption, and the equality case now
    treated as no vacuum;
  - a Weyl-independent oracle for Theorem 5.

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
  The size of the sphere is not predicted.
- 6D gravity and 6D gauge theory are non-renormalizable. This is an effective
  theory with cutoff of order M; it is not UV complete.
