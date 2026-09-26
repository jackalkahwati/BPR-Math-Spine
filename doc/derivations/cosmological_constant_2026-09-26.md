# The cosmological constant in BPR-6D: no classical self-tuning

2026-09-26. Status: exact classical computation. The implementation is
`bpr/cosmological_constant.py`, with tests in
`tests/test_cosmological_constant.py` and a demo in
`scripts/demo_cosmological_constant.py`. An independent review (section 5)
confirmed every equation; its repairs, mostly citations and scope, are
applied.

## 0. Question

The BPR-6D vacuum is flat only if the 6D cosmological constant is tuned,
Λ = 2M⁸e²/m² (flux note, Theorem 1). Six-dimensional brane worlds are the
classic setting for proposals to avoid this: the "football" or "rugby ball"
geometries and supersymmetric large extra dimensions (SLED). The idea is that
the 4D vacuum energy lives on branes, and a brane tension only changes the
deficit angle of the sphere, leaving 4D flat.

Does that work for BPR-6D?

## 1. Setup

Place two antipodal codimension-2 branes of tension T on the sphere. The
sphere metric becomes dθ² + α² sin²θ dφ², with

    α = 1 − δ/2π,   δ = T/M⁴.

Here M⁴ = 1/(8πG₆), from the (M⁴/2)R normalization. The local 6D equations
are unchanged. Flux quantization on the reduced area 4πr²α, however, turns
the flux m into m/α. Solving the full 6D equations on (A)dS₄ × football
(the same method as the frame-independent oracle in the flux note) gives

    Λu² − 2M⁴u + (3/8)(m/α)²/e² = 0,   u = r²,
    H² = 1/(3r²) − B²/(3M⁴),   B = m/(2eαr²).

**What is held fixed.** As the tension varies, the flux quantum m is held
fixed: dF = 0 conserves ∫F, and BPR-6D has no magnetically charged branes
(flux note, section 6). Carroll and Guica (hep-th/0302067) study the same
model holding the field strength B fixed instead, tuned against Λ, so that T
only changes the deficit. When the tension changes dynamically, it is the
conserved flux quantum, not B, that stays fixed.

**Checks** (tested):
- Solving the 6D Einstein equations directly on the football metric, with
  the flux fixed by integrating F over the football, reproduces H² at five
  tensions: two de Sitter, one flat, two anti-de Sitter.
- At a larger tension, both computations find no compactified vacuum.
- On a regularized cone, the integrated brane energy density M⁴G_tt gives
  T = 2πM⁴(1 − α), i.e. δ = T/M⁴ = 8πG₆T.

**Stability.** The brane action cancels the conical curvature term, so the
breathing-mode potential is the round-sphere one with m → m/α:

    V(u) ∝ Λ/u − M⁴/u² + (m/α)²/(8e²u³).

V′ = 0 reproduces the constraint above, and on shell V″ ∝ u(M⁴ − Λu). This
is positive at the smaller root and negative at the larger one, so the
smaller root is the stable branch (tested symbolically, and numerically at
three tensions).

## 2. Results

**Theorem 1 (no self-tuning).** Fix Λ, M, e and m, within the unwarped
ansatz with two equal tensions.
- 4D is flat for exactly one deficit. H = 0 together with the constraint
  forces u = M⁴/(2Λ) and B² = 2Λ on either root, which pins
  α_flat = m√(Λ/2)/(M⁴e). So every other tension curves 4D: this is a
  global statement, not only a first-order one.
- At the flat point, dH²/dT = √2·e·√Λ/(3π m M⁴) > 0 (tested symbolically;
  1/(90π) in the example). Added brane vacuum energy produces de Sitter.
- A large enough tension, α with (m/α)² above the landscape bound, has no
  compactified vacuum at all.

The brane tension therefore does not drop out. Every shift of the brane
vacuum energy must be cancelled by hand: for example the Standard-Model
contributions from the QCD and electroweak scales, or from loops up to 1/r.

**Proposition 2 (discrete flat tensions).** For fixed Λ, the flat tensions
form a discrete set, one per flux quantum, equally spaced in α. Flux
quantization is what removes the continuous freedom self-tuning would need.
This is the classical argument of Navarro ("Spheres, deficit angles and the
cosmological constant", CQG 20 (2003) 3603, hep-th/0305014) and of Garriga
and Porrati (JHEP 08 (2004) 028), now in BPR-6D's own equations.

**Size of the tuning.** The observed vacuum energy, ρ_Λ ≈ (2.3 meV)⁴,
compares with natural scales as follows:
- ≈ 10⁻¹²⁰ of M_Pl⁴, with the reduced Planck mass (≈ 10⁻¹²³ with the
  unreduced one);
- ≈ 10⁻¹¹⁶ of (1/r)⁴ for 1/r ≈ 10¹⁷ GeV;
- ≈ 10⁻⁵⁹ of (TeV)⁴.

BPR-6D offers no mechanism against any of these.

## 3. The supersymmetric route (cited, not derived)

Supersymmetric 6D models have been proposed as a way around this:
- Salam–Sezgin (1984) gives Minkowski × S² without tuning.
- SLED (Aghababaie, Burgess, Parameswaran and Quevedo, Nucl. Phys. B680
  (2004) 389) adds branes and argues for a technically small 4D vacuum
  energy.

These proposals face known objections:
- a classically flat dilaton–radius modulus;
- the Salam–Sezgin flux lies in the gauged U(1)_R, not in U(1)_F. Its
  quantum, ±1 in the R-charge normalization, does not itself conflict with
  n_gen = q|m| = 3, which also has |m| = 1. The real costs are that a
  vacuum with non-R flux is an open question, and that in an F-theory
  embedding 3 | n_gen is allowed but not forced (string_embedding,
  section 3). With SLED branes, flux quantization also involves α and
  brane-localized flux;
- loop and brane-coupling issues debated in the literature (Garriga–Porrati;
  Vinet–Cline 2005, hep-th/0501098; Burgess et al., later papers).

There is no consensus that they solve the problem. Weinberg's no-go theorem
for adjustment mechanisms (Rev. Mod. Phys. 61 (1989) 1) has assumptions: 4D,
finitely many fields and translation invariance. SLED proponents claim to
evade it. Garriga and Porrati argue that these models reduce to 4D and do
not. The applicability of the theorem is part of what is contested.

## 4. Status

The cosmological constant of BPR-6D is **tuned**, and the tuning cannot be
moved onto branes classically. The supersymmetric route is open and
contested. In an F-theory embedding it would leave the multiple-of-three
family number allowed but no longer forced (round 6).
This is recorded as an obstruction for this class, not a no-go for all
theories.

## 5. Independent review

An independent review, with its own SymPy and its own Ricci routine,
verified:
- δ = T/M⁴ on a regularized cone;
- that α drops out of the Einstein equations;
- the H² formula and the constraint;
- stability of the smaller root;
- the flat-point derivative in closed form;
- the tuning numbers and the main citations (arXiv abstracts via search
  results).

Its findings, all repaired:
- **Major** (citations and supporting text; none in the equations):
  - Navarro is now cited precisely (hep-th/0305014).
  - The Carroll–Guica contrast is stated, with the global flatness statement.
  - The "flux fixed at ±1" objection is rewritten: it does not conflict with
    n_gen = 3, and the real costs are named.
- **Minor:**
  - The test oracle now fixes B by integrating F, and a regularized-cone
    test of δ = T/M⁴ is added.
  - de Sitter tensions are added to the oracle test.
  - Theorem 1's derivative is tested in closed form.
  - The stability argument and the scope limits are stated.
  - The Weinberg claim is qualified.
  - The flat-tension test now checks H² = 0, and the Planck mass is
    labelled as reduced.

## 6. Limitations

- The analysis is classical, with idealized tension-only branes: no
  brane-localized flux, dilaton couplings or brane matter backreaction.
- Only the unwarped ansatz with two equal tensions is solved. Warped
  solutions with unequal tensions exist (e.g. Mukohyama et al. 2005, cited
  from memory); they also need one relation for flatness, so they do not
  self-tune either.
- Only the breathing mode is shown stable. The round-sphere stability result
  (Randjbar-Daemi–Salam–Strathdee 1983) does not cover the football, which
  breaks SU(2) to U(1); its other modes are not analysed.
- The supersymmetric claims are cited, not computed.
- Quantum corrections are represented only as shifts of T or Λ.
