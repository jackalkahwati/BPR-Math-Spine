# The cosmological constant in BPR-6D: no classical self-tuning

2026-09-26. Status: exact classical computation. The implementation is
`bpr/cosmological_constant.py`, with tests in
`tests/test_cosmological_constant.py` and a demo in
`scripts/demo_cosmological_constant.py`. The independent review is recorded
in section 5.

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

**Check.** Solving the 6D Einstein equations directly on the football metric,
with de Sitter slicing and flux on the reduced area, reproduces H² at three
tensions. At a larger tension both computations find no compactified vacuum.

## 2. Results

**Theorem 1 (no self-tuning).** Fix Λ, M, e and m.
- 4D is flat for exactly one deficit, α_flat = m√(Λ/2)/(M⁴e).
- At that point, dH²/dT ≠ 0. In the tested example it equals 1/(90π) > 0, so
  added brane vacuum energy produces de Sitter.
- A large enough tension, α with (m/α)² above the landscape bound, has no
  compactified vacuum at all.

The brane tension therefore does not drop out. Every shift of the brane
vacuum energy must be cancelled by hand: for example the Standard-Model
contributions from the QCD and electroweak scales, or from loops up to 1/r.

**Proposition 2 (discrete flat tensions).** For fixed Λ, the flat tensions
form a discrete set, one per flux quantum, equally spaced in α. Flux
quantization is what removes the continuous freedom self-tuning would need.
This is the classical argument of Garriga and Porrati (JHEP 08 (2004) 028) and
Navarro (2003), now in BPR-6D's own equations.

**Size of the tuning.** The observed vacuum energy, ρ_Λ ≈ (2.3 meV)⁴,
compares with natural scales as follows:
- ≈ 10⁻¹²⁰ of M_Pl⁴;
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
- flux fixed at ±1, which conflicts with the family count of BPR-6D;
- the anomaly structure of gauged U(1)_R (string_embedding, section 3);
- loop and brane-coupling issues debated in the literature (Garriga–Porrati;
  Vinet–Cline 2005; Burgess et al., later papers).

There is no consensus that they solve the problem. Weinberg's no-go theorem
for adjustment mechanisms (Rev. Mod. Phys. 61 (1989) 1) constrains any
field-theoretic self-tuning.

## 4. Status

The cosmological constant of BPR-6D is **tuned**, and the tuning cannot be
moved onto branes classically. The supersymmetric route is open and
contested, and it would cost the multiple-of-three family result (round 6).
This is recorded as an obstruction for this class, not a no-go for all
theories.

## 5. Independent review

Pending at the time of writing.

## 6. Limitations

- The analysis is classical, with idealized tension-only branes: no
  brane-localized flux, dilaton couplings or brane matter backreaction.
- The supersymmetric claims are cited, not computed.
- Quantum corrections are represented only as shifts of T or Λ.
