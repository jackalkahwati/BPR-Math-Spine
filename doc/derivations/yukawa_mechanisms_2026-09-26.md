# Yukawa mechanisms for BPR-6D: brane and bulk Higgs fields

2026-09-26. Status: exact symmetry analysis with explicit rotation matrices
and zero modes. The implementation is `bpr/yukawa_mechanisms.py`, with tests
in `tests/test_yukawa_mechanisms.py` and a demo in
`scripts/demo_yukawa_mechanisms.py`. The independent review is recorded in
section 5.

## 0. Question

Two earlier results frame this note:
- Round 3 ([family_symmetry_from_flux_2026-09-26.md](family_symmetry_from_flux_2026-09-26.md))
  showed that the minimal BPR-6D content has no Yukawa couplings, and that a
  bulk Higgs couples only through an internal one-form 10 or 126 in the J=2
  channel.
- Round 7 ([gut_breaking_2026-09-26.md](gut_breaking_2026-09-26.md)) showed
  that the Higgs sector must be supplied anyway.

So which supplied Higgs fields give usable Yukawa couplings? The **families
are an SU(2)-isometry triplet** (m = 1, 0, −1), and that structure is what
constrains every option below.

## 1. A Higgs localized at a point: never three distinct masses

The first option is a boundary mechanism: a 4D Higgs on a codimension-2
brane at a point of S² (by symmetry, the north pole). It carries F-charge −6,
so that it neutralizes two charge-3 families. The point is fixed by the
rotations about its axis, so J_z is exactly conserved.

The brane field carries some J_z charge c. Two things set c:
- its spin in the normal bundle;
- the lift of rotations to the U(1)_F bundle at the point, since a charged
  field at a pole of a monopole carries extra angular momentum.

The derivative counting of Proposition 2 suggests c = 1 for a scalar brane
Higgs, but c is left free here, because the result holds for every value.

**Theorem 1.** For a Higgs of J_z charge c at one point, the Yukawa satisfies
Y_{mm′} = 0 unless m + m′ = c. The J_z-covariant complex symmetric matrices
on the triplet give:
- **|c| ≥ 3:** no coupling at all.
- **c = ±2:** a single entry, rank 1, so one massive family.
- **c = ±1:** entries (1,0) or (0,−1), so the spectrum is (|a|, |a|, 0).
- **c = 0:** entries (1,−1) and (0,0), so the spectrum is (|a|, |a|, |b|).

**A single point brane never gives three distinct nonzero masses.**

*Checks.*
- The covariant matrices are computed for c = −4…4 as a nullspace, using
  explicit spin-1 rotation matrices, and random elements are classified by
  their spectra (tested).
- Every c=0 basis element commutes with diag(e^{−ima}).

The c=0 case is the J=0 singlet plus the M=0 component of J=2. It is
consistent with the round-3 orientation theorem, where the uniaxial J=2 vev
gave 2:1:1.

**Proposition 2 (derivative counting).** Near the point, the zero
modes vanish as θ^{n_m} with n_m = 1 − m, i.e. orders (0, 1, 2) for m = 1, 0, −1.
This is measured on the explicit spin-weighted harmonics, and gives (0…4) for
five families.

A brane operator that probes a family with vanishing order n needs n
derivatives, each suppressed by ε = 1/(M_* r). Entries with m + m′ = c need
n_m + n_{m′} = 2 − c derivatives. That is O(1) for c=2, O(ε) for c=1 and
O(ε²) for c=0. So within a single brane, every allowed entry sits at the same
order and there is no internal hierarchy.

The derivative count ties c to the operator. The only bilinear nonzero at the
point without derivatives is f₁f₁, but a scalar brane field cannot absorb its
free Γ^a index. The leading scalar operator is ψᵀCΓ^a D_aψ, which gives c = 1
and spectrum (|a|, |a|, 0). That identification is suggestive, not proved.
Theorem 1 does not depend on it.

**Two branes** at different points lift the degeneracy. With O(1)
coefficients the spectra are generic: the median of m₁/m₃ is O(0.1–1). A
ratio like m_u/m_t ~ 10⁻⁵ arises in fewer than 2% of random samples (tested).
Two branes also break the SU(2) family symmetry completely, which removes the
massless family gauge bosons of round 3. The hierarchy, however, is not
explained.

## 2. A bulk internal-vector Higgs: allowed, tuned

A non-gauge (Proca) internal one-form 10 or 126 of F-charge −6 has monopole
number |n| = 6 in unit flux. Its aligned lowest level is

    m² r² = |s_e| + 1 − g|n|/2 + M²r²,   |s_e| = |n|/2 − 1 = 2,

with g the gyromagnetic ratio. For the gauge value g = 2 and M = 0 this is
−3, matching round 3, where it was checked against Atiyah–Bott and a
Yang–Mills Hessian. The J=2 Higgs quintet can be made light only by tuning
M²r² to 3g − 3 to a precision of (m_H r)² ≈ 10⁻³⁰ for 1/r ≈ 10¹⁷ GeV. This is
the ordinary hierarchy problem; BPR-6D does not solve it.

When it works, round 3's Theorem 4 applies:
- a real vev obeys m₃ = m₁ + m₂, which is excluded;
- a complex vev reaches any spectrum;
- hierarchies need a near-null orientation.

## 3. Status of the Yukawa sector

| mechanism | Yukawa? | hierarchy? | status |
|---|---|---|---|
| bulk scalar 10/126 | no, at any derivative order (round 3) | — | excluded |
| SO(12) gauge–Higgs | paired families, vacuum instability (round 3) | — | excluded |
| one brane Higgs | rank 1 or a degenerate pair, whatever its J_z charge | no internal hierarchy | excluded as sole source |
| two brane Higgses | yes, generic | no (O(1) ratios) | allowed, hierarchy unexplained |
| bulk Proca 10/126 (J=2) | yes, any spectrum | only by near-null tuning | allowed, doubly tuned |

**Conclusion.** BPR-6D has no natural Yukawa mechanism. Yukawa couplings can
be added, with branes or a bulk vector, but the observed hierarchies are not
explained. The flavor sector stays open. The legacy flavor formulas remain
unconnected phenomenology.

## 4. Limitations

- Every Higgs field here is supplied.
- Brane dynamics, tension and backreaction are not modelled, and ε is a free
  parameter.
- The UV consistency of a charged Proca field is not addressed.
- J_z-charged brane fields, e.g. sections of the normal bundle, are not
  considered. They would open further entries.

## 5. Independent review

Pending at the time of writing.
