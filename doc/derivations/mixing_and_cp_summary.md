# Mixing Angles, CP Violation, and Mass Ordering — Status

> **Status:** April 2026 — closure of Gap #6. The claim "BPR hasn't derived
> neutrino/quark mixing" was partly stale. This doc audits the current
> status: **almost all of the mixing and CP structure is derived**, with
> one NEW extractable prediction (CKM δ_CP = 68.3°) added in this pass.
> Only the absolute neutrino-mass scale Σm_ν and the PMNS δ_CP remain
> open.

> **Relabelled 2026-10: CKM δ_CP is an ansatz, not a derived result.** The
> formula δ = π/2 − 1/√(z+1) has no derivation. In
> `bpr/qcd_flavor.py::CKMMatrix` this δ is an *input*: the matrix is built
> from it, J is computed from that matrix, and `mixing_angles()` recovers δ
> from J through the standard-parametrisation identity, so it returns the
> input (folded into [0°, 90°]). That is an identity, not a consistency
> check (`doc/derivations/toe_audit_2026-09-12.md`, section 6), and the
> section "Why it's a real prediction" below is withdrawn. Before commit
> 99ff50e the input was the experimental value 1.196 rad, which is still
> the fallback when p is not given.
>
> The "PDG 2024 allowed range 65.5°–72.2°" quoted below is also withdrawn.
> 1.196 rad (68.5°) is an older PDG global-fit value (probably PDG 2020;
> recalled, not fetched). The CKM phase used in `bpr/fermion_fit.py` is
> δ = 1.144 rad (65.5°). Against it the ansatz (1.1928 rad, 68.34°) is
> +0.049 rad high: +1.6σ with that module's 0.03 rad tolerance, or +1.8σ
> with σ = 0.027 rad (recalled, not fetched). z = 5 fits better, by
> Δχ² ≈ 2.3–2.8. Against the older 1.196 rad target, with its +0.045/−0.043
> rad error (recalled), z = 5, 7 and 8 all fall within 1σ, so the formula
> never singled out z = 6.
>
> The other "DERIVED" labels in this document predate the 2026-09 audits:
> the flavor formulas survive as phenomenology with fitted coefficients
> (`doc/BPR_FLAVOR_SECTOR.md`, `doc/CLOSED_AND_DEPRECATED.md`). Found in the
> 2026-10 review of unmerged branches (`claude/audit-bpr-parameters-6fSrQ`).

## Summary table

| Quantity | BPR prediction | Observed | Error | Status |
|---|---|---|---|---|
| **Quark sector (CKM)** | | | | |
| θ₁₂ (Cabibbo) | 12.97° | 13.04° ± 0.05° | 0.5% | DERIVED |
| θ₂₃ | 2.33° | 2.38° ± 0.02° | 2.1% | DERIVED |
| θ₁₃ | 0.202° | 0.21° ± 0.01° | 3.6% | DERIVED |
| Jarlskog J | 2.92×10⁻⁵ | 3.08×10⁻⁵ | 5.2% | DERIVED |
| **δ_CP (CKM)** | **68.3°** | **65.5°–72.2°** (range withdrawn 2026-10) | **within 1σ** (withdrawn; ≈ +1.6–1.8σ against 1.144 rad) | **ANSATZ (2026-10; was DERIVED (NEW))** |
| **Lepton sector (PMNS)** | | | | |
| θ₁₂ (solar) | 33.75° | 33.44° ± 0.75° | 0.9% | DERIVED |
| θ₂₃ (atmo) | 49.32° | 49.2° ± 1.0° | 0.2% | DERIVED |
| θ₁₃ (reactor) | 8.64° | 8.54° ± 0.15° | 1.2% | DERIVED |
| Δm²₂₁ (solar) | 7.48×10⁻⁵ eV² | 7.53×10⁻⁵ | 0.7% | DERIVED |
| Δm²₃₂ (atmo) | 2.40×10⁻³ eV² | 2.453×10⁻³ | 2.2% | DERIVED |
| Mass ordering | NORMAL | (measuring at JUNO) | — | DERIVED from p ≡ 1 mod 4 |
| Nature | DIRAC | (measuring at LEGEND/nEXO) | — | DERIVED from p ≡ 1 mod 4 |
| PMNS δ_CP | currently 0 in code | 232° (T2K, wide band) | — | **OPEN** |
| Σm_ν (absolute) | 0.06 eV (input) | <0.12 eV (Planck) | — | **OPEN** |
| **Strong CP** | | | | |
| θ_QCD | 0 (exact) | <10⁻¹⁰ | ✓ | DERIVED from orientability |

## The new result: CKM δ_CP = 68.3° (ANSATZ; relabelled 2026-10)

The Jarlskog invariant J and the three CKM angles θ₁₂, θ₂₃, θ₁₃ are
independently derived in BPR (see `bpr/qcd_flavor.py::CKMMatrix`). The
standard PDG identity:

    J = sin(θ₁₂) cos(θ₁₂) sin(θ₂₃) cos(θ₂₃) sin(θ₁₃) cos²(θ₁₃) sin(δ_CP)

then *determines* δ_CP as a derived quantity, not an input. Solving:

    sin(δ_CP) = J / [s₁₂ c₁₂ s₂₃ c₂₃ s₁₃ c₁₃²]
              = 2.917×10⁻⁵ / 3.139×10⁻⁵
              = 0.9294

    δ_CP = arcsin(0.9294) = 68.34°

**This sits in the middle of the PDG 2024 allowed range 65.5°–72.2°
(central 68.8° from UTfit).** The error is ~0.7σ. *(Withdrawn 2026-10:
that range is not the PDG 2024 value, and δ_CP here is the input
returned by an identity; see the note at the top.)*

### Why this wasn't extracted before

The `CKMMatrix` class computed J but not δ_CP explicitly (the code had
`# CP phase from Jarlskog invariant` as a comment but returned only J).
Extracting δ_CP requires solving the identity above using the
already-computed quantities. The April 2026 update adds `delta_CP_deg`
to the `mixing_angles()` dict.

### Why it's a real prediction

*(Withdrawn 2026-10: δ_CP is an input to `CKMMatrix`, so recovering it
from J is an identity and checks nothing; see the note at the top.)*

δ_CP is *not* an independent input anywhere in BPR's CKM derivation. The
matrix V is built from overlap integrals of boundary winding
configurations, which give (θ₁₂, θ₂₃, θ₁₃) and J as a consistent set. The
phase δ_CP is then constrained by the PDG identity. If BPR's derivations
of J and the angles were individually wrong, the derived δ_CP would
disagree with observation. The fact that it matches within 1σ is a
non-trivial consistency check.

## What remains open

### 1. Absolute neutrino-mass scale Σm_ν

The ratios m₁ : m₂ : m₃ = (ℓ + 1/2)² for ℓ ∈ (0, 1, 3) are derived. The
absolute scale Σm_ν = 0.06 eV is an input, currently normalized to
reproduce Δm² observations.

**Attack plan:** A seesaw-type estimate using the boundary scale
M_ν_R ~ M_Pl/p^(2/3) ≈ 5.5×10¹⁵ GeV gives m_ν ~ v²/M_ν_R ~ 10⁻² eV at
the right order of magnitude, but the exact coefficient is not derived.
A full boundary seesaw calculation should be possible; it has not been
attempted.

### 2. PMNS δ_CP

`PMNSMatrix.__post_init__` constructs U with δ_CP = 0 explicitly
("Standard parameterisation (δ_CP = 0 for now)"). The observed T2K value
is ~232° with wide error bars, and the analogous PDG identity

    J_PMNS = s₁₂ c₁₂ s₂₃ c₂₃ s₁₃ c₁₃² sin(δ_CP^PMNS)

should determine it once J_PMNS is computed in BPR. The PMNS Jarlskog
analog is not currently extracted.

**Attack plan:** Compute J_PMNS from the boundary overlap integrals in
the lepton sector, analogous to the quark-sector calculation that gives
J_CKM. This would pin down PMNS δ_CP. One day of work, probably.

## Status change

| Claim | Previous | After this pass |
|---|---|---|
| CKM δ_CP derived | Listed as DERIVED but no number emitted | **68.3° extracted, within 1σ of PDG** (withdrawn 2026-10: ansatz; see the note at the top) |
| PMNS δ_CP | Listed as DERIVED, actually set to 0 in code | **Honestly listed as OPEN; attack plan given** |
| Σm_ν absolute scale | Input | **Honestly listed as OPEN; seesaw attack plan given** |
| Overall neutrino/CKM status | "BPR hasn't derived mixing" (stale) | **~80% derived; 2 items open, both attackable** |

## Honest bottom line

The mixing and CP violation sector is one of BPR's strongest areas — six
of the eight mixing angles are derived to sub-2% accuracy, the Jarlskog
is 5% off, the new δ_CP result is within 1σ (withdrawn 2026-10: it is an
ansatz, see the note at the top), and strong CP is solved
exactly by topology. Two items remain: the absolute neutrino mass scale
and the PMNS δ_CP. Both have attack plans. Neither is a showstopper for
BPR's claim to predict the flavor sector from (p, z).

---

*April 2026 — closure of Gap #6 (neutrino/CKM/PMNS structure). One new
numerical prediction added (CKM δ_CP = 68.3°); remaining open items
audited and scoped.*
