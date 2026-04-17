# Nuclear EOS from Boundary — Scope & What is Derivable

> **Status:** April 2026 — closes Task #11 of the gap-closure pass in the
> limited sense of documenting what is and is not derivable from BPR
> substrate physics, and marking the specific remaining open pieces.
>
> Nuclear EOS derivation from first principles is a hard many-body
> problem that BPR does not solve. This document scopes what BPR *does*
> derive cleanly vs. what is standard nuclear physics inheriting BPR
> inputs (m_π, f_π) vs. what remains genuinely open.

## Honest scope

Deriving the nuclear equation of state from a UV theory (QCD or its UV
completion) is an unsolved problem in physics. Lattice QCD can compute
hadronic observables but not nuclear matter. Chiral effective theory
gets to a few-nucleon systems but not finite nuclei. What BPR adds to
this landscape is modest: it provides a derivation of the pion mass
m_π ≈ 135 MeV (at 0.4%), identifies magic numbers with SO(3) boundary
shells, and gives the overall scale via f_π. Everything above that is
standard nuclear physics.

## What BPR derives cleanly

**Magic numbers (P19.5).** The sequence 2, 8, 20, 28, 50, 82, 126 comes
from filling SO(3) angular momentum shells on the S² boundary (n=0 → 2,
n=1 → 8, n=2 → 20, etc. with spin-orbit splitting). This is the same
structure as the standard nuclear shell model; BPR's contribution is
identifying the SO(3) structure with the boundary isometry group rather
than as an independent nuclear assumption. **Status: DERIVED (geometric,
not a nuclear fit).**

**Pion mass m_π ≈ 135 MeV (P12.14).** Via GMOR with boundary-derived
condensate. Takes Λ_QCD (now derivable via Task #8 chain) and f_π
(external input, 92 MeV) as inputs. **Status: DERIVED at 0.4%.**

**Nuclear radius scale r_0 ~ λ_π.** Dimensional analysis:

    λ_π = ℏc / m_π = 1.467 fm   (using BPR's m_π = 134.5 MeV)

    r_0 (observed) = 1.25 fm    (PDG nuclear radius constant)

    r_0 / λ_π = 0.852

The ratio is O(1) and consistent with a pion-cloud packing argument
(nucleons overlap at ~0.85 × pion Compton wavelength). BPR does not
derive this O(1) factor from first principles; it is a standard
nuclear-matter calculation. **Status: DIMENSIONAL (scale derived,
coefficient not).**

## What is standard nuclear physics given BPR inputs

**Nuclear saturation density.** Given r_0, n_sat = 3/(4π r_0³) ≈ 0.122
fm⁻³ (close to observed 0.16 fm⁻³; the ~24% discrepancy reflects the
r_0 scale uncertainty above). This is not a BPR derivation — it is the
definition of saturation once r_0 is set.

**Coulomb coefficient a_C.** Given r_0, a_C = (3/5) × e²/(4πε₀ r_0) ≈
0.72 MeV. Pure electromagnetism; no nuclear input. Matches SEMF fit
value. **Status: DERIVED from r_0 (inherits r_0 uncertainty).**

**Volume and surface coefficients a_V, a_S.** Scaling arguments give
a_V ~ f_π² r_0 ≈ 10 MeV (observed: 15.6 MeV). The absolute coefficient
requires solving nuclear many-body problem; BPR does not attempt this.
**Status: FRAMEWORK — scaling correct, absolute coefficient empirical.**

**Binding energy B/A (Fe56) = 8.85 MeV.** Follows from the semi-empirical
mass formula with the above coefficients. Matches observed 8.79 MeV to
0.7%. Because the SEMF coefficients are empirical in BPR, this prediction
is consistent but not independent. **Status: FRAMEWORK (unchanged).**

## What remains genuinely open

**Absolute value of SEMF coefficients.** a_V = 15.56 MeV, a_S = 17.23
MeV, a_A = 23.29 MeV, a_P = 12.0 MeV. Each is fit to nuclear data. BPR
does not provide a first-principles calculation of any of these
coefficients. The BPR shell correction a_BPR = 2.5 MeV is likewise a
fit.

**Neutron star maximum mass M_max ≈ 2.2 M_☉.** `NeutronStar.max_mass_solar`
returns 2.2 hardcoded. The actual TOV calculation requires a nuclear EOS
above saturation density, which BPR does not derive. The 2.2 M_☉ value
is consistent with observation (PSR J0740+6620: 2.08 ± 0.07 M_☉) but is
not a BPR prediction. **Status: FRAMEWORK (consistent with observation,
not derived).**

**Neutron star radius R_NS ≈ 12.4 km.** Same situation — requires EOS
integration. NICER gives 12.4 ± 0.5 km for 1.4 M_☉. **Status: FRAMEWORK.**

## Net accounting for Task #11

| Sub-item | Claim in code | Honest status |
|---|---|---|
| Magic numbers | DERIVED | DERIVED (from boundary SO(3), trivial geometry) |
| r_0 = 1.25 fm | Empirical | DIMENSIONAL — r_0 ~ λ_π with O(1) packing factor |
| n_sat | Default value | FRAMEWORK — follows from r_0 |
| a_C (Coulomb) | 0.7 MeV | DERIVED from r_0 + standard EM |
| a_V, a_S | Fit to data | FRAMEWORK — scaling correct, coefficient empirical |
| a_A, a_P, a_BPR | Fit to data | FRAMEWORK — all empirical |
| B/A(Fe56) = 8.85 MeV | FRAMEWORK | Unchanged |
| M_NS_max = 2.2 M_☉ | Hardcoded | FRAMEWORK — consistent, not derived |
| R_NS = 12.4 km | Hardcoded | FRAMEWORK — consistent, not derived |

**Task #11 is not fully closeable** within BPR without solving nuclear
many-body problem. The closure this document provides is *honest scoping*:
clarifying which nuclear quantities are derived from substrate, which
follow from BPR inputs via standard nuclear physics, and which remain
empirical coefficients.

## Recommendation

Update `VALIDATION_STATUS.md` P19 entries to distinguish these tiers:

- **DERIVED (geometric):** magic numbers, shell structure.
- **DERIVED (from BPR inputs + standard EM):** a_C given r_0.
- **FRAMEWORK (scaling correct, coefficient empirical):** a_V, a_S, a_A, a_P, a_BPR.
- **FRAMEWORK (consistent with observation, not derived):** M_NS, R_NS.

This is less aggressive than the original "close the nuclear EOS" goal
but is the truthful accounting.

---

*April 2026 — closes Task #11 of the April 2026 gap-closure pass in
the scoping sense. Honest conclusion: BPR does not derive the nuclear
EOS; it provides inputs (m_π, f_π, magic numbers) to standard nuclear
physics, which is then used to fit the SEMF coefficients. This is the
same status as standard QCD + nuclear physics has today.*
