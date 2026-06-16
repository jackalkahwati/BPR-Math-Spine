# Closed and Deprecated — Negative Findings Registry

> **Status:** Single navigation point for everything in BPR that has been
> determined NOT to work or has been honestly closed as a negative finding.
> Existing for two reasons: (1) reviewers shouldn't have to scrape commit
> history to find what didn't pan out, (2) the framework's credibility
> depends on documented negatives as much as positive claims.

## 1. Structural negatives (architectural problems with no fix in current scope)

| Item | Status | Where | Notes |
|---|---|---|---|
| Original `RPSTHamiltonian` rank-1 problem | **STRUCTURALLY INCOMPATIBLE** | `bpr/rpst/hamiltonian.py`; locked in by `tests/test_gue_riemann_honest.py` | H = outer(leg, leg) is rank-1 by construction — one nonzero eigenvalue. Cannot show GUE level statistics. |
| Riemann/GUE conjecture for BPR | **DOWNGRADED** | `doc/conjectures/riemann_connection.md` | Originally "Tier 2 with KS p=0.92 numerical support" — not reproducible by any operator. Final survey verdict: no natural prime-modular Z_p Hermitian or unitary operator class reproduces Wigner-Dyson statistics. |
| Legendre Hankel/Multiplicative/Circulant Hermitian | **GAUSS-SUM DEGENERATE** | `bpr/substrate_hamiltonians.py`; tested in `tests/test_substrate_hamiltonians.py` | All ≤ 3 distinct eigenvalues (Hankel/Circulant) or rank-1 (Multiplicative). Cannot do level spacing. |
| Discrete Berry-Keating 1D and 2D | **POISSON (integrable)** | `bpr/substrate_hamiltonians.py` | Full rank, p (or p²) distinct eigenvalues, but Poisson level spacings rather than GUE. K-S D_Poisson ≈ 0.14 across primes 211–1009. |
| Hannay-Berry quantum cat map (unitary) | **POOR FIT to CUE** | `bpr/substrate_hamiltonians.py:quantum_cat_map_spectral_statistics` | Final attempt in the unitary class. K-S D ≈ 0.32 — much worse than the random-matrix sanity control (D = 0.07). Spectrum has arithmetic structure, not generic Wigner-Dyson. |

**Net result for the prime-substrate quantum-information program:** the
Hilbert-Pólya / Berry-Keating / GUE story for Riemann zeros via Z_p
operators is closed negatively. The prime-modular structure is
incompatible with generic random-matrix statistics in every natural
construction tested (Hermitian and unitary). This is the most substantive
negative finding of the project.

## 2. Refuted retrofits (artifacts we engaged with that didn't survive)

| Artifact | Verdict | Where |
|---|---|---|
| Buga sphere as Theory 0 device | **FALSIFIED on visual inspection** | `doc/experiments/BUGA_THEORY_0_ANALYSIS.md` (original analysis preserved as instructive failure case with FALSIFIED banner) |
| Integratron as Theory 0 device | **STRUCTURALLY INCOMPATIBLE** | 16-fold visible symmetry not in allowed substrate classes; no quasicrystal material; never operated |
| Pyramids as global coherence machines | **REJECTED** | `doc/RECOHERENCE_DEVICE_ROADMAP.md §8` (Bucket B); local-acoustic only, not global |
| Miami Bayside non-human entities | **TIER 0 EVIDENCE** | No instrumented documentation; framework engaged with the perception-cascade phenomenology, not the unverified entity claim |
| Giza pyramid 33 Hz pulse | **FABRICATED EVENT** | Not in any seismological network; Assange impersonator; numerological construction |
| LHCb new baryon as BPR-validating | **NON-DISCRIMINATING** | Standard Model particle predicted by quark model since 1964; BPR matches via inheritance, not unique derivation |

## 3. Refuted pseudoscientific claims (excluded from framework scope)

`doc/RECOHERENCE_DEVICE_ROADMAP.md §8` Bucket B + `doc/experiments/COERCIVE_AUDIO_CALIBRATION.md` exclusions:

- Subliminal audio messaging (Vokey & Read 1985, Greenwald 1991, Shanks 2020 all null)
- Solfeggio / 432 Hz / 528 Hz "healing frequency" claims (pseudoscientific)
- Hypersonic Effect (Oohashi 2000, 2006) — failed Meyer-Moran 2007 ABX replication (n=554)
- Backward masking ("satanic panic" era) — canonically refuted
- Audio steganography → human-brain behavioral decoding (no peer-reviewed evidence)
- Reverse speech (Oates) — pseudoscience
- Ambient consumer-router-style RF for coherence (10⁻⁶ T vs 1.5 T rTMS threshold; no mechanism)
- "Bioresonance" / scalar wave / tachyon / orgone / EM healing devices
- Acoustic weapons selectively disrupting DMN (no published frequency does this)
- Havana Syndrome as acoustic weapon (NIH/JAMA 2024 found no MRI-detectable injury; AARO investigation negative)

## 4. Fitted-but-presented-as-derived coefficients (parameter-honesty flags)

From the June 2026 parameter-honesty pass, complementary to Kontoyiannis-bound audit:

| Coefficient | Where | Status |
|---|---|---|
| θ_23 (PMNS) coefficient 1.35 | `bpr/neutrino.py` | FITTED, not derived from (p, z, n_gen) |
| θ_12 (PMNS) coefficient 3.5 | `bpr/neutrino.py` | FITTED, not derived |
| ln(p)/(ln(p)+1) finite-boundary correction in M_Pl/v_EW | `bpr/gauge_unification.py` | PHENOMENOLOGICAL; exponent z/2+1/3 is derived |
| 1/(4 ln p) interband factor in MgB₂ Tc | `bpr/superconductivity.py` | Coordination-shell 2/z alternative tested and REJECTED (overshoots); 1/(4 ln p) remains phenomenological |
| 1.57 in QCD 3-loop correction | `bpr/qcd_flavor.py` (LambdaQCD) | EMPIRICAL fit from lattice comparison |
| Eq (5) consciousness coupling exponents (α=1.2, β=1.5, γ=0.8, δ=1.0, ε=1.3, k=2.0) | `bpr/information.py` | Tunable floats, not derived rationals |

All flagged in their respective docstrings; this doc consolidates the
list for reviewers who want to see the honest parameter accounting in
one place.

## 5. Speculative sectors clearly bracketed (NOT closed — separated)

These are NOT negative findings, just sectors explicitly bracketed from
the flavor-sector evaluation because their evidence requirements are
separate. They remain valid framework content with their own evidence
streams:

- **Phason topological propulsion** — UNDECIDABLE at 31 orders below
  experimental sensitivity (`bpr/phason_sector.py:phason_defect_lift_budget`)
- **Eq (5) consciousness coupling** — empirically bounded ≤ 10⁻³
  behavioral, ≤ 10⁻⁵ at QRNG; consistent with null psi literature
- **Moral phenomenology / privation framework / exorcism mapping** —
  interpretive overlays, not derivations; documented honestly as such
- **Coercive audio detection device** — engineering target; calibrated
  against the audio-engineering literature; consent-required by design

These should NOT be cited as evidence for or against the flavor-sector
content. They live or die on their own evidence streams.

## 6. What this means for the framework's credibility

A framework's credibility lives or dies by what it admits doesn't work.
This doc consolidates BPR's documented negatives so they're visible at
one URL rather than scattered across the commit history. The framework
is meaningfully more credible than a pure-positive-claims version
because:

- Negative findings are explicit (rank-1 Hamiltonian, fabricated-event
  rejection, retrofit refusals, parameter-honesty flags)
- Discriminating tests are registered BEFORE measurement (Ξcc* hyperfine,
  δ=2 Casimir, doubly-bottom predictions)
- Speculative extensions are bracketed from the flavor-sector evaluation
- Information-theoretic audit independently confirms the parameter-honesty
  pass
- Mainstream-mapping positions BPR in the correct reference class

The framework's most ambitious claim — that BPR is *the* substrate of
reality — is unlikely correct (<1% probability per the comparative
assessment). But BPR's more modest claim — that it sits in the
holographic-emergent-spacetime family and makes specific sharp pre-
dictions in the SM flavor sector that LHCb can test — is defensible
and not dependent on the speculative extensions in §5.

## 7. Cross-references

- `doc/BPR_MAINSTREAM_MAPPING.md` — where BPR sits in the leading-frameworks landscape
- `doc/BPR_FLAVOR_SECTOR.md` — headline content (Asymptotic-Safety-style registered pre-dictions)
- `doc/PRIME_INFORMATION_STRUCTURE_CONNECTIONS.md` — Kontoyiannis + Latorre-Sierra connection
- `doc/LIMITATIONS_AND_FALSIFICATION.md` — original parameter accounting
- `doc/conjectures/riemann_connection.md` — Riemann conjecture downgrade
- `doc/conjectures/born_rule.md` — all four originally-stated gaps closed
- `tests/test_substrate_hamiltonians.py` — Hermitian + unitary spectral-statistics survey results
- `tests/test_gue_riemann_honest.py` — original rank-1 finding lock-in
