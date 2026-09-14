# BPR Experiments

> Historical literature comparisons and candidate experimental tests.
> Legacy verdicts are retained below; they do not uniformly constitute
> independent confirmation or falsification of BPR-specific mechanisms.
> Current claim status follows the [negative-findings registry](../CLOSED_AND_DEPRECATED.md)
> and [foundation audit](../derivations/foundation_prerequisites_2026-09-13.md).

## Quick Links

| Document | Purpose |
|----------|---------|
| [**papers.md**](papers.md) | All papers with results vs BPR; CONFIRM / FALSIFY / INCONCLUSIVE verdicts |
| [**THEORY_CONFIRMATION_BREAKDOWN.md**](THEORY_CONFIRMATION_BREAKDOWN.md) | Which part of each of the 21 theories is confirmed |
| [**EVIDENCE_PIPELINE.md**](EVIDENCE_PIPELINE.md) | Continuous evidence ingestion, staging, and triage workflow |
| [**evidence_queue.md**](evidence_queue.md) | Auto-generated queue of newly staged candidate evidence |
| [**EXPERIMENTAL_ROADMAP.md**](../EXPERIMENTAL_ROADMAP.md) | Future tests and falsification criteria |

## Historical summary (as of Feb 2026; legacy verdicts retained)

| Verdict | Count | Tests |
|---------|-------|-------|
| **CONFIRM** | **115** | All 21 theories + 0νββ, MOND, η, DM σ/m, Σm_ν, PMNS, CKM, Tc, proton τ, inflation, nuclear, NS, α, quarks, CKM/PMNS angles, Δm², dimensions, n_s, r, G₀, m_e, m_μ, m_τ, Koide, pion, Higgs, v_EW, m_p, Ω_DM, strong CP, ΔN_eff, magic numbers, R_K, GW speed, Tsirelson, 3D Ising, dark energy, B/A, n_sat, Planck length, memory, decoherence, info geometry, adiabatic, bioelectric, Kuramoto, Clifford, quantum chemistry, KZ |
| **FALSIFY** | 0 | — |
| **INCONCLUSIVE** | 11 | LIV, Casimir, mass ordering, Born rule, GUP, decoherence, Hubble tension, anyons, proton radius, muon g−2, GRB LIV, JWST H₀ |

**Papers:** 250+ cited across 129 tests (see papers.md totals). **All 21 theories** have CONFIRM. Feb 2026: added papers for BPR-unique topics (Casimir superconducting, Born rule many-photon, decoherence mass scaling, LIV CTA/GRB, m_s/m_d lattice).

The historical counts above have not been recomputed. They include inherited
relations, fitted or assumed inputs, bounds, and literature comparisons.
They are not a current tally of independent BPR confirmations and do not
supersede subsequently recorded closures or withdrawals. In particular,
the zero historical FALSIFY count does not erase the closed particle sector.

The legacy benchmark grade is a numerical comparison heuristic, not a
statistical hypothesis test. It mixes sigma thresholds, relative-error
fallbacks, exact-reference checks and one-sided bound checks. A PASS does
not establish independent predictive success or validate BPR's derivation.
No historical scores have been regenerated for this qualification.

## How to Use

```bash
# Run BPR predictions for comparison
python -c "
from bpr.first_principles import SubstrateDerivedTheories
sdt = SubstrateDerivedTheories.from_substrate(p=104761, N=10000, J_eV=1.0)
preds = sdt.predictions()
for k in ['P2.2_MOND_a0', 'P11.7_baryon_asymmetry_eta', 'P4.9_Tc_MgB2_K']:
    print(f'{k}: {preds[k]}')
"
```

## Contributing

When adding a new paper:
1. Add to `papers.md` with full citation
2. State BPR prediction and experimental result
3. Assign verdict: CONFIRM | FALSIFY | INCONCLUSIVE, with explicit provenance:
   measurement, bound, empirical/calibration input, inherited relation, or
   literature target. State whether the comparison discriminates a BPR-specific
   mechanism; numerical agreement alone is not confirmation.

## Continuous Intake

The repo now includes a first-pass evidence worker:

```bash
python3 scripts/research_loop.py show-policy
python3 scripts/research_loop.py evidence-scan
```

This stages machine-readable paper candidates under `data/evidence/staging/`,
updates `evidence_queue.md`, and appends an event to the research audit log in
`data/research/audit/`.
