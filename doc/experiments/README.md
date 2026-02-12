# BPR Experiments

> Experiments and papers that can **confirm** or **falsify** BPR predictions.
> Sourced from published literature with explicit verdicts.

## Quick Links

| Document | Purpose |
|----------|---------|
| [**papers.md**](papers.md) | All papers with results vs BPR; CONFIRM / FALSIFY / INCONCLUSIVE verdicts |
| [**THEORY_CONFIRMATION_BREAKDOWN.md**](THEORY_CONFIRMATION_BREAKDOWN.md) | Which part of each of the 21 theories is confirmed |
| [**EXPERIMENTAL_ROADMAP.md**](../EXPERIMENTAL_ROADMAP.md) | Future tests and falsification criteria |

## Summary (as of Feb 2026)

| Verdict | Count | Tests |
|---------|-------|-------|
| **CONFIRM** | **115** | All 21 theories + 0νββ, MOND, η, DM σ/m, Σm_ν, PMNS, CKM, Tc, proton τ, inflation, nuclear, NS, α, quarks, CKM/PMNS angles, Δm², dimensions, n_s, r, G₀, m_e, m_μ, m_τ, Koide, pion, Higgs, v_EW, m_p, Ω_DM, strong CP, ΔN_eff, magic numbers, R_K, GW speed, Tsirelson, 3D Ising, dark energy, B/A, n_sat, Planck length, memory, decoherence, info geometry, adiabatic, bioelectric, Kuramoto, Clifford, quantum chemistry, KZ |
| **FALSIFY** | 0 | — |
| **INCONCLUSIVE** | 11 | LIV, Casimir, mass ordering, Born rule, GUP, decoherence, Hubble tension, anyons, proton radius, muon g−2, GRB LIV, JWST H₀ |

**Papers:** 250+ cited across 129 tests. **All 21 theories** have CONFIRM.

## How to Use

```bash
# Run BPR predictions for comparison
python -c "
from bpr.first_principles import SubstrateDerivedTheories
sdt = SubstrateDerivedTheories.from_substrate(p=104729, N=10000, J_eV=1.0)
preds = sdt.predictions()
for k in ['P2.2_MOND_a0', 'P11.7_baryon_asymmetry_eta', 'P4.9_Tc_MgB2_K']:
    print(f'{k}: {preds[k]}')
"
```

## Contributing

When adding a new paper:
1. Add to `papers.md` with full citation
2. State BPR prediction and experimental result
3. Assign verdict: CONFIRM | FALSIFY | INCONCLUSIVE
