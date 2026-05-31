# Consciousness Coupling — Empirical Calibration

> **Purpose:** Anchor BPR's Eq (5) consciousness-φ coupling χ_b to the peer-reviewed
> empirical record on psi phenomena (telepathy, precognition, remote viewing,
> micro-PK). Provides the upper-bound envelope for χ_max · ε at behavioral and
> controlled-substrate readout levels.
> **Method:** Five-angle literature survey (May 2026), preregistered work and
> bias-corrected meta-analyses weighted over narrative reviews.

---

## 1. Effect-size envelope from the literature

| Phenomenon | Defensible upper bound (preregistered + bias-corrected) | Skeptic-side bound | Key source |
|---|---|---|---|
| Ganzfeld telepathy | r ≈ 0.08, CI [0.04, 0.12] | r ≈ 0.00–0.05 | Tressoldi & Storm 2024 Stage 2 RR ([F1000](https://f1000research.com/articles/10-234)); Milton & Wiseman 1999 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/11393304/)) |
| Precognition (Bem-style) | g ≈ 0.09 (shrinks to ~0 under PET-PEESE) | d ≈ 0.00–0.04 | Bem et al. 2015 ([F1000](https://f1000research.com/articles/4-1188)); Galak et al. 2012 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/22924750/)); Kekecs et al. 2023 Transparent Psi Project ([RSOS](https://royalsocietypublishing.org/doi/10.1098/rsos.191375)) — strong null |
| Remote viewing | d ≈ 0.1–0.25 (Utts SAIC, single lab) | ~0 in independent labs | Utts 1995 ([PDF](https://ics.uci.edu/~jutts/air.pdf)); Hyman 1995 ([HTML](https://ics.uci.edu/~jutts/hyman.html)); AIR 1995 ([PDF](https://nsarchive2.gwu.edu/NSAEBB/NSAEBB438/docs/doc_57.pdf)) — no operational utility, program terminated |
| RNG micro-PK | ~2×10⁻⁴ per bit (PEAR claim) | After trim-and-fill **CI includes zero** | Bösch et al. 2006 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/16822162/)); Maier et al. 2018 QRNG preregistered ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5872141/)) — **Bayes 10:1 favoring null** |

**Hostile-but-fair statistical reviewer's band:** d ≈ 0.00–0.10, with the upper
edge applying only to the most rigorous preregistered ganzfeld work and
remaining statistically consistent with zero under PET-PEESE / trim-and-fill
correction.

**Pattern across phenomena:**
1. Effects shrink with methodological rigor (preregistration, open data, blind judging, automated randomization).
2. The largest preregistered modern replications (Kekecs 2023 Transparent Psi Project, Maier 2018 QRNG) return strong evidence for the null.
3. Where any small effect persists, it is driven by a handful of "productive labs" within multi-lab meta-analyses (Lakens critique of Bem et al. 2015).
4. The PEAR three-lab consortium replication (Jahn et al. 2000) **failed** its primary prediction — PEAR itself could not replicate its claim under multi-lab control.

---

## 2. State-dependence: BPR's phase-transition prediction vs. the data

BPR Eq (5) gates χ_b through a sigmoid σ[k(Φ/Φ_c − 1)] — a phase-transition
switch in integrated information. The literature does **not** support this:

| State variable | Reported relationship | Source |
|---|---|---|
| Hypnotizability × altered-state shift (PCI) | Continuous correlation r=0.40 (high hypnotizables only); did not replicate cleanly | Cardeña & Marcusson-Clavertz 2011/2020 |
| Belief (sheep-goat) | Continuous correlation with belief score, **not threshold** | Schmeidler / Lawrence 1993 review (685k trials, 4500 participants) |
| Meditator selection × ganzfeld | Selected-participant advantage shrinks substantially in preregistered modern work | Storm/Tressoldi 2009–2018 vs Bem-Honorton 1994 |
| Dream/REM telepathy | Independent labs (Belvedere & Foulkes; Edinburgh) failed to replicate Maimonides REM enhancement | Krippner et al. vs. independent replications |
| **Anesthesia / DOC patients** | **No peer-reviewed studies exist** | — |

**Implication for Eq (5):**
- The sigmoid switching term is empirically unsupported.
- The relationship between Φ and reported effect is more consistent with a
  continuous monotone coupling than with a step-function.
- The phase-transition prediction is **not falsified — it is untested**. The
  framework can defend the sigmoid by specifying that the threshold lies below
  the Φ of any waking subject ever tested, and predicting effect collapse under
  general anesthesia or in DOC patients. That's a testable claim BPR should
  stake out explicitly.

---

## 3. Refinements warranted

### A. Add empirical-bound constants

`bpr/information.py` now exposes:

- `BEHAVIORAL_UPPER_BOUND_D = 0.10` — defensible upper bound on χ_max·ε·gain
  at the behavioral level, from preregistered psi meta-analyses
- `CONTROLLED_RNG_UPPER_BOUND_PER_BIT = 1e-5` — defensible upper bound at the
  controlled physical-readout level, from Bösch + Maier 2018
- `EMPIRICAL_ANCHOR_REFS` — citation strings for the bounds

These do not change any computed result. They make the empirical envelope a
first-class object in the module so future model construction reads against
the literature, not against the arbitrary default.

### B. Continuous-coupling alternative

`continuous_consciousness_coupling()` provides a softened-monotone alternative
to the sigmoid, parameterized by the same six factors but with the σ[k(...)]
switch replaced by a smooth (Φ/Φ_c) → tanh transition that approaches linear
for k ≪ 1. The default `k = 2.0` σ-switch remains for backward compatibility;
the continuous version represents what the state-dependence data actually
support. Use the sigmoid if you mean to defend the threshold prediction
empirically; use the continuous form if you mean to fit existing data.

### C. Testable phase-transition prediction (recorded)

If the sigmoid is retained, BPR commits to the following falsifiable
prediction:

> **P-Φc.1:** Under general anesthesia (Φ collapsed below Φ_c by clinical
> definition — propofol burst-suppression, sevoflurane at ≥1 MAC, or ketamine
> dissociation), no statistically significant psi effect should be observed in
> a previously-effective operator. Effect should collapse to chance levels
> within minutes of anesthetic onset and return within minutes of emergence.
>
> **Falsifier:** Any preregistered, multi-lab study showing a paired-operator
> effect (sender awake, receiver anesthetized, or vice versa) persisting at
> behavioral d ≥ 0.05 with high-channel EEG confirming anesthesia depth.

This is the BPR-specific test that distinguishes the framework from a
continuous-coupling null. No such study has been published.

### D. Parameter accounting

The six fitted exponents in χ_b — α=1.2, β=1.5, γ=0.8, δ=1.0, ε=1.3, k=2.0 —
are TUNABLE FLOATS, not derived rationals. Documented in
`LIMITATIONS_AND_FALSIFICATION.md` parallel to the existing 1.37→2 Casimir
accounting. Eq (5) is the most parametrized part of BPR and should be read
with that in the margin.

---

## 4. What this calibration costs and preserves

**Costs:**
- The sigmoid threshold is no longer the only supported coupling form.
- χ_max default (1e-3) is acknowledged as not anchored to data; the upper bound
  envelope is open to revision as preregistered work accumulates.
- The six fitted exponents are documented as model choices, not derivations.

**Preserves:**
- The mechanism: χ_b sources φ via Eq (5).
- The action term S_bio in the master action.
- The bridge to Hoffman's conscious-agent calculus in `conscious_agents.py`.
- The phase-transition prediction as a testable, named claim (P-Φc.1) rather than a derivation.

---

## 5. Limits of this calibration

- **Bidirectional sensitivity:** the literature bounds the operator → behavioral
  readout channel. Different readout substrates (Casimir-amplitude, quantum
  optics, MEMS) could in principle have very different gain factors. The
  empirical bound on χ_max·ε at QRNG ~10⁻⁵ per bit does not directly translate
  to a bound on the same product at sub-μm Casimir scales.
- **Publication bias asymmetry:** the skeptic-side bound assumes file-drawer
  problems are large; the proponent-side bound assumes they are small. Both
  bounds are conditional on that assumption.
- **No causal-direction test:** the literature cannot distinguish whether the
  small residual ganzfeld r ≈ 0.08 is psi or residual unblinded-judging artifact.

---

## 6. References

- Tressoldi & Storm (2024) Stage 2 Registered Report — F1000Research
- Milton & Wiseman (1999) Psychological Bulletin 125(4):387–391
- Storm, Tressoldi & Di Risio (2010) Psychological Bulletin 136(4):471–485
- Hyman (2010) Psychological Bulletin 136(4):486–490
- Cardeña (2018) American Psychologist 73(5):663–677
- Bem (2011) JPSP — "Feeling the Future"
- Wagenmakers et al. (2011) JPSP — Bayesian critique
- Galak et al. (2012) JPSP — preregistered failed replication
- Ritchie, Wiseman & French (2012) PLoS ONE
- Bem, Tressoldi, Rabeyron & Duggan (2015) F1000Research
- Kekecs et al. (2023) Royal Society Open Science — Transparent Psi Project
- Bösch, Steinkamp & Boller (2006) Psychological Bulletin 132:497–523
- Radin et al. (2006) Psychological Bulletin 132:529–532 (response)
- Maier, Dechamps & Plitsch (2018) Frontiers in Psychology
- Utts (1995) AIR report — SAIC remote viewing assessment
- Hyman (1995) AIR report — critique
- AIR/Mumford, Rose & Goslin (1995) — operational conclusion: terminated
- Marks & Kammann (1980/1981) Nature 292:177 — sensory cue invalidation
- Lawrence (1993) sheep-goat meta-analysis
- Cardeña & Marcusson-Clavertz (2011, 2020) — state-PCI correlations
- Mossbridge, Tressoldi & Utts (2012) — presentiment meta-analysis
- Watt & Cantiñho/Tressoldi prospective ganzfeld RR
- Jahn et al. (2000) PortREG three-lab consortium — primary replication failed
