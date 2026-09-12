# Closed and Deprecated — Negative Findings Registry

> **Status:** Single navigation point for everything in BPR that has been
> determined NOT to work or has been honestly closed as a negative finding.
> Existing for two reasons: (1) reviewers shouldn't have to scrape commit
> history to find what didn't pan out, (2) the framework's credibility
> depends on documented negatives as much as positive claims.
>
> **Last updated:** 2026-09-12 (gauge repair, classical alignment obstruction and conditional quantum-source diagnostic; earlier flavor-foundation withdrawals retained). The largest single
> entry: the BPR 1.0 "theory of everything" headline claim is **WITHDRAWN**
> (§1, first row). The current claim is a flavor-sector organizing framework
> plus a proposed, untested particle sector (`doc/BPR2_PATH_B_NONABELIAN_GAUGE.md`).

## 1. Structural negatives (architectural problems with no fix in current scope)

| Item | Status | Where | Notes |
|---|---|---|---|
| BPR 1.0 "theory of everything" headline claim | **WITHDRAWN** | `doc/BPR2_PATH_B_NONABELIAN_GAUGE.md`; status page `viz/bpr2-status.html` | The sealed glueball benchmark (row below) falsified the 1.0 particle sector structurally: scalar/Abelian boundary content provably cannot host a light parity-odd state or non-Abelian gauge matter. The claim was narrowed, not defended: BPR 2.0 = flavor-sector organizing framework + a proposed, untested particle sector (Postulate 0d, PROPOSED — not merged into the frozen core). Flavor predictions numerically unchanged (verified exactly, M3). |
| SU(3) color-bundle index step in l_t=283 | **WITHDRAWN** (2026-09-10) | `doc/derivations/color_bundle_index.md`; `bpr/flavor_foundations.py`; `bpr/qcd_flavor.derive_l_modes` | Claimed index(D_color)=c₁=n_gen=3 on S². For an ordinary SU(N_c) bundle det E is trivial, tr F=0, so c₁=0 and the twisted spin Dirac index is 0. A nonzero index requires an additional U(1) twist O(q) (index = N_c·q), which is not derived from (p,z); and an index counts net zero modes, it does not add +3 to an angular-momentum label. The formula l_t=(z²−1)(z+n_gen+2−N_c)+n_gen=283 is retained as phenomenology, now labelled CONJECTURAL. Numbers unchanged. |
| c=1 compact-boson proof of exactly three generations | **WITHDRAWN** (2026-09-10) | `doc/derivations/generations_from_CFT.md`; `bpr/boundary_topology.generation_count_from_topology`; `bpr/neutrino.number_of_generations` | Conformal spin of the untwisted momentum/winding lattice is s=mn, an integer, so it contains no spin-1/2 operators; the "3+1" SO(3) decomposition of four labels was asserted without a group action; V₍₁,₁₎·V₍₁,₁₎→V₍₂,₂₎ is a primary, not a descendant, so fusion does not exclude ℓ≥2; 10/3+1/16=163/48 not 167/48; the orbifold/4th-generation exclusion was blanket. Round-S² ℓ=1 multiplicity 3 is a spatial spin-1 fact, not a family count. n_gen=3 is an EMPIRICAL INPUT; the ℓ=1↔families identification is OPEN; no topology-only exclusion of a 4th generation. Conditional replacement: Dirac zero modes of O(q) on S² number |q|, with q an input. |
| "Coherence GUARANTEED / ROBUST" condensate verdict | **DOWNGRADED to CONDITIONAL** (2026-09-10) | `bpr/condensate_regime.py`; `tests/test_condensate_regime.py` | The exact result is a classical fixed-norm minimum of the frozen ring model; it does not compute the quantum vacuum of the interacting substrate. T*≈0.086C comes from Rayleigh–Jeans weak-coupling occupation on an assumed dispersion, cutoff l~√p and unit density, and the substrate temperature is not derived. Coherence is conditioned, not established. Test now fails if GUARANTEED/ROBUST wording returns. |
| Path B `{r,r^-1,s}` electric Hamiltonian | **GAUGE-INVARIANCE CLAIM WITHDRAWN** (2026-09-12) | `bpr/gauge_heat_kernel.py`; `doc/derivations/toe_constructive_extension_2026-09-12.md` | The generator sum is not central and fails one endpoint gauge symmetry. Quoted scalar Casimirs are trace averages. A NEW conjugacy-averaged operator has those scalar eigenvalues and a matching heat kernel. Existing Wilson simulations are a different model; a scalar beta-to-lambda identification is unsupported. No historical numerical output or sealed benchmark status changes. |
| Source-driven monopole flavor prototype | **CONDITIONAL, NOT PREDICTIVE SELECTION** (2026-09-12) | `bpr/chiral_flavor_prototype.py`; constructive note | An added sphere, chosen flux and external scalar sources define overlap Yukawas for an assumed 4D chiral EFT. Zero source gives degeneracy. Arbitrary low harmonics retain arbitrary Hermitian matrix freedom; the quadratic response action does not select the sources. No consistent 6D origin, flux selection, or physical mass prediction is established. |
| Attractive fixed-occupation flavor-source selection | **CONDITIONAL ALIGNMENT OBSTRUCTION** (2026-09-12) | `bpr/flavor_source_selection.py`; [derivation](derivations/flavor_source_selection_2026-09-12.md) | A NEW mean-field action selects coherent monopole occupations globally at fixed trace, rather than prescribing angular sources. Positive inter-sector coupling forces alignment: nondegenerate overlap spectra but identity absolute mixing and zero CP quartet. At zero coupling relative orientation is flat, not predicted. Equal filling gives degeneracy. The occupation/action assumptions and a quantum vacuum remain underived; no extra parameters were added to rescue mixing. |
| Classical source selection promoted to a quantum vacuum | **NOT ESTABLISHED; CONDITIONAL OPERATOR DIAGNOSTIC** (2026-09-12) | `bpr/quantum_flavor_sources.py`; [operator derivation](derivations/quantum_flavor_sources_2026-09-12.md) | In a separately stipulated six-mode fermion Hamiltonian, positive exchange gives a rank-five 1+1 ground space; zero exchange gives rank nine. Operator squares differ from squared mean densities by variance terms. Normal ordering shifts conserved-number sectors by a one-body counterterm. Neither statistics, chosen population nor temporal scalar elimination follows from the substrate; the valid classical theorem is not a quantum-vacuum derivation. |
| BPR 2.0 particle-physics bridge | **OPEN — no partial credit** (2026-09-10) | `doc/BPR2_PATH_B_NONABELIAN_GAUGE.md` §3.0; `viz/bpr2-status.html`; README | A gauged dihedral quantum double gives 2D non-Abelian anyons. That does not establish 4D SU(3) color, chiral SM matter, anomaly cancellation, or a spectrum. "Flavor predictions unchanged" is numerical preservation under a kinematic relabeling, not survival under the interacting M2 dynamics. |
| M4 parity-odd (A2) loop operator in `glueball_channels_mc` | **DEFECTIVE, superseded** (2026-09-10) | `bpr/gauge_mc_fast.py`; `tests/test_gauge_mc_fast.py`; addendum in `doc/derivations/path_b_open_problems_2026-08.md` §2 | The 8-link "L" loop was not a contiguous path, so its character was not gauge invariant (changes O(1) under a gauge transformation); and the notched-square shape is achiral, so its parity-odd combination vanishes identically for real characters. The earlier "A2 gate fails at laptop statistics" was therefore vacuous. Replaced by chiral polyomino loops, rotation-symmetrised and reflection-antisymmetrised, verified gauge invariant and parity-odd. A1/B1 operators were correct. Benchmark v3 remains sealed. |
| Flavor-label operator scan (preregistered) | **NEGATIVE** (2026-09-10) | `bpr/flavor_label_scan.py`; `doc/derivations/flavor_label_scan_2026-09.md` | None of eight frozen-ingredient operators (S² Laplacian, O_h/O-invariant harmonics, CCR selection, octahedral shell graphs, D_n Casimirs, winding-shifted spectrum) reproduces any flavor sector; no candidate beats chance after Bonferroni correction over 16 comparisons. Two sub-threshold coincidences recorded (30 = 5·6 and 210 = 14·15 are round-sphere Laplacian eigenvalues). Labels remain CONJECTURAL; next step is constructive, not a scan. |
| Original `RPSTHamiltonian` rank-1 problem | **STRUCTURALLY INCOMPATIBLE** | `bpr/rpst/hamiltonian.py`; locked in by `tests/test_gue_riemann_honest.py` | H = outer(leg, leg) is rank-1 by construction — one nonzero eigenvalue. Cannot show GUE level statistics. |
| Riemann/GUE conjecture for BPR | **DOWNGRADED** | `doc/conjectures/riemann_connection.md` | Originally "Tier 2 with KS p=0.92 numerical support" — not reproducible by any operator. Final survey verdict: no natural prime-modular Z_p Hermitian or unitary operator class reproduces Wigner-Dyson statistics. |
| Legendre Hankel/Multiplicative/Circulant Hermitian | **GAUSS-SUM DEGENERATE** | `bpr/substrate_hamiltonians.py`; tested in `tests/test_substrate_hamiltonians.py` | All ≤ 3 distinct eigenvalues (Hankel/Circulant) or rank-1 (Multiplicative). Cannot do level spacing. |
| Discrete Berry-Keating 1D and 2D | **POISSON (integrable)** | `bpr/substrate_hamiltonians.py` | Full rank, p (or p²) distinct eigenvalues, but Poisson level spacings rather than GUE. K-S D_Poisson ≈ 0.14 across primes 211–1009. |
| Hannay-Berry quantum cat map (unitary) | **POOR FIT to CUE** | `bpr/substrate_hamiltonians.py:quantum_cat_map_spectral_statistics` | Final attempt in the unitary class. K-S D ≈ 0.32 — much worse than the random-matrix sanity control (D = 0.07). Spectrum has arithmetic structure, not generic Wigner-Dyson. |
| Glueball sector (all branches) | **CLOSED** | `bpr/glueball_gates.py`, `bpr/phason_boundary_modes.py`; sealed protocol `doc/GLUEBALL_BENCHMARK_V1.md` | Blind test: targets locked before any spectrum. Positives: bound states exist (Gate 1); {0⁺⁺, 2⁺⁺} families with C=+ emerge from two J=1 quasiparticles. Fatal: extra light 1⁻ at 0.5×M(0⁺⁺); light 0⁻⁺ forbidden by parity/bose selection rules (lightest is (2,3,4) triple, ratio ≥ 3.7 vs target 1.497); 2⁺⁺/0⁺⁺ ∈ [0.87, 1.00] vs 1.387; no γ rescues it. All three loopholes then dispositioned: interactions can't beat symmetry-protected parity rules; lump branch shares the scalar parity lock; phason sector has NO propagating boundary branch (root-symmetry theorem + full-range scan — relaxation, not particles). Lessons kept as constraints: glueball spectrum fingerprints *vector* constituents; BPR lacked a derived state-selection principle — since derived (Z_p neutrality superselection, `bpr/zp_selection_principle.py`): confines bare quanta, yields Z_p baryons, but quasiparticles are exactly neutral so the 1⁻ defect and this closure stand. **Path B follow-up (2026-08, status pointer — this closure is unaffected):** the non-Abelian revision (Postulate 0d, PROPOSED) re-attacks the sector; Benchmark v3 remains SEALED. A PROPOSED physical-coupling derivation (`bpr/physical_lambda.py`) places the substrate ~5× into the deconfined phase — if it stands, the confining glueball-analog spectrum describes a phase the substrate does not occupy, and the particle-sector interpretation routes through anyons instead. See `doc/derivations/path_b_open_problems_2026-08.md`. |

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
| ln(p)/z ≈ β_c numerical coincidence | `bpr/physical_lambda.py` | RECORDED, NOT USED — lands on the measured transition for all 4 classes (0.88–0.99×), but no frozen derivation supports the z-division; logged as retrofit *risk*, flagged for future work |

All flagged in their respective docstrings; this doc consolidates the
list for reviewers who want to see the honest parameter accounting in
one place.

## 5. Speculative sectors clearly bracketed (NOT closed — separated)

These are NOT negative findings, just sectors explicitly bracketed from
the flavor-sector evaluation because their evidence requirements are
separate. They remain valid framework content with their own evidence
streams:

- **Phason topological propulsion** — UNDECIDABLE at 31 orders below
  experimental sensitivity (`bpr/phason_sector.py:phason_defect_lift_budget`).
  Two honest attacks on the gap, both resolved (`bpr/phason_coupling.py`):
  (1) *Mode census* — **NEGATIVE/CLOSED**: the internal space is a torus
  T^{d_⊥} whose only nonzero homotopy is π_1 = ℤ^{d_⊥} (higher homotopy of a
  torus vanishes), so the known dislocation sector is the *complete*
  topological-defect content — there is no missed lift channel. (2) *Coupling
  derivation* — **OPEN, retrofit-flagged**: reaching ε_required ≈ 6×10⁻³⁶
  needs ~7 powers of 1/p; no derivation predicts that exponent, and the
  (1/p)⁷ numerical coincidence is locked as a retrofit *risk*, not a result
  (tripwire test: `test_no_hypothesis_is_marked_derived`). The 31-order gap is
  an *undecidability* gap (the J⁴ reservoir is ~10³⁵× the lift need), not an
  energy shortfall. **Lagrangian attempt** (`bpr/phason_defect_lagrangian.py`,
  `doc/PHASON_DEFECT_LAGRANGIAN.md`): writing the LRT quasicrystal-elasticity
  action pins the energy convention (c=2, quadratic F) and the protection count
  (τ=1, single winding direction) from structure, fixing the *bulk* exponent
  k_bulk = 2(d⊥−1) = 6 for the 9-fold class. The odd power 7 would come from a
  single-site defect-**core** factor (g_core ~ 1/p) — physically natural but a
  hypothesis, not derived. Net: 2 of 3 ambiguities pinned; the gap is reduced to
  one number (the core coupling), still open.
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

- The headline claim itself was withdrawn when a sealed blind benchmark
  falsified it (TOE → flavor-sector framework) — the framework applied its
  own kill rules to its own biggest claim
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
- `doc/BPR2_PATH_B_NONABELIAN_GAUGE.md` — the PROPOSED non-Abelian revision responding to the glueball closure (§1); Benchmark v3 SEALED
- `doc/derivations/path_b_open_problems_2026-08.md` — M2 phase located, M4 machinery gated, √210 wrinkle resolution, physical-λ proposal (all PROPOSED-tier, none frozen)
