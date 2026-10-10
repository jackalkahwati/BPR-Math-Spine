# Release history

This file lists, by release, every change to the status or claim of a node in
the [unification map](derivations/unification_map_2026-09-25.json). Its format
follows the release log of the openai/math repository: dated entries, each
split into the same fixed classes. Only the format is borrowed.

Each entry names the **pre-change commit**, the last commit before the change,
and links to the repository as it was then, so earlier states stay readable.
The machine-readable record is the `revisions` list in the map JSON. Its
`node_changes` give, for each change, the node, its old and new status, its old
and new claim hashes, and the reason.
[status_snapshot.json](derivations/status_snapshot.json) stores every node's
current status and claim hash. `tests/test_unification_map.py` fails if either
changes without a matching `node_changes` entry.

Every entry uses these classes:
- **Withdrawals and downgrades**: a claim withdrawn or narrowed, or a node
  moved to a weaker status.
- **Repairs after review**: a result corrected after a review found an error.
  Wording changes that leave a claim's content alone are not listed.
- **Status upgrades**: a node moved to a stronger status.
- **External-input changes**: a node's use of an external result added,
  changed or removed.
- **Formalizations**: machine-checked statements or proofs. BPR has none.

New nodes are listed separately. An addition is not an upgrade.

**Adding an entry.**
1. Change the node in the map JSON.
2. Append a revision with `date`, `pre_change_commit`, `summary` and
   `node_changes`: one entry per changed status or claim, with its reason.
3. Add a dated section here.
4. Refresh `status_snapshot.json` only at a release, and raise its
   `revisions_included` in the same commit.

No entry reports a percentage of nodes verified or formalized.

## 2026-10-09: map schema 2

Pre-change commit
[16ca3ca](https://github.com/jackalkahwati/BPR-Math-Spine/tree/16ca3cae9a406d2c6ef4da31f2b5b0f1bf9014ec).

**Withdrawals and downgrades.** None. No node's status changes.

**Repairs after review.**
- `thermodynamic_condensation`: open → open; only the claim text changes. The
  claim now covers the superfluid region only: outside the Mott lobes, for
  example at non-integer filling, or at integer filling with g/C below the
  lobe tip. At integer filling and large g/C the ground state is expected to
  be a Mott insulator. At unit filling in 3D the transition is at
  (g/C)_c ≈ 29.3, a numerical result (Capogrosso-Sansone, Prokof'ev and
  Svistunov 2007). The limitation now names reflection positivity
  (Dyson–Lieb–Simon 1978; Kennedy–Lieb–Shastry 1988) as the method behind the
  half-filling hard-core result. The problem was found by a single agent
  during the integration review. openai/math is not the basis for this
  change; the sources named here are.

**Status upgrades.** None.

**External-input changes.** None. An external cross-reference index,
[doc/external/OPENAI_MATH.md](external/OPENAI_MATH.md), is added as context
only. No node depends on it, and no node's data refer to it.

**Formalizations.** None. BPR has no Lean formalization, and no node has a
formal verification level.

**Bookkeeping (no status change).**
- Map schema 2. `depends_on` is split into premises and context, each link
  with a reason. Nodes name their hypotheses. Every node has a verification
  record, kept separate from its status.
- The premise audit now runs as a test, and its findings on the schema-1
  graph are resolved:
  - the BPR-6D chain through `bpr6d_architecture` is context (section 9 of
    the architecture decision);
  - `common_limiting_speed` names `emergent_gauge_field` as a hypothesis for
    its photon clause;
  - `flavor_formulas` is context for `flavor_source_selection`.
- Standard "Independent review" headings are added to three notes:
  substrate_local_symmetry, flavor_source_selection and
  foundation_observation_gate.
- This file and `status_snapshot.json` are added. The ten earlier revision
  entries are backfilled from git history with pre-change commits and node
  changes.

## 2026-10-01: Phase 2b

Pre-change commit
[b207c17](https://github.com/jackalkahwati/BPR-Math-Spine/tree/b207c17bb3ca15cf0d5d982098e5256dcdbf3c9f).

**New nodes.** `kinetic_corrections` (phenomenological_fit).

**Withdrawals and downgrades.** The first version's verdict (commit d604c07),
that next-to-leading corrections do not rescue the pinned branes, is
withdrawn. See the repair below.

**Repairs after review.**
- The review found an omitted one-derivative brane vertex. It is allowed by
  the same J_z rule and is of the same order.
- With that vertex, pinning is not excluded at next-to-leading order:
  χ² = 4.5 at |δ| ≤ 0.3, the top of its natural range. Flavour is still not
  predicted.
- The `kinetic_corrections` claim was rewritten; its status is unchanged.
- The review's lower pinned point lowered Phase 2's pinned best fit from
  χ² ≈ 124 to ≈ 120. The `fermion_fit` claim changed accordingly; its status
  is unchanged.

**Status upgrades, external-input changes, formalizations.** None.

## 2026-09-27: Phases 1d and 2

Pre-change commit
[0b91547](https://github.com/jackalkahwati/BPR-Math-Spine/tree/0b915479aa384fdfa4d63dfcbe09d741fb2b26c3).

**New nodes.** `brane_positions` (conditional) and `fermion_fit`
(phenomenological_fit).

**Withdrawals and downgrades.** Before review, the Phase 1d draft reported
two claims that the review withdrew:
- kill check 11 (brane positions stabilized) as passed. It is now
  conditional, and the node entered the map as conditional;
- "no loss of fitting ability". In fact fixed positions reach only a proper
  subset of the Yukawa pairs.

**Repairs after review.** In Phase 2 the review found two blockers: two
missing physical phases, and a stored pinned point that was not exactly
reachable. Repairing them reversed the first version's numbers (χ² 36 and
609).

**Status upgrades, external-input changes, formalizations.** None.

## 2026-09-26: Phase 1

Pre-change commit
[51d38c9](https://github.com/jackalkahwati/BPR-Math-Spine/tree/51d38c9e9fedb3e5c5b91de6c6a7163e0830e33c).

**New nodes.** `bpr6d_minimal_model` (conditional).

**Withdrawals and downgrades.** Version A, with an independent Higgs copy on
each brane, fails kill checks 4 and 8 and is recorded as failing.

**Repairs after review.** The review's blocker was that brane Higgs copies do
not mix at tree level. The adopted model is version B, a single bulk F-charge
−6 Higgs.

**Status upgrades, external-input changes, formalizations.** None.

## 2026-09-26: independent-review repairs for rounds 6–10

Pre-change commit
[96bd2c2](https://github.com/jackalkahwati/BPR-Math-Spine/tree/96bd2c2f8cc47ca6d3dadfee53b27af6ab433f25).

**Withdrawals and downgrades.** `bpr6d_predictions` stays conditional, but
its claim is narrowed. The claim that BPR-6D predicts a QCD axion near the
compactification scale is withdrawn. With the F-charged Higgs included, a
viable axion needs a supplied F-charged singlet; with only the Higgs doublets
the axion is an excluded PQWW axion.

**Repairs after review.** No status changes. The repairs changed limitations
and notes:
- string embedding: the "intrinsically non-supersymmetric" fork was an
  artifact of T=0; on T=1, supersymmetric analogues can force charges into 3ℤ;
- Spin(10) breaking: the instability of every quantized flux is now the
  primary obstruction;
- Yukawa mechanisms: a brane Higgs has J_z charge c = s_h + 3, so a brane
  scalar never couples (the earlier c = 1 identification was wrong);
- cosmological constant: flatness pins the deficit globally, and citations
  and supporting text are corrected.

**Status upgrades, external-input changes, formalizations.** None.

## 2026-09-26: Round 10

Pre-change commit
[acb3271](https://github.com/jackalkahwati/BPR-Math-Spine/tree/acb3271bf1e9f5f43ca7139d54fbb4875ab7cc45).

**New nodes.** `bpr6d_predictions` (conditional).

**Withdrawals, repairs, upgrades, external-input changes, formalizations.**
None.

## 2026-09-26: Rounds 8–9

Pre-change commit
[93913ec](https://github.com/jackalkahwati/BPR-Math-Spine/tree/93913ec52e020b9e4232e82130c1f44ea70fca86).

**New nodes.** `cc_self_tuning` (obstructed). The `yukawa_sector` limitation
is extended with the brane and bulk-vector mechanisms (no hierarchy is
predicted); its status is unchanged.

**Withdrawals, repairs, upgrades, external-input changes, formalizations.**
None in this entry. The review repairs for these rounds are listed under
"independent-review repairs for rounds 6–10".

## 2026-09-26: Rounds 6–7

Pre-change commit
[dbafa21](https://github.com/jackalkahwati/BPR-Math-Spine/tree/dbafa2189a419c7a0d156fa4b5d174b349f13877).

**New nodes.** `string_embedding` (open) and `spin10_breaking` (obstructed).

**Withdrawals, repairs, upgrades, external-input changes, formalizations.**
None in this entry. The review repairs for these rounds are listed under
"independent-review repairs for rounds 6–10".

## 2026-09-26: Round 5

Pre-change commit
[80029d3](https://github.com/jackalkahwati/BPR-Math-Spine/tree/80029d3803ab2b5cd0fa94cc4da464ac4e63380e).

**New nodes.** `global_anomalies` (exact_theorem).

**Repairs after review.** The review found a ℤ/2 in Ω₇ for the twisted
Spin ×_{ℤ₂} Spin(10) × U(1) form. The claim was extended (commit a3c5655): the
anomaly on that ℤ/2 is trivial, by a reduction argument. The status is
unchanged.

**Withdrawals, upgrades, external-input changes, formalizations.** None.

## 2026-09-26: Round 4

Pre-change commit
[b506441](https://github.com/jackalkahwati/BPR-Math-Spine/tree/b506441759e8ee643fc8532cc679b5ff6899e028).

**New nodes.** `gs_quantization` (exact_theorem).

**Status upgrades.** `family_count`: stipulated → conditional. Dirac
quantization of the minimal completion's Green–Schwarz couplings makes the
family number a multiple of three, at the characteristic-class level. Three
itself still needs minimal choices.

**Repairs after review.** The `gs_quantization` claim was widened from
unimodular to integral 2-form lattices, and its U(1) normalization anchored
to Park–Taylor. Its limitation now says the minimal completion is unique
only in class 𝒞.

**Withdrawals, external-input changes, formalizations.** None.

## 2026-09-26: Round 3 (BPR-6D)

Pre-change commit
[f82453e](https://github.com/jackalkahwati/BPR-Math-Spine/tree/f82453e4268d0331db5473a87fce6bbac1cab6b6).

**New nodes.**
- stipulated: `bpr6d_architecture`, `six_d_gravity`;
- exact_theorem: `flux_vacuum`, `family_symmetry`;
- obstructed: `flux_selection`, `yukawa_sector`.

**Withdrawals and downgrades.**
- The implicit claim that a boundary substrate generates spacetime, gauge
  fields, matter and gravity is withdrawn (architecture decision, section 5).
- The `common_limiting_speed` claim is narrowed to species that emerge from
  the lattice substrate. It stays obstructed.
- The substrate nodes are reclassified in their limitations as analogue or
  regulator.

**Repairs after review.** These were note-level repairs; no map claim
changed:
- flux vacuum: the SU(2) isometry bosons are massless at tree level, not
  near 10¹⁷ GeV;
- architecture decision: option 3 does not meet every section-7 link.

**Status upgrades, external-input changes, formalizations.** None.

## Before the revisions list

The map was created on 2026-09-25 at
[a469921](https://github.com/jackalkahwati/BPR-Math-Spine/tree/a46992130465a96c008f7f2f4aee7797e2e9fcf3),
with baseline commit c27470f. Some changes on the same day predate the
`revisions` list:
- the chiral-completion correction after review;
- in Round 2, `common_limiting_speed` open → obstructed and
  `emergent_gauge_field` open → conditional;
- the new nodes `link_boson_substrate` and `nonabelian_gauge_field`.

They are recorded in the derivation notes and the map's prose, not as
`node_changes`, and are not backfilled.
