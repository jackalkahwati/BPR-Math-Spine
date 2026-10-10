# openai/math: how this repository uses it

2026-10-09. [openai/math](https://github.com/openai/math) is a catalogue of
719 mathematics manuscripts in 372 result families, produced by an internal
OpenAI model and released under the Apache License 2.0. Its README says the
results are at different stages of verification: not all have Lean
formalizations, and some unformalized results could have issues.

This repository **references** it, pinned to commit
`fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb` (`fd4aeeb2e`, retrieved
2026-10-09). The machine-readable record is
[openai_math_index.json](openai_math_index.json). It is validated by
`bpr/external_refs.py` and tested in `tests/test_openai_math_index.py`.

> References material from openai/math (https://github.com/openai/math),
> licensed under the Apache License 2.0. No upstream files are redistributed
> in this repository.

## Bottom line

**No openai/math result changes the status or limitation of any
unification-map node.** Every indexed entry is capped at `context_only`.

How the 372 families were reviewed:
- A triage pass covered every family.
- Seven candidates were read in depth.
- Four of those (102, 216, 223, 225) passed separate hypothesis and
  reliability reviews. A value review then found none of them load-bearing for
  any argument here.
- The rest are recorded as context, or as considered and not cited.

The review's most useful outputs were not citations:
- **Bookkeeping conventions**, adopted in schema 2 of the
  [unification map](../derivations/unification_map_2026-09-25.md) and in
  [HISTORY.md](../HISTORY.md):
  - a node's verification is recorded separately from its status;
  - premises are separated from motivating context, with a premise audit;
  - a release history records withdrawals and repairs.
- **Defects in this repository's own legacy modules** that came up while
  scoping the citations. They are fixed on their own evidence and are
  **not** credited to openai/math.

## Policy

1. **Reference only.** Never depend on openai/math: no Lake, pip or
   environment dependency, and nothing built, imported or executed from it.
   - Its Lean library is about 1.8 GB (122,458 `.lean` files, 30 direct
     dependencies, 23 patches). Its lakefile runs `run_cmd` and
     `post_update` hooks that clone and patch repositories while the build is
     being configured.
   - A search of the library found nothing this repository needs: no
     Bogoliubov, anomaly, bordism or Perron–Frobenius material.
2. **Clones live outside the repository and are untrusted data.** Read them
   with text parsing only, for example `python3 -I script.py <clone-path>`
   with the script kept outside the clone. Never open them with tools that
   execute their contents.
3. **Never copy upstream files, PDFs, abstracts or Lean code.**
   - Paraphrase statements in our own words (`statement_bpr`).
   - Storing a sha256 of an upstream file is fine.
   - If a file is ever copied, it must sit under `third_party/openai_math/`
     with the licence text, a `MODIFICATIONS.md` and any upstream `NOTICE`.
     The validator's rule 2 enforces this. Upstream has no `NOTICE` at the
     pinned commit.
4. **Cite a statement, never a family.**
   - Cite a specific theorem in a specific versioned manuscript, or a Lean
     comparator declaration, through a permalink at the pinned commit. Never
     use `blob/main`; the BibTeX in upstream READMEs uses `blob/main` and
     must be rewritten.
   - Always print the verification label and the classical or refereed
     fallback next to the citation. `citation()` renders the standard
     string.
5. **No status upgrades.** The map loader never reads the index, and no
   premise of any node may be an index entry. The `rigorous_input` cap needs
   all of:
   - a Lean main statement;
   - `bpr_check == built`, meaning this repository re-ran the comparator in
     an isolated sandbox, which it has never done;
   - a recorded fidelity check against the manuscript;
   - an active upstream status;
   - no unverified external dependencies.

   Below that cap, an entry used as an input or counterexample may touch only
   conditional or open nodes, or a named hypothesis.

## Verification labels

Labels describe the *cited statement*, not the family:

| label | meaning |
|---|---|
| `lean_main_statement` | the cited statement is a Lean comparator statement; `lean_scope_excerpt` records its scope |
| `lean_supporting_only` | a comparator exists but covers only supporting results |
| `manuscript_only` | no comparator covers it |

`verification_source` records where the comparator is listed:
- `formalization_yaml`: in `lean/formalization.yaml` `main_results`;
- `lean_docs_only`: only in `lean/docs/<family>.md`.

The two sources differ. 135 families appear in `formalization.yaml`, and 242
have a `lean/docs` page. Upstream's own "formalized" figures depend on the
denominator: 300 of 719 manuscripts, 242 of 372 families, 173 sources, 200 main
results. This repository computes no "% formalized".

Upstream marks `review.status: unchecked`: nobody upstream has confirmed that
the formal statements match the papers. A Lean label therefore never stands
in for a fidelity check.

## The headline-versus-formal-scope trap

A family's headline often claims more than its Lean comparator states,
including in the physics closest to this repository:

| family | headline | what Lean states |
|---|---|---|
| 267 | positive-temperature BEC and quantum depletion | ground-state condensation of the dilute hard-sphere gas only |
| 271 | Bloch's T^{3/2} law with exact coefficient | spontaneous magnetization ≥ S/4 at low temperature, d ≥ 3 |
| 090 | universal optimality, including the Sandier–Serfaty W energy | completely monotone potentials and the Cohn–Elkies certificate only |
| 215 | massive continuum limits and the O(3) particle pole | finite-volume exponential decay only |
| 260 | spacetime Penrose inequalities | a supporting gluing result |
| 261 | (spectral type) | the spectrum set only, not its pure-point type |
| 269 | Laughlin gap under disorder | the unperturbed gap inequality |
| 113 | perfect-matching counting | the sharp face-dimension bound is outside the Lean statement |

Rule 10 of the validator checks every `lean_*` label mechanically. The
cited statement must share anchor terms with the recorded Lean scope and
must use no term recorded as outside it.

## Citation template

`citation(entry)` renders:

> OpenAI, "<title>" (openai/math preprint, <date>; commit fd4aeeb2e;
> Apache-2.0), <result_ref>. Verification: <label> (…); <what BPR checked>.
> Unrefereed. Used as <relation>, capped at <usage_cap>. <permalink>.
> Classical or refereed: <fallback>.

Entries marked `considered_not_cited`, such as 003, the quasi-Riemann
zero-free half-plane, refuse to render a citation. 003 would be a landmark
result if correct, but no argument here uses it.

## What is indexed

| entries | relation | notes |
|---|---|---|
| 102 (optimal Max-Cut hardness) | context | Lean main statement; worst-case ceiling for the legacy `bpr/optimization.py` heuristic; classical fallbacks 16/17 (Håstad; TSSW) and α_GW under UGC (KKMO; MOO) |
| 216 (three planar-XY / BKT statements) | context | manuscript only; reaches the speculative `BKTCoherenceTransition` only by universality |
| 223 (Z² bond-percolation exponents) | context | manuscript only; the same values are classical for triangular-lattice site percolation |
| 225 (six-vertex GFF at c = 1) | analogue | manuscript only; depends on unverified DKLM2026 extensions; the 2D RK point is tangential to the 3D emergent photon |
| 267, 271 | context for `thermodynamic_condensation` | Lean labels for the narrow statements above; neither model is BPR's lattice Bose-Hubbard problem |
| 213, 214, 219, 281, 284, 113, 115, 098, 007, 215, 090 | context | single-agent triage, not adversarially verified (`context_only_unverified`) |
| 003 | none | considered, not cited |

## Triaged, no match

So that they are not re-triaged, these sections were read family by family
at the pinned commit and found to have no hook in this repository beyond the
entries above:
- number theory (28 of 31 families);
- algebraic and complex geometry (35 of 36);
- real and complex analysis (all 16);
- convex and metric geometry (13 of 15);
- theoretical computer science (36 of 40);
- dynamical systems (10 of 12);
- combinatorics (all 37);
- algebra (17 of 18);
- probability and statistical mechanics (18 of 29);
- mathematical logic (all 6);
- group theory (all 14);
- mathematical physics (18 of 25);
- operator algebras (all 19);
- topology (17 of 18; none computes Spin bordism of a classifying space, so
  `global_anomalies` gains nothing);
- functional analysis (10 of 11);
- differential geometry (28 of 29);
- partial differential equations (12 of 16).

A completeness check then re-examined the open, conditional and obstructed
nodes and found no further family for: `thermodynamic_condensation`,
`continuum_equilibrium_gate`, `emergent_gauge_field`,
`nonabelian_gauge_field`, `lattice_chiral_fermions`,
`common_limiting_speed`, `metric_dynamics`, `vacuum_energy`,
`string_embedding`, `brane_positions`, `ring_locality_obstruction`,
`spin10_breaking`, `flux_selection` and `cc_self_tuning`.

## Adding an entry

1. Read the theorem in the manuscript source at the pinned commit. Record
   `result_ref`, a paraphrased `statement_bpr`, `not_covered`, and the
   `source_file` with its sha256.
2. Check Lean coverage in both `lean/formalization.yaml` and
   `lean/docs/<family>.md`. If a comparator covers the statement, record:
   - the config and challenge file;
   - the declarations;
   - `permitted_axioms`;
   - the challenge sha256;
   - `lean_scope_excerpt`;
   - `lean_scope_terms` and `outside_lean_scope`.
3. Choose `relation`, `applicability` and `usage_cap` conservatively, and
   name a classical fallback if one exists.
4. Run `python3 -m pytest tests/test_openai_math_index.py`.

## Moving the pin

1. Fetch the new commit into a separate directory outside this repository.
2. Run `python3 -I scripts/check_openai_math_pin.py <clone> --index
   doc/external/openai_math_index.json`. It parses text only and reports,
   for indexed entries:
   - withdrawals and new versions;
   - Lean coverage added or removed;
   - changed statement hashes and changed permitted axioms;
   - a new `NOTICE` file.

   Upstream removes withdrawn papers from `CONTENTS.md` without any marker;
   only `history.md` and the preprint README record them. The checker
   detects text changes, not mathematical errors.
3. Update the entries, then `commit`, `commit_short` and `retrieved`, and
   record the move in [HISTORY.md](../HISTORY.md) under "External-input
   changes".

## Risks recorded

- **The source.** It is AI-generated, unrefereed and fast-changing. The
  2026-10-07 release alone recorded 3 withdrawals, 14 proof repairs and 13
  citation updates.
- **Lean labels:**
  - the headline and the formal scope often diverge;
  - upstream statement fidelity is unchecked;
  - the trust base is Lean and Mathlib at d13f23b.
- **Halo effect.** "Lean-verified" next to a physics claim invites
  over-reading, which is why labels sit on statements, not families, and why
  every entry is capped.
