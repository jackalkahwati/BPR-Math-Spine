# Common-foundation decision: current routes and amendment gate

2026-09-13. Draft work package 4 of the [foundation prerequisite campaign](foundation_prerequisites_2026-09-13.md). Mathematical arguments have independent support; the decision reasoning and citations have independent review, with scope corrections applied. New finite implementations have not completed overall verification. This is not approval of an amended theory.

## Decision

**Do not treat the existing ring, gauge target, internal-sphere chiral parent and supplied gravitational action as one derived unified theory.** Retain the exact results within their models. Stop construction branches that require the specific failed identifications below. Do not spend another campaign reproducing known negative results or increasing synthetic agreement counts.

The new theorems narrow two important routes; they do not establish that all BPR-like unification is impossible. No defensible common-foundation amendment meeting these requirements has been identified in the constructions reviewed for this campaign. This is an evidentiary stop condition, not a proof that no such amendment exists or a classification of minimal revisions.

## Dependency decisions

| Claimed connection | Evidence used | Disposition |
|---|---|---|
| Exact short-interval microscopic Gauss transformations on unrestricted Bose Fock space | [Bounded interval theorem](substrate_local_symmetry_2026-09-13.md), C>0 and at least two exterior sites | Nontrivial commuting bounded transformations of this support class are excluded. Constrained-subspace or extended transformations are different hypotheses, not covered by the theorem |
| 3D local site structure encoded in the ring with uniformly bounded dilation and fibers | [Graph-growth theorem](substrate_locality_dimension_2026-09-13.md) | Excluded for growing superlinear source neighborhoods under the stated uniform bounds. An abstract Hilbert-space packing is not a controlled local physical map |
| Given Bose occupation encoding supplies gauge-target dynamics | [Gauge-encoding derivation](substrate_gauge_encoding_2026-09-13.md) | Algebraic representation does not fix leakage or dynamics mismatch. Do not relabel it an invariant local gauge theory |
| Lone six-dimensional chiral Spin(10) parent supplies consistent physical matter | `bpr/chiral_parent_anomaly.py`, `bpr/chiral_content_constraints.py` | Existing irreducible p2 coefficient -s/90 blocks this parent without completion. Restricted lower-dimensional cancellation is insufficient; the identical opposite-chirality partner gives vectorlike reduction |
| Supplied local sources predict realistic flavor | `bpr/substrate_triplet_projection.py`, `doc/derivations/flavor_source_selection_2026-09-12.md` | The direct linear source image and aligned attractive-source model do not supply predictive realistic flavor. Composite/dressed source constructions are separate proposals; do not confuse the full generated algebra with the direct linear image |
| External source consistency gives unique gravity | [Source integrability](substrate_source_integrability_2026-09-13.md), `bpr/gravity_consistency.py` | Contact-completed identities survive, but no unique metric map/action or dynamical graviton follows. Bare/counterterm/cutoff and normalization ambiguities remain |
| More precise sampled response would establish BPR | [Observation gate](foundation_observation_gate_2026-09-13.md) | The specified BPR and conventional Bose models have identical predictions. Calibration or data acquisition alone cannot distinguish them |

These are scoped dispositions. The arbitrary bounded local theorem, full physical gauge constraints, graph maps and effective low-energy encodings are not interchangeable concepts. Likewise, internal Dirac zero modes are not automatically physical spacetime fermions.

## Optional proposal-development direction; no concrete amendment specified

The next approval request should target **one explicit spatial foundation first**, not add an anomaly-cancellation field and a metric independently onto the current ring. The candidate direction is a separately specified genuinely three-dimensional local bosonic graph Hamiltonian with conserved total number, treating the present ring as a controlled one-dimensional reference rather than the derived geometry of our world.

This ordering is a research-planning judgment, not a theorem-selected or uniquely preferred revision. Conserved total number would be a deliberate continuity assumption with the reference model. Alternative encoding classes remain unassessed and would need their own explicit observable and dynamical maps.

This would be a change of starting assumptions, not a consequence of this campaign. A three-dimensional graph by itself solves neither chiral matter nor gravity; no claim of minimality or sufficiency is made. No particular lattice connectivity, added interactions or operator assignments are adopted here, and no amended code is authorized.

Before requesting approval to construct that candidate, a proposal must actually specify:

1. The graph family, locality metric, site/link Hilbert spaces, self-adjoint Hamiltonian/action, couplings and symmetry constraints. “Use a 3D lattice” is not a complete model.
2. A vacuum/population selection rule and controlled large-system regime; assumptions must not masquerade as a derived vacuum.
3. Which gauge/matter operators are physical and which are supplied targets, with explicit support, products, commutators and dynamical-error estimates. Bosonic starting variables do not forbid emergence, but neither provide it automatically.
4. An anomaly-consistent proposed matter sector and physical spin/statistics construction, including mirror treatment if required. The current inconsistent parent cannot be inherited unchanged.
5. A testable geometric-dynamics mechanism: physical degrees of freedom, constraint algebra, positive-residue propagating tensor content in its claimed regime and universal coupling. Inserting a TT projector or curvature counterterm does not meet this requirement.
6. Parameter/input accounting, independent scale matching, falsification conditions and a distinguishing observable. No fit to the old benchmark targets is allowed as a substitute for derivation.

**Current proposal disposition: incomplete; construction branch stopped.** The reviewed record does not justify specifying a common action meeting these requirements, so no concrete revision is proposed for construction approval. The direction above is a prioritized proposal-development question, not an automatically scheduled research campaign. User approval of a future concrete proposal is necessary before fields, graph or interactions change.

## What survives a possible revision

The complete-ring implementations, scoped theorems, source-response identities, numerical failure records and comparison contracts remain valid reference work under their original assumptions. Graph-specific spectra, vacuum claims, propagation bounds, local-operator classification and observation maps cannot automatically be transplanted to a different graph or action. General algebraic tools may be reused only with their hypotheses rechecked.

The old chiral and gravitational constructions remain conditional target models until a common map and action are supplied. Their already-established anomaly/contact/normalization obstructions must be carried forward, not reset to “pending” merely because a new model is proposed.

Legacy cosmology normalization inconsistencies and stale registry labels beyond the four reconciled entries are recorded debt, not repaired physics. No sealed glueball/GUE/flavor benchmark is reopened or regraded by this decision.

## Experimental and numerical consequences

The observation gate ends this campaign's experimental branch with `no_distinguishing_prediction`, `calibration_missing` and `data_missing`. It does not authorize downloads, fitting, outreach or a generic data-loader project.

The prior sampled-response sign criterion remains unsatisfied. A prospective numerical amendment could use a stable generic three-node trapezoid reformulation: for `d=expm1(-nu*h)`,

`T = h/2 * [(s0+2*s1+s2) + 2*(s1+s2)*d + s2*d*d]`.

Its domain, roundoff contract and verification would need separate preregistration; it cannot guarantee every cancellation sign at fixed precision and must not replace a failed stimulus by its expected answer. No numerical revision is implemented here. Even a successful future repair would not change the identical-model discrimination result or derive missing physics.

## Completion requirements for this decision

Before marking this package complete: independent review of this reasoning and its citations; reconcile final theorem/API statuses and test evidence; verify no unapproved new physics or protected-code changes. Publish neither a TOE-completion claim nor a numeric percentage. No commit/push/merge is authorized by the present campaign approval.
