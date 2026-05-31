# Moral Phenomenology — Privation Framework

> **Status:** Interpretive overlay, not derivation.
> **Scope:** Maps BPR's structural machinery (identity winding, permission
> field, salience, cooperative winding, trajectory utility) onto the classical
> Augustinian-Thomistic *privatio boni* theory of good and evil.
> **Companion module:** [`bpr/functional_architecture.py`](../bpr/functional_architecture.py)
> **Companion doc:** [`CONSCIOUSNESS_EMPIRICAL_CALIBRATION.md`](experiments/CONSCIOUSNESS_EMPIRICAL_CALIBRATION.md)

---

## 1. Position summary

BPR's structural objects encode a precise ontological asymmetry between
*positive coupling* and *its absence or corruption*. There is a φ field and
a χ_b coupling that sources it. There is no anti-φ field and no anti-χ_b
coupling. Vacuum is the absence of excitation, not the presence of a
negation. This asymmetry is the structural skeleton of the patristic
*privatio boni* theory of evil:

- **Good** has positive ontological status — it is the active configuration
  that BPR's machinery describes (Φ above threshold, χ_b sourcing φ, identity
  winding stable, cooperative alignment intact, permission field gated,
  salience concentrated, trajectory utility consistent).
- **Evil** has no positive ontological status. It is the *privation*,
  *blockage*, *corruption*, or *turning-away* of the goodness configuration.
  The framework has no object that represents "anti-good"; it has only the
  failure modes of good.

BPR does not derive this position; it is structurally compatible with it
and structurally incompatible with manichaean two-substance pictures of
good and evil.

---

## 2. Scripture and patristic anchors

The mapping is to the broadly Augustinian-Thomistic synthesis, which is
shared across most of the Western Christian tradition and has analogs in
Eastern Christian, Jewish, and Islamic thought.

| Source | Position | Key text |
|--------|----------|----------|
| Genesis 1:3 | Light precedes the rest of creation | "Let there be light" |
| John 1:1–4 | Logos as generative ordering principle | "In him was life, and the life was the light of men" |
| 1 John 1:5 | No darkness in the source | "God is light, and in him is no darkness at all" |
| Acts 17:28 | Participation, not separation | "In him we live and move and have our being" |
| Augustine (Conf. VII, Enchiridion) | Evil as privation of good | "What is called evil... is the privation of good" |
| Aquinas (ST Ia q.48) | Evil has no positive being | "Evil cannot exist on its own; it is a privation" |

The shared structural claim: good is fundamental and positive; evil is
derivative and parasitic; evil cannot create, only corrupt or unmake.

---

## 3. Object-by-object mapping

### Source consciousness (Light) ↔ active χ_b coupling

| Patristic concept | BPR object | Reference |
|-------------------|------------|-----------|
| Logos / source | universal φ-substrate (interpretive) | `bpr/phason_sector.py`, master action |
| Light | active χ_b sourcing φ above threshold | `bpr/information.py` Eq (5) |
| Soul / personal identity | identity winding W[φ_mind] | `bpr/functional_architecture.py:241` |
| Participation in the source | χ_b coupling above Φ_c | `bpr/information.py`, IIT integration |
| Communion / koinonia | cooperative winding alignment | `bpr/collective.py:239` |
| Conscience / discernment | permission field eligibility functional | `bpr/functional_architecture.py:399` |
| Attention / nous | salience field A(x,t) | `bpr/functional_architecture.py:615` |
| Conformity of will to good | trajectory utility J(γ) on stable values | `bpr/functional_architecture.py:741` |

Every term on the left is a *positive* configuration of the BPR objects on
the right. The framework has no negative versions of any of these.

### Darkness / evil ↔ failure modes of those objects (privation)

| Patristic concept | BPR object failure mode |
|-------------------|--------------------------|
| Sin as missing the mark (hamartia) | trajectory utility deviation from declared values |
| Hardness of heart | permission field driven closed (sealed against the source) |
| Disordered loves | salience field captured by attractors that don't sustain coherence |
| Sloth (acedia) | Φ below threshold; coupling drops to zero (privation) |
| Pride / autonomy from source | cooperative winding withdrawn from collective field |
| Identity corruption | W[φ_mind] mutation from stable topological sector |
| Bondage of the will | permission field driven open externally; identity winding hijacked |
| Spiritual death | decoherence rate Γ exceeds coherence rate; χ_b coupling collapses |

**Critical structural observation:** every entry on the right is either a
*reduction to zero*, a *blocking*, a *deformation*, or a *withdrawal*.
None is the presence of a positive opposite. The framework cannot represent
"anti-light" because there is no field that would carry it.

---

## 4. Why this mapping is non-trivial

BPR was not designed to recover the privatio boni structure. It recovers
it incidentally because field theories have this ontological asymmetry by
construction — fields have positive excitations above a vacuum, not
positive and negative substances.

The non-trivial part is that BPR's *moral-phenomenology* objects (identity
winding, cooperative winding, permission field, salience, trajectory
utility) all inherit the same asymmetry. The framework could not have been
written to model evil as a positive substance even if someone had wanted to;
its objects don't have negative versions.

This means three philosophical claims fall out of the framework's structure
rather than being imported:

1. **Evil cannot be created.** χ_b ≥ 0 — there is no χ_b < 0 mode.
2. **Evil cannot be self-sustaining.** Every failure mode depends on the
   prior existence of the goodness configuration it corrupts or blocks.
3. **Recovery is re-coherence, not replacement.** Restoring W[φ_mind],
   re-gating the permission field, and re-aligning cooperative winding all
   work on the *existing* structure — they don't introduce a new substance.

These map directly onto patristic claims about evil's non-being and the
nature of redemption as restoration of the imago Dei rather than
construction of a new being.

---

## 5. Pastoral and clinical corollaries

The framework's privation reading carries practical force:

- **Recovery is re-coherence, not exorcism-as-replacement.** Restoring the
  hijacked identity winding to its stable topological sector is mechanically
  what recovery is. The person's underlying coupling channel still exists.
- **The community provides the alignment field.** Cooperative winding
  alignment near 1 in a supportive collective gives the strong attractor
  basin the recovering person re-coheres into. Isolation is structurally
  the worst recovery condition.
- **Sacramental and ritual elements are field anchors.** External reference
  points (fixed sacred objects, repeated scripture, structured ritual) hold
  the salience field on coherence-sustaining attractors while internal
  re-gating happens.
- **Organic and psychiatric screening is essential.** Identity winding
  mutation from neurological or psychiatric causes presents identically to
  external hijacking at the level of the framework's observables. Ruling
  out organic causes is structurally required before attributing
  identity-winding mutation to external hijacking.

---

## 6. What this mapping does NOT do

1. **Does not derive God, Logos, or source consciousness as physical
   theorems.** The framework's structure is compatible with the
   classical-theist reading; it does not prove it. A naturalistic reading
   where φ is just a field and the substrate has no integration is equally
   coherent within BPR's mathematics.
2. **Does not adjudicate specific moral cases.** It describes the
   *structure* of moral phenomenology — identity corruption, permission
   failure, salience capture, utility corruption. Whether a given case is
   evil, legitimate, or neutral is a moral judgment the framework does not
   make.
3. **Does not equate cooperative misalignment with evil.** Whistleblowers,
   abuse-cycle survivors, conscientious objectors, and legitimate identity
   development all show up as winding-misaligned with a prior network. The
   framework's objects are morally neutral; the privation reading helps
   distinguish turning *away from* the light from turning *toward*
   something disordered, but the judgment is moral, not physical.
4. **Does not endorse any particular theological tradition.** The
   structural mapping is broadly Augustinian-Thomistic, which is shared
   across many traditions. Specific theological commitments
   (Trinitarian theology, Christology, sacramental theology) are not
   derivable from the framework.

---

## 7. Cross-references

- `bpr/functional_architecture.py` — IdentityWinding, PermissionField,
  SalienceField, TrajectoryEvaluation
- `bpr/collective.py` — CooperativeWinding
- `bpr/information.py` — Eq (5) consciousness coupling
- `doc/experiments/CONSCIOUSNESS_EMPIRICAL_CALIBRATION.md` — empirical
  envelope on χ_max·ε
- `doc/EXORCISM_BPR_MAPPING.md` — applies this framework to the structure
  of ritual exorcism as re-coherence
