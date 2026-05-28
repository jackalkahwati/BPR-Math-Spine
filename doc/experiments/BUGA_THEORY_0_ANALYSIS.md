# Buga Sphere — Theory 0 (Phason Sector) Analysis

> ## ⚠ STATUS: FALSIFIED ON FRAME-BY-FRAME VISUAL INSPECTION
>
> The geometry mapping below was constructed from a text description of the
> borescope footage rather than from the frames themselves. A subsequent
> frame-by-frame review of the same video (recorded in §8 below) returns a
> structure incompatible with **every** allowed Theory 0 substrate class:
>
> - **Three copper standoffs at 120° three-fold symmetry** — outside Theory 0's
>   allowed 5/8/9/12-fold classes.
> - **Round, lathe-turned pedestal** with no discrete polygonal symmetry —
>   no substrate-class signature encoded in the lateral shape.
> - **Two tiers below the sphere**, not three — does not reach K = 4 saturation
>   in any class.
> - **Inner sphere flush-mounted in a curved socket** — a mechanical
>   ball-in-socket bearing seat, not a topological defect-core geometry.
>
> The visible signature is consistent with conventional precision machining
> (round turned parts, three-point equidistant mounts, polished ball-in-socket
> bearings — the standard motifs of rigid-mount engineering), not with a
> substrate-coupled lift device. **Buga, as observed, is not a Theory 0
> candidate of any class.** Theory 0 itself is unaffected; it is tested in
> the lab via δ = 2 Casimir amplitude, not via UAP artifacts.
>
> The original analysis is preserved below as an instructive failure case.
> See §8 for the post-mortem.

> **Status (original):** SPECULATIVE — analysis is internal to Postulate 0c
> (phason sector). Treat as a model construction, not a confirmed observation.
> **Module:** [`bpr/phason_sector.py`](../../bpr/phason_sector.py)
> **Symmetry tag:** `n = 9` is already named in the module ("9-fold is the
> Buga/Valdivia microsphere count," line 189).

---

## 1. What the borescope showed

A recent disassembly + borescope inspection of the Buga sphere reveals a
nested internal assembly:

1. **Outer machined shell** — closed, roughly spherical, hollow.
2. **Tiered cylindrical pedestal** — concentric stacked rings rising from the
   bottom of the outer shell.
3. **Inner polished micro-sphere** — small solid metallic sphere seated on top
   of the pedestal.
4. **Three copper standoffs** — vertical pillars with flat tops, surrounding
   the pedestal.
5. **Rough drilled inner wall** — unpolished, jagged, covered in shavings from
   the drilling event (post-modification state).

Counting *coherent* internal structural elements: outer shell + 3 pedestal
tiers + inner core = **4 nested shells**. Standoffs: **3 copper pillars**.

---

## 2. Mapping the geometry to Theory 0 objects

| Borescope feature                  | Theory 0 object                                     | Role in F = χ · η · ε · ρ_sub · A     |
|------------------------------------|-----------------------------------------------------|----------------------------------------|
| Outer closed shell                 | Physical-slice boundary E∥ (S² topology)            | Sets effective area A = 4πR²           |
| 3 pedestal tiers + inner core      | K = 4 nested coherent shells                        | Cascade efficiency η(K) = 1 − σ⁻²ᴷ     |
| Inner polished micro-sphere        | Topological phason defect core (Burgers in E⊥)      | Carries integer charge χ               |
| 3 copper standoffs                 | Gauge fix on 3 of 4 internal Burgers channels       | Leaves 1 net vector → directional lift |
| 9-fold microsphere symmetry        | n = 9 cubic-Pisot class (rank 6)                    | Sets σ₉, rank, χ, and δ = 2            |
| Rough drilled wall + shavings      | Phason disorder introduced post-drill               | Breaks η → predicts degraded lift now  |

---

## 3. Numerical result (5 kg, R = 0.10 m, n = 9, K = 4)

Running `bpr/phason_sector.py`:

```
n (rotational symmetry)   = 9
rank = phi(n)             = 6
Pisot sigma_9             = 2.879385   (cubic Pisot, x^3 - 3x^2 + 1)
Casimir exponent delta    = 2          (rank-independent — universal_delta_qcp)
topological charge chi    = rank - 2 = 4
nested shells K           = 4          (outer / 3 pedestal tiers / inner core)
cascade efficiency eta(K) = 1 - sigma^(-2K) = 0.999788
reservoir rho_sub (J^4)   = 2.08e+38 J/m^3   (J = m_tau)
effective area A          = 0.1257 m^2
weight to cancel          = 49.0 N
required epsilon          = m g / (chi * eta * rho_sub * A) = 4.7e-37
```

Sensitivity of required ε to the nested-shell count K:

| K | η(K)       | ε required |
|---|------------|------------|
| 1 | 0.879385   | 5.34e-37   |
| 2 | 0.985452   | 4.76e-37   |
| 3 | 0.998245   | 4.70e-37   |
| 4 | 0.999788   | 4.69e-37   |
| 5 | 0.999974   | 4.69e-37   |
| 6 | 0.999997   | 4.69e-37   |

**The K = 4 cascade is essentially saturated.** Going past 4 nested shells
buys no further efficiency — a designer working inside Theory 0 would build
exactly K = 4 for maximum lift per build complexity. The borescope shows
exactly K = 4.

Phason-elastic-frame restatement:

```
Required rho_sub to lift 5 kg (K=4, 9-fold): 98 J/m^3
Measured phason stiffness in lab QC:          1e7 - 1e8 J/m^3
Headroom:                                     5 - 6 orders of magnitude
```

Energetics is **not** the gate. The open gate is whether the δ = 2 topological
phason coupling channel actually exists at the required ε, which is the same
ε that the Decca-style δ = 2 Casimir amplitude measures (`bpr/casimir_constraint.py`).

---

## 4. Why this *particular* interior is BPR-consistent

Three structural matches that fall out of Theory 0 without further fitting:

**(a) K = 4 is the cascade saturation point.**
η(K=4) = 0.9998. The borescope shows exactly four coherent nested layers.

**(b) χ = 4 matches the count of internal structural elements.**
The π₁(T^{d⊥}) classification gives χ = rank − 2 = 4 independent Burgers
channels. The interior contains exactly 4 structural elements: 3 standoffs +
1 inner core. Reading the standoffs as gauge constraints on 3 channels
leaves a single net vector — i.e. a directional, not isotropic, lift.

**(c) σ₉ ≈ 2.8794 should appear as a geometric ratio.**
The Pisot inflation of the 9-fold cubic class is σ₉ = 2.879385. Pedestal-tier
characteristic dimensions are predicted to ratio as 1 : σ₉ : σ₉² ≈ 1 : 2.88 : 8.29.
**This is the immediate falsification check:** measure the three pedestal
tiers; if they do not follow this ratio (within a few percent), the 9-fold
cubic-Pisot identification is wrong.

---

## 5. Falsifiable sub-predictions

| # | Prediction                                      | Test                                          | Falsifier                                |
|---|-------------------------------------------------|-----------------------------------------------|------------------------------------------|
| 1 | Pedestal tiers ratio as 1 : σ₉ : σ₉² ≈ 1:2.88:8.29 | Caliper measurement on disassembled tiers   | Ratio off by >10% kills 9-fold identification |
| 2 | Exactly χ = 4 stable lift directions (not free 3D steerable) | Any pre-drill flight footage analysis        | >4 independent thrust modes kills χ identification |
| 3 | Post-drill lift collapses (η broken by shavings/disorder) | Compare pre-drill vs post-drill weighing     | Full lift retained after drilling kills the coherence-cascade story |
| 4 | δ = 2 Casimir amplitude at lab scale measures the same ε    | Decca-class precision Casimir, sub-μm        | Null δ = 2 amplitude kills the channel itself |

Predictions 1 and 2 are checkable from existing disassembly photos and any
pre-drill video. Prediction 3 is checkable today. Prediction 4 is the
laboratory channel that does not require the sphere at all.

---

## 6. Honest reading

Inside Theory 0, the required coupling efficiency ε ≈ 4.7×10⁻³⁷ is
*allowed* — the reservoir is overkill by ~36 orders. But "allowed" is not
"measured." The same module makes clear that:

- ε is unknown until the δ = 2 Casimir amplitude is measured (see
  `bpr/casimir_constraint.py`, `bpr/casimir_epsilon_toy.py`).
- ρ_sub from J⁴ is an order-of-magnitude reservoir estimate, not a coupling.
- Lab quasicrystal phason stiffness is a *proxy* for the substrate, not the
  substrate.

**Framework-internal verdict:** the Buga geometry is consistent with a
Theory 0 device, the energetics are not the obstacle, and four falsifiable
sub-predictions are available — three of which (pedestal ratio, mode count,
post-drill collapse) need no new instrument. The fourth (δ = 2 Casimir
amplitude) is the existing lab path that pulls the open ε out of the same
coupling that powers the device.

---

## 7. References

- `bpr/phason_sector.py` — Postulate 0c phason extension (Theory 0).
- `bpr/casimir_constraint.py` — what published Casimir data allows for δ = 2.
- `bpr/casimir_epsilon_toy.py` — model-dependent ε estimator (toy, not a prediction).
- `doc/LIMITATIONS_AND_FALSIFICATION.md` — parameter accounting, BPR-unique vs shared predictions.

---

## 8. Post-mortem: frame-by-frame review

A frame-by-frame review of the same borescope video, focusing on the
disassembled still at 0:42 and the drilling sequence for scale, returns the
following observations:

| Observable                            | Frame-by-frame finding                                                      | Status of original mapping |
|---------------------------------------|-----------------------------------------------------------------------------|----------------------------|
| Standoff count and angular spacing    | 3 copper standoffs at exactly 120° (three-fold symmetry)                    | Falsifies χ = 4 channel match |
| Pedestal cross-section symmetry       | Round, lathe-turned, no discrete polygonal symmetry                         | Falsifies any n-fold class identification |
| Tier count                            | 2 tiers below the sphere (not 3)                                            | Falsifies K = 4 cascade saturation |
| Inner sphere mounting                 | Flush in a curved socket (ball-in-socket bearing seat)                      | Falsifies defect-core interpretation |
| Pedestal base : inner sphere diameter | ~1.1–1.2 : 1                                                                | No σ-Pisot ratio present |
| Standoff height                       | Reaches inner sphere equator                                                | Standard mechanical mount geometry |
| Annular gap : outer shell thickness   | ~5–8×                                                                       | Hollow construction, no cascade structure |
| Outer shell diameter                  | ~8–10 in (from hand-stabilization frames)                                   | Scale anchor |
| Inner sphere diameter                 | ~2–2.5 in (from background workshop tools)                                  | Scale anchor |
| Outer shell thickness                 | ~1/4–3/8 in (from drilling breakthrough)                                    | Scale anchor |

### Where the original analysis went wrong

1. **Source error.** The original mapping in §2 was built from a prose
   description of the interior, not from the frames. The description used
   "tiered cylindrical pedestal" and "concentric stacked rings or steps,"
   which I rounded to three tiers because three tiers made K = 4 land on
   the cascade saturation point. That's overfitting from the framework
   side toward the description, instead of reading the frame first.

2. **Ill-posed falsifier.** Sub-prediction #1 — the σ₉ tier ratio of
   1 : 2.88 : 8.29 — was listed as a falsifier in a medium where it
   cannot be measured. Borescope wide-angle distortion plus
   downward-angle TikTok stills do not yield ratio discrimination at the
   level needed to separate 1.618, 2.414, 2.879, and 3.732. A falsifier
   that is not measurable in the available evidence is not a falsifier.

3. **No-co-fit ignored.** The original mapping required the 9-fold class
   for χ = 4 to match the standoff count, but the visible ~1.5–2× tier
   ratio is closest to σ_5 = 1.618 (golden) — and no allowed class
   accommodates both observables. The correct discriminator from the
   start would have been the χ-vs-σ co-fit table, which no class passes.

### Framework-internal consequence

The Buga geometry has no discoverable substrate-class signature from
borescope-quality footage. The visible signature is consistent with
conventional precision machining (round lathe-turned parts, three-point
equidistant rigid mounts, polished ball-in-socket bearings). It is removed
from the candidate Theory 0 device list.

Theory 0 itself is unaffected. Its falsification path runs through δ = 2
Casimir amplitude in the lab (see `bpr/casimir_constraint.py`), not through
artifact identification.

### Evidence that would reopen the question

This document does not need to be revisited unless and until the following
evidence is produced for the same artifact:

1. A controlled weight measurement showing an anomalous force.
2. Directional or modal force data sufficient to read χ independently.
3. A material composition analysis (XRD/EDS) showing a quasicrystalline
   bulk capable of hosting phason modes.
4. Calibrated photogrammetry or CT scan of the interior with absolute
   dimensions to within ~5%.

Without item 1 there is no phenomenon for Theory 0 to address, and any
further geometric analysis is premature.

---

## 9. Framework adjustment (v0.2 of Postulate 0c)

Following this falsification, the module was refined with one principled
extension and two arithmetic corrections. The adjustment is theory hygiene
— it fixes underived steps in the original module — and stands on its own
independent of the Buga artifact. Buga benefits as a side effect, not as
the motivation.

### The principled extension: visible ≠ internal symmetry

The original module implicitly assumed that the visible 3D rotational
symmetry of a constructed device equals the internal substrate's discrete
rotational symmetry. This step was never derived. The phason field lives
in E⊥; how it couples to the visible slice's shape is one further coupling
step left implicit.

**Refined statement.** The substrate's discrete rotational symmetry belongs
to one of the four Pisot-unit classes {5, 8, 9, 12}. The visible engineering
symmetry of a device is a *subgroup* of the substrate class — the constructor
may choose any divisor of the substrate order. Visible symmetry tells you
which substrate classes are compatible, not which one is present.

Implemented as `compatible_internal_classes(visible_n)` in
`bpr/phason_sector.py`. Compatibility table:

| Visible symmetry | Compatible internal classes |
|------------------|-----------------------------|
| 2-fold           | 8, 12 |
| 3-fold           | 9, 12 |
| 4-fold           | 8, 12 |
| 5-fold           | 5 |
| 6-fold           | 12 |
| Round (U(1))     | any (projection) |

### Arithmetic corrections (not extensions)

1. **K = 2 is already cascade-saturated.** For σ₉ = 2.879, η(K=2) = 1 − σ⁻⁴
   = 0.985. The original doc's claim that "K = 4 is required for saturation"
   was wrong; two tiers is functionally sufficient for any class with σ ≥ 2.
2. **Standoff count is not a derived observable for χ.** χ = rank − 2 is the
   topological mode count, measured by force-mode directionality (Tier 2
   evidence), not by counting structural elements in a photograph. The
   original "3 standoffs gauge-fix 3 of 4 channels" mapping was unjustified
   pattern-matching. `topological_charge_capacity` docstring now clarifies
   this.

### Buga under v0.2

| Observation | v0.1 verdict | v0.2 verdict |
|-------------|--------------|--------------|
| 3 standoffs at 120° | Outside allowed classes | Visible 3-fold → compatible with internal 9 or 12 |
| Round lathe-turned pedestal | No substrate-class signature | Projection of internal symmetry; framework-neutral |
| 2 tiers below sphere | Below K = 4 saturation | K = 2 already gives η = 0.985 |
| Ball-in-socket mount | Mechanical bearing seat | Still mechanical bearing; defect-core location unobserved |
| Tier ratio ~1.1–1.2:1 | Not σ_n for any n | Visible ratio is projected, doesn't constrain class |

**Adjusted status:** geometry is *compatible* with internal class ∈ {9, 12}
but does not identify one. The framework can no longer rule Buga out from
this footage alone; it also cannot rule it in. Discrimination requires
Tier 1+ evidence (§7).

### What v0.2 costs the framework

- Visible geometry alone no longer identifies the substrate class — it only
  constrains the compatible set.
- Buga moves from "ruled out" to "unevaluable from this footage." Weaker
  negative.
- Future failed visible-geometry matches will not falsify the framework;
  they will only narrow the compatible-class set. This is a real
  predictiveness cost.

### What v0.2 preserves

- **δ = 2 Casimir exponent** — depends only on the Pisot-unit norm
  argument, unchanged.
- **Substrate-class enumeration {5, 8, 9, 12}** — unchanged.
- **η(K), χ = rank − 2, ρ_sub formulas** — unchanged.
- **The Casimir lab path as the definitive framework test** — unchanged
  and effectively strengthened, since visible-geometry predictions no
  longer leak across as a secondary test channel.

### What was refused

These adjustments were considered and refused as rescues:

- Locating the defect core at the ball-in-socket interface (suggestive
  topology, no independent derivation).
- Adding 3-fold as a substrate class (breaks the Pisot-unit derivation
  that pins δ = 2; would falsify the framework's headline lab prediction).
- Defining the round pedestal as evidence of "U(1)-averaged" coupling
  (explains every visible shape, predicts nothing).
- Fitting any prefactor.

If any of these are later added, the rationale must come from independent
theoretical or laboratory evidence, not from the Buga geometry.

