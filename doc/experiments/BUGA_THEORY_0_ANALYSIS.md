# Buga Sphere — Theory 0 (Phason Sector) Analysis

> **Status:** SPECULATIVE — analysis is internal to Postulate 0c (phason sector).
> Treat as a model construction, not a confirmed observation.
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
