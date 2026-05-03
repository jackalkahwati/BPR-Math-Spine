# LunarFire v0.4 Plant-Net Geometry Re-Score

v0.4 reopens the geometry question using plant-net accounting.

The earlier downselect favored FRC because it fits the D-He3 thesis: compact,
high-beta, low-neutron, and direct-conversion friendly. But the engineering-net
and scale sweeps showed that the current FRC assumptions do not close. v0.4
therefore asks:

> If current drive, conversion losses, rejected heat, radiator burden, and
> geometry-specific assumptions are included, does another geometry become less
> bad than FRC?

Current answer: **mirror/nozzle geometry deserves a serious second look.**

It does not close plant-net either, but it misses by less than FRC under the
rough v0.4 assumptions.

## Generated Outputs

Run:

```bash
python3 scripts/lunarfire_geometry_engineering.py --output-dir data/helionis
```

Generated files:

- `data/helionis/lunarfire_geometry_engineering_v0_4.csv`
- `data/helionis/lunarfire_geometry_engineering_v0_4.md`
- `data/helionis/lunarfire_geometry_engineering_v0_4.png`

## Result At 50 MW Screening-Net Target

| Geometry | Feasible screening candidate | Plant-net | Gross fusion | Engineering load | Radius | Length | Field | Radiator area | Rejections | Closes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mirror / nozzle | Yes | -24.7 MW | 1,249 MW | 74.7 MW | 0.86 m | 8.56 m | 12.51 T | 64,196 m2 | None | No |
| FRC | Yes | -59.7 MW | 1,774 MW | 109.7 MW | 0.76 m | 4.55 m | 12.88 T | 92,398 m2 | None | No |
| Spherical torus | No | N/A | N/A | N/A | N/A | N/A | N/A | N/A | no_positive_screening_net=80 | No |

## Interpretation

The mirror profile is not a winner. It is the least bad plant-net case in this
rough comparison.

Why it improves relative to FRC:

- Better assumed charged-particle/direct-conversion access.
- Lower assumed current-drive fraction.
- Linear nozzle geometry aligns with the energy extraction problem.

Why it remains risky:

- End losses are modeled only as a transport penalty, not a true mirror
  confinement model.
- It still needs 12.5 T field in the best case.
- It still rejects more than a gigawatt-scale thermal burden for a 50 MW
  screening-net target.
- It still misses plant-net power by about 25 MW.

Spherical torus does not look promising under the current D-He3 assumptions. In
the tested grid, every spherical-torus operating point was rejected before
engineering-net accounting because it failed to produce positive screening-net
power per unit volume. That result is consistent with weak direct
charged-particle access and the gross-power burden, but the current model only
proves the rejection reason, not a detailed torus physics failure.

## What This Means

We should not say:

> FRC is wrong and mirror is right.

We can say:

> FRC is no longer the only serious geometry. Mirror/nozzle architecture now has
> enough plant-net advantage to justify a dedicated model.

The correct next step is not broad CAD. It is a **mirror/nozzle v0.5 model**:

1. Replace the mirror transport penalty with an explicit end-loss model.
2. Model charged-particle expansion and direct conversion in the nozzle.
3. Compare mirror ratio, plug field, length, and collector voltage.
4. Re-run the plant-net budget with actual mirror leakage/current-drive terms.
5. Keep FRC as the baseline until the mirror model survives that pass.

## Current Geometry Status

- FRC: still compact and strategically aligned, but plant-net burden is bad.
- Mirror/nozzle: now the most interesting alternate geometry.
- Spherical torus: useful baseline, not currently aligned with LunarFire's
  D-He3/direct-conversion wedge.

The v0.4 conclusion is:

> LunarFire should continue FRC as the baseline, but immediately build a
> dedicated mirror/nozzle model because plant-net accounting makes it the most
> credible challenger.
