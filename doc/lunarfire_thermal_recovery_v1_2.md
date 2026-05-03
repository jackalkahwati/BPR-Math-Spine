# LunarFire v1.2 Thermal Packaging Recovery

v1.2 continues the analysis from the CAD envelope result.

v1.1 showed that the reactor core can be parameterized, but the full machine is
not CAD-ready because:

- plant-net power is still negative
- radiator wing span is too large

v1.2 asks what minimum thermal recovery recipe clears both blockers.

## Generated Outputs

Run:

```bash
python3 scripts/lunarfire_thermal_recovery_v12.py --output-dir data/helionis
```

Generated files:

- `data/helionis/lunarfire_thermal_recovery_v1_2.csv`
- `data/helionis/lunarfire_thermal_recovery_v1_2.md`
- `data/helionis/lunarfire_thermal_recovery_v1_2.png`

## Result

The least aggressive CAD-ready recipe found in the sweep is:

| Metric | Value |
| --- | --- |
| Plant-net power | +2.3 MW |
| Direct heat recovery | 1% |
| Radiator temperature | 1000 K |
| Topology packing factor | 1.0 |
| Adjusted rejected heat | 699 MW |
| Adjusted radiator area | 14,510 m2 |
| Adjusted wing span per side | 343 m |

## Interpretation

This is a major design clarification.

The radiator problem is not solved by layout alone. The full system needs even
a small amount of direct-conversion heat recovery to close plant-net power.

But the recovery threshold is not enormous in this screening model:

- capturing about `1%` of inferred rejected heat as useful electric power closes the
  plant-net gap
- raising radiator operating temperature from `800 K` to `1000 K` reduces area
  enough to fit the `500 m` span constraint
- topology packing helps, but the least aggressive recipe does not require it

## What This Means

The system now has a plausible path from:

> controllable but net-negative and radiator-impossible

to:

> slightly plant-positive and thermally packageable

under a modest but important assumption: the collector/nozzle can recover about
`1%` of what was previously rejected as heat and the radiator can operate near
`1000 K`.

The highest-score CAD-ready row in the full default sweep reaches a readiness
score of `0.743` and about `+23.4 MW` plant-net, but it requires a more
aggressive recipe than the minimum-assumption case.

## What It Does Not Prove

This does not prove a working reactor.

It does not validate:

- plasma stability
- collector survivability
- radiator material feasibility at 1000 K
- high-voltage direct-conversion hardware
- detailed thermal hydraulics
- launch, deployment, or maintenance architecture

## Next Step

The next analysis should move from generic thermal recovery to a specific
collector/nozzle thermal design:

1. split rejected heat into bremsstrahlung, transport, collector waste, and
   power-conditioning heat
2. model which parts can realistically be recovered by direct conversion
3. model radiator materials at `1000 K`
4. feed that into the CAD envelope as real subsystem geometry

Plain-English conclusion:

> The design becomes CAD-plausible if we recover a small fraction of rejected
> heat and run hotter radiators. The next proof point is the collector/nozzle
> thermal architecture.
