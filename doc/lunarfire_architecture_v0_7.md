# LunarFire v0.7 Shared-Grid Architecture Comparison

v0.7 answers the question left open by v0.6:

> Does mirror/nozzle still look better than FRC when both are compared under the
> same target, plasma grid, thermal conversion assumption, and engineering-net
> accounting path?

Current answer: **under the nominal architecture-specific performance
assumptions, mirror/nozzle still beats the FRC baseline, but neither
architecture closes plant-net power.**

This is a cleaner comparison than the earlier cross-version statements, but it
is not a pure equal-assumption ablation. Direct-conversion and transport
assumptions remain architecture-specific. The result shows that, under this
shared-grid/shared-accounting setup, mirror/nozzle has the less-bad plant-net
result.

## Generated Outputs

Run:

```bash
python3 scripts/lunarfire_architecture_v07.py --output-dir data/helionis
```

Generated files:

- `data/helionis/lunarfire_architecture_v0_7.csv`
- `data/helionis/lunarfire_architecture_v0_7.md`
- `data/helionis/lunarfire_architecture_v0_7.png`

## Result At 50 MW Screening-Net Target

| Architecture | Plant-net | Gross fusion | Engineering load | Rejected heat | Transport | Direct eta | Field | Plug field | Radiator area | Closes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mirror/nozzle | -18.2 MW | 904 MW | 68.2 MW | 918 MW | 0.14 | 0.835 | 12.5 T | 62.6 T | 46,504 m2 | No |
| FRC baseline | -86.9 MW | 2,433 MW | 136.9 MW | 2,507 MW | 0.15 | 0.78 | 10.9 T | N/A | 127,008 m2 | No |

## What Changed

v0.6 tested mirror/nozzle alone with a static loss-cone transport mapping and
staged collector voltages. It found:

- Mirror/nozzle v0.6: `-17.1 MW` plant-net

v0.7 adds a shared-grid FRC baseline, a pitch-angle-scattering leakage proxy for
mirror/nozzle, and a first-order collector/nozzle auxiliary load. It finds:

- Mirror/nozzle v0.7: `-18.2 MW` plant-net
- FRC v0.7 baseline: `-86.9 MW` plant-net

The mirror result improves relative to v0.6 because the leakage proxy now
depends on loss-cone refilling over the requested confinement time rather than
applying the full loss-cone fraction as a static transport multiplier.

## Interpretation

Mirror/nozzle is now the stronger architecture candidate in the shared-grid
screening model.

The reason is not that mirror closes. It does not. The reason is that the FRC
baseline requires much more gross fusion power and rejects much more heat to hit
the same `50 MW` screening-net target.

The mirror/nozzle failure mode is narrower:

- It misses plant-net by `18.2 MW`, not `86.9 MW`.
- It still requires a `62.6 T` plug field.
- It still rejects nearly `0.9 GW` of heat.
- It still relies on high staged direct conversion around `0.835`.
- It now includes a `9.8 MW` collector/nozzle auxiliary load.

The FRC failure mode is broader:

- It needs `2.4 GW` gross fusion for the same target.
- It rejects `2.5 GW` of heat.
- Radiator burden is roughly `127,000 m2`.
- It does not benefit enough from compactness to overcome gross-power burden.

## Design Consequence

The working architecture hypothesis should now be:

> Keep FRC as the compact reference baseline, but prioritize mirror/nozzle
> sensitivity work next.

The next step is not another geometry downselect. It is a mirror/nozzle margin
recovery campaign:

1. Sweep pitch-angle scattering time and mirror stabilization factor.
2. Sweep plug-field cap and plug-coil mass coefficient.
3. Sweep staged collector efficiency and collector power-density cap.
4. Identify the minimum combination that moves mirror/nozzle from `-18.2 MW` to
   positive plant-net.
5. Add equal-direct-eta/equal-transport ablations before treating the result as
   a final geometry downselect.

The v0.7 conclusion is:

> Mirror/nozzle is the current lead hypothesis under shared-grid screening, but
> the design is still margin-negative. The next proof point is sensitivity: find
> out whether the missing `18.2 MW` is recoverable through plausible mirror
> confinement, plug-field, and direct-conversion improvements.
