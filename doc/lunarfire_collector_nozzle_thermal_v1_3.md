# LunarFire v1.3 Collector/Nozzle Thermal Architecture

v1.3 is the next proof point after the v1.2 thermal-packaging recovery sweep.

v1.2 showed that a generic `1%` direct heat recovery assumption plus hotter
radiators could make the design CAD-ready in the screening model. v1.3 replaces
that generic assumption with a channel-specific thermal model.

## Generated Outputs

Run:

```bash
python3 scripts/lunarfire_collector_nozzle_thermal_v13.py --output-dir data/helionis
```

Generated files:

- `data/helionis/lunarfire_collector_nozzle_thermal_v1_3.csv`
- `data/helionis/lunarfire_collector_nozzle_thermal_v1_3.md`
- `data/helionis/lunarfire_collector_nozzle_thermal_v1_3.png`

## Heat Channels

The v1.3 screen splits rejected heat into:

| Channel | Default fraction | Recoverable? |
| --- | --- | --- |
| Bremsstrahlung/core radiation | 68% | No |
| Transport/wall heat | 17% | No |
| Collector waste | 10% | Yes |
| Nozzle waste | 3% | Yes |
| Power-conditioning heat | 2% | Limited |

This is deliberately stricter than v1.2. The model does not allow collector or
nozzle design to recover bremsstrahlung/core heat.

## Result

The lowest heuristic-aggressiveness CAD-ready recipe found in the default sweep is:

| Metric | Value |
| --- | --- |
| Plant-net power | +0.9 MW |
| Collector capture efficiency | 8% |
| Nozzle capture efficiency | 0% |
| Radiator temperature | 1000 K |
| Topology packing factor | 1.0 |
| Recovered electric power | 5.7 MW |
| Recoverable channel heat | 106.0 MW |
| Adjusted radiator span per side | 344 m |

The highest-score CAD-ready row in the full default sweep reaches a readiness
score of `0.638` and about `+7.8 MW` plant-net.

## Interpretation

The generic v1.2 `1%` recovery requirement translates into a more concrete
collector/nozzle requirement:

> capture roughly `8%` of collector-side waste heat before it becomes radiator
> load, while keeping the radiator near `1000 K`.

This is more credible than saying "recover 1% of all rejected heat," because it
does not pretend that core bremsstrahlung or transport heat can be harvested by
the collector/nozzle subsystem.

## What This Means

The design path is narrower but still plausible in the screening model:

- most heat is still unrecoverable core heat
- the collector/nozzle only needs to recover a small slice of the total plant
  heat
- radiator temperature still matters because hundreds of MW remain as heat
- the next CAD package should focus on collector geometry, thermal isolation,
  high-voltage direct conversion, and radiator interfaces

## What It Does Not Prove

This is still not detailed thermal engineering. It does not prove:

- collector material survivability
- plasma-facing erosion limits
- high-voltage insulation in the collector stack
- detailed nozzle plume interception
- heat-pipe, pumped-loop, or radiator material feasibility
- transient loads during startup or control excursions

## Next Step

The next proof point should be a collector/nozzle subsystem model:

1. define collector stages and voltage levels
2. allocate charged alpha/proton power to each collector stage
3. compute intercepted heat flux and collector area
4. size a thermal isolation path from collector to radiator
5. check material temperature limits at `1000 K`

Plain-English conclusion:

> The thermal bottleneck is no longer "recover some magic 1%." It is now a
> specific subsystem target: recover about 8% of collector waste while leaving
> bremsstrahlung and transport heat as radiator load.
