# LunarFire v1.1 Control-Constrained CAD Envelope

v1.1 is the first CAD-facing layer for the Helionis / Modulus Fusion design loop.

It is not detailed CAD. It is a parametric envelope model that turns the current
physics and control outputs into first-order package constraints:

- plasma radius and length
- chamber clearance
- midplane and plug-coil radii
- magnetic-nozzle length
- staged collector surface area
- radiator area and wing span
- total machine length and outer radius
- CAD-readiness score

## Generated Outputs

Run:

```bash
python3 scripts/lunarfire_cad_envelope_v11.py --output-dir data/helionis
```

Generated files:

- `data/helionis/lunarfire_cad_envelope_v1_1.csv`
- `data/helionis/lunarfire_cad_envelope_v1_1.md`
- `data/helionis/lunarfire_cad_envelope_v1_1.png`

## Result

The highest-scoring v1.1 row is:

| Metric | Value |
| --- | --- |
| CAD-readiness score | 0.463 |
| CAD-ready | False |
| Machine length | 21.2 m |
| Outer radius | 13.6 m |
| Plasma radius | 0.96 m |
| Magnetic nozzle length | 6.0 m |
| Collector surface area | 408 m2 |
| Radiator area | 39,683 m2 |
| Radiator wing span per side | 936 m |
| Control score | 0.656 |
| Plant-net power | -5.1 MW |

## Interpretation

No current row is CAD-ready.

There are two primary blockers:

1. The source plant is still net-negative.
2. The radiator span is too large.

The reactor core envelope itself is not the first packaging blocker. The best
row is roughly a 20-meter-class machine with a 14-meter-class outer radius. That
is large, but still a coherent parametric envelope.

The radiator area is nearly `40,000 m2`, and the equivalent two-wing deployment
is close to `1 km` per side in the current layout assumption. That makes the row
not CAD-ready even though the plasma, coil, collector, and nozzle envelope can
be parameterized.

## Design Consequence

The next step should not be beautified CAD. It should be a thermal-packaging
recovery loop:

1. Reduce rejected heat through better direct conversion.
2. Revisit radiator temperature assumptions and materials.
3. Try alternate radiator topology: cylindrical sleeve, deployable panels,
   pumped-loop arrays, or lunar surface heat rejection.
4. Re-run the v1.1 envelope with radiator span as a hard constraint.

## Plain-English Conclusion

The machine is no longer an abstract reactor equation. It now has an envelope:

> The core is packageable enough to draw parametrically, but the full machine is
> not CAD-ready until plant-net power closes and radiator span drops.

So the next engineering bottleneck is thermal architecture, not the plasma
chamber shell.
