# LunarFire v0.8 Mirror/Nozzle Margin Recovery

v0.8 asks a narrower question than v0.7:

> The shared-grid mirror/nozzle case misses plant-net by `18.2 MW`. What is the
> least aggressive recovery recipe that closes the gap in the screening model?

Current answer: **the decisive lever is staged direct conversion, but it is not
just a cap change.**

In the nominal v0.8 sweep, mirror/nozzle closes plant-net when the assumed
direct-conversion cap rises from `0.86` to `0.88` and the collector voltage-match
bonus rises from `0.22` to `0.30`. The recovery is cap-limited: the uncapped
direct-conversion proxy is about `0.91`, but the row is clipped to `0.88`.

## Generated Outputs

Run:

```bash
python3 scripts/lunarfire_margin_recovery_v08.py --output-dir data/helionis
```

Generated files:

- `data/helionis/lunarfire_margin_recovery_v0_8.csv`
- `data/helionis/lunarfire_margin_recovery_v0_8.md`
- `data/helionis/lunarfire_margin_recovery_v0_8.png`

## Minimum Recovery Recipe

| Lever | v0.7 nominal | v0.8 minimum closing recipe |
| --- | --- | --- |
| Plant-net power | -18.2 MW | +3.3 MW |
| Pitch-angle scattering time | 80 s | 80 s |
| Mirror stabilization factor | 1.0 | 1.0 |
| Direct-conversion cap | 0.86 | 0.88 |
| Collector base efficiency | 0.58 | 0.58 |
| Collector match bonus | 0.22 | 0.30 |
| Cap-limited direct conversion | No | Yes |
| Plug-coil mass coefficient | 0.20 t/T2 | 0.20 t/T2 |
| Collector auxiliary load | 5.0 kW/m2 | 5.0 kW/m2 |
| Nozzle auxiliary fraction | 0.010 | 0.010 |

Best minimum-recovery operating point:

- Temperature: `200 keV`
- Ion density: `3.5e20 m^-3`
- Confinement: `20 s`
- Mirror ratio: `4.0`
- Transport multiplier: `0.160`
- Gross fusion power: `598 MW`
- Radiator area: `29,989 m2`

## Interpretation

This is the first result that gives a concrete recovery path:

> Mirror/nozzle does not need every subsystem to improve at once in the current
> screening model. It mainly needs the staged direct-conversion system to reach
> an assumed `0.88` effective efficiency with a stronger voltage-match term.

That is encouraging, but it is not a reactor proof.

The result depends on several assumptions that need hardware-level validation:

- The alpha and proton collectors can be staged near their ideal voltages.
- The effective direct-conversion cap can reach `0.88` after real
  collector losses, electrode loading, plasma exhaust spread, and power
  conditioning.
- The collector/nozzle auxiliary load proxy is not badly underestimated.
- The mirror leakage proxy remains valid when moved from screening math toward
  a confinement model.

## What This Means

The design priority should shift again.

Earlier versions asked:

- Which geometry is less bad?
- Does mirror/nozzle beat FRC?
- Can mirror/nozzle close under tougher leakage assumptions?

v0.8 says the next bottleneck is more specific:

> The critical subsystem is the direct-energy-conversion nozzle.

The next step should be a v0.9 **charged-product collector model**:

1. Split D-He3 charged power into alpha and proton channels.
2. Model ideal collector voltages, voltage staging, and energy spread.
3. Estimate electrode area and collector heat load from incident charged power.
4. Add conversion efficiency loss terms instead of a single cap.
5. Re-run the margin recovery sweep with collector physics replacing the `0.88`
   cap and `0.30` voltage-match bonus.

The v0.8 conclusion is:

> Mirror/nozzle can recover plant-net margin in the screening model, but only
> under a stronger staged-collector assumption: an effective `0.88` cap plus a
> higher voltage-match bonus. The next proof point is no longer broad reactor
> geometry. It is whether the charged-particle collector/nozzle can physically
> deliver that efficiency.
