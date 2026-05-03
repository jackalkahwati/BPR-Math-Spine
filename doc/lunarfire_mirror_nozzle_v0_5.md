# LunarFire v0.5 Mirror/Nozzle Model

v0.5 takes the v0.4 geometry re-score seriously and builds a dedicated
mirror/nozzle screen.

The v0.4 result did not prove that mirror geometry is better than FRC. It showed
that mirror/nozzle architecture was the least-bad geometry under plant-net
accounting. v0.5 replaces the generic mirror transport penalty with explicit
screening knobs:

- mirror ratio
- plug-field requirement
- axial end-loss proxy
- collector voltage
- direct-conversion efficiency proxy

This is still an order-of-magnitude screening model. It is not a mirror
stability calculation, not a particle-in-cell model, and not a detailed direct
energy converter design.

## Generated Outputs

Run:

```bash
python3 scripts/lunarfire_mirror_nozzle.py --output-dir data/helionis
```

Generated files:

- `data/helionis/lunarfire_mirror_nozzle_v0_5.csv`
- `data/helionis/lunarfire_mirror_nozzle_v0_5.md`
- `data/helionis/lunarfire_mirror_nozzle_v0_5.png`

## Best Current Candidate

At a `50 MW` screening-net target, the top v0.5 candidate is:

| Metric | Value |
| --- | --- |
| Plant-net power | +4.6 MW |
| Screening-net power | 50.0 MW |
| Engineering load | 45.4 MW |
| Gross fusion power | 666.9 MW |
| Mirror ratio | 5.0 |
| Midplane field | 12.5 T |
| Plug field | 62.6 T |
| Plug-coil mass proxy | 501 tonnes |
| Effective magnet mass proxy | 730 tonnes |
| Collector voltage | 1,500 kV |
| Direct conversion efficiency proxy | 0.88 |
| End-loss multiplier | 0.36 |
| Total transport multiplier | 0.42 |
| Radiator area | 33,382 m2 |

This is the first LunarFire screening result that closes plant-net power under
the current model family.

## Interpretation

The result is promising, but fragile.

Why it closes:

- Direct conversion is pushed to the model cap of `0.88`.
- Gross fusion power drops from the v0.4 mirror result's roughly `1.25 GW` to
  roughly `667 MW`.
- Current-drive load remains lower than the FRC reference assumption.
- The linear nozzle architecture is allowed to monetize charged-product energy
  more efficiently than the FRC baseline.

Why it is still hard:

- The plug field is `62.6 T`, which is extremely aggressive.
- The plug-field mass proxy adds roughly `501 tonnes` to the magnet proxy.
- The model treats collector voltage as a direct-conversion proxy, not as a
  detailed staged electrode design.
- End losses are still represented by a compact proxy, not by loss-cone physics.
- Radiator area remains tens of thousands of square meters.
- Bremsstrahlung remains a dominant system driver.

## FRC Comparison

The previous v0.4 geometry re-score at the same `50 MW` screening-net target
found:

- FRC: `-59.7 MW` plant-net
- Mirror/nozzle: `-24.7 MW` plant-net

The dedicated v0.5 mirror/nozzle model improves the mirror case to:

- Mirror/nozzle v0.5: `+4.6 MW` plant-net

That does not mean mirror wins permanently. It means the next design cycle
should move from "should we consider mirror?" to "can the mirror/nozzle closure
survive a more physical end-loss and direct-conversion model?"

## Design Consequence

FRC remains valuable as the compact baseline, but mirror/nozzle is now the lead
candidate for the D-He3 architecture path.

The immediate next step is v0.6:

1. Replace the end-loss proxy with a loss-cone model.
2. Split charged-product power by particle species for collector design.
3. Estimate collector voltage stages, electrode area, and power density.
4. Add plug-coil mass and cryogenic penalty as a function of plug field.
5. Re-run plant-net closure with the plug-field penalty included.

The v0.5 conclusion is:

> Mirror/nozzle is the first geometry to close in the screening model, but the
> closure depends on high plug field and optimistic direct conversion. It should
> become the primary architecture under test, with FRC retained as the compact
> baseline.
