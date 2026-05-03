# LunarFire v0.6 Mirror/Nozzle Loss-Cone Model

v0.6 stress-tests the v0.5 mirror/nozzle closure.

v0.5 was the first model version where mirror/nozzle closed plant-net power, but
that result depended on a compact end-loss proxy and a single generic collector
voltage. v0.6 replaces those with two more explicit screening terms:

- isotropic two-ended loss-cone fraction from mirror ratio
- staged alpha/proton collector voltage matching

This is still a screening model. It is not a full mirror stability calculation,
not a particle-orbit simulation, and not a detailed direct-energy-converter
design.

## Generated Outputs

Run:

```bash
python3 scripts/lunarfire_mirror_nozzle_v06.py --output-dir data/helionis
```

Generated files:

- `data/helionis/lunarfire_mirror_nozzle_v0_6.csv`
- `data/helionis/lunarfire_mirror_nozzle_v0_6.md`
- `data/helionis/lunarfire_mirror_nozzle_v0_6.png`

## Best Current Candidate

At a `50 MW` screening-net target, the top v0.6 candidate is:

| Metric | Value |
| --- | --- |
| Plant-net power | -17.1 MW |
| Screening-net power | 50.0 MW |
| Engineering load | 67.1 MW |
| Gross fusion power | 1,054.7 MW |
| Mirror ratio | 5.0 |
| Loss-cone fraction | 0.106 |
| Midplane field | 12.5 T |
| Plug field | 62.6 T |
| Alpha collector voltage | 1,800 kV |
| Proton collector voltage | 15,000 kV |
| Direct conversion efficiency proxy | 0.835 |
| Collector area at assumed power-density cap | 526 m2 |
| Plug-coil mass proxy | 501 tonnes |
| Effective magnet mass proxy | 850 tonnes |
| Radiator area | 54,021 m2 |

The v0.6 result **does not close plant-net power**.

## What Changed From v0.5

v0.5 best result:

- Plant-net: `+4.6 MW`
- Direct conversion proxy: `0.88`
- Gross fusion: `667 MW`
- Radiator area: `33,382 m2`

v0.6 best result:

- Plant-net: `-17.1 MW`
- Direct conversion proxy: `0.835`
- Gross fusion: `1,055 MW`
- Radiator area: `54,021 m2`

The closure disappeared because the more explicit loss-cone and collector model
increased required gross fusion power and heat rejection. The staged collector
is more physically meaningful than a single voltage proxy, but it also makes the
direct-conversion assumption harder to satisfy.

## Interpretation

Mirror/nozzle remains the most important alternate architecture to investigate,
but v0.6 is not enough to call it the lead design. That claim needs a
same-assumption FRC or other-geometry baseline.

What v0.6 does show:

- Its failure mode is now specific: loss-cone transport, plug-field burden, and
  collector staging.
- It gives us concrete subsystem targets instead of vague geometry preference.

The critical targets are now:

- Reduce effective loss-cone transport below the current proxy.
- Keep plug field below roughly `60 T`, or add a better magnet mass model.
- Push staged direct conversion above `0.84` without unrealistic collector area.
- Bring radiator area down from the `50,000 m2` class.

## Design Consequence

The correct conclusion is not:

> Mirror is solved.

The correct conclusion is:

> Mirror/nozzle remains worth investigating, but its apparent v0.5 closure is
> not robust under the v0.6 loss-cone and collector proxies.

For v0.7, the model should stop using power-balance proxies alone and add a
dedicated mirror physics layer:

1. Replace isotropic loss-cone fraction with a confinement/leakage model that
   includes pitch-angle scattering.
2. Add mirror ratio versus plug-field and coil-mass scaling as an explicit
   optimization, not a hard filter.
3. Split direct conversion into proton and alpha collector power flows,
   electrode area, and staged voltage losses.
4. Run sensitivity on the three decisive assumptions: loss-cone transport
   scale, direct-conversion cap, and plug-coil mass coefficient.
5. Compare mirror v0.7 against an FRC baseline using the same target and
   assumption level.

The v0.6 conclusion is:

> Mirror/nozzle remains a serious architecture candidate, but v0.6 no longer
> supports a plant-net closure claim. The next proof point is whether a
> same-assumption mirror/FRC comparison and a more physical confinement model
> can recover closure without relying on optimistic caps.
