# LunarFire v0.3 Minimum Viable Scale Sweep

The v0.3 scale sweep asks:

> Does LunarFire become plant-net positive if we simply make the FRC reactor
> bigger?

Current answer: **not under the present assumptions**.

The sweep tested 10, 25, 50, 100, and 250 MW screening-net targets. None close
plant-net power after the first-order engineering loads are subtracted.

## Generated Outputs

Run the sweep with:

```bash
python3 scripts/lunarfire_scale_sweep.py --output-dir data/helionis
```

Generated files:

- `data/helionis/lunarfire_scale_sweep_v0_3.csv`
- `data/helionis/lunarfire_scale_sweep_v0_3.md`
- `data/helionis/lunarfire_scale_sweep_v0_3.png`

## Results

| Target screening-net | Plant-net | Gross fusion | Engineering load | Radius | Length | Radiator area | Closes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 MW | -15.3 MW | 368 MW | 25.3 MW | 0.56 m | 3.37 m | 19,302 m2 | No |
| 25 MW | -33.5 MW | 919 MW | 58.5 MW | 0.76 m | 4.57 m | 48,018 m2 | No |
| 50 MW | -63.9 MW | 1,838 MW | 113.9 MW | 0.96 m | 5.76 m | 95,879 m2 | No |
| 100 MW | -124.6 MW | 3,677 MW | 224.6 MW | 1.21 m | 7.26 m | 191,601 m2 | No |
| 250 MW | -306.9 MW | 9,192 MW | 556.9 MW | 1.64 m | 9.86 m | 478,767 m2 | No |

## Interpretation

The sweep says the problem is **not just scale**.

If the engineering loads were mostly fixed, larger reactors would cross into
positive plant-net power. But in the current model, the dominant loads scale
with gross fusion power:

- Current drive is modeled as a fraction of gross fusion power.
- Power-conditioning loss is modeled as a fraction of useful converted power.
- Conversion waste scales with gross fusion output.
- Bremsstrahlung heat scales with the reactor output needed to hit the target.
- Radiator area scales with rejected heat.

So the plant-net margin remains negative even at 250 MW screening-net.

## What This Means

Going bigger helps packaging metrics, but it does not solve LunarFire by itself.
The real blockers are now clear:

1. **Bremsstrahlung / radiation handling**
   The reactor is producing enormous radiative heat relative to delivered power.

2. **Direct-conversion chain losses**
   A few percent loss on hundreds or thousands of MW becomes product-killing.

3. **Current-drive burden**
   Modeling current drive as 2% of gross fusion power is enough to erase margin.

4. **Gross-to-net leverage**
   At 10 MW screening-net, the top case needs about 368 MW gross fusion. At
   250 MW screening-net, it needs about 9.2 GW gross fusion. That is not a
   healthy product scaling law.

## Design Consequence

Do not move to CAD yet.

The next design step should not be “make it 100 MW.” It should be:

> Find a physics/architecture change that improves gross-to-screening-net
> leverage before scaling the plant.

The most valuable v0.4 work is therefore sensitivity search:

- How low must current-drive fraction fall?
- How high must direct-conversion efficiency rise?
- How much bremsstrahlung must be recovered or rejected at higher temperature?
- Does a different D-He3 operating point reduce gross fusion burden?
- Is the FRC model forcing an unrealistic confinement assumption?

## Current Go / No-Go

- FRC remains the current working geometry from the prior downselect; this
  sweep does not revisit the geometry choice.
- 10 MW is too small as a product point.
- Larger scale alone does not close plant-net under current assumptions.
- LunarFire needs a better loss/control architecture, not merely a bigger one.

The next milestone is:

> LunarFire v0.4 should run a sensitivity sweep over current drive, direct
> conversion efficiency, transport loss, bremsstrahlung handling, and operating
> temperature to identify which assumption has to change for plant-net closure.
