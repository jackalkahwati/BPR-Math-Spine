# LunarFire v0.9 BPR-Coupled Mirror/Nozzle Screen

v0.9 is the first pass that explicitly connects the Helionis/LunarFire reactor
screening model to the existing BPR Math Spine.

Earlier Helionis versions lived inside the same repository, but they did not
use `bpr/` primitives. v0.9 adds a small bridge layer:

- `bpr.impedance.TopologicalImpedance`
- `bpr.resonance.load_riemann_zeros`

These are used as **bounded correction factors** for mirror/nozzle transport and
direct-conversion coupling. They are not treated as energy sources.

The mapping is intentionally conservative and explicit:

- mirror ratio is used as an effective-winding proxy for BPR impedance mismatch.
- mirror/nozzle aspect ratio is compared with normalized BPR/Riemann-zero
  spacings as a resonance-alignment proxy.
- the resulting factors can reduce transport loss and improve collector/nozzle
  coupling only within hard caps.

## Generated Outputs

Run:

```bash
python3 scripts/lunarfire_bpr_coupled_v09.py --output-dir data/helionis
```

Generated files:

- `data/helionis/lunarfire_bpr_coupled_v0_9.csv`
- `data/helionis/lunarfire_bpr_coupled_v0_9.md`
- `data/helionis/lunarfire_bpr_coupled_v0_9.png`

## Result

At the `50 MW` screening-net target, the best BPR-coupled row is:

| Metric | Value |
| --- | --- |
| Plant-net power | -3.2 MW |
| Mirror ratio | 4.0 |
| Mirror/nozzle aspect ratio | 8.0 |
| Temperature | 200 keV |
| Ion density | 3.5e20 m^-3 |
| Confinement | 20 s |
| BPR resonance alignment | 0.993 |
| BPR impedance match | 0.862 |
| Transport multiplier factor | 0.785 |
| Direct-conversion multiplier | 1.047 |
| Gross fusion power | 697 MW |
| Radiator area | 35,284 m2 |

This does **not** close plant-net power, but it is much closer than v0.7.

## What Changed

v0.7 shared-grid mirror/nozzle:

- Plant-net: `-18.2 MW`
- No BPR primitives used

v0.8 margin recovery:

- Plant-net: `+3.3 MW`
- Required stronger staged-collector assumptions
- Still did not use BPR primitives

v0.9 BPR-coupled screen:

- Plant-net: `-3.2 MW`
- Uses BPR impedance and resonance proxy factors
- Does not require treating BPR as an energy source
- Still misses closure without the stronger v0.8 collector assumption

## Interpretation

With this bounded BPR-proxy coupling active, the mirror/nozzle case is much
closer to plant-net closure than the earlier v0.7 shared-grid result. This is
not yet a strict same-grid attribution study, because v0.9 also sweeps
mirror/nozzle aspect ratio.

But the model is still conservative in two ways:

- BPR factors are bounded to modest corrections.
- BPR factors modify transport/coupling only; they do not add net power.
- Resonance alignment is currently an aspect-ratio proxy, not a validated
  plasma eigenmode calculation.

So the current conclusion is:

> BPR math improves the mirror/nozzle case, but does not remove the need for a
> real charged-particle collector/nozzle model.

## Design Consequence

The next step should combine v0.8 and v0.9:

1. Keep the BPR bridge active for resonance/impedance factors.
2. Replace the direct-conversion cap with an explicit charged-product collector
   model.
3. Expand the BPR resonance bridge from aspect-ratio proxy to an explicit
   collector/nozzle eigenmode model.
4. Re-run the margin recovery sweep with BPR-coupled collector physics.

The v0.9 conclusion is:

> Helionis is now using the BPR Math Spine. The first integration shows that
> BPR impedance/resonance factors materially reduce the mirror/nozzle plant-net
> gap, but collector/nozzle physics remains the decisive proof point.
