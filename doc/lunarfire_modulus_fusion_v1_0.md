# LunarFire / Modulus Fusion v1.0 Control Twin

v1.0 adds the first Modulus Fusion control layer on top of the Helionis
BPR-coupled mirror/nozzle screen.

The purpose is not to claim that the physical plasma has zero drift. The purpose
is narrower and more defensible:

> Modulus Fusion adds zero numerical drift in the deterministic control math,
> while still reporting physical plasma drift, sensor error, latency, and
> actuator limits separately.

## Generated Outputs

Run:

```bash
python3 scripts/lunarfire_modulus_fusion_v10.py --output-dir data/helionis
```

Generated files:

- `data/helionis/lunarfire_modulus_fusion_v1_0.csv`
- `data/helionis/lunarfire_modulus_fusion_v1_0.md`
- `data/helionis/lunarfire_modulus_fusion_v1_0.png`

## What v1.0 Measures

The control twin takes top rows from `run_bpr_coupled_v09_sweep()` and computes:

- midplane field from plug field and mirror ratio
- per-update physical drift allowance
- sensor-error allowance
- required field correction per control update
- actuator slew fraction
- latency margin
- equilibrium residual
- numerical drift fraction
- controllability score

## Result

At the default `1 ms` update period, the best control-score row is controllable:

| Metric | Value |
| --- | --- |
| Controllability score | 0.656 |
| Controllable | True |
| Update period | 1.0 ms |
| Coil command fraction | 0.397 |
| Equilibrium residual | 0.064 |
| Numerical drift fraction | 0.0 |
| Physical drift fraction per update | 0.001 |
| Sensor error fraction | 0.002 |

The best plant-net v0.9 geometry remains near `-3.2 MW`, while the strongest
control-score row is lower in plant-net power. That is a valuable
design signal: the geometry that is best for plant power is not automatically
the geometry with the cleanest control margin.

## Interpretation

Modulus Fusion is best framed as the real-time control brain for Helionis:

1. Helionis chooses the reactor architecture and operating point.
2. BPR-coupled screening narrows the mirror/nozzle design space.
3. Modulus Fusion evaluates whether the resulting geometry can be controlled in
   real time without adding numerical drift.
4. CAD should be parametric around the geometry that is both power-plausible and
   control-plausible.

## What “Zero Drift” Means

It means:

- exact deterministic control updates introduce no accumulated numerical
  roundoff drift
- the controller can separate numerical error from physical drift and sensor
  error

It does not mean:

- the plasma never moves
- sensors have zero noise
- coils have infinite bandwidth
- the system is validated as a working MHD controller

## Next Step

The next useful step is a v1.1 control/CAD interface:

- expose coil locations and nozzle length as explicit geometry variables
- compute actuator authority from coil spacing and field gradients
- pass those dimensions into a parametric CAD envelope
- feed CAD-derived coil/collector/radiator constraints back into the v1.0
  control score

The conclusion for now:

> Start CAD after the control twin, not before it. The CAD model should be
> constrained by the exact-math controller and the direct-energy-conversion
> nozzle geometry.
