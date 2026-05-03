# LunarFire 10 MW FRC Reference Design v0.1

LunarFire should use **10 MW screening-net power** as the first serious
product-scale reference point. Smaller systems are still useful as subsystem
testbeds, but the 10 MW point is where the reactor starts to look like a
lunar/space power module rather than a lab apparatus.

This document summarizes the first 10 MW FRC sizing sweep. It is a reference
design screen, not an engineering drawing and not a commercial feasibility
claim.

## Generated Outputs

Run the sweep with:

```bash
python3 scripts/lunarfire_reference_design.py --output-dir data/helionis
```

Generated files:

- `data/helionis/lunarfire_reference_design_10mw.csv`
- `data/helionis/lunarfire_reference_design_10mw.md`
- `data/helionis/lunarfire_reference_design_10mw.png`

## Reference Target

- Screening-net output target: 10 MW.
- Fuel: D-He3.
- Geometry: FRC-centered linear plasmoid.
- Magnet architecture: HTS solenoids plus pulsed formation coils and end
  mirror/nozzle coils.
- Beta target: 0.85.
- Direct conversion efficiency assumption: 78%.
- Thermal residual conversion efficiency assumption: 38%.
- Transport loss multiplier: 0.15.
- D-D side-reaction fraction: 3%.
- Maximum field in the sweep: 12 T.
- Maximum separatrix radius: 3 m.
- Maximum FRC length: 20 m.

These are optimistic assumptions. The point of v0.1 is to find the shape of the
design window, not to prove the machine is buildable.

## Top Candidate

The top-ranked candidate closes exactly 10 MW screening-net power under the
current model:

- Temperature: 200 keV.
- Ion density: 3.50e20 m^-3.
- Confinement time: 20 s.
- FRC radius: 0.73 m.
- FRC length: 4.36 m.
- Required magnetic field: 9.11 T.
- Plasma volume: 6.13 m^3.
- Gross fusion power: 389.03 MW.
- Useful converted power before modeled losses: 335.59 MW.
- Bremsstrahlung loss: 323.66 MW.
- Transport loss: 1.94 MW.
- Neutron power: 0.77 MW.
- Neutron wall load: 0.0334 MW/m^2.
- Magnetic energy proxy: 202 MJ.
- Magnet mass proxy: 66.8 tonnes.

The uncomfortable result is important: the design closes only because the model
uses optimistic transport and direct-conversion assumptions. Gross fusion power
is nearly 39x the target screening-net output because bremsstrahlung dominates
the loss budget. This does not include recirculating power, cryogenics, current
drive, formation power, thermal rejection, or fixed plant loads.

## Interpretation

The 10 MW point is still the right reference target, but LunarFire should treat
this first closure as a **stress-case operating point**, not as a clean product
design.

The model says:

- D-He3 FRC geometry can be compact on paper.
- Neutron burden stays low enough to preserve the LunarFire strategic thesis.
- HTS field levels around 9-11 T are in the right regime for a compact design.
- Bremsstrahlung and confinement are the central blockers.
- Direct conversion must be genuinely high-efficiency; it cannot be marketing.

## Why Not Start Smaller

The sweep reinforces the earlier intuition: smaller-than-10 MW systems are
probably test articles, not first products.

At small scale, fixed losses and fixed subsystems dominate:

- Formation power.
- Cryogenic load.
- Magnet stored energy.
- Control systems.
- Shielding and structure.
- Direct-conversion hardware.
- Thermal rejection.

A 100 kW or 1 MW machine may be the right hardware demonstrator, but the design
math should be anchored to a 10 MW product reference so the subsystem tests are
aimed at a useful machine.

## Design Implications

The next reactor-design step is not CAD. It is to replace the optimistic knobs
with deeper subsystem models:

1. **Bremsstrahlung mitigation and recovery**
   The top candidate loses 323.66 MW to bremsstrahlung in the current model.
   LunarFire needs a more serious radiation model and an explicit recovery or
   rejection strategy.

2. **Confinement scaling**
   The top candidate assumes 20 s confinement at 200 keV. That is an aggressive
   requirement for an FRC. The next model needs confinement-time scaling rather
   than a free input.

3. **Recirculating power**
   Formation coils, RMF/current drive, cryogenics, power electronics, and
   direct-conversion conditioning must be modeled as internal loads.

4. **Magnet engineering**
   The 9.11 T reference case belongs in HTS territory. The next model should
   include coil pack radius, current density, stress, stored energy, quench
   margin, and cryogenic load.

5. **Direct-conversion geometry**
   The FRC end regions need a first-pass charged-particle expansion/nozzle model
   rather than a single efficiency scalar.

## Current Conclusion

Continue with the 10 MW FRC design point, but label it honestly:

> LunarFire 10 MW FRC v0.1 is a compact, high-field, high-temperature stress
> case that identifies the subsystem requirements a D-He3 reactor must beat.

The next credible milestone is not "reactor designed." It is:

> Show which assumptions must improve for 10 MW screening-net power to close
> with margin before recirculating and plant loads are added.
