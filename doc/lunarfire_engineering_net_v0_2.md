# LunarFire v0.2 Engineering Net-Power Budget

LunarFire v0.1 closed a **10 MW screening-net** FRC point. LunarFire v0.2
answers the harder question:

> Does that reference point still close after first-order reactor engineering
> loads are subtracted?

Current answer: **no**.

The best v0.2 case misses plant-net power by about **15.3 MW** under the first
engineering budget.

## Generated Outputs

Run the budget with:

```bash
python3 scripts/lunarfire_engineering_net.py --output-dir data/helionis
```

Generated files:

- `data/helionis/lunarfire_engineering_net_v0_2.csv`
- `data/helionis/lunarfire_engineering_net_v0_2.md`
- `data/helionis/lunarfire_engineering_net_v0_2.png`

## What Changed From v0.1

v0.1 reported screening-net power:

`screening_net = useful_converted_power - bremsstrahlung_loss - transport_loss`

v0.2 subtracts first-order engineering loads:

`plant_net = screening_net - engineering_loads`

Engineering loads currently include:

- RMF/current-drive power.
- Pulsed formation average power.
- HTS cryogenic wall power.
- Direct-conversion / power-conditioning losses.
- Thermal-rejection parasitic load.
- Fixed balance-of-plant load.

This is still not a full plant model. It does not include detailed coil stress,
quench protection, first-wall design, vacuum systems, startup sequence,
structural mass, fault handling, or full thermal architecture.

## Top Engineering Case

The best v0.2 engineering case:

- Screening-net power: 10.00 MW.
- Engineering load: 25.27 MW.
- Plant-net power: -15.27 MW.
- Closes plant net: false.
- Current drive: 7.35 MW.
- Formation average: 1.50 MW.
- Cryogenic wall power: 0.32 MW.
- Power-conditioning loss: 12.69 MW.
- Fusion conversion waste heat: 50.51 MW.
- Thermal-rejection parasitic: 1.91 MW.
- Rejected heat: 381.05 MW.
- Radiator area: 19,302 m2.

The strongest conclusion is that power conditioning and current drive are now
as important as the plasma math. Even if the screening plasma point closes,
the machine does not yet close as a reactor.

## Why It Misses

The dominant issue remains bremsstrahlung. The engineering budget does not add
bremsstrahlung as a new loss; v0.1 already subtracted it. But v0.2 shows what
bremsstrahlung does to the surrounding machine:

- It creates a large rejected-heat burden.
- It drives radiator area.
- It reduces margin for current drive and power conditioning.
- It makes a 10 MW product look like a much larger thermal-management problem.

The next largest engineering loads are:

- Power conditioning: roughly 12.7 MW in the best case.
- Current drive: roughly 7.4 MW in the best case.
- Thermal rejection parasitic: roughly 1.7 MW in the best case.

Cryogenic wall power is not the first-order blocker in this rough model. That
will change once coil stress, quench margin, radiation heating, and real HTS
coil-pack design are included.

## Design Implication

The current LunarFire FRC concept is not dead. It is now correctly constrained.

The design target should change from:

> Can we close 10 MW screening-net?

to:

> Can we create at least 25-30 MW of screening-net margin at the same compact
> geometry, or cut engineering loads enough for 10 MW plant-net to close?

That means the next design work should focus on four levers:

1. **Reduce current-drive burden**
   Model RMF/current-drive physics instead of using a gross-fusion fraction.

2. **Reduce power-conditioning losses**
   The direct-conversion chain must be designed explicitly. A few percent loss
   on hundreds of MW is a product-scale problem.

3. **Improve radiation handling**
   Bremsstrahlung recovery, reflection, spectral handling, or high-temperature
   rejection must move from an assumption to a subsystem model.

4. **Increase screening-net margin**
   The reference point barely clears gain proxy 1.0. It needs margin before
   engineering loads can be absorbed.

## Go / No-Go Status

Current status:

- Geometry: FRC remains the right first architecture.
- Screening plasma point: closes, but barely.
- Engineering plant-net: does not close.
- Main blocker: not He3 fuel supply; reactor-side power accounting.

The next credible milestone is:

> LunarFire v0.3 should search for operating points that produce positive
> plant-net power after engineering loads, not just positive screening-net
> power.
