# Helionis Technical MVP

Helionis is the reactor-side design-intelligence layer for the helium-3
economy. The first MVP does not claim that a commercial D-He3 reactor is ready.
It quantifies the operating assumptions that would have to become true before
lunar helium-3 supply becomes valuable as fusion fuel.

## One-Week Build

The MVP is a deterministic Python trade-study package in `helionis/`. It models
reaction energy partitioning, coarse reactivity, bremsstrahlung loss, transport
loss, direct charged-particle conversion, D-D side-neutron production, and a
simple shielding mass proxy.

Run it with:

```bash
python3 scripts/helionis_trade_study.py --output-dir data/helionis
```

Generated outputs:

- `data/helionis/helionis_trade_study.csv`
- `data/helionis/helionis_trade_study.md`
- `data/helionis/helionis_trade_study.png`

## What The First Model Says

The default run compares four D-He3 architecture scenarios against one D-T
reference block. Under the current conservative loss assumptions, every default
scenario is net-negative. That is a useful result, not a failure: the MVP makes
the design gap explicit instead of hiding it behind reactor marketing.

Key first-run outputs:

- D-He3 neutron fraction stays between roughly 0.20% and 0.46% once D-D side
  reactions are included.
- The D-T reference case carries roughly 80% of fusion power in neutrons.
- D-He3 shielding proxy ranges from 5.8 to 44.9 tonnes across the default cases.
- The D-T reference shielding proxy is 977.4 tonnes in the same coarse metric.
- The best default D-He3 case reaches a gain proxy of 0.60 before any advanced
  confinement, radiation recovery, or detailed direct-conversion design.

The immediate technical implication is clear: Helionis should focus its first
validation work on loss suppression, charged-particle capture, and D-D side
reaction accounting, not on broad reactor claims.

## Model Boundaries

This is an order-of-magnitude screening model. It is suitable for scenario
comparison and investor/partner diligence, but it is not a plasma simulation.

Included:

- D-He3, D-T, and averaged D-D reaction energetics.
- Rounded Maxwellian reactivity anchors with log-log interpolation.
- Lawson-style triple-product reporting.
- Bremsstrahlung and thermal transport loss proxies.
- Direct charged-particle conversion sensitivity.
- D-D side-reaction neutron burden for D-He3 scenarios.
- Comparative shielding mass proxy tied to neutron power.

Excluded:

- Kinetic plasma behavior.
- Stability and turbulence physics.
- Magnet, wall, blanket, and divertor engineering.
- Fuel ash removal.
- Lunar mining economics.
- Tritium breeding and full fuel-cycle logistics.

## Why This Is The Right First Wedge

LH3M is working on helium-3 supply. The missing counterpart is a reactor-side
model that tells partners how much He3 demand can exist, under what assumptions,
and for which markets.

Helionis should therefore start as:

> The technical bridge between lunar helium-3 supply and real fusion demand.

The MVP creates the first version of that bridge. It converts a broad thesis
into a repeatable artifact: change the temperature, confinement, conversion
efficiency, side-reaction fraction, and shielding assumptions, then see whether
the architecture improves or collapses.

## Partner-Facing Message

LH3M mines the fuel. Helionis quantifies and designs the engine demand curve.

For a Chris/LH3M conversation, the clean ask is not "fund a reactor." It is:

> Let us jointly define the first credible reactor-side demand model for lunar
> helium-3, then use that model to identify which subsystem demonstrations make
> the lunar fuel market investable.

## Next Validation Milestones

1. Replace rounded reactivity anchors with a cited Bosch-Hale/NIST-style
   reaction-rate module.
2. Add sensitivity sweeps for electron temperature, Z_eff, confinement time,
   and direct-conversion efficiency.
3. Model charged-particle direct conversion as a subsystem with capture
   efficiency, voltage limits, and recirculating-power assumptions.
4. Add a D-D side-reaction module that depends on deuterium inventory and
   temperature instead of using a fixed fraction.
5. Build a hardware validation roadmap around charged-particle capture,
   low-neutron shielding, plasma operating-point measurement, and fuel-cycle
   accounting.

## Credibility Line

Helionis should not say: "We can build a commercial He3 reactor now."

Helionis can credibly say:

> We are building the reactor-side architecture model that tells the helium-3
> economy what has to be true before lunar fuel becomes bankable.
