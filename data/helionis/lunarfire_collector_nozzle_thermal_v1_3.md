# LunarFire v1.3 Collector/Nozzle Thermal Architecture

Lowest heuristic-aggressiveness CAD-ready collector/nozzle recipe:

- Plant-net power: `0.9 MW`
- Collector capture efficiency: `0.08`
- Nozzle capture efficiency: `0.00`
- Radiator temperature: `1000 K`
- Topology packing factor: `1.0`
- Recovered electric power: `5.7 MW`
- Recoverable channel heat: `106.0 MW`
- Adjusted wing span per side: `344 m`

Highest-score CAD-ready row: `0.638` at `7.8 MW` plant-net.

| Ready | Plant MW | Agg | Collector eta | Nozzle eta | Recovered MW | Recoverable MW | Brem MW | Transport MW | Collector MW | Nozzle MW | Rad K | Pack | Span m | Blockers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | 0.9 | 3.0 | 0.08 | 0.00 | 5.7 | 106.0 | 480 | 120 | 71 | 21 | 1000 | 1.0 | 344 | none |
| True | 0.6 | 3.0 | 0.08 | 0.00 | 5.7 | 106.8 | 484 | 121 | 71 | 21 | 1000 | 1.0 | 346 | none |
| True | 0.5 | 3.0 | 0.08 | 0.00 | 5.7 | 106.9 | 485 | 121 | 71 | 21 | 1000 | 1.0 | 340 | none |
| True | 1.4 | 3.0 | 0.08 | 0.00 | 5.7 | 107.1 | 486 | 121 | 71 | 21 | 1000 | 1.0 | 390 | none |
| True | 1.0 | 3.0 | 0.08 | 0.00 | 5.8 | 108.0 | 490 | 122 | 72 | 22 | 1000 | 1.0 | 393 | none |
| True | 1.4 | 3.0 | 0.08 | 0.00 | 5.5 | 103.8 | 470 | 118 | 69 | 21 | 1000 | 1.0 | 416 | none |
| True | 0.9 | 3.0 | 0.08 | 0.00 | 5.7 | 106.0 | 480 | 120 | 71 | 21 | 800 | 2.0 | 420 | none |
| True | 1.2 | 3.0 | 0.08 | 0.00 | 5.6 | 104.4 | 473 | 118 | 70 | 21 | 1000 | 1.0 | 418 | none |
| True | 0.6 | 3.0 | 0.08 | 0.00 | 5.7 | 106.8 | 484 | 121 | 71 | 21 | 800 | 2.0 | 422 | none |
| True | 1.1 | 3.0 | 0.08 | 0.00 | 5.6 | 104.5 | 474 | 118 | 70 | 21 | 1000 | 1.0 | 411 | none |
| True | 0.5 | 3.0 | 0.08 | 0.00 | 5.7 | 106.9 | 485 | 121 | 71 | 21 | 800 | 2.0 | 415 | none |
| True | 1.4 | 3.0 | 0.08 | 0.00 | 5.7 | 107.1 | 486 | 121 | 71 | 21 | 800 | 2.0 | 476 | none |
| True | 1.0 | 3.0 | 0.08 | 0.00 | 5.8 | 108.0 | 490 | 122 | 72 | 22 | 800 | 2.0 | 479 | none |
| True | 0.1 | 3.5 | 0.06 | 0.03 | 4.9 | 106.0 | 480 | 120 | 71 | 21 | 1000 | 1.0 | 344 | none |
| True | 0.6 | 3.5 | 0.06 | 0.03 | 4.9 | 107.1 | 486 | 121 | 71 | 21 | 1000 | 1.0 | 391 | none |
| True | 0.2 | 3.5 | 0.06 | 0.03 | 5.0 | 108.0 | 490 | 122 | 72 | 22 | 1000 | 1.0 | 393 | none |
| True | 0.6 | 3.5 | 0.06 | 0.03 | 4.8 | 103.8 | 470 | 118 | 69 | 21 | 1000 | 1.0 | 416 | none |
| True | 0.1 | 3.5 | 0.06 | 0.03 | 4.9 | 106.0 | 480 | 120 | 71 | 21 | 800 | 2.0 | 420 | none |
| True | 0.4 | 3.5 | 0.06 | 0.03 | 4.8 | 104.4 | 473 | 118 | 70 | 21 | 1000 | 1.0 | 418 | none |
| True | 0.4 | 3.5 | 0.06 | 0.03 | 4.8 | 104.5 | 474 | 118 | 70 | 21 | 1000 | 1.0 | 411 | none |
| True | 0.6 | 3.5 | 0.06 | 0.03 | 4.9 | 107.1 | 486 | 121 | 71 | 21 | 800 | 2.0 | 477 | none |
| True | 0.2 | 3.5 | 0.06 | 0.03 | 5.0 | 108.0 | 490 | 122 | 72 | 22 | 800 | 2.0 | 480 | none |
| True | 0.9 | 4.0 | 0.08 | 0.00 | 5.7 | 106.0 | 480 | 120 | 71 | 21 | 1200 | 1.0 | 166 | none |
| True | 0.9 | 4.0 | 0.08 | 0.00 | 5.7 | 106.0 | 480 | 120 | 71 | 21 | 1000 | 2.0 | 172 | none |

Interpretation notes:

- v1.3 splits rejected heat into bremsstrahlung, transport, collector, nozzle, and power-conditioning channels.
- Only collector/nozzle/conditioning channels are recoverable channel heat in this screen.
- Bremsstrahlung and transport heat remain radiator load.
- A CAD-ready row must close plant-net and keep residual radiator span/area inside constraints.
- CAD-ready here means parametric envelope-ready, not detailed CAD or validated thermal hardware.
