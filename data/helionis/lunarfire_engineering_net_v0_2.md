# LunarFire v0.2 Engineering Net-Power Output

Model label: `order_of_magnitude_trade_study`.

Top-ranked engineering case:

- Screening net power: 10.00 MW
- Engineering load: 25.27 MW
- Plant net power: -15.27 MW
- Closes plant net: False
- Current drive: 7.35 MW
- Formation average: 1.50 MW
- Cryogenic wall power: 0.32 MW
- Power conditioning loss: 12.69 MW
- Fusion conversion waste heat: 50.51 MW
- Thermal rejection parasitic: 1.91 MW
- Rejected heat: 381.05 MW
- Radiator area: 19302 m^2
- Warnings: D-D side reactions included as neutron source; reference design uses optimistic transport/direct-conversion assumptions; high-field HTS magnet regime; plant-net includes first-order engineering loads; does not close plant-net power; bremsstrahlung dominates engineering design

| Rank | Plant MW | Load MW | Current MW | Cond MW | Cryo MW | Reject MW | Radiator m2 | Closes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | -15.3 | 25.3 | 7.4 | 12.7 | 0.3 | 381.1 | 19302 | False |
| 2 | -16.0 | 26.0 | 7.6 | 13.1 | 0.3 | 393.9 | 19951 | False |
| 3 | -16.6 | 26.6 | 7.8 | 13.4 | 0.4 | 403.7 | 20447 | False |
| 4 | -17.3 | 27.3 | 8.0 | 13.9 | 0.3 | 417.3 | 21135 | False |
| 5 | -17.8 | 27.8 | 8.2 | 14.1 | 0.4 | 424.5 | 21504 | False |
| 6 | -18.8 | 28.8 | 8.4 | 14.6 | 0.6 | 438.2 | 22198 | False |
| 7 | -20.1 | 30.1 | 8.9 | 15.4 | 0.5 | 464.5 | 23527 | False |
| 8 | -20.5 | 30.5 | 9.1 | 15.7 | 0.4 | 473.3 | 23972 | False |
| 9 | -20.8 | 30.8 | 9.1 | 15.7 | 0.6 | 473.5 | 23985 | False |
| 10 | -24.7 | 34.7 | 10.5 | 18.1 | 0.4 | 546.3 | 27673 | False |
| 11 | -26.3 | 36.3 | 11.0 | 18.9 | 0.6 | 571.6 | 28952 | False |
| 12 | -38.6 | 48.6 | 15.1 | 26.0 | 0.6 | 788.3 | 39928 | False |

Interpretation notes:

- Plant-net power subtracts first-order engineering loads from screening-net power.
- This still omits detailed coil stress, quench protection, wall design, and full balance-of-plant engineering.
- Negative plant-net power means the reference point misses after internal reactor loads.
