# LunarFire v0.4 Plant-Net Geometry Re-Score Output

Model label: `order_of_magnitude_trade_study`.

Best current geometry by plant-net power: `mirror`.
Best plant-net power: `-24.7 MW`.

| Geometry | Feasible | Plant MW | Gross MW | Load MW | R m | L m | B T | Radiator m2 | Direct eta | CD frac | Transport | Rejections | Closes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mirror | True | -24.7 | 1249 | 74.7 | 0.86 | 8.56 | 12.51 | 64196 | 0.82 | 0.015 | 0.30 |  | False |
| frc | True | -59.7 | 1774 | 109.7 | 0.76 | 4.55 | 12.88 | 92398 | 0.78 | 0.020 | 0.15 |  | False |
| spherical_torus | False | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.35 | 0.010 | 0.08 | no_positive_screening_net=80 | False |

Interpretation notes:

- This is a plant-net re-score, not a final geometry selection.
- Profiles use rough geometry-specific assumptions for beta, transport, conversion, and current drive.
- Negative plant-net means the geometry misses under current assumptions.
