# LunarFire v1.2 Thermal Packaging Recovery

Minimum-assumption CAD-ready thermal recovery recipe:

- Plant-net power: `2.3 MW`
- Direct heat recovery: `0.01`
- Radiator temperature: `1000 K`
- Topology packing factor: `1.0`
- Adjusted radiator area: `14510 m2`
- Adjusted wing span per side: `343 m`

Highest-score CAD-ready row: `0.743` at `23.4 MW` plant-net.

| Ready | Plant MW | Agg | Recover frac | Recovered MW | Rad K | Pack | Heat MW | Area m2 | Span m | Blockers | Source ID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | 2.3 | 2.0 | 0.01 | 7.1 | 1000 | 1.0 | 699 | 14510 | 343 | none | r0.954_l21.149_or13.513_plant-4.756 |
| True | 2.0 | 2.0 | 0.01 | 7.1 | 1000 | 1.0 | 705 | 14629 | 345 | none | r0.956_l21.199_or13.578_plant-5.137 |
| True | 1.9 | 2.0 | 0.01 | 7.1 | 1000 | 1.0 | 706 | 14645 | 339 | none | r0.888_l21.576_or14.574_plant-5.190 |
| True | 2.8 | 2.0 | 0.01 | 7.1 | 1000 | 1.0 | 707 | 14671 | 390 | none | r0.957_l18.832_or16.815_plant-4.356 |
| True | 2.5 | 2.0 | 0.01 | 7.2 | 1000 | 1.0 | 713 | 14794 | 392 | none | r0.960_l18.877_or16.899_plant-4.750 |
| True | 2.3 | 2.0 | 0.01 | 7.1 | 800 | 2.0 | 699 | 35425 | 419 | none | r0.954_l21.149_or13.513_plant-4.756 |
| True | 2.8 | 2.0 | 0.01 | 6.9 | 1000 | 1.0 | 685 | 14212 | 415 | none | r0.757_l17.117_or16.498_plant-4.155 |
| True | 2.5 | 2.0 | 0.01 | 7.0 | 1000 | 1.0 | 689 | 14293 | 417 | none | r0.758_l17.145_or16.554_plant-4.414 |
| True | 2.0 | 2.0 | 0.01 | 7.1 | 800 | 2.0 | 705 | 35714 | 421 | none | r0.956_l21.199_or13.578_plant-5.137 |
| True | 2.5 | 2.0 | 0.01 | 7.0 | 1000 | 1.0 | 689 | 14304 | 410 | none | r0.704_l17.442_or17.779_plant-4.449 |
| True | 1.9 | 2.0 | 0.01 | 7.1 | 800 | 2.0 | 706 | 35754 | 414 | none | r0.888_l21.576_or14.574_plant-5.190 |
| True | 2.8 | 2.0 | 0.01 | 7.1 | 800 | 2.0 | 707 | 35819 | 475 | none | r0.957_l18.832_or16.815_plant-4.356 |
| True | 2.5 | 2.0 | 0.01 | 7.2 | 800 | 2.0 | 713 | 36118 | 478 | none | r0.960_l18.877_or16.899_plant-4.750 |
| True | 9.4 | 3.0 | 0.02 | 14.1 | 1000 | 1.0 | 692 | 14363 | 340 | none | r0.954_l21.149_or13.513_plant-4.756 |
| True | 9.1 | 3.0 | 0.02 | 14.2 | 1000 | 1.0 | 698 | 14481 | 342 | none | r0.956_l21.199_or13.578_plant-5.137 |
| True | 9.1 | 3.0 | 0.02 | 14.3 | 1000 | 1.0 | 699 | 14497 | 336 | none | r0.888_l21.576_or14.574_plant-5.190 |
| True | 2.3 | 3.0 | 0.01 | 7.1 | 1200 | 1.0 | 699 | 6997 | 165 | none | r0.954_l21.149_or13.513_plant-4.756 |
| True | 2.3 | 3.0 | 0.01 | 7.1 | 1000 | 2.0 | 699 | 14510 | 172 | none | r0.954_l21.149_or13.513_plant-4.756 |
| True | 2.0 | 3.0 | 0.01 | 7.1 | 1200 | 1.0 | 705 | 7055 | 166 | none | r0.956_l21.199_or13.578_plant-5.137 |
| True | 2.0 | 3.0 | 0.01 | 7.1 | 1000 | 2.0 | 705 | 14629 | 173 | none | r0.956_l21.199_or13.578_plant-5.137 |
| True | 1.9 | 3.0 | 0.01 | 7.1 | 1200 | 1.0 | 706 | 7063 | 164 | none | r0.888_l21.576_or14.574_plant-5.190 |
| True | 9.9 | 3.0 | 0.02 | 14.3 | 1000 | 1.0 | 700 | 14523 | 386 | none | r0.957_l18.832_or16.815_plant-4.356 |
| True | 1.9 | 3.0 | 0.01 | 7.1 | 1000 | 2.0 | 706 | 14645 | 170 | none | r0.888_l21.576_or14.574_plant-5.190 |
| True | 9.7 | 3.0 | 0.02 | 14.4 | 1000 | 1.0 | 706 | 14644 | 388 | none | r0.960_l18.877_or16.899_plant-4.750 |

Interpretation notes:

- direct conversion recovery improves plant-net power and reduces rejected heat.
- hotter radiators and topology packing improve packaging, but do not create net power.
- CAD-ready means plant-net closes and radiator span/area fit the v1.2 constraints.
