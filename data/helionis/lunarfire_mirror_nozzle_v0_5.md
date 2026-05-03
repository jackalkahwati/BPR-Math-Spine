# LunarFire v0.5 Mirror/Nozzle Output

Best plant-net power: `4.6 MW`.
Best mirror ratio: `5.0`.
Best collector voltage: `1500 kV`.

| Plant MW | T keV | n m^-3 | tau s | Mirror ratio | Midplane T | Plug T | Collector kV | Direct eta | End loss | Transport | Gross MW | Radiator m2 | Plug mass t | Eff magnet t | Closes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.6 | 200 | 3.5e+20 | 20 | 5.0 | 12.5 | 62.6 | 1500 | 0.88 | 0.36 | 0.42 | 667 | 33382 | 501 | 730 | True |
| 2.7 | 200 | 3.5e+20 | 16 | 5.0 | 12.5 | 62.6 | 1500 | 0.88 | 0.36 | 0.42 | 699 | 35115 | 501 | 740 | True |
| 1.8 | 200 | 2.5e+20 | 20 | 5.0 | 10.6 | 52.9 | 1500 | 0.88 | 0.36 | 0.42 | 720 | 36221 | 358 | 689 | True |
| -1.0 | 200 | 3.5e+20 | 12 | 5.0 | 12.5 | 62.6 | 1500 | 0.88 | 0.36 | 0.42 | 761 | 38413 | 501 | 759 | False |
| -1.5 | 200 | 2.5e+20 | 16 | 5.0 | 10.6 | 52.9 | 1500 | 0.88 | 0.36 | 0.42 | 775 | 39128 | 358 | 713 | False |
| -1.5 | 200 | 3.5e+20 | 20 | 4.0 | 12.5 | 50.1 | 1500 | 0.87 | 0.44 | 0.50 | 799 | 40339 | 282 | 552 | False |
| -4.8 | 200 | 3.5e+20 | 16 | 4.0 | 12.5 | 50.1 | 1500 | 0.87 | 0.44 | 0.50 | 855 | 43362 | 282 | 569 | False |
| -7.2 | 200 | 2.5e+20 | 20 | 4.0 | 10.6 | 42.3 | 1500 | 0.87 | 0.44 | 0.50 | 894 | 45399 | 201 | 608 | False |
| -8.2 | 200 | 2.5e+20 | 12 | 5.0 | 10.6 | 52.9 | 1500 | 0.88 | 0.36 | 0.42 | 886 | 45089 | 358 | 761 | False |
| -10.6 | 200 | 3.5e+20 | 8 | 5.0 | 12.5 | 62.6 | 1500 | 0.88 | 0.36 | 0.42 | 924 | 47132 | 501 | 810 | False |
| -11.6 | 200 | 3.5e+20 | 12 | 4.0 | 12.5 | 50.1 | 1500 | 0.87 | 0.44 | 0.50 | 970 | 49481 | 282 | 605 | False |
| -13.4 | 200 | 2.5e+20 | 16 | 4.0 | 10.6 | 42.3 | 1500 | 0.87 | 0.44 | 0.50 | 997 | 50919 | 201 | 652 | False |

Interpretation notes:

- Mirror ratio lowers the explicit end-loss proxy but raises plug-field burden.
- Collector voltage changes the direct-conversion efficiency proxy.
- This is still an order-of-magnitude model, not a mirror stability calculation.
