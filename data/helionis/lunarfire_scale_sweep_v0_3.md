# LunarFire v0.3 Minimum Viable Scale Output

Model label: `order_of_magnitude_trade_study`.

No tested target closes plant-net power under current assumptions.

| Target MW | Plant MW | Margin % | Gross MW | Load MW | R m | L m | B T | Radiator m2 | Closes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | -15.3 | -152.7 | 368 | 25.3 | 0.56 | 3.37 | 10.88 | 19302 | False |
| 25 | -33.5 | -134.0 | 919 | 58.5 | 0.76 | 4.57 | 10.88 | 48018 | False |
| 50 | -63.9 | -127.7 | 1838 | 113.9 | 0.96 | 5.76 | 10.88 | 95879 | False |
| 100 | -124.6 | -124.6 | 3677 | 224.6 | 1.21 | 7.26 | 10.88 | 191601 | False |
| 250 | -306.9 | -122.8 | 9192 | 556.9 | 1.64 | 9.86 | 10.88 | 478767 | False |

Interpretation notes:

- Targets are screening-net power levels, not guaranteed delivered plant output.
- Plant-net power subtracts first-order engineering loads from each target case.
- Positive plant-net means the target closes under the current rough assumptions.
