# Modulus Fusion Control Twin

Best ranked-row score: `0.656`.
Best row controllable: `True`.
Drift claim: `zero numerical drift in deterministic control math`.

| Score | Controllable | Update ms | Coil cmd | Eq residual | Numerical drift | Physical drift | Sensor error | R | Aspect | Plant MW |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.656 | True | 1.00 | 0.397 | 0.064 | 0.0e+00 | 1.000e-03 | 2.000e-03 | 5.0 | 8 | -5.1 |
| 0.655 | True | 1.00 | 0.397 | 0.065 | 0.0e+00 | 1.000e-03 | 2.000e-03 | 4.0 | 8 | -4.8 |
| 0.639 | True | 1.00 | 0.469 | 0.064 | 0.0e+00 | 1.000e-03 | 2.000e-03 | 5.0 | 8 | -4.4 |
| 0.639 | True | 1.00 | 0.397 | 0.070 | 0.0e+00 | 1.000e-03 | 2.000e-03 | 5.0 | 8 | -4.8 |
| 0.638 | True | 1.00 | 0.469 | 0.065 | 0.0e+00 | 1.000e-03 | 2.000e-03 | 4.0 | 8 | -3.4 |
| 0.633 | True | 1.00 | 0.397 | 0.072 | 0.0e+00 | 1.000e-03 | 2.000e-03 | 4.0 | 8 | -4.4 |
| 0.622 | True | 1.00 | 0.469 | 0.070 | 0.0e+00 | 1.000e-03 | 2.000e-03 | 5.0 | 8 | -4.2 |
| 0.621 | True | 1.00 | 0.397 | 0.076 | 0.0e+00 | 1.000e-03 | 2.000e-03 | 5.0 | 10 | -5.2 |
| 0.620 | True | 1.00 | 0.397 | 0.076 | 0.0e+00 | 1.000e-03 | 2.000e-03 | 4.0 | 10 | -4.8 |
| 0.616 | True | 1.00 | 0.469 | 0.072 | 0.0e+00 | 1.000e-03 | 2.000e-03 | 4.0 | 8 | -3.2 |
| 0.604 | True | 1.00 | 0.469 | 0.076 | 0.0e+00 | 1.000e-03 | 2.000e-03 | 5.0 | 10 | -4.4 |
| 0.603 | True | 1.00 | 0.469 | 0.076 | 0.0e+00 | 1.000e-03 | 2.000e-03 | 4.0 | 10 | -3.5 |

Interpretation notes:

- `zero numerical drift` means the deterministic control math adds no roundoff drift.
- This is not zero plasma motion, zero sensor noise, or zero hardware latency.
- The current twin is a control-screening model, not a validated MHD controller.
