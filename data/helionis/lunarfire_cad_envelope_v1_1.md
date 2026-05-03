# LunarFire v1.1 Parametric CAD Envelope

Best CAD-readiness score: `0.463`.
Best row CAD-ready: `False`.

No CAD-ready rows. Readiness blockers: `control row is not controllable` (3), `outer radius exceeds CAD envelope limit` (2), `radiator wing span exceeds CAD envelope limit` (12), `source plant is net-negative` (12).

| CAD score | Ready | Length m | Outer R m | Plasma R m | Nozzle m | Collector m2 | Radiator m2 | Wing span m | Control score | Plant MW | Blockers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.463 | False | 21.2 | 13.6 | 0.96 | 6.0 | 408 | 39683 | 936 | 0.656 | -5.1 | radiator wing span exceeds CAD envelope limit; source plant is net-negative |
| 0.457 | False | 21.1 | 13.5 | 0.95 | 6.0 | 405 | 39361 | 931 | 0.639 | -4.8 | radiator wing span exceeds CAD envelope limit; source plant is net-negative |
| 0.438 | False | 21.6 | 14.6 | 0.89 | 5.5 | 408 | 39727 | 921 | 0.621 | -5.2 | radiator wing span exceeds CAD envelope limit; source plant is net-negative |
| 0.436 | False | 17.1 | 16.6 | 0.76 | 4.7 | 399 | 38772 | 1131 | 0.639 | -4.4 | radiator wing span exceeds CAD envelope limit; source plant is net-negative |
| 0.434 | False | 21.5 | 14.5 | 0.89 | 5.5 | 405 | 39400 | 915 | 0.607 | -4.8 | radiator wing span exceeds CAD envelope limit; control row is not controllable; source plant is net-negative |
| 0.433 | False | 18.9 | 16.9 | 0.96 | 4.8 | 413 | 40131 | 1063 | 0.655 | -4.8 | radiator wing span exceeds CAD envelope limit; source plant is net-negative |
| 0.431 | False | 17.1 | 16.5 | 0.76 | 4.7 | 397 | 38552 | 1126 | 0.622 | -4.2 | radiator wing span exceeds CAD envelope limit; source plant is net-negative |
| 0.426 | False | 18.8 | 16.8 | 0.96 | 4.8 | 409 | 39798 | 1057 | 0.633 | -4.4 | radiator wing span exceeds CAD envelope limit; source plant is net-negative |
| 0.426 | False | 15.3 | 20.6 | 0.76 | 3.8 | 402 | 39038 | 1277 | 0.638 | -3.4 | outer radius exceeds CAD envelope limit; radiator wing span exceeds CAD envelope limit; source plant is net-negative |
| 0.423 | False | 22.0 | 15.4 | 0.83 | 5.2 | 407 | 39553 | 897 | 0.607 | -5.0 | radiator wing span exceeds CAD envelope limit; control row is not controllable; source plant is net-negative |
| 0.422 | False | 22.1 | 15.5 | 0.84 | 5.2 | 410 | 39899 | 903 | 0.607 | -5.4 | radiator wing span exceeds CAD envelope limit; control row is not controllable; source plant is net-negative |
| 0.418 | False | 15.3 | 20.5 | 0.76 | 3.8 | 400 | 38812 | 1272 | 0.616 | -3.2 | outer radius exceeds CAD envelope limit; radiator wing span exceeds CAD envelope limit; source plant is net-negative |

Interpretation notes:

- v1.1 is a control-constrained parametric envelope, not detailed CAD.
- Dimensions are derived from Helionis v0.9 geometry and Modulus Fusion v1.0 control rows.
- The collector, nozzle, coil, and radiator envelopes are sizing constraints for the next CAD pass.
