# Helionis Geometry Downselect Output

Model label: `order_of_magnitude_trade_study`.

| Scenario | Family | Score | B T | R minor m | L m | Wall MW/m2 | Direct | Rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lunar_infrastructure_dhe3 | frc | 0.715 | 5.44 | 1.71 | 10.26 | 0.0239 | 0.95 | Best fit for high-beta compact D-He3 with direct end access. |
| lunar_infrastructure_dhe3 | mirror | 0.620 | 7.48 | 1.37 | 13.66 | 0.0238 | 0.88 | Strong direct-conversion access, penalized for end-loss confinement risk. |
| lunar_infrastructure_dhe3 | spherical_torus | 0.477 | 9.48 | 1.06 | 5.73 | 0.0264 | 0.32 | Mature equilibrium baseline, penalized for toroidal mass and poor charged-product access. |
| orbital_data_center_dhe3 | frc | 0.667 | 6.53 | 2.24 | 13.44 | 0.0492 | 0.95 | Best fit for high-beta compact D-He3 with direct end access. |
| orbital_data_center_dhe3 | mirror | 0.580 | 8.97 | 1.79 | 17.89 | 0.0491 | 0.88 | Strong direct-conversion access, penalized for end-loss confinement risk. |
| orbital_data_center_dhe3 | spherical_torus | 0.453 | 11.38 | 1.39 | 7.50 | 0.0543 | 0.32 | Mature equilibrium baseline, penalized for toroidal mass and poor charged-product access. |
| compact_space_reactor_dhe3 | frc | 0.744 | 4.62 | 1.36 | 8.14 | 0.0133 | 0.95 | Best fit for high-beta compact D-He3 with direct end access. |
| compact_space_reactor_dhe3 | mirror | 0.656 | 6.35 | 1.08 | 10.84 | 0.0133 | 0.88 | Strong direct-conversion access, penalized for end-loss confinement risk. |
| compact_space_reactor_dhe3 | spherical_torus | 0.509 | 8.04 | 0.84 | 4.54 | 0.0147 | 0.32 | Mature equilibrium baseline, penalized for toroidal mass and poor charged-product access. |
| terrestrial_demonstrator_dhe3 | frc | 0.698 | 6.66 | 1.96 | 11.74 | 0.0328 | 0.95 | Best fit for high-beta compact D-He3 with direct end access. |
| terrestrial_demonstrator_dhe3 | mirror | 0.606 | 9.16 | 1.56 | 15.63 | 0.0327 | 0.88 | Strong direct-conversion access, penalized for end-loss confinement risk. |
| terrestrial_demonstrator_dhe3 | spherical_torus | 0.474 | 11.61 | 1.21 | 6.55 | 0.0362 | 0.32 | Mature equilibrium baseline, penalized for toroidal mass and poor charged-product access. |

Interpretation notes:

- Scores are zero-shot architecture fit metrics, not final reactor performance.
- FRC is expected to rank first when compactness and direct charged-particle access matter.
- Magnetic mass is a comparative proxy from magnetic energy and field strength.
- Neutron wall load comes from the existing D-He3 trade-study side-reaction model.
