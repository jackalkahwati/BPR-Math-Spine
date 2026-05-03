# LunarFire v0.7 Shared-Grid Architecture Output

This is a shared-grid/shared-accounting FRC vs mirror/nozzle comparison.
Top plant-net architecture: `mirror_nozzle`.
Top plant-net power: `-18.2 MW`.
Plant-net status: `does not close`.

| Family | Plant MW | Closes | Gross MW | Load MW | Collector MW | Reject MW | Transport | Direct eta | R m | L m | B T | Plug T | Radiator m2 | Warnings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mirror_nozzle | -18.2 | False | 904 | 68.2 | 9.8 | 918 | 0.14 | 0.84 | 0.77 | 7.69 | 12.5 | 62.6 | 46504 | D-D side reactions included as neutron source; plant-net includes first-order engineering loads; does not close plant-net power; bremsstrahlung dominates engineering design; v0.7 shared-grid/shared-accounting architecture comparison; mirror leakage uses pitch-angle-scattering proxy; collector/nozzle auxiliary load included: 9.8 MW |
| frc | -86.9 | False | 2433 | 136.9 | 0.0 | 2507 | 0.15 | 0.78 | 1.05 | 6.33 | 10.9 | N/A | 127008 | D-D side reactions included as neutron source; plant-net includes first-order engineering loads; does not close plant-net power; bremsstrahlung dominates engineering design; v0.7 shared-grid/shared-accounting architecture comparison; FRC row uses same target/grid/engineering path |

Interpretation notes:

- Both rows use the same target, plasma grid, thermal conversion, and engineering-net path.
- Direct-conversion and transport assumptions remain architecture-specific.
- Mirror/nozzle uses a pitch-angle-scattering leakage proxy.
- Mirror/nozzle includes a first-order collector/nozzle auxiliary load.
- FRC remains a baseline row here, not a high-fidelity FRC physics model.
