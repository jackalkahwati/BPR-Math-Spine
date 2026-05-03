# Helionis Geometry Downselect

The zero-shot geometry conclusion is clear:

> Helionis should start with an FRC-centered linear reactor architecture, with
> mirror/nozzle-style end regions for charged-particle expansion and direct
> conversion.

This is not a final commercial reactor selection. It is the first simulation
geometry because it best matches the helium-3 economy wedge: compact, high-beta,
lower-neutron, direct-conversion-first fusion architecture.

## Why FRC First

D-He3 fusion pushes the reactor toward different constraints than mainstream
D-T power plants:

- It needs high operating temperature and high plasma pressure.
- It only becomes strategically interesting if lower neutron burden reduces
  shielding and activation mass.
- Most primary reaction energy is in charged protons and alpha particles, so
  the architecture should expose those products to direct conversion instead
  of defaulting to a thermal blanket.
- Lunar, orbital, and mobile space-power markets punish large toroidal systems.

FRC geometry fits that problem better than a tokamak-style first architecture.
It is linear, compact, high-beta, and naturally gives access to end regions
where charged fusion products can expand into a direct-conversion stage.

## Zero-Shot Scorecard

The new geometry model compares three first-pass candidates:

- `frc`: compact field-reversed configuration / linear plasmoid.
- `mirror`: linear mirror/nozzle geometry.
- `spherical_torus`: compact toroidal baseline.

Generated outputs:

- `data/helionis/helionis_geometry_downselect.csv`
- `data/helionis/helionis_geometry_downselect.md`
- `data/helionis/helionis_geometry_scores.png`
- `data/helionis/helionis_frc_field_map.png`

Across all default D-He3 scenarios, FRC ranks first:

- Lunar infrastructure: FRC score 0.715 vs mirror 0.620 vs spherical torus 0.477.
- Orbital data center: FRC score 0.667 vs mirror 0.580 vs spherical torus 0.453.
- Compact space reactor: FRC score 0.744 vs mirror 0.656 vs spherical torus 0.509.
- Terrestrial demonstrator: FRC score 0.698 vs mirror 0.606 vs spherical torus 0.474.

The score is not a physics proof. It is an architecture-fit metric combining
direct conversion access, compactness, stability confidence, engineering
simplicity, magnetic mass proxy, neutron wall loading, and the existing
power-balance proxy.

## What The Math Includes

The geometry layer adds 0D sizing and comparison equations:

- Plasma pressure: `p = (n_i + n_e) kT`.
- Magnetic pressure: `p_B = B^2 / 2 mu0`.
- Required field: `B = sqrt(2 mu0 p / beta)`.
- FRC volume proxy: `V = pi r_s^2 L * shape_factor`.
- Mirror volume proxy: `V = pi r^2 L`.
- Spherical torus volume proxy: `V = 2 pi^2 R a^2 kappa`.
- Neutron wall load: `P_n / surface_area`.
- Magnetic energy proxy: `p_B * V`.
- Magnet mass proxy: geometry-weighted function of magnetic energy and field.
- Geometry score: weighted fit to direct conversion, compactness, stability,
  engineering simplicity, shielding, and net-power proxy.

## First FRC Dimensions

For the default scenarios, the FRC screen produces first-pass separatrix-scale
dimensions:

- Lunar infrastructure: radius 1.71 m, length 10.26 m, field 5.44 T.
- Orbital data center: radius 2.24 m, length 13.44 m, field 6.53 T.
- Compact space reactor: radius 1.36 m, length 8.14 m, field 4.62 T.
- Terrestrial demonstrator: radius 1.96 m, length 11.74 m, field 6.66 T.

These are geometry-screening values, not engineering drawings. The next step is
to stress them with better confinement, stability, wall, and magnet models.

## Why Not Start With A Tokamak

A spherical torus is useful as a baseline because the equilibrium math is more
mature. It is not the best first Helionis architecture because it fights the
company wedge:

- Charged products are less naturally available for direct conversion.
- Toroidal coil and blanket geometry increases mass and packaging complexity.
- The architecture trends toward thermal power-plant assumptions instead of
  compact space/lunar power assumptions.

Helionis should keep spherical torus in the scorecard, but not make it the lead
design.

## Why Mirror Is Second

Mirror geometry is attractive for direct charged-particle extraction and linear
packaging. The reason it ranks second is confinement risk: end losses are the
central problem. The best near-term architecture may borrow mirror/nozzle end
regions for FRC exhaust and direct conversion rather than use a pure mirror as
the primary confinement device.

## Simulation Path

The next simulation layer should stay sequenced:

1. Improve the FRC 0D equations with confinement scaling, separatrix elongation,
   and recirculating-power estimates.
2. Replace the analytic field-map proxy with an axisymmetric Grad-Shafranov-like
   equilibrium approximation.
3. Add a charged-particle expansion/direct-conversion stage downstream of the
   FRC ends.
4. Sweep radius, length, beta, field strength, density, temperature, and
   direct-conversion efficiency to find viable geometry windows.
5. Keep mirror and spherical torus as baselines so the FRC choice remains
   defensible as the math improves.

## Credibility Line

Helionis should say:

> Our zero-shot reactor geometry analysis selects an FRC-centered linear
> architecture as the first simulation target because it best matches D-He3's
> high-beta, low-neutron, direct-conversion design requirements.

Helionis should not say:

> We have found the final commercial helium-3 reactor geometry.
