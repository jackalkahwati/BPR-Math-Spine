# M4 compute run — Benchmark v3 gate status, 2026-09-10

> **Status: gates NOT passed; envelope SEALED.** First production run with
> the corrected operators and the vectorised code. Records the numbers so
> the next run starts from evidence rather than from "insufficient
> statistics". Machinery: `bpr/gauge_mc_fast.py`. No comparison with the
> sealed targets was made and no ratio is reported.

## Run 1: isotropic, near the transition

| Setting | Value |
|---|---|
| Group | D₅ (faithful E1 irrep), frozen Wilson action |
| Lattice | 8 × 8 × 16 (time = 16), and 6 × 6 × 16 for the volume gate |
| β | 1.8 (β_c ≈ 1.9–2.0 at this volume; confining side) |
| Configurations | 100,000 per volume (stride 4, 3,000 equilibration sweeps) |
| Basis | 3 operators per channel; GEVP at t₀ = 1; 50 jackknife blocks |
| Operators | A1 plaquette/rectangles, B1 rectangle differences, A2 chiral polyominoes (gauge invariant and parity-odd, verified) |

**Diagonal correlators of the first operator, 8³×16 (connected for A1):**

| t | A1 | B1 | A2 |
|---|---|---|---|
| 0 | 3.8 × 10⁻³ | 4.6 × 10⁻³ | 1.9 × 10⁻² |
| 1 | 2.8 × 10⁻⁴ | 3.1 × 10⁻⁵ | 2.4 × 10⁻⁵ |
| 2 | 2.6 × 10⁻⁵ | noise | noise |
| 3 | 1.6 × 10⁻⁶ | noise | noise |

Statistical noise floor at this budget ≈ 1–3 × 10⁻⁶.

**Reading.** The A1 channel decays by a factor ≈ 14 per time slice, i.e.
an effective mass ≈ 2.6 in lattice units, and reaches the noise floor by
t = 3. The B1 and A2 channels decay faster still (effective mass ≈ 5–6
from t = 0 to 1) and are noise from t = 2. Every state is heavy in units
of the lattice spacing. No plateau exists in any channel, so Gates 1 and 2
fail and Gate 3 is moot.

**What this changes.** The blocker is no longer "statistics" in the sense
of configuration count: the decay is exponential in t with a large mass,
so a hundred-fold increase in configurations buys one more usable time
slice. The lever is the lattice spacing in the time direction, which is
standard: an anisotropic action (β_t > β_s) makes the temporal spacing
finer so each state decays over more slices. Mass **ratios** between
channels in the same units do not require the anisotropy to be
calibrated, so the benchmark, which needs ratios only, is not blocked by
the calibration.

## Run 2 (launched; results appended below when complete)

| Tag | Lattice | β_s | β_t | Purpose |
|---|---|---|---|---|
| aniso_b18_bt36 | 8 × 8 × 32 | 1.8 | 3.6 | anisotropy ≈ 2 |
| aniso_b18_bt54 | 8 × 8 × 48 | 1.8 | 5.4 | anisotropy ≈ 3 |
| iso_b195 | 8 × 8 × 24 | 1.95 | 1.95 | closer to β_c, lighter states if the transition is continuous |
| aniso_b18_bt36_L6 | 6 × 6 × 32 | 1.8 | 3.6 | volume gate partner |

Each 100,000 configurations. **Completed 2026-09-11; gates fail in all
four.** Two lessons, one of them a mistake of mine:

| Tag | Plaquette | A1 decay per slice | Reading |
|---|---|---|---|
| iso_b195 | 0.573 | ≈ ×8 (m_eff ≈ 2.1) | slightly lighter than at β = 1.8; noise by t = 3 |
| aniso_b18_bt36 | 0.997 | ≈ ×10, amplitude 40× smaller | **wrong phase** |
| aniso_b18_bt54 | 0.9998 | frozen; A2 amplitude 10⁻⁸ | **wrong phase** |

Raising β_t while holding β_s = 1.8 does not make the temporal spacing
finer at fixed physics; it moves the whole system across the transition
into the ordered (deconfined) phase, where the loops barely fluctuate.
The anisotropic action needs β_s **lowered** and β_t raised together, and
the transition line in the (β_s, β_t) plane has to be located first. That
scan is recorded in the next section. Closer to β_c on the isotropic
line the states are lighter (m_eff 2.6 → 2.1), which is the expected
direction if the transition is continuous, but not enough on its own.

## Anisotropic transition line (2026-09-11)

Spatial-plaquette susceptibility along β_t = ξ β_s, D₅, L_s = 8, L_t = 16,
600 measurements per point (cheap with the vectorised code):

| ξ | β_s at the jump | β_t | Character |
|---|---|---|---|
| 1 | ≈ 2.0 (peak at the end of the coarse scan; earlier work: 1.9–2.0) | 2.0 | susceptibility peak |
| 2 | 1.29 ± 0.01 | 2.58 | plaquette jumps 0.52 → 0.72 → 0.85 across 0.04 in β_s |
| 3 | 0.97 ± 0.01 | 2.91 | plaquette jumps 0.42 → 0.55 → 0.66 → 0.90 |

The jumps are sharp at this volume, which is what a first-order (or
weakly first-order) transition looks like; if that holds, the correlation
length stays finite at the transition and the states never become light
in lattice units. This is a property of the frozen dynamics, not of the
algorithm. It does not block the benchmark, which needs ratios, but it
means "go closer to β_c" is not a route to arbitrarily clean plateaus.

## Run 3 (launched; results appended when complete)

| Tag | Lattice | β_s | β_t | Purpose |
|---|---|---|---|---|
| xi2_b126 (+ L6 partner) | 8 × 8 × 32 | 1.26 | 2.52 | ξ = 2, just below the line |
| xi3_b094 (+ L6 partner) | 8 × 8 × 48 | 0.94 | 2.82 | ξ = 3, just below the line |
| iso_b198 | 8 × 8 × 24 | 1.98 | 1.98 | isotropic, just below the line |

## Rules kept

- No target value appears in any module (grep-enforced).
- Ratios are produced only if all three gates pass; none did.
- The comparison with the sealed targets, if it ever happens, is a
  separate, deliberate, one-time act documented in
  `doc/GLUEBALL_BENCHMARK_V1.md` terms.
