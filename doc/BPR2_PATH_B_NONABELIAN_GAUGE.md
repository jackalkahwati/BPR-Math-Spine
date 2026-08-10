# BPR 2.0 Path B — Gauging the Frozen Point Group (Postulate 0d, PROPOSED v0.1)

> **Status: PROPOSED, not merged into the frozen core.** This is the structural
> revision chosen after the glueball benchmark closed: not a patch to the
> failed sector, but a rebuild that promotes non-Abelian structure the theory
> *already contains* into gauge structure. Milestone 1 (kinematics) is done and
> verified; Milestones 2–4 (dynamics, flavor survival, sealed re-benchmark) are
> open, and **no spectrum claim exists at this stage**.

## 1. Why this revision is principled and not a retrofit

The glueball benchmark (`doc/GLUEBALL_BENCHMARK_V1.md`) closed with a
structural diagnosis, not a numerical near-miss: BPR's boundary content is
**scalar and Abelian**, and such content provably cannot contain a light
parity-odd (0⁻) state (parity locks to (−1)^J) or non-Abelian gauge-boson
matter. The 2026 BESIII flavor-singlet result anchored that missing sector to
measured reality.

The retrofit move would be to invent new fields for that sector. The
principled move is to ask what the *frozen* theory already contains and never
used. Answer: **Postulate 0c's quasicrystal classes n ∈ {5, 8, 9, 12} carry
dihedral point groups D_n — non-Abelian discrete groups** — and the Abelian
phase field only ever used the (Abelian) translation/phase structure. Path B
proposes:

> **Postulate 0d (v0.1, PROPOSED):** the dihedral point-group symmetry D_n of
> the substrate class is a *gauge* symmetry of the boundary theory, not a
> global one.

Blindness discipline carried over from Benchmark v1: no glueball mass, ratio,
or experimental value appears in the gauge-sector module — enforced by a
mechanical grep test, exactly as before. The eventual spectrum faces a **new
sealed benchmark (v3) with the same lattice targets**.

## 2. Milestone 1 — the gauged kinematics (DONE, exact)

Gauging a finite group G on a 2D boundary yields a discrete gauge theory whose
excitation content is fixed by representation theory alone (Kitaev quantum
double D(G)): anyons = (conjugacy class C, irrep R of its centralizer), with
quantum dimension d = |C|·dim R. Everything below is computed from the frozen
group and verified exactly (`bpr/nonabelian_gauge_sector.py`):

| n | \|G\| | classes | anyon types | non-Abelian (d>1) | A2 pseudoscalar | E_k doublets |
|---|---|---|---|---|---|---|
| 5 | 10 | 4 | 16 | 14 | ✓ | k = 1..2 |
| 8 | 16 | 7 | 46 | 38 | ✓ | k = 1..3 |
| 9 | 18 | 6 | 44 | 42 | ✓ | k = 1..4 |
| 12 | 24 | 9 | 86 | 78 | ✓ | k = 1..5 |

(Character orthogonality, Burnside, and the quantum-double sum rule
Σd² = |G|² all verified exactly; D_5 hand-checked: 16 types, Σd² = 100.)

**Structural result 1 — the missing ingredient appears.** Every D_n contains
the 1-dimensional **A2 (sign) irrep**: rotations → +1, reflections → −1. A
gauge charge in A2 is invariant under the substrate's rotations (discrete
spin-0) and *odd* under its reflections (discrete parity −1): **the discrete
precursor of a 0⁻ state.** This is precisely the content whose absence was
*provable* in the Abelian scalar theory — and it emerges here from structure
that was frozen years before the benchmark, not inserted to pass it.

**Structural result 2 — vector-constituent skeleton.** The E_k doublet irreps
carry discrete angular momentum ±k (rotation eigenvalues e^{±2πik/n}), the
precursors of higher-J content — the skeleton of "vector constituents" that
the glueball spectrum fingerprints.

**Structural result 3 — genuinely non-Abelian.** The overwhelming majority of
anyon types have quantum dimension > 1: non-trivial fusion and braiding, the
discrete-gauge analog of self-interacting force carriers.

## 3. What Milestone 1 does NOT deliver — the honest gap list

1. **No spectrum.** At its exactly solvable point, a 2+1D discrete gauge
   theory is *topological*: gapped, deconfined, degenerate — no propagating
   particles, no mass ratios. Claiming glueball physics from Milestone 1 alone
   would be false. A spectrum requires a frozen dynamics away from the
   fixed point (Milestone 2) — e.g., explicit flux/charge kinetic terms or an
   anyon-condensation transition. This is the hard, genuinely open physics.
2. **Flavor-sector survival unproven.** The flavor integers (the framework's
   live LHCb pre-dictions) were derived in the Abelian sector. Whether they
   survive gauging — untouched, modified, or destroyed — must be shown
   (Milestone 3). If gauging destroys the flavor sector, Path B dies by its
   own rules.
3. **Continuum quantum numbers not established.** A2 and E_k are *discrete
   precursors* of P = −1 and higher J. The emergent continuum assignment
   requires the S² eigenproblem of the gauged dynamics — part of the sealed
   Benchmark v3 (Milestone 4).

## 4. The frozen roadmap

| Milestone | Content | Status |
|---|---|---|
| **M1** | Gauge-sector kinematics from the frozen point group | **DONE** |
| **M2** | Freeze a dynamics away from the topological point | **FORM FROZEN** — phase location OPEN (`bpr/gauge_dynamics_m2_m3.py`) |
| **M3** | Prove flavor-sector survival under gauging | **PASS (kinematic)** — see §5 |
| **M4** | Sealed Benchmark v3: same lattice targets, same blindness, emergent J^PC from the gauged dynamics | **SEALED** — blocker documented in §5 |

## 5. Milestones 2–4 — executed 2026-08-10 (`bpr/gauge_dynamics_m2_m3.py`)

**M3 (the kill condition) — PASS, kinematic.** The flavor mass formulas consume
only gauge-inert inputs (mode integers l_i and J, p, z, n_gen). Gauging
organizes boundary states by D_n charge and adds flux sectors, but does not
touch the untwisted Hamiltonian — verified exactly: the charge-sector spectra
partition the full spectrum for every allowed class (no energy moves). **Every
registered LHCb pre-diction is numerically unchanged.** What changes is
interpretation: flavor modes now carry D_n gauge charge fixed by l mod n
(e.g. n=9: d,u → E1; b,c → E3; s,t → E4), directly parallel to quarks carrying
color. Caveats on record: the lepton labels include √210 (l mod n undefined —
open wrinkle), and if the M2 dynamics confines at the physical coupling,
binding-energy corrections are unquantified until the phase is located.

**M2 — dynamics FROZEN (form), phase OPEN.** The canonical finite-group Wilson
Hamiltonian with one coupling λ: H(λ) = (1/λ)Σ_links Δ_G + λΣ_plaq(1 −
Re χ_F/d_F), generating set {r, r⁻¹, s} — the textbook form, not tuned.
Exactly computed: the electric Casimirs ε(A1)=0, ε(A2)=2, ε(B1)=4, ε(B2)=6,
ε(E_k)=3−2cos(2πk/n) for all four classes. Limits: λ→∞ deconfined/topological
(no spectrum); λ→0 confining, string tensions σ(R) ∝ ε(R), glueball analogs =
closed electric loops with LO mass 4ε_min/λ. Structural note recorded without
significance claims: for n=5 the lightest nontrivial charge sector is A2 (the
pseudoscalar precursor); for n=8, 9, 12 it is E1. The confinement–deconfinement
phase location in λ is OPEN — it requires simulation.

**M4 — Benchmark v3 stays SEALED, blocker precise.** The leading-order
strong-coupling spectrum is J^PC-**degenerate** (every smallest-loop state has
mass 4ε_min/λ); quantum numbers split only at higher orders in the
strong-coupling expansion or via Monte Carlo, with emergent J^PC read from
loop multiplets on S². Rather than manufacture ratios, v3 remains sealed with
that single blocker on record. Targets and pass bands inherited unchanged from
v1; mechanical blindness greps enforced on every Path B module.

Rules frozen now: M4 uses the *identical* sealed targets and pass bands as
Benchmark v1 (`doc/GLUEBALL_BENCHMARK_V1.md` §4). No result from M1–M3 may be
tuned against them. If M4 fails, the non-Abelian revision joins the registry
and BPR's particle-sector program ends. If M3 fails, Path B ends regardless of
M4. The Abelian-sector closures (`doc/CLOSED_AND_DEPRECATED.md`) are
unaffected either way — Postulate 0d is a *proposal on top of* the frozen
core, not an amendment to it.

## Cross-references
- `bpr/nonabelian_gauge_sector.py` — Milestone 1 computations + honesty guards
- `tests/test_nonabelian_gauge_sector.py` — exact rep-theory locks + blindness grep
- `doc/GLUEBALL_BENCHMARK_V1.md` — the diagnosis this responds to; the sealed targets M4 inherits
- `doc/derivations/zp_neutrality_superselection.md` — the Abelian selection principle (still holds; complementary)
- `bpr/phason_sector.py:ALLOWED_INTERNAL_CLASSES` — the frozen classes supplying D_n
