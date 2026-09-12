# Path B Open Problems — August 2026 Working Session

Three open problems were on record after M1–M4
(`doc/BPR2_PATH_B_NONABELIAN_GAUGE.md` §5). This documents what was done on
each, what closed, and what remains. Modules:
`bpr/gauge_phase_mc.py`, `bpr/glueball_channels_mc.py`,
`bpr/lepton_dn_assignment.py`; locks in `tests/test_gauge_open_problems.py`.
Blindness to the sealed v1/v3 targets is grep-enforced in all three.

---

## 2026-09-12 model-identification correction

The scans below locate finite-volume features of the **Euclidean
character-Wilson action**, not the originally declared `{r,r^-1,s}` electric
Hamiltonian. That electric operator is noncentral and fails endpoint gauge
invariance. The separate conjugacy-averaged model in `bpr/gauge_heat_kernel.py`
has its own heat kernel. The Wilson transfer energies are not proportional to
its electric spectrum: the monotone beta-to-lambda identification asserted
below is unsupported, not merely awaiting calibration. Neither these scans nor
the physical-lambda proposal locates the repaired Hamiltonian's physical phase.
All historical scan numbers are retained. See
[toe_constructive_extension_2026-09-12.md](toe_constructive_extension_2026-09-12.md).

## 1. Historical Wilson phase scan — **ANSWERED at scan resolution** (not repaired-Hamiltonian phase location)

**Method.** 3D Euclidean Wilson lattice theory for D_n, plaquette action in
the faithful E1 irrep, L³ periodic lattice, Metropolis; pseudo-critical
coupling β_c from the plaquette-susceptibility peak.

**Validation anchor.** The identical code for gauge group Z₂ reproduces the
known 3D Z₂ gauge transition (β_c ≈ 0.7613, dual 3D Ising): peak at
β = 0.65 ± 0.05 at L=4 — consistent within coarse-scan + finite-size shift.

**Result (scan resolution ± 0.1, susceptibility peak, seeds fixed):**

| Class | β_c (L=4) | β_c (L=6) | Volume-stable |
|---|---|---|---|
| D_5  | 1.90 | 2.00 | ✓ (within resolution) |
| D_8  | 2.10 | 2.10 | ✓ |
| D_9  | 2.10 | 2.10 | ✓ |
| D_12 | 2.20 | 2.20 | ✓ |

Strong coupling (β < β_c) is the confining phase — where a spectrum exists;
above it the theory is deconfined/topological, as identified exactly in M2.

**What remains open (sharpened, not hidden):** BPR provides no derivation of
the substrate's *physical* λ from (p, z, n). The transition's location is now
known; **which side the substrate sits on is not.** This is the sharpest
remaining form of the M2 problem. Also recorded: β_c here is pseudo-critical
(no finite-size-scaling extrapolation), and the Euclidean β ↔ Hamiltonian λ
map is monotone but not calibrated (needs an anisotropic-lattice study).

## 2. M4 J^PC-splitting blocker — **MACHINERY BUILT; envelope stays SEALED**

**Method.** Zero-momentum loop-operator correlators in three lattice-symmetry
channels on time slices of the same 3D theory: A1 (symmetric plaquette sum →
0⁺⁺ precursor), B1 (x−y rectangle difference → 2⁺⁺ precursor), A2 (chiral
L-loop minus mirror → parity-odd precursor; bent loops are required because
D_n plaquette characters are real). Jackknife effective masses, plateau
detection.

**Sealed-envelope discipline, mechanically enforced.** Benchmark v3 permits
exactly one comparison. The module hard-gates its own output:
Gate 1 plateau in every channel; Gate 2 jackknife error < 15%; Gate 3
volume stability of the A1 mass. **Ratios are not even computed unless all
gates pass**, and comparison to the locked targets is a separate deliberate
act not performed by any code here.

**Status at current compute budget (laptop, ~10⁴ configs, L ≤ 6):** signals
in all channels die into noise within 1–2 time slices (correlators drop from
~10⁻² to ~10⁻⁴ immediately — the states are heavy in lattice units near β_c
at these volumes). Gates fail; **the envelope stays sealed.** The blocker has
changed character: from "no machinery exists" to **"insufficient statistics"
— a compute problem, not a physics wall.** The remedy is standard and known:
larger lattices, anisotropic coupling (finer time direction), operator
smearing/variational basis, and ~10⁶ configs. None of that changes the
frozen dynamics or the targets.

**Addendum 2026-09-10 (`bpr/gauge_mc_fast.py`, `tests/test_gauge_mc_fast.py`).**
Two defects were found in the A2 (parity-odd) operator above while building
a faster measurement code, and both are fixed there:

1. The 8-link "L" and its "mirror" in `glueball_channels_mc._l_loop` are not
   contiguous link paths (a backward step is taken from the wrong site), so
   their characters are **not gauge invariant** — verified: they change by
   O(1) under a random gauge transformation while plaquettes and rectangles
   are invariant to machine precision. The earlier A2 "signal" was gauge noise.
2. Even a contiguous notched-square loop is **achiral**: its x-reflection is
   a rotation of itself with reversed orientation, and D_n characters are
   real, so the rotation-symmetrised (loop − mirror) combination vanishes
   identically. A parity-odd zero-momentum operator needs a chiral shape.
   The fix uses polyomino boundaries (L-tetromino, S-tetromino,
   P-pentomino), rotation-symmetrised and antisymmetrised under x → −x;
   these are verified gauge invariant, non-zero, and odd under reflection
   of the configuration.

The A1 and B1 operators were correct. Consequently the earlier statement
"gates fail at laptop statistics" was true for A1/B1 and vacuous for A2.
The compute side is also addressed: a vectorised checkerboard Metropolis
(validated against `WilsonMC` on Z₂ and D₅) runs 8×8×16 at ~1 ms/sweep,
with an L_t longer than L_s, a 3-operator variational basis per channel, and
a GEVP for effective masses. Results of the first production run are
recorded in `doc/derivations/m4_compute_run_2026-09.md`.

## 3. M3 √210 lepton wrinkle — **CLOSED for n=5; reduced elsewhere**

**Resolution (no new postulate).** The l_μ derivation itself supplies the
missing structure: l_μ = √(z(z²−1)) = √210 = √(14×15) is explicitly a
geometric mean of two consecutive shell counts — i.e. the label of a
**two-mode composite** with integer constituents (14, 15). Composites carry
the tensor product of their constituents' D_n charges (standard rep theory).
Computed exactly with dihedral fusion rules:

| Class | e | μ | τ | μ verdict |
|---|---|---|---|---|
| D_5  | E1 | **E1** (E1 ⊗ A) | E1 | **determinate — wrinkle CLOSED** |
| D_8  | E1 | E1 + E3 | E3 | component ambiguity remains |
| D_9  | E1 | E1 + E2 | E4 | component ambiguity remains |
| D_12 | E1 | E1 + E5 | E1 | component ambiguity remains |

For n=5, 15 ≡ 0 (mod 5) is A-type, so E1 ⊗ A = E1 is a single irrep: the
muon's D_5 charge is unique and the wrinkle closes outright. For n ∈ {8, 9, 12}
the product decomposes into two irreps; picking a component would require a
dynamical input we do not have, so it is recorded as a **reduced** open item
(from "undefined" to "two-way ambiguity"), not closed.

**Proved regardless:** lepton masses consume l² (integer for all three), so
every candidate assignment is mass-inert, and M3's PASS is unaffected.

---

## 4. The M2 remainder: physical λ from substrate quantities — **PROPOSED v0.1**

The sharpened M2 question ("which side of β_c does the substrate sit on?")
was worked in `bpr/physical_lambda.py` (locks:
`tests/test_physical_lambda.py`). Two independent derivations from frozen
inputs, no tuning:

| Route | Basis | Provenance | β |
|---|---|---|---|
| A (thermal) | β = J/T_eff with T_eff = J/ln p | frozen in `bpr/rpst/boundary_energy.py` | **ln p = 11.56** |
| B (gauge) | β = 1/g², g² = 4πα, frozen α formula | Postulate 0c (`bpr/consistency.py`) | **1/(4πα) = 10.91** |

The routes agree to 6% — with the caveat stated plainly: the agreement holds
because ln p ≈ 4π for the selected prime; it is a property of the frozen p,
not an independent identity.

**Verdict (PROPOSED):** β_phys ≈ 11.2 ± 0.3 against measured β_c ≈ 1.9–2.2 —
the substrate sits on the **deconfined/topological side, a factor ~5 above
the transition**, robust to any O(1) convention ambiguity below that factor.

**Kill condition (explicit):** an anisotropic-lattice calibration of the
Euclidean-β ↔ Hamiltonian-λ map (or an exact duality) shifting the effective
coupling by more than ~5× overturns the verdict. Nothing smaller can.

**Consequence if it stands:** the physical excitation content is the M1 anyon
spectrum (topological phase) — consistent with M1's quantum-double
kinematics. The confining glueball-analog spectrum that the M4 machinery
targets belongs to a phase the substrate does not occupy; Benchmark v3 then
tests the confining phase of the frozen dynamics as a theory exercise, and
the particle-sector interpretation must route through anyonic excitations.

**Recorded, not used:** the per-link value ln(p)/z = 1.93 lands *on* the
measured β_c (ratios 0.88–0.99 across all four classes). No frozen derivation
supports dividing by z, so this is logged as a numerical observation only —
but a per-link convention would place the substrate essentially AT
criticality, and the coincidence across all four classes is flagged for
future work.

## Net status change

| Problem | Before | After |
|---|---|---|
| M2 phase location | OPEN — "requires simulation" | **β_c located for all 4 classes** (± 0.1, volume-stable, Z₂-anchored); remainder worked in §4: β_phys ≈ 11.2 from two frozen routes → **deconfined side (PROPOSED, kill condition on record)** |
| M4 J^PC splitting | BLOCKER — no machinery | **Machinery built + gated**; blocker recharacterized as statistics/compute; envelope SEALED |
| M3 √210 wrinkle | OPEN — "l mod n undefined" | **CLOSED for D_5**; reduced to component ambiguity for D_8/9/12 |

No sealed target was read, computed against, or approached. All three modules
pass the same mechanical blindness grep as the rest of Path B.
