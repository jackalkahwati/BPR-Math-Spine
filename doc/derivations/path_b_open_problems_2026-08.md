# Path B Open Problems — August 2026 Working Session

Three open problems were on record after M1–M4
(`doc/BPR2_PATH_B_NONABELIAN_GAUGE.md` §5). This documents what was done on
each, what closed, and what remains. Modules:
`bpr/gauge_phase_mc.py`, `bpr/glueball_channels_mc.py`,
`bpr/lepton_dn_assignment.py`; locks in `tests/test_gauge_open_problems.py`.
Blindness to the sealed v1/v3 targets is grep-enforced in all three.

---

## 1. M2 phase location — **ANSWERED at scan resolution** (was: OPEN)

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

## Net status change

| Problem | Before | After |
|---|---|---|
| M2 phase location | OPEN — "requires simulation" | **β_c located for all 4 classes** (± 0.1, volume-stable, Z₂-anchored); open remainder sharpened to "physical λ underived" |
| M4 J^PC splitting | BLOCKER — no machinery | **Machinery built + gated**; blocker recharacterized as statistics/compute; envelope SEALED |
| M3 √210 wrinkle | OPEN — "l mod n undefined" | **CLOSED for D_5**; reduced to component ambiguity for D_8/9/12 |

No sealed target was read, computed against, or approached. All three modules
pass the same mechanical blindness grep as the rest of Path B.
