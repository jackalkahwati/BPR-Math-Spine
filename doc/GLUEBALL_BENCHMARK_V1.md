# BPR Glueball Benchmark v1 — sealed protocol

> **Status: PROTOCOL LOCKED, ENVELOPE SEALED.** This document freezes the rules
> and the comparison targets BEFORE the BPR spectrum is computed. The solver
> never reads the targets; the comparison happens only after the S² eigenproblem
> produces a spectrum. Locked 2026-08-10 on branch
> `claude/bpr-experiment-review-Gpd2R`.

## 1. The test

Treat BPR as an eigenmode problem and judge it against the QCD glueball
spectrum with no tuning:

```
frozen BPR equations → stationary/topological modes → {J^PC} → {E_n} → E_n/E_0
```

compared against lattice-QCD glueball ratios and (secondarily) the BESIII
X(2370) pseudoscalar-glueball candidate.

**Rules (fixed now, before any spectrum exists):**
1. The BPR equations are frozen as of this commit (§2). No new terms after
   looking at any spectrum.
2. No glueball mass or ratio enters any solver, seed, loss function, or
   convergence criterion. The targets live only in this document and its tests.
3. Absolute scale is not judged. Exactly ONE overall scale may be fixed to one
   state; all remaining ratios are then predictions.
4. Quantum numbers must emerge from the symmetry of the solutions (rotations →
   J, inversion → P, complex conjugation of the U(1) field → C), never inserted
   by hand.
5. The envelope (§4) is opened only after Gates 2–4 produce numbers.

## 2. The frozen equations

Exactly what is already committed in this repo — nothing added for this test:

- Substrate: Z_p lattice; linear dispersion from nearest-neighbour hopping,
  ω_k = −2C cos(2πk/p) (`bpr/condensate_mechanism.py`).
- Interaction: the substrate-symmetric quartic |ψ|⁴ whose Fourier vertex is the
  momentum-conserving (translation-invariant) coupling derived in
  `bpr/harmonic_coupling_derivation.py`.
- Elastic/phason sector: the LRT free energy + phason dislocation content of
  `bpr/phason_defect_lagrangian.py` (π₁ line defects only; higher homotopy
  vanishes — `bpr/phason_coupling.py`).
- Conserved U(1) norm of the complex boundary field (charge conjugation C acts
  as ψ → ψ*).

## 3. The four gates

| Gate | Question | Status |
|---|---|---|
| 1 | Does the frozen theory generate discrete localized finite-energy states at all? | **PASS** (§5) |
| 2 | Do the solutions fall into the correct J^PC families (0^{++}, 2^{++}, 0^{−+}, …) without hand-inserted labels? | **PARTIAL FAIL** (§8 — envelope opened) |
| 3 | Is the ordering of the lowest states M(0^{++}) < M(2^{++}) < M(0^{−+})? | **FAIL** (§8) |
| 4 | After fixing one overall scale, do the remaining ratios land near lattice QCD? | **FAIL** (§8) |

## 4. The sealed envelope — locked comparison targets

These numbers are recorded now so the later comparison is blind. **No solver
may import them.**

**Primary: lattice QCD, quenched SU(3)** (Morningstar & Peardon, PRD 60, 034509, 1999):

| State | Mass (MeV) | Ratio to 0^{++} |
|---|---|---|
| 0^{++} | 1730 ± 80 | 1.000 |
| 2^{++} | 2400 ± 120 | **1.387** |
| 0^{−+} | 2590 ± 130 | **1.497** |

**Cross-check** (Chen et al., PRD 73, 014516, 2006): 0^{++} 1710, 2^{++} 2390,
0^{−+} 2560 → ratios 1.398, 1.497. Consistent.

**Experimental context** (not a fit target): BESIII observes X(2370) at
≈ 2.37–2.40 GeV with quantum numbers increasingly consistent with 0^{−+}; its
ratio to the lattice 0^{++} (1730) is ≈ 1.37–1.39, in mild tension with the
lattice 0^{−+}/0^{++} = 1.50 (possibly from scalar-sector mixing). That tension
belongs to QCD phenomenology, not to this benchmark; BPR is judged against the
lattice ratios.

*Update (post-envelope, 2026 BESIII Letter with 10B J/ψ events):* the
experimental anchor has strengthened substantially since the envelope was
sealed and opened. BESIII established the X(2370) as a **flavor singlet** (the
K*(892)⁰K̄⁰ mode is suppressed, B < 2.7×10⁻⁶ at 90% C.L. — forbidden for a
0^{−+} flavor singlet by generalized G-parity), the first flavor-singlet light
hadron above 1 GeV. Combined mass/width: 2359 MeV / 170 MeV; J^PC = 0^{−+} at
>9.8σ; high production in gluon-rich J/ψ radiative decays; ηc-like decay
pattern; narrow partial widths; γω/γφ strongly suppressed. BESIII's conclusion:
a dominant lightest-0^{−+}-glueball component is essential to explain all
properties simultaneously; η–η′ excitation, q q̄, multiquark, hybrid, and Σ Σ̄
baryonium interpretations are disfavored. This does not change the sealed
lattice targets or any verdict — it means the state this benchmark judged BPR
against is now the best-established glueball in nature, so the Gate 2–4
failure is anchored to *measured* physics, not only to lattice prediction
(experimental ratio 2359/1710 ≈ 1.38, inside the sealed 0^{−+} band; BPR's
computed range was [3.7, 9.3]).

**Pass bands (fixed now):** Gate 3 requires the strict ordering above. Gate 4
requires 2^{++}/0^{++} ∈ [1.25, 1.55] and 0^{−+}/0^{++} ∈ [1.35, 1.65] (lattice
central values ± ~10%), with BOTH ratios from a single parameter set.

## 5. Gate 1 result (this commit)

**Structural part (Derrick's theorem).** The frozen continuum scalar sector
(quadratic elastic + quartic) admits NO stable static localized solitons in 3D:
under x → λx both terms drive scale collapse. So a naive continuum reading of
BPR fails Gate 1 immediately — this is recorded, not hidden. The frozen theory
contains exactly three Derrick escapes, all already present before this test:

1. **Lattice discreteness** — the Z_p substrate has no continuous scaling
   symmetry, which is the textbook loophole permitting intrinsic localized
   modes / discrete breathers (Flach & Willis).
2. **Conserved U(1) norm** — permits time-periodic Q-ball-type states
   ψ = φ e^{−iμt} evading the static assumption.
3. **π₁ line defects** (phason dislocations) — topologically stable but
   string-like (infinite energy as point particles in 3D), so they are NOT
   glueball candidates; excluded.

**Numerical part.** Solving the frozen stationary equation
μφ = −C∆φ − |φ|²φ on the Z_p ring (`bpr/glueball_benchmark.py`):

- Newton converges to residual ~10⁻¹³ on **two distinct** localized states
  (site-centered and bond-centered), localization ~10⁵ over 20 sites.
- The linearized (Bogoliubov) fluctuation operator has the exact U(1) phase
  zero mode and **discrete internal modes below the continuum band edge**
  (ω² ≈ 0.073, 0.223 below the edge at 0.250 for C=1, μ=−2.5).

**Verdict: Gate 1 PASSES** — the frozen equations generate discrete, localized,
finite-energy bound states with discrete fluctuation spectra, via escapes 1+2.

**What Gate 1 does NOT claim:** the ring calculation carries **no J^PC
content** — a 1D winding number is not angular momentum, and no glueball
identification is made or implied. These states demonstrate existence and the
solver machinery, nothing more.

## 6. What Gates 2–4 require (defined now)

The eigenproblem must be posed on the 2-sphere boundary (BPR's derived boundary
topology, `bpr/boundary_topology.py`) so that solutions transform under real
SO(3) rotations:

1. Discretize the frozen equations on S² (or solve in the continuum with the
   lattice-derived dispersion as UV completion).
2. Find stationary states Φ_n (Newton, as validated in Gate 1).
3. Linearize: H_n ψ_k = ω_k² ψ_k.
4. Read off J from the SO(3) irrep of each solution/fluctuation family, P from
   inversion, C from ψ → ψ*.
5. Compute E[Φ_n]; form ratios; **then** open §4.

## 8. Envelope opened — results (2026-08-10, same day; spectrum computed blind first)

The S² eigenproblem was computed in `bpr/glueball_gates.py` on the branch our
own condensate chain forces (excitations of the uniform k=0 condensate — see
`bpr/condensate_mechanism.py`). Emergence was verified, not asserted: a
real-space solver discovered the SO(3) multiplets numerically (degeneracies
1, 3, 5, 7, 9, 11 → J = 0…5; ε = J(J+1) to 4 decimals; parity (−1)^J from the
antipodal map on the eigenvectors), and composite selection rules came from
SO(3) character integrals, not hand-coded tables.

**Gate 2 — PARTIAL FAIL.** The genuinely nontrivial positive: two J=1
quasiparticles give *exactly* {0^{++}, 2^{++}} (J=1 bose-forbidden), C=+
throughout — the right lowest glueball families emerge without insertion. But
two disqualifying features also emerge: (a) the lightest excitation is a single
1^− quasiparticle at 0.50 × M(0^{++}), a family with no counterpart in the low
glueball spectrum; (b) a light 0^{−+} is *forbidden*: parity arithmetic kills
every 1- and 2-quasiparticle J=0^− state, and character integrals kill (1,1,1),
(2,2,1), (2,2,3), (3,3,3). The lightest pseudoscalar is the (2,3,4) triple.

**Gate 3 — FAIL.** Computed ordering: 1^− < {0^{++} = 2^{++} degenerate} ≪
0^{−+}, versus the required 0^{++} < 2^{++} < 0^{−+}.

**Gate 4 — FAIL.** Over the full interaction range γ ∈ [0.1, 1000]:
2^{++}/0^{++} ∈ [0.87, 1.00] (target 1.387; never reaches the band) and
0^{−+}/0^{++} ∈ [3.7, 9.3] (target 1.497; off by ≥ 2.5×). No γ rescues either
ratio — this is not a tuning failure, it is structural.

**Overall: the condensate branch of BPR fails the glueball benchmark.**
Recorded in `doc/CLOSED_AND_DEPRECATED.md` §1 as promised. Loopholes recorded
(not used to soften the verdict): leading-order treatment of quasiparticle
interactions; the self-bound-lump (focusing) branch was not quantized; the
phason sector's boundary parity content was not computed.

## 9. Loopholes dispositioned — sector closed (2026-08-10)

The go/no-go for a Benchmark v2 through the phason sector was computed in
`bpr/phason_boundary_modes.py`. The frozen phason dynamics (inertial phonon +
overdamped phason + D coupling) has a root-symmetry theorem: the dispersion
cubic satisfies P(−ω̄) = conj(P(ω)), so at most ONE propagating (mirror) pair
exists — and it is continuously connected to the D=0 phonon. The full-range
scan (ε ∈ [10⁻², 316], D up to √(KC), Γ ∈ [0.1, 10]) confirms: the
phason-dominated branch sits *exactly* on the imaginary axis everywhere
(relaxation, not particles), and the propagating-family count never exceeds
one. Honest caveat, reported: at short wavelength and strong coupling the
single phonon pair carries up to ~98% *slaved* phason amplitude — dressing of
the same family, no new quantum numbers.

**Verdict: v2 is NO-GO.** With that, all three recorded loopholes close for
the fatal gate (the missing light 0^{−+}):

1. *Beyond-leading-order interactions* — cannot help: the 0^{−+} absence is
   symmetry-protected (parity selection rules); interactions shift energies,
   they do not create parity-odd light states.
2. *Self-bound-lump branch* — built from the same scalar quanta, hence the
   same parity lock.
3. *Phason sector* — no propagating boundary branch exists.

**The glueball sector is CLOSED for BPR as frozen.** What the failure taught
(recorded as constraints, not patched): (a) the lattice glueball spectrum is a
fingerprint of *vector* constituents — any scalar-constituent boundary theory
is structurally incapable of it; (b) BPR lacks a derived physical-state
selection principle (confinement analog) — the extra light 1⁻ exists because
nothing forbids single-quasiparticle states; any future version must derive
such a principle from the substrate, not postulate it to fix a spectrum.

*Update:* lesson (b) has since been partially addressed — a Z_p neutrality
superselection rule was derived from the substrate
(`doc/derivations/zp_neutrality_superselection.md`,
`bpr/zp_selection_principle.py`): it confines bare quanta and yields a
Z_p-baryon sector, but Bogoliubov quasiparticles are exactly neutral, so the
light 1⁻ survives it and **this section's closure stands unchanged**.

## 7. Cross-references

- `bpr/glueball_benchmark.py` — Gate 1 solver + Derrick analysis (never reads §4)
- `tests/test_glueball_benchmark.py` — locks targets, Gate 1 results, honesty guards
- `bpr/harmonic_coupling_derivation.py` — the frozen coupling
- `bpr/condensate_mechanism.py` — the frozen dispersion + nonlinearity
- `doc/CLOSED_AND_DEPRECATED.md` — where a Gate 2–4 failure will be recorded if it fails
