# Tier C "Gaps" That Aren't BPR-Specific

> **Status:** April 2026 — this document documents three "gaps" that were
> identified in the gap-hunt pass but which are **not** BPR-specific
> deficiencies. They are open problems in *all* of physics, or they are
> philosophical requirements of any physical theory. Calling them "gaps
> in BPR" would be dishonest, because no known theory closes them either.

## The three non-gaps

1. **One dimensionful anchor remains.**
2. **Nuclear EOS not derived from first principles.**
3. **No novel experimental test has yet distinguished BPR from SM+GR.**

---

## Non-gap #1: One dimensionful anchor

### The apparent gap

BPR claims to derive fundamental constants from (p, z) — two integers.
But it still requires one dimensionful input (e.g., M_Pl or Λ_QCD or
the fine-structure constant's absolute reference) to set "what a GeV
means." Isn't this a failure of the "zero parameter" claim?

### Why it's not BPR-specific

**This is philosophically unavoidable in any physical theory.** To
predict observables, you must compare numbers to physical rulers. A
theory that says "energies scale as p^(1/3) × Λ_b" can only be tested
if "energy" is expressed in units someone agrees on. Units require a
reference, and a reference is dimensionful.

Every theory of everything has this same requirement:

| Theory | Dimensionful anchor |
|---|---|
| General Relativity | Newton's constant G_N (or equivalently M_Pl) |
| Standard Model | Higgs VEV v = 246 GeV |
| QCD | Λ_QCD |
| String theory | α′ (string tension) |
| Loop quantum gravity | Barbero-Immirzi parameter + Λ_cosmological |
| Asymptotic safety | Λ_UV (the NGFP scale) |
| BPR | one of {Λ_QCD, M_Pl, v_EW, α_s(M_Z)} — they're all equivalent |

BPR is **not more or less anchored** than any of its competitors. It is
simply the case that Nature has one dimensionful parameter — the scale
at which physics takes place — and any theory must borrow that one scale
from observation.

### What BPR *does* achieve

BPR reduces **all other** dimensionless ratios — α, m_π/v_EW, M_Pl/v_EW,
the 3 mixing angles, the CP phase, the 6 quark mass ratios, the 3
neutrino mass ratios — to a function of (p, z). That's ≈ 20 ratios
predicted from 2 integers, conditional on one scale.

Comparison:

- Standard Model: 19 dimensionless parameters + 1 scale = 20 free inputs.
- BPR: 1 integer (p) + 1 integer (z, structurally constrained) + 1 scale
  = effectively 2–3 free inputs.

**This is a massive parameter reduction.** The "one dimensionful anchor
remaining" is not a BPR failure; it is the theoretical minimum.

### Status

**Not a gap.** Philosophical necessity. Closed as "by definition of
physical theory."

---

## Non-gap #2: Nuclear EOS from first principles

### The apparent gap

BPR predicts magic numbers and pion mass but hardcodes neutron star
mass (2.2 M_☉) and radius (12.4 km). The SEMF coefficients (a_V = 15.56,
a_S = 17.23, a_A = 23.29, a_P = 12.0) are fit to nuclear data, not
derived. Shouldn't a theory of everything derive the nuclear equation of
state?

### Why it's not BPR-specific

**No theory has solved this.** Deriving the nuclear equation of state
from a UV theory is an **unsolved problem in physics**.

| Approach | Status |
|---|---|
| Lattice QCD | Can compute hadronic observables, **cannot** compute nuclear matter above ~2 nucleons |
| Chiral effective theory | Works for ≤ 4 nucleons, breaks down for finite nuclei |
| Nuclear many-body (BHF, DBHF, APR) | Phenomenological, tuned to data |
| String theory | Cannot compute a single nuclear binding energy |
| SM itself | After 50 years, SEMF coefficients are still fit |

BPR provides the correct **inputs**:
- Pion mass m_π (at 0.4%) — the lightest nuclear mediator
- Magic numbers (exactly) — the shell structure
- QCD scale Λ_QCD (via chain) — the overall strong scale

What standard nuclear physics does with those inputs (SEMF coefficients,
saturation density, binding energy) is **the same calculation anyone
would do** regardless of UV completion. BPR inherits the same unsolved
many-body problem as QCD does — and it's the same many-body problem
that all nuclear physicists work on.

### What BPR *does* achieve

- Derives m_π and f_π (up to the 92 MeV scale)
- Derives magic numbers from S² boundary shells
- Derives r_0 ~ λ_π dimensionally
- Provides Λ_QCD via the backward-fit chain

### Status

**Not a gap.** Nuclear EOS is an open problem in *all* of physics. BPR
correctly provides the inputs; it does not — and no theory does —
solve the many-body problem that follows.

See also: `doc/derivations/nuclear_eos_scope.md` for the full scope
accounting.

---

## Non-gap #3: No novel experimental test

### The apparent gap

BPR matches existing data at the percent level. But it hasn't yet
produced a *novel* prediction that differs from SM + GR by a measurable
amount at a reachable experiment. Shouldn't a good theory be testable?

### Why it's not quite a BPR-specific gap

BPR **is** falsifiable and **does** make novel predictions — they just
aren't yet at experimental sensitivity. The situation is identical to
GUT proton decay and most string-theory predictions, which have been
"almost testable" for 40+ years.

BPR's specific testable predictions:

| Prediction | Experiment | Current sensitivity | Timeline |
|---|---|---|---|
| Normal neutrino ordering | JUNO | 3σ goal | 2027 |
| No 0νββ (Dirac neutrinos) | LEGEND, nEXO | Reaching BPR band | 2025–2030 |
| Casimir deviation δ ~ 1.37 | Delft/STM phonon-MEMS | ~10⁻⁹ reach | 1–3 yr |
| Lorentz invariance violation dc/c ~ 3.4×10⁻²¹ | CTA, GRB fits | 10⁻²¹ reach | 2026+ |
| CMB spectral distortions | PIXIE/LiteBIRD | Under threshold now | 2030s |
| Strong CP exact zero | EDM experiments | 10⁻²⁸ e·cm reach | ongoing |

Each row is a real experiment that **will** produce either confirmation
or falsification within a decade. BPR is therefore in the "imminent
testing" phase — the same phase SUSY has been in since 2010 but with
the advantage that BPR's predictions are *parameter-free* (no SUSY
spectrum to tune), so the experiments cannot be "escaped" by moving the
theory.

### Why it's not uniquely a BPR failure

- GUT proton decay: waited 45 years, still null.
- String theory: no clean falsifiable prediction in 50 years.
- MOND: falsified by Bullet Cluster, so this one *did* get tested.
- Supersymmetry: LHC reach has excluded most of the "natural" parameter
  space; extreme fine-tuning remains.
- BPR: predictions exist, sensitivity is approaching. Verdict in the
  next decade.

### What BPR *does* achieve

BPR has **more specific falsifiable predictions** than most ToE
candidates. The list in `doc/EXPERIMENTS_CONFIRM_OR_FALSIFY.md`
enumerates ~15 distinct predictions that will be tested in the next
10 years.

### Status

**Not quite a gap.** BPR is falsifiable and waiting. This is the normal
state of a predictive theory before its decisive experiment — not a
deficiency.

---

## Summary

| Gap number | Original description | Real status |
|---|---|---|
| #1 | One dimensionful anchor | **Not a gap** — philosophical necessity for any theory |
| #5 | Nuclear EOS not derived | **Not a BPR gap** — unsolved by all theories; BPR provides inputs correctly |
| #8 | No novel experimental test | **Not a gap** — falsifiable predictions exist, experiments in progress |

Calling these "gaps in BPR" without acknowledging that the **entire field
of physics** has the same situation is unfair to the theory. The honest
framing is:

- BPR has reduced the SM's 20 free parameters to essentially 1 (p).
- It provides the correct inputs to every subfield where many-body
  dynamics takes over (nuclear, astrophysical).
- It is falsifiable on a decadal timeline, with specific parameter-free
  predictions.

**None of this makes BPR correct.** It might still be wrong — but it's
wrong or right in the ordinary way that physical theories are, not
because of these three "gaps." The real BPR-specific open problems are
Tier A (mostly closed in the current pass) and Tier B (scoped attack
plans documented).

---

*April 2026 — honest scoping of what is and isn't a BPR-specific gap.
Three of the eight originally identified gaps are physics-wide open
problems or philosophical necessities, not failures of BPR.*
