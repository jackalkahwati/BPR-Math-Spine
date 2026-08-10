# Z_p Neutrality Superselection — the derived state-selection principle

> **What this is.** The physical-state selection principle (confinement analog)
> that the glueball post-mortem said BPR was missing — derived from the Z_p
> substrate, with its one load-bearing assumption flagged, its consequences
> verified numerically, and its honest limitation stated: **it does not reopen
> the glueball sector.**
>
> Not to be confused with `p_selection_principle.md` (which concerns the
> selection *of* the prime p). This document concerns the selection of
> physical *states*.

## 1. The derivation

**Premise 1 (frozen, verified).** Every term of the frozen boundary action is
built from phase differences: the hopping term depends on e^{i(φ_y − φ_x)},
the quartic on |ψ|⁴ alone. A global shift φ_x → φ_x + c (mod p) leaves the
action exactly invariant (verified to 10⁻¹⁴ in
`bpr/zp_selection_principle.py:global_shift_invariance`), and no frozen
observable measures absolute phase.

**Premise 2 (the load-bearing assumption — flagged, not hidden).** The
substrate is *self-contained*: BPR's boundary is not embedded in anything that
could serve as an external phase reference. In laboratory condensed matter the
environment provides such a reference, which is why a BEC can spontaneously
"break" U(1). In a self-contained substrate there is nothing to measure the
absolute phase against, so two configurations differing by a global shift are
the *same physical state*: the shift is a **redundancy** (gauge
identification), not a symmetry. This is the standard closed-universe
argument. Everything below rides on this premise; if the substrate is ever
given an external embedding, the principle reverts to an ordinary global
symmetry and the rule dissolves.

**Conclusion.** Physical states are invariant under the shift generator
S = e^{2πiQ/p}. Invariance forces

$$Q \equiv 0 \pmod p$$

— the **Z_p neutrality superselection rule**. Note the *mod p*: the substrate
shift group is Z_p, not U(1), so charge is only forced to vanish modulo p.

## 2. Verified consequences

| Consequence | Check | Result |
|---|---|---|
| Rule is consistent with dynamics | [H, Q] on a 27-dim bosonic Hilbert space | exactly 0 |
| Sectors never mix | max \|⟨Q\|H\|Q′⟩\| across sectors | exactly 0 |
| Confinement analog | is a bare Q=1 quantum physical? | **No** |
| Z_p "baryons" | is a p-quanta composite physical? | **Yes** (Q = p ≡ 0) |
| Sector counting | physical dim on the toy space | dim/p exactly (9 of 27) |

The Z_p-baryon consequence is the substrate's structural fingerprint — the
exact analog of the Z_N center of SU(N) permitting N-quark baryons. With
p = 104761 such composites are energetically absurd, so this is a structural
prediction, not a phenomenological one.

## 3. The honest limitation — the glueball sector stays closed

The principle **does not remove the light 1⁻** that helped kill the glueball
benchmark. Number-conserving Bogoliubov quasiparticles are operators of the
form b†_k ~ a†_k a_0 (move one quantum out of the condensate): they commute
with Q *exactly* (verified) — they are neutral, hence physical. The neutrality
rule confines bare quanta, but every quasiparticle of the condensate spectrum
survives it, including the offending 1⁻.

**Therefore: the glueball sector remains CLOSED.** The post-mortem lesson is
only partially addressed — BPR now *has* a derived confinement analog (a real
structural gain), but the specific spectroscopic defect survives, because that
defect was already neutral.

Dead ends searched and recorded (so nobody retries them silently):
- **Multiplicative/QR grading** as a conserved Z₂ charge: the Legendre
  character is multiplicative, the interaction vertex conserves *sums* mod p —
  the grading is not conserved. Not a superselection charge.
- **Reflection identification x ↔ −x**: would be a new gauge identification
  not present in the frozen postulates — a retrofit. Rejected.

## 4. Status

- **Derived, not postulated:** the rule follows from Premise 1 (verified) +
  Premise 2 (flagged interpretive assumption). It was NOT constructed to fix
  the glueball spectrum — and indeed it does not fix it.
- **What it adds to BPR:** a confinement analog for bare quanta; a Z_p-baryon
  sector; exact charge-sector block structure consistent with the frozen
  dynamics.
- **What it leaves open:** removing neutral single quasiparticles from the
  physical spectrum would require structure the frozen substrate does not
  supply. If a future frozen extension supplies it, the glueball benchmark
  re-runs under the same sealed targets.

## Cross-references
- `bpr/zp_selection_principle.py` — derivation + all numerical checks
- `tests/test_zp_selection_principle.py` — locks the checks and the honest negative
- `doc/GLUEBALL_BENCHMARK_V1.md` §9 — the post-mortem that demanded this principle
- `doc/CLOSED_AND_DEPRECATED.md` §1 — glueball closure (unchanged by this)
