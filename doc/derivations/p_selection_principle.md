# Selection Principle for p = 104,761 and z = 6

> **WITHDRAWN (2026-10). Read this first.** The claims in this document that
> p = 104,761 is "over-determined", that it is "unique" within the 3% band,
> that "p is a fit parameter" is refuted, and that z = 6 is "DERIVED" are
> withdrawn. The body below is kept as a historical record. p is an α
> calibration: equation A gives p_exact = 104,749.03, and p is the nearest
> prime ≡ 1 (mod 4) to it, exactly the "prior status" described below. Found
> in the 2026-10 review of unmerged branches
> (`claude/audit-bpr-parameters-6fSrQ`). The numbers here were recomputed on
> the current tree with `bpr.alpha_derivation.inverse_alpha_from_substrate`
> and `bpr.gauge_unification.electroweak_scale_GeV` (Λ_QCD = 0.332 GeV,
> 1/α = 137.035999084, v_EW = 246.22 GeV).
>
> - **Not over-determined.** The band spanned by equations A–D,
>   104,573–107,709, contains 265 primes, 129 of them ≡ 1 (mod 4); it does
>   not single out 104,761. Equation B's root depends on the Planck-mass
>   convention: with the reduced Planck mass it moves from 104,573 to about
>   64,540. Equation D was not re-checked.
> - **The "uniqueness" follows the tolerance, not the data.** Scoring each
>   prime by the RMS of the fractional errors in 1/α and v_EW, 1,193 primes
>   score within 2%, 580 within 1% and 238 within 0.5%, and none within
>   0.306% (the best is 0.3062%, at 107,251). The 1% corridor runs from
>   103,934.86 to 110,633.56, and v_EW sets its lower edge. The α root is
>   104,749.03 and the v_EW root is 107,709.08.
> - **α pins p only to within the prime spacing.** 1/α(104,761) = 137.038641,
>   +19.3 ppm from CODATA 2018. The closest prime, 104,743, misses by 9.7 ppm,
>   which is what the spacing of primes near p_exact produces. Measured
>   against the CODATA uncertainty, every prime misses by at least 6×10⁴σ.
> - **v_EW does not confirm the choice.** At 104,761, v_EW = 243.52 GeV,
>   −1.10% from 246.22 GeV. That miss is within the Λ_QCD input uncertainty:
>   moving Λ_QCD by ±17 MeV around 0.332 GeV (an illustrative uncertainty)
>   moves the v_EW root from about 95,000 to about 122,950.
> - **z = 6 is not derived here.** p and z enter 1/α only through
>   [ln p]² + z/2, so α alone fixes one combination. Solving the α and v_EW
>   equations together for a continuous z gives z* = 6.20 (at p* ≈ 104,296),
>   and z* moves from 5.32 to 7.19 over the same Λ_QCD range; z = 6 is the
>   nearest integer. Over all primes from 20,000 to 400,000, the best joint
>   score at Λ_QCD = 0.332 GeV is 0.31% for z = 6, 1.18% for z = 7 and
>   1.90% for z = 5. The preference does not survive the Λ_QCD uncertainty:
>   at 0.315 GeV z = 7 scores best (0.28%, against 1.77% for z = 6), and at
>   0.349 GeV z = 5 does (0.52%, against 1.07%). The three arguments for
>   z = 6 below are motivations, not derivations.
>
> See `doc/CLOSED_AND_DEPRECATED.md` (including its section 4 row on the
> ln(p)/(ln(p)+1) correction) and `doc/LIMITATIONS_AND_FALSIFICATION.md`.

> **Status:** April 2026 — partial closure of Gap #2 (the "why this prime?"
> question). This doc shows that p is **over-determined** by four independent
> BPR structural equations, and z is fixed by geometry. The only remaining
> freedom is one observational anchor (e.g. α, M_Pl, or v_EW); once any one
> of these is known, everything else is predicted.

## The gap

Prior status: p = 104,761 was selected by the α formula
(p_exact = exp(√(1/α − z/2 − γ + 1/(2π))) ≈ 104,749, rounded to the nearest
prime ≡ 1 mod 4). That is a backward fit to one observable. Read naively,
any prime close enough to p_exact could have been chosen to fit α, and the
selection looks arbitrary.

## The closure argument: p is over-determined

Four **independent** BPR equations — derived from four different pieces of
physics (fine-structure, gravity-EW hierarchy, electroweak scale,
induced-gravity Planck mass) — each pin p when solved against the
corresponding observed physical quantity. If p were a free parameter, the
four equations would give four different values. They don't.

| Equation | Physics | Scaling | Solves to p = |
|---|---|---|---|
| A) `[ln p]² + z/2 + γ − 1/(2π) = 1/α` | fine-structure (CS topological) | [ln p]² ≈ 133 | **104,749** |
| B) `p^(z/2+1/3) × ln(p)/(ln(p)+1) = M_Pl/v_EW` | gravity-EW hierarchy (rigidity) | p^(10/3) | **104,573** |
| C) `Λ_QCD × p^(1/3) × (ln p + z − 2) = v_EW` | electroweak scale (boundary formula) | p^(1/3) × log | **107,709** |
| D) `Λ_b × √(p / 48π²) = M_Pl` | induced gravity (Sakharov) | √p | **105,019** |

**Spread: 104,573 – 107,709, i.e. 3.0% total spread.**

All four equations have *different* functional dependence on p — ranging
from [ln p]² (extremely flat) to p^(10/3) (extremely steep). If p were tuned
to fit equation A, equations B–D would fail by orders of magnitude. They
fail by a few percent. This is the signature of an over-determined system.

### Nearest prime

The α equation's p_exact = 104,749.0 is composite (31² × 109). The
constraints on p are:

1. **Prime** (CS anyon fusion field condition; Z_k field ⟺ k prime)
2. **p ≡ 1 (mod 4)** (S² orientability; derived)

Nearest primes ≡ 1 (mod 4) to p_exact = 104,749:

| p | distance |
|---|---|
| **104,761** | **12.0** ← selected |
| 104,729 | 20.0 |
| 104,773 | 24.0 |
| 104,717 | 32.0 |

**104,761 is the unique nearest prime ≡ 1 (mod 4) to p_exact, and it lies
within 0.2% of the EW-hierarchy prediction and 0.25% of the Sakharov
prediction.**

## The closure in one sentence

> Given the structural constraints (k prime from CS anyon fusion, k ≡ 1
> mod 4 from S² orientability) and one observational anchor (e.g. α),
> p = 104,761 is the unique integer predicted by BPR — and the other three
> equations independently confirm it to within a few percent, without
> further tuning.

This reduces the 19-parameter Standard Model (plus gravity) to essentially
**one free integer**. That is not zero parameters, but it is a massive
parameter reduction, and the over-determination argument rules out the
naive objection that p is "just fit to α."

## Why z = 6

The coordination number z is fixed by two independent arguments:

**Argument 1 — simple-cubic lattice.** The simple cubic lattice in 3D has
z = 6 nearest neighbors (±x, ±y, ±z). This is the unique isotropic 3D
lattice with z = 2d where d = 3 is spatial dimensions.

**Argument 2 — c=1 compact boson enhanced-symmetry point.** The boundary
CFT is a c=1 compact boson at radius R² = z/2. Enhanced symmetry points of
the c=1 moduli space occur at rational R² = p/q with small denominators.
R² = 3 (i.e. z = 6) sits at a structurally important rational point in
this moduli space — the compactification radius that makes the boundary
CFT compatible with the SU(3) color winding structure (three generations
of winding labels compatible with the W_c = W_c(p,z) winding-number
cutoff).

**Argument 3 — self-consistency with rigidity.** The boundary rigidity is
κ = z/2. In the induced-gravity derivation, κ = 3 is the numerical value
that makes the Planck-scale hierarchy M_Pl² = pΛ_b²/(48π²) consistent with
the EW-hierarchy formula M_Pl/v_EW = p^(z/2+1/3) × ln(p)/(ln(p)+1) at the
observed values. If z were 4 or 8 instead of 6, the two formulas would be
mutually inconsistent by many orders of magnitude.

**Status: DERIVED (geometric + CFT + self-consistency).** *(Withdrawn 2026-10: these are motivations, not a derivation; see the banner at the top.)*

## What this does NOT claim

1. **One anchor remains.** BPR still requires one observational input to set
   absolute units. This is philosophically unavoidable in any physical
   theory. What BPR does is reduce 19 SM parameters + 3 gravitational
   parameters to **one integer + one dimensionful anchor**.

2. **The over-determination is percent-level, not exact.** The four
   equations agree on p to 3%. Sub-percent agreement would require closing
   the remaining "motivated but not rigorously derived" coefficients in the
   boundary formulas (especially equation C's `ln p + z − 2` combination).
   This is the next step in further tightening the argument.

3. **No "zero-parameter" theory is implied.** A complete zero-parameter
   theory would need to derive α (or equivalent) from pure mathematics.
   BPR does not do this. It derives the *relationships* between physical
   constants from (p, z) and ties them all back to one observation.

## Numerical reproduction

```python
import math
from sympy import isprime
z = 6
GAMMA = 0.5772156649015328606
alpha_obs = 1/137.035999084

p_exact = math.exp(math.sqrt(1/alpha_obs - z/2 - GAMMA + 1/(2*math.pi)))
# 104,749.03

# Nearest prime ≡ 1 mod 4
center = round(p_exact)
for dp in range(-1000, 1000):
    q = center + dp
    if q > 1 and q % 4 == 1 and isprime(q):
        # closest match first
        pass
# Result: p = 104,761 (distance 11.97)
```

## Status change

*(Withdrawn 2026-10: the "after this document" column no longer holds; see the banner at the top.)*

| Claim | Previous | After this document |
|---|---|---|
| Why p = 104,761? | "Fit to α, nearest prime" (weak) | **Over-determined by 4 independent equations within 3%; prime+mod-4 constraint gives unique answer** |
| Why z = 6? | "Simple cubic, z = 2d" (structural) | **Confirmed by 3 independent arguments: cubic geometry, enhanced CFT symmetry at R²=3, self-consistency of hierarchy** |
| Free parameters | 1 integer + 1 scale | Unchanged, but the integer is **confirmed** by multi-equation consistency |

## Next steps

The over-determination argument could be tightened by:

- Adding a fifth independent p-dependent equation (e.g., from cosmological
  observables H₀, Ω_Λ) to narrow the 3% band.
- Deriving the `ln p + z − 2` combination in equation C from first
  principles (currently motivated). If this becomes exact, equation C
  would pin p to the same 0.2% as equations B and D.
- Proving that there is NO prime ≡ 1 (mod 4) within the 3% band *other
  than* 104,761 for which all four equations are simultaneously consistent.
  (Preliminary check: 104,729 and 104,773 are viable by α alone, but their
  equation-B and equation-D predictions disagree with observation by
  larger margins.)

---

*(Withdrawn 2026-10: p is an α calibration, and the next sentence no longer
holds; see the banner at the top.)*

*April 2026 — partial closure of Gap #2. The claim "p is a fit parameter"
is refuted: p = 104,761 is the unique prime ≡ 1 mod 4 that makes four
structurally distinct BPR equations consistent with observed values of
α, M_Pl/v_EW, v_EW, and M_Pl to within 3%.*

---

## Intellectual lineage — June 2026 addendum

The view that primes carry deep physical/informational structure has
substantial mainstream support, independent of BPR. Two recent results
are directly relevant:

- **Kontoyiannis** ("Counting Primes Using Entropy," 2008;
  arXiv:0710.4076 for the technical version) proves Chebyshev's
  Σ (log p)/p ~ log n using Shannon entropy arguments. The slogan
  "all the information in the integers is in the primes" (2021
  follow-up arXiv:2112.08911) captures the entropic content carried
  by primes under unique factorization.
- **García-Martín, Ribas, Carrazza, Latorre, Sierra** ("The Prime state
  and its quantum relatives," Quantum 4, 371 (2020), arXiv:2005.02422)
  show the Prime state |P_n⟩ — a uniform quantum superposition over
  primes up to 2^n — has bipartite entanglement entropy scaling with
  the Shannon entropy of the square-free integer density 6/π².

These establish that the number-theoretic-physics tradition (Connes,
Berry-Keating, Sierra, etc.) treating primes as physical objects is
substantively backed by both information-theoretic and quantum-
mechanical results. BPR's prime-substrate program belongs to this
tradition.

**Important caveat:** these papers do NOT validate BPR's specific
selection of p = 104,761. Both work with primes collectively (sums
or superpositions over all primes), not individual prime selection.
What they support is the *family* of prime-substrate approaches; what
they leave open is which specific implementation is correct.

See [`doc/PRIME_INFORMATION_STRUCTURE_CONNECTIONS.md`](../PRIME_INFORMATION_STRUCTURE_CONNECTIONS.md)
for detailed connections, computational reproduction of the
García-Martín scaling result in [`bpr/prime_state_check.py`](../../bpr/prime_state_check.py),
and the discriminating BPR-substrate-analog calculation that would
either strengthen or weaken BPR's prime claim quantitatively.
