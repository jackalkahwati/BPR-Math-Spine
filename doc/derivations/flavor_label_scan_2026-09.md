# Flavor-label operator scan (preregistered) — 2026-09-10

> **Status: NEGATIVE.** No frozen-ingredient operator in the candidate list
> reproduces any flavor sector's labels, and no candidate beats chance after
> correcting for the number of comparisons. The nine labels remain
> CONJECTURAL. Module: `bpr/flavor_label_scan.py`; lock:
> `tests/test_flavor_label_scan.py`.

## Why this scan

The status page lists "derive at least one fermion mass label as an
eigenvalue of an explicit Hamiltonian" as a condition for restoring the
narrowed claim. Before building any new operator it is worth asking whether
an operator the framework already has produces the labels. This is a cheap
test with a fixed protocol, so a positive result would be informative and a
negative one costs nothing but honesty.

## Preregistration (fixed before any spectrum was computed)

**Targets.** Label integers {1, 4, 24, 30, 59, 283}; eigenvalue integers
{1, 4, 24, 30, 210, 283, 3481}. (210 is the squared muon label.)

**Candidates** (frozen ingredients only; no tunable parameter anywhere):

| | Operator | Ingredient it uses |
|---|---|---|
| C1 | Round S² scalar Laplacian, ℓ(ℓ+1) | Postulate: S² boundary |
| C2 | S² Laplacian on O_h-invariant harmonics (Molien 1/((1−t⁴)(1−t⁶))) | Cubic tiling symmetry that fixes z=6 |
| C3 | Same with chiral O (adds ℓ=9) | as above |
| C4 | Compact boson with CCR rule m ≡ 0 mod 6 | Postulate 0 |
| C5 | Graph Laplacian / adjacency of the z=6 shell (octahedron) | z=6 |
| C6 | Star and star+shell graphs (7 sites) | z=6 |
| C7 | D_n electric Casimirs, n ∈ {5,8,9,12} | Postulate 0d |
| C8 | Winding-shifted ℓ(ℓ+√3) | Down-type rule |

**Scoring.** Spectra truncated at 300. A hit is an exact integer match. The
chance rate is the fraction of integers in [1,300] present in the spectrum.
Report a candidate only if the exact binomial tail probability of its hit
count is below 0.05 after Bonferroni correction for the 16 (candidate, test)
comparisons, and it has at least two hits.

## Result

| Candidate | Eigenvalue hits | Expected by chance | p (tail) | p × 16 |
|---|---|---|---|---|
| C1 round S² | 30, 210 | 0.32 | 0.037 | 0.59 |
| C2 O_h harmonics | 210 | 0.14 | 0.13 | 1 |
| C3 O harmonics | 210 | 0.20 | 0.18 | 1 |
| C4 CCR | 30, 210 | 0.32 | 0.037 | 0.59 |
| C5 octahedron | 4 | 0.04 | 0.039 | 0.63 |
| C6 star graphs | 1 | 0.06 | 0.059 | 0.94 |
| C7 D_n Casimirs | 4 | 0.10 | 0.096 | 1 |
| C8 ℓ(ℓ+√3) | none | 0 | 1 | 1 |

Label tests are uninformative: every candidate with integer labels has
density near 1 in [1,300], so hits are guaranteed.

**Interesting after correction: none.**

## Observations recorded, not claimed

- 30 = 5·6 and 210 = 14·15 are exact Laplacian eigenvalues ℓ(ℓ+1) on the
  round sphere (ℓ = 5 and ℓ = 14). The retained ansatz treats 30 as a label
  (mass ∝ ℓ(ℓ+W_c)) and 210 as a squared label; neither usage is the
  Laplacian usage. Two coincidences among seven targets at 5% density is a
  p ≈ 0.04 event before the look-elsewhere correction and p ≈ 0.6 after.
- ℓ = 4 is the first non-trivial O_h-invariant harmonic on S² (the cubic
  harmonic). That makes "the first excited mode a cubic tiling can see is
  ℓ = 4" a true statement about the sphere. It does not extend to the rest
  of the down sector: ℓ_d = 1 is not O_h-invariant, and ℓ_b = 30 is one of
  many allowed degrees.
- The octahedron Laplacian has eigenvalue 4 with multiplicity 3 and 6 with
  multiplicity 2. Single hits do not pass the rule.

## What this rules out and what it does not

It rules out the cheap route: none of the operators already in the
framework, applied without adjustment, produces a flavor sector. It does not
rule out an operator that has not been written down. The next step on this
problem is therefore constructive, not a scan: specify a fermionic boundary
action, compute its spectrum, and compare with a frozen uncertainty. Until
then the labels stay CONJECTURAL and this negative stays on record.
