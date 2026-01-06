# BPR Math Spine: Validation Summary (Python vs Wolfram Language)

## Executive summary

The repo currently has **two validated implementations**:

- **Wolfram Language (WL)**: validated for the *WL-native spine modules implemented so far* (boundary spectrum/solver, Eq(3/6b), Eq(4), Eq(5), Eq(7) calibrated Casimir, plus RPST scaffolding).
- **Python**: includes additional modules (HKLL, symbolic conservation verification, resonance/GUE tooling) that are **not yet ported/validated in WL**.

This doc is intended to be a precise “what is validated where” snapshot, suitable for sharing with reviewers.

---

## 1) WL: passing test suites (as-run via `wolframscript`)

Run:

```bash
cd /Users/jackal-kahwati/BPR-Math-Spine
wolframscript -timeout 60 -code 'TestReport["wolfram/tests/BPRTests.wlt"]'
wolframscript -timeout 60 -code 'TestReport["wolfram/tests/BPREquationTests.wlt"]'
wolframscript -timeout 60 -code 'TestReport["wolfram/tests/BPRRPSTTests.wlt"]'
```

### What those WL tests cover

- **Checkpoint 1 (sphere spectrum)**: \( \lambda_\ell=\ell(\ell+1) \) ladder.
- **Eq (6a)**: spectral residual check for \( \kappa \Delta\phi=f \) on \(S^2\).
- **Eq (3)/(6b)**: metric perturbation structure + symmetry sanity.
- **Eq (4)**: information action is numeric and nonpositive for \(\xi>0\).
- **Eq (5)**: consciousness coupling is numeric and nonnegative.
- **Eq (7)**:
  - **λ_theory** (\(\lambda=\kappa \ell_P^2\)) gives essentially zero deviation (λ→0 recovery).
  - **λ_eff** constructed from an experimental bound satisfies \(|\Delta F/F|\le \varepsilon\) at the reference separation.
- **RPST (Eq 0a–0e, partial)**: finite-field closure, symplectic reversibility, winding number sanity, Hamiltonian matrix/eigenvalues basic properties.

### Not yet covered in WL (important)

- **Checkpoint 2 / full covariant conservation** \( \nabla_\mu T^{\mu\nu}=0 \): WL has a scaffold, but not a full symbolic proof test suite.
- **HKLL**, **Riemann-zero mode checks**, **GUE statistics verification**: present in Python, not yet ported/validated in WL.

---

## 2) Casimir CSV artifacts (WL)

### 2.1 `data/laplacian_eigenvalues_wl.csv`

This should match exactly:
\[
\lambda_\ell=\ell(\ell+1)
\]

### 2.2 `data/casimir_deviation_wl.csv` (λ_theory sweep)

This is the **Planck-normalized λ_theory regime**. Typical interpretation:

- baseline \(F_0(d)\) is attractive (negative)
- \( \Delta F \) is so small that \(F_\text{total}\approx F_0\)
- \(|\Delta F/F_0|\) is \(\sim 10^{-58}\)–\(10^{-55}\) in the demo sweep range (unmeasurable at accessible separations)

### 2.3 `data/casimir_coupling_comparison_wl.csv` (λ_theory vs λ_eff vs λ_demo)

This comparison CSV includes the **100 nm anchor** and shows:

- **λ_eff** hits \(|\Delta F/F_0|=10^{-3}\) at **100 nm** by construction.
- **λ_theory** remains ~\(10^{-58}\) at 100 nm.
- **λ_demo** is a visualization knob only (not physically meaningful under the calibrated model).

---

## 3) Geometry convention (WL vs Python)

The WL demo’s default `"Geometry" -> "parallel_plates"` uses a toy plate-area model:

- \(A=(2d)^2\) so the *force* scales like \(F\propto d^{-2}\).

Many Python pipelines use a per-unit-area convention:

- \(F/A\propto d^{-4}\).

Both are internally consistent, but **λ_eff depends on which baseline you constrain against**. For paper-facing falsifiable comparisons, a per-area or experimentally matched geometry (e.g. sphere–plate) is usually the standard.

---

## 4) Recommended next steps

1. Keep WL CSVs labeled with **`d [m]`** (not `R [m]`) to avoid reviewer confusion.
2. Add a WL geometry option for **per-unit-area** and/or **sphere–plate** to match common experimental reporting.
3. Port + validate in WL (in order): **GUE/Riemann checks**, **symbolic conservation**, **HKLL**.


