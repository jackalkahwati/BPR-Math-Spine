# BPR in Wolfram Language (for Wolfram Research review)

This folder is a **Wolfram Language** implementation of the BPR “math spine” APIs in this repo, intended to be runnable by Wolfram Research (e.g. via `wolframscript`) and to reproduce the **headline artifacts**:

- **Checkpoint 1**: Sphere Laplacian eigenvalues \( \lambda_\ell = \ell(\ell+1) \)
- **Eq (7) artifact**: a `casimir_deviation.csv` prediction curve with columns compatible with the Python pipeline
- **Eq (3) artifact**: a symbolic metric perturbation tensor \( \Delta g_{\mu\nu} \) you can inspect/compute with in WL

## Requirements

- Wolfram Engine / Mathematica / Wolfram Desktop that includes `wolframscript`

## Quick run (CLI)

From the repo root:

```bash
wolframscript -file wolfram/run_bpr_demo.wls --output data/
```

If you want a fast run first:

```bash
# Option 1 (recommended): use -script so flags are passed through
wolframscript -script wolfram/run_bpr_demo.wls --output data/ --quick

# Option 2: use an environment variable (works even with -file)
BPR_QUICK=1 wolframscript -file wolfram/run_bpr_demo.wls --output data/
```

Outputs:
- `data/laplacian_eigenvalues_wl.csv`
- `data/casimir_deviation_wl.csv`

## Run all WL tests (equation-by-equation)

```bash
wolframscript -code 'TestReport[{"wolfram/tests/BPRTests.wlt","wolfram/tests/BPREquationTests.wlt"}]'
```

## If `TestReport` trips a license error (fallback smoke runner)

Some Wolfram Engine licenses are sensitive to longer-running `TestReport` sessions. This runner does the same checks but prints PASS/FAIL line-by-line and exits quickly:

```bash
wolframscript -script wolfram/tests/run_equation_smoke.wls
```

## WL-native scope (what’s implemented)

- **Boundary Laplacian / solver**: WL-native spectral Laplace–Beltrami solve on \(S^2\) using spherical harmonics (`wolfram/BPR/BoundaryField.wl`).
- **Casimir Eq (7) artifact**: WL-native sweep that computes boundary energy from the solved phase field and exports a CSV compatible with the Python pipeline (`wolfram/BPR/Casimir.wl`).
- **Metric + conservation scaffold**: WL-native symbolic tensor + simplified conservation check matching the current repo’s “math spine” intent (`wolfram/BPR/Metric.wl`).
- **Information + consciousness**: WL-native IIT-inspired Φ and six-factor coupling (`wolfram/BPR/Information.wl`).
- **E8**: WL-native E8 objects via `RootSystemData` (`wolfram/BPR/E8.wl`).

## WL notebooks

WL notebooks mirroring the Python notebooks live in `wolfram/notebooks/`.


