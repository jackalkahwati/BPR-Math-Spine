# Wolfram Language Port Map (what to change)

Wolfram Research asked that BPR be presented in **Wolfram Language** and be **runnable**. This map describes the minimal changes needed to make the repo “WL-first” while preserving the existing Python pipeline.

## What Wolfram’s email implies (deliverables)

- **Runnable Wolfram Language code** (not just pseudo-code)
- A **demo** that reproduces your headline claim artifacts (at minimum Eq (7) curve + your “math checkpoints”)
- A **test harness** in WL (`TestReport`) so their team can run and validate quickly

## Current Python → Wolfram mapping (by module)

- **`bpr/geometry.py` → WL region/mesh helpers**
  - Python: `make_boundary(mesh_size, geometry="sphere", radius=...)`
  - WL target: `BPRMakeBoundary[meshSize, geometry, radius]`
  - Suggested WL building blocks:
    - `Sphere[]`, `Cylinder[]`, `BoundaryDiscretizeRegion`, `DiscretizeRegion`
    - (Later) `NDSolveValue` with FEM for surface PDEs

- **`bpr/boundary_field.py` → WL Laplacian / eigenmodes / PDE solve**
  - Python: `solve_phase(boundary_mesh, source, kappa)`
  - Python: `solve_eigenvalue_problem(boundary_mesh, n_modes)`
  - WL target:
    - `BPRSolvePhase[mesh, source, kappa]` (phase field)
    - `BPRLaplacianEigenmodes[mesh, nModes]` (numerical checkpoint)
  - Minimal starting point (portable):
    - Analytic sphere eigenvalues: `BPRLaplacianEigenvaluesSphere[lMax]` (already implemented in `wolfram/BPR.wl`)
  - “Full port” step:
    - Replace analytic-only / toy models with `NDEigensystem` / FEM on a discretized sphere boundary.

- **`bpr/metric.py` → WL symbolic tensor construction**
  - Python uses SymPy; WL can do this natively with symbolic expressions and matrices.
  - WL target:
    - `BPRMetricPerturbation[phi, lambda, coords] -> Δg`
    - `BPRStressTensor[...]`
    - `BPRVerifyConservation[...]`
  - Note: your current Python conservation verification is simplified (connection terms omitted). WL version should match the repo’s “math spine” intent (start simplified, then upgrade).

- **`bpr/casimir.py` → WL prediction curve + export**
  - Python: `casimir_force`, `sweep_radius` export CSV for Eq (7)
  - WL target:
    - `BPRCasimirForce[...]` + `BPRCasimirSweep[...]` exporting `casimir_deviation_wl.csv`
  - Minimal runnable version is already in `wolfram/BPR.wl` with:
    - Standard force model + explicit energy model + fractal scaling term

- **`bpr/information.py` → WL integrated information + coupling**
  - Python uses histograms, pairwise mutual information, and a 6-factor coupling.
  - WL is well-suited here:
    - `HistogramList`, `Entropy`, `MutualInformation` (or compute MI from joint histogram)
    - `ClusteringComponents` / graph-based partitions
  - WL target:
    - `BPRIntegratedInformation[phiSamples, partitions]`
    - `BPRConsciousnessCoupling[Phi, E, S, U, I, params]`

- **`notebooks/*.ipynb` → `.nb` / `.wl` notebooks**
  - `01_boundary_laplacian.ipynb` → `BPR_BoundaryLaplacian.nb`
  - `02_metric_perturbation.ipynb` → `BPR_MetricPerturbation.nb`
  - `03_casimir_prediction.ipynb` → `BPR_CasimirPrediction.nb`
  - `04_e8_index.ipynb` → `BPR_E8Index.nb`
  - For E8 specifically, WL has strong built-ins (`RootSystemData["E8", ...]`) which can replace custom Python group theory.

## Repo changes to make (practical checklist)

### 1) Add a Wolfram folder with package + demo + tests (done in this commit)

- `wolfram/BPR.wl` (package)
- `wolfram/run_bpr_demo.wls` (CLI demo)
- `wolfram/tests/BPRTests.wlt` (tests)
- `wolfram/README_Wolfram.md` (how to run)

### 2) Produce the “review artifacts” Wolfram will look for

- **CSV outputs**: Eq (7) curve + eigenvalue checkpoint
- **One-command run**: `wolframscript -file ...`
- **One-command test**: `wolframscript -code "TestReport[...]"` (or a `.wls` wrapper)

### 3) (Optional but recommended) Add WL notebooks mirroring your Jupyter ones

Wolfram will digest `.nb` much faster than a large `.wl` file.

### 4) Decide how deep the FEM port needs to go

If the goal is “meaningful critique,” you usually need:

- the **same public equation objects** (symbols/tensors) in WL
- the **same falsifiable curve generation** in WL
- at least one of:
  - analytic derivations with plots, or
  - numerical PDE solve in WL (FEM) to replace FEniCS

## Current status in this repo

- WL scaffold is present in `wolfram/` and runnable via `wolframscript`.
- Next “deep port” steps are: surface PDE solve + information module + E8 notebook conversion.




