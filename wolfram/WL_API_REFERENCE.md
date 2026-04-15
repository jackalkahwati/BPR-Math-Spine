# BPR Wolfram Language API Reference

Load the package: `Get["path/to/wolfram/BPR.wl"]`

All functions are in the `BPR`` context and prefixed `BPR*`.

---

## Core.wl — Constants and Utilities

| Function | Signature | Description |
|----------|-----------|-------------|
| `BPRConstants` | `BPRConstants[]` | Returns an Association of physical constants (c, G, hbar, k_B, l_P, m_Pl, p, z, kappa). |
| `BPRGet` | `BPRGet[obj, key]` | Gets a value from a spine object (Association or rules list) in a runtime-compatible way. |
| `BPRPhysicalCouplingLambda` | `BPRPhysicalCouplingLambda[kappaBoundary]` | Returns lambda = kappaBoundary * l_P^2 (Planck-length-squared coupling). |
| `BPRAssociationToCSV` | `BPRAssociationToCSV[list, file]` | Exports a list of Associations to CSV with stable column order. |

---

## BoundaryField.wl — Laplacian Eigenvalues and Phase Solver

| Function | Signature | Description |
|----------|-----------|-------------|
| `BPRLaplacianEigenvaluesSphere` | `BPRLaplacianEigenvaluesSphere[lMax]` | Returns `{{l, l(l+1)}, ...}` for l=0..lMax (Checkpoint 1). Analytic sphere eigenvalues. |
| `BPRSphericalHarmonicCoefficients` | `BPRSphericalHarmonicCoefficients[f, lMax]` | Numerically estimates spherical harmonic coefficients of a function f on the unit sphere. |
| `BPRSolvePhaseSphereSpectral` | `BPRSolvePhaseSphereSpectral[f, kappa, lMax]` | Solves kappa * Delta_{S^2} phi = f on the unit sphere spectrally. Returns an Association with coefficients and residual. |
| `BPRPhaseEnergySphereSpectral` | `BPRPhaseEnergySphereSpectral[solution]` | Returns the boundary phase energy integral_{S^2} |grad phi|^2 dS from a spectral solution. |
| `BPRVerifyPhaseEquationSphereSpectral` | `BPRVerifyPhaseEquationSphereSpectral[solution]` | Verifies Eq (6a) kappa * Delta_{S^2} phi = f. Returns max residual (should be < 1e-10). |

---

## Casimir.wl — Force Prediction and Coupling Sweeps

| Function | Signature | Description |
|----------|-----------|-------------|
| `BPRStandardCasimirForce` | `BPRStandardCasimirForce[radius, opts]` | Returns the standard Casimir force for a sphere-plate geometry at given radius. |
| `BPRCasimirForce` | `BPRCasimirForce[radius, opts]` | Returns an Association with standard force, BPR correction dF, relative deviation dF/F, and coupling lambda used. |
| `BPRCasimirForceRow` | `BPRCasimirForceRow[radius, opts]` | Returns a plain numeric row `{R, F_Casimir, dF_BPR, dF/F}` for CSV export. |
| `BPRPhenomenologicalCouplingLambda` | `BPRPhenomenologicalCouplingLambda[experimentalBound, referenceSeparation, opts]` | Back-calculates lambda_eff from an experimental dF/F bound at a reference separation. |
| `BPRExperimentalBounds` | `BPRExperimentalBounds` | Association of named experimental dF/F bounds and reference separations (StateOfArt2024, Lamoreaux1997, etc.). |
| `BPRPhenomenologicalCouplingLambdaFromName` | `BPRPhenomenologicalCouplingLambdaFromName[name, opts]` | Computes lambda_eff using the bound named in BPRExperimentalBounds. |
| `BPRCasimirSweepRows` | `BPRCasimirSweepRows[rMin, rMax, n, opts]` | Returns a list of numeric rows for radii in [rMin, rMax] with n steps. |
| `BPRCasimirSweep` | `BPRCasimirSweep[rMin, rMax, n, opts]` | Sweeps radii and optionally exports a CSV matching the Python Eq(7) artifact. |

---

## Metric.wl — Symbolic Metric Perturbation (Eq 3 / 6b)

| Function | Signature | Description |
|----------|-----------|-------------|
| `BPRMetricPerturbation` | `BPRMetricPerturbation[phiExpr, couplingLambda, coords, coordSystem]` | Returns an Association with the metric perturbation tensor Delta g_{mu nu} = lambda * (grad phi)^2 * eta_{mu nu} and its trace. Works symbolically. |
| `BPRStressTensorScalarField` | `BPRStressTensorScalarField[phiExpr, coords]` | Returns a simplified scalar-field stress tensor T^{mu nu} (flat-space, no Christoffel terms). |
| `BPRVerifyConservation` | `BPRVerifyConservation[T, coords]` | Returns the (simplified) divergence of T^{mu}{}_nu. Should vanish for a conserved stress tensor. |

**Note:** Conservation check uses flat-space simplification (no Christoffel symbol terms). Full covariant conservation requires curved-space extension.

---

## Information.wl — Integrated Information and Consciousness Coupling

| Function | Signature | Description |
|----------|-----------|-------------|
| `BPRPartitionFieldValues` | `BPRPartitionFieldValues[values, nPartitions]` | Partitions a vector of samples into nPartitions groups for information estimation. |
| `BPRMutualInformationFromHistograms` | `BPRMutualInformationFromHistograms[x, y, nBins]` | Estimates mutual information I(x;y) from binned histograms. |
| `BPRIntegratedInformationPhi` | `BPRIntegratedInformationPhi[values, nPartitions]` | Computes an IIT-inspired integrated information Phi from field samples. |
| `BPRInformationAction` | `BPRInformationAction[values, xi, nPartitions]` | Returns S_info = -xi * Area * Phi (Eq 4). |
| `BPRConsciousnessCoupling` | `BPRConsciousnessCoupling[values, params]` | Computes the six-factor chi_b coupling (Eq 5) combining Phi, membrane voltage, temperature, and coherence. |

---

## E8.wl — E8 Root System

| Function | Signature | Description |
|----------|-----------|-------------|
| `BPRE8Roots` | `BPRE8Roots[]` | Returns the list of 240 E8 roots as 8-vectors. Uses WL built-in `RootSystemData`. |
| `BPRE8CartanMatrix` | `BPRE8CartanMatrix[]` | Returns the 8x8 E8 Cartan matrix. |
| `BPRE8SimpleRoots` | `BPRE8SimpleRoots[]` | Returns a choice of 8 simple roots for E8. |

---

## RPST.wl — Relativistic Prime Substrate Theory

| Function | Signature | Description |
|----------|-----------|-------------|
| `BPRRPSTLegendreSymbol` | `BPRRPSTLegendreSymbol[a, p]` | Returns the Legendre symbol (a/p) for odd prime p: 0, 1, or -1. |
| `BPRRPSTPrimeField` | `BPRRPSTPrimeField[p]` | Returns an Association representing arithmetic in Z_p with mod-p add, mul, and inverse. |
| `BPRRPSTSubstrateState` | `BPRRPSTSubstrateState[q, pi, p, t]` | Constructs an RPST phase-space state on Z_p with position q, momentum pi, prime p, and time t. |
| `BPRRPSTSymplecticEvolution` | `BPRRPSTSymplecticEvolution[p, J]` | Constructs a reversible update rule on Z_p with coupling J. Returns an evolution function. |
| `BPRRPSTMinimalSignedDifference` | `BPRRPSTMinimalSignedDifference[a, b, p]` | Returns the minimal signed difference (a-b) mod p, in range (-p/2, p/2]. |
| `BPRRPSTComputeWindingNumber` | `BPRRPSTComputeWindingNumber[qValues, loopIndices, p]` | Computes the winding number W = sum of minimal signed differences around a loop in Z_p. |
| `BPRRPSTVerifyChargeConservation` | `BPRRPSTVerifyChargeConservation[q0, q1, p, loops]` | Verifies winding conservation across a time step: W(q1) = W(q0) for all loops. |
| `BPRRPSTCoarseGraining` | `BPRRPSTCoarseGraining[positions, coarseScale]` | Constructs a kernel coarse-grainer mapping fine-grained Z_p positions to smooth field values. |
| `BPRRPSTVerifyWaveEquation` | `BPRRPSTVerifyWaveEquation[trajectory, positions, evalPoints, dt, coarseScale, c, tolerance]` | Verifies that a substrate trajectory satisfies the discrete wave equation at given evaluation points. |
| `BPRRPSTHamiltonianMatrix` | `BPRRPSTHamiltonianMatrix[p]` | Builds the Legendre-symbol outer-product Hamiltonian matrix H_{ij} = (i*j / p) for substrate prime p. |
| `BPRRPSTHamiltonianEigenvalues` | `BPRRPSTHamiltonianEigenvalues[p]` | Returns sorted eigenvalues of the RPST Hamiltonian for substrate prime p. |

---

## AdjacentTheories.wl — Cross-Domain Predictions

| Function | Signature | Description |
|----------|-----------|-------------|
| `BPRDMRelicAbundance` | `BPRDMRelicAbundance[opts]` | Returns Omega_DM h^2 from thermal freeze-out via BPR coupling. |
| `BPRDMBoundaryCollectiveEnhancement` | `BPRDMBoundaryCollectiveEnhancement[p]` | Returns N_coh with decoherence factor f_decoh = 1 - Sqrt[E]/p^(1/4). |
| `BPRChargedLeptonMasses` | `BPRChargedLeptonMasses[opts]` | Returns `{m_e, m_mu, m_tau}` in MeV derived from S^2 l-mode boundary spectrum. |
| `BPRLeptonLModes` | `BPRLeptonLModes[]` | Returns the l-modes for (e, mu, tau) with l_mu = Sqrt[14*15] derived from winding. |
| `BPRNuclearSaturationDensity` | `BPRNuclearSaturationDensity[rCh]` | Returns nuclear saturation density rho_0 in fm^-3 from charge radius rCh. |
| `BPRBindingEnergyPerNucleon` | `BPRBindingEnergyPerNucleon[A, Z]` | Returns B/A in MeV using Bethe-Weizsacker with BPR-derived coefficients. |
| `BPRMagicNumbers` | `BPRMagicNumbers[]` | Returns the nuclear magic numbers {2, 8, 20, 28, 50, 82, 126} from boundary shell structure. |
| `BPRSuperconductorTc` | `BPRSuperconductorTc[N0V, TDebye]` | Returns T_c in K via BCS + Eliashberg vertex correction. |
| `BPRSuperconductorN0VDerived` | `BPRSuperconductorN0VDerived[EFermi, TDebye, p, z]` | Returns N(0)V from BPR boundary coupling, coordination number z^2, and Eliashberg correction. |
| `BPRPionMass` | `BPRPionMass[opts]` | Returns m_pi in MeV via the Gell-Mann-Oakes-Renner relation with BPR quark condensate. |
| `BPRInverseAlphaFromSubstrate` | `BPRInverseAlphaFromSubstrate[p, z]` | Returns 1/alpha from substrate winding-sector boundary coupling. Default: p=104761, z=6 -> 137.036. |
| `BPRElectroweakScaleGeV` | `BPRElectroweakScaleGeV[p, z, LambdaQCD]` | Returns v_EW in GeV from BPR impedance function and QCD scale. |
| `BPRConstraintPotentialDoubleWell` | `BPRConstraintPotentialDoubleWell[kappa, eta, lambdaK]` | Double-well potential V(kappa) for boundary phase constraint. |
| `BPRConstraintPotentialDerivative` | `BPRConstraintPotentialDerivative[kappa, eta, lambdaK]` | Derivative dV/dkappa of the double-well constraint potential. |

---

## Geometry.wl — Boundary Discretization (Placeholder)

| Function | Signature | Description |
|----------|-----------|-------------|
| `BPRMakeBoundary` | `BPRMakeBoundary[meshSize, geometry, radius]` | Returns a discretized boundary region. Currently supports `"Sphere"` via analytic eigenvalues; FEM extension planned. |

---

## Quick Start

```mathematica
(* Load package *)
Get["/path/to/wolfram/BPR.wl"]

(* Checkpoint 1: Laplacian eigenvalues on S^2 *)
BPRLaplacianEigenvaluesSphere[4]
(* {{0,0},{1,2},{2,6},{3,12},{4,20}} *)

(* Derive 1/alpha from substrate integers *)
BPRInverseAlphaFromSubstrate[104761, 6]
(* ~137.036 *)

(* Casimir force deviation at 100nm *)
BPRCasimirForce[100*^-9]["deltaFoverF"]

(* Run full Eq(7) sweep and export *)
BPRCasimirSweep[50*^-9, 500*^-9, 20, "Export" -> True, "File" -> "casimir_wl.csv"]
```

---

## Demo Scripts

```bash
# Checkpoint 1 + Eq(7) curve
wolframscript -script wolfram/run_bpr_demo.wls

# Coupling comparison (theory vs phenomenological)
wolframscript -script wolfram/run_casimir_coupling_comparison.wls
```

## Test Suite

```bash
# WL smoke tests (13 assertions, no full license required)
wolframscript -script wolfram/tests/run_equation_smoke.wls

# Python coherence tests
python3 -m pytest tests/test_coherence_verification.py -p no:asyncio -q

# Full Python suite (1225 tests)
python3 -m pytest tests/ -p no:asyncio -q
```
