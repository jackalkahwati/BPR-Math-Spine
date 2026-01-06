(* Run via:
   wolframscript -file wolfram/run_bpr_demo.wls --output data/
*)

Get[FileNameJoin[{DirectoryName[$InputFileName], "BPR.wl"}]];

(* Robust arg parsing:
   - wolframscript -file does not reliably populate $ScriptCommandLine with user args.
   - wolframscript -script does.
   Support:
     - BPR_QUICK=1 env var
     - --quick flag (when available)
     - --output <dir> (when available)
*)
argv = Join[$ScriptCommandLine, $CommandLine];
outputDir = "data/";
posOut = FirstPosition[argv, "--output", None];
If[ListQ[posOut] && posOut[[1]] < Length[argv], outputDir = argv[[posOut[[1]] + 1]]];
quick = (Environment["BPR_QUICK"] === "1") || MemberQ[argv, "--quick"];

If[!DirectoryQ[outputDir], CreateDirectory[outputDir, CreateIntermediateDirectories -> True]];

(* Demo header: coupling constants (theory vs experimental upper bound vs demo knob) *)
Print["═══════════════════════════════════════════════════════"];
Print["BPR Casimir Spine - Coupling Constants"];
Print["═══════════════════════════════════════════════════════"];
Print[""];
Print["λ_theory  = κ ℓ_P² = ", ToString[ScientificForm[BPR`BPRPhysicalCouplingLambda[], 3], OutputForm]];
Print["  → Prediction: ΔF/F unmeasurably small at all accessible scales"];
Print[""];
Print["λ_eff     = ", ToString[ScientificForm[BPR`BPRPhenomenologicalCouplingLambdaFromName["StateOfArt2024"], 3], OutputForm]];
Print["  → Upper bound from |ΔF/F| < 10⁻³ at d = 100 nm"];
Print["  → Source: Decca et al. / Yale precision Casimir experiments (conservative)"];
Print[""];
Print["λ_demo    = 10⁻³ (shape demonstration only, not physical)"];
Print["═══════════════════════════════════════════════════════"];
Print[""];

(* Checkpoint 1 artifact *)
eig = BPR`BPRLaplacianEigenvaluesSphere[10];
eigFile = FileNameJoin[{outputDir, "laplacian_eigenvalues_wl.csv"}];
Export[eigFile, Prepend[eig, {"l", "lambda"}], "CSV"];

(* Eq (7) artifact *)
casimirFile = FileNameJoin[{outputDir, "casimir_deviation_wl.csv"}];
If[quick,
  Print["Running QUICK mode (fewer points / lower LMax) ..."];
  data = BPR`BPRCasimirSweepRows[0.2*10^-6, 5.0*10^-6, 12,
    "Geometry" -> "parallel_plates",
    "CouplingLambda" -> Automatic,
    "CorrectionModel" -> "calibrated",
    "LMax" -> 4,
    "QuadraturePointsTheta" -> 24,
    "QuadraturePointsPhi" -> 48,
    "OutputCSV" -> casimirFile
  ];,
  Print["Running FULL mode ..."];
  data = BPR`BPRCasimirSweepRows[0.2*10^-6, 5.0*10^-6, 40,
    "Geometry" -> "parallel_plates",
    "CouplingLambda" -> Automatic,
    "CorrectionModel" -> "calibrated",
    "LMax" -> 8,
    "QuadraturePointsTheta" -> 48,
    "QuadraturePointsPhi" -> 96,
    "OutputCSV" -> casimirFile
  ];
];

Print["BPR Wolfram demo complete."];
Print["- Eigenvalues CSV: ", eigFile];
Print["- Casimir sweep CSV: ", casimirFile];


