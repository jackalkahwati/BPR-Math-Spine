(* Run via:
   wolframscript -script wolfram/run_casimir_coupling_comparison.wls --output data/ --quick
*)

Get[FileNameJoin[{DirectoryName[$InputFileName], "BPR.wl"}]];

argv = Join[$ScriptCommandLine, $CommandLine];
outputDir = "data/";
posOut = FirstPosition[argv, "--output", None];
If[ListQ[posOut] && posOut[[1]] < Length[argv], outputDir = argv[[posOut[[1]] + 1]]];
quick = (Environment["BPR_QUICK"] === "1") || MemberQ[argv, "--quick"];

If[!DirectoryQ[outputDir], CreateDirectory[outputDir, CreateIntermediateDirectories -> True]];

(* Match demo defaults *)
geometry = "parallel_plates";
model = "calibrated";
kappa = 1.0;
delta = 1.37;
rF = 10^-6;
alphaBpr = 0.1;
src = Function[{θ, ϕ}, Re[SphericalHarmonicY[2, 0, θ, ϕ]]];

If[quick,
  lMax = 4; nθ = 24; nϕ = 48; nPts = 24; dMin = 0.1*10^-6; dMax = 5.0*10^-6;,
  lMax = 8; nθ = 48; nϕ = 96; nPts = 60; dMin = 0.1*10^-6; dMax = 5.0*10^-6;
];

(* Compute boundary energy ONCE (expensive step) *)
sol = BPR`BPRSolvePhaseSphereSpectral[
  src, kappa, lMax,
  "Radius" -> 1.0,
  "QuadraturePointsTheta" -> nθ,
  "QuadraturePointsPhi" -> nϕ
];
energy = N @ BPR`BPRPhaseEnergySphereSpectral[sol];

(* Couplings *)
lamTheory = N @ BPR`BPRPhysicalCouplingLambda[];
lamEff = N @ BPR`BPRPhenomenologicalCouplingLambdaFromName[
  "StateOfArt2024",
  "Geometry" -> geometry,
  "Kappa" -> kappa,
  "LMax" -> lMax,
  "QuadraturePointsTheta" -> nθ,
  "QuadraturePointsPhi" -> nϕ,
  "DeltaBPR" -> delta,
  "ReferenceScale" -> rF,
  "AlphaBPR" -> alphaBpr,
  "BoundarySource" -> src,
  "CorrectionModel" -> model,
  "EnergyOverride" -> energy
];
lamDemo = 10^-3;

(* Log-spaced grid without Subdivide (portable across runtimes) *)
dGrid = Module[{a = Log[dMin], b = Log[dMax]},
  If[nPts == 1, {dMin}, Table[Exp[a + (b - a) * i/(nPts - 1)], {i, 0, nPts - 1}]]
];

rowFor[d_, lam_] := Module[{res},
  res = BPR`BPRCasimirForce[d,
    "Geometry" -> geometry,
    "CouplingLambda" -> lam,
    "CorrectionModel" -> model,
    "Kappa" -> kappa,
    "LMax" -> lMax,
    "QuadraturePointsTheta" -> nθ,
    "QuadraturePointsPhi" -> nϕ,
    "DeltaBPR" -> delta,
    "ReferenceScale" -> rF,
    "AlphaBPR" -> alphaBpr,
    "BoundarySource" -> src,
    "FieldEnergyOverride" -> energy
  ];
  {
    N @ BPR`BPRGet[res, "F_Casimir [N]"],
    N @ BPR`BPRGet[res, "ΔF_BPR [N]"],
    N @ BPR`BPRGet[res, "relative_deviation"]
  }
];

rows = Table[
  Module[{d = dGrid[[i]], f0, dF1, rel1, dF2, rel2, dF3, rel3, tmp},
    tmp = rowFor[d, lamTheory]; f0 = tmp[[1]]; dF1 = tmp[[2]]; rel1 = tmp[[3]];
    tmp = rowFor[d, lamEff];   dF2 = tmp[[2]]; rel2 = tmp[[3]];
    tmp = rowFor[d, lamDemo];  dF3 = tmp[[2]]; rel3 = tmp[[3]];
    {
      N[d],
      f0,
      energy,
      lamTheory, dF1, rel1,
      lamEff,   dF2, rel2,
      lamDemo,  dF3, rel3
    }
  ],
  {i, 1, Length[dGrid]}
];

header = {
  "d [m]",
  "F_Casimir [N]",
  "field_energy",
  "λ_theory", "ΔF_theory [N]", "rel_theory",
  "λ_eff",   "ΔF_eff [N]",   "rel_eff",
  "λ_demo",  "ΔF_demo [N]",  "rel_demo"
};

outFile = FileNameJoin[{outputDir, "casimir_coupling_comparison_wl.csv"}];
Export[outFile, Prepend[rows, header], "CSV"];

Print["BPR coupling comparison complete."];
Print["- Output CSV: ", outFile];
Print["- quick=", quick, "  energy=", ScientificForm[energy, 6]];
Print["- λ_theory=", ScientificForm[lamTheory, 6]];
Print["- λ_eff=", ScientificForm[lamEff, 6]];
Print["- λ_demo=", ScientificForm[lamDemo, 6]];


