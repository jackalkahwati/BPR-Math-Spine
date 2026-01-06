(* ::Package:: *)

BeginPackage["BPR`"];

BPRStandardCasimirForce::usage =
  "BPRStandardCasimirForce[radius, opts] returns the standard Casimir force for a simple geometry model.";

BPRCasimirForce::usage =
  "BPRCasimirForce[radius, opts] returns an Association with standard force, BPR correction, and totals (Eq 7 artifact).";

BPRCasimirForceRow::usage =
  "BPRCasimirForceRow[radius, opts] returns a plain numeric row {R, F_Casimir, ΔF_BPR, F_total, relative_deviation, field_energy}. Useful for lightweight WL runtimes (e.g. Mathics).";

BPRCasimirSweepRows::usage =
  "BPRCasimirSweepRows[rMin, rMax, n, opts] returns a list of numeric rows (and can export CSV).";

BPRCasimirSweep::usage =
  "BPRCasimirSweep[rMin, rMax, n, opts] sweeps radii and optionally exports a CSV matching Python columns.";

Begin["`Private`"];

Needs["BPR`"]; (* already loaded when used via BPR.wl; harmless otherwise *)

(* Mathics compatibility: avoid Subdivide by using explicit log-spacing. *)
ClearAll[$BPRLogSpace];
$BPRLogSpace[rMin_?NumericQ, rMax_?NumericQ, n_Integer?Positive] := Module[
  {a = Log[rMin], b = Log[rMax]},
  If[n == 1, {rMin}, Table[Exp[a + (b - a) * i/(n - 1)], {i, 0, n - 1}]]
];

(* Constants *)
$CasimirPrefactor := BPRConstants[]["CASIMIR_PREFACTOR"];

BPRStandardCasimirForce::badgeom = "Unknown geometry: `1`.";

Options[BPRStandardCasimirForce] = {"Geometry" -> "parallel_plates"};
BPRStandardCasimirForce[radius_?NumericQ, opts : OptionsPattern[]] := Module[
  {geometry = OptionValue["Geometry"], r = radius, area, gap, length},
  Which[
    geometry === "parallel_plates",
    area = (2 r)^2; (* match python demo assumption *)
    -$CasimirPrefactor * area / r^4,

    geometry === "sphere",
    gap = 0.1 r;
    -(Pi^3 * BPRConstants[]["hbar"] * BPRConstants[]["c"]) * r / (360 gap^3),

    geometry === "cylinder",
    length = 2 r;
    gap = 0.1 r;
    -(Pi^2 * BPRConstants[]["hbar"] * BPRConstants[]["c"]) * length / (240 gap^3),

    True,
    (Message[BPRStandardCasimirForce::badgeom, geometry]; Indeterminate)
  ]
];

(* Eq (7) correction model: use solved boundary field energy, then apply radius scaling + fractal factor.
   This matches the current Python repo's structure but makes the “energy” computable WL-natively. *)
Options[BPRCasimirForce] = {
  "Geometry" -> "parallel_plates",
  "CouplingLambda" -> 10^-3,
  "Kappa" -> 1.0,
  "LMax" -> 8,
  "QuadraturePointsTheta" -> Automatic,
  "QuadraturePointsPhi" -> Automatic,
  "DeltaBPR" -> 1.37,
  "ReferenceScale" -> 10^-6,
  "BoundarySource" -> Automatic
};

BPRCasimirForce[radius_?NumericQ, opts : OptionsPattern[]] := Module[
  {
    geometry = OptionValue["Geometry"],
    lam = OptionValue["CouplingLambda"],
    kappa = OptionValue["Kappa"],
    lMax = OptionValue["LMax"],
    nθ = OptionValue["QuadraturePointsTheta"],
    nϕ = OptionValue["QuadraturePointsPhi"],
    delta = OptionValue["DeltaBPR"],
    rF = OptionValue["ReferenceScale"],
    src = OptionValue["BoundarySource"],
    f0,
    solution,
    energy,
    geometricFactor,
    baseCorrection,
    radiusScaling,
    fractalFactor,
    dF,
    fT
  },

  (* default source: a smooth l=2,m=0 excitation in θ,ϕ coordinates *)
  If[src === Automatic,
    src = Function[{θ, ϕ}, Re[SphericalHarmonicY[2, 0, θ, ϕ]]];
  ];

  f0 = BPRStandardCasimirForce[radius, "Geometry" -> geometry];

  (* Solve on S^2 (unit sphere); radius enters only through the Casimir scaling and the BPR correction model. *)
  solution = BPRSolvePhaseSphereSpectral[
    src,
    kappa,
    lMax,
    "Radius" -> 1.0,
    Sequence @@ DeleteCases[
      {
        If[nθ === Automatic, Nothing, "QuadraturePointsTheta" -> nθ],
        If[nϕ === Automatic, Nothing, "QuadraturePointsPhi" -> nϕ]
      },
      Nothing
    ]
  ];
  energy = BPRPhaseEnergySphereSpectral[solution];

  geometricFactor = Which[
    geometry === "parallel_plates", 1.0,
    geometry === "sphere", 4 Pi,
    geometry === "cylinder", 2 Pi,
    True, 1.0
  ];

  baseCorrection = lam * energy / radius^2;
  radiusScaling = (radius/10^-6)^(-1); (* match python demo radius scaling *)
  fractalFactor = lam * (radius/rF)^(-delta);

  dF = geometricFactor * baseCorrection * radiusScaling * (1 + fractalFactor);
  fT = f0 + dF;

  {
    "R [m]" -> radius,
    "F_Casimir [N]" -> f0,
    "ΔF_BPR [N]" -> dF,
    "F_total [N]" -> fT,
    "relative_deviation" -> If[f0 === 0, Indeterminate, dF/Abs[f0]],
    "field_energy" -> energy,
    "coupling_lambda" -> lam,
    "kappa" -> kappa,
    "lMax" -> lMax
  }
];

(* Plain numeric row, avoiding Associations/Lookups (for Mathics compatibility) *)
BPRCasimirForceRow[radius_?NumericQ, opts : OptionsPattern[]] := Module[
  {a = BPRCasimirForce[radius, FilterRules[{opts}, Options[BPRCasimirForce]]]},
  {
    N[BPRGet[a, "R [m]"]],
    N[BPRGet[a, "F_Casimir [N]"]],
    N[BPRGet[a, "ΔF_BPR [N]"]],
    N[BPRGet[a, "F_total [N]"]],
    N[BPRGet[a, "relative_deviation"]],
    N[BPRGet[a, "field_energy"]]
  }
];

Options[BPRCasimirSweep] = Join[
  Options[BPRCasimirForce],
  {"OutputCSV" -> None}
];

BPRCasimirSweep[rMin_?NumericQ, rMax_?NumericQ, n_Integer?Positive, opts : OptionsPattern[]] := Module[
  {radii, results, out = OptionValue["OutputCSV"]},
  radii = $BPRLogSpace[rMin, rMax, n];
  results = BPRCasimirForce[#, FilterRules[{opts}, Options[BPRCasimirForce]]] & /@ radii;

  If[StringQ[out],
    BPRAssociationToCSV[
      results,
      out,
      "ColumnOrder" -> {"R [m]", "F_Casimir [N]", "ΔF_BPR [N]", "F_total [N]", "relative_deviation", "field_energy", "coupling_lambda", "kappa", "lMax"}
    ];
  ];

  results
];

Options[BPRCasimirSweepRows] = Join[
  Options[BPRCasimirForce],
  {"OutputCSV" -> None}
];

BPRCasimirSweepRows[rMin_?NumericQ, rMax_?NumericQ, n_Integer?Positive, opts : OptionsPattern[]] := Module[
  {
    radii, rows, out = OptionValue["OutputCSV"], header,
    geometry, lam, kappa, lMax, nθ, nϕ, delta, rF, src,
    solution, energy, geometricFactor
  },

  (* Pull options once *)
  geometry = OptionValue["Geometry"];
  lam = OptionValue["CouplingLambda"];
  kappa = OptionValue["Kappa"];
  lMax = OptionValue["LMax"];
  nθ = OptionValue["QuadraturePointsTheta"];
  nϕ = OptionValue["QuadraturePointsPhi"];
  delta = OptionValue["DeltaBPR"];
  rF = OptionValue["ReferenceScale"];
  src = OptionValue["BoundarySource"];

  If[src === Automatic,
    src = Function[{θ, ϕ}, Re[SphericalHarmonicY[2, 0, θ, ϕ]]];
  ];

  (* Compute boundary energy ONCE; this is the expensive step in lightweight runtimes. *)
  solution = BPRSolvePhaseSphereSpectral[
    src,
    kappa,
    lMax,
    "Radius" -> 1.0,
    Sequence @@ DeleteCases[
      {
        If[nθ === Automatic, Nothing, "QuadraturePointsTheta" -> nθ],
        If[nϕ === Automatic, Nothing, "QuadraturePointsPhi" -> nϕ]
      },
      Nothing
    ]
  ];
  energy = BPRPhaseEnergySphereSpectral[solution];

  geometricFactor = Which[
    geometry === "parallel_plates", 1.0,
    geometry === "sphere", 4 Pi,
    geometry === "cylinder", 2 Pi,
    True, 1.0
  ];

  radii = $BPRLogSpace[rMin, rMax, n];
  rows = Table[
    Module[{r = radii[[i]], f0, baseCorrection, radiusScaling, fractalFactor, dF, fT, rel},
      f0 = BPRStandardCasimirForce[r, "Geometry" -> geometry];
      baseCorrection = lam * energy / r^2;
      radiusScaling = (r/10^-6)^(-1);
      fractalFactor = lam * (r/rF)^(-delta);
      dF = geometricFactor * baseCorrection * radiusScaling * (1 + fractalFactor);
      fT = f0 + dF;
      rel = If[f0 === 0, Indeterminate, dF/Abs[f0]];
      {N[r], N[f0], N[dF], N[fT], N[rel], N[energy]}
    ],
    {i, 1, Length[radii]}
  ];

  header = {"R [m]", "F_Casimir [N]", "ΔF_BPR [N]", "F_total [N]", "relative_deviation", "field_energy"};
  If[StringQ[out],
    Export[out, Prepend[rows, header], "CSV"];
  ];
  rows
];

End[];
EndPackage[];


