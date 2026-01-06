(* ::Package:: *)

BeginPackage["BPR`"];

BPRStandardCasimirForce::usage =
  "BPRStandardCasimirForce[radius, opts] returns the standard Casimir force for a simple geometry model.";

BPRCasimirForce::usage =
  "BPRCasimirForce[radius, opts] returns an Association with standard force, BPR correction, and totals (Eq 7 artifact).";

BPRCasimirForceRow::usage =
  "BPRCasimirForceRow[radius, opts] returns a plain numeric row {R, F_Casimir, ΔF_BPR, F_total, relative_deviation, field_energy}. Useful for lightweight WL runtimes (e.g. Mathics).";

BPRPhenomenologicalCouplingLambda::usage =
  "BPRPhenomenologicalCouplingLambda[experimentalBound, referenceSeparation, opts] returns a coupling λ_eff such that |ΔF/F|<=experimentalBound at referenceSeparation, using the current (calibrated) BPR correction model.";

BPRExperimentalBounds::usage =
  "BPRExperimentalBounds is an Association of named experimental |ΔF/F| bounds and reference separations for Casimir constraints.";

BPRPhenomenologicalCouplingLambdaFromName::usage =
  "BPRPhenomenologicalCouplingLambdaFromName[name, opts] computes λ_eff using the bound/separation stored in BPRExperimentalBounds[name].";

BPRCasimirSweepRows::usage =
  "BPRCasimirSweepRows[rMin, rMax, n, opts] returns a list of numeric rows (and can export CSV).";

BPRCasimirSweep::usage =
  "BPRCasimirSweep[rMin, rMax, n, opts] sweeps radii and optionally exports a CSV matching Python columns.";

Begin["`Private`"];

Needs["BPR`"]; (* already loaded when used via BPR.wl; harmless otherwise *)

(* Default experimental bounds table (user-supplied summary; conservative defaults). *)
BPRExperimentalBounds = <|
  "Lamoreaux1997" -> <|"bound" -> 5*10^-2, "separation" -> 600*10^-9|>,
  "Mohideen1998" -> <|"bound" -> 10^-2, "separation" -> 100*10^-9|>,
  "Decca2003" -> <|"bound" -> 5*10^-3, "separation" -> 160*10^-9|>,
  "Decca2007" -> <|"bound" -> 2*10^-3, "separation" -> 200*10^-9|>,
  "StateOfArt2024" -> <|"bound" -> 10^-3, "separation" -> 100*10^-9|>
|>;

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
  (* If Automatic, use λ = κ_boundary * ℓ_P^2 (physically tiny) *)
  "CouplingLambda" -> Automatic,
  "KappaBoundary" -> 1.0,
  (* Boundary PDE stiffness κ in κ Δ φ = f *)
  "Kappa" -> 1.0,
  "LMax" -> 8,
  "QuadraturePointsTheta" -> Automatic,
  "QuadraturePointsPhi" -> Automatic,
  "DeltaBPR" -> 1.37,
  "ReferenceScale" -> 10^-6,
  "AlphaBPR" -> 0.1,
  (* Select correction model:
     - \"legacy\": old demo scaling (can be huge)
     - \"calibrated\": λ=κℓ_P^2 normalization + fractal enhancement (sub-dominant)
  *)
  "CorrectionModel" -> "calibrated",
  (* If numeric, skip PDE solve and use provided boundary energy directly *)
  "FieldEnergyOverride" -> Automatic,
  "BoundarySource" -> Automatic
};

BPRPhenomenologicalCouplingLambda::badmodel =
  "Phenomenological λ_eff is only implemented for CorrectionModel -> \"calibrated\" (linear in λ). Got: `1`.";

BPRPhenomenologicalCouplingLambdaFromName::unk =
  "Unknown experimental bound name: `1`. Known names: `2`.";

Options[BPRPhenomenologicalCouplingLambda] = {
  "Geometry" -> "parallel_plates",
  "Kappa" -> 1.0,
  "LMax" -> 8,
  "QuadraturePointsTheta" -> Automatic,
  "QuadraturePointsPhi" -> Automatic,
  "DeltaBPR" -> 1.37,
  "ReferenceScale" -> 10^-6,
  "AlphaBPR" -> 0.1,
  "BoundarySource" -> Automatic,
  "CorrectionModel" -> "calibrated",
  (* If numeric, skip PDE solve and use provided boundary energy directly *)
  "EnergyOverride" -> Automatic
};

(* Compute an experimental upper-bound on λ (units depend on model; here it matches the calibrated linear-in-λ usage).
   Given:
     |ΔF/F| <= eps at separation d0
   and (calibrated model):
     ΔF(d0) = λ * E * (1 + α (d0/Rf)^(-δ))
   => λ_eff <= eps * |F_Casimir(d0)| / (E * (1 + α (d0/Rf)^(-δ)))
*)
BPRPhenomenologicalCouplingLambda[experimentalBound_: 10^-3, referenceSeparation_: 100*10^-9, opts : OptionsPattern[]] := Module[
  {
    eps = experimentalBound,
    d0 = referenceSeparation,
    geometry = OptionValue["Geometry"],
    kappa = OptionValue["Kappa"],
    lMax = OptionValue["LMax"],
    nθ = OptionValue["QuadraturePointsTheta"],
    nϕ = OptionValue["QuadraturePointsPhi"],
    delta = OptionValue["DeltaBPR"],
    rF = OptionValue["ReferenceScale"],
    alphaBpr = OptionValue["AlphaBPR"],
    src = OptionValue["BoundarySource"],
    model = OptionValue["CorrectionModel"],
    energyOverride = OptionValue["EnergyOverride"],
    f0, solution, energy, fractalFactor
  },

  If[model =!= "calibrated",
    Message[BPRPhenomenologicalCouplingLambda::badmodel, model];
    Return[$Failed];
  ];

  If[!NumericQ[eps] || eps <= 0, Return[$Failed]];
  If[!NumericQ[d0] || d0 <= 0, Return[$Failed]];

  If[src === Automatic,
    src = Function[{θ, ϕ}, Re[SphericalHarmonicY[2, 0, θ, ϕ]]];
  ];

  f0 = BPRStandardCasimirForce[d0, "Geometry" -> geometry];

  energy =
    If[NumericQ[energyOverride],
      N[energyOverride],
      Module[{sol},
        sol = BPRSolvePhaseSphereSpectral[
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
        BPRPhaseEnergySphereSpectral[sol]
      ]
    ];

  fractalFactor = 1 + alphaBpr * (d0/rF)^(-delta);

  N[eps * Abs[f0] / (energy * fractalFactor)]
];

BPRPhenomenologicalCouplingLambdaFromName[name_String, opts : OptionsPattern[]] := Module[
  {entry},
  entry = Lookup[BPRExperimentalBounds, name, Missing["NotFound"]];
  If[Head[entry] === Missing,
    Message[BPRPhenomenologicalCouplingLambdaFromName::unk, name, Keys[BPRExperimentalBounds]];
    Return[$Failed];
  ];
  BPRPhenomenologicalCouplingLambda[
    entry["bound"],
    entry["separation"],
    FilterRules[{opts}, Options[BPRPhenomenologicalCouplingLambda]]
  ]
];

BPRCasimirForce[radius_?NumericQ, opts : OptionsPattern[]] := Module[
  {
    geometry = OptionValue["Geometry"],
    lamOpt = OptionValue["CouplingLambda"],
    kappaBoundary = OptionValue["KappaBoundary"],
    model = OptionValue["CorrectionModel"],
    kappa = OptionValue["Kappa"],
    lMax = OptionValue["LMax"],
    nθ = OptionValue["QuadraturePointsTheta"],
    nϕ = OptionValue["QuadraturePointsPhi"],
    delta = OptionValue["DeltaBPR"],
    rF = OptionValue["ReferenceScale"],
    alphaBpr = OptionValue["AlphaBPR"],
    energyOverride = OptionValue["FieldEnergyOverride"],
    src = OptionValue["BoundarySource"],
    lam,
    f0,
    solution,
    energy,
    baseCorrection,
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
  energy =
    If[NumericQ[energyOverride],
      N[energyOverride],
      Module[{sol},
        sol = BPRSolvePhaseSphereSpectral[
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
        BPRPhaseEnergySphereSpectral[sol]
      ]
    ];

  (* Choose coupling *)
  lam = Which[
    lamOpt === Automatic, BPRPhysicalCouplingLambda[kappaBoundary],
    Head[lamOpt] === String && lamOpt === "phenomenological",
    BPRPhenomenologicalCouplingLambda[10^-3, 100*10^-9,
      "Geometry" -> geometry,
      "Kappa" -> kappa,
      "LMax" -> lMax,
      "QuadraturePointsTheta" -> nθ,
      "QuadraturePointsPhi" -> nϕ,
      "DeltaBPR" -> delta,
      "ReferenceScale" -> rF,
      "AlphaBPR" -> alphaBpr,
      "BoundarySource" -> src,
      "CorrectionModel" -> model
    ],
    Head[lamOpt] === String && KeyExistsQ[BPRExperimentalBounds, lamOpt],
    BPRPhenomenologicalCouplingLambdaFromName[lamOpt,
      "Geometry" -> geometry,
      "Kappa" -> kappa,
      "LMax" -> lMax,
      "QuadraturePointsTheta" -> nθ,
      "QuadraturePointsPhi" -> nϕ,
      "DeltaBPR" -> delta,
      "ReferenceScale" -> rF,
      "AlphaBPR" -> alphaBpr,
      "BoundarySource" -> src,
      "CorrectionModel" -> model
    ],
    True, lamOpt
  ];

  (* Correction models *)
  Which[
    model === "legacy",
    (* legacy demo scaling (can be huge; kept for backward comparability) *)
    baseCorrection = lam * energy / radius^2;
    fractalFactor = lam * (radius/rF)^(-delta);
    dF = baseCorrection * (radius/10^-6)^(-1) * (1 + fractalFactor),

    True,
    (* calibrated: matches the intent of physical_coupling_lambda + fractal enhancement in your snippet *)
    baseCorrection = lam * energy; (* separation cancels under the unit-sphere scaling assumption *)
    fractalFactor = 1 + alphaBpr * (radius/rF)^(-delta);
    dF = baseCorrection * fractalFactor
  ];

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
  {
    "OutputCSV" -> None,
    (* Output-only: rename the distance column for CSV clarity without changing internal keys. *)
    "DistanceColumnName" -> "R [m]"
  }
];

BPRCasimirSweepRows[rMin_?NumericQ, rMax_?NumericQ, n_Integer?Positive, opts : OptionsPattern[]] := Module[
  {
    radii, rows, out = OptionValue["OutputCSV"], header,
    dColName = OptionValue["DistanceColumnName"],
    geometry, lamOpt, kappaBoundary, model, alphaBpr,
    lam, kappa, lMax, nθ, nϕ, delta, rF, src,
    solution, energy
  },

  (* Pull options once *)
  geometry = OptionValue["Geometry"];
  lamOpt = OptionValue["CouplingLambda"];
  kappaBoundary = OptionValue["KappaBoundary"];
  model = OptionValue["CorrectionModel"];
  alphaBpr = OptionValue["AlphaBPR"];
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

  lam = If[lamOpt === Automatic, BPRPhysicalCouplingLambda[kappaBoundary], lamOpt];

  radii = $BPRLogSpace[rMin, rMax, n];
  rows = Table[
    Module[{r = radii[[i]], f0, baseCorrection, radiusScaling, fractalFactor, dF, fT, rel},
      f0 = BPRStandardCasimirForce[r, "Geometry" -> geometry];
      Which[
        model === "legacy",
        baseCorrection = lam * energy / r^2;
        radiusScaling = (r/10^-6)^(-1);
        fractalFactor = lam * (r/rF)^(-delta);
        dF = baseCorrection * radiusScaling * (1 + fractalFactor),

        True,
        baseCorrection = lam * energy;
        fractalFactor = 1 + alphaBpr * (r/rF)^(-delta);
        dF = baseCorrection * fractalFactor
      ];
      fT = f0 + dF;
      rel = If[f0 === 0, Indeterminate, dF/Abs[f0]];
      {N[r], N[f0], N[dF], N[fT], N[rel], N[energy]}
    ],
    {i, 1, Length[radii]}
  ];

  header = {dColName, "F_Casimir [N]", "ΔF_BPR [N]", "F_total [N]", "relative_deviation", "field_energy"};
  If[StringQ[out],
    Export[out, Prepend[rows, header], "CSV"];
  ];
  rows
];

End[];
EndPackage[];


