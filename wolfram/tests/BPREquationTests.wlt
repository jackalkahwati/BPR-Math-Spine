(* Run:
   wolframscript -code "TestReport[\"wolfram/tests/BPREquationTests.wlt\"]"
*)

Get[FileNameJoin[{DirectoryName[$TestFileName], "..", "BPR.wl"}]];

(* Keep equation tests fast and license-safe:
   - Avoid PDE solves in this file (solver correctness is already covered by BPRTests.wlt).
   - Use an analytic single-mode example: f = Y_{2,0}, κ=1, R=1.
     Then φ_{2,0} = -(1/(l(l+1))) f_{2,0} = -1/6, and energy = l(l+1)|φ|^2 = 1/6.
*)
energyY20 = N[1/6];
epsSOA = BPR`BPRExperimentalBounds["StateOfArt2024"]["bound"];
d0SOA = BPR`BPRExperimentalBounds["StateOfArt2024"]["separation"];
lambdaEffSOA = BPR`BPRPhenomenologicalCouplingLambdaFromName["StateOfArt2024",
  "EnergyOverride" -> energyY20
];

(* Note: this file is a standard .wlt: it contains VerificationTest expressions.
   Run it via:
     wolframscript -code 'TestReport["wolfram/tests/BPREquationTests.wlt"]'
*)

(* Eq (6a): κ Δ φ = f on S^2 (spectral coefficient verification) *)
VerificationTest[
  Module[{sol, metrics},
    sol = {
      "type" -> "SphereSpectral",
      "radius" -> 1.0,
      "kappa" -> 1.0,
      "lMax" -> 2,
      "fCoefficients" -> {{2, 0} -> 1.0},
      "phiCoefficients" -> {{2, 0} -> (-1.0/6.0)}
    };
    metrics = BPR`BPRVerifyPhaseEquationSphereSpectral[sol];
    metrics["max_abs_residual"] < 10^-12
  ],
  True,
  TestID -> "Eq6a-PhaseEquationResidualSmall"
]

(* Eq (3)/(6b): metric perturbation structure exists and is symmetric on selected entries *)
VerificationTest[
  Module[{t, x, y, z, coords, phi, dg, m},
    coords = {t, x, y, z};
    phi = Sin[Pi x] Cos[Pi y] Exp[-(x^2 + y^2 + z^2)];
    dg = BPR`BPRMetricPerturbation[phi, 0.01, coords];
    m = BPR`BPRGet[dg, "delta_g"];
    (* check symmetry and nontriviality at a numeric sample point *)
    Module[{mn},
      mn = Quiet @ N[m /. {t -> 0.0, x -> 0.1, y -> 0.2, z -> 0.3}];
      MatrixQ[mn] &&
        Dimensions[mn] === {4, 4} &&
        Abs[mn[[1, 2]] - mn[[2, 1]]] < 10^-12 &&
        Abs[mn[[2, 3]] - mn[[3, 2]]] < 10^-12 &&
        Abs[mn[[2, 2]]] > 0
    ]
  ],
  True,
  TestID -> "Eq3-6b-MetricPerturbationBasic"
]

(* Eq (4): information action is numeric and negative for xi>0 *)
VerificationTest[
  Module[{vals, sInfo},
    vals = Table[Sin[i/7.], {i, 1, 200}];
    sInfo = BPR`BPRInformationAction[vals, "Xi" -> 10^-3, "Partitions" -> 8, "Bins" -> 10, "Area" -> 4 Pi];
    NumericQ[sInfo] && sInfo <= 0
  ],
  True,
  TestID -> "Eq4-InformationActionNumeric"
]

(* Eq (5): six-factor consciousness coupling is numeric and nonnegative *)
VerificationTest[
  Module[{vals, chi},
    vals = Table[Cos[i/11.], {i, 1, 400}];
    chi = BPR`BPRConsciousnessCoupling[vals, "Partitions" -> 8, "Bins" -> 10];
    NumericQ[chi] && chi >= 0
  ],
  True,
  TestID -> "Eq5-ConsciousnessCouplingNumeric"
]

(* Eq (7): calibrated mode with λ_theory should be extremely small relative deviation *)
VerificationTest[
  Module[{d, f0, res},
    d = 10^-6;
    f0 = BPR`BPRStandardCasimirForce[d, "Geometry" -> "parallel_plates"];
    res = BPR`BPRCasimirForce[d,
      "CorrectionModel" -> "calibrated",
      "CouplingLambda" -> Automatic,
      "FieldEnergyOverride" -> energyY20,
      "LMax" -> 2
    ];
    Abs[BPR`BPRGet[res, "relative_deviation"]] < 10^-20 && Abs[BPR`BPRGet[res, "F_total [N]"] - f0] < 10^-30
  ],
  True,
  TestID -> "Eq7-Calibrated-TheoryCouplingTiny"
]

(* Eq (7): phenomenological λ_eff enforces |ΔF/F| <= bound at reference point (StateOfArt2024 = 1e-3 at 100nm) *)
VerificationTest[
  Module[{d0, eps, res, rel},
    eps = epsSOA;
    d0 = d0SOA;
    res = BPR`BPRCasimirForce[d0,
      "CorrectionModel" -> "calibrated",
      "CouplingLambda" -> lambdaEffSOA,
      "FieldEnergyOverride" -> energyY20,
      "LMax" -> 2
    ];
    rel = Abs[BPR`BPRGet[res, "relative_deviation"]];
    rel <= 1.05 * eps
  ],
  True,
  TestID -> "Eq7-Phenomenological-BoundSatisfied"
]


