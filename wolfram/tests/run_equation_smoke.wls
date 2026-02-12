(* Run:
   wolframscript -script wolfram/tests/run_equation_smoke.wls
*)

Get[FileNameJoin[{DirectoryName[$InputFileName], "..", "BPR.wl"}]];

pass = 0;
fail = 0;

assert[name_, expr_] := Module[{ok},
  ok = Quiet @ Check[TrueQ[expr], False];
  If[ok,
    pass++; Print["PASS: ", name],
    fail++; Print["FAIL: ", name, "  (expr returned: ", ToString[expr, InputForm], ")"]
  ];
];

(* Eq (6a) analytic coefficient check *)
assert["Eq6a (analytic coefficient residual)",
  Module[{sol, m},
    sol = {
      "type" -> "SphereSpectral",
      "radius" -> 1.0,
      "kappa" -> 1.0,
      "lMax" -> 2,
      "fCoefficients" -> {{2, 0} -> 1.0},
      "phiCoefficients" -> {{2, 0} -> (-1.0/6.0)}
    };
    m = BPR`BPRVerifyPhaseEquationSphereSpectral[sol];
    m["max_abs_residual"] < 10^-12
  ]
];

(* Eq (3)/(6b) metric perturbation basic numeric sanity *)
assert["Eq3/Eq6b (metric perturbation symmetry at sample point)",
  Module[{t, x, y, z, coords, phi, dg, M, subs, Mn},
    coords = {t, x, y, z};
    phi = Sin[Pi x] Cos[Pi y] Exp[-(x^2 + y^2 + z^2)];
    dg = BPR`BPRMetricPerturbation[phi, 0.01, coords, "CoordinateSystem" -> "cartesian", "Simplify" -> False];
    M = dg["delta_g"];
    subs = {x -> 0.1, y -> 0.2, z -> 0.3, t -> 0.0};
    Mn = N[M /. subs];
    MatrixQ[Mn] && Abs[Mn[[1, 2]] - Mn[[2, 1]]] < 10^-12 && Abs[Mn[[2, 3]] - Mn[[3, 2]]] < 10^-12
  ]
];

(* Eq (4) information action sign/numeric *)
assert["Eq4 (information action numeric)",
  Module[{vals, sInfo},
    vals = Table[Sin[i/7.], {i, 1, 200}];
    sInfo = BPR`BPRInformationAction[vals, "Xi" -> 10^-3, "Partitions" -> 8, "Bins" -> 10, "Area" -> 4 Pi];
    NumericQ[sInfo] && sInfo <= 0
  ]
];

(* Eq (5) consciousness coupling numeric *)
assert["Eq5 (consciousness coupling numeric)",
  Module[{vals, chi},
    vals = Table[Cos[i/11.], {i, 1, 400}];
    chi = BPR`BPRConsciousnessCoupling[vals, "Partitions" -> 8, "Bins" -> 10];
    NumericQ[chi] && chi >= 0
  ]
];

(* Eq (7) calibrated + theory coupling is tiny *)
assert["Eq7 (calibrated, λ_theory gives tiny ΔF/F)",
  Module[{d, res},
    d = 10^-6;
    res = BPR`BPRCasimirForce[d,
      "CorrectionModel" -> "calibrated",
      "CouplingLambda" -> Automatic,
      "FieldEnergyOverride" -> N[1/6],
      "LMax" -> 2
    ];
    Abs[BPR`BPRGet[res, "relative_deviation"]] < 10^-20
  ]
];

(* Eq (7) phenomenological bound satisfied at reference point *)
assert["Eq7 (phenomenological StateOfArt2024 bound satisfied)",
  Module[{eps, d0, lamEff, res, rel},
    eps = BPR`BPRExperimentalBounds["StateOfArt2024"]["bound"];
    d0 = BPR`BPRExperimentalBounds["StateOfArt2024"]["separation"];
    lamEff = BPR`BPRPhenomenologicalCouplingLambdaFromName["StateOfArt2024", "EnergyOverride" -> N[1/6]];
    res = BPR`BPRCasimirForce[d0,
      "CorrectionModel" -> "calibrated",
      "CouplingLambda" -> lamEff,
      "FieldEnergyOverride" -> N[1/6],
      "LMax" -> 2
    ];
    rel = Abs[BPR`BPRGet[res, "relative_deviation"]];
    rel <= 1.05 * eps
  ]
];

(* Adjacent theories: P11.15, P18.2, P19.8, P19.9, P4.7, P12.14 *)
(* P11.15: W_c = Sqrt[3] from first principles (kappa = z/2, W_c = Sqrt[kappa]) *)
assert["P11.15 (DM relic Omega h^2 in range)",
  Module[{omega}, omega = BPR`BPRDMRelicAbundance[Sqrt[3], 104729];
    omega > 0.10 && omega < 0.14]
];

assert["P18.2 (charged lepton masses)",
  Module[{m}, m = BPR`BPRChargedLeptonMasses[];
    m["mu"] > 105 && m["mu"] < 110]
];

assert["P19.8 (B/A He4)",
  Module[{b}, b = BPR`BPRBindingEnergyPerNucleon[4, 2];
    b > 6.9 && b < 7.2]
];

assert["P19.9 (saturation density)",
  Module[{r}, r = BPR`BPRNuclearSaturationDensity[1.25];
    r > 0.15 && r < 0.17]
];

assert["P4.7 (Tc Nb)",
  Module[{tc}, tc = BPR`BPRSuperconductorTc[0.325, 275.0];
    tc > 9.0 && tc < 9.5]
];

assert["P12.14 (pion mass)",
  Module[{mpi}, mpi = BPR`BPRPionMass[2.16, 4.67, 92.1];
    mpi > 120 && mpi < 130]
];

Print["---"];
Print["Equation smoke tests: ", pass, " passed, ", fail, " failed."];
If[fail > 0, Exit[1], Exit[0]];


