(* ::Package:: *)

BeginPackage["BPR`"];

(* Boundary Laplacian / phase solver on S^2 (WL-native, spectral) *)

BPRLaplacianEigenvaluesSphere::usage =
  "BPRLaplacianEigenvaluesSphere[lMax] returns {{l, l(l+1)}, ...} for l=0..lMax (Checkpoint 1 baseline).";

BPRSphericalHarmonicCoefficients::usage =
  "BPRSphericalHarmonicCoefficients[f, lMax] numerically estimates spherical harmonic coefficients a_{l,m} for a scalar function on S^2.";

BPRSolvePhaseSphereSpectral::usage =
  "BPRSolvePhaseSphereSpectral[f, kappa, lMax] solves κ Δ_{S^2} φ = f on the unit sphere via spherical harmonics, returning an Association with coefficients and evaluators.";

BPRPhaseEnergySphereSpectral::usage =
  "BPRPhaseEnergySphereSpectral[solution] returns ∫_{S^2} |∇_Σ φ|^2 dS computed from spectral coefficients.";

Begin["`Private`"];

(* Helpers *)

(* WL convention: SphericalHarmonicY[l,m,θ,φ] uses θ=colatitude, φ=longitude. *)
ClearAll[$BPRThetaPhiMeasure];
$BPRThetaPhiMeasure[θ_, ϕ_] := Sin[θ];

(* Mathics compatibility: Subdivide is not always implemented; use explicit tables. *)
ClearAll[$BPRLinearSubdivide];
$BPRLinearSubdivide[a_?NumericQ, b_?NumericQ, n_Integer?NonNegative] := Module[
  {k = n},
  If[k == 0, {a}, Table[a + (b - a) * i/k, {i, 0, k}]]
];

(* Allow f to be provided either as f[{θ,ϕ}] or f[x,y,z]. *)
ClearAll[$BPRToThetaPhiFunction];
$BPRToThetaPhiFunction[f_] := Module[{},
  Which[
    Head[f] === Function,
    Module[{vars, n},
      (* In Wolfram Language, Function[vars, body] stores vars at Part 1. *)
      vars = Quiet @ Check[f[[1]], Null];
      n = Which[
        ListQ[vars], Length[vars],
        vars === Null, 0,
        True, 1
      ];

      Which[
        (* Function[{θ,ϕ}, ...] *)
        n == 2,
        Function[{θ, ϕ}, Evaluate @ f[θ, ϕ]],

        (* Function[{x,y,z}, ...] expects cartesian *)
        n == 3,
        Function[{θ, ϕ},
          With[
            {x = Sin[θ] Cos[ϕ], y = Sin[θ] Sin[ϕ], z = Cos[θ]},
            Evaluate @ f[x, y, z]
          ]
        ],

        (* Function[{u}, ...] (or Function[u, ...]) assume u is {θ,ϕ} *)
        True,
        Function[{θ, ϕ}, Evaluate @ f[{θ, ϕ}]]
      ]
    ],

    True,
    (* if already an expression in θ,ϕ, treat as pure function *)
    Function[{θ, ϕ}, Evaluate @ f]
  ]
];

BPRLaplacianEigenvaluesSphere[lMax_Integer?NonNegative] := Table[{l, l (l + 1)}, {l, 0, lMax}];

Options[BPRSphericalHarmonicCoefficients] = {
  (* Defaults chosen to be fast enough even in lightweight WL runtimes. Increase for higher accuracy. *)
  "QuadraturePointsTheta" -> 24,
  "QuadraturePointsPhi" -> 48,
  "Assumptions" -> True
};

(* Estimate coefficients a_{l,m} for f on unit sphere:
   a_{l,m} = ∫ f(θ,ϕ) Conjugate[Y_{l,m}(θ,ϕ)] dΩ
   with dΩ = sinθ dθ dϕ
*)
BPRSphericalHarmonicCoefficients[f_, lMax_Integer?NonNegative, opts : OptionsPattern[]] := Module[
  {
    ff = $BPRToThetaPhiFunction[f],
    nθ = OptionValue["QuadraturePointsTheta"],
    nϕ = OptionValue["QuadraturePointsPhi"],
    θs, ϕs, wθ, wϕ,
    coeffs
  },

  (* simple tensor-product quadrature (trapezoidal in ϕ, midpoint-ish in θ) *)
  θs = Table[(i + 1/2) * Pi/nθ, {i, 0, nθ - 1}];
  ϕs = Table[(j + 1/2) * 2 Pi/nϕ, {j, 0, nϕ - 1}];
  wθ = N[Pi/nθ];
  wϕ = N[2 Pi/nϕ];

  coeffs = Flatten[
    Table[
      With[
        {
          integrandSum =
            Total[
              Table[
                ff[θ, ϕ] * Conjugate[SphericalHarmonicY[l, m, θ, ϕ]] * $BPRThetaPhiMeasure[θ, ϕ],
                {θ, θs}, {ϕ, ϕs}
              ],
              2
            ]
        },
        Rule[{l, m}, N[integrandSum * wθ * wϕ]]
      ],
      {l, 0, lMax}, {m, -l, l}
    ]
  ];

  coeffs
];

Options[BPRSolvePhaseSphereSpectral] = Join[
  Options[BPRSphericalHarmonicCoefficients],
  {
    "Radius" -> 1.0
  }
];

(* Solve κ Δ φ = f on S^2:
   Δ Y_{l,m} = -l(l+1)/R^2 Y_{l,m}
   => κ (-l(l+1)/R^2) φ_{l,m} = f_{l,m}
   => φ_{l,m} = -(R^2/(κ l(l+1))) f_{l,m} for l>0
   l=0 is the nullspace; we set φ_{0,0}=0 (zero-mean gauge).
*)
BPRSolvePhaseSphereSpectral[f_, kappa_?NumericQ, lMax_Integer?Positive, opts : OptionsPattern[]] := Module[
  {R = OptionValue["Radius"], fCoeff, phiCoeff, phiFn, fFn},

  fCoeff = BPRSphericalHarmonicCoefficients[f, lMax, FilterRules[{opts}, Options[BPRSphericalHarmonicCoefficients]]];

  (* Avoid KeyValueMap for compatibility with lightweight WL runtimes (e.g. Mathics). *)
  phiCoeff = Map[
    Function[{rule},
      With[
        {lm = First[rule], val = Last[rule], l = First[First[rule]]},
        If[l == 0,
          lm -> 0.0,
          lm -> (-(R^2/(kappa * l (l + 1)))) * val
        ]
      ]
    ],
    fCoeff
  ];

  phiFn = Function[{θ, ϕ},
    Total[
      Map[
        Function[{rule},
          With[{lm = First[rule], a = Last[rule]},
            a * SphericalHarmonicY[lm[[1]], lm[[2]], θ, ϕ]
          ]
        ],
        phiCoeff
      ]
    ]
  ];

  fFn = $BPRToThetaPhiFunction[f];

  {
    "type" -> "SphereSpectral",
    "radius" -> R,
    "kappa" -> kappa,
    "lMax" -> lMax,
    "fCoefficients" -> fCoeff,
    "phiCoefficients" -> phiCoeff,
    "phiThetaPhi" -> phiFn,
    "fThetaPhi" -> fFn,
    "gauge" -> "ZeroMean"
  }
];

(* Energy on S^2:
   ∫ |∇φ|^2 dΩ = Σ_{l,m} l(l+1)/R^2 |φ_{l,m}|^2  (orthonormal Y_{l,m})
*)
BPRPhaseEnergySphereSpectral[solution_] := Module[
  {R, coeff, energy},
  If[BPRGet[solution, "type"] =!= "SphereSpectral", Return[$Failed]];
  R = BPRGet[solution, "radius"];
  coeff = BPRGet[solution, "phiCoefficients"];
  (* Avoid KeyValueMap for compatibility with lightweight WL runtimes (e.g. Mathics). *)
  energy = Total @ Map[
    Function[{rule},
      With[{lm = First[rule], a = Last[rule], l = First[First[rule]]},
        If[l == 0, 0.0, (l (l + 1)/R^2) * Abs[a]^2]
      ]
    ],
    coeff
  ];
  N[energy]
];

End[];
EndPackage[];


