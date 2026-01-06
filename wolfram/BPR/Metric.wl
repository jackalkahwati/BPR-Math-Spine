(* ::Package:: *)

BeginPackage["BPR`"];

(* Metric perturbation + stress tensor (WL-native, symbolic-first) *)

BPRMetricPerturbation::usage =
  "BPRMetricPerturbation[phiExpr, couplingLambda, coords, coordSystem] returns an Association with Δg_{μν} as a symbolic 4x4 matrix (Eq 3 / 6b scaffold).";

BPRStressTensorScalarField::usage =
  "BPRStressTensorScalarField[phiExpr, coords] returns a simplified scalar-field stress tensor T_{μν} (static, flat background).";

BPRVerifyConservation::usage =
  "BPRVerifyConservation[T, coords] returns the (simplified) divergence ∂_μ T^{μ}{}_{ν} as a 4-vector (Checkpoint 2 scaffold).";

Begin["`Private`"];

Options[BPRMetricPerturbation] = {
  "CoordinateSystem" -> "cartesian",
  (* Simplify can be expensive; keep off by default for fast tests/CLI usage *)
  "Simplify" -> False
};

(* This mirrors the current repo's intent: a boundary-localized coupling factor.
   For a full GR-consistent implementation, we'd compute Christoffel symbols etc.
*)
BPRMetricPerturbation[phiExpr_, couplingLambda_?NumericQ, coords_List, opts : OptionsPattern[]] := Module[
  {coordSystem = OptionValue["CoordinateSystem"], doSimplify = OptionValue["Simplify"], t, x, y, z, deltaG, boundaryFactor},

  If[Length[coords] =!= 4, Return[$Failed]];
  {t, x, y, z} = coords;

  deltaG = ConstantArray[0, {4, 4}];

  Which[
    coordSystem === "cartesian",
    boundaryFactor = Exp[-(((x^2 + y^2 + z^2) - 1)^2)];

    ,
    coordSystem === "spherical",
    (* coords expected as {t,r,θ,ϕ} *)
    boundaryFactor = Exp[-((coords[[2]] - 1)^2)];

    ,
    True,
    boundaryFactor = 1
  ];

  (* spatial diagonal terms *)
  deltaG[[2, 2]] = couplingLambda * phiExpr * boundaryFactor;
  deltaG[[3, 3]] = couplingLambda * phiExpr * boundaryFactor;
  deltaG[[4, 4]] = couplingLambda * phiExpr * boundaryFactor;

  (* time-space mixing sample term (mirrors python) *)
  deltaG[[1, 2]] = couplingLambda * phiExpr * D[phiExpr, x] * boundaryFactor;
  deltaG[[2, 1]] = deltaG[[1, 2]];

  (* cross term *)
  deltaG[[2, 3]] = couplingLambda * D[phiExpr, x] * D[phiExpr, y];
  deltaG[[3, 2]] = deltaG[[2, 3]];

  <|
    "coords" -> coords,
    "coupling_lambda" -> couplingLambda,
    "delta_g" -> If[TrueQ[doSimplify], Simplify[deltaG], deltaG],
    "coordinateSystem" -> coordSystem
  |>
];

(* Simplified scalar field stress tensor on flat background, static case:
   T_00 = 1/2 |∇phi|^2 ; T_ij = 1/2 δ_ij |∇phi|^2 (isotropic, repo-style simplification)
*)
BPRStressTensorScalarField[phiExpr_, coords_List] := Module[
  {t, x, y, z, grad2, T},
  If[Length[coords] =!= 4, Return[$Failed]];
  {t, x, y, z} = coords;
  grad2 = D[phiExpr, x]^2 + D[phiExpr, y]^2 + D[phiExpr, z]^2;

  T = ConstantArray[0, {4, 4}];
  T[[1, 1]] = (1/2) grad2;
  Do[
    T[[i, i]] = (1/2) T[[1, 1]],
    {i, 2, 4}
  ];

  Simplify[T]
];

(* Simplified conservation check: ∂_μ T_{μν} (no connection terms) *)
BPRVerifyConservation[T_?MatrixQ, coords_List] := Module[
  {div},
  If[Length[coords] =!= 4, Return[$Failed]];
  div = Table[
    Sum[D[T[[μ, ν]], coords[[μ]]], {μ, 1, 4}],
    {ν, 1, 4}
  ];
  Simplify[div]
];

End[];
EndPackage[];




