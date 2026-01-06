(* ::Package:: *)

BeginPackage["BPR`"];

BPRMakeBoundary::usage =
  "BPRMakeBoundary[meshSize, geometry, radius] returns a discretized boundary region/mesh representation for WL-native computations.";

Begin["`Private`"];

Options[BPRMakeBoundary] = {
  "Geometry" -> "sphere",
  "Radius" -> 1.0
};

(* For now, we prioritize S^2 (the repo's default and checkpoints). We return both the Region and a metadata Association. *)
BPRMakeBoundary[meshSize_?NumericQ, opts : OptionsPattern[]] := Module[
  {geom = OptionValue["Geometry"], r = OptionValue["Radius"], region, mesh},
  region = Which[
    geom === "sphere", Sphere[{0, 0, 0}, r],
    geom === "cylinder", Cylinder[{{0, 0, -r}, {0, 0, r}}, r],
    True, (Message[BPRMakeBoundary::badgeom, geom]; $Failed)
  ];

  (* Discretize boundary as surface mesh; if not available, fall back to the symbolic region. *)
  mesh = Quiet @ Check[
    BoundaryDiscretizeRegion[region, MaxCellMeasure -> meshSize^2],
    region
  ];

  <|
    "geometry" -> geom,
    "radius" -> r,
    "meshSize" -> meshSize,
    "region" -> region,
    "mesh" -> mesh
  |>
];

BPRMakeBoundary::badgeom = "Unknown geometry: `1`.";

End[];
EndPackage[];




