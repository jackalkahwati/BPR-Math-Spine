(* ::Package:: *)

BeginPackage["BPR`"];

(* Shared constants and utilities *)

BPRConstants::usage = "BPRConstants[] returns an Association of physical constants used in the WL-native spine.";
BPRAssociationToCSV::usage = "BPRAssociationToCSV[list, file] exports a list of Associations to CSV with stable column order.";
BPRGet::usage = "BPRGet[obj, key] gets a value from a spine object (Association or rules list) in a runtime-compatible way.";
BPRPhysicalCouplingLambda::usage = "BPRPhysicalCouplingLambda[kappaBoundary] returns λ = kappaBoundary * ℓ_P^2 (Planck-length-squared coupling).";

Begin["`Private`"];

(* Numeric constants for maximal portability (works in Wolfram Engine without Quantity infrastructure). *)
$HBar = 1.054571817*10^-34; (* J*s *)
$C = 299792458; (* m/s *)
$PlanckLength = 1.616255*10^-35; (* m *)

BPRConstants[] := <|
  "hbar" -> $HBar,
  "c" -> $C,
  "planck_length" -> $PlanckLength,
  "pi" -> Pi,
  "CASIMIR_PREFACTOR" -> (Pi^2 * $HBar * $C / 240)
|>;

BPRPhysicalCouplingLambda[kappaBoundary_: 1.0] := N[kappaBoundary * $PlanckLength^2];

(* Runtime-friendly getter:
   - In full Wolfram Language, Associations are supported.
   - In lightweight interpreters (e.g. Mathics), Associations may exist but Lookup/part access may be incomplete.
   - Using rules replacement is the most portable.
*)
BPRGet[obj_, key_] := Module[{rules},
  rules = Which[
    Head[obj] === Association, Quiet @ Check[Normal[obj], obj],
    ListQ[obj], obj,
    True, {}
  ];
  key /. rules /. key -> Missing["KeyAbsent", key]
];

Options[BPRAssociationToCSV] = {
  "ColumnOrder" -> {"R [m]", "F_Casimir [N]", "ΔF_BPR [N]", "F_total [N]", "relative_deviation"}
};

BPRAssociationToCSV[list_List, file_String, opts : OptionsPattern[]] := Module[
  {cols = OptionValue["ColumnOrder"], rows},
  rows = Prepend[(BPRGet[#, col] & /@ cols) & /@ list, cols];
  Export[file, rows, "CSV"]
];

End[];
EndPackage[];


