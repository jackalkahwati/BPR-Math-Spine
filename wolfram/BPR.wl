(* ::Package:: *)
(* Single entrypoint that loads WL-native spine modules. *)

BeginPackage["BPR`"];

(* Public API is declared in the module files. *)

Begin["`Private`"];

$BPRRootDir = DirectoryName[$InputFileName];
Get[FileNameJoin[{$BPRRootDir, "BPR", "Core.wl"}]];
Get[FileNameJoin[{$BPRRootDir, "BPR", "Geometry.wl"}]];
Get[FileNameJoin[{$BPRRootDir, "BPR", "BoundaryField.wl"}]];
Get[FileNameJoin[{$BPRRootDir, "BPR", "Metric.wl"}]];
Get[FileNameJoin[{$BPRRootDir, "BPR", "Casimir.wl"}]];
Get[FileNameJoin[{$BPRRootDir, "BPR", "Information.wl"}]];
Get[FileNameJoin[{$BPRRootDir, "BPR", "E8.wl"}]];
Get[FileNameJoin[{$BPRRootDir, "BPR", "RPST.wl"}]];

End[];
EndPackage[];

