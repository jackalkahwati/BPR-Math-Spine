(* ::Package:: *)

BeginPackage["BPR`"];

(* E8 utilities (WL-native via RootSystemData) *)

BPRE8Roots::usage = "BPRE8Roots[] returns the list of E8 roots (as vectors).";
BPRE8CartanMatrix::usage = "BPRE8CartanMatrix[] returns the E8 Cartan matrix.";
BPRE8SimpleRoots::usage = "BPRE8SimpleRoots[] returns a choice of simple roots for E8.";

Begin["`Private`"];

BPRE8Roots[] := RootSystemData["E8", "RootVectors"];
BPRE8CartanMatrix[] := RootSystemData["E8", "CartanMatrix"];
BPRE8SimpleRoots[] := RootSystemData["E8", "SimpleRootVectors"];

End[];
EndPackage[];




