(* ::Package:: *)

BeginPackage["BPR`"];

(* Eq (4) + Eq (5) in WL-native form *)

BPRPartitionFieldValues::usage =
  "BPRPartitionFieldValues[values, nPartitions] partitions a vector of samples into subsystems (IIT-style scaffolding).";

BPRMutualInformationFromHistograms::usage =
  "BPRMutualInformationFromHistograms[x, y, nBins] estimates mutual information I(x;y) from a 2D histogram.";

BPRIntegratedInformationPhi::usage =
  "BPRIntegratedInformationPhi[values, nPartitions] computes an IIT-inspired integrated information Φ as an average pairwise mutual information across partitions.";

BPRInformationAction::usage =
  "BPRInformationAction[values, xi, nPartitions] returns S_info = -xi * Area * Φ (Area defaults to 4π unless provided).";

BPRConsciousnessCoupling::usage =
  "BPRConsciousnessCoupling[values, params] computes the six-factor χ_b coupling (Eq 5) from samples and parameters.";

Begin["`Private`"];

Options[BPRPartitionFieldValues] = {"Partitions" -> 8};
BPRPartitionFieldValues[values_List, opts : OptionsPattern[]] := Module[
  {n = OptionValue["Partitions"], v = values, sz, parts},
  If[Length[v] < 2, Return[{v}]];
  sz = Max[1, Floor[Length[v]/n]];
  parts = Partition[v, sz, sz, 1, {}];
  If[Length[parts] < 2, {v}, parts]
];

Options[BPRMutualInformationFromHistograms] = {"Bins" -> 10, "Epsilon" -> 10^-12};
BPRMutualInformationFromHistograms[x_List, y_List, opts : OptionsPattern[]] := Module[
  {nBins = OptionValue["Bins"], eps = OptionValue["Epsilon"], h2, pxy, px, py, mi},
  If[Length[x] == 0 || Length[y] == 0, Return[0.0]];
  h2 = HistogramList[{x, y}, {nBins, nBins}][[3]];
  If[Total[h2, 2] == 0, Return[0.0]];
  pxy = N[h2/Total[h2, 2]];
  px = Total[pxy, {2}];
  py = Total[pxy, {1}];
  mi = Total[
    MapIndexed[
      Function[{p, ij},
        If[p <= 0, 0,
          With[{i = ij[[1, 1]], j = ij[[1, 2]]},
            p * Log[(p + eps)/((px[[i]] + eps) (py[[j]] + eps))]
          ]
        ]
      ],
      pxy,
      {2}
    ],
    2
  ];
  Max[0.0, N[mi]]
];

Options[BPRIntegratedInformationPhi] = {"Partitions" -> 8, "Bins" -> 10};
BPRIntegratedInformationPhi[values_List, opts : OptionsPattern[]] := Module[
  {parts, n, pairs, miVals},
  If[Length[values] < 2, Return[0.0]];
  parts = BPRPartitionFieldValues[values, "Partitions" -> OptionValue["Partitions"]];
  n = Length[parts];
  If[n < 2, Return[N[Variance[values]]]];
  pairs = Flatten[Table[{i, j}, {i, 1, n}, {j, i + 1, n}], 1];
  miVals = BPRMutualInformationFromHistograms[parts[[#[[1]]]], parts[[#[[2]]]], "Bins" -> OptionValue["Bins"]] & /@ pairs;
  N[Mean[miVals]]
];

Options[BPRInformationAction] = {"Xi" -> 10^-3, "Partitions" -> 8, "Bins" -> 10, "Area" -> 4 Pi};
BPRInformationAction[values_List, opts : OptionsPattern[]] := Module[
  {xi = OptionValue["Xi"], area = OptionValue["Area"], phi},
  phi = BPRIntegratedInformationPhi[values, "Partitions" -> OptionValue["Partitions"], "Bins" -> OptionValue["Bins"]];
  N[-xi * area * phi]
];

(* Six-factor coupling: χ_b = χ_max σ[k(Φ/Φ_c - 1)] E^α (Φ/Φ_c)^β τ S^γ U^δ I^ε *)
Options[BPRConsciousnessCoupling] = {
  "chi_max" -> 10^-3,
  "Phi_c" -> 1.0,
  "alpha" -> 1.2,
  "beta" -> 1.5,
  "gamma" -> 0.8,
  "delta" -> 1.0,
  "epsilon" -> 1.3,
  "k" -> 2.0,
  "tau" -> 1.0,
  "Partitions" -> 8,
  "Bins" -> 10
};

BPRConsciousnessCoupling[values_List, opts : OptionsPattern[]] := Module[
  {
    chiMax = OptionValue["chi_max"],
    phiC = OptionValue["Phi_c"],
    α = OptionValue["alpha"],
    β = OptionValue["beta"],
    γ = OptionValue["gamma"],
    δ = OptionValue["delta"],
    ϵ = OptionValue["epsilon"],
    k = OptionValue["k"],
    τ = OptionValue["tau"],
    vals = values,
    E, S, U, I, Phi, ratio, sigma
  },

  If[Length[vals] < 2, Return[0.0]];

  (* energy proxy *)
  E = Max[10^-10, Mean[vals^2]];

  (* entropy proxy from histogram *)
  S = Module[{h = HistogramList[vals, 20][[2]], p},
    If[Total[h] == 0, 1.0,
      p = N[h/Total[h]];
      Max[0.1, -Total[Select[p, # > 0 &] * Log[Select[p, # > 0 &]]]]
    ]
  ];

  (* utility proxy: smoothness *)
  U = Module[{v = Variance[vals], dv = Variance[Differences[vals]]},
    Max[0.1, 0.6 * (1/(1 + v)) + 0.4 * (1/(1 + dv))]
  ];

  (* information proxy: normalized Φ *)
  Phi = BPRIntegratedInformationPhi[vals, "Partitions" -> OptionValue["Partitions"], "Bins" -> OptionValue["Bins"]];
  I = Max[0.01, Phi/(1 + Phi)];

  ratio = If[phiC == 0, 0.0, Phi/phiC];
  sigma = 1/(1 + Exp[-k * (ratio - 1)]);

  N[chiMax * sigma * (E^α) * (ratio^β) * τ * (S^γ) * (U^δ) * (I^ϵ)]
];

End[];
EndPackage[];




