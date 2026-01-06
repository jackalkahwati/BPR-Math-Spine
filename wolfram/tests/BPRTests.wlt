(* Run (example):
   wolframscript -code "TestReport[\"wolfram/tests/BPRTests.wlt\"]"
*)

Get[FileNameJoin[{DirectoryName[$TestFileName], "..", "BPR.wl"}]];

TestReport[{

  VerificationTest[
    BPR`BPRLaplacianEigenvaluesSphere[3],
    {{0, 0}, {1, 2}, {2, 6}, {3, 12}},
    TestID -> "Checkpoint1-AnalyticSphereEigenvalues"
  ],

  VerificationTest[
    Head @ BPR`BPRSolvePhaseSphereSpectral[Function[{θ, ϕ}, Re[SphericalHarmonicY[2, 0, θ, ϕ]]], 1.0, 6],
    List,
    TestID -> "BoundarySolve-SphereSpectral-ReturnsAssociation"
  ],

  VerificationTest[
    NumericQ @ BPR`BPRPhaseEnergySphereSpectral[
      BPR`BPRSolvePhaseSphereSpectral[Function[{θ, ϕ}, Re[SphericalHarmonicY[2, 0, θ, ϕ]]], 1.0, 6]
    ],
    True,
    TestID -> "BoundarySolve-SphereSpectral-EnergyNumeric"
  ],

  VerificationTest[
    ListQ @ BPR`BPRCasimirForce[10^-6, "CouplingLambda" -> 10^-3],
    True,
    TestID -> "CasimirForce-ReturnsAssociation"
  ],

  VerificationTest[
    NumericQ @ Part[BPR`BPRCasimirForceRow[10^-6, "CouplingLambda" -> 10^-3, "LMax" -> 4], 3],
    True,
    TestID -> "CasimirForce-HasBPRCorrectionKey"
  ]

}]


