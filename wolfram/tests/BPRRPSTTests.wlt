(* Run:
   wolframscript -code "TestReport[\"wolfram/tests/BPRRPSTTests.wlt\"]"
*)

Get[FileNameJoin[{DirectoryName[$TestFileName], "..", "BPR.wl"}]];

(* Standard .wlt: just VerificationTest expressions (no nested TestReport). *)

VerificationTest[
  Module[{field, p, a, b, out},
    p = 17;
    field = BPR`BPRRPSTPrimeField[p];
    a = {15, 16, 0, 1};
    b = {5, 5, 5, 5};
    out = field["Add"][a, b];
    VectorQ[out, IntegerQ] && Min[out] >= 0 && Max[out] < p
  ],
  True,
  TestID -> "RPST-PrimeField-Closure"
]

VerificationTest[
  Module[{p, n, q, pi, state, evo},
    p = 101;
    n = 20;
    SeedRandom[0];
    q = RandomInteger[{0, p - 1}, n];
    pi = RandomInteger[{0, p - 1}, n];
    state = BPR`BPRRPSTSubstrateState[q, pi, p, 0];
    evo = BPR`BPRRPSTSymplecticEvolution[p];
    TrueQ[evo["VerifyReversibility"][state, 50]]
  ],
  True,
  TestID -> "RPST-Symplectic-Reversibility"
]

VerificationTest[
  Module[{p, q, loop, q2, loop2, w0, w1},
    p = 31;
    q = ConstantArray[5, 10];
    loop = Range[10];
    w0 = BPR`BPRGet[BPR`BPRRPSTComputeWindingNumber[q, loop, p], "winding"];

    q2 = Range[0, p - 1];
    loop2 = Range[p];
    w1 = BPR`BPRGet[BPR`BPRRPSTComputeWindingNumber[q2, loop2, p], "winding"];

    (w0 === 0) && (w1 === 1)
  ],
  True,
  TestID -> "RPST-Winding-Trivial-And-Single"
]

VerificationTest[
  Module[{p, m, eigs},
    p = 31;
    m = BPR`BPRRPSTHamiltonianMatrix[p];
    eigs = BPR`BPRRPSTHamiltonianEigenvalues[p];
    MatrixQ[m] && Dimensions[m] === {p, p} && Chop[m - Transpose[m]] === ConstantArray[0, {p, p}] && VectorQ[eigs, NumericQ] && Length[eigs] === p
  ],
  True,
  TestID -> "RPST-Hamiltonian-Basic"
]


