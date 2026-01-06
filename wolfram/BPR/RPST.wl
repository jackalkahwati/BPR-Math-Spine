(* ::Package:: *)
BeginPackage["BPR`"];

(* RPST (Resonant Prime Substrate Theory) - Eq 0a–0e scaffolding *)

BPRRPSTLegendreSymbol::usage =
  "BPRRPSTLegendreSymbol[a,p] returns the Legendre symbol (a/p) for odd prime p (0, 1, or -1).";

BPRRPSTPrimeField::usage =
  "BPRRPSTPrimeField[p] returns an Association representing arithmetic in Z_p with keys \"p\",\"Mod\",\"Add\",\"Sub\",\"Mul\".";

BPRRPSTSubstrateState::usage =
  "BPRRPSTSubstrateState[q,pi,p,t] constructs an RPST phase-space state on Z_p with integer vectors q and pi.";

BPRRPSTSymplecticEvolution::usage =
  "BPRRPSTSymplecticEvolution[p,J] constructs a reversible update rule on Z_p with optional coupling matrix J.";

BPRRPSTMinimalSignedDifference::usage =
  "BPRRPSTMinimalSignedDifference[a,b,p] returns the minimal signed difference (a-b) on Z_p in roughly [-(p-1)/2,(p-1)/2].";

BPRRPSTComputeWindingNumber::usage =
  "BPRRPSTComputeWindingNumber[qValues, loopIndices, p] computes the winding number along a closed loop (Eq 0c).";

BPRRPSTVerifyChargeConservation::usage =
  "BPRRPSTVerifyChargeConservation[q0,q1,p,loops] verifies winding conservation across a set of loops; returns {passes,maxDeviation}.";

BPRRPSTCoarseGraining::usage =
  "BPRRPSTCoarseGraining[positions, coarseScale] constructs a kernel coarse-grainer (Eq 0d/0e scaffolding).";

BPRRPSTVerifyWaveEquation::usage =
  "BPRRPSTVerifyWaveEquation[trajectory, positions, evalPoints, dt, coarseScale, c, tolerance] checks a crude discrete wave residual.";

BPRRPSTHamiltonianMatrix::usage =
  "BPRRPSTHamiltonianMatrix[p] builds the Legendre-symbol outer-product Hamiltonian (rank-1 scaffold).";

BPRRPSTHamiltonianEigenvalues::usage =
  "BPRRPSTHamiltonianEigenvalues[p] returns sorted eigenvalues of the RPST Hamiltonian.";

Begin["`Private`"];

BPRRPSTLegendreSymbol[a_Integer, p_Integer] := Module[{aa = Mod[a, p]},
  If[p < 2, Return[$Failed]];
  If[aa == 0, Return[0]];
  (* For odd primes, JacobiSymbol equals Legendre symbol. *)
  JacobiSymbol[aa, p]
];

BPRRPSTPrimeField[p_Integer] := Module[{pp = int[p]},
  If[pp <= 1, Return[$Failed]];
  <|
    "p" -> pp,
    "Mod" -> Function[x, Mod[x, pp]],
    "Add" -> Function[{a, b}, Mod[a + b, pp]],
    "Sub" -> Function[{a, b}, Mod[a - b, pp]],
    "Mul" -> Function[{a, b}, Mod[a b, pp]]
  |>
];

(* helper: robust integer coercion *)
int[x_] := Quiet @ Check[Round[x], x];

(* State is an Association: keep it lightweight and easy to inspect. *)
BPRRPSTSubstrateState[q_List, pi_List, p_Integer, t_Integer : 0] := Module[
  {pp = int[p], qq, ppi},
  qq = Mod[int /@ q, pp];
  ppi = Mod[int /@ pi, pp];
  If[Length[qq] =!= Length[ppi], Return[$Failed]];
  <|
    "q" -> qq,
    "pi" -> ppi,
    "p" -> pp,
    "t" -> int[t],
    "n" -> Length[qq]
  |>
];

(* Private: validate prime modulus quickly (sufficient for tests). *)
rpstIsPrime[n_Integer] := Module[{nn = int[n], i},
  If[nn < 2, Return[False]];
  If[EvenQ[nn], Return[nn == 2]];
  i = 3;
  While[i i <= nn,
    If[Mod[nn, i] == 0, Return[False]];
    i += 2;
  ];
  True
];

BPRRPSTSymplecticEvolution[p_Integer, J_: Automatic] := Module[{pp = int[p], field, JJ},
  If[!TrueQ[rpstIsPrime[pp]], Return[$Failed]];
  field = BPRRPSTPrimeField[pp];
  JJ = J;
  <|
    "p" -> pp,
    "J" -> JJ,
    "Step" -> Function[state, rpstStep[field, JJ, state]],
    "StepInverse" -> Function[state, rpstStepInverse[field, JJ, state]],
    "Evolve" -> Function[{state, steps}, rpstEvolve[field, JJ, state, steps]],
    "VerifyReversibility" -> Function[{state, steps}, rpstVerifyReversibility[field, JJ, state, steps]]
  |>
];

rpstCoupling[field_Association, JJ_, q_List] := Module[{p = field["p"], qv, Jm},
  If[JJ === Automatic || JJ === None, Return[ConstantArray[0, Length[q]]]];
  Jm = JJ;
  qv = int /@ q;
  If[!MatrixQ[Jm] || Dimensions[Jm] =!= {Length[qv], Length[qv]}, Return[$Failed]];
  Mod[Jm . qv, p]
];

rpstStep[field_Association, JJ_, state_Association] := Module[
  {p = field["p"], q, pi, force, piNext, qNext, t},
  q = BPRGet[state, "q"];
  pi = BPRGet[state, "pi"];
  t = int @ BPRGet[state, "t"];
  force = rpstCoupling[field, JJ, q];
  If[force === $Failed, Return[$Failed]];
  piNext = Mod[pi - force, p];
  qNext = Mod[q + piNext, p];
  BPRRPSTSubstrateState[qNext, piNext, p, t + 1]
];

rpstStepInverse[field_Association, JJ_, state_Association] := Module[
  {p = field["p"], qNext, piNext, q, force, pi, t},
  qNext = BPRGet[state, "q"];
  piNext = BPRGet[state, "pi"];
  t = int @ BPRGet[state, "t"];
  q = Mod[qNext - piNext, p];
  force = rpstCoupling[field, JJ, q];
  If[force === $Failed, Return[$Failed]];
  pi = Mod[piNext + force, p];
  BPRRPSTSubstrateState[q, pi, p, t - 1]
];

rpstEvolve[field_Association, JJ_, state_Association, steps_Integer] := Module[{s = state, traj = {state}, k},
  For[k = 1, k <= int[steps], k++,
    s = rpstStep[field, JJ, s];
    If[s === $Failed, Return[$Failed]];
    AppendTo[traj, s];
  ];
  traj
];

rpstVerifyReversibility[field_Association, JJ_, state_Association, steps_Integer : 50] := Module[
  {traj, forward, s, k},
  traj = rpstEvolve[field, JJ, state, steps];
  If[traj === $Failed, Return[False]];
  forward = Last[traj];
  s = forward;
  For[k = 1, k <= int[steps], k++,
    s = rpstStepInverse[field, JJ, s];
    If[s === $Failed, Return[False]];
  ];
  And[
    Mod[BPRGet[s, "q"], field["p"]] === Mod[BPRGet[state, "q"], field["p"]],
    Mod[BPRGet[s, "pi"], field["p"]] === Mod[BPRGet[state, "pi"], field["p"]]
  ]
];

BPRRPSTMinimalSignedDifference[a_Integer, b_Integer, p_Integer] := Module[
  {pp = int[p], diff, halfP},
  diff = Mod[int[a] - int[b], pp];
  halfP = Floor[pp/2];
  If[diff > halfP, diff - pp, diff]
];

BPRRPSTComputeWindingNumber[qValues_List, loopIndices_List, p_Integer] := Module[
  {pp = int[p], q = Mod[int /@ qValues, int[p]], n, totalDelta = 0, k, i, j, delta, wCont, w},
  n = Length[loopIndices];
  If[n < 2, Return[<|"winding" -> 0, "loop_path" -> loopIndices, "total_phase" -> 0.0, "continuous_winding" -> 0.0|>]];
  For[k = 1, k <= n, k++,
    i = loopIndices[[k]];
    j = loopIndices[[Mod[k, n] + 1]];
    delta = BPRRPSTMinimalSignedDifference[q[[j]], q[[i]], pp];
    totalDelta += int[delta];
  ];
  wCont = N[totalDelta/pp];
  w = Round[wCont];
  If[Abs[wCont - w] > 0.01, Return[$Failed]];
  <|
    "winding" -> int[w],
    "loop_path" -> loopIndices,
    "total_phase" -> N[2 Pi wCont],
    "continuous_winding" -> wCont
  |>
];

BPRRPSTVerifyChargeConservation[q0_List, q1_List, p_Integer, loops_List] := Module[
  {maxDev = 0.0, loop, w0, w1},
  Do[
    loop = loops[[k]];
    w0 = BPRGet[BPRRPSTComputeWindingNumber[q0, loop, p], "winding"];
    w1 = BPRGet[BPRRPSTComputeWindingNumber[q1, loop, p], "winding"];
    maxDev = Max[maxDev, Abs[int[w0] - int[w1]]],
    {k, 1, Length[loops]}
  ];
  {maxDev < 0.5, N[maxDev]}
];

BPRRPSTCoarseGraining[positions_List, coarseScale_ : 0.5] := Module[
  {pos = N[positions], cs = N[coarseScale]},
  If[cs <= 0, Return[$Failed]];
  <|
    "positions" -> pos,
    "coarse_scale" -> cs,
    "FieldFromState" -> Function[{state, evalPoints}, rpstFieldFromState[pos, cs, state, evalPoints]],
    "VerifyWaveEquation" -> Function[{trajectory, dt, evalPoints, c, tol}, rpstVerifyWaveEquation[pos, cs, trajectory, dt, evalPoints, c, tol]]
  |>
];

rpstKernel[cs_?NumericQ, x_List, y_List] := Module[{r2 = Total[(x - y)^2]},
  Exp[-r2/(2 cs^2)]
];

rpstFieldFromState[pos_, cs_, state_Association, evalPointsIn_] := Module[
  {p = int @ BPRGet[state, "p"], q, qReal, evalPoints, out, xp, w, sw},
  q = Mod[int /@ BPRGet[state, "q"], p];
  qReal = N[q/p - 0.5];
  evalPoints = N[evalPointsIn];
  If[VectorQ[evalPoints], evalPoints = List /@ evalPoints];
  out = ConstantArray[0.0, Length[evalPoints]];
  Do[
    xp = evalPoints[[i]];
    w = rpstKernel[cs, xp, #] & /@ pos;
    sw = Total[w];
    out[[i]] = If[sw == 0, 0.0, N[Total[w qReal]/sw]],
    {i, 1, Length[evalPoints]}
  ];
  out
];

rpstVerifyWaveEquation[pos_, cs_, trajectory_List, dt_?NumericQ, evalPoints_, c_?NumericQ, tol_?NumericQ] := Module[
  {cg, u0, u1, u2, utt, x, dx, uxx, residual, maxRes},
  If[Length[trajectory] < 3, Return[{False, Infinity}]];
  cg = BPRRPSTCoarseGraining[pos, cs];
  u0 = cg["FieldFromState"][trajectory[[1]], evalPoints];
  u1 = cg["FieldFromState"][trajectory[[2]], evalPoints];
  u2 = cg["FieldFromState"][trajectory[[3]], evalPoints];
  utt = (u2 - 2 u1 + u0)/(dt^2);
  x = N[Flatten[evalPoints]];
  If[Length[x] < 3, Return[{False, Infinity}]];
  dx = Mean[Differences[x]];
  If[dx == 0, Return[{False, Infinity}]];
  uxx = ConstantArray[0.0, Length[u1]];
  uxx[[2 ;; -2]] = (u1[[3 ;;]] - 2 u1[[2 ;; -2]] + u1[[;; -3]])/(dx^2);
  uxx[[1]] = uxx[[2]];
  uxx[[-1]] = uxx[[-2]];
  residual = utt - (c^2) uxx;
  maxRes = Max[Abs[residual]];
  {maxRes < tol, N[maxRes]}
];

BPRRPSTVerifyWaveEquation[trajectory_List, positions_List, evalPoints_, dt_?NumericQ, coarseScale_ : 0.5, c_ : 1.0, tolerance_ : 1.0] := Module[
  {cg = BPRRPSTCoarseGraining[positions, coarseScale]},
  If[cg === $Failed, Return[$Failed]];
  cg["VerifyWaveEquation"][trajectory, dt, evalPoints, N[c], N[tolerance]]
];

BPRRPSTHamiltonianMatrix[p_Integer] := Module[{pp = int[p], leg},
  leg = N@Table[BPRRPSTLegendreSymbol[a, pp], {a, 0, pp - 1}];
  Outer[Times, leg, leg]
];

BPRRPSTHamiltonianEigenvalues[p_Integer] := Sort[Eigenvalues[BPRRPSTHamiltonianMatrix[p]]];

End[];
EndPackage[];


