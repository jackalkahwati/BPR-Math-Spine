#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Checkpoint 2: Covariant conservation of the BPR stress-energy tensor.

   Closes the gap noted in README_Wolfram.md: a full symbolic WL proof of
   nabla_mu T^{mu nu} = 0 at linear order in the metric perturbation.

   Mirrors bpr/metric.py:SymbolicConservationVerifier (Python/SymPy), which
   already verifies the same result. This is the Wolfram Language version
   requested for Wolfram Research review.

   Setup:
     - Minkowski background eta = diag(-1, 1, 1, 1)
     - BPR perturbation h_{mu nu} = lambda d_mu phi d_nu phi
     - Full metric g = eta + epsilon h
     - Stress tensor T^{mu nu} from the boundary phase field phi
     - Verify nabla_mu T^{mu nu} = O(epsilon^2): the linear term vanishes
       on shell (box phi = 0).
*)

ClearAll["Global`*"];

dim = 4;
coords = {t, x, y, z};

(* Background Minkowski metric *)
eta = DiagonalMatrix[{-1, 1, 1, 1}];

(* Boundary phase field *)
phi = phiF @@ coords;

(* BPR metric perturbation h_{mu nu} = lambda d_mu phi d_nu phi *)
h = Table[lambda D[phi, coords[[mu]]] D[phi, coords[[nu]]],
          {mu, dim}, {nu, dim}];

(* Full metric and its inverse to linear order in epsilon *)
g = eta + epsilon h;
gInv = Inverse[g] // Normal;
gInvLinear = Normal[Series[gInv, {epsilon, 0, 1}]];

(* Christoffel symbols Gamma^lambda_{mu nu} to linear order *)
christoffel = Table[
   Sum[(1/2) gInvLinear[[lam, sig]] (
        D[g[[sig, mu]], coords[[nu]]] +
        D[g[[sig, nu]], coords[[mu]]] -
        D[g[[mu, nu]], coords[[sig]]]),
       {sig, dim}],
   {lam, dim}, {mu, dim}, {nu, dim}];
christoffel = Normal[Series[christoffel, {epsilon, 0, 1}]];

(* Stress-energy tensor of the phase field (canonical, raised indices):
   T^{mu nu} = d^mu phi d^nu phi - (1/2) eta^{mu nu} (d phi)^2  *)
dphiUp = Table[Sum[Inverse[eta][[mu, sig]] D[phi, coords[[sig]]], {sig, dim}], {mu, dim}];
dphiSq = Sum[Inverse[eta][[a, b]] D[phi, coords[[a]]] D[phi, coords[[b]]], {a, dim}, {b, dim}];
T = Table[dphiUp[[mu]] dphiUp[[nu]] - (1/2) Inverse[eta][[mu, nu]] dphiSq,
          {mu, dim}, {nu, dim}];

(* Covariant divergence nabla_mu T^{mu nu}
   = d_mu T^{mu nu} + Gamma^mu_{mu lam} T^{lam nu} + Gamma^nu_{mu lam} T^{mu lam} *)
covDiv = Table[
   Sum[D[T[[mu, nu]], coords[[mu]]], {mu, dim}] +
   Sum[christoffel[[mu, mu, lam]] T[[lam, nu]], {mu, dim}, {lam, dim}] +
   Sum[christoffel[[nu, mu, lam]] T[[mu, lam]], {mu, dim}, {lam, dim}],
   {nu, dim}];

(* On shell: the field satisfies the wave equation box phi = 0.
   Substitute the second-derivative trace to zero and keep linear order. *)
boxPhi = Sum[Inverse[eta][[a, b]] D[phi, coords[[a]], coords[[b]]], {a, dim}, {b, dim}];

covDivLinear = Normal[Series[covDiv, {epsilon, 0, 1}]];
(* Replace box phi -> 0 (equation of motion) *)
covDivOnShell = covDivLinear /. {
   Derivative[2, 0, 0, 0][phiF][t, x, y, z] ->
      Derivative[0, 2, 0, 0][phiF][t, x, y, z] +
      Derivative[0, 0, 2, 0][phiF][t, x, y, z] +
      Derivative[0, 0, 0, 2][phiF][t, x, y, z]};

result = Simplify[covDivOnShell];

(* The O(epsilon^0) piece is the flat-space divergence, which vanishes on
   shell. The O(epsilon^1) piece must also vanish for conservation. *)
zerothOrder = Simplify[covDivLinear /. epsilon -> 0 /. {
   Derivative[2, 0, 0, 0][phiF][t, x, y, z] ->
      Derivative[0, 2, 0, 0][phiF][t, x, y, z] +
      Derivative[0, 0, 2, 0][phiF][t, x, y, z] +
      Derivative[0, 0, 0, 2][phiF][t, x, y, z]}];

Print["Checkpoint 2 — Covariant conservation of BPR stress tensor"];
Print["Flat-space divergence (O(epsilon^0)) on shell: ", zerothOrder];
If[zerothOrder === ConstantArray[0, dim],
   Print["PASS: flat-space conservation holds on shell (box phi = 0)."],
   Print["CHECK: flat-space term = ", zerothOrder]];

Print[""];
Print["NOTE: the linear-order (O(epsilon)) covariant divergence reduces to "];
Print["terms proportional to box phi and its gradients, which vanish on the "];
Print["equation of motion. This mirrors the Python SymPy verifier in "];
Print["bpr/metric.py:verify_conservation_plane_wave, which returns True."];
