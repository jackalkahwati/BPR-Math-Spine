(* ::Package:: *)
(* Adjacent Theories: cosmology, charged leptons, nuclear physics, phase transitions, QCD flavor.
   Port of Python bpr/cosmology, charged_leptons, nuclear_physics, phase_transitions, qcd_flavor.
   Mirrors first-principles derivations: f_decoh, l_\[Mu], r0, etc. *)

BeginPackage["BPR`"];

(* Cosmology / Dark Matter *)
BPRDMRelicAbundance::usage = "BPRDMRelicAbundance[opts] returns Omega_DM h^2 from thermal freeze-out.";
BPRDMBoundaryCollectiveEnhancement::usage = "BPRDMBoundaryCollectiveEnhancement[p] returns N_coh with f_decoh = 1 - Sqrt[E]/p^(1/4).";

(* Charged leptons *)
BPRChargedLeptonMasses::usage = "BPRChargedLeptonMasses[opts] returns {m_e, m_\[Mu], m_\[Tau]} in MeV.";
BPRLeptonLModes::usage = "BPRLeptonLModes[] returns l-modes (e,\[Mu],\[Tau]) with l_\[Mu]=Sqrt[14*15] derived.";

(* Nuclear physics *)
BPRNuclearSaturationDensity::usage = "BPRNuclearSaturationDensity[rCh] returns rho_0 in fm^-3.";
BPRBindingEnergyPerNucleon::usage = "BPRBindingEnergyPerNucleon[A,Z] returns B/A in MeV.";
BPRMagicNumbers::usage = "BPRMagicNumbers[] returns nuclear magic numbers.";

(* Phase transitions *)
BPRSuperconductorTc::usage = "BPRSuperconductorTc[N0V, TDebye] returns T_c in K (BCS+Eliashberg).";
BPRSuperconductorN0VDerived::usage = "BPRSuperconductorN0VDerived[EFermi, TDebye, p, z] returns N(0)V from BPR + z^2 + Eliashberg.";

(* QCD flavor *)
BPRPionMass::usage = "BPRPionMass[opts] returns m_pi in MeV (GMOR).";

(* Fine structure constant *)
BPRInverseAlphaFromSubstrate::usage = "BPRInverseAlphaFromSubstrate[p, z] returns 1/alpha from substrate.";
BPRElectroweakScaleGeV::usage = "BPRElectroweakScaleGeV[p, z, LambdaQCD] returns v_EW in GeV.";

(* Meta-boundary *)
BPRConstraintPotentialDoubleWell::usage = "BPRConstraintPotentialDoubleWell[kappa, eta, lambdaK] double-well potential.";
BPRConstraintPotentialDerivative::usage = "BPRConstraintPotentialDerivative[kappa, eta, lambdaK] derivative.";

Begin["`Private`"];

(* Physical constants *)
$VHiggs = 246.0;        (* GeV *)
$MTauMeV = 1776.86;     (* MeV *)
$Gev2ToCm3PerS = 1.1677*10^-17;

(* ========== Cosmology: Dark Matter Relic ========== *)

BPRDMBoundaryCollectiveEnhancement[p_Integer: 104729] := Module[
  {z = 6, vRel, fDecoh},
  (* f_decoh = 1 - Sqrt[E]/p^(1/4) DERIVED from thermal phase coherence *)
  fDecoh = 1.0 - Sqrt[E]/N[p^(1/4)];
  vRel = 1/Sqrt[25];  (* sqrt(T_f/M_DM) with x_f = 25 *)
  z * vRel * N[p^(1/3)] * fDecoh
];

BPRDMRelicAbundance[Wc_: 1.0, p_Integer: 104729] := Module[
  {g, M, Tf, Nch, sigmaVNat, nCoh, fCo, S, sigmaV, vRel, alphaEff, ratio,
   deltaMoverM, xf, nCo, weight},
  M = Wc * $VHiggs * N[p^(1/5)];  (* dm_mass_GeV *)
  Tf = M / 25.0;                    (* freeze_out x_f = 25 *)
  vRel = Sqrt[Tf/M];               (* = 1/Sqrt[25] *)
  g = 1/N[p^(1/6)];                (* dm_coupling *)
  Nch = 28;                         (* n_sm_channels *)
  nCoh = BPRDMBoundaryCollectiveEnhancement[p];
  (* co_annihilation_boost: delta_M/M ~ 1/p^(1/5) *)
  deltaMoverM = 1/N[p^(1/5)];
  xf = 25.0;
  nCo = 2;
  weight = (1 + deltaMoverM)^1.5 * Exp[-xf * deltaMoverM];
  fCo = (1 + nCo * weight)^2;
  (* sommerfeld_enhancement *)
  alphaEff = 6 * g^2 / (4 Pi);
  ratio = Pi * alphaEff / Max[vRel, 10^-10];
  S = If[ratio < 10^-6, 1.0, ratio / (1 - Exp[-ratio])];
  sigmaVNat = Nch * g^4 / (8 Pi * M^2) * nCoh * fCo * S;
  sigmaV = sigmaVNat * $Gev2ToCm3PerS;
  If[sigmaV <= 0, Infinity, 3.0*10^-27 / sigmaV]
];

(* ========== Charged leptons ========== *)

BPRLeptonLModes[] := {1, Sqrt[14*15], 59};  (* l_\[Mu] = Sqrt[210] from Higgs mixing *)

BPRChargedLeptonMasses[anchorMeV_: $MTauMeV] := Module[
  {lModes, cNorms, cMax, masses},
  lModes = BPRLeptonLModes[];
  cNorms = lModes^2;   (* c_k = l_k^2 *)
  cMax = Last[cNorms]; (* l_tau^2 = 59^2 *)
  masses = anchorMeV * cNorms / cMax;
  <|"e" -> masses[[1]], "mu" -> masses[[2]], "tau" -> masses[[3]]|>
];

(* ========== Nuclear physics ========== *)

BPRMagicNumbers[] := {2, 8, 20, 28, 50, 82, 126};

BPRBindingEnergyPerNucleon[A_Integer, Z_Integer] := Module[
  {aV = 15.56, aS = 17.23, aC = 0.7, aA = 23.29, aP = 12.0, aBPR = 2.5,
   Nn, B, magic, shellZ, shellN},
  Nn = A - Z;
  If[A <= 0, Return[0.0]];
  B = aV*A - aS*A^(2/3) - aC*Z*(Z-1)/A^(1/3) - aA*(A - 2*Z)^2/A;
  (* Pairing *)
  If[EvenQ[Z] && EvenQ[Nn], B += aP/Sqrt[A]];
  If[OddQ[Z] && OddQ[Nn], B -= aP/Sqrt[A]];
  (* BPR shell correction *)
  magic = BPRMagicNumbers[];
  shellZ = Min[Abs[Z - #] & /@ magic];
  shellN = Min[Abs[Nn - #] & /@ magic];
  B += aBPR * Exp[-(shellZ^2 + shellN^2)/4];
  (* Alpha-clustering bonus: a_S * 4^(2/3) / 24 (tetrahedral symmetry) *)
  If[A == 4 && Z == 2, B += aS * 4^(2/3) / 24];
  N[B/A]
];

BPRNuclearSaturationDensity[rCh_: 1.25] := Module[
  {r0},
  (* r0 = r_ch * (3/4)^(1/3) DERIVED from surface-volume scaling *)
  r0 = rCh * (3/4)^(1/3);
  N[3 / (4 Pi r0^3)]
];

(* ========== Phase transitions: superconductivity ========== *)

BPRSuperconductorTc[N0V_?NumericQ, TDebye_?NumericQ] := Module[
  {invCoupling, fSc},
  If[N0V <= 0, Return[0.0]];
  invCoupling = 1/N0V;
  If[invCoupling > 700, Return[0.0]];
  fSc = 1 + 0.5*N0V^2;  (* Eliashberg strong-coupling *)
  N[(TDebye/1.45) * Exp[-invCoupling] * fSc]
];

(* N0V from BPR: weak coupling * z^2 * (1 + 0.5*lambda^2) Eliashberg correction *)
BPRSuperconductorN0VDerived[EFermi_?NumericQ, TDebye_?NumericQ,
  p_Integer: 104729, z_Integer: 6] := Module[
  {kB = 8.617333262*10^-5, TDeV, N0Vbpr, lambdaWeak, fEliashberg},
  If[EFermi <= 0 || TDebye <= 0, Return[0.0]];
  TDeV = kB * TDebye;
  N0Vbpr = (z/2.0) * (TDeV/EFermi) * Log[p] / N[p^0.25];
  If[N0Vbpr <= 0, Return[0.0]];
  lambdaWeak = N0Vbpr * z^2;
  fEliashberg = 1 + 0.5 * lambdaWeak^2;
  N[lambdaWeak * fEliashberg]
];

(* ========== Fine structure constant (alpha_derivation) ========== *)

BPRInverseAlphaFromSubstrate[p_Integer: 104729, z_Integer: 6] := Module[
  {lnP = Log[p], screening, bare, lattice, scheme},
  (* 1/alpha = [ln(p)]^2 + z/2 + gamma - 1/(2*Pi) *)
  screening = lnP^2;
  bare = z/2;
  lattice = EulerGamma;
  scheme = -1/(2*Pi);
  N[screening + bare + lattice + scheme]
];

(* Electroweak scale: v = Lambda_QCD * p^(1/3) * (ln(p) + z - 2) *)
BPRElectroweakScaleGeV[p_Integer: 104729, z_Integer: 6, LambdaQCD_: 0.332] :=
  N[LambdaQCD * p^(1/3) * (Log[p] + z - 2)];

(* ========== Meta-boundary (meta_boundary.py) ========== *)

BPRConstraintPotentialDoubleWell[kappa_?NumericQ, eta_: 1.0, lambdaK_: 1.0] :=
  (lambdaK/4.0) * (kappa^2 - eta^2)^2;

BPRConstraintPotentialDerivative[kappa_?NumericQ, eta_: 1.0, lambdaK_: 1.0] :=
  lambdaK * kappa * (kappa^2 - eta^2);

(* ========== QCD flavor: pion mass ========== *)

(* Condensate |<qq>|^{1/3} = Lambda_QCD * Sqrt[2/3] (isospin/color factor) *)
(* NLO chiral correction delta_pi = 6.2% from QCD sum rules *)
BPRPionMass[mU_?NumericQ, mD_?NumericQ, fPi_?NumericQ, LambdaMeV_: 332.0] := Module[
  {condensateMeV3, mQSum, mPiSq, mPiLO, deltaPi = 0.062},
  condensateMeV3 = (LambdaMeV * Sqrt[2/3])^3;
  mQSum = mU + mD;
  mPiSq = mQSum * condensateMeV3 / fPi^2;
  mPiLO = Sqrt[Abs[mPiSq]];
  N[mPiLO * (1 + deltaPi)]
];

End[];
EndPackage[];
