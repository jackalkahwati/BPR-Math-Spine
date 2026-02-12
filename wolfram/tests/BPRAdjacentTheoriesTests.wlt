(* Run:
   wolframscript -code "TestReport[\"wolfram/tests/BPRAdjacentTheoriesTests.wlt\"]"
*)

Get[FileNameJoin[{DirectoryName[$TestFileName], "..", "BPR.wl"}]];

(* Reference values from Python benchmark (p=104729) *)
(* P11.15: Omega_DM h^2 ~ 0.1197 (Planck: 0.120) *)
(* P18.2: m_mu ~ 107.2 MeV (exp: 105.66) *)
(* P19.8: B/A(He4) ~ 7.07 MeV *)
(* P19.9: rho_0 ~ 0.163 fm^-3 (exp: 0.16) *)
(* P4.7: Tc(Nb) ~ 9.2 K with N0V=0.325, T_D=275 *)
(* P12.14: m_pi ~ 126 MeV with condensate 270^3 *)

(* P11.15: Dark matter relic abundance (W_c = Sqrt[3] from first principles) *)
VerificationTest[
  Module[{omega},
    omega = BPR`BPRDMRelicAbundance[Sqrt[3], 104729];
    omega > 0.10 && omega < 0.14
  ],
  True,
  TestID -> "P11.15-DMRelicInRange"
]

(* f_decoh derivation: 1 - Sqrt[E]/p^(1/4) *)
VerificationTest[
  Module[{p, fDecoh},
    p = 104729;
    fDecoh = 1.0 - Sqrt[E]/N[p^(1/4)];
    fDecoh > 0.90 && fDecoh < 0.95
  ],
  True,
  TestID -> "P11.15-FDecohDerived"
]

(* P18.2: Charged lepton masses *)
VerificationTest[
  Module[{masses, me, mmu, mtau},
    masses = BPR`BPRChargedLeptonMasses[];
    me = masses["e"];
    mmu = masses["mu"];
    mtau = masses["tau"];
    me > 0.5 && me < 0.52 &&
    mmu > 105 && mmu < 110 &&
    mtau > 1776 && mtau < 1778
  ],
  True,
  TestID -> "P18.2-ChargedLeptonMasses"
]

(* P18.2: l_mu = Sqrt[14*15] *)
VerificationTest[
  Module[{lModes},
    lModes = BPR`BPRLeptonLModes[];
    Abs[lModes[[2]] - Sqrt[210]] < 0.01
  ],
  True,
  TestID -> "P18.2-LMuDerived"
]

(* P19.8: Binding energy per nucleon He4 *)
VerificationTest[
  Module[{bPerA},
    bPerA = BPR`BPRBindingEnergyPerNucleon[4, 2];
    bPerA > 6.9 && bPerA < 7.2
  ],
  True,
  TestID -> "P19.8-BPerAHe4"
]

(* P19.8: Fe56 *)
VerificationTest[
  Module[{bPerA},
    bPerA = BPR`BPRBindingEnergyPerNucleon[56, 26];
    bPerA > 8.7 && bPerA < 9.0
  ],
  True,
  TestID -> "P19.8-BPerAFe56"
]

(* P19.9: Nuclear saturation density *)
VerificationTest[
  Module[{rho0},
    rho0 = BPR`BPRNuclearSaturationDensity[1.25];
    rho0 > 0.15 && rho0 < 0.17
  ],
  True,
  TestID -> "P19.9-SaturationDensity"
]

(* P4.7: Superconductor Tc Nb *)
VerificationTest[
  Module[{tc},
    tc = BPR`BPRSuperconductorTc[0.325, 275.0];
    tc > 9.0 && tc < 9.5
  ],
  True,
  TestID -> "P4.7-TcNb"
]

(* P4.9: MgB2 *)
VerificationTest[
  Module[{tc},
    tc = BPR`BPRSuperconductorTc[0.355, 900.0];
    tc > 38 && tc < 41
  ],
  True,
  TestID -> "P4.9-TcMgB2"
]

(* P12.14: Pion mass *)
VerificationTest[
  Module[{mpi},
    mpi = BPR`BPRPionMass[2.16, 4.67, 92.1];
    mpi > 120 && mpi < 130
  ],
  True,
  TestID -> "P12.14-PionMass"
]
