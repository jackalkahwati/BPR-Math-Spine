"""
BPR Physical Constants (Canonical Source)
==========================================

All physical constants used across BPR modules, defined once.
Import from here instead of hardcoding values.

Usage:
    from bpr.constants import HBAR, C, G, L_PLANCK, K_B, ...

References: CODATA 2018 recommended values
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Fundamental constants
# ---------------------------------------------------------------------------

C = 299792458.0                    # Speed of light [m/s]
HBAR = 1.054571817e-34             # Reduced Planck constant [J·s]
H_PLANCK = 6.62607015e-34          # Planck constant [J·s]
G = 6.67430e-11                    # Gravitational constant [m³/(kg·s²)]
K_B = 1.380649e-23                 # Boltzmann constant [J/K]
E_CHARGE = 1.602176634e-19         # Elementary charge [C]
EPSILON_0 = 8.8541878128e-12       # Vacuum permittivity [F/m]
MU_0 = 1.25663706212e-6            # Vacuum permeability [H/m]
Z_0 = 376.730313668                # Vacuum impedance [Ω]
ALPHA_EM = 1.0 / 137.035999084     # Fine structure constant (q² → 0)
ALPHA_EM_MZ = 1.0 / 127.952       # α_EM at M_Z scale

# ---------------------------------------------------------------------------
# Planck units (derived)
# ---------------------------------------------------------------------------

L_PLANCK = np.sqrt(HBAR * G / C**3)       # 1.616255e-35 m
T_PLANCK = np.sqrt(HBAR * G / C**5)       # 5.391247e-44 s
M_PLANCK = np.sqrt(HBAR * C / G)          # 2.176434e-8 kg
E_PLANCK = M_PLANCK * C**2                 # 1.956e9 J = 1.22e28 eV
M_PLANCK_GEV = E_PLANCK / E_CHARGE / 1e9  # ~1.22e19 GeV

# ---------------------------------------------------------------------------
# Particle physics
# ---------------------------------------------------------------------------

V_HIGGS = 246.22                   # Electroweak VEV [GeV] (PDG 2024)
M_Z = 91.1876                      # Z boson mass [GeV]
M_W = 80.377                       # W boson mass [GeV]
M_PROTON = 938.272088              # Proton mass [MeV]
M_NEUTRON = 939.565420             # Neutron mass [MeV]
M_NUCLEON_AVG = 938.918            # Average nucleon mass [MeV]
M_ELECTRON = 0.51099895            # Electron mass [MeV]
HBAR_C = 197.3269804               # ℏc [MeV·fm]
LAMBDA_QCD = 0.217                 # QCD scale [GeV]

# ---------------------------------------------------------------------------
# Astrophysical / cosmological
# ---------------------------------------------------------------------------

M_SUN = 1.989e30                   # Solar mass [kg]
R_SUN = 6.957e8                    # Solar radius [m]
H_0 = 67.4                        # Hubble constant [km/s/Mpc]
R_HUBBLE = C / (H_0 * 1e3 / 3.086e22)  # Hubble radius [m]
T_CMB = 2.7255                     # CMB temperature [K]
OMEGA_LAMBDA = 0.685               # Dark energy fraction (Planck 2018)

# ---------------------------------------------------------------------------
# Nuclear
# ---------------------------------------------------------------------------

R_NUCLEAR = 1.25                   # Nuclear radius constant r₀ [fm]

# ---------------------------------------------------------------------------
# BPR-specific derived constants
# ---------------------------------------------------------------------------

# Default substrate prime
P_DEFAULT = 104761
Z_DEFAULT = 6

# Golden ratio
PHI_GOLDEN = (1 + np.sqrt(5)) / 2  # ≈ 1.6180339887

# Riemann zero imaginary parts (first 5)
GAMMA_ZEROS = np.array([14.1347, 21.0220, 25.0109, 30.4249, 32.9351])
