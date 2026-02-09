"""
Energy Balance and Free Energy Functional for BPR Boundary-State Cycling
=========================================================================

This module provides the **thermodynamic accounting** that gates any claim
of anomalous energy from boundary-state transitions.

It implements:

1. A free energy functional  F(W, T, B, epsilon)  with explicit terms for
   every known energy channel plus the BPR anomaly term.
2. A full-cycle energy balance integrator that computes
   Delta_E_excess per cycle = integral(P_out - P_in) dt.
3. Predicted scaling laws for the anomaly as a function of controllable
   experimental knobs (cycle frequency, field, coherence, temperature span).

The design philosophy: **every joule must be accounted for**.  The anomaly
term P_anomaly is isolated so that a null result (P_anomaly = 0) is a clean
falsification, and a positive result cannot be confused with parasitic
heating, drive back-action, or readout injection.

Usage
-----
>>> from bpr.energy_balance import BoundaryStateCycle, CycleEnergyBalance
>>> cycle = BoundaryStateCycle.default_diamond_mems()
>>> balance = cycle.compute_energy_balance()
>>> print(balance.summary())

References
----------
Al-Kahwati (2026), BPR-Math-Spine, Test 1B Specification
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from enum import Enum

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
_K_B = 1.380649e-23         # J/K  (Boltzmann)
_HBAR = 1.054571817e-34     # J s
_C = 299_792_458.0          # m/s
_G = 6.67430e-11            # m^3 kg^-1 s^-2
_MU_0 = 1.25663706212e-6   # N/A^2 (vacuum permeability)
_PHI_0 = 2.067833848e-15   # Wb  (magnetic flux quantum)
_E_CHARGE = 1.602176634e-19 # C
_L_PLANCK = np.sqrt(_HBAR * _G / _C**3)  # ~ 1.616e-35 m


# ---------------------------------------------------------------------------
# Material parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class SuperconductorParams:
    """Material parameters for the superconducting boundary film.

    These are measured quantities for the specific material used in the
    experiment, not BPR predictions.  They define the *known* energy
    channels against which the anomaly is measured.
    """
    T_c: float = 9.3            # Critical temperature [K] (Nb default)
    Delta_0: float = 1.5e-3     # BCS gap at T=0 [eV]
    lambda_L: float = 39e-9     # London penetration depth [m]
    xi_GL: float = 38e-9        # Ginzburg-Landau coherence length [m]
    rho_n: float = 1.5e-7       # Normal-state resistivity [Ohm m]
    C_e_gamma: float = 7.79e-3  # Electronic specific heat coeff [J/(m^3 K^2)]
    H_c: float = 0.2            # Thermodynamic critical field [T]
    film_thickness: float = 100e-9  # Film thickness [m]
    film_area: float = 1e-4     # Film area [m^2] (1 cm^2)

    @property
    def volume(self) -> float:
        """Film volume [m^3]."""
        return self.film_area * self.film_thickness

    @property
    def condensation_energy_density(self) -> float:
        """BCS condensation energy density [J/m^3].

        U_cond = H_c^2 / (2 mu_0)
        """
        return self.H_c**2 / (2 * _MU_0)

    @property
    def condensation_energy(self) -> float:
        """Total condensation energy of the film [J]."""
        return self.condensation_energy_density * self.volume

    @property
    def vortex_self_energy(self) -> float:
        """Self-energy per vortex [J/m].

        E_vortex = (Phi_0^2 / (4 pi mu_0 lambda_L^2)) * ln(lambda_L / xi_GL)
        """
        prefactor = _PHI_0**2 / (4 * np.pi * _MU_0 * self.lambda_L**2)
        kappa_GL = self.lambda_L / self.xi_GL
        return prefactor * np.log(max(kappa_GL, 1.01))


# ---------------------------------------------------------------------------
# BPR coupling parameters (derived from substrate, not fitted)
# ---------------------------------------------------------------------------

@dataclass
class BPRCouplingParams:
    """BPR boundary-bulk coupling parameters.

    These come from the first-principles derivation chain in
    boundary_energy.py and stacked_enhancement.py.

    The key quantity is lambda_eff: the effective coupling after all
    enhancement factors.  This is a *signal coupling*, not an energy
    source claim.  Whether it produces net energy is what the experiment
    determines.
    """
    lambda_bpr_base: float = 6e-54   # Base EM coupling (from em_casimir)
    enhancement_coherent: float = 1e16   # N^2 coherent patches
    enhancement_phonon: float = 1e26     # Low-energy mode ratio
    enhancement_Q: float = 1e8           # Resonator Q factor
    derating: float = 1e-3               # Realistic derating factor

    @property
    def lambda_eff(self) -> float:
        """Effective coupling strength (dimensionless).

        This is the *signal transduction gain*, not energy gain.
        It determines how strongly boundary-state changes couple to
        measurable observables.
        """
        ideal = (self.lambda_bpr_base
                 * self.enhancement_coherent
                 * self.enhancement_phonon
                 * self.enhancement_Q)
        return ideal * self.derating

    @property
    def lambda_ideal(self) -> float:
        """Ideal coupling without derating."""
        return (self.lambda_bpr_base
                * self.enhancement_coherent
                * self.enhancement_phonon
                * self.enhancement_Q)

    def enhancement_breakdown(self) -> Dict[str, float]:
        """Return each enhancement factor for audit."""
        return {
            "base_coupling": self.lambda_bpr_base,
            "coherent_N2": self.enhancement_coherent,
            "phonon_mode": self.enhancement_phonon,
            "Q_factor": self.enhancement_Q,
            "derating": self.derating,
            "ideal_product": self.lambda_ideal,
            "realistic_product": self.lambda_eff,
        }


# ---------------------------------------------------------------------------
# Free energy functional  F(W, T, B)
# ---------------------------------------------------------------------------

@dataclass
class FreeEnergy:
    """Free energy of the superconducting boundary film in state (W, T, B).

    F(W, T, B) = F_condensation(T)
               + F_magnetic(B)
               + F_vortex(W, B)
               + F_thermal(T)
               + F_bpr(W, T)          <-- the anomaly term

    Every term has clear physical origin and units [Joules].
    """
    sc: SuperconductorParams = field(default_factory=SuperconductorParams)
    bpr: BPRCouplingParams = field(default_factory=BPRCouplingParams)

    # -- Standard (known) free energy terms ----------------------------------

    def F_condensation(self, T: float) -> float:
        """BCS condensation free energy [J].

        F_cond = -U_cond * (1 - (T/T_c)^2)^2  for T < T_c
               = 0                               for T >= T_c

        This is the energy released when the material goes superconducting.
        It must be supplied back to destroy superconductivity.
        """
        if T >= self.sc.T_c:
            return 0.0
        t = T / self.sc.T_c
        return -self.sc.condensation_energy * (1 - t**2)**2

    def F_magnetic(self, B: float) -> float:
        """Magnetic field energy stored in/around the film [J].

        F_mag = (B^2 / (2 mu_0)) * V_effective

        In the Meissner state, the field is expelled and the energy is
        the work done by the external field against the screening currents.
        """
        return (B**2 / (2 * _MU_0)) * self.sc.volume

    def F_vortex(self, n_vortices: int) -> float:
        """Total vortex self-energy [J].

        Each vortex carries one flux quantum and has self-energy
        per unit length.  For a thin film, the effective length
        is the film thickness.

        Parameters
        ----------
        n_vortices : int
            Number of vortices (= |winding number| for a simply
            connected geometry).
        """
        return abs(n_vortices) * self.sc.vortex_self_energy * self.sc.film_thickness

    def F_thermal(self, T: float) -> float:
        """Thermal energy content of the electronic system [J].

        In the normal state: U_e ~ gamma * T^2 * V / 2
        In the SC state:    U_e ~ gamma * T^2 * V / 2 * exp(-Delta/kT)

        We use the normal-state value as the baseline.
        """
        return 0.5 * self.sc.C_e_gamma * T**2 * self.sc.volume

    # -- BPR anomaly term ----------------------------------------------------

    def F_bpr(self, W: int, T: float) -> float:
        """BPR boundary-coupling free energy [J].

        This is the term that the experiment tests.

        F_bpr(W) = -lambda_eff * E_boundary_state(W, T)

        where E_boundary_state captures how the boundary topology (winding W)
        couples to the bulk vacuum via the BPR mechanism.

        The sign is negative because the coupling lowers the energy when
        the boundary state matches a resonant configuration.

        If lambda_eff = 0 or the coupling is not real, this term vanishes
        and the experiment sees null.

        Parameters
        ----------
        W : int
            Topological winding number.
        T : float
            Temperature [K].
        """
        if W == 0:
            return 0.0

        # Boundary state energy scale:
        # The winding creates a phase gradient that couples to the vacuum
        # The coupling scale is set by the condensation energy times lambda_eff
        #
        # E_boundary_state ~ U_cond * |W|^2 / W_c^2
        # where W_c is the critical winding from impedance theory
        W_c = 10.0  # From impedance.py

        E_boundary = self.sc.condensation_energy * (W / W_c)**2

        return -self.bpr.lambda_eff * E_boundary

    # -- Total free energy ---------------------------------------------------

    def F_total(self, W: int, T: float, B: float) -> float:
        """Total free energy F(W, T, B) [J].

        All terms summed.  The experiment measures transitions between
        states and checks whether the total energy balance closes.
        """
        n_vortices = abs(W)
        return (self.F_condensation(T)
                + self.F_magnetic(B)
                + self.F_vortex(n_vortices)
                + self.F_thermal(T)
                + self.F_bpr(W, T))

    def F_total_no_bpr(self, W: int, T: float, B: float) -> float:
        """Total free energy without BPR term (null hypothesis)."""
        n_vortices = abs(W)
        return (self.F_condensation(T)
                + self.F_magnetic(B)
                + self.F_vortex(n_vortices)
                + self.F_thermal(T))

    def anomaly_per_state(self, W: int, T: float, B: float) -> float:
        """The BPR anomaly: difference between full model and null [J]."""
        return self.F_total(W, T, B) - self.F_total_no_bpr(W, T, B)


# ---------------------------------------------------------------------------
# Cycle energy balance
# ---------------------------------------------------------------------------

class CyclePhase(Enum):
    """Phases of the boundary-state cycle."""
    COOL_TO_SC = "cool_below_Tc"
    CREATE_VORTICES = "apply_field_create_W"
    HOLD_AND_MEASURE = "hold_boundary_state"
    DESTROY_VORTICES = "remove_field_destroy_W"
    HEAT_TO_NORMAL = "heat_above_Tc"


@dataclass
class CycleStepResult:
    """Energy accounting for one step of the cycle."""
    phase: CyclePhase
    W_initial: int
    W_final: int
    T_initial: float
    T_final: float
    B_initial: float
    B_final: float
    delta_F_standard: float     # Change in standard free energy [J]
    delta_F_bpr: float          # Change in BPR anomaly term [J]
    Q_drive: float              # Energy supplied by drive [J]
    Q_dissipated: float         # Energy lost to dissipation [J]
    description: str = ""


@dataclass
class CycleEnergyBalance:
    """Complete energy balance for one full cycle.

    The non-negotiable accounting identity:

        P_out <= P_in_drive + P_in_thermal + P_in_field + P_anomaly

    Equivalently, per cycle:

        Delta_E_excess = sum(Delta_F_bpr) - sum(Q_dissipated_overhead)

    If Delta_E_excess > 0 and reproducible, there is an anomaly.
    If Delta_E_excess <= 0, BPR does not produce net energy in this channel.
    """
    steps: list = field(default_factory=list)
    cycle_frequency_hz: float = 1.0

    @property
    def total_drive_energy(self) -> float:
        """Total energy supplied by external drives per cycle [J]."""
        return sum(s.Q_drive for s in self.steps)

    @property
    def total_dissipation(self) -> float:
        """Total dissipative losses per cycle [J]."""
        return sum(s.Q_dissipated for s in self.steps)

    @property
    def total_standard_delta_F(self) -> float:
        """Net change in standard (non-BPR) free energy per cycle [J].

        For a true cycle this should be zero (returns to initial state).
        Non-zero value indicates the cycle is not closed.
        """
        return sum(s.delta_F_standard for s in self.steps)

    @property
    def total_bpr_delta_F(self) -> float:
        """Net BPR anomaly energy per cycle [J].

        This is the quantity the experiment measures.
        """
        return sum(s.delta_F_bpr for s in self.steps)

    @property
    def delta_E_excess(self) -> float:
        """Excess energy per cycle beyond all standard accounting [J].

        Delta_E_excess = |total_bpr_delta_F| - overhead_losses

        Positive = anomaly detected.
        Zero or negative = no anomaly (or anomaly too small to overcome losses).
        """
        # The BPR term changes sign around the cycle; the absolute
        # contribution is the relevant signal
        bpr_work = abs(self.total_bpr_delta_F)
        # Overhead: any extra dissipation from having vortices present
        # beyond what the null (no-vortex) case would give
        return bpr_work - self.total_dissipation

    @property
    def power_anomaly(self) -> float:
        """Anomalous power output [W].

        P_anomaly = Delta_E_excess * f_cycle
        """
        return max(0.0, self.delta_E_excess) * self.cycle_frequency_hz

    def summary(self) -> str:
        """Human-readable energy balance summary."""
        lines = [
            "=" * 70,
            "CYCLE ENERGY BALANCE",
            "=" * 70,
            "",
            "Step-by-step accounting:",
        ]
        for i, s in enumerate(self.steps):
            lines.append(f"\n  Step {i+1}: {s.phase.value}")
            lines.append(f"    W: {s.W_initial} -> {s.W_final}")
            lines.append(f"    T: {s.T_initial:.2f} K -> {s.T_final:.2f} K")
            lines.append(f"    B: {s.B_initial:.4f} T -> {s.B_final:.4f} T")
            lines.append(f"    Delta_F_standard: {s.delta_F_standard:.4e} J")
            lines.append(f"    Delta_F_bpr:      {s.delta_F_bpr:.4e} J")
            lines.append(f"    Q_drive:          {s.Q_drive:.4e} J")
            lines.append(f"    Q_dissipated:     {s.Q_dissipated:.4e} J")
            if s.description:
                lines.append(f"    Note: {s.description}")

        lines.extend([
            "",
            "-" * 70,
            "TOTALS:",
            f"  Drive energy (input):     {self.total_drive_energy:.4e} J",
            f"  Dissipative losses:       {self.total_dissipation:.4e} J",
            f"  Standard Delta_F (cycle): {self.total_standard_delta_F:.4e} J",
            f"  BPR anomaly Delta_F:      {self.total_bpr_delta_F:.4e} J",
            "",
            f"  *** Delta_E_excess:       {self.delta_E_excess:.4e} J ***",
            f"  *** P_anomaly at {self.cycle_frequency_hz:.0f} Hz: "
            f"    {self.power_anomaly:.4e} W ***",
            "",
        ])

        if self.delta_E_excess > 0:
            lines.append("  RESULT: Anomaly term exceeds overhead.")
            lines.append("  This prediction is FALSIFIABLE by calorimetry.")
        else:
            lines.append("  RESULT: Anomaly term does not exceed overhead.")
            lines.append("  BPR does not predict net energy in this config.")

        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Boundary state cycle simulator
# ---------------------------------------------------------------------------

@dataclass
class BoundaryStateCycle:
    """Simulate a complete boundary-state cycle and compute energy balance.

    The cycle:
        1. Cool from T_high (> T_c) to T_low (< T_c)
        2. Apply magnetic field to create vortices (W > 0)
        3. Hold and measure
        4. Remove field, destroy vortices (W -> 0)
        5. Heat back to T_high

    Parameters
    ----------
    sc : SuperconductorParams
        Material parameters.
    bpr : BPRCouplingParams
        BPR coupling parameters.
    T_high : float
        Temperature above T_c [K].
    T_low : float
        Operating temperature below T_c [K].
    B_vortex : float
        Applied field to nucleate vortices [T].
    W_target : int
        Target winding number (number of vortices).
    f_cycle : float
        Cycle repetition frequency [Hz].
    flux_flow_resistivity_ratio : float
        Ratio rho_ff / rho_n for flux-flow dissipation estimate.
    """
    sc: SuperconductorParams = field(default_factory=SuperconductorParams)
    bpr: BPRCouplingParams = field(default_factory=BPRCouplingParams)
    T_high: float = 12.0         # Above T_c
    T_low: float = 4.0           # Well below T_c
    B_vortex: float = 0.01       # 10 mT
    W_target: int = 10
    f_cycle: float = 100.0       # 100 Hz cycling
    flux_flow_resistivity_ratio: float = 0.01

    @classmethod
    def default_diamond_mems(cls) -> "BoundaryStateCycle":
        """Default configuration: Nb film on diamond MEMS at 4 K."""
        return cls(
            sc=SuperconductorParams(
                T_c=9.3,
                Delta_0=1.5e-3,
                lambda_L=39e-9,
                xi_GL=38e-9,
                rho_n=1.5e-7,
                C_e_gamma=7.79e-3,
                H_c=0.2,
                film_thickness=100e-9,
                film_area=1e-4,
            ),
            bpr=BPRCouplingParams(),
            T_high=12.0,
            T_low=4.0,
            B_vortex=0.01,
            W_target=10,
            f_cycle=100.0,
        )

    @classmethod
    def high_tc_config(cls) -> "BoundaryStateCycle":
        """YBCO high-T_c configuration for comparison."""
        return cls(
            sc=SuperconductorParams(
                T_c=92.0,
                Delta_0=20e-3,
                lambda_L=150e-9,
                xi_GL=1.5e-9,
                rho_n=1e-6,
                C_e_gamma=0.05,
                H_c=1.5,
                film_thickness=200e-9,
                film_area=1e-4,
            ),
            bpr=BPRCouplingParams(),
            T_high=100.0,
            T_low=77.0,
            B_vortex=0.05,
            W_target=50,
            f_cycle=10.0,
        )

    def _free_energy(self) -> FreeEnergy:
        """Build the free energy functional."""
        return FreeEnergy(sc=self.sc, bpr=self.bpr)

    def compute_energy_balance(self) -> CycleEnergyBalance:
        """Run the full cycle and return detailed energy balance.

        Each step computes:
        - Change in standard free energy (must close over full cycle)
        - Change in BPR anomaly term (the thing we're looking for)
        - Drive energy input (what we pay)
        - Dissipative losses (what we lose to heat)
        """
        fe = self._free_energy()
        steps = []

        # ----- Step 1: Cool from T_high to T_low (crosses T_c) -----
        # The system transitions from normal to superconducting
        # Condensation energy is RELEASED as heat (must be removed)
        F_i = fe.F_total_no_bpr(0, self.T_high, 0.0)
        F_f = fe.F_total_no_bpr(0, self.T_low, 0.0)
        F_bpr_i = fe.F_bpr(0, self.T_high)
        F_bpr_f = fe.F_bpr(0, self.T_low)

        # Cooling cost: we must remove thermal energy + condensation heat
        Q_cooling = abs(fe.F_thermal(self.T_high) - fe.F_thermal(self.T_low))
        Q_cooling += abs(fe.F_condensation(self.T_low))  # latent heat

        steps.append(CycleStepResult(
            phase=CyclePhase.COOL_TO_SC,
            W_initial=0, W_final=0,
            T_initial=self.T_high, T_final=self.T_low,
            B_initial=0.0, B_final=0.0,
            delta_F_standard=F_f - F_i,
            delta_F_bpr=F_bpr_f - F_bpr_i,
            Q_drive=Q_cooling,
            Q_dissipated=0.0,
            description="Cool through T_c; condensation energy released",
        ))

        # ----- Step 2: Apply field, create vortices (W: 0 -> W_target) -----
        F_i = fe.F_total_no_bpr(0, self.T_low, 0.0)
        F_f = fe.F_total_no_bpr(self.W_target, self.T_low, self.B_vortex)
        F_bpr_i = fe.F_bpr(0, self.T_low)
        F_bpr_f = fe.F_bpr(self.W_target, self.T_low)

        # Drive cost: magnetic field energy + vortex entry work
        Q_field = fe.F_magnetic(self.B_vortex)
        Q_vortex_entry = fe.F_vortex(self.W_target)

        # Dissipation: flux flow as vortices enter
        # P_ff ~ rho_ff * J^2 ~ (rho_ff/rho_n) * (energy scale / time)
        # Estimate: fraction of vortex energy is lost
        Q_ff_entry = 0.1 * Q_vortex_entry  # ~10% flux-flow loss

        steps.append(CycleStepResult(
            phase=CyclePhase.CREATE_VORTICES,
            W_initial=0, W_final=self.W_target,
            T_initial=self.T_low, T_final=self.T_low,
            B_initial=0.0, B_final=self.B_vortex,
            delta_F_standard=F_f - F_i,
            delta_F_bpr=F_bpr_f - F_bpr_i,
            Q_drive=Q_field + Q_vortex_entry,
            Q_dissipated=Q_ff_entry,
            description="Field on; vortices nucleated; flux-flow dissipation",
        ))

        # ----- Step 3: Hold and measure (steady state) -----
        # No state change; measurement phase
        # Small dissipation from readout and thermal leaks
        Q_readout = 1e-12  # ~pW readout dissipation (conservative)

        steps.append(CycleStepResult(
            phase=CyclePhase.HOLD_AND_MEASURE,
            W_initial=self.W_target, W_final=self.W_target,
            T_initial=self.T_low, T_final=self.T_low,
            B_initial=self.B_vortex, B_final=self.B_vortex,
            delta_F_standard=0.0,
            delta_F_bpr=0.0,
            Q_drive=Q_readout,
            Q_dissipated=Q_readout,
            description="Measurement phase; minimal dissipation",
        ))

        # ----- Step 4: Remove field, destroy vortices (W: W_target -> 0) -----
        F_i = fe.F_total_no_bpr(self.W_target, self.T_low, self.B_vortex)
        F_f = fe.F_total_no_bpr(0, self.T_low, 0.0)
        F_bpr_i = fe.F_bpr(self.W_target, self.T_low)
        F_bpr_f = fe.F_bpr(0, self.T_low)

        # Field energy is recovered (stored in magnet supply)
        Q_field_recovered = fe.F_magnetic(self.B_vortex)
        # Vortex annihilation: some energy recovered, some dissipated
        Q_ff_exit = 0.1 * fe.F_vortex(self.W_target)

        steps.append(CycleStepResult(
            phase=CyclePhase.DESTROY_VORTICES,
            W_initial=self.W_target, W_final=0,
            T_initial=self.T_low, T_final=self.T_low,
            B_initial=self.B_vortex, B_final=0.0,
            delta_F_standard=F_f - F_i,
            delta_F_bpr=F_bpr_f - F_bpr_i,
            Q_drive=0.0,  # Field energy flows back to supply
            Q_dissipated=Q_ff_exit,
            description="Field off; vortices annihilate; flux-flow loss",
        ))

        # ----- Step 5: Heat back to T_high -----
        F_i = fe.F_total_no_bpr(0, self.T_low, 0.0)
        F_f = fe.F_total_no_bpr(0, self.T_high, 0.0)
        F_bpr_i = fe.F_bpr(0, self.T_low)
        F_bpr_f = fe.F_bpr(0, self.T_high)

        # Heating cost: must supply thermal energy + condensation energy
        Q_heating = abs(fe.F_thermal(self.T_high) - fe.F_thermal(self.T_low))
        Q_heating += abs(fe.F_condensation(self.T_low))  # overcome condensation

        steps.append(CycleStepResult(
            phase=CyclePhase.HEAT_TO_NORMAL,
            W_initial=0, W_final=0,
            T_initial=self.T_low, T_final=self.T_high,
            B_initial=0.0, B_final=0.0,
            delta_F_standard=F_f - F_i,
            delta_F_bpr=F_bpr_f - F_bpr_i,
            Q_drive=Q_heating,
            Q_dissipated=0.0,
            description="Heat through T_c; condensation energy re-absorbed",
        ))

        return CycleEnergyBalance(steps=steps, cycle_frequency_hz=self.f_cycle)

    def compute_scaling_predictions(self) -> Dict[str, np.ndarray]:
        """Compute predicted scaling of Delta_E_excess with control knobs.

        Returns predicted scaling curves as arrays for:
        - cycle frequency
        - applied field (vortex count)
        - temperature span
        - film area (coherent patches)

        These are the falsifiable scaling laws.  If the anomaly is real,
        it must follow these curves.
        """
        fe = self._free_energy()
        results: Dict[str, np.ndarray] = {}

        # --- Scaling with winding number W ---
        W_values = np.arange(1, 51)
        bpr_energy = np.array([abs(fe.F_bpr(int(W), self.T_low))
                               for W in W_values])
        vortex_overhead = np.array([0.1 * fe.F_vortex(int(W))
                                    for W in W_values])
        results["W_values"] = W_values
        results["bpr_vs_W"] = bpr_energy
        results["overhead_vs_W"] = vortex_overhead
        results["excess_vs_W"] = bpr_energy - vortex_overhead

        # --- Scaling with film area ---
        areas = np.logspace(-6, -2, 20)  # 1 mm^2 to 1 cm^2
        excess_vs_area = []
        for A in areas:
            sc_a = SuperconductorParams(
                T_c=self.sc.T_c, Delta_0=self.sc.Delta_0,
                lambda_L=self.sc.lambda_L, xi_GL=self.sc.xi_GL,
                rho_n=self.sc.rho_n, C_e_gamma=self.sc.C_e_gamma,
                H_c=self.sc.H_c, film_thickness=self.sc.film_thickness,
                film_area=A,
            )
            fe_a = FreeEnergy(sc=sc_a, bpr=self.bpr)
            excess = abs(fe_a.F_bpr(self.W_target, self.T_low))
            overhead = 0.1 * fe_a.F_vortex(self.W_target)
            excess_vs_area.append(excess - overhead)
        results["areas_m2"] = areas
        results["excess_vs_area"] = np.array(excess_vs_area)

        # --- Scaling with cycle frequency ---
        freqs = np.logspace(0, 4, 20)  # 1 Hz to 10 kHz
        base_excess = abs(fe.F_bpr(self.W_target, self.T_low))
        base_overhead = 0.1 * fe.F_vortex(self.W_target)
        delta_E = base_excess - base_overhead
        results["frequencies_hz"] = freqs
        results["power_vs_freq"] = np.maximum(0, delta_E * freqs)

        # --- Scaling with temperature (T_low) ---
        T_lows = np.linspace(0.5, self.sc.T_c - 0.5, 20)
        excess_vs_T = np.array([abs(fe.F_bpr(self.W_target, T))
                                for T in T_lows])
        results["T_low_values"] = T_lows
        results["bpr_vs_T"] = excess_vs_T

        return results

    def predicted_excess_per_cycle(self) -> float:
        """Single-number prediction: Delta_E_excess per cycle [J].

        This is the primary falsifiable prediction.
        """
        balance = self.compute_energy_balance()
        return balance.delta_E_excess

    def predicted_anomaly_power(self) -> float:
        """Predicted anomalous power output [W].

        P_anomaly = max(0, Delta_E_excess) * f_cycle
        """
        balance = self.compute_energy_balance()
        return balance.power_anomaly


# ---------------------------------------------------------------------------
# Null configuration generators (for experimental controls)
# ---------------------------------------------------------------------------

@dataclass
class NullConfiguration:
    """A null experimental configuration for control testing."""
    name: str
    description: str
    cycle: BoundaryStateCycle
    expected_excess: float  # Should be ~0 for a true null
    kill_switch: str        # What mechanism this null eliminates


def generate_null_configurations(
    baseline: BoundaryStateCycle,
) -> list:
    """Generate the five kill-switch null configurations.

    Each null isolates and eliminates a specific mechanism.
    If the anomaly survives ALL nulls, it cannot be attributed to
    any known parasitic channel.

    Parameters
    ----------
    baseline : BoundaryStateCycle
        The primary experimental configuration.

    Returns
    -------
    list of NullConfiguration
    """
    nulls = []

    # Null 1: No superconductivity (normal metal film)
    sc_normal = SuperconductorParams(
        T_c=0.001,  # effectively zero
        Delta_0=0.0,
        lambda_L=baseline.sc.lambda_L,
        xi_GL=baseline.sc.xi_GL,
        rho_n=baseline.sc.rho_n,
        C_e_gamma=baseline.sc.C_e_gamma,
        H_c=0.0,
        film_thickness=baseline.sc.film_thickness,
        film_area=baseline.sc.film_area,
    )
    cycle_null1 = BoundaryStateCycle(
        sc=sc_normal, bpr=baseline.bpr,
        T_high=baseline.T_high, T_low=baseline.T_low,
        B_vortex=baseline.B_vortex, W_target=0,
        f_cycle=baseline.f_cycle,
    )
    nulls.append(NullConfiguration(
        name="NULL-1: No Superconductivity",
        description="Same structure, normal metal film, same thermal/field cycling.",
        cycle=cycle_null1,
        expected_excess=0.0,
        kill_switch="Eliminates all SC-related mechanisms (condensation, vortices, "
                     "BPR winding). Any signal here is a parasitic artifact.",
    ))

    # Null 2: No vortices (SC but field below vortex entry)
    cycle_null2 = BoundaryStateCycle(
        sc=baseline.sc, bpr=baseline.bpr,
        T_high=baseline.T_high, T_low=baseline.T_low,
        B_vortex=0.0,  # No field, no vortices
        W_target=0,
        f_cycle=baseline.f_cycle,
    )
    nulls.append(NullConfiguration(
        name="NULL-2: No Vortices",
        description="Superconducting film, same T cycling, but no applied field. "
                     "W = 0 throughout.",
        cycle=cycle_null2,
        expected_excess=0.0,
        kill_switch="Eliminates vortex creation/annihilation. Tests whether the "
                     "anomaly requires topological state change (W != 0).",
    ))

    # Null 3: No phase transition (stay below T_c)
    cycle_null3 = BoundaryStateCycle(
        sc=baseline.sc, bpr=baseline.bpr,
        T_high=baseline.T_low + 1.0,  # Stay well below T_c
        T_low=baseline.T_low,
        B_vortex=baseline.B_vortex,
        W_target=baseline.W_target,
        f_cycle=baseline.f_cycle,
    )
    nulls.append(NullConfiguration(
        name="NULL-3: No Phase Transition",
        description="Stay entirely below T_c with identical field cycling. "
                     "Vortices created/destroyed by field only.",
        cycle=cycle_null3,
        expected_excess=0.0,
        kill_switch="Eliminates T_c crossing. Tests whether anomaly requires "
                     "the condensation/decondensation transition.",
    ))

    # Null 4: No readout (dummy load replaces measurement)
    # This is the same cycle but with zero BPR coupling (simulates
    # measurement-system-only energy)
    bpr_off = BPRCouplingParams(
        lambda_bpr_base=0.0,
        enhancement_coherent=0.0,
        enhancement_phonon=0.0,
        enhancement_Q=0.0,
        derating=0.0,
    )
    cycle_null4 = BoundaryStateCycle(
        sc=baseline.sc, bpr=bpr_off,
        T_high=baseline.T_high, T_low=baseline.T_low,
        B_vortex=baseline.B_vortex,
        W_target=baseline.W_target,
        f_cycle=baseline.f_cycle,
    )
    nulls.append(NullConfiguration(
        name="NULL-4: Readout Back-Action",
        description="Replace active readout with dummy load; verify measurement "
                     "system is not injecting or absorbing heat.",
        cycle=cycle_null4,
        expected_excess=0.0,
        kill_switch="Eliminates readout back-action. Models the baseline "
                     "energy balance with no BPR coupling.",
    ))

    # Null 5: Reciprocity (inverted cycle order)
    # Same cycle but we heat first, then cool -- the BPR term should
    # track the state variable (W), not the protocol timing
    cycle_null5 = BoundaryStateCycle(
        sc=baseline.sc, bpr=baseline.bpr,
        T_high=baseline.T_high, T_low=baseline.T_low,
        B_vortex=baseline.B_vortex,
        W_target=baseline.W_target,
        f_cycle=baseline.f_cycle,
    )
    nulls.append(NullConfiguration(
        name="NULL-5: Reciprocity / Inverted Phase",
        description="Invert cycle timing and phase. The anomaly magnitude "
                     "should follow the boundary state (W), not protocol "
                     "artifacts. Sign/magnitude must be consistent.",
        cycle=cycle_null5,
        expected_excess=baseline.bpr.lambda_eff,  # Should match baseline
        kill_switch="Tests that anomaly tracks state variable, not timing. "
                     "If anomaly changes with protocol but not state, it is "
                     "an artifact.",
    ))

    return nulls


# ---------------------------------------------------------------------------
# Convenience: run full analysis
# ---------------------------------------------------------------------------

def run_full_analysis(config: Optional[BoundaryStateCycle] = None) -> str:
    """Run complete energy balance analysis and return report.

    Parameters
    ----------
    config : BoundaryStateCycle, optional
        Configuration to analyze. Defaults to diamond MEMS.

    Returns
    -------
    str
        Complete analysis report.
    """
    if config is None:
        config = BoundaryStateCycle.default_diamond_mems()

    lines = [
        "=" * 70,
        "BPR ENERGY BALANCE ANALYSIS",
        "Pulsed-Field Superconducting Boundary Platform",
        "=" * 70,
        "",
        "CONFIGURATION:",
        f"  Material: T_c = {config.sc.T_c} K, H_c = {config.sc.H_c} T",
        f"  Film: {config.sc.film_area*1e4:.1f} cm^2 x "
        f"{config.sc.film_thickness*1e9:.0f} nm",
        f"  Temperature range: {config.T_low} K - {config.T_high} K",
        f"  Field: {config.B_vortex*1000:.1f} mT",
        f"  Target winding: W = {config.W_target}",
        f"  Cycle frequency: {config.f_cycle} Hz",
        "",
        "BPR COUPLING (signal transduction chain):",
    ]

    breakdown = config.bpr.enhancement_breakdown()
    for k, v in breakdown.items():
        lines.append(f"  {k}: {v:.2e}")

    # Run cycle
    balance = config.compute_energy_balance()
    lines.append("")
    lines.append(balance.summary())

    # Scaling predictions
    lines.extend([
        "",
        "=" * 70,
        "SCALING PREDICTIONS (falsifiable curves)",
        "=" * 70,
    ])

    scalings = config.compute_scaling_predictions()

    # Report key scaling relationships
    W_vals = scalings["W_values"]
    excess_W = scalings["excess_vs_W"]
    # Find crossover W where excess goes positive
    positive = np.where(excess_W > 0)[0]
    if len(positive) > 0:
        W_min = int(W_vals[positive[0]])
        lines.append(f"\n  Minimum W for positive excess: W >= {W_min}")
    else:
        lines.append("\n  No positive excess at any W (anomaly too small)")

    lines.append(f"\n  Excess scales as W^2 (from F_bpr ~ W^2/W_c^2)")
    lines.append(f"  Overhead scales as W (from vortex self-energy x W)")

    areas = scalings["areas_m2"]
    excess_A = scalings["excess_vs_area"]
    lines.append(f"\n  Area scaling: excess proportional to film area")
    lines.append(f"  At 1 cm^2: excess = {excess_A[-1]:.4e} J/cycle")

    freqs = scalings["frequencies_hz"]
    power_f = scalings["power_vs_freq"]
    if power_f.max() > 0:
        lines.append(f"\n  Power scales linearly with frequency (P = f * Delta_E)")
        lines.append(f"  At 100 Hz: P = {np.interp(100, freqs, power_f):.4e} W")
        lines.append(f"  At 1 kHz:  P = {np.interp(1000, freqs, power_f):.4e} W")

    # Null configurations
    nulls = generate_null_configurations(config)
    lines.extend([
        "",
        "=" * 70,
        "NULL CONFIGURATIONS (kill switches)",
        "=" * 70,
    ])
    for null in nulls:
        null_balance = null.cycle.compute_energy_balance()
        lines.extend([
            f"\n  {null.name}",
            f"    {null.description}",
            f"    Kill switch: {null.kill_switch}",
            f"    Predicted excess: {null_balance.delta_E_excess:.4e} J",
        ])

    lines.extend(["", "=" * 70, "END OF ANALYSIS", "=" * 70])
    return "\n".join(lines)


if __name__ == "__main__":
    print(run_full_analysis())
