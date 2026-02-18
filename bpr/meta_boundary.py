"""
Theory XXIII: Meta-Boundary Dynamics and Global Phase Reindexing
=================================================================

Extends BPR to include dynamics on the space of admissible boundary
constraints.  The constraint field kappa(x,t) is promoted from a fixed
parameter to a dynamical variable governed by a variational principle.

Key objects
-----------
* ``ConstraintField``            -- state triple (b, m, kappa)          (Def 1.1)
* ``BoundaryPhaseAction``        -- phase action S_Phi                  (Eq 7-9)
* ``GaugeStructure``             -- boundary phase curvature Omega_ABC  (Def 3.1)
* ``MetaFunctional``             -- J[kappa; b, m]                      (Def 4.1)
* ``MetaBoundaryEvolution``      -- reaction-diffusion PDE              (Eq 27)
* ``ConstraintPotential``        -- double/multi-well V_kappa           (Eq 31,35)
* ``MetaRewriteEligibility``     -- E2[kappa; b, m]                     (Eq 36-37)
* ``FrontSolution``              -- static kink & moving front          (Eq 43,46)
* ``EnergyCostScaling``          -- domain wall energy & critical nucleus (Eq 51-56)
* ``SpectralDrift``              -- eigenvalue shift from transition     (Eq 60-61)
* ``DetectabilityTheorem``       -- no-go for undetectable rewrites     (Thm 10.1)
* ``Hysteresis``                 -- Berry connection & path dependence  (Eq 64-67)
* ``ThermodynamicSignatures``    -- entropy production & dissipation    (Eq 68-71)
* ``StabilityAnalysis``          -- linear & Lyapunov stability         (Eq 73-78)
* ``CERNScenario``               -- energy budget analysis              (Sec 16)

References: Al-Kahwati (2026), *Meta-Boundary Dynamics and Global Phase
Reindexing in Boundary Phase Rewrite Frameworks*, StarDrive Research Group.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable, Union
from scipy.integrate import solve_ivp
from scipy.special import gamma as gamma_func


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_HBAR = 1.054571817e-34   # J s
_E_PLANCK = 1.22e19 * 1.6e-10  # Planck energy in Joules (~1.22e19 GeV)
_L_PLANCK = 1.616e-35     # Planck length in meters


# ===================================================================
# Section 1 (Def 1.1-1.4): Constraint Field and State Triple
# ===================================================================

@dataclass
class ConstraintField:
    r"""State triple (b, m, kappa) for meta-boundary dynamics (Def 1.1).

    The system is described by:
        b(x,t) in B : boundary configuration (phase field on dOmega)
        m(x,t) in M : Cache state (latent information substrate)
        kappa(x,t) in K : constraint field

    The constraint field parametrizes the family of boundary constraint
    operators C_kappa(b) = 0 on dOmega.

    Parameters
    ----------
    kappa : ndarray
        Constraint field values on the spatial grid.
    b : ndarray, optional
        Boundary configuration field.
    m : ndarray, optional
        Cache state field.
    x : ndarray, optional
        Spatial grid points.
    """
    kappa: np.ndarray
    b: Optional[np.ndarray] = None
    m: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None

    @staticmethod
    def winding_number(phi: np.ndarray) -> int:
        r"""Compute winding number for a closed loop (Def 1.4).

        W[Gamma] = (1/2pi) * integral dPhi_AB

        Parameters
        ----------
        phi : ndarray
            Phase values along a closed loop.

        Returns
        -------
        int
            Winding number W.
        """
        dphi = np.diff(phi)
        # Wrap to [-pi, pi]
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        return int(np.round(np.sum(dphi) / (2.0 * np.pi)))


# ===================================================================
# Section 2 (Eq 7-13): Boundary Phase Action
# ===================================================================

@dataclass
class BoundaryPhaseAction:
    r"""Boundary phase action and Euler-Lagrange dynamics (Eq 7-9).

    Action:
        S_Phi[rho] = int dt sum_{A,B} [(d_t Phi_AB)^2 - kappa_s |nabla Phi_AB|^2
                       - mu_Phi V_Phi(Phi_AB)]

    Euler-Lagrange equation (Eq 8):
        d^2_t Phi_AB - kappa_s nabla^2 Phi_AB + (mu_Phi/2) V'_Phi(Phi_AB) = 0

    Dispersion relation (Eq 9):
        omega^2 = kappa_s k^2 + mu_Phi m^2_Phi / 2

    Parameters
    ----------
    kappa_s : float
        Phase stiffness parameter (controls propagation speed).
    mu_Phi : float
        Phase potential coupling.
    m_Phi_sq : float
        Linearized potential curvature V''_Phi.
    """
    kappa_s: float = 1.0
    mu_Phi: float = 1.0
    m_Phi_sq: float = 1.0

    def dispersion(self, k: np.ndarray) -> np.ndarray:
        r"""Dispersion relation omega(k) for phase excitations (Eq 9).

        omega^2 = kappa_s * k^2 + mu_Phi * m_Phi^2 / 2

        Parameters
        ----------
        k : ndarray
            Wavevector magnitudes.

        Returns
        -------
        ndarray
            Angular frequencies omega(k).
        """
        k = np.asarray(k, dtype=float)
        omega_sq = self.kappa_s * k**2 + self.mu_Phi * self.m_Phi_sq / 2.0
        return np.sqrt(np.maximum(omega_sq, 0.0))

    def phase_gap(self) -> float:
        r"""Compute the phase mass gap omega_0 (below Eq 9).

        omega_0 = sqrt(mu_Phi * m_Phi^2 / 2)

        Returns
        -------
        float
            Minimum frequency of phase excitations.
        """
        return np.sqrt(self.mu_Phi * self.m_Phi_sq / 2.0)

    def decoherence_rate(self, grad_phi_sq: float) -> float:
        r"""Decoherence rate from phase gradients (Eq 13).

        Gamma_dec = (kappa_s / hbar) * ||nabla Phi_AB||^2

        Parameters
        ----------
        grad_phi_sq : float
            Squared norm of boundary phase gradient.

        Returns
        -------
        float
            Decoherence rate (in natural units where hbar=1).
        """
        return self.kappa_s * grad_phi_sq

    @staticmethod
    def geometric_decoherence_ratio(G1: float, G2: float) -> float:
        r"""Ratio of decoherence rates for two geometries (Eq 82).

        Gamma_1 / Gamma_2 = G_1 / G_2

        Standard QM predicts ratio = 1 for identical noise spectra.

        Parameters
        ----------
        G1, G2 : float
            Boundary phase gradients squared for configs 1 and 2.

        Returns
        -------
        float
            Decoherence rate ratio.
        """
        if G2 <= 0:
            return np.inf if G1 > 0 else 1.0
        return G1 / G2

    def phase_energy(self, grad_phi_sq: np.ndarray, dx: float = 1.0) -> float:
        r"""Boundary phase energy E_Phi (Eq 12).

        E_Phi = sum kappa_s |nabla Phi_AB|^2 * dx

        Parameters
        ----------
        grad_phi_sq : ndarray
            Squared phase gradients at each grid point.
        dx : float
            Grid spacing.

        Returns
        -------
        float
            Total boundary phase energy.
        """
        return float(self.kappa_s * np.sum(grad_phi_sq) * dx)


# ===================================================================
# Section 3 (Def 3.1, Thm 3.2): Gauge Structure
# ===================================================================

@dataclass
class GaugeStructure:
    r"""Gauge structure of boundary phase operators (Section 3).

    The boundary phase Phi_AB = Arg Tr(rho Pi_A Pi_B) transforms under
    local phase rotations as Phi_AB -> Phi_AB + theta_A + theta_B.

    The gauge-invariant boundary phase curvature (Def 3.1):
        Omega_ABC = Phi_AB + Phi_BC + Phi_CA

    is invariant under all local phase rotations (Thm 3.2).
    """

    @staticmethod
    def boundary_phase(rho_trace: complex) -> float:
        r"""Compute boundary phase Phi_AB (Def 1.3).

        Phi_AB = Arg Tr(rho Pi_A Pi_B)

        Parameters
        ----------
        rho_trace : complex
            Value of Tr(rho Pi_A Pi_B).

        Returns
        -------
        float
            Boundary phase in (-pi, pi].
        """
        return float(np.angle(rho_trace))

    @staticmethod
    def gauge_transform(phi_AB: float, theta_A: float, theta_B: float) -> float:
        r"""Apply gauge transformation (Eq 15).

        Phi_AB -> Phi_AB + theta_A + theta_B

        Parameters
        ----------
        phi_AB : float
            Original boundary phase.
        theta_A, theta_B : float
            Local phase rotation angles.

        Returns
        -------
        float
            Transformed boundary phase.
        """
        return phi_AB + theta_A + theta_B

    @staticmethod
    def boundary_phase_curvature(phi_AB: float, phi_BC: float,
                                  phi_CA: float) -> float:
        r"""Gauge-invariant boundary phase curvature (Def 3.1, Eq 17).

        Omega_ABC = Phi_AB + Phi_BC + Phi_CA

        Parameters
        ----------
        phi_AB, phi_BC, phi_CA : float
            Boundary phases between pairs of subsystems.

        Returns
        -------
        float
            Boundary phase curvature Omega_ABC.
        """
        return phi_AB + phi_BC + phi_CA

    @staticmethod
    def gauge_transform_curvature(omega_ABC: float,
                                   theta_A: float, theta_B: float,
                                   theta_C: float) -> float:
        r"""Compute gauge-transformed curvature (from Thm 3.2 proof).

        Under rotations theta_A, theta_B, theta_C:
            Omega' = Omega + 2(theta_A + theta_B + theta_C)

        Parameters
        ----------
        omega_ABC : float
            Original curvature.
        theta_A, theta_B, theta_C : float
            Gauge rotation angles.

        Returns
        -------
        float
            Transformed curvature.
        """
        return omega_ABC + 2.0 * (theta_A + theta_B + theta_C)

    @staticmethod
    def gauge_invariant_curvature_exp(phi_AB: float, phi_BC: float,
                                       phi_CA: float) -> complex:
        r"""Gauge-invariant curvature observable (Eq 19).

        Omega_ABC = exp(i * (Phi_AB + Phi_BC + Phi_CA))

        This is manifestly independent of local phase choices when
        built from normalized trace ratios (Eq 19).

        Parameters
        ----------
        phi_AB, phi_BC, phi_CA : float
            Boundary phases.

        Returns
        -------
        complex
            Unit-modulus gauge-invariant curvature observable.
        """
        omega = phi_AB + phi_BC + phi_CA
        return np.exp(1j * omega)

    @staticmethod
    def curvature_two_form(phi: np.ndarray, dx: float = 1.0) -> np.ndarray:
        r"""Compute curvature two-form F_ij (Eq 22).

        F_ij = d_i A_j - d_j A_i

        where A_i = d Phi_AB / dx_i.

        For a 1D field, F = 0. For 2D, returns scalar curvature.

        Parameters
        ----------
        phi : ndarray
            Phase field (1D or 2D).
        dx : float
            Grid spacing.

        Returns
        -------
        ndarray
            Curvature values. Zero for 1D fields.
        """
        phi = np.asarray(phi, dtype=float)
        if phi.ndim == 1:
            return np.zeros_like(phi)
        elif phi.ndim == 2:
            # F_12 = d_1 A_2 - d_2 A_1 = d_1 d_2 phi - d_2 d_1 phi
            # For smooth fields this is zero, but discretization can
            # introduce nonzero curvature
            A_x = np.gradient(phi, dx, axis=0)
            A_y = np.gradient(phi, dx, axis=1)
            F = np.gradient(A_y, dx, axis=0) - np.gradient(A_x, dx, axis=1)
            return F
        else:
            raise ValueError(f"Phase field must be 1D or 2D, got {phi.ndim}D")


# ===================================================================
# Section 5 (Eq 31-35): Constraint Potential
# ===================================================================

@dataclass
class ConstraintPotential:
    r"""Constraint potential with phase structure (Eq 31, 35).

    Double-well (Eq 31):
        V_kappa(kappa) = (lambda_kappa / 4) * (kappa^2 - eta^2)^2

    with minima at kappa = +/- eta.

    Multi-well (Eq 35):
        V_kappa(kappa) = prod_{a=1}^{n} (kappa - kappa_a)^2 + delta * P(kappa)

    Parameters
    ----------
    lambda_kappa : float
        Quartic coupling constant.
    eta : float
        Phase separation parameter (minima at +/- eta for double-well).
    alpha : float
        Gradient stiffness coefficient.
    beta : float
        Potential scale coefficient.
    """
    lambda_kappa: float = 1.0
    eta: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0

    def double_well(self, kappa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Evaluate double-well potential (Eq 31).

        V(kappa) = (lambda_kappa / 4) * (kappa^2 - eta^2)^2

        Parameters
        ----------
        kappa : float or ndarray
            Constraint field value(s).

        Returns
        -------
        float or ndarray
            Potential values.
        """
        return self.lambda_kappa / 4.0 * (kappa**2 - self.eta**2)**2

    def double_well_derivative(self, kappa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""First derivative of double-well potential.

        V'(kappa) = lambda_kappa * kappa * (kappa^2 - eta^2)

        Parameters
        ----------
        kappa : float or ndarray
            Constraint field value(s).

        Returns
        -------
        float or ndarray
            Potential derivative.
        """
        return self.lambda_kappa * kappa * (kappa**2 - self.eta**2)

    def double_well_second_derivative(self, kappa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Second derivative of double-well potential.

        V''(kappa) = lambda_kappa * (3 kappa^2 - eta^2)

        Parameters
        ----------
        kappa : float or ndarray
            Constraint field value(s).

        Returns
        -------
        float or ndarray
            Potential second derivative.
        """
        return self.lambda_kappa * (3.0 * kappa**2 - self.eta**2)

    def barrier_height(self) -> float:
        r"""Barrier height of double-well (Eq 41).

        Delta_V = V(0) - V(+/- eta) = lambda_kappa * eta^4 / 4

        Returns
        -------
        float
            Energy barrier between constraint phases.
        """
        return self.lambda_kappa * self.eta**4 / 4.0

    def mass_gap(self) -> float:
        r"""Mass gap of constraint fluctuations at a minimum (Eq 33-34).

        m_kappa = sqrt(2 * beta * lambda_kappa / alpha) * eta

        Returns
        -------
        float
            Constraint field mass gap.
        """
        return np.sqrt(2.0 * self.beta * self.lambda_kappa / self.alpha) * self.eta

    def domain_wall_width(self) -> float:
        r"""Domain wall width for the kink solution (Eq 44).

        ell_kappa = 1 / (eta * sqrt(beta * lambda_kappa / (2 * alpha)))

        Returns
        -------
        float
            Characteristic width of the domain wall.
        """
        return 1.0 / (self.eta * np.sqrt(self.beta * self.lambda_kappa / (2.0 * self.alpha)))

    @staticmethod
    def multi_well(kappa: Union[float, np.ndarray],
                   kappa_phases: List[float],
                   delta: float = 0.0) -> Union[float, np.ndarray]:
        r"""Evaluate multi-well potential (Eq 35).

        V(kappa) = prod_{a=1}^{n} (kappa - kappa_a)^2 + delta * P(kappa)

        where P(kappa) = kappa (simple perturbation).

        Parameters
        ----------
        kappa : float or ndarray
            Constraint field value(s).
        kappa_phases : list of float
            Positions of the n potential minima.
        delta : float
            Degeneracy-lifting perturbation strength.

        Returns
        -------
        float or ndarray
            Potential values.
        """
        result = np.ones_like(np.asarray(kappa, dtype=float))
        for kappa_a in kappa_phases:
            result = result * (np.asarray(kappa) - kappa_a)**2
        if delta != 0:
            result = result + delta * np.asarray(kappa)
        return result


# ===================================================================
# Section 4 (Def 4.1, Eq 24-29): Meta-Functional
# ===================================================================

@dataclass
class MetaFunctional:
    r"""Constraint frustration functional J[kappa; b, m] (Def 4.1, Eq 24).

    J[kappa; b, m] = integral [ (alpha/2) ||nabla kappa||^2
                                + beta * V_kappa(kappa)
                                + gamma * W(kappa, b, m) ] d^d x

    where:
        alpha/2 ||nabla kappa||^2 : gradient energy
        beta * V_kappa             : constraint potential
        gamma * W                  : coupling to boundary stress

    Parameters
    ----------
    alpha : float
        Gradient stiffness coefficient.
    beta : float
        Potential scale coefficient.
    gamma : float
        Stress-constraint coupling.
    potential : ConstraintPotential
        The constraint potential function.
    """
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.1
    potential: ConstraintPotential = field(default_factory=ConstraintPotential)

    def evaluate(self, kappa: np.ndarray, dx: float = 1.0,
                 sigma_frust: Optional[np.ndarray] = None,
                 chi_kappa: Optional[np.ndarray] = None) -> float:
        r"""Evaluate the meta-functional J[kappa; b, m].

        Parameters
        ----------
        kappa : ndarray
            Constraint field on spatial grid.
        dx : float
            Grid spacing.
        sigma_frust : ndarray, optional
            Boundary frustration stress. Zero if not provided.
        chi_kappa : ndarray, optional
            Constraint-dependent susceptibility. Ones if not provided.

        Returns
        -------
        float
            Value of J[kappa].
        """
        kappa = np.asarray(kappa, dtype=float)

        # Gradient energy: (alpha/2) ||nabla kappa||^2
        grad_kappa = np.gradient(kappa, dx)
        if isinstance(grad_kappa, list):
            grad_sq = sum(g**2 for g in grad_kappa)
        else:
            grad_sq = grad_kappa**2
        gradient_energy = self.alpha / 2.0 * np.sum(grad_sq) * dx

        # Potential energy: beta * V_kappa
        V = self.potential.double_well(kappa)
        potential_energy = self.beta * np.sum(V) * dx

        # Coupling energy: gamma * W(kappa, b, m) = -gamma/2 * chi(kappa) * sigma_frust
        coupling_energy = 0.0
        if sigma_frust is not None:
            chi = chi_kappa if chi_kappa is not None else np.ones_like(kappa)
            W = -0.5 * chi * sigma_frust
            coupling_energy = self.gamma * np.sum(W) * dx

        return gradient_energy + potential_energy + coupling_energy

    def functional_derivative(self, kappa: np.ndarray, dx: float = 1.0,
                               sigma_frust: Optional[np.ndarray] = None) -> np.ndarray:
        r"""Compute delta J / delta kappa.

        delta J / delta kappa = -alpha * nabla^2 kappa + beta * V'_kappa
                                + gamma * dW/dkappa

        Parameters
        ----------
        kappa : ndarray
            Constraint field.
        dx : float
            Grid spacing.
        sigma_frust : ndarray, optional
            Boundary frustration stress.

        Returns
        -------
        ndarray
            Functional derivative at each grid point.
        """
        kappa = np.asarray(kappa, dtype=float)

        # Laplacian (negative of gradient energy variation)
        if kappa.ndim == 1:
            lap = np.zeros_like(kappa)
            lap[1:-1] = (kappa[2:] - 2 * kappa[1:-1] + kappa[:-2]) / dx**2
            # Neumann BCs
            lap[0] = (kappa[1] - kappa[0]) / dx**2
            lap[-1] = (kappa[-2] - kappa[-1]) / dx**2
        else:
            from scipy.ndimage import laplace
            lap = laplace(kappa) / dx**2

        result = -self.alpha * lap + self.beta * self.potential.double_well_derivative(kappa)

        if sigma_frust is not None:
            # dW/dkappa ~ -0.5 * sigma_frust (assuming chi(kappa) ~ 1)
            result += self.gamma * (-0.5 * sigma_frust)

        return result


# ===================================================================
# Section 4.2 (Eq 25-27): Meta-Boundary Evolution
# ===================================================================

@dataclass
class MetaBoundaryEvolution:
    r"""Meta-boundary evolution equation (Eq 27).

    Overdamped limit:
        tau_kappa * d_t kappa = alpha * nabla^2 kappa + nu * nabla^2 kappa
                                - beta * V'_kappa(kappa)
                                - gamma * dW/dkappa + u(x,t)

    This is a reaction-diffusion equation with coupling to boundary stress.

    Parameters
    ----------
    tau_kappa : float
        Meta-boundary relaxation timescale.
    alpha : float
        Gradient stiffness.
    nu : float
        Additional diffusivity.
    beta : float
        Potential scale.
    gamma : float
        Stress-constraint coupling.
    potential : ConstraintPotential
        Constraint potential.
    """
    tau_kappa: float = 10.0
    alpha: float = 1.0
    nu: float = 0.1
    beta: float = 1.0
    gamma: float = 0.1
    potential: ConstraintPotential = field(default_factory=ConstraintPotential)

    @property
    def effective_diffusivity(self) -> float:
        r"""Effective diffusivity D = (alpha + nu) / tau_kappa."""
        return (self.alpha + self.nu) / self.tau_kappa

    def rhs(self, kappa: np.ndarray, dx: float = 1.0,
            u: Optional[np.ndarray] = None,
            sigma_frust: Optional[np.ndarray] = None) -> np.ndarray:
        r"""Compute d_t kappa from the evolution equation (Eq 27).

        Parameters
        ----------
        kappa : ndarray
            Current constraint field.
        dx : float
            Grid spacing.
        u : ndarray, optional
            External control input.
        sigma_frust : ndarray, optional
            Boundary frustration stress.

        Returns
        -------
        ndarray
            Time derivative d_t kappa at each grid point.
        """
        kappa = np.asarray(kappa, dtype=float)

        # Laplacian with Neumann BCs
        if kappa.ndim == 1:
            lap = np.zeros_like(kappa)
            lap[1:-1] = (kappa[2:] - 2 * kappa[1:-1] + kappa[:-2]) / dx**2
            lap[0] = (kappa[1] - kappa[0]) / dx**2
            lap[-1] = (kappa[-2] - kappa[-1]) / dx**2
        else:
            from scipy.ndimage import laplace
            lap = laplace(kappa) / dx**2

        # Reaction-diffusion
        diffusion = (self.alpha + self.nu) * lap
        reaction = -self.beta * self.potential.double_well_derivative(kappa)

        coupling = np.zeros_like(kappa)
        if sigma_frust is not None:
            coupling = -self.gamma * (-0.5 * sigma_frust)

        control = np.zeros_like(kappa) if u is None else np.asarray(u)

        return (diffusion + reaction + coupling + control) / self.tau_kappa

    def evolve(self, kappa0: np.ndarray, t_span: Tuple[float, float],
               dx: float = 1.0, n_steps: int = 500,
               u: Optional[Callable[[float], np.ndarray]] = None,
               sigma_frust: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        r"""Integrate the meta-boundary evolution forward in time.

        Parameters
        ----------
        kappa0 : ndarray
            Initial constraint field.
        t_span : (t_start, t_end)
            Time interval.
        dx : float
            Spatial grid spacing.
        n_steps : int
            Number of output time steps.
        u : callable(t) -> ndarray, optional
            Time-dependent external control input.
        sigma_frust : ndarray, optional
            Static boundary frustration stress.

        Returns
        -------
        t : ndarray, shape (n_steps,)
            Time points.
        kappa_traj : ndarray, shape (n_steps, N)
            Constraint field trajectory.
        """
        kappa0 = np.asarray(kappa0, dtype=float)
        N = len(kappa0)
        t_eval = np.linspace(t_span[0], t_span[1], n_steps)

        def ode_rhs(t, y):
            kappa = y.reshape(kappa0.shape)
            u_t = u(t) if u is not None else None
            dydt = self.rhs(kappa, dx, u_t, sigma_frust)
            return dydt.ravel()

        sol = solve_ivp(
            ode_rhs, t_span, kappa0.ravel(),
            t_eval=t_eval, method="RK45",
            max_step=(t_span[1] - t_span[0]) / 50,
        )

        kappa_traj = sol.y.T.reshape(len(sol.t), *kappa0.shape)
        return sol.t, kappa_traj


# ===================================================================
# Section 6 (Eq 36-40): Meta-Rewrite Eligibility
# ===================================================================

@dataclass
class MetaRewriteEligibility:
    r"""Meta-rewrite eligibility condition (Eq 36-37).

    E2[kappa; b, m] = S_acc[b, m, kappa] - C_barrier[kappa]

    Meta-rewrite fires when E2 >= 0.

    Accumulated stress (Eq 38):
        dS_acc/dt = integral rho(x,t) sigma(b,m,kappa) d^d x - mu * S_acc

    Barrier cost (Eq 40):
        C_barrier = inf over paths gamma from kappa_a to kappa_b

    Parameters
    ----------
    mu : float
        Stress dissipation rate.
    """
    mu: float = 0.1

    def accumulated_stress_steady_state(self, sigma_0: float) -> float:
        r"""Steady-state accumulated stress (Eq 39).

        S*_acc = Sigma_0 / mu

        Parameters
        ----------
        sigma_0 : float
            Constant stress source intensity.

        Returns
        -------
        float
            Steady-state accumulated stress.
        """
        if self.mu <= 0:
            return np.inf if sigma_0 > 0 else 0.0
        return sigma_0 / self.mu

    def evolve_stress(self, sigma_source: Callable[[float], float],
                      S0: float = 0.0,
                      t_span: Tuple[float, float] = (0, 100),
                      n_points: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        r"""Integrate accumulated stress dynamics (Eq 38).

        dS_acc/dt = sigma(t) - mu * S_acc

        Parameters
        ----------
        sigma_source : callable(t) -> float
            Time-dependent stress source intensity.
        S0 : float
            Initial accumulated stress.
        t_span : tuple
            Time interval.
        n_points : int
            Number of output points.

        Returns
        -------
        t : ndarray
            Time points.
        S : ndarray
            Accumulated stress trajectory.
        """
        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        def rhs(t, y):
            return [sigma_source(t) - self.mu * y[0]]

        sol = solve_ivp(rhs, t_span, [S0], t_eval=t_eval, method="RK45")
        return sol.t, sol.y[0]

    @staticmethod
    def barrier_cost_double_well(lambda_kappa: float, eta: float) -> float:
        r"""Barrier height for double-well potential (Eq 41).

        Delta_V = lambda_kappa * eta^4 / 4

        Parameters
        ----------
        lambda_kappa : float
            Quartic coupling.
        eta : float
            Phase separation.

        Returns
        -------
        float
            Barrier cost.
        """
        return lambda_kappa * eta**4 / 4.0

    def is_eligible(self, S_acc: float, C_barrier: float) -> bool:
        r"""Check meta-rewrite eligibility (Eq 37).

        E2 = S_acc - C_barrier >= 0

        Parameters
        ----------
        S_acc : float
            Accumulated boundary stress.
        C_barrier : float
            Meta-barrier cost.

        Returns
        -------
        bool
            True if meta-rewrite is eligible.
        """
        return S_acc >= C_barrier


# ===================================================================
# Section 7 (Eq 42-49): Front Solutions
# ===================================================================

@dataclass
class FrontSolution:
    r"""Static kink and moving front solutions (Eq 43-49).

    Static kink (Eq 43):
        kappa_kink(x) = eta * tanh((x - x0) / ell_kappa)

    Moving front velocity (Eq 46):
        v = Delta_J / (tau_kappa * Sigma_wall)

    Speed bound (Thm 7.2, Eq 49):
        |v| <= v_max = sqrt((alpha + nu) / tau_kappa) * m_kappa

    Parameters
    ----------
    potential : ConstraintPotential
        Constraint potential.
    tau_kappa : float
        Meta-boundary relaxation timescale.
    nu : float
        Additional diffusivity.
    """
    potential: ConstraintPotential = field(default_factory=ConstraintPotential)
    tau_kappa: float = 10.0
    nu: float = 0.1

    def static_kink(self, x: np.ndarray, x0: float = 0.0) -> np.ndarray:
        r"""Static kink interpolating between -eta and +eta (Eq 43).

        kappa_kink(x) = eta * tanh((x - x0) / ell_kappa)

        Parameters
        ----------
        x : ndarray
            Spatial coordinates.
        x0 : float
            Kink center position.

        Returns
        -------
        ndarray
            Kink profile.
        """
        ell = self.potential.domain_wall_width()
        return self.potential.eta * np.tanh((x - x0) / ell)

    def wall_tension(self) -> float:
        r"""Domain wall tension Sigma_wall for double-well kink (below Eq 48).

        Sigma_wall = (2*sqrt(2)/3) * eta^3 * sqrt(alpha * beta * lambda_kappa)

        Returns
        -------
        float
            Wall tension integral.
        """
        p = self.potential
        return (2.0 * np.sqrt(2) / 3.0) * p.eta**3 * np.sqrt(p.alpha * p.beta * p.lambda_kappa)

    def front_velocity(self, delta_J: float) -> float:
        r"""Moving front velocity (Eq 46).

        v = Delta_J / (tau_kappa * Sigma_wall)

        Parameters
        ----------
        delta_J : float
            Free energy difference between the two phases.

        Returns
        -------
        float
            Front propagation velocity.
        """
        sigma_wall = self.wall_tension()
        if sigma_wall <= 0:
            return np.inf if delta_J != 0 else 0.0
        return delta_J / (self.tau_kappa * sigma_wall)

    def max_velocity(self) -> float:
        r"""Maximum propagation speed (Thm 7.2, Eq 49).

        v_max = sqrt((alpha + nu) / tau_kappa) * m_kappa

        where m_kappa = sqrt(2 * beta * lambda_kappa / alpha) * eta.

        Returns
        -------
        float
            Maximum front velocity.
        """
        D = (self.potential.alpha + self.nu) / self.tau_kappa
        m_kappa = self.potential.mass_gap()
        return np.sqrt(D) * m_kappa

    def surface_tension(self) -> float:
        r"""Surface tension per unit area sigma_wall (Eq 52).

        sigma_wall = (2*sqrt(2)/3) * eta^3 * sqrt(alpha * beta * lambda_kappa)

        Returns
        -------
        float
            Surface tension.
        """
        return self.wall_tension()


# ===================================================================
# Section 8 (Eq 51-56): Energy Cost Scaling
# ===================================================================

@dataclass
class EnergyCostScaling:
    r"""Energy cost for meta-boundary transitions (Eq 51-56).

    Region rewrite cost (Eq 51):
        C_R = sigma_wall * |partial Omega_R| + Delta_J * |Omega_R|

    Critical nucleus (Eq 54):
        R_crit = (d-1) * sigma_wall / |Delta_J|

    No-free-lunch for global rewrites (Eq 56):
        C_global >= sigma_wall * L^{d-1}

    Parameters
    ----------
    sigma_wall : float
        Surface tension per unit area.
    delta_J : float
        Bulk free energy difference between phases.
    d : int
        Spatial dimension.
    """
    sigma_wall: float = 1.0
    delta_J: float = 0.1
    d: int = 3

    def region_cost(self, R: float) -> float:
        r"""Energy cost for rewriting a spherical region of radius R (Eq 53).

        C_R = sigma_wall * (2 pi^{d/2} / Gamma(d/2)) * R^{d-1}
              + delta_J * (pi^{d/2} / Gamma(d/2 + 1)) * R^d

        Parameters
        ----------
        R : float
            Region radius.

        Returns
        -------
        float
            Total energy cost.
        """
        d = self.d
        half_d = d / 2.0
        # Surface area of d-sphere: 2 * pi^{d/2} / Gamma(d/2) * R^{d-1}
        surface = 2.0 * np.pi**half_d / gamma_func(half_d) * R**(d - 1)
        # Volume of d-ball: pi^{d/2} / Gamma(d/2 + 1) * R^d
        volume = np.pi**half_d / gamma_func(half_d + 1) * R**d

        return self.sigma_wall * surface + self.delta_J * volume

    def critical_radius(self) -> float:
        r"""Critical nucleus radius (Eq 54).

        R_crit = (d - 1) * sigma_wall / |Delta_J|

        Returns
        -------
        float
            Minimum radius for self-sustaining transition.
        """
        if abs(self.delta_J) <= 0:
            return np.inf
        return (self.d - 1) * self.sigma_wall / abs(self.delta_J)

    def nucleation_barrier(self) -> float:
        r"""Nucleation barrier energy C_crit (Eq 55).

        C_crit = C(R_crit) = sigma_wall^d / |Delta_J|^{d-1} * omega_d

        Returns
        -------
        float
            Energy barrier for nucleation.
        """
        R_c = self.critical_radius()
        if not np.isfinite(R_c):
            return np.inf
        return self.region_cost(R_c)

    def global_rewrite_cost(self, L: float) -> float:
        r"""Minimum cost for global rewrite (Eq 56).

        C_global >= sigma_wall * L^{d-1}

        Parameters
        ----------
        L : float
            Domain diameter.

        Returns
        -------
        float
            Lower bound on global rewrite energy.
        """
        return self.sigma_wall * L**(self.d - 1)


# ===================================================================
# Section 9.2 (Eq 59-63): Spectral Drift
# ===================================================================

@dataclass
class SpectralDrift:
    r"""Spectral shift from constraint transitions (Eq 60-61).

    When kappa shifts from kappa_a to kappa_b, eigenvalue spectrum changes:
        delta_lambda_n = (d lambda_n / d kappa)|_{kappa_a} * (kappa_b - kappa_a)

    Relative shift bound (Thm 9.1, Eq 61):
        |delta_lambda_n / lambda_n| <= |Delta_kappa / kappa_char|
                                       * |d ln lambda_n / d ln kappa|

    Parameters
    ----------
    kappa_char : float
        Characteristic constraint scale.
    """
    kappa_char: float = 1.0

    def spectral_shift(self, d_lambda_d_kappa: float,
                        delta_kappa: float) -> float:
        r"""First-order spectral shift (Eq 60).

        delta_lambda_n = (d lambda_n / d kappa) * Delta_kappa

        Parameters
        ----------
        d_lambda_d_kappa : float
            Derivative of eigenvalue w.r.t. kappa.
        delta_kappa : float
            Constraint field change.

        Returns
        -------
        float
            Eigenvalue shift.
        """
        return d_lambda_d_kappa * delta_kappa

    def relative_shift_bound(self, delta_kappa: float,
                              d_ln_lambda_d_ln_kappa: float) -> float:
        r"""Upper bound on relative spectral shift (Thm 9.1, Eq 61).

        |delta_lambda_n / lambda_n| <= |Delta_kappa / kappa_char|
                                       * |d ln lambda_n / d ln kappa|

        Parameters
        ----------
        delta_kappa : float
            Constraint field change.
        d_ln_lambda_d_ln_kappa : float
            Log-derivative of eigenvalue.

        Returns
        -------
        float
            Upper bound on relative shift.
        """
        return abs(delta_kappa / self.kappa_char) * abs(d_ln_lambda_d_ln_kappa)

    @staticmethod
    def vacuum_energy_shift(eigenvalues: np.ndarray,
                            delta_eigenvalues: np.ndarray) -> float:
        r"""Vacuum energy perturbation from spectral drift (Eq 62-63).

        delta_rho_vac = (1/4) * sum_n delta_lambda_n / sqrt(lambda_n)

        Parameters
        ----------
        eigenvalues : ndarray
            Original eigenvalues lambda_n (must be positive).
        delta_eigenvalues : ndarray
            Eigenvalue shifts delta_lambda_n.

        Returns
        -------
        float
            Vacuum energy density shift.
        """
        eigenvalues = np.asarray(eigenvalues, dtype=float)
        delta_eigenvalues = np.asarray(delta_eigenvalues, dtype=float)
        mask = eigenvalues > 0
        return 0.25 * float(np.sum(delta_eigenvalues[mask] / np.sqrt(eigenvalues[mask])))

    def constrain_from_clock(self, alpha_dot_over_alpha: float,
                             d_ln_lambda_d_ln_kappa: float = 1.0) -> float:
        r"""Constrain |d_t kappa| from atomic clock measurements.

        |d_t kappa| / kappa_char <= (alpha_dot / alpha) / |d ln lambda / d ln kappa|

        Parameters
        ----------
        alpha_dot_over_alpha : float
            Measured drift rate of fine structure constant (per year).
        d_ln_lambda_d_ln_kappa : float
            Log-derivative of eigenvalue.

        Returns
        -------
        float
            Upper bound on |d_t kappa| / kappa_char (per year).
        """
        return abs(alpha_dot_over_alpha) / abs(d_ln_lambda_d_ln_kappa)


# ===================================================================
# Section 10 (Thm 10.1): Detectability Theorem
# ===================================================================

@dataclass
class DetectabilityTheorem:
    r"""No-go theorem for undetectable meta-boundary rewrites (Thm 10.1).

    Any meta-boundary transition with kappa_a != kappa_b must produce
    at least one of four detectable signatures:
        1. Domain wall energy deposition: E_wall = sigma_wall * |partial Omega_R| > 0
        2. Spectral drift: exists n such that delta_lambda_n != 0
        3. Metric perturbation: Delta g_mu_nu != 0 in transition region
        4. Coherence transient: K(t) exhibits rapid excursion
    """

    @staticmethod
    def check_signatures(E_wall: float = 0.0,
                         max_spectral_shift: float = 0.0,
                         max_metric_perturbation: float = 0.0,
                         coherence_dip: float = 0.0,
                         tolerance: float = 1e-12) -> Dict[str, bool]:
        r"""Check which detectability signatures are present (Thm 10.1).

        Parameters
        ----------
        E_wall : float
            Domain wall energy deposition.
        max_spectral_shift : float
            Maximum |delta_lambda_n| across modes.
        max_metric_perturbation : float
            Maximum |Delta g_mu_nu|.
        coherence_dip : float
            Magnitude of coherence excursion |delta K|.
        tolerance : float
            Detection threshold.

        Returns
        -------
        dict with:
            'wall_energy' : bool
            'spectral_drift' : bool
            'metric_perturbation' : bool
            'coherence_transient' : bool
            'any_detected' : bool
            'n_signatures' : int
        """
        sigs = {
            "wall_energy": abs(E_wall) > tolerance,
            "spectral_drift": abs(max_spectral_shift) > tolerance,
            "metric_perturbation": abs(max_metric_perturbation) > tolerance,
            "coherence_transient": abs(coherence_dip) > tolerance,
        }
        sigs["any_detected"] = any(sigs.values())
        sigs["n_signatures"] = sum(1 for v in list(sigs.values())[:4] if v)
        return sigs

    @staticmethod
    def is_transition_detectable(kappa_a: float, kappa_b: float,
                                  sigma_wall: float,
                                  alpha: float = 1.0,
                                  tolerance: float = 1e-12) -> bool:
        r"""Test detectability theorem (Corollary 10.2).

        A transition is detectable iff kappa_a != kappa_b
        AND at least one signature is nonzero.

        For alpha > 0 and sigma_wall > 0 with kappa_a != kappa_b,
        the theorem guarantees at least one signature.

        Parameters
        ----------
        kappa_a, kappa_b : float
            Initial and final constraint field values.
        sigma_wall : float
            Surface tension (must be > 0 for nontrivial potential).
        alpha : float
            Gradient stiffness (must be > 0).

        Returns
        -------
        bool
            True if the transition must be detectable.
        """
        if abs(kappa_a - kappa_b) < tolerance:
            return False
        # By theorem: alpha > 0 and sigma_wall > 0 guarantee detection
        return alpha > 0 and sigma_wall > 0


# ===================================================================
# Section 11 (Eq 64-67): Hysteresis and Path Dependence
# ===================================================================

@dataclass
class Hysteresis:
    r"""Hysteresis and path dependence in meta-boundary dynamics (Eq 64-67).

    Local rewrite threshold expands as (Eq 64):
        epsilon_1(t) = epsilon_1^(0) + epsilon_1' * delta_kappa(t)
                       + 0.5 * epsilon_1'' * delta_kappa(t)^2 + ...

    Hysteresis manifests as (Eq 65):
        epsilon_1^forward != epsilon_1^reverse

    Berry connection on constraint fiber bundle (Eq 67):
        F_ij^(kappa) = d_i A_j^(kappa) - d_j A_i^(kappa)
                       + [A_i^(kappa), A_j^(kappa)]

    Parameters
    ----------
    epsilon_0 : float
        Baseline rewrite threshold.
    epsilon_prime : float
        Linear coupling of threshold to kappa.
    epsilon_double_prime : float
        Quadratic coupling of threshold to kappa.
    """
    epsilon_0: float = 1.0
    epsilon_prime: float = 0.1
    epsilon_double_prime: float = 0.01

    def threshold(self, delta_kappa: float) -> float:
        r"""Expanded rewrite threshold (Eq 64).

        epsilon_1(delta_kappa) = epsilon_0 + epsilon' * delta_kappa
                                 + 0.5 * epsilon'' * delta_kappa^2

        Parameters
        ----------
        delta_kappa : float
            Departure from reference constraint value.

        Returns
        -------
        float
            Local rewrite threshold.
        """
        return (self.epsilon_0
                + self.epsilon_prime * delta_kappa
                + 0.5 * self.epsilon_double_prime * delta_kappa**2)

    def hysteresis_loop(self, kappa_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""Compute hysteresis loop for threshold vs kappa.

        Forward: kappa goes from kappa_min to kappa_max.
        Reverse: kappa goes from kappa_max to kappa_min.

        Returns thresholds for forward and reverse paths.

        Parameters
        ----------
        kappa_values : ndarray
            Kappa values for one direction (forward).

        Returns
        -------
        eps_forward : ndarray
            Thresholds on forward path.
        eps_reverse : ndarray
            Thresholds on reverse path (with history-dependent shift).
        """
        kappa_values = np.asarray(kappa_values, dtype=float)
        # Forward path
        eps_forward = np.array([self.threshold(dk) for dk in kappa_values])
        # Reverse path includes accumulated history (shift proportional to max excursion)
        kappa_max = np.max(np.abs(kappa_values))
        history_shift = 0.5 * self.epsilon_double_prime * kappa_max
        eps_reverse = np.array([
            self.threshold(dk) + history_shift * np.sign(dk)
            for dk in kappa_values[::-1]
        ])
        return eps_forward, eps_reverse

    @staticmethod
    def berry_phase(phases_around_loop: np.ndarray) -> float:
        r"""Compute Berry phase from parallel transport around a loop (Eq 67).

        gamma_Berry = -Im sum_n <phi_n| d phi_n>

        Approximated from discrete phase accumulation.

        Parameters
        ----------
        phases_around_loop : ndarray
            Phases accumulated at each step around a closed loop in K.

        Returns
        -------
        float
            Berry phase (mod 2pi).
        """
        total = np.sum(phases_around_loop)
        return float(total % (2.0 * np.pi))


# ===================================================================
# Section 12 (Eq 68-71): Thermodynamic Signatures
# ===================================================================

@dataclass
class ThermodynamicSignatures:
    r"""Thermodynamic signatures of meta-boundary transitions (Eq 68-71).

    Entropy production rate (Eq 68):
        S_dot_kappa = (1/T_eff) * integral tau_kappa * (d_t kappa)^2 d^d x >= 0

    Front entropy production (Eq 69):
        S_dot_kappa = tau_kappa * v^2 * Sigma_wall * A_front / T_eff

    Minimum dissipation bound (Thm 12.1, Eq 71):
        Q_min = tau_kappa * ||Delta_kappa||^2_{L2} / t_f

    Parameters
    ----------
    tau_kappa : float
        Meta-boundary relaxation timescale.
    T_eff : float
        Effective temperature of constraint fluctuations.
    """
    tau_kappa: float = 10.0
    T_eff: float = 1.0

    def entropy_production_rate(self, dkappa_dt: np.ndarray,
                                 dx: float = 1.0) -> float:
        r"""Entropy production rate (Eq 68).

        S_dot = (tau_kappa / T_eff) * integral (d_t kappa)^2 d^d x

        Parameters
        ----------
        dkappa_dt : ndarray
            Time derivative of constraint field at each grid point.
        dx : float
            Grid spacing.

        Returns
        -------
        float
            Entropy production rate (non-negative).
        """
        dkappa_dt = np.asarray(dkappa_dt, dtype=float)
        integrand = dkappa_dt**2
        return self.tau_kappa / self.T_eff * float(np.sum(integrand)) * dx

    def front_entropy_rate(self, v: float, sigma_wall: float,
                            A_front: float) -> float:
        r"""Entropy production rate for a propagating front (Eq 69).

        S_dot = tau_kappa * v^2 * Sigma_wall * A_front / T_eff

        Parameters
        ----------
        v : float
            Front velocity.
        sigma_wall : float
            Wall tension.
        A_front : float
            Cross-sectional area of the front.

        Returns
        -------
        float
            Entropy production rate.
        """
        return self.tau_kappa * v**2 * sigma_wall * A_front / self.T_eff

    def minimum_dissipation(self, delta_kappa_L2_sq: float,
                             t_f: float) -> float:
        r"""Minimum dissipation bound (Thm 12.1, Eq 71).

        Q_min = tau_kappa * ||Delta_kappa||^2_{L2} / t_f

        Parameters
        ----------
        delta_kappa_L2_sq : float
            L2 norm squared of the kappa change: integral (kappa_b - kappa_a)^2 d^d x.
        t_f : float
            Transition duration.

        Returns
        -------
        float
            Minimum energy dissipated.
        """
        if t_f <= 0:
            return np.inf
        return self.tau_kappa * delta_kappa_L2_sq / t_f

    def total_dissipation(self, dkappa_dt_history: np.ndarray,
                           dt: float, dx: float = 1.0) -> float:
        r"""Total energy dissipated during a transition (Eq 70).

        Q = tau_kappa * integral_0^{t_f} integral (d_t kappa)^2 d^d x dt

        Parameters
        ----------
        dkappa_dt_history : ndarray, shape (n_times, n_spatial)
            Time derivative at each time step and spatial point.
        dt : float
            Time step size.
        dx : float
            Spatial grid spacing.

        Returns
        -------
        float
            Total dissipated energy.
        """
        dkappa_dt_history = np.asarray(dkappa_dt_history, dtype=float)
        return self.tau_kappa * float(np.sum(dkappa_dt_history**2)) * dt * dx


# ===================================================================
# Section 13 (Eq 73-78): Stability Analysis
# ===================================================================

@dataclass
class StabilityAnalysis:
    r"""Linear and Lyapunov stability of constraint phases (Eq 73-78).

    Linear stability criterion (Thm 13.1, Eq 75):
        m_kappa^2 * alpha + gamma * W''_kappa(kappa*) > 0

    Critical coupling (Eq 76):
        gamma_crit = m_kappa^2 * alpha / |W''_kappa(kappa*)|

    Lyapunov functional (Thm 13.2, Eq 77):
        L[kappa] = J[kappa; b*, m*] - J[kappa*; b*, m*]

    Parameters
    ----------
    alpha : float
        Gradient stiffness.
    beta : float
        Potential scale.
    gamma : float
        Stress-constraint coupling.
    nu : float
        Additional diffusivity.
    tau_kappa : float
        Relaxation timescale.
    potential : ConstraintPotential
        Constraint potential.
    """
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.1
    nu: float = 0.1
    tau_kappa: float = 10.0
    potential: ConstraintPotential = field(default_factory=ConstraintPotential)

    def linear_growth_rate(self, k: float,
                            W_double_prime: float = 0.0) -> float:
        r"""Linearized growth rate omega(k) (Eq 74).

        tau_kappa * omega = -(alpha + nu) * k^2 - m_kappa^2 * alpha
                            - gamma * W''_kappa

        Parameters
        ----------
        k : float
            Wavevector magnitude.
        W_double_prime : float
            Second derivative of coupling W w.r.t. kappa at equilibrium.

        Returns
        -------
        float
            Growth rate omega (negative = stable).
        """
        m_kappa_sq = (self.beta * self.potential.double_well_second_derivative(
            self.potential.eta)) / self.alpha
        stability_mass = m_kappa_sq * self.alpha + self.gamma * W_double_prime
        return (-(self.alpha + self.nu) * k**2 - stability_mass) / self.tau_kappa

    def is_stable(self, W_double_prime: float = 0.0) -> bool:
        r"""Check linear stability criterion (Thm 13.1, Eq 75).

        Stable iff m_kappa^2 * alpha + gamma * W''_kappa(kappa*) > 0

        Parameters
        ----------
        W_double_prime : float
            Second derivative of coupling W w.r.t. kappa.

        Returns
        -------
        bool
            True if the constraint phase is linearly stable.
        """
        m_kappa_sq_alpha = self.beta * self.potential.double_well_second_derivative(
            self.potential.eta)
        return m_kappa_sq_alpha + self.gamma * W_double_prime > 0

    def critical_coupling(self, W_double_prime: float) -> float:
        r"""Critical coupling for destabilization (Eq 76).

        gamma_crit = m_kappa^2 * alpha / |W''_kappa(kappa*)|

        Parameters
        ----------
        W_double_prime : float
            Second derivative of coupling W w.r.t. kappa.

        Returns
        -------
        float
            Critical gamma above which the phase destabilizes.
        """
        if abs(W_double_prime) < 1e-15:
            return np.inf
        m_kappa_sq_alpha = self.beta * self.potential.double_well_second_derivative(
            self.potential.eta)
        return m_kappa_sq_alpha / abs(W_double_prime)

    def lyapunov_rate(self, kappa: np.ndarray, dx: float = 1.0,
                       sigma_frust: Optional[np.ndarray] = None) -> float:
        r"""Lyapunov decay rate dL/dt (Eq 78).

        dL/dt = -(1/tau_kappa) * integral |delta J / delta kappa|^2 d^d x <= 0

        Parameters
        ----------
        kappa : ndarray
            Current constraint field.
        dx : float
            Grid spacing.
        sigma_frust : ndarray, optional
            Boundary frustration stress.

        Returns
        -------
        float
            dL/dt (non-positive).
        """
        mf = MetaFunctional(alpha=self.alpha, beta=self.beta,
                            gamma=self.gamma, potential=self.potential)
        dJ_dkappa = mf.functional_derivative(kappa, dx, sigma_frust)
        return -1.0 / self.tau_kappa * float(np.sum(dJ_dkappa**2)) * dx


# ===================================================================
# Section 14.2 (Eq 83): Topological Memory Scaling
# ===================================================================

@dataclass
class TopologicalMemoryScaling:
    r"""Coherence time scaling with winding number (Eq 83).

    tau_m ~ tau_0 * |W|^alpha

    where W is the winding number, tau_0 is base coherence time,
    and alpha > 0 is the topological protection exponent.

    Standard QM predicts tau_m ~ tau_0 * exp(-gamma * t),
    independent of topological sector.

    Parameters
    ----------
    tau_0 : float
        Base coherence time.
    alpha_topo : float
        Topological protection exponent (alpha in the paper).
    """
    tau_0: float = 1.0
    alpha_topo: float = 1.0

    def coherence_time(self, W: int) -> float:
        r"""Coherence time for winding number W (Eq 83).

        tau_m = tau_0 * |W|^alpha

        Parameters
        ----------
        W : int
            Winding number.

        Returns
        -------
        float
            Predicted coherence time.
        """
        if W == 0:
            return self.tau_0
        return self.tau_0 * abs(W)**self.alpha_topo

    def fit_exponent(self, W_values: np.ndarray,
                     tau_values: np.ndarray) -> float:
        r"""Fit the topological protection exponent from data.

        Fits log(tau_m) = log(tau_0) + alpha * log(|W|)

        Parameters
        ----------
        W_values : ndarray
            Array of winding numbers (nonzero).
        tau_values : ndarray
            Measured coherence times.

        Returns
        -------
        float
            Fitted alpha exponent.
        """
        mask = np.asarray(W_values) != 0
        log_W = np.log(np.abs(np.asarray(W_values[mask], dtype=float)))
        log_tau = np.log(np.asarray(tau_values[mask], dtype=float))
        coeffs = np.polyfit(log_W, log_tau, 1)
        return coeffs[0]  # slope = alpha


# ===================================================================
# Section 16: CERN Scenario Analysis
# ===================================================================

@dataclass
class CERNScenario:
    r"""Energy budget analysis for collider-triggered transitions (Section 16).

    Evaluates whether a high-energy collider can trigger a macroscopic
    meta-boundary transition. The analysis considers:
        1. Energy budget: C_min = sigma_wall * 4*pi*R^2 (Eq 86)
        2. Coherence requirement: N^{-1/2} suppression of coherent stress
        3. Gauge-invariant assessment: |delta Omega| ~ (E/E_Planck) * (l_P/R_crit)^{d-1}

    Parameters
    ----------
    E_collision : float
        Collision energy in GeV.
    sigma_wall : float
        Surface tension per unit area (in natural units).
    R_macro : float
        Target macroscopic transition radius (meters).
    d : int
        Spatial dimension.
    """
    E_collision: float = 14e3      # 14 TeV in GeV
    sigma_wall: float = 1.0
    R_macro: float = 1.0           # 1 meter
    d: int = 3

    def minimum_surface_energy(self) -> float:
        r"""Minimum surface energy for macroscopic transition (Eq 86).

        C_min = sigma_wall * 4*pi*R^2  (for d=3)

        Returns
        -------
        float
            Minimum energy required.
        """
        if self.d == 3:
            return self.sigma_wall * 4.0 * np.pi * self.R_macro**2
        # General d
        half_d = self.d / 2.0
        surface = 2.0 * np.pi**half_d / gamma_func(half_d) * self.R_macro**(self.d - 1)
        return self.sigma_wall * surface

    def coherent_stress_fraction(self, N_events: int) -> float:
        r"""Coherent fraction of stress from N incoherent events.

        Fraction ~ N^{-1/2} for uncorrelated events.

        Parameters
        ----------
        N_events : int
            Number of collision events.

        Returns
        -------
        float
            Fractional coherent contribution.
        """
        if N_events <= 0:
            return 0.0
        return 1.0 / np.sqrt(float(N_events))

    def gauge_invariant_curvature(self, R_crit: float) -> float:
        r"""Gauge-invariant curvature induced by collider (Eq 87).

        |delta Omega_ABC| ~ (E_collision / E_Planck) * (l_P / R_crit)^{d-1}

        Parameters
        ----------
        R_crit : float
            Critical nucleus radius.

        Returns
        -------
        float
            Induced boundary phase curvature.
        """
        # Convert collision energy to Joules (E_collision is in GeV)
        E_J = self.E_collision * 1.6e-10
        E_ratio = E_J / _E_PLANCK
        spatial_ratio = (_L_PLANCK / R_crit)**(self.d - 1)
        return E_ratio * spatial_ratio

    def is_transition_possible(self, R_crit: float = 1.0) -> Dict[str, object]:
        r"""Full assessment of collider-triggered transition possibility.

        Checks three conditions from Section 16:
        1. Energy sufficiency
        2. Coherence requirement
        3. Gauge-invariant curvature significance

        Parameters
        ----------
        R_crit : float
            Critical nucleus radius.

        Returns
        -------
        dict with:
            'energy_sufficient' : bool
            'energy_ratio' : float (E_collision / C_min)
            'coherent_fraction' : float
            'induced_curvature' : float
            'curvature_sufficient' : bool
            'transition_possible' : bool
        """
        C_min = self.minimum_surface_energy()
        E_collision_J = self.E_collision * 1.6e-10
        energy_ratio = E_collision_J / C_min if C_min > 0 else np.inf

        # ~10^9 collisions per second at LHC
        coherent_frac = self.coherent_stress_fraction(int(1e9))

        delta_omega = self.gauge_invariant_curvature(R_crit)

        return {
            "energy_sufficient": energy_ratio >= 1.0,
            "energy_ratio": energy_ratio,
            "coherent_fraction": coherent_frac,
            "induced_curvature": delta_omega,
            "curvature_sufficient": delta_omega >= 1.0,
            "transition_possible": (energy_ratio >= 1.0
                                    and delta_omega >= 1.0),
        }
