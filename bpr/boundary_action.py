"""
Master Boundary Action: Universal Cross-Sector Observable
==========================================================

The "Rosetta Stone" of BPR: a single boundary action functional
from which all physics sectors emerge as limits.

Key equations:
    S_d = int sqrt|gamma| * (L_ph + L_cpl + L_ct) d^3x
    sigma_eff(omega) = 1 - ||S(omega; Z_s)||^2   (effective cross-section)
    Q_eff = omega_0 * E_stored / (P_ohmic + P_rad(Z_s))  (quality factor)

Sectors: EM (Maxwell), QM (Schroedinger), GR (Einstein), NS (Navier-Stokes)

References: Al-Kahwati (2026), Toward a Unification
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_Z0_VACUUM = 376.730313668          # Ohm  (impedance of free space)
_C = 299792458.0                    # m/s
_G_NEWTON = 6.67430e-11             # m^3 kg^-1 s^-2


# ===========================================================================
#  Master Boundary Action
# ===========================================================================

@dataclass
class BoundaryAction:
    """Master boundary action functional S_d.

    The action integrates sqrt|gamma| * (L_ph + L_cpl + L_ct) over the
    three-dimensional boundary hypersurface.

    Parameters
    ----------
    gamma_metric : np.ndarray
        Induced metric on the boundary (3x3 matrix at each point, or scalar
        determinant for simplified 1-D computations).
    Z_impedance : float or complex
        Boundary impedance Z_s characterising the interface.
    sector : str
        Physics sector: 'em', 'qm', 'gr', or 'ns'.
    """
    gamma_metric: np.ndarray
    Z_impedance: complex
    sector: str = "em"

    # ---- Lagrangian densities -------------------------------------------

    def lagrangian_phase(
        self, field: np.ndarray, grad_field: np.ndarray
    ) -> np.ndarray:
        """Phase (bulk kinetic) Lagrangian L_ph for the chosen sector.

        EM:  L_ph = (1/2)(|grad A|^2 - (1/c^2)|dA/dt|^2)  (static approx)
        QM:  L_ph = (hbar^2/2m)|grad psi|^2
        GR:  L_ph = (1/16 pi G) R_boundary ~ |grad g|^2 proxy
        NS:  L_ph = (rho/2)|v|^2
        """
        if self.sector == "em":
            # Electromagnetic: half the squared gradient (Coulomb gauge)
            return 0.5 * np.sum(grad_field ** 2, axis=-1)
        elif self.sector == "qm":
            # Quantum: kinetic density |grad psi|^2
            return 0.5 * np.sum(np.abs(grad_field) ** 2, axis=-1)
        elif self.sector == "gr":
            # Gravitational: linearised Ricci scalar proxy
            return (1.0 / (16.0 * np.pi * _G_NEWTON)) * np.sum(
                grad_field ** 2, axis=-1
            )
        elif self.sector == "ns":
            # Navier-Stokes: kinetic energy density (rho=1 normalisation)
            return 0.5 * np.sum(field ** 2, axis=-1)
        raise ValueError(f"Unknown sector: {self.sector}")

    def lagrangian_coupling(
        self, field: np.ndarray, boundary_data: np.ndarray
    ) -> np.ndarray:
        """Coupling Lagrangian L_cpl between bulk field and boundary data.

        Models the impedance-weighted overlap:
            L_cpl = Re(Z_s) * field * boundary_data
        """
        Z_re = np.real(self.Z_impedance)
        return Z_re * field * boundary_data

    def lagrangian_counterterm(self, field: np.ndarray) -> np.ndarray:
        """Counter-term Lagrangian L_ct for UV regulation.

        L_ct = -(1/2)|Z_s|^{-1} * field^2
        Removes boundary divergences analogous to holographic renormalisation.
        """
        Z_abs = np.abs(self.Z_impedance)
        if Z_abs == 0.0:
            return np.zeros_like(field)
        return -0.5 * field ** 2 / Z_abs

    # ---- Full action ----------------------------------------------------

    def _sqrt_gamma(self) -> np.ndarray:
        """Compute sqrt|det gamma| from the metric data.

        Accepts either a scalar/array of determinant values, or a stack of
        3x3 matrices from which determinants are computed.
        """
        g = np.asarray(self.gamma_metric, dtype=float)
        if g.ndim == 0 or (g.ndim == 1):
            # Scalar or 1-D array of determinant values
            return np.sqrt(np.abs(g))
        if g.ndim == 2 and g.shape == (3, 3):
            return np.sqrt(np.abs(np.linalg.det(g))) * np.ones(1)
        if g.ndim == 3 and g.shape[-2:] == (3, 3):
            # Stack of 3x3 matrices
            return np.sqrt(np.abs(np.linalg.det(g)))
        # Fallback: treat as determinant values
        return np.sqrt(np.abs(g))

    def action(
        self,
        field: np.ndarray,
        grad_field: np.ndarray,
        boundary_data: np.ndarray,
        dx: float,
        ccr_action: Optional["CCRAction"] = None,
        ccr_inputs: Optional[dict] = None,
    ) -> float:
        """Compute the boundary action S_d = int sqrt|gamma| * L d^3x.

        Uses a simple Riemann sum with volume element dx^3.

        Parameters
        ----------
        field : np.ndarray
            Field values on the boundary grid.
        grad_field : np.ndarray
            Gradient of the field (last axis = spatial components).
        boundary_data : np.ndarray
            External / boundary source data.
        dx : float
            Grid spacing (uniform cubic lattice assumed).
        ccr_action : CCRAction, optional
            Postulate-0 constraint action.  When supplied, S_CCR is added
            to S_d enforcing C_n equivariance and scale covariance.
        ccr_inputs : dict, optional
            Required when ``ccr_action`` is given, with keys:
            ``mode_amplitudes``, ``m_indices``, ``phi_at_x``, ``phi_at_sx``.
        """
        L_ph = self.lagrangian_phase(field, grad_field)
        L_cpl = self.lagrangian_coupling(field, boundary_data)
        L_ct = self.lagrangian_counterterm(field)
        L_total = L_ph + L_cpl + L_ct

        sqrt_g = self._sqrt_gamma()
        # Broadcast sqrt_g to the field shape if needed
        if sqrt_g.size == 1:
            sqrt_g = sqrt_g.flat[0]

        integrand = sqrt_g * L_total
        dV = dx ** 3
        S = float(np.sum(integrand) * dV)

        if ccr_action is not None:
            if ccr_inputs is None:
                raise ValueError(
                    "ccr_inputs required when ccr_action is supplied"
                )
            S += ccr_action.lagrangian(
                ccr_inputs["mode_amplitudes"],
                ccr_inputs["m_indices"],
                ccr_inputs["phi_at_x"],
                ccr_inputs["phi_at_sx"],
            )
        return S

    # ---- Equations of motion --------------------------------------------

    def stationarity_equations(
        self, field: np.ndarray, dx: float
    ) -> np.ndarray:
        r"""Compute delta S / delta Phi numerically (functional derivative).

        Uses a centred finite-difference approximation:
            (delta S / delta Phi)_i ~ (S[Phi+eps*e_i] - S[Phi-eps*e_i]) / (2*eps*dx^3)

        Returns the EoM residual array (should vanish on-shell).
        """
        eps = 1e-6
        flat = field.ravel()
        eom = np.zeros_like(flat)
        grad_field = np.gradient(field, dx)
        if isinstance(grad_field, list):
            grad_field = np.stack(grad_field, axis=-1)
        bd = np.zeros_like(field)  # zero boundary data for free EoM

        for i in range(flat.size):
            f_plus = flat.copy()
            f_plus[i] += eps
            f_minus = flat.copy()
            f_minus[i] -= eps

            fp = f_plus.reshape(field.shape)
            fm = f_minus.reshape(field.shape)

            gp = np.gradient(fp, dx)
            gm = np.gradient(fm, dx)
            if isinstance(gp, list):
                gp = np.stack(gp, axis=-1)
                gm = np.stack(gm, axis=-1)

            S_plus = self.action(fp, gp, bd, dx)
            S_minus = self.action(fm, gm, bd, dx)
            eom[i] = (S_plus - S_minus) / (2.0 * eps * dx ** 3)

        return eom.reshape(field.shape)


# ===========================================================================
#  Scattering observables
# ===========================================================================

def sigma_effective(
    omega: float | np.ndarray,
    Z_s: complex | np.ndarray,
    S_matrix: Optional[np.ndarray] = None,
    Z_0: float = _Z0_VACUUM,
) -> float | np.ndarray:
    """Effective cross-section sigma_eff(omega) = 1 - ||S(omega; Z_s)||^2.

    Parameters
    ----------
    omega : float or array
        Angular frequency.
    Z_s : complex or array
        Boundary impedance (may be frequency-dependent).
    S_matrix : np.ndarray, optional
        Pre-computed scattering matrix.  If None, uses the single-interface
        reflection coefficient S = (Z_s - Z_0) / (Z_s + Z_0).
    Z_0 : float
        Reference (vacuum) impedance.
    """
    omega = np.asarray(omega, dtype=complex)
    Z_s = np.asarray(Z_s, dtype=complex)

    if S_matrix is None:
        # Simple single-interface scattering model
        S = (Z_s - Z_0) / (Z_s + Z_0)
    else:
        S = np.asarray(S_matrix, dtype=complex)

    return np.real(1.0 - np.abs(S) ** 2)


def quality_factor_effective(
    omega_0: float,
    E_stored: float,
    P_ohmic: float,
    P_rad: float,
) -> float:
    """Effective quality factor Q_eff = omega_0 * E_stored / (P_ohmic + P_rad).

    Parameters
    ----------
    omega_0 : float
        Resonant angular frequency.
    E_stored : float
        Energy stored in the resonator.
    P_ohmic : float
        Ohmic (dissipative) power loss.
    P_rad : float
        Radiative power loss through the boundary (impedance-dependent).
    """
    P_total = P_ohmic + P_rad
    if P_total <= 0.0:
        return np.inf
    return omega_0 * E_stored / P_total


# ===========================================================================
#  Boundary RG flow
# ===========================================================================

def boundary_rg_flow(
    couplings: Dict[str, float],
    mu_range: np.ndarray,
    beta_functions: Dict[str, Callable[[float], float]],
) -> Dict[str, np.ndarray]:
    """Integrate boundary renormalisation group flow equations.

    Solves dg_i/d(ln mu) = beta_i(g_i) for each coupling over the given
    range of the RG scale parameter mu.

    Parameters
    ----------
    couplings : dict
        Initial coupling values {name: value} at mu_range[0].
    mu_range : np.ndarray
        Array of RG scale values (must be positive).
    beta_functions : dict
        Maps coupling name -> callable beta(g) returning dg/d(ln mu).

    Returns
    -------
    dict
        Trajectories {name: array of coupling values} evaluated at mu_range.
    """
    names = sorted(couplings.keys())
    y0 = np.array([couplings[n] for n in names])

    # Use ln(mu) as the independent variable for the ODE
    ln_mu = np.log(mu_range)

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        dydt = np.zeros_like(y)
        for i, name in enumerate(names):
            dydt[i] = beta_functions[name](y[i])
        return dydt

    sol = solve_ivp(
        rhs,
        t_span=(ln_mu[0], ln_mu[-1]),
        y0=y0,
        t_eval=ln_mu,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )

    result: Dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        result[name] = sol.y[i]
    return result


def sigma_eff_at_fixed_point(
    omega_range: np.ndarray,
    Z_fixed: complex,
    Z_0: float = _Z0_VACUUM,
) -> np.ndarray:
    """Compute sigma_eff over a frequency range at an RG fixed-point impedance.

    At a fixed point the impedance Z_fixed is scale-invariant, so sigma_eff
    is frequency-independent (a hallmark of conformal boundary conditions).

    Parameters
    ----------
    omega_range : np.ndarray
        Array of angular frequencies.
    Z_fixed : complex
        Fixed-point boundary impedance.
    Z_0 : float
        Reference impedance.

    Returns
    -------
    np.ndarray
        sigma_eff at each frequency (constant if truly at fixed point).
    """
    return sigma_effective(omega_range, Z_fixed, Z_0=Z_0)


# ===========================================================================
#  Sectoral limits
# ===========================================================================

@dataclass
class SectoralLimit:
    """Extract the sector-specific field equation from the master action.

    Each physics sector arises as a particular limit of the boundary action
    when the impedance and field content are specialised.

    Parameters
    ----------
    sector : str
        One of 'em', 'qm', 'gr', 'ns'.
    """
    sector: str

    def __post_init__(self) -> None:
        allowed = {"em", "qm", "gr", "ns"}
        if self.sector not in allowed:
            raise ValueError(f"sector must be one of {allowed}")

    def field_equation(
        self, Z_s: complex, sources: Optional[np.ndarray] = None
    ) -> Tuple[str, Callable]:
        """Return (description, callable) for the sector field equation."""
        dispatch = {
            "em": self._em_description,
            "qm": self._qm_description,
            "gr": self._gr_description,
            "ns": self._ns_description,
        }
        return dispatch[self.sector](Z_s, sources)

    # ---- EM sector ------------------------------------------------------

    def em_limit(
        self, Z_s: complex, J_source: np.ndarray
    ) -> np.ndarray:
        r"""Electromagnetic boundary condition:
            n_mu F^{mu i} = Z_s * A^i  -  J^i

        Parameters
        ----------
        Z_s : complex
            Boundary impedance.
        J_source : np.ndarray
            External current source on the boundary.

        Returns
        -------
        np.ndarray
            Residual of the boundary Maxwell equation (shape matches J).
            For a solution, provide A and compute n_mu F externally.
        """
        # Returns a callable that computes the residual given (nF, A)
        def residual(nF: np.ndarray, A: np.ndarray) -> np.ndarray:
            return nF - Z_s * A + J_source
        return residual

    def _em_description(self, Z_s, sources):
        desc = "n_mu F^{mu i} = Z_s * A^i - J^i  (impedance boundary Maxwell)"
        J = sources if sources is not None else np.zeros(3)

        def eqn(nF: np.ndarray, A: np.ndarray) -> np.ndarray:
            return nF - Z_s * A + J
        return desc, eqn

    # ---- QM sector ------------------------------------------------------

    def qm_limit(
        self, Z_s: complex, beta: complex
    ) -> Callable:
        r"""Quantum (Robin) boundary condition:
            n . grad(psi) = alpha * psi  -  beta

        where alpha = Z_s (impedance maps to Robin parameter).

        Returns a callable residual(grad_n_psi, psi) -> residual.
        """
        alpha = Z_s

        def residual(grad_n_psi: np.ndarray, psi: np.ndarray) -> np.ndarray:
            return grad_n_psi - alpha * psi + beta
        return residual

    def _qm_description(self, Z_s, sources):
        desc = "n.grad(psi) = alpha*psi - beta  (Robin BC -> Schroedinger)"
        beta = sources if sources is not None else 0.0

        def eqn(grad_n_psi: np.ndarray, psi: np.ndarray) -> np.ndarray:
            return grad_n_psi - Z_s * psi + beta
        return desc, eqn

    # ---- GR sector ------------------------------------------------------

    def gr_limit(self, Z_s: complex) -> Callable:
        r"""Gravitational (quasilocal stress-energy) boundary condition:
            T^{ij}_boundary = (1/8 pi G)(K_{ij} - K gamma_{ij})

        The impedance Z_s enters through the extrinsic curvature
        normalisation.  Returns a callable computing the Brown-York
        stress tensor.

        Parameters
        ----------
        Z_s : complex
            Gravitational boundary impedance (sets the curvature scale).
        """
        prefactor = 1.0 / (8.0 * np.pi * _G_NEWTON)

        def stress_tensor(
            K_ij: np.ndarray, gamma_ij: np.ndarray
        ) -> np.ndarray:
            """Brown-York quasilocal stress tensor."""
            K_trace = np.trace(K_ij)
            return prefactor * (K_ij - K_trace * gamma_ij)
        return stress_tensor

    def _gr_description(self, Z_s, sources):
        desc = "T^ij_boundary = (1/8piG)(K_ij - K*gamma_ij)  (quasilocal stress)"
        return desc, self.gr_limit(Z_s)

    # ---- NS sector ------------------------------------------------------

    def ns_limit(
        self, Z_s: complex, nu: float
    ) -> Callable:
        r"""Boundary-driven Navier-Stokes condition:
            nu * n.grad(v) = Z_s * v  (impedance slip condition)

        Interpolates between no-slip (|Z_s| -> inf) and free-slip (Z_s -> 0).

        Parameters
        ----------
        Z_s : complex
            Boundary impedance for the fluid interface.
        nu : float
            Kinematic viscosity.
        """
        def residual(
            grad_n_v: np.ndarray, v: np.ndarray
        ) -> np.ndarray:
            return nu * grad_n_v - Z_s * v
        return residual

    def _ns_description(self, Z_s, sources):
        desc = "nu * n.grad(v) = Z_s * v  (impedance slip Navier-Stokes BC)"
        nu = float(np.real(sources)) if sources is not None else 1.0
        return desc, self.ns_limit(Z_s, nu)


# ===========================================================================
#  Anomaly inflow
# ===========================================================================

def anomaly_inflow(
    gauge_charges: np.ndarray,
    n_families: int = 3,
) -> Tuple[float, bool, str]:
    """Check anomaly cancellation via the cubic gauge anomaly Tr[Q^3].

    In the Standard Model, anomaly freedom requires Tr[Q^3] = 0 summed
    over all fermion representations within each family.  The boundary
    action must reproduce this cancellation via anomaly inflow from the
    bulk.

    Parameters
    ----------
    gauge_charges : np.ndarray
        Array of U(1) charges for each fermion representation in one family.
        E.g. SM hypercharges: [1/6, 1/6, 1/6, -1/2, -1/2, 2/3, -1/3, 1].
    n_families : int
        Number of fermion families (default 3).

    Returns
    -------
    tuple
        (Tr_Q3, is_cancelled, constraint_message)
        - Tr_Q3: float, the cubic anomaly coefficient per family
        - is_cancelled: bool, True if |Tr_Q3| < tolerance
        - constraint_message: str, description of the constraint on n_families
    """
    Q = np.asarray(gauge_charges, dtype=float)
    Tr_Q3 = float(np.sum(Q ** 3))
    Tr_Q3_total = n_families * Tr_Q3

    tol = 1e-10
    cancelled = abs(Tr_Q3_total) < tol

    if cancelled:
        msg = (
            f"Anomaly cancelled: Tr[Q^3] = {Tr_Q3:.6e} per family, "
            f"total = {Tr_Q3_total:.6e} for {n_families} families. "
            f"n_families is unconstrained by cubic anomaly alone."
        )
    else:
        msg = (
            f"Anomaly NOT cancelled: Tr[Q^3] = {Tr_Q3:.6e} per family, "
            f"total = {Tr_Q3_total:.6e} for {n_families} families. "
            f"Boundary inflow must compensate {Tr_Q3_total:.6e}."
        )

    return Tr_Q3, cancelled, msg
