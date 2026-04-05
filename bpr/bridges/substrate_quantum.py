"""
Substrate-Quantum-Spacetime Bridge Equations
=============================================

Mathematical bridges connecting the BPR impedance, decoherence, metric,
memory, Hilbert-space, resonance, TDGL, stability, geometry, and
gravitational-wave modules into a unified computational framework.

Each bridge function imports from actual BPR modules, calls their real
interfaces, and returns a dict with all intermediate values for
transparency and testing.

References: Al-Kahwati (2026), *Ten Adjacent Theories*, bridge layer
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_HBAR = 1.054571817e-34   # J s
_K_B = 1.380649e-23       # J/K
_G = 6.67430e-11          # m^3 kg^-1 s^-2
_C = 299792458.0           # m/s
_Z0 = 376.730313668        # Ohm (vacuum impedance)


# ===================================================================
# Bridge 1: Impedance-Modulated Metric Perturbation
# ===================================================================

def impedance_modulated_metric(
    phi_field,
    W: float,
    coupling_lambda: float,
    p: int = 104729,
) -> Dict[str, Any]:
    r"""Impedance-modulated metric perturbation.

    lambda_eff(W) = lambda / Z(W)
    Delta g_mu_nu = lambda_eff * phi(x_boundary) * [geometric coupling]

    The topological impedance Z(W) screens the gravitational coupling:
    high-winding sectors couple more weakly to the metric.

    Parameters
    ----------
    phi_field : symbolic expression, callable, or BoundaryFieldSolution
        Boundary phase field phi(x).
    W : float
        Topological winding number of the excitation.
    coupling_lambda : float
        Bare coupling strength between boundary field and metric.
    p : int
        Substrate prime modulus (sets W_c scale).

    Returns
    -------
    dict with keys: Z_W, lambda_eff, metric_perturbation, screening_factor
    """
    try:
        from ..impedance import TopologicalImpedance
        from ..metric import metric_perturbation, MetricPerturbation
    except ImportError as e:
        raise ImportError(
            "impedance_modulated_metric requires bpr.impedance and bpr.metric"
        ) from e

    # Compute impedance at winding W
    Z_imp = TopologicalImpedance(Z0=_Z0, W_c=np.sqrt(float(p)))
    Z_W = float(Z_imp(W))

    # Effective coupling screened by impedance
    lambda_eff = coupling_lambda / (Z_W / _Z0)  # normalise so Z0 -> no screening

    # Screening factor: how much the coupling is suppressed
    screening_factor = _Z0 / Z_W

    # Compute the metric perturbation with the screened coupling
    mp = metric_perturbation(phi_field, lambda_eff)

    return {
        "Z_W": Z_W,
        "Z0": _Z0,
        "lambda_eff": lambda_eff,
        "screening_factor": screening_factor,
        "metric_perturbation": mp,
        "winding": W,
        "bare_lambda": coupling_lambda,
    }


# ===================================================================
# Bridge 2: Winding-Dependent Decoherence Rate
# ===================================================================

def winding_decoherence_rate(
    W_system: float,
    W_environment: float,
    T: float,
    A_eff: float,
    lambda_dB: float,
    p: int = 104729,
) -> Dict[str, Any]:
    r"""Decoherence rate from winding-number mismatch.

    Delta_Z = |Z(W_sys) - Z(W_env)|
    Gamma_dec = (k_B T / hbar) * (Delta_Z / Z0)^2 * (A_eff / lambda_dB^2)

    Prediction: matched winding (W_sys ~ W_env) yields anomalously
    low decoherence -- a testable signature of BPR.

    Parameters
    ----------
    W_system : float
        Winding number of the quantum system.
    W_environment : float
        Winding number of the environment.
    T : float
        Temperature (K).
    A_eff : float
        Effective boundary surface area (m^2).
    lambda_dB : float
        Thermal de Broglie wavelength (m).
    p : int
        Substrate prime modulus.

    Returns
    -------
    dict with keys: Z_sys, Z_env, delta_Z, gamma_dec, decoherence_time,
                    matched_winding
    """
    try:
        from ..impedance import TopologicalImpedance
        from ..decoherence import DecoherenceRate
    except ImportError as e:
        raise ImportError(
            "winding_decoherence_rate requires bpr.impedance and bpr.decoherence"
        ) from e

    Z_imp = TopologicalImpedance(Z0=_Z0, W_c=np.sqrt(float(p)))
    Z_sys = float(Z_imp(W_system))
    Z_env = float(Z_imp(W_environment))
    delta_Z = abs(Z_sys - Z_env)

    # Use the DecoherenceRate dataclass
    dec = DecoherenceRate(
        T=T,
        Z_system=Z_sys,
        Z_environment=Z_env,
        A_eff=A_eff,
        lambda_dB=lambda_dB,
    )

    gamma = dec.gamma_dec
    tau = dec.decoherence_time

    # Matched-winding flag: anomalously low decoherence
    matched = abs(W_system - W_environment) < 0.1

    return {
        "Z_sys": Z_sys,
        "Z_env": Z_env,
        "delta_Z": delta_Z,
        "gamma_dec": gamma,
        "decoherence_time": tau,
        "matched_winding": matched,
        "T": T,
        "A_eff": A_eff,
        "lambda_dB": lambda_dB,
        "prediction": (
            "Anomalously low decoherence for matched winding"
            if matched
            else "Standard decoherence from impedance mismatch"
        ),
    }


# ===================================================================
# Bridge 3: Memory Kernel as Boundary Phase Operator
# ===================================================================

def memory_kernel_as_operator(
    tau: float,
    omega: float,
    dim: int = 2,
) -> Dict[str, Any]:
    r"""Construct a BoundaryPhaseOperator from memory kernel parameters.

    M(t,t') = exp(-|t-t'|/tau) cos(omega(t-t'))
    -> T = [[A, B], [C, D]]
    where A = exp(-dt/tau)*cos(omega*dt), B = exp(-dt/tau)*sin(omega*dt), etc.

    The operator T propagates the memory state one step forward.
    Spectral radius rho(T) < 1 implies convergent (fading) memory.

    Parameters
    ----------
    tau : float
        Memory decay timescale.
    omega : float
        Memory oscillation frequency (rad/s).
    dim : int
        Dimension of each subspace H_c, H_d (total dim = 2*dim).

    Returns
    -------
    dict with keys: operator, spectral_radius, is_contractive, eigenvalues,
                    convergent_memory
    """
    try:
        from ..hilbert_bpr import BoundaryPhaseOperator
    except ImportError as e:
        raise ImportError(
            "memory_kernel_as_operator requires bpr.hilbert_bpr"
        ) from e

    # Use dt = 1 as the canonical step
    dt = 1.0
    decay = np.exp(-dt / tau) if tau > 0 else 0.0
    c = decay * np.cos(omega * dt)
    s = decay * np.sin(omega * dt)

    # Build block operator: rotation-with-decay structure
    # A (coherent-coherent): preserves oscillation
    # B (damped-coherent): couples damping into coherent sector
    # C (coherent-damped): couples coherent into damped sector
    # D (damped-damped): pure decay
    A = c * np.eye(dim)
    B = s * np.eye(dim)
    C = -s * np.eye(dim)  # antisymmetric coupling
    D = c * np.eye(dim)   # same decay rate in damped sector

    op = BoundaryPhaseOperator(A=A, B=B, C=C, D=D)
    rho = op.spectral_radius()
    contractive = op.is_contractive()
    eigvals = np.linalg.eigvals(op.full_matrix())

    return {
        "operator": op,
        "spectral_radius": rho,
        "is_contractive": contractive,
        "eigenvalues": eigvals,
        "convergent_memory": contractive,
        "tau": tau,
        "omega": omega,
        "decay_per_step": decay,
        "interpretation": (
            "Memory fades (contractive)" if contractive
            else "Memory persists or grows (non-contractive)"
        ),
    }


# ===================================================================
# Bridge 4: TDGL-Landau Phase Transition Bridge
# ===================================================================

def tdgl_landau_bridge(
    T: float,
    T_c: float,
    a_0: float = 1.0,
    beta: float = 1.0,
    kappa: float = 1.0,
    lam: float = 0.5,
) -> Dict[str, Any]:
    r"""Connect Landau mean-field theory to the TDGL simulation.

    alpha(T) = a_0 * (T/T_c - 1)
    T < T_c => alpha < 0 => ordered phase (spontaneous symmetry breaking)
    T > T_c => alpha > 0 => disordered phase

    Parameters
    ----------
    T : float
        Temperature.
    T_c : float
        Critical temperature.
    a_0 : float
        Landau coefficient scale.
    beta : float
        Nonlinear saturation coefficient.
    kappa : float
        Gradient stiffness coefficient.
    lam : float
        Boundary coupling strength.

    Returns
    -------
    dict with keys: alpha, config, phase, order_parameter_eq,
                    correlation_length, sweep_function
    """
    try:
        from ..tdgl_bpr import TDGLConfig
    except ImportError as e:
        raise ImportError(
            "tdgl_landau_bridge requires bpr.tdgl_bpr"
        ) from e

    # Landau coefficient
    alpha = a_0 * (T / T_c - 1.0)

    # Create TDGL configuration
    config = TDGLConfig(alpha=alpha, beta=beta, kappa=kappa, lam=lam)

    # Equilibrium order parameter magnitude
    if alpha < 0 and beta > 0:
        psi_eq = np.sqrt(-alpha / beta)
        phase = "ordered"
    else:
        psi_eq = 0.0
        phase = "disordered"

    # Correlation length xi = sqrt(kappa / |alpha|) (diverges at T_c)
    xi = np.sqrt(kappa / abs(alpha)) if abs(alpha) > 1e-15 else np.inf

    def sweep_critical_exponents(T_values):
        """Sweep temperature and extract critical exponents.

        Returns arrays of (T, alpha, psi_eq, xi) for each temperature.
        """
        results = []
        for T_val in T_values:
            a = a_0 * (T_val / T_c - 1.0)
            if a < 0 and beta > 0:
                psi = np.sqrt(-a / beta)
            else:
                psi = 0.0
            xi_val = np.sqrt(kappa / abs(a)) if abs(a) > 1e-15 else np.inf
            results.append({
                "T": T_val,
                "alpha": a,
                "psi_eq": psi,
                "xi": xi_val,
            })
        return results

    return {
        "alpha": alpha,
        "config": config,
        "phase": phase,
        "order_parameter_eq": psi_eq,
        "correlation_length": xi,
        "T": T,
        "T_c": T_c,
        "sweep_function": sweep_critical_exponents,
    }


# ===================================================================
# Bridge 5: Substrate Spectral Dynamics to Riemann Zeros
# ===================================================================

def substrate_spectral_to_riemann(
    p: int = 7,
    n_sites: int = 4,
    n_steps: int = 256,
) -> Dict[str, Any]:
    r"""Compare Z_p lattice spectral dynamics with Riemann zero spacings.

    Z_p dynamics -> FFT -> peak wavenumbers k_n
    Compare spacing Delta_k with Riemann zero spacing Delta_gamma
    delta_derived = 2*pi / (<Delta_k> * log(gamma_N / 2*pi))

    Parameters
    ----------
    p : int
        Prime modulus for the substrate lattice (must be prime).
    n_sites : int
        Number of lattice sites.
    n_steps : int
        Number of symplectic evolution steps.

    Returns
    -------
    dict with keys: peak_frequencies, riemann_zeros, delta_k_mean,
                    delta_gamma_mean, delta_derived, trajectory
    """
    try:
        from ..rpst.dynamics import SymplecticEvolution
        from ..rpst.substrate import SubstrateState
        from ..resonance import load_riemann_zeros
    except ImportError as e:
        raise ImportError(
            "substrate_spectral_to_riemann requires bpr.rpst.dynamics, "
            "bpr.rpst.substrate, and bpr.resonance"
        ) from e

    # Initialize substrate state
    rng = np.random.default_rng(42)
    q0 = rng.integers(0, p, size=n_sites)
    pi0 = rng.integers(0, p, size=n_sites)
    state0 = SubstrateState(q=q0, pi=pi0, p=p)

    # Create nearest-neighbour coupling
    J = np.zeros((n_sites, n_sites), dtype=int)
    for i in range(n_sites - 1):
        J[i, i + 1] = 1
        J[i + 1, i] = 1

    evo = SymplecticEvolution(p=p, J=J)
    trajectory = evo.evolve(state0, steps=n_steps)

    # Extract q-trajectory as real-valued signal (mod p -> centred)
    q_traj = np.array([s.q for s in trajectory], dtype=float)  # (n_steps+1, n_sites)
    # Centre around p/2
    q_centred = q_traj - p / 2.0

    # FFT of summed signal (total field amplitude)
    signal = q_centred.sum(axis=1)
    fft_vals = np.fft.rfft(signal)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(len(signal))

    # Find peaks (above median power, excluding DC)
    median_power = np.median(power[1:])
    peak_mask = power[1:] > 2.0 * median_power
    peak_indices = np.where(peak_mask)[0] + 1
    peak_freqs = freqs[peak_indices]

    # Spacings between consecutive peaks
    if len(peak_freqs) >= 2:
        delta_k = np.diff(np.sort(peak_freqs))
        delta_k_mean = float(np.mean(delta_k))
    else:
        delta_k = np.array([])
        delta_k_mean = float("nan")

    # Riemann zero spacings
    zeros = load_riemann_zeros(20)
    delta_gamma = np.diff(zeros)
    delta_gamma_mean = float(np.mean(delta_gamma))

    # Derive fractal exponent
    # delta = 2*pi / (<Delta_k> * log(gamma_N / 2*pi))
    gamma_N = zeros[-1]
    if delta_k_mean > 0 and not np.isnan(delta_k_mean):
        log_factor = np.log(gamma_N / (2.0 * np.pi))
        delta_derived = (2.0 * np.pi) / (delta_k_mean * log_factor) if log_factor > 0 else float("nan")
    else:
        delta_derived = float("nan")

    return {
        "peak_frequencies": peak_freqs,
        "peak_power": power[peak_indices] if len(peak_indices) > 0 else np.array([]),
        "riemann_zeros": zeros,
        "delta_k": delta_k,
        "delta_k_mean": delta_k_mean,
        "delta_gamma": delta_gamma,
        "delta_gamma_mean": delta_gamma_mean,
        "delta_derived": delta_derived,
        "trajectory_shape": q_traj.shape,
        "p": p,
        "n_sites": n_sites,
        "n_steps": n_steps,
    }


# ===================================================================
# Bridge 6: Geometry to Gravitational Wave Strain
# ===================================================================

def geometry_to_gw_strain(
    boundary_type: str = "sphere",
    radius: float = 1.0,
    coupling_lambda: float = 0.1,
    freq_Hz: float = 100.0,
) -> Dict[str, Any]:
    r"""Boundary shape -> eigenmode spectrum -> GW quadrupole emission.

    Boundary shape -> eigenmode spectrum -> time-evolving phi(x,t)
    -> quadrupole Q_ij(t) -> GW power P = G/(5c^5)|Q_dddot|^2

    Parameters
    ----------
    boundary_type : str
        Boundary geometry: "sphere" or "cylinder".
    radius : float
        Characteristic radius of the boundary (m).
    coupling_lambda : float
        Coupling strength for the boundary field.
    freq_Hz : float
        Characteristic frequency of the boundary oscillation (Hz).

    Returns
    -------
    dict with keys: mesh, quadrupole, power, strain_estimate, freq_Hz
    """
    try:
        from ..geometry import make_boundary
        from ..gravitational_waves import GWQuadrupole
    except ImportError as e:
        raise ImportError(
            "geometry_to_gw_strain requires bpr.geometry and "
            "bpr.gravitational_waves"
        ) from e

    # Build boundary mesh (make_boundary returns BoundaryMesh; inner .mesh is NumpyMesh)
    boundary = make_boundary(mesh_size=0.3, geometry=boundary_type, radius=radius)
    mesh = boundary.mesh
    coords = mesh.coordinates()  # (n_points, 3)
    n_points = coords.shape[0]

    # Construct time-evolving field from eigenmode superposition
    # phi(x, t) = lambda * sum_n A_n * f_n(x) * cos(2*pi*n*freq*t)
    omega = 2.0 * np.pi * freq_Hz
    n_times = 128
    dt = 1.0 / (4.0 * freq_Hz)  # quarter-period steps
    times = np.arange(n_times) * dt

    # Use spherical harmonics approximation: radial modes
    r_pts = np.linalg.norm(coords, axis=1)
    r_pts = np.where(r_pts > 0, r_pts, 1e-10)

    # Superpose first 3 modes
    phi = np.zeros((n_points, n_times))
    for n_mode in range(1, 4):
        # Spatial part: standing wave on boundary
        spatial = np.sin(n_mode * np.pi * r_pts / radius) / r_pts
        # Temporal part
        for t_idx, t_val in enumerate(times):
            phi[:, t_idx] += (coupling_lambda / n_mode) * spatial * np.cos(
                n_mode * omega * t_val
            )

    # Compute quadrupole moment
    Q = GWQuadrupole.compute_quadrupole(phi, coords, dt)

    # Third time-derivative via finite differences
    if n_times >= 4:
        # d^3Q/dt^3 via central differences
        Q3 = np.zeros_like(Q)
        for t_idx in range(2, n_times - 2):
            Q3[:, :, t_idx] = (
                -0.5 * Q[:, :, t_idx - 2]
                + Q[:, :, t_idx - 1]
                - Q[:, :, t_idx + 1]
                + 0.5 * Q[:, :, t_idx + 2]
            ) / dt ** 3
        Q3_valid = Q3[:, :, 2:-2]
    else:
        Q3_valid = np.zeros((3, 3, 1))

    # GW power
    gw = GWQuadrupole(Q_phi_dddot=Q3_valid)
    power = gw.power

    # Strain estimate at distance r_obs
    r_obs = 1e6  # 1 Mpc for illustration
    # h ~ (2G / (r c^4)) * Q_ddot
    Q_ddot_rms = np.sqrt(np.mean(np.diff(Q[:, :, :], n=2, axis=2) ** 2))
    strain = (2.0 * _G / (r_obs * _C ** 4)) * Q_ddot_rms

    return {
        "mesh_vertices": n_points,
        "quadrupole_shape": Q.shape,
        "power": power,
        "strain_estimate": strain,
        "freq_Hz": freq_Hz,
        "radius": radius,
        "boundary_type": boundary_type,
        "n_times": n_times,
        "dt": dt,
    }


# ===================================================================
# Bridge 7: Hilbert Stability Analysis
# ===================================================================

def hilbert_stability_bridge(
    T_operator,
) -> Dict[str, Any]:
    r"""Map BoundaryPhaseOperator spectral data to stability analysis.

    rho(T) -> Lyapunov exponent lambda_L = ln(rho)
    Stability manifold S = {x : E_Phi = 0} <-> ker(T - I)
    dim(H_res) = number of zero modes on stability manifold

    Parameters
    ----------
    T_operator : BoundaryPhaseOperator
        Block operator T = [[A,B],[C,D]] on H_c (+) H_d.

    Returns
    -------
    dict with keys: spectral_radius, lyapunov_exponent, eigenvalues,
                    n_zero_modes, H_res_dim, H_damp_dim, stable
    """
    try:
        from ..hilbert_bpr import BoundaryPhaseOperator
    except ImportError as e:
        raise ImportError(
            "hilbert_stability_bridge requires bpr.hilbert_bpr"
        ) from e

    T_mat = T_operator.full_matrix()
    eigvals = np.linalg.eigvals(T_mat)
    rho = float(np.max(np.abs(eigvals)))

    # Lyapunov exponent: lambda_L = ln(rho)
    lyapunov = np.log(rho) if rho > 0 else -np.inf

    # Zero modes: eigenvalues with |lambda| ~ 1 (on the unit circle)
    unit_circle_tol = 1e-6
    on_unit_circle = np.abs(np.abs(eigvals) - 1.0) < unit_circle_tol
    n_zero_modes = int(np.sum(on_unit_circle))

    # Decompose into resonant (|lambda| ~ 1) and damped (|lambda| < 1) subspaces
    inside_circle = np.abs(eigvals) < 1.0 - unit_circle_tol
    H_res_dim = n_zero_modes
    H_damp_dim = int(np.sum(inside_circle))

    # Kernel of (T - I): exact fixed points
    T_minus_I = T_mat - np.eye(T_mat.shape[0])
    sv = np.linalg.svd(T_minus_I, compute_uv=False)
    kernel_dim = int(np.sum(sv < 1e-10))

    stable = rho < 1.0

    return {
        "spectral_radius": rho,
        "lyapunov_exponent": lyapunov,
        "eigenvalues": eigvals,
        "eigenvalue_magnitudes": np.abs(eigvals),
        "n_zero_modes": n_zero_modes,
        "kernel_dim": kernel_dim,
        "H_res_dim": H_res_dim,
        "H_damp_dim": H_damp_dim,
        "stable": stable,
        "classification": (
            "asymptotically stable" if rho < 1.0
            else "marginally stable" if abs(rho - 1.0) < unit_circle_tol
            else "unstable"
        ),
    }


# ===================================================================
# Bridge 8: Fractal Spectral Degeneracy
# ===================================================================

def fractal_spectral_degeneracy(
    D_S: float,
    n_eigenvalues: int = 20,
    sigma: float = 1.0,
) -> Dict[str, Any]:
    r"""Spectral degeneracy enhancement on fractal boundaries.

    J_deg(D_S) = Sum_{i<j} exp(-(lambda_i - lambda_j)^2 / (2*sigma^2))
    On fractal boundary: lambda_i ~ i^{2/D_S}
    Prediction: J_deg ~ L^{2*D_S - 2}

    Compares spectral crowding for fractal (D_S != 2) vs smooth (D_S = 2)
    boundaries.

    Parameters
    ----------
    D_S : float
        Fractal (Hausdorff) dimension of the boundary.
    n_eigenvalues : int
        Number of eigenvalues to generate.
    sigma : float
        Width parameter for the near-degeneracy functional.

    Returns
    -------
    dict with keys: eigenvalues_fractal, eigenvalues_smooth, J_fractal,
                    J_smooth, enhancement_ratio, D_S
    """
    try:
        from ..geometry import near_degeneracy_functional
    except ImportError as e:
        raise ImportError(
            "fractal_spectral_degeneracy requires bpr.geometry"
        ) from e

    indices = np.arange(1, n_eigenvalues + 1, dtype=float)

    # Fractal boundary eigenvalues: Weyl law generalisation
    # lambda_i ~ i^{2/D_S}
    eigenvalues_fractal = indices ** (2.0 / D_S)

    # Smooth boundary eigenvalues (D_S = 2): lambda_i ~ i
    eigenvalues_smooth = indices ** (2.0 / 2.0)  # = i^1

    # Near-degeneracy functional from geometry.py
    J_fractal = near_degeneracy_functional(eigenvalues_fractal, sigma=sigma)
    J_smooth = near_degeneracy_functional(eigenvalues_smooth, sigma=sigma)

    # Enhancement ratio
    enhancement = J_fractal / J_smooth if J_smooth > 0 else float("inf")

    # Theoretical prediction: J ~ L^{2*D_S - 2}
    # For D_S > 2, enhancement grows with system size
    # For D_S < 2, degeneracy is suppressed
    predicted_scaling_exponent = 2.0 * D_S - 2.0

    return {
        "eigenvalues_fractal": eigenvalues_fractal,
        "eigenvalues_smooth": eigenvalues_smooth,
        "J_fractal": J_fractal,
        "J_smooth": J_smooth,
        "enhancement_ratio": enhancement,
        "D_S": D_S,
        "predicted_scaling_exponent": predicted_scaling_exponent,
        "n_eigenvalues": n_eigenvalues,
        "sigma": sigma,
        "prediction": (
            f"Fractal degeneracy enhancement ~ L^{predicted_scaling_exponent:.2f}; "
            f"J_fractal/J_smooth = {enhancement:.4f}"
        ),
    }
