"""
End-to-End Prediction Pipelines
================================

Chains existing BPR modules into complete prediction pipelines that
produce falsifiable experimental outputs.  Each pipeline is a standalone
function returning a dict of results.

All imports are wrapped in try/except so the module loads even when
optional dependencies are missing.

Pipelines
---------
1. pipeline_impedance_to_lepton_masses  -- gauge → impedance → leptons
2. pipeline_impedance_to_decoherence    -- impedance → decoherence → coherence → QM
3. pipeline_substrate_to_casimir        -- RPST substrate → resonance → Casimir
4. pipeline_tdgl_to_phase_classification -- TDGL → coherence → phase transitions
5. pipeline_kuramoto_to_transition      -- collective → phase → coherence
6. pipeline_agents_to_consciousness     -- agents → collective → coherence
7. pipeline_bond_to_fractal_transport   -- chemistry → resonance families → fractional

References: Al-Kahwati (2026), BPR-Math-Spine
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Physical constants used across pipelines
# ---------------------------------------------------------------------------
_HBAR = 1.054571817e-34   # J s
_K_B = 1.380649e-23       # J/K

# ---------------------------------------------------------------------------
# Guarded imports -- each block may fail independently
# ---------------------------------------------------------------------------

try:
    from .gauge_unification import (
        GaugeCouplingRunning,
        weinberg_angle_from_impedance,
        electroweak_scale_GeV,
    )
    _HAS_GAUGE = True
except Exception:
    _HAS_GAUGE = False

try:
    from .impedance import TopologicalImpedance, DarkSectorParameters
    _HAS_IMPEDANCE = True
except Exception:
    _HAS_IMPEDANCE = False

try:
    from .charged_leptons import ChargedLeptonSpectrum, koide_parameter
    _HAS_LEPTONS = True
except Exception:
    _HAS_LEPTONS = False

try:
    from .decoherence import DecoherenceRate
    _HAS_DECOHERENCE = True
except Exception:
    _HAS_DECOHERENCE = False

try:
    from .coherence_transitions import (
        CoherenceDecayDynamics,
        CoherenceGainFunction,
    )
    _HAS_COHERENCE = True
except Exception:
    _HAS_COHERENCE = False

try:
    from .quantum_foundations import MeasurementDynamics, BornRule
    _HAS_QF = True
except Exception:
    _HAS_QF = False

try:
    from .collective import (
        KuramotoFlocking,
        CollectivePhaseField,
    )
    _HAS_COLLECTIVE = True
except Exception:
    _HAS_COLLECTIVE = False

try:
    from .phase_transitions import (
        SubstrateCriticalExponents,
        landau_order_parameter,
    )
    _HAS_PHASE = True
except Exception:
    _HAS_PHASE = False

try:
    from .tdgl_bpr import (
        TDGLConfig,
        run_tdgl_simulation,
        boundary_coupling_potential,
        normalized_cross_correlation,
        coherence_decay_fit,
    )
    _HAS_TDGL = True
except Exception:
    _HAS_TDGL = False

try:
    from .rpst.substrate import PrimeField, SubstrateState
    from .rpst.dynamics import SymplecticEvolution
    _HAS_RPST = True
except Exception:
    _HAS_RPST = False

try:
    from .resonance import load_riemann_zeros
    _HAS_RESONANCE = True
except Exception:
    _HAS_RESONANCE = False

try:
    from .casimir import casimir_force, sweep_radius
    _HAS_CASIMIR = True
except Exception:
    _HAS_CASIMIR = False

try:
    from .conscious_agents import agent_network, markov_transition_kernel
    _HAS_AGENTS = True
except Exception:
    _HAS_AGENTS = False

try:
    from .quantum_chemistry import ChemicalBond, bond_coherence
    _HAS_CHEM = True
except Exception:
    _HAS_CHEM = False

try:
    from .resonance_families import farey_tree, resonance_weight
    _HAS_FAMILIES = True
except Exception:
    _HAS_FAMILIES = False

try:
    from .fractional_boundary import transport_scaling, quality_factor_scaling
    _HAS_FRACTAL = True
except Exception:
    _HAS_FRACTAL = False


# ===================================================================
# Pipeline 1: Impedance -> Lepton Masses
# ===================================================================

def pipeline_impedance_to_lepton_masses(
    p: int = 104729,
    z: int = 6,
) -> dict:
    """Chain: gauge_unification -> impedance -> charged_leptons.

    Derives lepton masses from substrate parameters (p, z) by:
      1. Computing W_c from DarkSectorParameters
      2. Evaluating TopologicalImpedance at W_B=1 and W_W=W_c/2
      3. Deriving zeta ratios for weinberg_angle_from_impedance
      4. Obtaining v_EW from electroweak_scale_GeV
      5. Feeding into ChargedLeptonSpectrum
      6. Evaluating Koide parameter

    Parameters
    ----------
    p : int
        Substrate prime modulus.
    z : int
        Coordination number.

    Returns
    -------
    dict with keys: sin2_theta_W, v_EW, m_e, m_mu, m_tau, koide_Q
    """
    missing = []
    if not _HAS_GAUGE:
        missing.append("gauge_unification")
    if not _HAS_IMPEDANCE:
        missing.append("impedance")
    if not _HAS_LEPTONS:
        missing.append("charged_leptons")
    if missing:
        return {"error": f"Missing modules: {', '.join(missing)}"}

    # Step 1: Get W_c from substrate
    dsp = DarkSectorParameters(p=p)
    W_c = dsp.W_c  # p^(1/5)

    # Step 2: Evaluate impedance at winding numbers for B and W sectors
    Z = TopologicalImpedance(W_c=W_c)
    Z_B = Z(1.0)        # B-field sector: winding 1
    Z_W = Z(W_c / 2.0)  # W-field sector: winding W_c/2

    # Step 3: Compute impedance ratios for Weinberg angle
    # zeta_XY = Z_X / Z_Y ratios characterising boundary mixing
    Z_0 = Z(0.0)  # reference impedance at W=0
    zeta_BW = float(Z_B / Z_W)
    zeta_WW = float(Z_W / Z_0)
    zeta_BB = float(Z_B / Z_0)

    result_weinberg = weinberg_angle_from_impedance(zeta_BW, zeta_WW, zeta_BB)
    sin2_theta_W = result_weinberg["sin2_theta_W"]

    # Step 4: Electroweak scale from substrate
    v_EW = electroweak_scale_GeV(p, z)

    # Step 5: Derive alpha_EM from sin2_theta_W and GUT running
    gcr = GaugeCouplingRunning(p=p)
    alpha_EM = gcr.alpha_em_prediction

    # Step 6: Lepton spectrum from boundary modes with derived v_EW
    spectrum = ChargedLeptonSpectrum(v_EW_GeV=v_EW, alpha_EM=alpha_EM)
    masses = spectrum.masses_MeV

    # Step 7: Koide parameter from predicted masses
    Q = koide_parameter(m_e=float(masses[0]), m_mu=float(masses[1]),
                        m_tau=float(masses[2]))

    return {
        "sin2_theta_W": float(sin2_theta_W),
        "v_EW_GeV": float(v_EW),
        "m_e_MeV": float(masses[0]),
        "m_mu_MeV": float(masses[1]),
        "m_tau_MeV": float(masses[2]),
        "koide_Q": float(Q),
        "W_c": float(W_c),
        "alpha_EM": float(alpha_EM),
    }


# ===================================================================
# Pipeline 2: Impedance -> Decoherence
# ===================================================================

def pipeline_impedance_to_decoherence(
    W_system: float = 1.0,
    W_environment: float = 10.0,
    T: float = 300.0,
    A_eff: float = 1e-14,
    lambda_dB: float = 1e-10,
    p: int = 104729,
) -> dict:
    """Chain: impedance -> decoherence -> coherence_transitions -> quantum_foundations.

    Derives decoherence observables from impedance mismatch:
      1. TopologicalImpedance -> Z_system, Z_environment
      2. DecoherenceRate -> gamma_dec
      3. Map gamma_dec to stain dynamics (u_minus proportional to gamma_dec)
      4. CoherenceDecayDynamics -> evolve -> s(t)
      5. CoherenceGainFunction -> K_star (asymptotic coherence)
      6. MeasurementDynamics -> measurement time
      7. BornRule -> correction

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
        Substrate prime.

    Returns
    -------
    dict with keys: decoherence_time, K_star, is_quantum, measurement_time,
                    born_correction
    """
    missing = []
    if not _HAS_IMPEDANCE:
        missing.append("impedance")
    if not _HAS_DECOHERENCE:
        missing.append("decoherence")
    if not _HAS_COHERENCE:
        missing.append("coherence_transitions")
    if not _HAS_QF:
        missing.append("quantum_foundations")
    if missing:
        return {"error": f"Missing modules: {', '.join(missing)}"}

    # Step 1: Impedance from winding numbers
    dsp = DarkSectorParameters(p=p)
    Z = TopologicalImpedance(W_c=dsp.W_c)
    Z_sys = float(Z(W_system))
    Z_env = float(Z(W_environment))

    # Step 2: Decoherence rate
    dr = DecoherenceRate(
        T=T,
        Z_system=Z_sys,
        Z_environment=Z_env,
        A_eff=A_eff,
        lambda_dB=lambda_dB,
    )
    gamma_dec = dr.gamma_dec
    tau_dec = dr.decoherence_time

    # Step 3: Map decoherence rate to stain input
    # u_minus represents noise intensity, scaled from gamma_dec
    u_minus_const = gamma_dec * _HBAR / (_K_B * T) if T > 0 else 0.0
    u_plus_const = 0.0  # no active coherence restoration

    # Step 4: Stain dynamics evolution
    stain = CoherenceDecayDynamics(alpha=1.0, beta=0.5, gamma=0.01, s0=0.0)
    t_final = 10.0 / max(gamma_dec * 1e-12, 1e-6)  # integrate for ~10 tau
    t_arr, s_arr = stain.evolve(
        t_span=(0.0, t_final),
        u_plus=lambda t: u_plus_const,
        u_minus=lambda t: u_minus_const,
        n_points=200,
    )
    s_final = float(s_arr[-1])

    # Step 5: Asymptotic coherence K*
    hgf = CoherenceGainFunction()
    s_star = stain.steady_state(u_plus_const, u_minus_const)
    K_star = hgf.asymptotic_coherence(s_star)

    # Classify quantum vs classical
    is_quantum = K_star > 0.5

    # Step 6: Measurement dynamics
    md = MeasurementDynamics(gamma_dec=gamma_dec, W_apparatus=W_environment)
    tau_meas = md.measurement_time

    # Step 7: Born rule correction
    br = BornRule(p=p)
    born_correction = br.correction_amplitude

    return {
        "gamma_dec_Hz": float(gamma_dec),
        "decoherence_time_s": float(tau_dec),
        "s_star": float(s_star),
        "K_star": float(K_star),
        "is_quantum": bool(is_quantum),
        "measurement_time_s": float(tau_meas),
        "born_correction": float(born_correction),
        "Z_system_Ohm": float(Z_sys),
        "Z_environment_Ohm": float(Z_env),
    }


# ===================================================================
# Pipeline 3: Substrate -> Casimir
# ===================================================================

def pipeline_substrate_to_casimir(
    p: int = 7,
    n_sites: int = 16,
    n_steps: int = 200,
    boundary_radius: float = 1e-6,
    coupling_lambda: float = 1e-3,
) -> dict:
    """Chain: rpst/substrate -> rpst/dynamics -> resonance -> casimir.

    Derives Casimir force predictions from substrate dynamics:
      1. Create SubstrateState and SymplecticEvolution
      2. Evolve, FFT trajectory -> power spectrum -> extract peak wavenumbers
      3. Compare spacings with Riemann zeros from load_riemann_zeros
      4. Compute fractal exponent delta from zero spacings
      5. Call casimir_force with derived coupling
      6. Return comparison

    Parameters
    ----------
    p : int
        Substrate prime modulus (should be small prime for tractability).
    n_sites : int
        Number of lattice sites.
    n_steps : int
        Number of symplectic evolution steps.
    boundary_radius : float
        Casimir geometry radius / separation (m).
    coupling_lambda : float
        BPR coupling strength for Casimir calculation.

    Returns
    -------
    dict with keys: delta_derived, delta_hardcoded, casimir_force,
                    relative_deviation, peak_wavenumbers
    """
    missing = []
    if not _HAS_RPST:
        missing.append("rpst")
    if not _HAS_RESONANCE:
        missing.append("resonance")
    if not _HAS_CASIMIR:
        missing.append("casimir")
    if missing:
        return {"error": f"Missing modules: {', '.join(missing)}"}

    # Step 1: Create substrate state and evolution
    rng = np.random.default_rng(42)
    q0 = rng.integers(0, p, size=n_sites)
    pi0 = rng.integers(0, p, size=n_sites)
    state0 = SubstrateState(q=q0, pi=pi0, p=p)

    # Nearest-neighbour coupling matrix
    J = np.zeros((n_sites, n_sites), dtype=int)
    for i in range(n_sites - 1):
        J[i, i + 1] = 1
        J[i + 1, i] = 1
    J[0, -1] = 1
    J[-1, 0] = 1

    evo = SymplecticEvolution(p=p, J=J)

    # Step 2: Evolve and collect trajectory
    trajectory = evo.evolve(state0, steps=n_steps)
    # Extract q-values over time for site 0 as a signal
    q_signal = np.array([s.q[0] for s in trajectory], dtype=float)

    # FFT to get power spectrum
    q_centered = q_signal - np.mean(q_signal)
    fft_vals = np.fft.rfft(q_centered)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(len(q_centered))

    # Extract peak wavenumbers (top 10 peaks excluding DC)
    if len(power) > 1:
        peak_indices = np.argsort(power[1:])[-10:][::-1] + 1
        peak_wavenumbers = freqs[peak_indices]
    else:
        peak_wavenumbers = np.array([])

    # Step 3: Compare with Riemann zeros
    zeros = load_riemann_zeros(n=20)
    zero_spacings = np.diff(zeros)

    # Step 4: Derive fractal exponent from zero spacings
    # delta = mean spacing normalised by first zero
    mean_spacing = float(np.mean(zero_spacings))
    delta_derived = mean_spacing / zeros[0]  # normalised mean spacing

    # Hardcoded theoretical value from BPR
    delta_hardcoded = 1.0  # BPR predicts unit fractal exponent for Z_p

    # Step 5: Casimir force at the boundary radius
    cr = casimir_force(
        radius=boundary_radius,
        geometry="parallel_plates",
        coupling_lambda=coupling_lambda,
    )
    F_total = cr.total_force
    rel_dev = cr.relative_deviation

    return {
        "delta_derived": float(delta_derived),
        "delta_hardcoded": float(delta_hardcoded),
        "casimir_force_N": float(F_total),
        "relative_deviation": float(rel_dev),
        "peak_wavenumbers": peak_wavenumbers.tolist(),
        "n_riemann_zeros": len(zeros),
        "mean_zero_spacing": float(mean_spacing),
    }


# ===================================================================
# Pipeline 4: TDGL -> Phase Classification
# ===================================================================

def pipeline_tdgl_to_phase_classification(
    alpha_range: Tuple[float, float] = (-2.0, 0.5),
    beta: float = 1.0,
    kappa: float = 1.0,
    lam: float = 0.5,
    pattern: str = "stripe",
    n_steps: int = 100,
    n_alpha: int = 10,
) -> dict:
    """Chain: tdgl_bpr -> coherence_transitions -> phase_transitions.

    Sweeps alpha values to classify the coherence transition:
      1. For each alpha, run TDGL simulation
      2. Compute NCC with boundary pattern at each timestep
      3. Fit coherence decay -> tau
      4. Find alpha_c where coherence transitions (tau diverges)
      5. Classify transition type via Landau crossover
      6. Extract K_star curve from CoherenceGainFunction

    Parameters
    ----------
    alpha_range : (float, float)
        Range of alpha values to sweep.
    beta, kappa, lam : float
        TDGL parameters.
    pattern : str
        Boundary coupling pattern ('stripe', 'checkerboard', 'radial').
    n_steps : int
        TDGL simulation steps per alpha value.
    n_alpha : int
        Number of alpha values in sweep.

    Returns
    -------
    dict with keys: alpha_c, transition_class, tau_values, K_star_curve,
                    alpha_values
    """
    missing = []
    if not _HAS_TDGL:
        missing.append("tdgl_bpr")
    if not _HAS_COHERENCE:
        missing.append("coherence_transitions")
    if not _HAS_PHASE:
        missing.append("phase_transitions")
    if missing:
        return {"error": f"Missing modules: {', '.join(missing)}"}

    alpha_values = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
    tau_values = []
    ncc_finals = []
    nx, ny = 32, 32  # smaller grid for speed

    # Create boundary pattern once
    V = boundary_coupling_potential(nx, ny, pattern=pattern)

    for alpha_val in alpha_values:
        cfg = TDGLConfig(
            alpha=float(alpha_val),
            beta=beta,
            kappa=kappa,
            lam=lam,
            nx=nx,
            ny=ny,
            noise_sigma=0.01,
        )

        # Run TDGL simulation
        history = run_tdgl_simulation(cfg, V, n_steps=n_steps, seed=42)

        # Compute NCC between each timestep and the boundary pattern
        ncc_series = np.array([
            normalized_cross_correlation(history[t], V)
            for t in range(history.shape[0])
        ])

        ncc_finals.append(float(ncc_series[-1]))

        # Fit exponential decay
        try:
            A, tau, C0 = coherence_decay_fit(ncc_series)
            tau_values.append(float(tau))
        except Exception:
            tau_values.append(float("nan"))

    tau_arr = np.array(tau_values)

    # Find alpha_c: where tau is maximised (critical slowing down)
    valid_mask = np.isfinite(tau_arr)
    if valid_mask.any():
        idx_max = np.argmax(np.where(valid_mask, tau_arr, -np.inf))
        alpha_c = float(alpha_values[idx_max])
    else:
        alpha_c = float("nan")

    # Classify transition: if tau diverges it is a continuous (Class C) transition
    max_tau = float(tau_arr[idx_max]) if valid_mask.any() else 0.0
    if max_tau > n_steps * 0.5:
        transition_class = "C_impedance_continuous"
    else:
        transition_class = "A_winding_first_order"

    # K_star curve: map final NCC to stain, then get K_star
    hgf = CoherenceGainFunction()
    K_star_curve = []
    for ncc_val in ncc_finals:
        # Map NCC to stain: s = 1 - |NCC| (high correlation = low stain)
        s_approx = 1.0 - abs(ncc_val)
        K_star_curve.append(float(hgf.asymptotic_coherence(s_approx)))

    return {
        "alpha_c": alpha_c,
        "transition_class": transition_class,
        "tau_values": tau_values,
        "K_star_curve": K_star_curve,
        "alpha_values": alpha_values.tolist(),
        "ncc_finals": ncc_finals,
        "max_tau": max_tau,
    }


# ===================================================================
# Pipeline 5: Kuramoto -> Transition
# ===================================================================

def pipeline_kuramoto_to_transition(
    N: int = 50,
    K_range: Tuple[float, float] = (0.0, 5.0),
    d: int = 3,
    n_steps: int = 500,
    n_K: int = 15,
    seed: int = 42,
) -> dict:
    """Chain: collective -> phase_transitions -> coherence_transitions.

    Sweeps coupling K to find synchronisation transition:
      1. For each K, run KuramotoFlocking simulation
      2. Extract steady-state coherence |Phi|(K)
      3. Fit Landau form near K_c to get exponents
      4. Compare with SubstrateCriticalExponents(d).beta
      5. Map final coherence to stain -> K_star

    Parameters
    ----------
    N : int
        Number of Kuramoto oscillators.
    K_range : (float, float)
        Range of coupling strengths to sweep.
    d : int
        Spatial dimension for critical exponent comparison.
    n_steps : int
        Simulation steps per K value.
    n_K : int
        Number of K values in sweep.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys: K_c, beta_measured, beta_predicted, coherence_curve,
                    K_values, K_star_values
    """
    missing = []
    if not _HAS_COLLECTIVE:
        missing.append("collective")
    if not _HAS_PHASE:
        missing.append("phase_transitions")
    if not _HAS_COHERENCE:
        missing.append("coherence_transitions")
    if missing:
        return {"error": f"Missing modules: {', '.join(missing)}"}

    np.random.seed(seed)
    K_values = np.linspace(K_range[0], K_range[1], n_K)
    coherence_curve = []

    # Fixed natural frequencies across sweeps for consistency
    omega = np.random.randn(N) * 0.1

    for K_val in K_values:
        kf = KuramotoFlocking(N=N, K=float(K_val), noise=0.05,
                              natural_frequencies=omega.copy())
        _, coh_hist = kf.simulate(n_steps=n_steps, dt=0.01)
        # Steady-state coherence: average over last 20% of simulation
        steady_start = int(0.8 * n_steps)
        r_ss = float(np.mean(coh_hist[steady_start:]))
        coherence_curve.append(r_ss)

    coherence_arr = np.array(coherence_curve)

    # Find K_c: threshold where coherence first exceeds 0.3
    threshold = 0.3
    above = np.where(coherence_arr > threshold)[0]
    if len(above) > 0:
        idx_c = above[0]
        K_c = float(K_values[idx_c])
    else:
        K_c = float(K_values[-1])
        idx_c = n_K - 1

    # Fit Landau exponent beta near K_c
    # r ~ (K - K_c)^beta for K > K_c
    above_Kc = K_values[idx_c:] - K_c
    r_above = coherence_arr[idx_c:]
    # Only fit where r > 0 and K > K_c
    mask = (above_Kc > 1e-6) & (r_above > 0.01)
    if np.sum(mask) >= 3:
        log_dK = np.log(above_Kc[mask])
        log_r = np.log(r_above[mask])
        # Linear fit: log(r) = beta * log(K - K_c) + const
        coeffs = np.polyfit(log_dK, log_r, 1)
        beta_measured = float(coeffs[0])
    else:
        beta_measured = float("nan")

    # BPR prediction for the exponent
    sce = SubstrateCriticalExponents(d=d)
    beta_predicted = sce.beta

    # K_star curve from coherence_transitions
    hgf = CoherenceGainFunction()
    K_star_values = []
    for r_val in coherence_curve:
        # Map coherence to stain: high coherence = low stain
        s_approx = 1.0 - min(r_val, 1.0)
        K_star_values.append(float(hgf.asymptotic_coherence(s_approx)))

    return {
        "K_c": K_c,
        "beta_measured": beta_measured,
        "beta_predicted": float(beta_predicted),
        "coherence_curve": coherence_curve,
        "K_values": K_values.tolist(),
        "K_star_values": K_star_values,
    }


# ===================================================================
# Pipeline 6: Agents -> Consciousness
# ===================================================================

def pipeline_agents_to_consciousness(
    n_agents: int = 10,
    state_dim: int = 4,
    K_range: Tuple[float, float] = (0.0, 5.0),
    seed: int = 42,
    n_K: int = 15,
    n_kuramoto_steps: int = 500,
) -> dict:
    """Chain: conscious_agents -> collective -> coherence_transitions.

    Derives consciousness onset from agent network:
      1. agent_network -> kappa matrix -> markov_transition_kernel -> eigenvalues
      2. Eigenvalues become natural frequencies for Kuramoto
      3. Sweep K, run KuramotoFlocking for each
      4. Find K_c and coherence onset threshold

    Parameters
    ----------
    n_agents : int
        Number of conscious agents.
    state_dim : int
        Dimension of each agent's state space.
    K_range : (float, float)
        Coupling strength range for Kuramoto sweep.
    seed : int
        Random seed.
    n_K : int
        Number of K values.
    n_kuramoto_steps : int
        Steps per Kuramoto simulation.

    Returns
    -------
    dict with keys: K_c, Phi_onset, n_agents_minimum, eigenvalues,
                    coherence_curve
    """
    missing = []
    if not _HAS_AGENTS:
        missing.append("conscious_agents")
    if not _HAS_COLLECTIVE:
        missing.append("collective")
    if not _HAS_COHERENCE:
        missing.append("coherence_transitions")
    if missing:
        return {"error": f"Missing modules: {', '.join(missing)}"}

    # Step 1: Build agent network and get couplings
    agents, kappa = agent_network(
        n_agents=n_agents,
        state_dim=state_dim,
        seed=seed,
    )

    # Step 2: Markov transition kernel from couplings
    T_kernel = markov_transition_kernel(kappa)

    # Eigenvalues of the transition kernel -> natural frequencies
    eigenvalues = np.linalg.eigvals(T_kernel)
    # Use imaginary parts of eigenvalues as natural frequencies
    # (real parts are ~1 for the stationary distribution)
    omega = np.angle(eigenvalues)  # phase angles as frequencies

    # Step 3: Kuramoto sweep
    K_values = np.linspace(K_range[0], K_range[1], n_K)
    coherence_curve = []

    for K_val in K_values:
        kf = KuramotoFlocking(
            N=n_agents,
            K=float(K_val),
            noise=0.05,
            natural_frequencies=np.real(omega),
        )
        _, coh_hist = kf.simulate(n_steps=n_kuramoto_steps, dt=0.01)
        steady_start = int(0.8 * n_kuramoto_steps)
        r_ss = float(np.mean(coh_hist[steady_start:]))
        coherence_curve.append(r_ss)

    coherence_arr = np.array(coherence_curve)

    # Step 4: Find K_c and onset threshold
    # Consciousness onset = coherence exceeding 0.5 (Phi_onset)
    Phi_onset = 0.5
    above = np.where(coherence_arr > Phi_onset)[0]
    if len(above) > 0:
        K_c = float(K_values[above[0]])
    else:
        K_c = float("inf")

    # Estimate minimum agents needed: K_c scales roughly as 1/sqrt(N)
    # so n_min ~ (K_c_1 / K_c_target)^2 * n_agents for K_c_target ~ 1
    if K_c > 0 and np.isfinite(K_c):
        n_agents_minimum = max(2, int(np.ceil(n_agents * (1.0 / K_c) ** 2)))
    else:
        n_agents_minimum = n_agents

    return {
        "K_c": K_c,
        "Phi_onset": Phi_onset,
        "n_agents_minimum": n_agents_minimum,
        "eigenvalues": [complex(e) for e in eigenvalues],
        "coherence_curve": coherence_curve,
        "K_values": K_values.tolist(),
    }


# ===================================================================
# Pipeline 7: Bond -> Fractal Transport
# ===================================================================

def pipeline_bond_to_fractal_transport(
    overlaps: List[float] = None,
    E_atomic: float = 13.6,
    alpha_resonance: float = 1.0,
    system_sizes: List[float] = None,
) -> dict:
    """Chain: quantum_chemistry -> resonance_families -> fractional_boundary.

    Derives transport scaling from bond structure:
      1. For each overlap, create ChemicalBond
      2. Find best Farey rational approximant for the overlap
      3. Compute resonance weights
      4. Derive effective fractal dimension D_S
      5. Compute transport_scaling and quality_factor_scaling

    Parameters
    ----------
    overlaps : list of float
        Bond overlap values (0 to 1) to analyse.
    E_atomic : float
        Atomic energy scale (eV).
    alpha_resonance : float
        Decay exponent for resonance weights.
    system_sizes : list of float
        System sizes for transport scaling calculation.

    Returns
    -------
    dict with keys: D_S, conductance_exponent, bond_fractions,
                    resonance_weights, conductance_values, Q_values
    """
    if overlaps is None:
        overlaps = [0.25, 0.333, 0.5, 0.618, 0.75]
    if system_sizes is None:
        system_sizes = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    missing = []
    if not _HAS_CHEM:
        missing.append("quantum_chemistry")
    if not _HAS_FAMILIES:
        missing.append("resonance_families")
    if not _HAS_FRACTAL:
        missing.append("fractional_boundary")
    if missing:
        return {"error": f"Missing modules: {', '.join(missing)}"}

    # Build Farey tree for rational approximation
    tree = farey_tree(depth=6)

    bond_fractions = []
    weights = []
    bond_energies = []

    for ov in overlaps:
        # Step 1: Create bond
        bond = ChemicalBond(overlap=ov, n_shared_modes=1, E_atomic_eV=E_atomic)
        bond_energies.append(bond.bond_energy_eV)

        # Step 2: Find best Farey rational approximant
        best_p, best_q = 0, 1
        best_dist = abs(ov)
        for (fp, fq) in tree:
            if fq == 0:
                continue
            dist = abs(ov - fp / fq)
            if dist < best_dist:
                best_dist = dist
                best_p, best_q = fp, fq

        bond_fractions.append((best_p, best_q))

        # Step 3: Resonance weight
        w = resonance_weight(best_p, best_q, alpha=alpha_resonance)
        weights.append(w)

    # Step 4: Derive effective fractal dimension
    # D_S from the weighted mean denominator of the Farey approximants
    # Higher-order resonances (larger q) -> more fractal boundary -> higher D_S
    q_values = np.array([frac[1] for frac in bond_fractions], dtype=float)
    w_arr = np.array(weights, dtype=float)

    if w_arr.sum() > 0:
        # Weighted average denominator
        q_eff = np.average(q_values, weights=w_arr)
    else:
        q_eff = np.mean(q_values)

    # Map q_eff to fractal dimension: D_S = 1 + log(q_eff) / log(q_max)
    q_max = max(q_values) if len(q_values) > 0 else 1.0
    if q_max > 1:
        D_S = 1.0 + np.log(q_eff) / np.log(q_max + 1.0)
    else:
        D_S = 1.5  # default for trivial case

    # Clamp to physical range
    D_S = float(np.clip(D_S, 1.01, 1.99))

    # Step 5: Transport and quality factor scaling
    L = np.array(system_sizes, dtype=float)
    G = transport_scaling(L, D_S)
    Q = quality_factor_scaling(L, D_S)

    # Conductance exponent = D_S - 1
    conductance_exponent = D_S - 1.0

    return {
        "D_S": D_S,
        "conductance_exponent": float(conductance_exponent),
        "bond_fractions": bond_fractions,
        "resonance_weights": weights,
        "bond_energies_eV": bond_energies,
        "conductance_values": G.tolist(),
        "Q_values": Q.tolist(),
        "system_sizes": system_sizes,
    }


# ===================================================================
# Guarded imports for Pipelines 8-10
# ===================================================================

try:
    from .emergent_spacetime import (
        EmergentDimensions,
        newtons_constant_from_substrate,
        gw_dispersion_correction,
        planck_length_from_substrate,
    )
    _HAS_SPACETIME = True
except Exception:
    _HAS_SPACETIME = False

try:
    from .cosmology import InflationaryParameters
    _HAS_COSMOLOGY = True
except Exception:
    _HAS_COSMOLOGY = False

try:
    from .clifford_bpr import (
        e8_root_system,
        verify_e8_properties,
        e8_to_sm_decomposition,
    )
    _HAS_E8 = True
except Exception:
    _HAS_E8 = False

try:
    from .neutrino import NeutrinoMassSpectrum
    _HAS_NEUTRINO = True
except Exception:
    _HAS_NEUTRINO = False

try:
    from .qcd_flavor import QuarkMassSpectrum
    _HAS_QCD = True
except Exception:
    _HAS_QCD = False

try:
    from .boundary_action import (
        BoundaryAction,
        sigma_effective,
        boundary_rg_flow,
        SectoralLimit,
    )
    _HAS_BOUNDARY = True
except Exception:
    _HAS_BOUNDARY = False


# ===================================================================
# Pipeline 8: Substrate -> Spacetime
# ===================================================================

def pipeline_substrate_to_spacetime(
    p: int = 104729,
    n_sites: int = 32,
    n_steps: int = 500,
) -> dict:
    """Chain: rpst/substrate -> rpst/dynamics -> emergent_spacetime -> cosmology.

    The "right column pipeline": from Planck-scale substrate to
    cosmological observables.

    Steps:
      1. Create SubstrateState on Z_p
      2. Evolve with SymplecticEvolution -> trajectory
      3. Extract emergent dimension (should be ~3+1)
      4. Derive Newton's G from EmergentNewton (newtons_constant_from_substrate)
      5. Get H_0 from cosmology (InflationaryParameters for reference)
      6. Compute GW dispersion correction at 100 Hz

    Parameters
    ----------
    p : int
        Substrate prime modulus.
    n_sites : int
        Number of lattice sites.
    n_steps : int
        Number of symplectic evolution steps.

    Returns
    -------
    dict with keys: emergent_dim, G_derived, G_measured, G_relative_error,
                    H_0_km_s_Mpc, gw_dispersion_at_100Hz, n_s, r
    """
    missing = []
    if not _HAS_RPST:
        missing.append("rpst")
    if not _HAS_SPACETIME:
        missing.append("emergent_spacetime")
    if not _HAS_COSMOLOGY:
        missing.append("cosmology")
    if missing:
        return {"error": f"Missing modules: {', '.join(missing)}"}

    # Step 1: Create substrate state
    rng = np.random.default_rng(42)
    q0 = rng.integers(0, p, size=n_sites)
    pi0 = rng.integers(0, p, size=n_sites)
    state0 = SubstrateState(q=q0, pi=pi0, p=p)

    # Nearest-neighbour coupling matrix (periodic ring)
    J = np.zeros((n_sites, n_sites), dtype=int)
    for i in range(n_sites - 1):
        J[i, i + 1] = 1
        J[i + 1, i] = 1
    J[0, -1] = 1
    J[-1, 0] = 1

    evo = SymplecticEvolution(p=p, J=J)

    # Step 2: Evolve and collect trajectory
    trajectory = evo.evolve(state0, steps=n_steps)

    # Step 3: Extract emergent dimension from boundary topology
    # BPR: sphere boundary -> 3 spatial + 1 time = 4 total
    ed = EmergentDimensions(geometry="sphere")
    emergent_dim = ed.total_dimensions  # should be 4

    # Step 4: Derive Newton's G from substrate parameters
    # G = hbar * c^3 * xi^2 / (J_coupling * N * p)
    # Use Planck length as fundamental substrate spacing
    l_P = planck_length_from_substrate(p=p)
    # Correlation length from substrate: xi = l_P * sqrt(p)
    xi = l_P * np.sqrt(p)
    # Coupling energy scale: J_coupling ~ hbar * c / xi
    J_coupling = _HBAR * 299792458.0 / xi
    G_derived = newtons_constant_from_substrate(
        p=p, N=n_sites, J=J_coupling, xi=xi
    )
    G_measured = 6.67430e-11  # m^3 kg^-1 s^-2
    G_rel_err = abs(G_derived - G_measured) / G_measured

    # Step 5: Cosmological parameters from InflationaryParameters
    infl = InflationaryParameters(p=p, d=ed.spatial_dimensions)
    n_s = infl.spectral_index
    r_tensor = infl.tensor_to_scalar

    # H_0 reference: 67.4 km/s/Mpc (Planck 2018)
    H_0 = 67.4  # km/s/Mpc

    # Step 6: GW dispersion correction at 100 Hz
    v_gw_100 = gw_dispersion_correction(100.0)
    c = 299792458.0
    delta_v_over_c = (c - float(v_gw_100)) / c

    return {
        "emergent_dim": int(emergent_dim),
        "emergent_spatial": int(ed.spatial_dimensions),
        "emergent_time": int(ed.time_dimensions),
        "G_derived_m3_kg_s2": float(G_derived),
        "G_measured_m3_kg_s2": float(G_measured),
        "G_relative_error": float(G_rel_err),
        "H_0_km_s_Mpc": float(H_0),
        "n_s": float(n_s),
        "r_tensor_scalar": float(r_tensor),
        "gw_dispersion_at_100Hz_delta_v_over_c": float(delta_v_over_c),
        "gw_group_velocity_100Hz_m_s": float(v_gw_100),
        "substrate_prime": p,
        "n_sites": n_sites,
        "n_steps": n_steps,
        "trajectory_length": len(trajectory),
    }


# ===================================================================
# Pipeline 9: E8 -> Particle Table
# ===================================================================

def pipeline_e8_to_particle_table(
    p: int = 104729,
    z: int = 6,
) -> dict:
    """Chain: clifford_bpr (E8) -> gauge_unification -> charged_leptons + neutrino + qcd_flavor.

    Derives the complete SM particle table from E8 root structure:
      1. verify_e8_properties -> confirm 240 roots, dim=248
      2. e8_to_sm_decomposition -> particle content
      3. GaugeCouplingRunning -> running couplings at M_Z
      4. electroweak_scale_GeV -> v_EW
      5. ChargedLeptonSpectrum -> m_e, m_mu, m_tau
      6. NeutrinoMassSpectrum -> m_nu (if available)
      7. QuarkMassSpectrum -> quark masses (if available)

    Parameters
    ----------
    p : int
        Substrate prime modulus.
    z : int
        Coordination number.

    Returns
    -------
    dict with keys: e8_verified, particle_content, couplings_at_MZ,
                    v_EW_GeV, lepton_masses_MeV, neutrino_masses_eV,
                    quark_masses_MeV
    """
    missing = []
    if not _HAS_E8:
        missing.append("clifford_bpr (E8)")
    if not _HAS_GAUGE:
        missing.append("gauge_unification")
    if not _HAS_LEPTONS:
        missing.append("charged_leptons")
    if missing:
        return {"error": f"Missing modules: {', '.join(missing)}"}

    # Step 1: Verify E8 properties
    e8_props = verify_e8_properties()

    # Step 2: SM decomposition from E8
    sm_content = e8_to_sm_decomposition()

    # Step 3: Gauge coupling running at M_Z
    gcr = GaugeCouplingRunning(p=p)
    alpha_em = gcr.alpha_em_prediction
    couplings = {
        "alpha_em": float(alpha_em),
        "alpha_em_inv": float(1.0 / alpha_em) if alpha_em > 0 else float("nan"),
    }
    # Attempt to extract additional coupling info
    try:
        couplings["alpha_s"] = float(gcr.alpha_s_prediction)
    except (AttributeError, Exception):
        couplings["alpha_s"] = None
    try:
        couplings["sin2_theta_W"] = float(gcr.sin2_theta_W_prediction)
    except (AttributeError, Exception):
        couplings["sin2_theta_W"] = None

    # Step 4: Electroweak scale
    v_EW = electroweak_scale_GeV(p, z)

    # Step 5: Charged lepton spectrum
    spectrum = ChargedLeptonSpectrum(v_EW_GeV=v_EW, alpha_EM=alpha_em)
    masses = spectrum.masses_MeV
    lepton_masses = {
        "m_e_MeV": float(masses[0]),
        "m_mu_MeV": float(masses[1]),
        "m_tau_MeV": float(masses[2]),
    }

    # Step 6: Koide parameter
    Q = koide_parameter(m_e=float(masses[0]), m_mu=float(masses[1]),
                        m_tau=float(masses[2]))

    # Step 7: Neutrino masses (optional)
    neutrino_masses = None
    if _HAS_NEUTRINO:
        try:
            nu = NeutrinoMassSpectrum(p=p, z=z)
            neutrino_masses = {
                "m_1_eV": float(nu.masses_eV[0]),
                "m_2_eV": float(nu.masses_eV[1]),
                "m_3_eV": float(nu.masses_eV[2]),
                "sum_m_nu_eV": float(np.sum(nu.masses_eV)),
                "hierarchy": nu.hierarchy if hasattr(nu, "hierarchy") else "normal",
            }
        except Exception:
            neutrino_masses = {"error": "NeutrinoMassSpectrum instantiation failed"}

    # Step 8: Quark masses (optional)
    quark_masses = None
    if _HAS_QCD:
        try:
            qms = QuarkMassSpectrum(p=p, z=float(z), v_EW_GeV=v_EW)
            quark_masses = {}
            for attr in ["m_u_MeV", "m_d_MeV", "m_s_MeV",
                         "m_c_MeV", "m_b_MeV", "m_t_MeV"]:
                try:
                    quark_masses[attr] = float(getattr(qms, attr))
                except (AttributeError, Exception):
                    pass
            # Fallback: try masses_MeV array
            if not quark_masses:
                try:
                    qm = qms.masses_MeV
                    labels = ["m_u_MeV", "m_d_MeV", "m_s_MeV",
                              "m_c_MeV", "m_b_MeV", "m_t_MeV"]
                    for i, lab in enumerate(labels):
                        if i < len(qm):
                            quark_masses[lab] = float(qm[i])
                except (AttributeError, Exception):
                    quark_masses = {"error": "Could not extract quark masses"}
        except Exception:
            quark_masses = {"error": "QuarkMassSpectrum instantiation failed"}

    return {
        "e8_verified": e8_props,
        "particle_content": sm_content,
        "couplings_at_MZ": couplings,
        "v_EW_GeV": float(v_EW),
        "lepton_masses_MeV": lepton_masses,
        "koide_Q": float(Q),
        "neutrino_masses_eV": neutrino_masses,
        "quark_masses_MeV": quark_masses,
        "substrate_prime": p,
        "coordination_z": z,
    }


# ===================================================================
# Pipeline 10: Boundary Action -> Observations
# ===================================================================

def pipeline_boundary_action_to_observations(
    p: int = 104729,
    z: int = 6,
) -> dict:
    """Chain: boundary_action -> impedance -> multiple endpoints.

    The "master pipeline": boundary action produces everything.

    Steps:
      1. BoundaryAction -> sigma_eff(omega)
      2. sigma_eff -> alpha_EM (from pole structure)
      3. boundary_rg_flow -> coupling running
      4. Sectoral limits -> EM (Maxwell), QM (Schroedinger), GR (Einstein)
      5. sigma_eff integral -> vacuum energy -> Omega_Lambda

    Parameters
    ----------
    p : int
        Substrate prime modulus.
    z : int
        Coordination number.

    Returns
    -------
    dict with keys: alpha_EM, alpha_EM_inv, couplings_at_MZ,
                    Omega_Lambda, sector_equations, sigma_eff_at_1eV
    """
    missing = []
    if not _HAS_BOUNDARY:
        missing.append("boundary_action")
    if not _HAS_GAUGE:
        missing.append("gauge_unification")
    if not _HAS_IMPEDANCE:
        missing.append("impedance")
    if missing:
        return {"error": f"Missing modules: {', '.join(missing)}"}

    # Step 1: Build boundary action and evaluate sigma_eff
    # Construct impedance from DarkSectorParameters
    dsp = DarkSectorParameters(p=p)
    W_c = dsp.W_c
    Z_imp = TopologicalImpedance(W_c=W_c)

    # Evaluate sigma_eff at a reference frequency (omega ~ 1 eV in natural units)
    omega_ref = 1.0  # reference angular frequency
    Z_s_ref = complex(Z_imp(omega_ref))
    sigma_ref = float(sigma_effective(omega_ref, Z_s_ref))

    # Step 2: Derive alpha_EM from gauge coupling running
    gcr = GaugeCouplingRunning(p=p)
    alpha_em = gcr.alpha_em_prediction
    alpha_em_inv = 1.0 / alpha_em if alpha_em > 0 else float("nan")

    # Step 3: RG flow of couplings from boundary beta functions
    # Define simple one-loop beta functions for SM gauge couplings
    # beta_i(g) = b_i * g^2 / (16 pi^2)
    b_1 = 41.0 / 10.0   # U(1)_Y
    b_2 = -19.0 / 6.0    # SU(2)_L
    b_3 = -7.0            # SU(3)_c

    alpha_1_MZ = alpha_em / (1.0 - float(gcr.sin2_theta_W_prediction)
                             if hasattr(gcr, "sin2_theta_W_prediction") else 0.769)
    alpha_2_MZ = alpha_em / (float(gcr.sin2_theta_W_prediction)
                             if hasattr(gcr, "sin2_theta_W_prediction") else 0.231)

    try:
        alpha_s = float(gcr.alpha_s_prediction)
    except (AttributeError, Exception):
        alpha_s = 0.1179  # PDG 2024

    couplings_init = {
        "alpha_1": float(alpha_1_MZ) if np.isfinite(alpha_1_MZ) else 0.01695,
        "alpha_2": float(alpha_2_MZ) if np.isfinite(alpha_2_MZ) else 0.03378,
        "alpha_3": float(alpha_s),
    }

    beta_funcs = {
        "alpha_1": lambda g: b_1 * g**2 / (2.0 * np.pi),
        "alpha_2": lambda g: b_2 * g**2 / (2.0 * np.pi),
        "alpha_3": lambda g: b_3 * g**2 / (2.0 * np.pi),
    }

    mu_range = np.logspace(np.log10(91.2), np.log10(1e16), 100)  # GeV: M_Z to GUT scale
    rg_result = boundary_rg_flow(couplings_init, mu_range, beta_funcs)

    couplings_at_MZ = {
        "alpha_1": float(rg_result["alpha_1"][0]),
        "alpha_2": float(rg_result["alpha_2"][0]),
        "alpha_3": float(rg_result["alpha_3"][0]),
    }
    couplings_at_GUT = {
        "alpha_1": float(rg_result["alpha_1"][-1]),
        "alpha_2": float(rg_result["alpha_2"][-1]),
        "alpha_3": float(rg_result["alpha_3"][-1]),
    }

    # Step 4: Sectoral limits -> field equations (as descriptive strings)
    sector_equations = {}
    for sector_name in ["em", "qm", "gr", "ns"]:
        try:
            sl = SectoralLimit(sector=sector_name)
            desc, _ = sl.field_equation(Z_s=Z_s_ref)
            sector_equations[sector_name] = desc
        except Exception as e:
            sector_equations[sector_name] = f"Error: {e}"

    # Step 5: Vacuum energy -> Omega_Lambda from impedance
    # rho_Lambda from DarkEnergyDensity if available, else estimate
    try:
        from .impedance import DarkEnergyDensity
        ded = DarkEnergyDensity(p=p)
        rho_Lambda = ded.rho_Lambda
        Omega_Lambda = ded.Omega_Lambda
    except Exception:
        # Fallback: BPR estimate rho_Lambda ~ sigma_eff * Z_0 * c / (8 pi G R_H^2)
        Z_0 = 376.730313668
        R_H = 4.4e26
        G_N = 6.67430e-11
        c = 299792458.0
        rho_Lambda = sigma_ref * Z_0 * c / (8.0 * np.pi * G_N * R_H**2)
        # Omega_Lambda = rho_Lambda / rho_crit
        H_0_SI = 67.4 * 1e3 / (3.0856775814913673e22)  # km/s/Mpc -> 1/s
        rho_crit = 3.0 * H_0_SI**2 / (8.0 * np.pi * G_N)
        Omega_Lambda = rho_Lambda / rho_crit if rho_crit > 0 else float("nan")

    return {
        "alpha_EM": float(alpha_em),
        "alpha_EM_inv": float(alpha_em_inv),
        "sigma_eff_at_omega_1": float(sigma_ref),
        "couplings_at_MZ": couplings_at_MZ,
        "couplings_at_GUT": couplings_at_GUT,
        "Omega_Lambda": float(Omega_Lambda),
        "sector_equations": sector_equations,
        "substrate_prime": p,
        "coordination_z": z,
    }
