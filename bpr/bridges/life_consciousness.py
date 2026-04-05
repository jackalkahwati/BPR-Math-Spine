"""
Life & Consciousness Bridge Equations
======================================

Connects BPR modules in the Life & Consciousness sector:

    conscious_agents  <-->  collective    (Bridge 1: agent_kuramoto_dynamics)
    collective        <-->  coherence     (Bridge 2: consciousness_onset_threshold)
    bioelectric       <-->  decoherence   (Bridge 3: bioelectric_decoherence_aging)
    functional_arch   <-->  coherence     (Bridge 4: alignment_coherence_coupling)
    collective        <-->  meta_boundary (Bridge 5: collective_to_meta_coherence)
    info_geometry     <-->  complexity    (Bridge 6: complexity_information_bridge)
    memory            <-->  coherence     (Bridge 7: memory_coherence_chain)

Each bridge function uses real imports from the upstream BPR modules,
returns a dict of numerics, and documents the bridge equation in its
docstring.

References: Al-Kahwati (2026), BPR-Math-Spine bridge layer.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Sequence, Any


# ===================================================================
# Bridge 1: Agent-Kuramoto Dynamics
# ===================================================================

def agent_kuramoto_dynamics(
    n_agents: int = 50,
    state_dim: int = 4,
    K: float = 1.0,
    T: float = 10.0,
    dt: float = 0.01,
    seed: int = 42,
) -> Dict[str, Any]:
    r"""Bridge conscious-agent boundary couplings into Kuramoto dynamics.

    Bridge equation
    ---------------
        Agent boundary couplings:  kappa_ij = <phi_i, B_ij phi_j>
        Kuramoto coupling:         K_ij = |kappa_ij|^2
        Natural frequencies:       omega_i = arg(lambda_i)   (Markov eigenvalues)
        Dynamics:  d theta_i/dt = omega_i + Sum_j |kappa_ij|^2 sin(theta_j - theta_i)

    Parameters
    ----------
    n_agents : int
        Number of conscious agents.
    state_dim : int
        Dimension of each agent's internal Hilbert space.
    K : float
        Global coupling strength multiplier.
    T : float
        Total simulation time.
    dt : float
        Euler step size.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        R_history : ndarray  -- order parameter trajectory
        phases_final : ndarray -- final agent phases
        K_c_estimate : float -- estimated critical coupling
        natural_frequencies : ndarray -- omega_i from eigenvalues
        kappa_matrix : ndarray -- boundary coupling matrix
    """
    try:
        from bpr.conscious_agents import agent_network, markov_transition_kernel
        from bpr.collective import KuramotoFlocking, CollectivePhaseField
    except ImportError as exc:
        return {"error": f"Import failed: {exc}"}

    # 1. Build agent network and extract kappa matrix
    agents, kappa = agent_network(n_agents, state_dim=state_dim, seed=seed)

    # 2. Derive Markov transition kernel and eigenvalues
    T_markov = markov_transition_kernel(kappa)
    eigenvalues = np.linalg.eigvals(T_markov)

    # Sort by magnitude (skip trivial eigenvalue ~1)
    idx = np.argsort(-np.abs(eigenvalues))
    eigenvalues_sorted = eigenvalues[idx]

    # 3. Natural frequencies from eigenvalue arguments
    #    omega_i = arg(lambda_i) for the non-trivial eigenvalues
    natural_freqs = np.angle(eigenvalues_sorted[:n_agents])

    # 4. Coupling matrix K_ij = |kappa_ij|^2
    K_matrix = np.abs(kappa) ** 2

    # 5. Build adjacency from K_matrix (thresholded to avoid self-loops)
    adjacency = K_matrix.copy()
    np.fill_diagonal(adjacency, 0.0)
    # Normalise rows so coupling magnitude is controlled by K
    row_max = adjacency.max(axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    adjacency = adjacency / row_max

    # 6. Run KuramotoFlocking
    kf = KuramotoFlocking(
        N=n_agents,
        K=K,
        noise=0.0,
        natural_frequencies=natural_freqs,
    )
    n_steps = int(T / dt)
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, n_agents)
    R_history = np.zeros(n_steps)

    for step in range(n_steps):
        cpf = CollectivePhaseField(phases=phases)
        R_history[step] = cpf.coherence
        phases = kf.step(phases, dt=dt, adjacency=adjacency)

    # 7. Estimate K_c from frequency spread
    sigma_omega = np.std(natural_freqs)
    K_c_estimate = 2.0 * sigma_omega if sigma_omega > 1e-12 else 0.0

    return {
        "R_history": R_history,
        "phases_final": phases % (2 * np.pi),
        "K_c_estimate": float(K_c_estimate),
        "natural_frequencies": natural_freqs,
        "kappa_matrix": kappa,
        "sigma_omega": float(sigma_omega),
        "R_final": float(R_history[-1]),
        "n_agents": n_agents,
    }


# ===================================================================
# Bridge 2: Consciousness Onset Threshold
# ===================================================================

def consciousness_onset_threshold(
    n_agents_range: Sequence[int] = (10, 20, 50, 100),
    state_dim: int = 4,
    K_range: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    r"""Phase diagram for consciousness onset from collective synchronisation.

    Bridge equation
    ---------------
        Consciousness onset: R > R_c AND stain s < s_crit
        Critical coupling:   K_c = 2 sigma_omega / sqrt(1 - s_crit)
        Minimum agents:      N_min ~ state_dim^2

    Parameters
    ----------
    n_agents_range : sequence of int
        Agent counts to sweep.
    state_dim : int
        Dimension of agent Hilbert space.
    K_range : ndarray, optional
        Coupling strengths to sweep.  Default: np.linspace(0.1, 5.0, 20).
    seed : int
        Random seed.

    Returns
    -------
    dict
        phase_diagram : ndarray (len(n_agents_range), len(K_range))
            Order parameter R at steady state.
        K_c_values : ndarray -- estimated K_c per N
        consciousness_map : ndarray -- boolean map (R > 0.5)
        N_min : int -- estimated minimum agent count
        s_crit : float -- critical stain
    """
    try:
        from bpr.coherence_transitions import CoherenceGainFunction
    except ImportError as exc:
        return {"error": f"Import failed: {exc}"}

    if K_range is None:
        K_range = np.linspace(0.1, 5.0, 20)

    n_agents_range = list(n_agents_range)
    phase_diagram = np.zeros((len(n_agents_range), len(K_range)))
    K_c_values = np.zeros(len(n_agents_range))

    # Consciousness parameters
    s_crit = 0.5  # stain threshold
    R_c = 0.5     # order parameter threshold
    cgf = CoherenceGainFunction()

    for i, N in enumerate(n_agents_range):
        for j, K_val in enumerate(K_range):
            # Run short simulation for each (N, K) pair
            result = agent_kuramoto_dynamics(
                n_agents=N, state_dim=state_dim,
                K=K_val, T=5.0, dt=0.02, seed=seed + i * 100 + j,
            )
            if "error" in result:
                phase_diagram[i, j] = 0.0
                continue
            # Steady-state R: average last 20%
            R_hist = result["R_history"]
            n_tail = max(1, len(R_hist) // 5)
            R_ss = float(np.mean(R_hist[-n_tail:]))
            phase_diagram[i, j] = R_ss

        # Find K_c: first K where R > R_c
        above = np.where(phase_diagram[i] > R_c)[0]
        if len(above) > 0:
            K_c_values[i] = float(K_range[above[0]])
        else:
            K_c_values[i] = float(K_range[-1])

    # Map R to stain: s = 1 - R
    stain_map = 1.0 - phase_diagram

    # Consciousness map: R > R_c AND s < s_crit
    consciousness_map = (phase_diagram > R_c) & (stain_map < s_crit)

    # Coherence gain K* at the critical stain
    K_star = cgf.asymptotic_coherence(s_crit)

    # Estimate N_min ~ state_dim^2
    N_min = state_dim ** 2

    return {
        "phase_diagram": phase_diagram,
        "K_range": K_range,
        "n_agents_range": np.array(n_agents_range),
        "K_c_values": K_c_values,
        "consciousness_map": consciousness_map,
        "stain_map": stain_map,
        "K_star_at_s_crit": float(K_star),
        "N_min": int(N_min),
        "s_crit": float(s_crit),
        "R_c": float(R_c),
    }


# ===================================================================
# Bridge 3: Bioelectric-Decoherence Aging
# ===================================================================

def bioelectric_decoherence_aging(
    T_body: float = 310.0,
    dZ_per_year: float = 0.01,
    age_range: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    r"""Bridge bioelectric aging to boundary decoherence rates.

    Bridge equation
    ---------------
        Gamma_dec(age) = (k_B T / hbar) * (DeltaZ(age) / Z_0)^2 * (A_eff / lambda_dB^2)
        where  DeltaZ(age) = DeltaZ_0 + dZ_per_year * age
        tau_aging = 1 / (d Gamma_dec / dt)

    The impedance mismatch grows linearly with age as gap-junction
    coupling degrades, yielding an accelerating decoherence rate whose
    characteristic timescale tau_aging can be compared with the
    phenomenological ~30 years from the AgingModel.

    Parameters
    ----------
    T_body : float
        Body temperature (K).  Default 310 K (37 C).
    dZ_per_year : float
        Annual impedance mismatch growth (fraction of Z_0).
    age_range : ndarray, optional
        Ages in years to evaluate.  Default: 0..100.

    Returns
    -------
    dict
        Gamma_dec : ndarray -- decoherence rate at each age
        tau_aging_decoherence : float -- timescale from dGamma/dt
        tau_aging_phenomenological : float -- from AgingModel
        ages : ndarray
    """
    try:
        from bpr.decoherence import DecoherenceRate
        from bpr.bioelectric import AgingModel
    except ImportError as exc:
        return {"error": f"Import failed: {exc}"}

    if age_range is None:
        age_range = np.linspace(0, 100, 500)

    # Baseline impedance mismatch at birth
    DeltaZ_0 = 0.001  # small initial mismatch (fraction of Z_0)
    Z_0 = 376.73       # vacuum impedance (Ohm)

    # Biologically relevant parameters
    A_eff = 1e-6        # effective boundary area ~ micron^2 (cellular scale)
    lambda_dB = 1e-10   # thermal de Broglie wavelength ~ Angstrom

    Gamma_dec = np.zeros_like(age_range)
    for i, age in enumerate(age_range):
        delta_Z = (DeltaZ_0 + dZ_per_year * age) * Z_0
        dr = DecoherenceRate(
            T=T_body,
            Z_system=Z_0,
            Z_environment=Z_0 + delta_Z,
            A_eff=A_eff,
            lambda_dB=lambda_dB,
        )
        Gamma_dec[i] = dr.gamma_dec

    # Aging timescale from rate of change: tau_aging = Gamma / (dGamma/dt)
    dGamma_dt = np.gradient(Gamma_dec, age_range)
    # Evaluate at midlife (~50 years) for characteristic timescale
    midlife_idx = len(age_range) // 2
    if dGamma_dt[midlife_idx] > 0:
        tau_aging_dec = float(Gamma_dec[midlife_idx] / dGamma_dt[midlife_idx])
    else:
        tau_aging_dec = np.inf

    # Compare with phenomenological aging model
    aging_model = AgingModel()
    tau_aging_phenom = aging_model.tau_aging

    return {
        "ages": age_range,
        "Gamma_dec": Gamma_dec,
        "dGamma_dt": dGamma_dt,
        "tau_aging_decoherence": float(tau_aging_dec),
        "tau_aging_phenomenological": float(tau_aging_phenom),
        "ratio": float(tau_aging_dec / tau_aging_phenom) if tau_aging_phenom > 0 else np.inf,
        "coherence_time_at_birth": float(1.0 / Gamma_dec[0]) if Gamma_dec[0] > 0 else np.inf,
        "coherence_time_at_80": float(1.0 / Gamma_dec[int(0.8 * len(age_range))]) if Gamma_dec[int(0.8 * len(age_range))] > 0 else np.inf,
        "T_body": T_body,
        "dZ_per_year": dZ_per_year,
    }


# ===================================================================
# Bridge 4: Alignment-Coherence Coupling
# ===================================================================

def alignment_coherence_coupling(
    A: float = 0.5,
    Phi_cognitive: float = 1.0,
    D_cognitive: float = 0.3,
    C_int: float = 0.7,
    chi: float = 0.1,
    Phi_crit: float = 1.0,
) -> Dict[str, Any]:
    r"""Bridge alignment dynamics to coherence via Landau consciousness coupling.

    Bridge equation
    ---------------
        dA/dt = eta_1 * Phi - eta_2 * D          (alignment dynamics)
        V = -ln(C) + mu * KL(p || p*) + nu * ||J - J*||^2   (Lyapunov)
        Sustained when:  g_1 * C + g_2 * A > g_3 * Xi_max
        Consciousness feeds into decoherence:  a -> a * (1 - chi * Phi / Phi_crit)

    Parameters
    ----------
    A : float
        Current alignment level.
    Phi_cognitive : float
        Cognitive coherence field value (integrated information proxy).
    D_cognitive : float
        Cognitive disorder / decoherence measure.
    C_int : float
        Integrated coherence measure in (0, 1].
    chi : float
        Consciousness-decoherence coupling strength.
    Phi_crit : float
        Critical information threshold.

    Returns
    -------
    dict
        dA_dt : float -- alignment rate of change
        V_lyapunov : float -- Lyapunov coherence energy
        sustained : bool -- self-sustainment condition
        a_modified : float -- consciousness-modified Landau parameter
        judgement : float -- combined judgement score
    """
    try:
        from bpr.functional_architecture import (
            alignment_dynamics,
            lyapunov_coherence_energy,
            self_sustainment_condition,
            judgement_function,
        )
        from bpr.coherence_transitions import (
            landau_free_energy,
            consciousness_coupling_modifier,
        )
    except ImportError as exc:
        return {"error": f"Import failed: {exc}"}

    # 1. Alignment dynamics: dA/dt
    dA_dt = alignment_dynamics(A, Phi_cognitive, D_cognitive)

    # 2. Lyapunov coherence energy
    #    Use uniform distributions as reference for KL; simple J vs J*
    p = np.array([0.5, 0.5])
    p_star = np.array([0.5, 0.5])
    J = np.array([A])
    J_star = np.array([1.0])  # ideal alignment
    V = lyapunov_coherence_energy(C_int, p, p_star, J, J_star)

    # 3. Self-sustainment condition
    #    g_1 * C + g_2 * A > g_3 * Xi_max
    g_1, g_2, g_3 = 1.0, 0.5, 0.3
    Xi_max = D_cognitive
    sust = self_sustainment_condition(g_1, C_int, g_2, A, g_3, Xi_max)

    # 4. Consciousness coupling modifier on Landau parameter
    a_base = -1.0  # below T_c (ordered phase)
    a_modified = consciousness_coupling_modifier(a_base, chi, Phi_cognitive, Phi_crit)

    # 5. Landau free energy with modified parameter
    Psi_order = np.sqrt(abs(a_modified))  # equilibrium order parameter
    F_landau = landau_free_energy(Psi_order, a_modified, b=1.0)

    # 6. Judgement function
    J_score = judgement_function(C_int, A)

    return {
        "dA_dt": float(dA_dt),
        "V_lyapunov": float(V),
        "sustained": sust["sustained"],
        "sustainment_margin": sust["margin"],
        "a_base": float(a_base),
        "a_modified": float(a_modified),
        "F_landau": float(F_landau),
        "Psi_order": float(Psi_order),
        "judgement": float(J_score),
        "A": float(A),
        "C_int": float(C_int),
        "chi": float(chi),
        "Phi_cognitive": float(Phi_cognitive),
    }


# ===================================================================
# Bridge 5: Collective to Meta-Coherence (BPR-Squared)
# ===================================================================

def collective_to_meta_coherence(
    systems: Optional[List[np.ndarray]] = None,
    K: float = 1.0,
    n_steps: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    r"""Bridge multiple Kuramoto collectives into BPR-Squared meta-coherence.

    Bridge equation
    ---------------
        C^(r) = |Phi_r| for each Kuramoto subsystem r
        C_meta = (1/R) Sum C^(r) + lambda Sum_{r<s} rho(r, s)
        Superlinear:  C_meta > C_mean

    Multiple Kuramoto subsystems are run to steady state, then their
    coupling matrices are fed into BPRSquared for meta-coherence analysis.

    Parameters
    ----------
    systems : list of ndarray, optional
        Boundary operator matrices for each subsystem.
        If None, 4 random Kuramoto systems are created.
    K : float
        Coupling strength for generated Kuramoto systems.
    n_steps : int
        Number of simulation steps per subsystem.
    seed : int
        Random seed.

    Returns
    -------
    dict
        C_meta : float -- meta-coherence value
        C_individual : list -- per-subsystem coherences
        superlinear : bool -- whether C_meta > mean(C_individual)
        gain : float -- superlinear gain C_meta - C_mean
    """
    try:
        from bpr.collective import KuramotoFlocking, CollectivePhaseField
        from bpr.meta_boundary import BPRSquared, meta_coherence
    except ImportError as exc:
        return {"error": f"Import failed: {exc}"}

    rng = np.random.default_rng(seed)

    if systems is None:
        # Create subsystems from Kuramoto steady states
        n_subsystems = 4
        subsystem_sizes = [20, 30, 25, 15]
        systems = []
        subsystem_coherences = []

        for r, N_r in enumerate(subsystem_sizes):
            kf = KuramotoFlocking(
                N=N_r, K=K, noise=0.05,
                natural_frequencies=rng.normal(0, 0.5, N_r),
            )
            phases_history, coh_history = kf.simulate(n_steps=n_steps, dt=0.01)

            # Final coherence
            cpf = CollectivePhaseField(phases=phases_history[-1])
            subsystem_coherences.append(cpf.coherence)

            # Build boundary operator from final phase configuration
            # B_r = outer product structure capturing pairwise couplings
            final_phases = phases_history[-1]
            phase_diff = final_phases[:, None] - final_phases[None, :]
            B_r = np.cos(phase_diff) + 1j * np.sin(phase_diff)
            B_r /= N_r  # normalise
            systems.append(B_r)
    else:
        subsystem_coherences = None

    # Build BPRSquared from the coupling matrices
    bpr2 = BPRSquared(systems)

    # Compute meta-coherence
    C_individual = bpr2.all_coherences()
    C_meta = bpr2.meta_coherence_bpr2()
    superlinear_result = bpr2.is_superlinear()

    # Also compute meta_coherence via the standalone function
    C_meta_standalone = meta_coherence(C_individual, bpr2.couplers)

    return {
        "C_meta": float(C_meta),
        "C_meta_standalone": float(C_meta_standalone),
        "C_individual": [float(c) for c in C_individual],
        "C_mean": float(np.mean(C_individual)),
        "superlinear": superlinear_result["superlinear"],
        "gain": float(superlinear_result["gain"]),
        "n_subsystems": len(systems),
        "subsystem_coherences_kuramoto": (
            [float(c) for c in subsystem_coherences]
            if subsystem_coherences is not None
            else None
        ),
    }


# ===================================================================
# Bridge 6: Complexity-Information Bridge
# ===================================================================

def complexity_information_bridge(
    phi_field: Optional[np.ndarray] = None,
    p: int = 104729,
    n_samples: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    r"""Bridge Fisher information geometry to substrate complexity.

    Bridge equation
    ---------------
        Fisher information:  I_F = E[(d/d theta  ln p)^2]  from info_geometry
        Complexity:          C = d S / dt  from complexity module
        Relationship:        C propto sqrt(I_F)  at mesoscale

    Parameters
    ----------
    phi_field : ndarray, optional
        Boundary phase field samples (n_samples, n_params).
        If None, generates random samples.
    p : int
        Substrate prime modulus (for complexity scaling).
    n_samples : int
        Number of Monte-Carlo samples for Fisher metric estimation.
    seed : int
        Random seed.

    Returns
    -------
    dict
        I_F : float -- Fisher information (trace of Fisher metric)
        complexity : float -- topological complexity measure
        ratio : float -- C / sqrt(I_F) proportionality constant
    """
    try:
        from bpr.info_geometry import FisherMetric
        from bpr.complexity import TopologicalParallelism
    except ImportError as exc:
        return {"error": f"Import failed: {exc}"}

    rng = np.random.default_rng(seed)

    # 1. Generate or use phi_field samples for Fisher metric
    n_params = 4  # boundary field parameter dimension
    if phi_field is None:
        # Simulate field derivative samples: d phi / d theta
        phi_field = rng.normal(0, 1.0, (n_samples, n_params))

    # 2. Compute Fisher metric from samples
    fm = FisherMetric.from_samples(phi_field, sigma_phi=1.0)
    g = fm.g

    # Fisher information scalar: trace of the metric
    I_F = float(np.trace(g))

    # 3. Complexity from topological parallelism
    #    Use winding number W as proxy for computational complexity
    #    The effective complexity scales with log of parallel sectors
    W_eff = np.log(I_F + 1)  # map Fisher info to effective winding
    tp = TopologicalParallelism(p=7, W=W_eff)
    complexity = tp.effective_qubits()  # log2(N_parallel)

    # 4. Check proportionality: C ~ sqrt(I_F)
    sqrt_I_F = np.sqrt(I_F)
    ratio = complexity / sqrt_I_F if sqrt_I_F > 1e-12 else np.inf

    # 5. Substrate correction from prime modulus
    substrate_correction = 1.0 / p

    return {
        "I_F": float(I_F),
        "Fisher_metric": g,
        "complexity": float(complexity),
        "sqrt_I_F": float(sqrt_I_F),
        "ratio_C_over_sqrt_IF": float(ratio),
        "W_eff": float(W_eff),
        "substrate_correction": float(substrate_correction),
        "n_parallel_sectors": float(tp.n_parallel),
        "p": p,
        "n_samples": n_samples,
    }


# ===================================================================
# Bridge 7: Memory-Coherence Chain
# ===================================================================

def memory_coherence_chain(
    tau: float = 1e-9,
    omega: float = 1e12,
    n_steps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    r"""Bridge substrate memory to consciousness via coherence chain.

    Bridge equation
    ---------------
        Memory kernel:  M(t,t') = exp(-|t-t'|/tau_m) cos(omega_r (t-t'))
        -> stain dynamics:  ds/dt = alpha * u_minus * (1-s) - beta * u_plus * s
        -> coherence gain:  K* = K_bar * G(s*) / (K_bar * G(s*) + nu)
        -> consciousness:   K* > K*_crit  =>  conscious

    The full chain from substrate memory through decoherence tracking
    to consciousness onset.

    Parameters
    ----------
    tau : float
        Memory kernel decay timescale (s).
    omega : float
        Memory kernel resonance frequency (rad/s).
    n_steps : int
        Number of time steps for stain evolution.
    seed : int
        Random seed.

    Returns
    -------
    dict
        memory_strength : float -- integrated memory kernel
        s_star : float -- steady-state stain
        K_star : float -- asymptotic coherence
        is_conscious : bool -- whether K* exceeds threshold
    """
    try:
        from bpr.memory import MemoryKernelParams, memory_kernel
        from bpr.coherence_transitions import (
            CoherenceDecayDynamics,
            CoherenceGainFunction,
        )
    except ImportError as exc:
        return {"error": f"Import failed: {exc}"}

    # 1. Build memory kernel
    mk_params = MemoryKernelParams(tau_0=tau, p=7, n=1, W=1.0, alpha=1.0)
    # The __post_init__ sets tau_m from W and omega_r from p

    # Evaluate kernel over a time window
    t_window = np.linspace(0, 10 * mk_params.tau_m, n_steps)
    t_prime = np.zeros_like(t_window)
    M_values = memory_kernel(t_window, t_prime, mk_params)

    # Integrated memory strength
    dt_mem = t_window[1] - t_window[0] if len(t_window) > 1 else 1.0
    memory_strength = float(np.trapz(np.abs(M_values), dx=dt_mem))

    # 2. Map memory strength to noise level
    #    Stronger memory -> less decoherence noise
    #    u_minus ~ 1 / (1 + memory_strength), u_plus ~ memory_strength
    u_minus_const = 1.0 / (1.0 + memory_strength)
    u_plus_const = np.tanh(memory_strength)

    # 3. Stain dynamics: find steady state
    stain = CoherenceDecayDynamics(alpha=1.0, beta=0.5, gamma=0.01, s0=0.0)
    s_star = stain.steady_state(u_plus=u_plus_const, u_minus=u_minus_const)

    # 4. Also evolve the stain dynamics over time
    t_span = (0.0, 50.0)
    t_stain, s_trajectory = stain.evolve(
        t_span,
        u_plus=lambda t: u_plus_const,
        u_minus=lambda t: u_minus_const,
        n_points=n_steps,
    )

    # 5. Coherence gain at steady-state stain
    cgf = CoherenceGainFunction(K_bar=1.0, nu=0.1, kappa_s=2.0, sigma_scale=1.0)
    K_star = cgf.asymptotic_coherence(s_star)
    G_at_sstar = cgf.G(s_star)

    # 6. Consciousness threshold
    K_star_crit = 0.5  # consciousness requires at least half coherence
    is_conscious = bool(K_star > K_star_crit)

    # 7. Evolve coherence given stain trajectory
    s_interp_func = lambda t: float(np.interp(t, t_stain, s_trajectory))
    t_coh, K_trajectory = cgf.evolve_coherence(
        t_span, K0=0.01, s_trajectory=s_interp_func, n_points=n_steps,
    )

    return {
        "memory_kernel_params": {
            "tau_m": float(mk_params.tau_m),
            "omega_r": float(mk_params.omega_r),
            "W": float(mk_params.W),
            "p": mk_params.p,
        },
        "memory_strength": float(memory_strength),
        "u_plus": float(u_plus_const),
        "u_minus": float(u_minus_const),
        "s_star": float(s_star),
        "G_at_s_star": float(G_at_sstar),
        "K_star": float(K_star),
        "K_star_crit": float(K_star_crit),
        "is_conscious": is_conscious,
        "t_stain": t_stain,
        "s_trajectory": s_trajectory,
        "t_coherence": t_coh,
        "K_trajectory": K_trajectory,
        "M_kernel_sample": M_values,
    }
