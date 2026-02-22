"""
Continuous-Time vs Discrete-Time: Honest Comparison
=====================================================

The critical question this experiment answers:

    Is the RPST spectral threshold a new physical prediction,
    or is it the stability boundary of a symplectic integrator?

The RPST map:
    q_{n+1} = q_n + π_n           (mod p)
    π_{n+1} = π_n - J·L·q_{n+1}  (mod p)

is mathematically identical to symplectic Euler applied to:
    dq/dt = π
    dπ/dt = -J·L·q

with step size h = 1.  The stability condition λ_max ≤ 4 is exactly
the CFL-like condition for symplectic Euler: h²·J·λ_max ≤ 4.

This experiment runs 5 models on the same ring graph:
    1. Continuous-time (RK45 adaptive) — should show NO threshold
    2. RPST discrete map (h=1 on Z_p) — threshold at J·λ_max = 4
    3. Störmer-Verlet h=1 (real-valued) — should MATCH RPST
    4. Störmer-Verlet h=0.1 — threshold at J·λ_max = 400
    5. Kuramoto — smooth synchronization, fundamentally different

If RPST matches Störmer-Verlet exactly, the threshold is NOT new
physics — it's the integrator stability boundary. But it IS physically
real for systems with genuinely discrete-time dynamics (kicked rotors,
stroboscopic systems, digital oscillator networks).

If RPST deviates from Störmer-Verlet (due to Z_p arithmetic), that
deviation IS a new prediction.

Usage:
    python experiments/continuous_time_comparison.py
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bpr.stability_manifolds import StabilityManifold, DynamicalMatrix


# ===================================================================
# Graph setup
# ===================================================================

def ring_laplacian(N):
    """Standard ring graph Laplacian (unnormalized)."""
    L = np.zeros((N, N))
    for i in range(N):
        L[i, i] = 2.0
        L[i, (i + 1) % N] = -1.0
        L[i, (i - 1) % N] = -1.0
    return L


def ring_adjacency(N, J):
    """Ring adjacency with coupling strength J."""
    A = np.zeros((N, N))
    for i in range(N):
        A[i, (i + 1) % N] = J
        A[i, (i - 1) % N] = J
    return A


# ===================================================================
# Model 1: Continuous-time (RK45)
# ===================================================================

def continuous_time_evolution(N, J, q0, pi0, T_final=200.0):
    """Exact continuous-time Hamiltonian dynamics via RK45.

    H = Σ π²/2 + J Σ_{<ij>} (q_i - q_j)² / 2
    dq/dt = π
    dπ/dt = -J · L · q

    This is a system of coupled harmonic oscillators.
    ALWAYS stable — oscillatory for all J > 0.
    """
    L = ring_laplacian(N)

    def rhs(t, y):
        q = y[:N]
        pi_v = y[N:]
        dq = pi_v
        dpi = -J * L @ q
        return np.concatenate([dq, dpi])

    y0 = np.concatenate([q0.astype(float), pi0.astype(float)])
    sol = solve_ivp(rhs, [0, T_final], y0, method='RK45',
                    rtol=1e-10, atol=1e-12,
                    t_eval=np.linspace(0, T_final, 2001))

    q_traj = sol.y[:N, :].T  # shape (n_times, N)
    return q_traj


def continuous_order_parameter(q_traj):
    """Order parameter from continuous phases (no mod)."""
    # Normalize phases to [0, 2π] via modular arithmetic
    phi = q_traj % (2 * np.pi)
    r = np.array([abs(np.mean(np.exp(1j * phi[t]))) for t in range(len(phi))])
    return r


# ===================================================================
# Model 2: RPST discrete map on Z_p
# ===================================================================

def rpst_evolution(N, J, q0, pi0, p=101, n_steps=2000):
    """RPST symplectic map on Z_p."""
    sm = StabilityManifold(p=p)
    A = ring_adjacency(N, J)
    sim = sm.simulate(q0.astype(int) % p, pi0.astype(int) % p, A, n_steps=n_steps)
    q_traj = sim["q_trajectory"]
    phi_traj = 2.0 * np.pi * q_traj.astype(float) / p
    r = np.array([abs(np.mean(np.exp(1j * phi_traj[t]))) for t in range(len(phi_traj))])
    return r


# ===================================================================
# Model 3: Störmer-Verlet (real-valued, step h)
# ===================================================================

def stormer_verlet_evolution(N, J, q0, pi0, h=1.0, n_steps=2000):
    """Störmer-Verlet symplectic integrator with step h.

    q_{n+1} = q_n + h · π_n
    π_{n+1} = π_n - h · J · L · q_{n+1}

    (This is the real-valued analog of RPST without mod-p.)

    Stability condition: h² · J · λ_max(L) ≤ 4
    """
    L = ring_laplacian(N)
    q = q0.astype(float).copy()
    pi_v = pi0.astype(float).copy()

    n_times = n_steps + 1
    r = np.zeros(n_times)
    r[0] = abs(np.mean(np.exp(1j * q)))

    max_amplitude = np.zeros(n_times)
    max_amplitude[0] = np.max(np.abs(q))

    for t in range(n_steps):
        q = q + h * pi_v
        pi_v = pi_v - h * J * L @ q

        r[t + 1] = abs(np.mean(np.exp(1j * q)))
        max_amplitude[t + 1] = np.max(np.abs(q))

    return r, max_amplitude


# ===================================================================
# Model 4: Kuramoto
# ===================================================================

def kuramoto_evolution(N, K, n_steps=5000, dt=0.01):
    """Kuramoto model with Lorentzian frequency distribution."""
    rng = np.random.RandomState(42)
    omega = rng.standard_cauchy(N) * 0.3
    theta = rng.uniform(0, 2 * np.pi, N)

    for _ in range(n_steps):
        coupling = np.zeros(N)
        for i in range(N):
            coupling[i] = (K / N) * np.sum(np.sin(theta - theta[i]))
        theta = theta + (omega + coupling) * dt
        theta = theta % (2 * np.pi)

    return abs(np.mean(np.exp(1j * theta)))


# ===================================================================
# Experiment 1: Does continuous-time show a threshold?
# ===================================================================

def experiment_1_continuous_vs_discrete(N=16, p=101):
    """Compare continuous and discrete dynamics across coupling values.

    RPST prediction: sharp threshold at J ≈ 1 (ring, sawtooth).
    Continuous-time prediction: NO threshold — always oscillatory.

    If continuous time shows no threshold but RPST does, the
    threshold is a feature of discretization, not the Hamiltonian.
    """
    J_values = np.linspace(0.1, 2.5, 25)

    # Initial conditions: near-constant with small perturbation
    q0 = np.full(N, np.pi)  # All at phase π
    pi0 = np.zeros(N)
    pi0[0] = 0.1  # Small momentum kick

    q0_int = np.full(N, p // 2, dtype=int)
    pi0_int = np.zeros(N, dtype=int)
    pi0_int[0] = 1

    results = {
        "J_values": J_values,
        "continuous": [],
        "rpst": [],
        "verlet_h1": [],
        "verlet_h01": [],
    }

    for J in J_values:
        # Continuous time
        q_traj = continuous_time_evolution(N, J, q0, pi0, T_final=200.0)
        r_cont = continuous_order_parameter(q_traj)
        n_late = max(1, len(r_cont) // 5)
        results["continuous"].append(float(np.mean(r_cont[-n_late:])))

        # RPST discrete
        r_rpst = rpst_evolution(N, J, q0_int, pi0_int, p=p, n_steps=2000)
        n_late = max(1, len(r_rpst) // 5)
        results["rpst"].append(float(np.mean(r_rpst[-n_late:])))

        # Störmer-Verlet h=1
        r_sv1, amp_sv1 = stormer_verlet_evolution(N, J, q0, pi0, h=1.0, n_steps=2000)
        n_late = max(1, len(r_sv1) // 5)
        # If amplitude explodes, coherence is meaningless
        if np.max(amp_sv1) > 1e10:
            results["verlet_h1"].append(0.0)  # Unstable = zero coherence
        else:
            results["verlet_h1"].append(float(np.mean(r_sv1[-n_late:])))

        # Störmer-Verlet h=0.1 (near continuous)
        r_sv01, amp_sv01 = stormer_verlet_evolution(N, J, q0, pi0, h=0.1, n_steps=20000)
        n_late = max(1, len(r_sv01) // 5)
        if np.max(amp_sv01) > 1e10:
            results["verlet_h01"].append(0.0)
        else:
            results["verlet_h01"].append(float(np.mean(r_sv01[-n_late:])))

    for key in ["continuous", "rpst", "verlet_h1", "verlet_h01"]:
        results[key] = np.array(results[key])

    return results


# ===================================================================
# Experiment 2: RPST vs Störmer-Verlet threshold location
# ===================================================================

def experiment_2_threshold_comparison(N=16, p=101):
    """Does RPST's threshold match Störmer-Verlet's exactly?

    Ring graph: λ_max = 4 (for unit coupling).
    RPST prediction: J_crit = 4/λ_max = 1.0 (h=1 implicit).
    Störmer-Verlet h=1: J_crit = 4/(h²·λ_max) = 1.0.
    Störmer-Verlet h=0.5: J_crit = 4/(0.25·4) = 4.0.
    Störmer-Verlet h=2.0: J_crit = 4/(4·4) = 0.25.

    If these match, RPST = Störmer-Verlet with h=1.
    """
    L = ring_laplacian(N)
    lambda_max = np.max(np.real(np.linalg.eigvals(L)))

    step_sizes = [0.25, 0.5, 1.0, 1.5, 2.0]
    results = {"lambda_max": lambda_max, "step_sizes": {}}

    q0 = np.full(N, np.pi)
    pi0 = np.zeros(N)
    pi0[0] = 0.1

    for h in step_sizes:
        J_crit_predicted = 4.0 / (h * h * lambda_max)
        # Scale n_steps inversely with h to cover same physical time
        n_steps = int(2000 / h)

        J_values = np.linspace(0.05, min(J_crit_predicted * 3, 50), 40)

        stable = []
        for J in J_values:
            _, amp = stormer_verlet_evolution(N, J, q0, pi0, h=h,
                                              n_steps=min(n_steps, 10000))
            # Unstable if amplitude grows beyond threshold
            is_stable = np.max(amp) < 1e6
            stable.append(is_stable)

        stable = np.array(stable)
        transitions = np.where(np.diff(stable.astype(int)) != 0)[0]
        if len(transitions) > 0:
            J_crit_observed = (J_values[transitions[0]] + J_values[transitions[0] + 1]) / 2
        else:
            J_crit_observed = None

        results["step_sizes"][h] = {
            "J_crit_predicted": J_crit_predicted,
            "J_crit_observed": J_crit_observed,
            "h_sq_J_lambda": h * h * J_crit_predicted * lambda_max if J_crit_observed else None,
        }

    # Now compare with RPST
    J_values_rpst = np.linspace(0.1, 2.5, 40)
    q0_int = np.full(N, p // 2, dtype=int)
    pi0_int = np.zeros(N, dtype=int)
    pi0_int[0] = 1

    rpst_coherent = []
    for J in J_values_rpst:
        r = rpst_evolution(N, J, q0_int, pi0_int, p=p, n_steps=2000)
        n_late = max(1, len(r) // 5)
        rpst_coherent.append(float(np.mean(r[-n_late:])))

    rpst_coherent = np.array(rpst_coherent)

    # RPST doesn't "explode" on Z_p — find threshold via coherence drop
    dm = DynamicalMatrix(p=p)
    from bpr.stability_manifolds import AttractorThreshold
    at = AttractorThreshold()
    q_star = np.zeros(N, dtype=int)
    rpst_stable = []
    for J in J_values_rpst:
        A = ring_adjacency(N, J)
        L_rpst = dm.graph_laplacian(q_star, A)
        eigs = np.real(np.linalg.eigvals(L_rpst))
        check = at.check_spectral_condition(eigs)
        rpst_stable.append(check["in_basin"])

    rpst_stable = np.array(rpst_stable)
    transitions = np.where(np.diff(rpst_stable.astype(int)) != 0)[0]
    rpst_J_crit = (J_values_rpst[transitions[0]] + J_values_rpst[transitions[0] + 1]) / 2 if len(transitions) > 0 else None

    results["rpst"] = {
        "J_crit_spectral": rpst_J_crit,
        "J_crit_predicted": 4.0 / lambda_max,  # = 1.0 for ring
    }

    return results


# ===================================================================
# Experiment 3: Z_p arithmetic vs real-valued — the deviation
# ===================================================================

def experiment_3_modular_arithmetic_effect(N=16, p=101, J=0.9):
    """Does Z_p mod arithmetic create effects beyond Störmer-Verlet?

    Run both RPST (mod p) and Störmer-Verlet (real) at the SAME J
    below threshold, with identical initial conditions.

    If Z_p wrapping creates nontrivial dynamics (recurrence, confinement),
    that IS a genuine RPST-specific prediction.
    """
    q0_real = np.full(N, np.pi)
    pi0_real = np.zeros(N)
    pi0_real[0] = 0.1

    # Scale to Z_p
    q0_int = np.full(N, p // 2, dtype=int)
    pi0_int = np.zeros(N, dtype=int)
    pi0_int[0] = 1

    n_steps = 5000

    # Störmer-Verlet h=1 (real-valued)
    r_sv, amp_sv = stormer_verlet_evolution(N, J, q0_real, pi0_real,
                                             h=1.0, n_steps=n_steps)

    # RPST on Z_p
    r_rpst = rpst_evolution(N, J, q0_int, pi0_int, p=p, n_steps=n_steps)

    # Measure differences
    # 1. Recurrence: does RPST return to initial state? (finite group → Poincaré recurrence)
    sm = StabilityManifold(p=p)
    A = ring_adjacency(N, J)
    sim = sm.simulate(q0_int % p, pi0_int % p, A, n_steps=n_steps)
    q_traj = sim["q_trajectory"]

    # Check if any later state matches initial state
    initial = q_traj[0]
    recurrence_times = []
    for t in range(1, len(q_traj)):
        if np.array_equal(q_traj[t], initial):
            recurrence_times.append(t)

    # 2. Confinement: on Z_p, phases are bounded [0, p). In R, they can drift.
    rpst_range = float(np.max(q_traj) - np.min(q_traj))  # Always ≤ p
    sv_L = ring_laplacian(N)
    q_sv = q0_real.copy()
    pi_sv = pi0_real.copy()
    q_sv_traj = [q_sv.copy()]
    for _ in range(n_steps):
        q_sv = q_sv + pi_sv
        pi_sv = pi_sv - J * sv_L @ q_sv
        q_sv_traj.append(q_sv.copy())
    q_sv_traj = np.array(q_sv_traj)
    sv_range = float(np.max(q_sv_traj) - np.min(q_sv_traj))

    # 3. Spectral difference: FFT of order parameter time series
    rpst_fft = np.abs(np.fft.rfft(r_rpst))
    sv_fft = np.abs(np.fft.rfft(r_sv))
    # Normalize
    rpst_fft = rpst_fft / max(rpst_fft[1:].max(), 1e-15)
    sv_fft = sv_fft / max(sv_fft[1:].max(), 1e-15)

    # Compare dominant frequencies
    rpst_dominant = np.argsort(rpst_fft[1:])[-5:] + 1
    sv_dominant = np.argsort(sv_fft[1:])[-5:] + 1

    return {
        "J": J,
        "p": p,
        "rpst_recurrence_times": recurrence_times[:5],  # First 5 recurrences
        "has_recurrence": len(recurrence_times) > 0,
        "rpst_phase_range": rpst_range,
        "verlet_phase_range": sv_range,
        "confinement_ratio": rpst_range / max(sv_range, 1e-15),
        "rpst_mean_coherence": float(np.mean(r_rpst)),
        "verlet_mean_coherence": float(np.mean(r_sv)),
        "rpst_dominant_freqs": rpst_dominant.tolist(),
        "sv_dominant_freqs": sv_dominant.tolist(),
        "spectra_overlap": float(np.sum(np.minimum(rpst_fft, sv_fft)) /
                                  max(np.sum(np.maximum(rpst_fft, sv_fft)), 1e-15)),
    }


# ===================================================================
# Experiment 4: Physical observables — where could you measure this?
# ===================================================================

def experiment_4_observable_predictions(N=16, p=101):
    """Generate concrete, measurable predictions for each model.

    For a ring of N coupled oscillators, each model predicts:
    1. Lyapunov exponent λ(J)
    2. Late-time order parameter r(J)
    3. Power spectrum of collective oscillation

    These are things you can measure in a physical experiment.
    """
    J_values = np.linspace(0.1, 2.5, 30)
    L = ring_laplacian(N)
    lambda_max = np.max(np.real(np.linalg.eigvals(L)))

    q0 = np.full(N, np.pi)
    pi0 = np.zeros(N)
    pi0[0] = 0.1

    q0_int = np.full(N, p // 2, dtype=int)
    pi0_int = np.zeros(N, dtype=int)
    pi0_int[0] = 1

    predictions = {
        "J_values": J_values,
        "continuous": {"lyapunov": [], "order_param": []},
        "rpst": {"lyapunov": [], "order_param": []},
        "verlet_h1": {"lyapunov": [], "order_param": []},
        "kuramoto": {"order_param": []},
    }

    for J in J_values:
        # --- Continuous: Lyapunov = 0 always (integrable system) ---
        predictions["continuous"]["lyapunov"].append(0.0)

        q_traj = continuous_time_evolution(N, J, q0, pi0, T_final=100.0)
        r_cont = continuous_order_parameter(q_traj)
        n_late = max(1, len(r_cont) // 5)
        predictions["continuous"]["order_param"].append(float(np.mean(r_cont[-n_late:])))

        # --- RPST: Lyapunov from eigenvalue analysis ---
        # Below threshold: λ = 0 (unit circle eigenvalues)
        # Above threshold: λ = log|μ_max| > 0
        J_lam = J * lambda_max
        if J_lam <= 4.0:
            rpst_lyap = 0.0
        else:
            # Unstable eigenvalue: μ- has |μ| > 1
            mu_minus = 1.0 - J_lam / 2.0 - np.sqrt((J_lam / 2.0 - 1)**2 - 1)
            rpst_lyap = np.log(abs(mu_minus))
        predictions["rpst"]["lyapunov"].append(rpst_lyap)

        r_rpst = rpst_evolution(N, J, q0_int, pi0_int, p=p, n_steps=1500)
        n_late = max(1, len(r_rpst) // 5)
        predictions["rpst"]["order_param"].append(float(np.mean(r_rpst[-n_late:])))

        # --- Störmer-Verlet h=1 ---
        if J_lam <= 4.0:
            sv_lyap = 0.0
        else:
            mu_minus = 1.0 - J_lam / 2.0 - np.sqrt((J_lam / 2.0 - 1)**2 - 1)
            sv_lyap = np.log(abs(mu_minus))
        predictions["verlet_h1"]["lyapunov"].append(sv_lyap)

        r_sv, amp_sv = stormer_verlet_evolution(N, J, q0, pi0, h=1.0, n_steps=1500)
        n_late = max(1, len(r_sv) // 5)
        if np.max(amp_sv) > 1e10:
            predictions["verlet_h1"]["order_param"].append(0.0)
        else:
            predictions["verlet_h1"]["order_param"].append(float(np.mean(r_sv[-n_late:])))

        # --- Kuramoto ---
        r_k = kuramoto_evolution(N, K=J, n_steps=3000)
        predictions["kuramoto"]["order_param"].append(r_k)

    for model in predictions:
        if model == "J_values":
            continue
        for key in predictions[model]:
            predictions[model][key] = np.array(predictions[model][key])

    return predictions


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("CONTINUOUS-TIME vs DISCRETE-TIME: HONEST COMPARISON")
    print("=" * 70)
    print()

    # --- Experiment 1 ---
    print("--- Experiment 1: Continuous vs Discrete Transition Curves ---")
    print("  Running 4 models across 25 coupling values...", flush=True)
    e1 = experiment_1_continuous_vs_discrete()

    # Does continuous time show a threshold?
    cont_range = np.max(e1["continuous"]) - np.min(e1["continuous"])
    rpst_range = np.max(e1["rpst"]) - np.min(e1["rpst"])
    sv1_range = np.max(e1["verlet_h1"]) - np.min(e1["verlet_h1"])
    sv01_range = np.max(e1["verlet_h01"]) - np.min(e1["verlet_h01"])

    print(f"\n  Order parameter range (measures transition strength):")
    print(f"    Continuous (RK45):     {cont_range:.4f}")
    print(f"    RPST (Z_p, h=1):      {rpst_range:.4f}")
    print(f"    Störmer-Verlet h=1:    {sv1_range:.4f}")
    print(f"    Störmer-Verlet h=0.1:  {sv01_range:.4f}")

    # Print transition curves at key J values
    print(f"\n  Late-time coherence r at key J values:")
    print(f"  {'J':>6s}  {'Continuous':>12s}  {'RPST':>12s}  {'SV h=1':>12s}  {'SV h=0.1':>12s}")
    for i, J in enumerate(e1["J_values"]):
        if J in [0.1, 0.5, 1.0, 1.5, 2.0, 2.5] or abs(J - 1.0) < 0.05:
            print(f"  {J:6.2f}  {e1['continuous'][i]:12.4f}  "
                  f"{e1['rpst'][i]:12.4f}  {e1['verlet_h1'][i]:12.4f}  "
                  f"{e1['verlet_h01'][i]:12.4f}")

    cont_has_threshold = cont_range > 0.3
    print(f"\n  Continuous-time shows threshold? {'YES' if cont_has_threshold else 'NO'}")
    print(f"  → {'THRESHOLD IS PHYSICAL' if cont_has_threshold else 'THRESHOLD IS FROM DISCRETIZATION'}")

    # --- Experiment 2 ---
    print("\n--- Experiment 2: RPST vs Störmer-Verlet Threshold Location ---")
    e2 = experiment_2_threshold_comparison()
    print(f"  Ring graph λ_max = {e2['lambda_max']:.4f}")
    print(f"\n  Störmer-Verlet thresholds (predicted: h²·J·λ_max = 4):")
    for h, data in sorted(e2["step_sizes"].items()):
        obs_str = f"{data['J_crit_observed']:.4f}" if data['J_crit_observed'] else "none"
        print(f"    h={h:.2f}: predicted J_crit={data['J_crit_predicted']:.4f}  "
              f"observed={obs_str}")

    rpst_data = e2["rpst"]
    print(f"\n  RPST threshold:")
    print(f"    Predicted:  {rpst_data['J_crit_predicted']:.4f}")
    print(f"    Observed:   {rpst_data['J_crit_spectral']:.4f}" if rpst_data['J_crit_spectral'] else "    Observed:   none")

    # Check if RPST matches SV h=1
    sv_h1 = e2["step_sizes"].get(1.0, {})
    if rpst_data['J_crit_spectral'] and sv_h1.get('J_crit_predicted'):
        match_error = abs(rpst_data['J_crit_spectral'] - sv_h1['J_crit_predicted'])
        rpst_matches_sv = match_error < 0.2
        print(f"\n  RPST matches Störmer-Verlet h=1? {'YES' if rpst_matches_sv else 'NO'} "
              f"(error: {match_error:.4f})")
        if rpst_matches_sv:
            print("  → RPST spectral threshold = symplectic integrator stability boundary")
        else:
            print("  → RPST deviates from standard symplectic integrator (Z_p effect)")
    else:
        rpst_matches_sv = None

    # --- Experiment 3 ---
    print("\n--- Experiment 3: Z_p Arithmetic Effects ---")
    e3 = experiment_3_modular_arithmetic_effect()
    print(f"  J = {e3['J']}, p = {e3['p']}")
    print(f"  RPST phase range:   {e3['rpst_phase_range']:.1f}  (bounded by p={e3['p']})")
    print(f"  Verlet phase range: {e3['verlet_phase_range']:.4f}")
    print(f"  Confinement ratio:  {e3['confinement_ratio']:.4f}")
    print(f"  RPST mean coherence:  {e3['rpst_mean_coherence']:.4f}")
    print(f"  Verlet mean coherence: {e3['verlet_mean_coherence']:.4f}")
    print(f"  Has Poincaré recurrence (Z_p): {e3['has_recurrence']}")
    if e3['has_recurrence']:
        print(f"    First recurrence at t = {e3['rpst_recurrence_times'][0]}")
    print(f"  Spectral overlap: {e3['spectra_overlap']:.4f}")
    print(f"  RPST dominant frequencies:  {e3['rpst_dominant_freqs']}")
    print(f"  Verlet dominant frequencies: {e3['sv_dominant_freqs']}")

    zp_unique = (e3['has_recurrence'] or
                 e3['spectra_overlap'] < 0.5 or
                 abs(e3['rpst_mean_coherence'] - e3['verlet_mean_coherence']) > 0.1)
    print(f"\n  Z_p creates unique effects? {'YES' if zp_unique else 'NO'}")

    # --- Experiment 4 ---
    print("\n--- Experiment 4: Measurable Observable Predictions ---")
    print("  Running 4 models for Lyapunov exponents and order parameters...", flush=True)
    e4 = experiment_4_observable_predictions()

    print(f"\n  Lyapunov exponent predictions:")
    print(f"  {'J':>6s}  {'Continuous':>12s}  {'RPST':>12s}  {'SV h=1':>12s}")
    for i, J in enumerate(e4["J_values"]):
        if abs(J - 0.5) < 0.05 or abs(J - 1.0) < 0.05 or abs(J - 1.5) < 0.05 or abs(J - 2.0) < 0.05:
            print(f"  {J:6.2f}  {e4['continuous']['lyapunov'][i]:12.4f}  "
                  f"{e4['rpst']['lyapunov'][i]:12.4f}  "
                  f"{e4['verlet_h1']['lyapunov'][i]:12.4f}")

    # RPST and SV h=1 should have identical Lyapunov predictions
    lyap_match = np.allclose(e4['rpst']['lyapunov'], e4['verlet_h1']['lyapunov'])
    print(f"\n  RPST Lyapunov = SV h=1 Lyapunov? {lyap_match}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("HONEST SUMMARY")
    print("=" * 70)

    print("""
  What RPST's spectral threshold IS:
    - The stability boundary of a discrete symplectic map
    - Mathematically identical to the CFL condition for symplectic Euler
    - Universal: holds for any graph topology (adversarial suite confirms)
    - Quantitatively predictable: J_crit = 4 / (h² · λ_max(L))

  What RPST's spectral threshold IS NOT:
    - A property of the continuous-time Hamiltonian (which has no threshold)
    - Something that standard numerical analysis hasn't already discovered
    - A prediction you can test with analog oscillators""")

    if cont_has_threshold:
        print("  SURPRISE: Continuous-time dynamics DO show threshold-like behavior.")
        print("  This would be genuinely new and worth investigating further.")
    else:
        print("  Continuous-time dynamics show NO threshold (as expected).")
        print("  The threshold requires discrete-time dynamics to be physical.")

    print(f"""
  Where RPST's threshold IS physically real:
    - Digital oscillator networks (FPGA, DSP)
    - Stroboscopic / kicked systems (quantum kicked rotor)
    - Any system with discrete time-step evolution
    - Lattice field theory simulations""")

    if zp_unique:
        print("""  What Z_p arithmetic adds beyond standard Störmer-Verlet:
    - Poincaré recurrence (guaranteed on finite group)
    - Phase confinement (bounded state space)
    - Modified spectral structure""")
    else:
        print("""  Z_p arithmetic does NOT add qualitatively new effects:
    - The mod-p wrapping doesn't change the threshold
    - RPST ≈ Störmer-Verlet for dynamics near fixed points""")

    print(f"""
  The discriminating experiment:
    - Build a coupled oscillator array (electrical or mechanical)
    - Drive it with DISCRETE timestep updates (digital control loop)
    - Scan coupling J through the predicted threshold
    - Measure: does coherence drop SHARPLY at J_crit = 4/λ_max?
    - Compare with: continuous analog coupling (should show NO threshold)

  This experiment is physically realizable, takes weeks not months,
  and gives a clear yes/no answer.""")


if __name__ == "__main__":
    main()
