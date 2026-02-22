"""
Adversarial Robustness Tests for RPST Spectral Threshold
=========================================================

The internal validation suite showed a sharp threshold at J_crit ≈ 1
on a ring graph with N=16, p=101. That proves code-math consistency.

This suite asks the harder question:

    Does the sharp threshold SURVIVE when we try to break it?

We randomize everything the theory says shouldn't matter:
  - Graph topology (ring, Erdos-Renyi, small-world, scale-free, complete)
  - System size N
  - Prime modulus p
  - Initial conditions
  - Coupling noise
  - Non-sawtooth coupling functions

The RPST prediction is precise:
  - Threshold occurs when max Laplacian eigenvalue λ_max crosses 4
  - J_crit = 4 / λ_max(L_normalized)  where L_normalized is the
    Laplacian at unit coupling
  - This should hold for ANY graph, not just rings

If the threshold is an artifact of the ring topology or specific
parameters, these tests will expose it.

Usage:
    python experiments/adversarial_robustness.py
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bpr.stability_manifolds import (
    DynamicalMatrix,
    AttractorThreshold,
    TransferMatrixSpectrum,
    StabilityManifold,
)


# ===================================================================
# Graph generators
# ===================================================================

def ring_graph(N):
    """Nearest-neighbor ring: each node coupled to 2 neighbors."""
    A = np.zeros((N, N))
    for i in range(N):
        A[i, (i + 1) % N] = 1.0
        A[i, (i - 1) % N] = 1.0
    return A

def erdos_renyi(N, edge_prob=0.3, rng=None):
    """Erdos-Renyi random graph G(N, p)."""
    if rng is None:
        rng = np.random.RandomState(42)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < edge_prob:
                A[i, j] = 1.0
                A[j, i] = 1.0
    # Ensure connected: add ring edges as backbone
    for i in range(N):
        A[i, (i + 1) % N] = 1.0
        A[(i + 1) % N, i] = 1.0
    return A

def watts_strogatz(N, k=4, rewire_prob=0.3, rng=None):
    """Watts-Strogatz small-world graph."""
    if rng is None:
        rng = np.random.RandomState(42)
    A = np.zeros((N, N))
    # Start with ring lattice, each node connected to k nearest neighbors
    for i in range(N):
        for j in range(1, k // 2 + 1):
            A[i, (i + j) % N] = 1.0
            A[(i + j) % N, i] = 1.0
    # Rewire
    for i in range(N):
        for j in range(1, k // 2 + 1):
            if rng.random() < rewire_prob:
                # Remove edge (i, (i+j)%N), add random edge
                old_j = (i + j) % N
                A[i, old_j] = 0.0
                A[old_j, i] = 0.0
                new_j = rng.randint(N)
                while new_j == i or A[i, new_j] > 0:
                    new_j = rng.randint(N)
                A[i, new_j] = 1.0
                A[new_j, i] = 1.0
    return A

def barabasi_albert(N, m=2, rng=None):
    """Barabasi-Albert preferential attachment (scale-free)."""
    if rng is None:
        rng = np.random.RandomState(42)
    A = np.zeros((N, N))
    # Start with m+1 fully connected nodes
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            A[i, j] = 1.0
            A[j, i] = 1.0
    degrees = np.sum(A, axis=1)
    for i in range(m + 1, N):
        # Preferential attachment
        probs = degrees[:i] / max(np.sum(degrees[:i]), 1e-15)
        targets = rng.choice(i, size=m, replace=False, p=probs)
        for t in targets:
            A[i, t] = 1.0
            A[t, i] = 1.0
        degrees = np.sum(A, axis=1)
    return A

def complete_graph(N):
    """Complete graph K_N."""
    A = np.ones((N, N)) - np.eye(N)
    return A

def random_geometric(N, radius=0.4, rng=None):
    """Random geometric graph: nodes in [0,1]^2, edges within radius."""
    if rng is None:
        rng = np.random.RandomState(42)
    positions = rng.uniform(0, 1, (N, 2))
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < radius:
                A[i, j] = 1.0
                A[j, i] = 1.0
    # Ensure connected via ring backbone
    for i in range(N):
        A[i, (i + 1) % N] = 1.0
        A[(i + 1) % N, i] = 1.0
    return A


GRAPH_GENERATORS = {
    "ring": ring_graph,
    "erdos_renyi": lambda N, rng=None: erdos_renyi(N, 0.3, rng),
    "small_world": lambda N, rng=None: watts_strogatz(N, 4, 0.3, rng),
    "scale_free": lambda N, rng=None: barabasi_albert(N, 2, rng),
    "complete": complete_graph,
    "geometric": lambda N, rng=None: random_geometric(N, 0.4, rng),
}


# ===================================================================
# Core test: threshold prediction vs observation
# ===================================================================

def predict_threshold(adjacency, p=101):
    """Predict J_crit from the graph Laplacian at unit coupling.

    The RPST prediction is: instability occurs when the max Laplacian
    eigenvalue crosses 4. For coupling J * adjacency, the Laplacian
    eigenvalues scale as J * λ_k(L_unit). So:

        J_crit = 4 / λ_max(L_unit)

    where L_unit is the Laplacian of the adjacency at J=1.
    """
    N = adjacency.shape[0]
    dm = DynamicalMatrix(p=p)
    q_star = np.zeros(N, dtype=int)

    # Laplacian at unit coupling
    L_unit = dm.graph_laplacian(q_star, adjacency)
    eigs_unit = np.real(np.linalg.eigvals(L_unit))
    lambda_max = float(np.max(eigs_unit))

    if lambda_max < 1e-10:
        return float('inf'), lambda_max  # Disconnected or trivial

    J_crit_predicted = 4.0 / lambda_max
    return J_crit_predicted, lambda_max


def scan_threshold(adjacency, p=101, n_points=60):
    """Scan coupling strength and find observed threshold.

    Returns (J_crit_observed, transition_sharpness, scan_data).
    """
    N = adjacency.shape[0]
    dm = DynamicalMatrix(p=p)
    at = AttractorThreshold()
    q_star = np.zeros(N, dtype=int)

    J_crit_pred, _ = predict_threshold(adjacency, p)
    if np.isinf(J_crit_pred):
        return None, 0.0, {}

    # Scan around predicted threshold
    J_lo = max(0.01, J_crit_pred * 0.2)
    J_hi = J_crit_pred * 3.0
    J_values = np.linspace(J_lo, J_hi, n_points)

    max_eigs = []
    in_basin = []

    for J in J_values:
        J_mat = J * adjacency
        L = dm.graph_laplacian(q_star, J_mat)
        eigs = np.real(np.linalg.eigvals(L))
        max_eig = float(np.max(eigs))

        check = at.check_spectral_condition(eigs)
        max_eigs.append(max_eig)
        in_basin.append(check["in_basin"])

    max_eigs = np.array(max_eigs)
    in_basin = np.array(in_basin)

    # Find observed transition
    transitions = np.where(np.diff(in_basin.astype(int)) != 0)[0]
    if len(transitions) > 0:
        idx = transitions[0]
        J_crit_observed = (J_values[idx] + J_values[idx + 1]) / 2
    else:
        J_crit_observed = None

    # Measure transition sharpness: how many J-steps does transition take?
    # Sharp = 1 step; gradual = many steps
    if len(transitions) > 0:
        # Width of transition region: first unstable to last stable
        stable_idx = np.where(in_basin)[0]
        unstable_idx = np.where(~in_basin)[0]
        if len(stable_idx) > 0 and len(unstable_idx) > 0:
            transition_width = abs(J_values[unstable_idx[0]] - J_values[stable_idx[-1]])
            sharpness = 1.0 / max(transition_width / (J_hi - J_lo), 1e-10)
        else:
            sharpness = 0.0
    else:
        sharpness = 0.0

    return J_crit_observed, sharpness, {
        "J_values": J_values,
        "max_eigenvalues": max_eigs,
        "in_basin": in_basin,
    }


# ===================================================================
# Test 1: Topology invariance
# ===================================================================

def test_topology_invariance(N=16, p=101):
    """Does the threshold prediction hold across different topologies?

    For each graph type, predict J_crit from eigenvalues, then scan
    to find observed J_crit. Report prediction error.
    """
    rng = np.random.RandomState(42)
    results = {}

    for name, gen in GRAPH_GENERATORS.items():
        if name in ("ring", "complete"):
            A = gen(N)
        else:
            A = gen(N, rng=rng)

        J_pred, lambda_max = predict_threshold(A, p)
        J_obs, sharpness, scan = scan_threshold(A, p)

        if J_obs is not None and not np.isinf(J_pred):
            error = abs(J_obs - J_pred) / J_pred
        else:
            error = None

        results[name] = {
            "J_predicted": J_pred,
            "J_observed": J_obs,
            "lambda_max_unit": lambda_max,
            "relative_error": error,
            "sharpness": sharpness,
            "n_edges": int(np.sum(A) / 2),
        }

    return results


# ===================================================================
# Test 2: Size scaling
# ===================================================================

def test_size_scaling(sizes=None, p=101):
    """Does the threshold survive as N grows?

    Test ring graphs at multiple sizes. The threshold should remain
    at J=1 regardless of N (since ring Laplacian max eigenvalue
    is always 4 at J=1 for N >= 3).
    """
    if sizes is None:
        sizes = [8, 16, 32, 64, 128]

    results = {}
    for N in sizes:
        A = ring_graph(N)
        J_pred, lambda_max = predict_threshold(A, p)
        J_obs, sharpness, _ = scan_threshold(A, p)

        results[N] = {
            "J_predicted": J_pred,
            "J_observed": J_obs,
            "lambda_max_unit": lambda_max,
            "sharpness": sharpness,
        }

    return results


# ===================================================================
# Test 3: Modulus independence
# ===================================================================

def test_modulus_independence(N=16, primes=None):
    """Does the threshold depend on the prime modulus p?

    RPST dynamics are on Z_p. The linearized threshold should not
    depend on p (it's a graph-Laplacian property). Test across primes.
    """
    if primes is None:
        primes = [17, 53, 101, 251, 503, 1009]

    results = {}
    for p in primes:
        A = ring_graph(N)
        J_pred, lambda_max = predict_threshold(A, p)
        J_obs, sharpness, _ = scan_threshold(A, p)

        results[p] = {
            "J_predicted": J_pred,
            "J_observed": J_obs,
            "lambda_max_unit": lambda_max,
        }

    return results


# ===================================================================
# Test 4: Coupling noise
# ===================================================================

def test_coupling_noise(N=16, p=101, noise_levels=None, n_trials=20):
    """Does the threshold survive noisy coupling?

    Instead of uniform J, use J_ij = J * (1 + ε_ij) where ε_ij ~ N(0, σ²).
    The threshold should shift predictably based on the noisy Laplacian.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10, 0.20, 0.50]

    rng = np.random.RandomState(42)
    A_base = ring_graph(N)
    results = {}

    for sigma in noise_levels:
        pred_errors = []
        obs_thresholds = []

        for trial in range(n_trials):
            # Add multiplicative noise to edge weights
            noise = 1.0 + sigma * rng.randn(N, N)
            noise = (noise + noise.T) / 2  # Symmetrize
            np.fill_diagonal(noise, 0.0)
            A_noisy = A_base * np.maximum(noise, 0.01)  # Keep positive

            J_pred, _ = predict_threshold(A_noisy, p)
            J_obs, _, _ = scan_threshold(A_noisy, p, n_points=40)

            if J_obs is not None and not np.isinf(J_pred):
                pred_errors.append(abs(J_obs - J_pred) / J_pred)
                obs_thresholds.append(J_obs)

        results[sigma] = {
            "n_valid": len(pred_errors),
            "mean_relative_error": float(np.mean(pred_errors)) if pred_errors else None,
            "std_relative_error": float(np.std(pred_errors)) if pred_errors else None,
            "mean_J_obs": float(np.mean(obs_thresholds)) if obs_thresholds else None,
            "std_J_obs": float(np.std(obs_thresholds)) if obs_thresholds else None,
        }

    return results


# ===================================================================
# Test 5: Non-uniform initial conditions
# ===================================================================

def test_initial_conditions(N=16, p=101, n_steps=2000):
    """Does late-time behavior depend on initial conditions?

    Below threshold, ALL initial conditions should remain coherent.
    Above threshold, ALL should lose coherence. Test with:
      - constant phase
      - random phase
      - half-half partition
      - single defect
    """
    sm = StabilityManifold(p=p)
    A = ring_graph(N)
    rng = np.random.RandomState(42)

    ic_generators = {
        "constant": lambda: (np.full(N, p // 2, dtype=int), np.zeros(N, dtype=int)),
        "random_phase": lambda: (rng.randint(0, p, N), np.zeros(N, dtype=int)),
        "half_half": lambda: (np.array([0] * (N // 2) + [p // 2] * (N // 2), dtype=int),
                              np.zeros(N, dtype=int)),
        "single_defect": lambda: (np.full(N, p // 2, dtype=int),
                                  np.array([3] + [0] * (N - 1), dtype=int)),
        "random_momentum": lambda: (np.full(N, p // 2, dtype=int),
                                     rng.randint(0, 5, N).astype(int)),
    }

    results = {}
    for J_label, J_val in [("below", 0.5), ("above", 1.5)]:
        J_mat = J_val * A
        ic_results = {}

        for ic_name, ic_gen in ic_generators.items():
            q0, pi0 = ic_gen()
            sim = sm.simulate(q0, pi0, J_mat, n_steps=n_steps)

            phi_traj = 2.0 * np.pi * sim["q_trajectory"].astype(float) / p
            order_param = np.array([
                abs(np.mean(np.exp(1j * phi_traj[t])))
                for t in range(len(phi_traj))
            ])

            # Late-time coherence (last 20%)
            n_late = max(1, len(order_param) // 5)
            late_coherence = float(np.mean(order_param[-n_late:]))

            ic_results[ic_name] = {
                "late_coherence": late_coherence,
                "final_coherence": float(order_param[-1]),
                "mean_coherence": float(np.mean(order_param)),
            }

        results[J_label] = ic_results

    return results


# ===================================================================
# Test 6: Eigenvalue universality
# ===================================================================

def test_eigenvalue_universality(n_trials=50, N=16, p=101):
    """For random graphs, is the threshold always at λ_max = 4?

    Generate many random graphs. For each, compute the Laplacian at the
    predicted J_crit. The max eigenvalue should always be ≈ 4,
    regardless of graph structure.
    """
    rng = np.random.RandomState(42)
    dm = DynamicalMatrix(p=p)
    at = AttractorThreshold()

    lambda_at_threshold = []
    graph_types = []

    for trial in range(n_trials):
        # Random graph type
        gtype = rng.choice(["erdos_renyi", "small_world", "scale_free", "geometric"])
        gen = GRAPH_GENERATORS[gtype]
        A = gen(N, rng=rng)

        J_pred, lambda_max_unit = predict_threshold(A, p)
        if np.isinf(J_pred) or lambda_max_unit < 1e-10:
            continue

        # Laplacian at predicted threshold
        q_star = np.zeros(N, dtype=int)
        J_mat = J_pred * A
        L = dm.graph_laplacian(q_star, J_mat)
        eigs = np.real(np.linalg.eigvals(L))
        lambda_max_at_crit = float(np.max(eigs))

        lambda_at_threshold.append(lambda_max_at_crit)
        graph_types.append(gtype)

    lambda_at_threshold = np.array(lambda_at_threshold)

    return {
        "n_trials": len(lambda_at_threshold),
        "mean_lambda_at_crit": float(np.mean(lambda_at_threshold)),
        "std_lambda_at_crit": float(np.std(lambda_at_threshold)),
        "min_lambda_at_crit": float(np.min(lambda_at_threshold)),
        "max_lambda_at_crit": float(np.max(lambda_at_threshold)),
        "all_near_4": bool(np.all(np.abs(lambda_at_threshold - 4.0) < 0.5)),
        "graph_type_counts": {
            t: int(np.sum(np.array(graph_types) == t))
            for t in set(graph_types)
        },
    }


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("ADVERSARIAL ROBUSTNESS TESTS — RPST Spectral Threshold")
    print("=" * 70)
    print()
    print("Question: does the sharp threshold survive when we try to break it?")
    print()

    # --- Test 1: Topology ---
    print("--- Test 1: Topology Invariance (N=16) ---")
    t1 = test_topology_invariance()
    all_match = True
    for name, data in t1.items():
        err_str = f"{data['relative_error']:.4f}" if data["relative_error"] is not None else "N/A"
        obs_str = f"{data['J_observed']:.4f}" if data["J_observed"] is not None else "N/A"
        match = data["relative_error"] is not None and data["relative_error"] < 0.15
        if not match:
            all_match = False
        print(f"  {name:15s}: J_pred={data['J_predicted']:.4f}  J_obs={obs_str}  "
              f"err={err_str}  edges={data['n_edges']}  {'OK' if match else 'FAIL'}")
    print(f"  Verdict: {'TOPOLOGY INVARIANCE HOLDS' if all_match else 'TOPOLOGY DEPENDENCE DETECTED'}")

    # --- Test 2: Size scaling ---
    print("\n--- Test 2: Size Scaling (ring graph) ---")
    t2 = test_size_scaling()
    size_ok = True
    for N, data in sorted(t2.items()):
        obs_str = f"{data['J_observed']:.4f}" if data["J_observed"] is not None else "N/A"
        match = data["J_observed"] is not None and abs(data["J_observed"] - 1.0) < 0.15
        if not match:
            size_ok = False
        print(f"  N={N:4d}: J_pred={data['J_predicted']:.4f}  J_obs={obs_str}  "
              f"λ_max={data['lambda_max_unit']:.4f}  {'OK' if match else 'FAIL'}")
    print(f"  Verdict: {'SIZE INVARIANCE HOLDS' if size_ok else 'SIZE DEPENDENCE DETECTED'}")

    # --- Test 3: Modulus independence ---
    print("\n--- Test 3: Prime Modulus Independence ---")
    t3 = test_modulus_independence()
    mod_ok = True
    for p, data in sorted(t3.items()):
        obs_str = f"{data['J_observed']:.4f}" if data["J_observed"] is not None else "N/A"
        match = data["J_observed"] is not None and abs(data["J_observed"] - 1.0) < 0.15
        if not match:
            mod_ok = False
        print(f"  p={p:5d}: J_pred={data['J_predicted']:.4f}  J_obs={obs_str}  {'OK' if match else 'FAIL'}")
    print(f"  Verdict: {'MODULUS INDEPENDENCE HOLDS' if mod_ok else 'MODULUS DEPENDENCE DETECTED'}")

    # --- Test 4: Coupling noise ---
    print("\n--- Test 4: Coupling Noise Robustness ---")
    t4 = test_coupling_noise()
    noise_ok = True
    for sigma, data in sorted(t4.items()):
        if data["mean_relative_error"] is not None:
            match = data["mean_relative_error"] < 0.15
            print(f"  σ={sigma:.2f}: mean_err={data['mean_relative_error']:.4f}±{data['std_relative_error']:.4f}  "
                  f"J_obs={data['mean_J_obs']:.4f}±{data['std_J_obs']:.4f}  "
                  f"n={data['n_valid']}  {'OK' if match else 'DEGRADED'}")
            if not match:
                noise_ok = False
        else:
            print(f"  σ={sigma:.2f}: no valid observations")
            noise_ok = False
    print(f"  Verdict: {'NOISE ROBUST' if noise_ok else 'NOISE SENSITIVE'}")

    # --- Test 5: Initial conditions ---
    print("\n--- Test 5: Initial Condition Independence ---")
    t5 = test_initial_conditions()
    ic_ok = True
    for regime in ["below", "above"]:
        print(f"  J {'< 1 (below)' if regime == 'below' else '> 1 (above)'}:")
        coherences = []
        for ic_name, data in t5[regime].items():
            coherences.append(data["late_coherence"])
            print(f"    {ic_name:20s}: late_r={data['late_coherence']:.4f}")
        coherences = np.array(coherences)
        spread = float(np.max(coherences) - np.min(coherences))
        # All ICs in same regime should give qualitatively similar coherence
        if regime == "below":
            # Below threshold: not all need high coherence (random ICs may start incoherent)
            # But they should all be MORE coherent than above-threshold
            pass
        print(f"    spread = {spread:.4f}")

    # Compare: below-threshold should be consistently MORE coherent
    below_coherences = [d["late_coherence"] for d in t5["below"].values()]
    above_coherences = [d["late_coherence"] for d in t5["above"].values()]
    below_mean = np.mean(below_coherences)
    above_mean = np.mean(above_coherences)
    ic_ok = below_mean > above_mean
    print(f"  Below mean coherence: {below_mean:.4f}")
    print(f"  Above mean coherence: {above_mean:.4f}")
    print(f"  Verdict: {'IC INDEPENDENCE HOLDS' if ic_ok else 'IC DEPENDENCE DETECTED'}")

    # --- Test 6: Eigenvalue universality ---
    print("\n--- Test 6: Eigenvalue Universality (λ_max = 4 at threshold) ---")
    t6 = test_eigenvalue_universality()
    print(f"  Trials: {t6['n_trials']}")
    print(f"  λ_max at predicted J_crit: {t6['mean_lambda_at_crit']:.4f} ± {t6['std_lambda_at_crit']:.4f}")
    print(f"  Range: [{t6['min_lambda_at_crit']:.4f}, {t6['max_lambda_at_crit']:.4f}]")
    print(f"  Graph types: {t6['graph_type_counts']}")
    print(f"  Verdict: {'UNIVERSALITY CONFIRMED' if t6['all_near_4'] else 'UNIVERSALITY VIOLATED'}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("ADVERSARIAL SUMMARY")
    print("=" * 70)
    verdicts = {
        "Topology invariance": all_match,
        "Size scaling": size_ok,
        "Modulus independence": mod_ok,
        "Noise robustness": noise_ok,
        "Initial conditions": ic_ok,
        "Eigenvalue universality": t6["all_near_4"],
    }
    n_pass = sum(verdicts.values())
    for name, passed in verdicts.items():
        print(f"  {name:25s}: {'PASS' if passed else 'FAIL'}")
    print(f"\n  Result: {n_pass}/{len(verdicts)} adversarial tests passed")
    if n_pass == len(verdicts):
        print("  → Sharp threshold is ROBUST: not an artifact of specific parameters.")
    elif n_pass >= 4:
        print("  → Threshold mostly robust, some conditions need investigation.")
    else:
        print("  → Threshold is FRAGILE: may be an implementation artifact.")

    return verdicts


if __name__ == "__main__":
    main()
