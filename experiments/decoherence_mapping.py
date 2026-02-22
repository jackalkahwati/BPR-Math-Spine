"""
RPST Decoherence vs Superconducting Qubit Experiments
=======================================================

Maps RPST predictions to measurable quantities in superconducting
qubit arrays and identifies where RPST diverges from standard models.

The mapping:
  - Qubit phases φ_i  →  RPST boundary phase field Φ
  - Qubit-resonator coupling g_ij  →  RPST coupling matrix J
  - Coupling graph (bus resonators)  →  RPST graph Laplacian L
  - Qubit decoherence  →  RPST boundary phase gradient Γ_dec

SINGLE-QUBIT LEVEL (not discriminating):
  RPST: Γ_dec = (κ_s/ℏ) |∇Φ|²
  Purcell: Γ_P = (g/Δ)² κ
  These give the same scaling when κ_s ~ ℏκ and |∇Φ| ~ g/Δ.

MULTI-QUBIT LEVEL (discriminating):
  Standard: each qubit decoheres independently.
    Γ_i = Γ_Purcell,i + Γ_dielectric,i  (per-qubit rates)
    Collective modes inherit rates: Γ_mode = Σ |c_ik|² Γ_i

  RPST: collective modes decohere via graph Laplacian.
    Γ_k = (κ_s/ℏ) λ_k(L_coupling)
    Modes with λ_k = 0 (in stability manifold) are PROTECTED.
    Decoherence rate depends on MODE STRUCTURE, not just
    individual qubit quality.

The discriminating prediction:
  Standard says:  worst qubit limits all modes
  RPST says:      some modes are protected by graph symmetry,
                  independent of individual qubit quality

This is testable with existing multi-qubit experiments.

Usage:
    python experiments/decoherence_mapping.py
"""

import numpy as np
from scipy import linalg
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bpr.stability_manifolds import (
    DynamicalMatrix,
    TransferMatrixSpectrum,
    PhaseGradientDecoherence,
    AttractorThreshold,
)


# ===================================================================
# Qubit array model
# ===================================================================

class QubitArray:
    """Model of a superconducting transmon qubit array.

    Parameters use typical published values:
      - Qubit frequency: 4-6 GHz
      - Resonator frequency: 6-8 GHz
      - Coupling g/2π: 10-200 MHz
      - Resonator linewidth κ/2π: 0.1-10 MHz
      - T1: 20-300 μs
    """

    def __init__(self, qubit_freqs, resonator_freqs, couplings_g,
                 adjacency, kappa_r, T1_individual):
        """
        Parameters
        ----------
        qubit_freqs : ndarray, shape (N,)
            Qubit frequencies ω_q/2π in GHz.
        resonator_freqs : ndarray, shape (N_edges,)
            Bus resonator frequencies ω_r/2π in GHz.
        couplings_g : ndarray, shape (N, N)
            Coupling strengths g_ij/2π in GHz.
        adjacency : ndarray, shape (N, N)
            Graph adjacency (1 if coupled, 0 otherwise).
        kappa_r : float
            Resonator linewidth κ/2π in GHz.
        T1_individual : ndarray, shape (N,)
            Measured T1 for each qubit in μs.
        """
        self.N = len(qubit_freqs)
        self.qubit_freqs = np.array(qubit_freqs)
        self.resonator_freqs = np.array(resonator_freqs)
        self.couplings_g = np.array(couplings_g)
        self.adjacency = np.array(adjacency)
        self.kappa_r = kappa_r
        self.T1_individual = np.array(T1_individual)

    def purcell_rates(self):
        """Standard Purcell decay rates per qubit.

        Γ_P,i = Σ_j (g_ij/Δ_ij)² κ_j
        """
        rates = np.zeros(self.N)
        edge_idx = 0
        for i in range(self.N):
            for j in range(self.N):
                if self.adjacency[i, j] > 0 and j > i:
                    g = self.couplings_g[i, j]
                    omega_r = self.resonator_freqs[edge_idx % len(self.resonator_freqs)]
                    delta_i = abs(self.qubit_freqs[i] - omega_r)
                    delta_j = abs(self.qubit_freqs[j] - omega_r)

                    if delta_i > 0:
                        rates[i] += (g / delta_i) ** 2 * self.kappa_r
                    if delta_j > 0:
                        rates[j] += (g / delta_j) ** 2 * self.kappa_r
                    edge_idx += 1

        return rates  # in GHz → convert to 1/μs later

    def standard_collective_rates(self):
        """Standard model: collective mode rates from individual rates.

        Standard assumption: each qubit decoheres independently.
        Collective mode k inherits rate: Γ_k = Σ_i |v_{ki}|² Γ_i
        where v_k is the k-th normal mode vector.

        This means: worst qubit limits ALL modes.
        """
        # Effective coupling Laplacian for normal modes
        L = self.effective_laplacian()
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        # Individual decay rates (1/T1)
        gamma_individual = 1.0 / self.T1_individual  # 1/μs

        # Each mode's rate is weighted sum of individual rates
        mode_rates = np.zeros(self.N)
        for k in range(self.N):
            v_k = eigenvectors[:, k]
            mode_rates[k] = np.sum(np.abs(v_k) ** 2 * gamma_individual)

        return eigenvalues, eigenvectors, mode_rates

    def effective_laplacian(self):
        """Effective coupling Laplacian for the qubit array.

        L_ij = -J_eff,ij for i ≠ j
        L_ii = Σ_j J_eff,ij

        where J_eff,ij = g_ij² / Δ_ij (dispersive coupling).
        """
        N = self.N
        J_eff = np.zeros((N, N))

        edge_idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                if self.adjacency[i, j] > 0:
                    g = self.couplings_g[i, j]
                    omega_r = self.resonator_freqs[edge_idx % len(self.resonator_freqs)]
                    delta = abs((self.qubit_freqs[i] + self.qubit_freqs[j]) / 2 - omega_r)
                    if delta > 0:
                        J_eff[i, j] = g ** 2 / delta
                        J_eff[j, i] = J_eff[i, j]
                    edge_idx += 1

        L = np.diag(np.sum(J_eff, axis=1)) - J_eff
        return L

    def rpst_collective_rates(self, kappa_s=None):
        """RPST prediction: collective mode rates from graph Laplacian
        PLUS intrinsic decoherence.

        The correct RPST prediction has two terms:
          Γ_k = κ_s · λ_k(L)  +  Σ_i |v_ki|² · Γ_intrinsic,i

        Term 1: coupling-induced decoherence (graph Laplacian eigenvalues)
        Term 2: intrinsic decoherence (same as standard model)

        The discriminating prediction is the ADDITIVE graph term:
          ΔΓ_k = κ_s · λ_k

        This makes high-eigenvalue modes decohere FASTER than the
        standard model predicts. The zero mode (λ_k=0) matches standard.
        """
        L = self.effective_laplacian()
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        # Intrinsic contribution (same as standard model)
        gamma_individual = 1.0 / self.T1_individual
        intrinsic_rates = np.zeros(self.N)
        for k in range(self.N):
            v_k = eigenvectors[:, k]
            intrinsic_rates[k] = np.sum(np.abs(v_k) ** 2 * gamma_individual)

        # Determine κ_s: the coupling-induced decoherence scale
        # Calibrate so that the coupling term is ~10-50% of intrinsic
        # (otherwise it would have already been detected)
        if kappa_s is None:
            mean_gamma = np.mean(gamma_individual)
            mean_lambda = np.mean(eigenvalues[eigenvalues > 1e-12])
            if mean_lambda > 0:
                # Set coupling term to 20% of intrinsic at mean eigenvalue
                kappa_s = 0.2 * mean_gamma / mean_lambda
            else:
                kappa_s = 1.0

        coupling_rates = kappa_s * np.maximum(eigenvalues, 0)
        total_rates = intrinsic_rates + coupling_rates

        return eigenvalues, eigenvectors, total_rates, intrinsic_rates, coupling_rates, kappa_s


# ===================================================================
# Published qubit array geometries
# ===================================================================

def ibm_5q_t_topology():
    """IBM Falcon 5-qubit T-shaped topology (like ibmq_manila).

    Q0 - Q1 - Q2 - Q3 - Q4
               |
              (T-junction at Q2 connects to Q3 and Q4 differently)

    Actually: linear chain Q0-Q1-Q2-Q3-Q4 with a T at Q2.
    Let's use the standard 5-qubit linear chain for simplicity.

    Published parameters (approximate, from IBM Quantum docs):
      ω_q ≈ 4.9-5.2 GHz
      g/2π ≈ 0.003-0.004 GHz (3-4 MHz effective ZZ)
      T1 ≈ 80-150 μs
      κ/2π ≈ 0.001 GHz
    """
    N = 5
    qubit_freqs = np.array([5.01, 4.95, 5.10, 4.98, 5.05])  # GHz

    # Linear chain adjacency
    adj = np.zeros((N, N))
    for i in range(N - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0

    # Coupling strengths
    g = np.zeros((N, N))
    for i in range(N - 1):
        g[i, i + 1] = 0.030  # 30 MHz
        g[i + 1, i] = 0.030

    res_freqs = np.array([6.5, 6.6, 6.4, 6.55])  # 4 bus resonators
    kappa = 0.001  # 1 MHz linewidth
    T1 = np.array([100.0, 120.0, 90.0, 110.0, 105.0])  # μs

    return QubitArray(qubit_freqs, res_freqs, g, adj, kappa, T1)


def google_sycamore_subset():
    """Google Sycamore-like 9-qubit grid subset.

    3×3 grid with nearest-neighbor coupling.

    Published parameters (approximate, from Arute et al. 2019):
      ω_q ≈ 5.0-5.5 GHz (tunable)
      g/2π ≈ 0.003 GHz (3 MHz ZZ)
      T1 ≈ 15-20 μs (early Sycamore)
      κ/2π ≈ 0.002 GHz
    """
    N = 9
    qubit_freqs = np.array([
        5.10, 5.20, 5.05,
        5.15, 5.25, 5.08,
        5.12, 5.18, 5.22,
    ])

    # 3×3 grid adjacency
    adj = np.zeros((N, N))
    for row in range(3):
        for col in range(3):
            idx = row * 3 + col
            if col < 2:
                adj[idx, idx + 1] = 1.0
                adj[idx + 1, idx] = 1.0
            if row < 2:
                adj[idx, idx + 3] = 1.0
                adj[idx + 3, idx] = 1.0

    g = adj * 0.003  # 3 MHz coupling
    res_freqs = np.array([6.5] * 12)  # 12 bus resonators
    kappa = 0.002
    T1 = np.array([18.0, 16.0, 19.0, 17.0, 15.0, 20.0, 18.5, 16.5, 17.5])

    return QubitArray(qubit_freqs, res_freqs, g, adj, kappa, T1)


def heavy_hex_7q():
    """IBM heavy-hex 7-qubit subset.

    Heavy-hex: each qubit connected to max 3 others.
    Central qubit has 3 connections, edge qubits have 1-2.

    This topology has important graph-theoretic properties:
    different Laplacian spectrum than grid or chain.

    Published parameters (approximate):
      ω_q ≈ 4.8-5.3 GHz
      g/2π ≈ 0.004 GHz
      T1 ≈ 100-300 μs (newer processors)
      κ/2π ≈ 0.0005 GHz
    """
    N = 7
    qubit_freqs = np.array([4.95, 5.10, 5.00, 5.20, 4.90, 5.05, 5.15])

    # Heavy-hex-like adjacency (central node + hexagonal edges)
    adj = np.zeros((N, N))
    # Center is Q3
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0)]
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    g = adj * 0.004
    res_freqs = np.array([6.5] * 7)
    kappa = 0.0005
    T1 = np.array([200.0, 180.0, 250.0, 150.0, 220.0, 190.0, 210.0])

    return QubitArray(qubit_freqs, res_freqs, g, adj, kappa, T1)


# ===================================================================
# Experiment 1: Single-qubit correspondence
# ===================================================================

def experiment_1_single_qubit_mapping():
    """Show that RPST Γ_dec matches Purcell at single-qubit level.

    This is the consistency check, not the discriminating test.
    Both models should agree for isolated qubits.
    """
    # Scan coupling g/Δ ratio
    g_over_delta = np.linspace(0.001, 0.1, 50)

    # Purcell: Γ = (g/Δ)² κ
    kappa = 0.001  # 1 MHz
    purcell_rates = (g_over_delta) ** 2 * kappa

    # RPST: Γ = κ_s |∇Φ|² where |∇Φ| ∝ g/Δ
    # Calibrate κ_s so they match at g/Δ = 0.01
    ref_idx = np.argmin(np.abs(g_over_delta - 0.01))
    kappa_s = purcell_rates[ref_idx] / (g_over_delta[ref_idx]) ** 2
    rpst_rates = kappa_s * (g_over_delta) ** 2

    # They should be identical (same functional form)
    max_relative_error = np.max(np.abs(rpst_rates - purcell_rates) /
                                 np.maximum(purcell_rates, 1e-20))

    return {
        "g_over_delta": g_over_delta,
        "purcell_rates": purcell_rates,
        "rpst_rates": rpst_rates,
        "kappa_s_calibrated": kappa_s,
        "max_relative_error": max_relative_error,
        "same_scaling": max_relative_error < 1e-10,
    }


# ===================================================================
# Experiment 2: Multi-qubit collective mode divergence (CORRECTED)
# ===================================================================

def experiment_2_collective_modes(array):
    """Compare RPST vs standard collective mode decoherence.

    CORRECTED VERSION: both models include intrinsic decoherence.

    Standard: Γ_k = Σ_i |v_ki|² Γ_i  (weighted average of individual rates)
    RPST:     Γ_k = Σ_i |v_ki|² Γ_i + κ_s · λ_k  (same + graph correction)

    The difference is the additive term κ_s · λ_k.
    - Zero mode (λ_0=0): RPST = standard (no correction)
    - High-λ modes: RPST predicts FASTER decoherence than standard
    - The spread of rates is WIDER in RPST than standard
    """
    # Standard prediction
    std_eigs, std_vecs, std_rates = array.standard_collective_rates()

    # RPST prediction (now includes intrinsic + coupling)
    rpst_eigs, rpst_vecs, rpst_total, rpst_intrinsic, rpst_coupling, kappa_s = \
        array.rpst_collective_rates()

    # The discriminating observable: rate difference per mode
    rate_excess = rpst_total - std_rates  # Should be ≥ 0

    # Maximum fractional excess: how much faster does the worst mode decohere?
    max_excess_frac = np.max(rate_excess / np.maximum(std_rates, 1e-20))

    # Rate spread comparison
    std_spread = np.max(std_rates) / max(np.min(std_rates), 1e-20)
    rpst_spread = np.max(rpst_total) / max(np.min(rpst_total), 1e-20)

    return {
        "N": array.N,
        "laplacian_eigenvalues": rpst_eigs,
        "standard_mode_rates": std_rates,
        "rpst_total_rates": rpst_total,
        "rpst_intrinsic_rates": rpst_intrinsic,
        "rpst_coupling_rates": rpst_coupling,
        "rate_excess": rate_excess,
        "max_excess_fraction": max_excess_frac,
        "kappa_s": kappa_s,
        "std_rate_spread": std_spread,
        "rpst_rate_spread": rpst_spread,
    }


# ===================================================================
# Experiment 3: Topology-dependent decoherence (CORRECTED)
# ===================================================================

def experiment_3_topology_dependence():
    """Compare decoherence across different qubit array topologies.

    CORRECTED: Both models include intrinsic decoherence.

    RPST predicts: topology changes the COUPLING correction.
    Standard predicts: same qubits → same rates (topology only enters
    through mode mixing, not through an additive correction).

    The discriminating observable: does the highest-mode decoherence rate
    change when you rewire the graph?
    """
    N = 6
    qubit_freqs = np.array([5.0, 5.05, 5.10, 4.95, 5.08, 5.02])
    g_strength = 0.003
    res_freqs = np.array([6.5] * 10)
    kappa = 0.001
    T1 = np.array([100.0] * N)  # ALL qubits identical T1

    topologies = {}

    adj_chain = np.zeros((N, N))
    for i in range(N - 1):
        adj_chain[i, i + 1] = 1.0
        adj_chain[i + 1, i] = 1.0
    topologies["chain"] = adj_chain

    adj_ring = adj_chain.copy()
    adj_ring[0, N - 1] = 1.0
    adj_ring[N - 1, 0] = 1.0
    topologies["ring"] = adj_ring

    adj_star = np.zeros((N, N))
    for i in range(1, N):
        adj_star[0, i] = 1.0
        adj_star[i, 0] = 1.0
    topologies["star"] = adj_star

    adj_complete = np.ones((N, N)) - np.eye(N)
    topologies["complete"] = adj_complete

    results = {}
    for name, adj in topologies.items():
        g_mat = adj * g_strength
        array = QubitArray(qubit_freqs, res_freqs, g_mat, adj, kappa, T1)

        std_eigs, std_vecs, std_rates = array.standard_collective_rates()
        rpst_eigs, rpst_vecs, rpst_total, rpst_intr, rpst_coup, ks = \
            array.rpst_collective_rates()

        results[name] = {
            "laplacian_eigenvalues": rpst_eigs,
            "standard_rates": std_rates,
            "rpst_total_rates": rpst_total,
            "coupling_correction": rpst_coup,
            "std_max_rate": float(np.max(std_rates)),
            "rpst_max_rate": float(np.max(rpst_total)),
            "max_coupling_correction": float(np.max(rpst_coup)),
            "correction_fraction": float(np.max(rpst_coup) / np.max(rpst_total)),
        }

    return results


# ===================================================================
# Experiment 4: Experimental protocol (CORRECTED)
# ===================================================================

def experiment_4_measurement_protocol():
    """What would you actually measure to test RPST?

    CORRECTED: The prediction is not a "protected mode" but an
    EXCESS decoherence on high-eigenvalue modes.

    Standard: all modes decay at rates within [min(Γ_i), max(Γ_i)]
    RPST:     high-λ modes decay FASTER than the standard envelope

    The observable: for each collective mode k, measure
      excess_k = Γ_measured,k - Σ_i |v_ki|² Γ_i
    RPST predicts: excess_k ∝ λ_k (graph Laplacian eigenvalue)
    Standard predicts: excess_k = 0
    """
    arrays = {
        "ibm_5q_chain": ibm_5q_t_topology(),
        "google_9q_grid": google_sycamore_subset(),
        "ibm_7q_hex": heavy_hex_7q(),
    }

    results = {}
    for name, array in arrays.items():
        std_eigs, std_vecs, std_rates = array.standard_collective_rates()
        rpst_eigs, rpst_vecs, rpst_total, rpst_intr, rpst_coup, ks = \
            array.rpst_collective_rates()

        T1_avg = np.mean(array.T1_individual)

        # Convert coupling correction to T1 equivalent
        # If coupling correction is X% of total rate, mode T1 is shorter by X%
        correction_pct = rpst_coup / np.maximum(rpst_total, 1e-20) * 100

        results[name] = {
            "N": array.N,
            "T1_avg": T1_avg,
            "laplacian_spectrum": rpst_eigs,
            "standard_rates": std_rates,
            "rpst_total_rates": rpst_total,
            "coupling_correction": rpst_coup,
            "correction_pct": correction_pct,
            "max_correction_pct": float(np.max(correction_pct)),
            "kappa_s": ks,
        }

    return results


# ===================================================================
# Experiment 5: Sensitivity — what precision is needed?
# ===================================================================

def experiment_5_sensitivity(N=5):
    """How precise must measurements be to see the RPST correction?

    Scan coupling strength and system size. Report the maximum
    fractional correction (RPST - standard) / standard as a percentage.
    """
    g_values = np.linspace(0.001, 0.010, 20)
    excess_vs_g = []

    for g_val in g_values:
        qubit_freqs = np.array([5.0 + 0.02 * i for i in range(N)])
        adj = np.zeros((N, N))
        for i in range(N - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        g_mat = adj * g_val
        res_freqs = np.array([6.5] * (N - 1))
        T1 = np.array([100.0] * N)
        array = QubitArray(qubit_freqs, res_freqs, g_mat, adj, 0.001, T1)

        result = experiment_2_collective_modes(array)
        excess_vs_g.append(result["max_excess_fraction"])

    N_values = [3, 5, 7, 9, 11, 15]
    excess_vs_N = []

    for N_val in N_values:
        qubit_freqs = np.array([5.0 + 0.02 * i for i in range(N_val)])
        adj = np.zeros((N_val, N_val))
        for i in range(N_val - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        g_mat = adj * 0.003
        res_freqs = np.array([6.5] * (N_val - 1))
        T1 = np.array([100.0] * N_val)
        array = QubitArray(qubit_freqs, res_freqs, g_mat, adj, 0.001, T1)

        result = experiment_2_collective_modes(array)
        excess_vs_N.append(result["max_excess_fraction"])

    return {
        "g_values": g_values,
        "excess_vs_g": np.array(excess_vs_g),
        "N_values": N_values,
        "excess_vs_N": np.array(excess_vs_N),
    }


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("RPST DECOHERENCE MAPPING TO QUBIT EXPERIMENTS")
    print("(CORRECTED: honest comparison with intrinsic decoherence)")
    print("=" * 70)

    # --- Experiment 1: Single-qubit consistency ---
    print("\n--- Experiment 1: Single-Qubit Consistency Check ---")
    e1 = experiment_1_single_qubit_mapping()
    print(f"  RPST and Purcell have same scaling? {e1['same_scaling']}")
    print(f"  Calibrated κ_s = {e1['kappa_s_calibrated']:.6f} GHz")
    print(f"  → At single-qubit level, RPST = Purcell. No discrimination.")

    # --- Experiment 2: Multi-qubit collective modes ---
    print("\n--- Experiment 2: Multi-Qubit Collective Mode Rates ---")
    print("  RPST adds κ_s·λ_k coupling correction on top of intrinsic rates")

    for label, array_fn in [("IBM 5Q chain", ibm_5q_t_topology),
                              ("Google 9Q grid", google_sycamore_subset),
                              ("IBM 7Q hex", heavy_hex_7q)]:
        array = array_fn()
        e2 = experiment_2_collective_modes(array)

        print(f"\n  {label} (N={e2['N']}):")
        print(f"    λ(L):       {np.array2string(e2['laplacian_eigenvalues'], precision=6, separator=', ')}")
        print(f"    Γ_standard: {np.array2string(e2['standard_mode_rates'], precision=6, separator=', ')}")
        print(f"    Γ_rpst:     {np.array2string(e2['rpst_total_rates'], precision=6, separator=', ')}")
        print(f"    Γ_coupling: {np.array2string(e2['rpst_coupling_rates'], precision=6, separator=', ')}")
        print(f"    Max excess: {e2['max_excess_fraction']*100:.1f}% faster than standard")
        print(f"    Standard spread: {e2['std_rate_spread']:.2f}x   RPST spread: {e2['rpst_rate_spread']:.2f}x")

    # --- Experiment 3: Topology dependence ---
    print("\n--- Experiment 3: Topology-Dependent Correction ---")
    print("  Same qubits (identical T1), different wiring")
    e3 = experiment_3_topology_dependence()

    print(f"\n  {'Topology':12s} | {'Std max':>10s} {'RPST max':>10s} {'Coupling':>10s} | {'Correction':>10s}")
    print(f"  {'-'*12}-+-{'-'*10}-{'-'*10}-{'-'*10}-+-{'-'*10}")
    for name, data in e3.items():
        print(f"  {name:12s} | {data['std_max_rate']:10.6f} {data['rpst_max_rate']:10.6f} "
              f"{data['max_coupling_correction']:10.2e} | "
              f"{data['correction_fraction']*100:8.1f}%")

    # Key test: standard max rates should be identical across topologies
    # (because T1 is identical and modes are just rotations of same rates)
    std_rates_all = [e3[t]["std_max_rate"] for t in e3]
    rpst_rates_all = [e3[t]["rpst_max_rate"] for t in e3]
    std_spread_topo = max(std_rates_all) / max(min(std_rates_all), 1e-20)
    rpst_spread_topo = max(rpst_rates_all) / max(min(rpst_rates_all), 1e-20)

    print(f"\n  Standard max-rate variation across topologies: {std_spread_topo:.4f}x")
    print(f"  RPST max-rate variation across topologies:     {rpst_spread_topo:.4f}x")

    # --- Experiment 4: Measurement protocol ---
    print("\n--- Experiment 4: What to Measure ---")
    e4 = experiment_4_measurement_protocol()

    for name, data in e4.items():
        print(f"\n  {name} (N={data['N']}, <T1>={data['T1_avg']:.0f} μs):")
        print(f"    Mode-by-mode coupling correction (% of total rate):")
        for k in range(data["N"]):
            bar = "#" * int(data["correction_pct"][k] / 2)
            print(f"      mode {k}: λ={data['laplacian_spectrum'][k]:.2e}  "
                  f"correction={data['correction_pct'][k]:5.1f}%  {bar}")
        print(f"    Maximum correction: {data['max_correction_pct']:.1f}%")
        print(f"    κ_s = {data['kappa_s']:.2e}")

    # --- Experiment 5: Sensitivity ---
    print("\n--- Experiment 5: Required Measurement Precision ---")
    e5 = experiment_5_sensitivity()

    print(f"  Max RPST excess vs coupling g/2π (5-qubit chain):")
    for i in range(0, len(e5["g_values"]), 5):
        print(f"    g={e5['g_values'][i]*1000:.1f} MHz: excess = {e5['excess_vs_g'][i]*100:.1f}%")

    print(f"\n  Max RPST excess vs system size N (g=3 MHz chain):")
    for i, N_val in enumerate(e5["N_values"]):
        print(f"    N={N_val:2d}: excess = {e5['excess_vs_N'][i]*100:.1f}%")

    # --- Honest Summary ---
    print("\n" + "=" * 70)
    print("HONEST SUMMARY")
    print("=" * 70)

    max_corrections = [e4[n]["max_correction_pct"] for n in e4]
    avg_correction = np.mean(max_corrections)

    print(f"""
  WHAT RPST ACTUALLY PREDICTS (corrected):
    RPST does NOT predict a "protected" mode with infinite lifetime.
    The zero-mode (λ_k=0) has the same intrinsic decoherence as standard.

    RPST predicts an ADDITIVE coupling correction: ΔΓ_k = κ_s · λ_k
    This makes high-eigenvalue collective modes decohere FASTER
    than the weighted average of individual qubit rates.

  PREDICTED EFFECT SIZE:
    Maximum coupling correction: {avg_correction:.1f}% of total rate
    (calibrated at κ_s such that coupling = 20% of intrinsic at mean λ)

  THE HONEST PROBLEM:
    The coupling correction κ_s is a FREE PARAMETER.
    We calibrated it to give a 20% effect. We could set it to 0.01%
    or 200%. Without an independent way to determine κ_s, the model
    is not yet falsifiable — it can fit any data by tuning κ_s.

  TO MAKE THIS FALSIFIABLE:
    1. Measure mode-resolved decoherence on TWO different topologies
       with the same qubits
    2. Fit κ_s from topology #1
    3. PREDICT topology #2's mode rates (no free parameters)
    4. If the prediction matches, RPST adds explanatory power
    5. If not, the coupling correction is wrong

  WHAT THIS EXPERIMENT NEEDS:
    - Multi-qubit processor with reconfigurable coupling topology
    - Mode-resolved decoherence measurement (not just single-qubit T1)
    - Two different wiring configurations of the same qubits
    - Precision: need to resolve ~{avg_correction:.0f}% rate differences

  WHERE RPST COULD WIN:
    If mode-resolved decoherence rates correlate with Laplacian
    eigenvalues (after subtracting intrinsic rates), that is evidence
    for boundary-phase-gradient-driven decoherence.
    This is a real, measurable, non-trivial prediction.
    """)


if __name__ == "__main__":
    main()
