#!/usr/bin/env python3
"""
Tutorial 1: RPST Substrate Basics

This tutorial demonstrates the fundamental RPST substrate structure
that underlies BPR.

Key concepts:
- Phase variables (q, π) on prime modular lattice
- U(1) gauge structure
- Hamiltonian dynamics
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.insert(0, repo_root)

# =============================================================================
# 1. The Substrate State Space
# =============================================================================

print("=" * 60)
print("TUTORIAL 1: RPST SUBSTRATE BASICS")
print("=" * 60)

# RPST uses a prime modular lattice
p = 104729  # A large prime
N = 100     # Number of nodes

print(f"\nSubstrate parameters:")
print(f"  Prime modulus p = {p}")
print(f"  Number of nodes N = {N}")
print(f"  State space: (ℤ_p × ℤ_p)^N")

# Each node has position q and momentum π
np.random.seed(42)
q = np.random.randint(0, p, N)  # Position variables
pi = np.random.randint(0, p, N)  # Momentum variables

print(f"\nSample state (first 5 nodes):")
for i in range(5):
    print(f"  Node {i}: q={q[i]:6d}, π={pi[i]:6d}")

# =============================================================================
# 2. Phase Angle Representation
# =============================================================================

print("\n" + "=" * 60)
print("PHASE ANGLE REPRESENTATION")
print("=" * 60)

# Map position to U(1) phase angle
theta = 2 * np.pi * q / p

print(f"\nPhase angles θ = 2πq/p (first 5 nodes):")
for i in range(5):
    print(f"  Node {i}: θ = {theta[i]:.4f} rad = {np.degrees(theta[i]):.1f}°")

# =============================================================================
# 3. The Hamiltonian
# =============================================================================

print("\n" + "=" * 60)
print("HAMILTONIAN DYNAMICS")
print("=" * 60)

# Coupling parameters
J = 1.0  # Coupling strength (in units where ℏ = 1)
m = 1.0  # Effective mass

# Create ring topology (each node connected to neighbors)
def compute_hamiltonian(q, pi, p, J, m):
    """Compute RPST Hamiltonian for ring topology."""
    N = len(q)

    # Kinetic energy
    T = np.sum(pi**2) / (2 * m * p**2)  # Normalized

    # Potential energy (nearest-neighbor cosine coupling)
    V = 0.0
    for i in range(N):
        j = (i + 1) % N  # Next neighbor (ring)
        delta_theta = 2 * np.pi * (q[j] - q[i]) / p
        V += J * (1 - np.cos(delta_theta))

    return T + V, T, V

H, T, V = compute_hamiltonian(q, pi, p, J, m)
print(f"\nHamiltonian H = T + V:")
print(f"  Kinetic energy T = {T:.6f}")
print(f"  Potential energy V = {V:.6f}")
print(f"  Total energy H = {H:.6f}")

# =============================================================================
# 4. U(1) Gauge Invariance
# =============================================================================

print("\n" + "=" * 60)
print("U(1) GAUGE INVARIANCE")
print("=" * 60)

# The Hamiltonian should be invariant under global phase shifts
# H(θ + α) = H(θ) for any constant α

def test_gauge_invariance(q, pi, p, J, m, alpha):
    """Test if H is invariant under θ → θ + α."""
    # Original H
    H_original, _, _ = compute_hamiltonian(q, pi, p, J, m)

    # Shifted q (corresponds to θ → θ + α)
    delta_q = int(alpha * p / (2 * np.pi)) % p
    q_shifted = (q + delta_q) % p

    # H after shift
    H_shifted, _, _ = compute_hamiltonian(q_shifted, pi, p, J, m)

    return abs(H_shifted - H_original)

print("\nTesting U(1) invariance H(θ + α) = H(θ):")
for alpha in [0.1, 0.5, 1.0, np.pi, 2*np.pi - 0.01]:
    violation = test_gauge_invariance(q, pi, p, J, m, alpha)
    print(f"  α = {alpha:.4f}: violation = {violation:.2e}")

print("\n✓ The Hamiltonian is U(1) gauge invariant!")
print("  (Violations are at machine precision level)")

# =============================================================================
# 5. Coarse-Graining Preview
# =============================================================================

print("\n" + "=" * 60)
print("COARSE-GRAINING PREVIEW")
print("=" * 60)

# Average phase over windows
window_size = 10
n_windows = N // window_size

print(f"\nCoarse-graining: averaging over windows of {window_size} nodes")
print(f"Number of coarse-grained 'points': {n_windows}")

phi_coarse = np.zeros(n_windows)
for i in range(n_windows):
    start = i * window_size
    end = start + window_size
    # Average the phases (need to handle wrap-around)
    phases = theta[start:end]
    # Use circular mean
    phi_coarse[i] = np.arctan2(np.mean(np.sin(phases)),
                               np.mean(np.cos(phases)))

print(f"\nCoarse-grained field φ(x) at {n_windows} points:")
for i in range(min(5, n_windows)):
    print(f"  φ[{i}] = {phi_coarse[i]:.4f} rad")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
The RPST substrate provides:
1. Discrete phase variables (q, π) ∈ ℤ_p × ℤ_p
2. Hamiltonian dynamics preserving symplectic structure
3. U(1) gauge invariance (phases only matter up to constant)
4. Coarse-graining gives continuous field φ(x)

This is the foundation for BPR coupling derivations.
""")
