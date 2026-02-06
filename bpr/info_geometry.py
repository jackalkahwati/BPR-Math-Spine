"""
Theory VI: Substrate Information Geometry
==========================================

Constructs the Fisher information metric on the space of coarse-grained
boundary configurations, derives topological Cramér–Rao bounds, and
interprets geodesics as optimal measurement protocols.

Key objects
-----------
* ``FisherMetric``           – g_ij on substrate state space
* ``TopologicalCramerRao``   – enhanced bounds for |W| > 0
* ``ThermodynamicLength``    – minimum-dissipation path length
* ``CouplingKernelGeometry`` – K_r as parallel transport

References: Al-Kahwati (2026), *Ten Adjacent Theories*, §8
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable


# ---------------------------------------------------------------------------
# §8.2  Fisher information metric
# ---------------------------------------------------------------------------

@dataclass
class FisherMetric:
    """Fisher information metric on coarse-grained boundary configurations.

    g_ij(θ) = ⟨(∂_i log p)(∂_j log p)⟩

    In the Gaussian approximation (near equilibrium):
        g_ij = ⟨∂_i φ  ∂_j φ⟩ / σ_φ²

    Parameters
    ----------
    cov_matrix : ndarray, shape (n_params, n_params)
        Covariance matrix of boundary field derivatives ⟨∂_i φ ∂_j φ⟩.
    sigma_phi : float
        Field variance σ_φ.
    """
    cov_matrix: Optional[np.ndarray] = None
    sigma_phi: float = 1.0

    @property
    def g(self) -> np.ndarray:
        """Fisher metric tensor g_ij."""
        if self.cov_matrix is None:
            raise ValueError("Covariance matrix not set.")
        return self.cov_matrix / self.sigma_phi ** 2

    def distance(self, dtheta: np.ndarray) -> float:
        """Fisher distance δs = √(g_ij δθ^i δθ^j)."""
        dtheta = np.asarray(dtheta)
        return float(np.sqrt(dtheta @ self.g @ dtheta))

    def geodesic(self, theta_A: np.ndarray, theta_B: np.ndarray,
                 n_steps: int = 100) -> np.ndarray:
        """Straight-line geodesic in the Gaussian approximation.

        For a flat Fisher manifold (Gaussian) the geodesic is a
        straight line in parameter space.
        """
        theta_A = np.asarray(theta_A)
        theta_B = np.asarray(theta_B)
        t = np.linspace(0, 1, n_steps)[:, None]
        return theta_A[None, :] + t * (theta_B - theta_A)[None, :]

    @staticmethod
    def from_samples(phi_samples: np.ndarray,
                     sigma_phi: float = 1.0) -> "FisherMetric":
        """Estimate Fisher metric from Monte-Carlo samples of ∂φ/∂θ.

        Parameters
        ----------
        phi_samples : ndarray, shape (n_samples, n_params)
            Samples of field derivatives.
        """
        cov = np.cov(phi_samples, rowvar=False)
        return FisherMetric(cov_matrix=cov, sigma_phi=sigma_phi)


# ---------------------------------------------------------------------------
# §8.3  Topological Cramér–Rao bound
# ---------------------------------------------------------------------------

@dataclass
class TopologicalCramerRao:
    """Topological Cramér–Rao bound (Theorem, §8.3).

    Var(θ̂) ≥ 1 / (N F(θ)) ≥ 1 / (N F_max |W|²)

    Systems with |W| > 0 have enhanced Fisher information.
    """
    N: int = 1                  # number of measurements
    F_max: float = 1.0          # max Fisher info per winding unit
    W: float = 1.0              # winding number

    @property
    def fisher_information(self) -> float:
        """F(θ) ≤ F_max |W|²."""
        return self.F_max * self.W ** 2

    @property
    def variance_lower_bound(self) -> float:
        """Minimum variance of any unbiased estimator."""
        fi = self.fisher_information
        if fi <= 0:
            return np.inf
        return 1.0 / (self.N * fi)

    def quantum_advantage_factor(self) -> float:
        """How much better W ≠ 0 systems are than W = 0.

        Factor = |W|² (i.e. Heisenberg-like scaling).
        """
        return self.W ** 2 if abs(self.W) > 0 else 0.0


def topological_cramer_rao(N: int, F_max: float, W: float) -> float:
    """Return the topological Cramér–Rao lower bound on variance."""
    return TopologicalCramerRao(N=N, F_max=F_max, W=W).variance_lower_bound


# ---------------------------------------------------------------------------
# §8.4  Thermodynamic length (minimum dissipation)
# ---------------------------------------------------------------------------

def thermodynamic_length(g_metric: np.ndarray,
                         path: np.ndarray) -> float:
    """Thermodynamic length along a path in parameter space.

    L_thermo = ∫ √(g_ij dθ^i dθ^j)

    Minimum entropy production ∝ L_thermo².

    Parameters
    ----------
    g_metric : ndarray, shape (n_params, n_params)
        Fisher metric tensor (constant along path in Gaussian approx).
    path : ndarray, shape (n_steps, n_params)
        Discretised path θ(t).
    """
    length = 0.0
    for i in range(1, len(path)):
        dtheta = path[i] - path[i - 1]
        ds = np.sqrt(float(dtheta @ g_metric @ dtheta))
        length += ds
    return length


# ---------------------------------------------------------------------------
# §8.5  Coupling kernel as parallel transport
# ---------------------------------------------------------------------------

def coupling_kernel_parallel_transport(phi_fine: np.ndarray,
                                        connection: np.ndarray) -> np.ndarray:
    """Apply the coupling kernel K_r as parallel transport.

    φ_coarse(θ) = K_r[φ_fine(θ)] = P-exp(∫ Γ_scale dr) φ_fine

    Simplified to matrix exponential of the connection for demonstration.

    Parameters
    ----------
    phi_fine : ndarray, shape (n,)
        Fine-scale field.
    connection : ndarray, shape (n, n)
        Scale connection matrix Γ_scale.
    """
    from scipy.linalg import expm
    return expm(connection) @ phi_fine
