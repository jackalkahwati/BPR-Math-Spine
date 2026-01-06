"""
HKLL Boundary-to-Bulk Reconstruction (Eq 2 enhancement; scaffolding)

This is a numerical mode-expansion reconstruction:
  Ψ_bulk(X) = ∫_{S²} dΩ' K(X|Ω') φ_boundary(Ω', t)

The goal here is a runnable, testable implementation consistent with the repo's
"math spine" ethos. It is not a full AdS/CFT production HKLL implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.special import sph_harm


@dataclass(frozen=True)
class BulkPoint:
    t: float
    r: float
    theta: float
    phi: float


@dataclass(frozen=True)
class BoundaryPoint:
    t: float
    theta: float
    phi: float


class HKLLReconstruction:
    """
    Simple spherical-harmonic smearing-kernel reconstruction on a ball of radius R.
    """

    def __init__(self, boundary_radius: float, mass_parameter: float = 0.0, l_max: int = 20):
        self.R = float(boundary_radius)
        if self.R <= 0:
            raise ValueError("boundary_radius must be > 0")
        self.m = float(mass_parameter)
        self.l_max = int(l_max)
        # For bookkeeping (not used heavily in this scaffold)
        self.Delta = 1.5 + float(np.sqrt(2.25 + (self.m * self.R) ** 2))

    def smearing_kernel(self, bulk_point: BulkPoint, boundary_point: BoundaryPoint) -> complex:
        r = float(bulk_point.r)
        thB, phB = float(bulk_point.theta), float(bulk_point.phi)
        th, ph = float(boundary_point.theta), float(boundary_point.phi)

        kernel = 0.0 + 0.0j
        for l in range(self.l_max + 1):
            if r < self.R:
                K_l = (r / self.R) ** l
            else:
                K_l = (self.R / r) ** (l + 1)
            for m in range(-l, l + 1):
                Y_bulk = sph_harm(m, l, phB, thB)
                Y_bdy = sph_harm(m, l, ph, th)
                kernel += K_l * Y_bulk * np.conj(Y_bdy)
        return kernel

    def reconstruct_bulk_field(
        self,
        bulk_point: BulkPoint,
        boundary_field: Callable[[float, float, float], float],
        n_theta: int = 24,
        n_phi: int = 48,
    ) -> complex:
        """
        Quadrature on S² using Gauss-Legendre in cos(theta) and uniform phi.
        """

        n_theta = int(n_theta)
        n_phi = int(n_phi)
        if n_theta < 4 or n_phi < 8:
            raise ValueError("n_theta>=4 and n_phi>=8 recommended")

        # leggauss returns points on [-1,1] for u=cos(theta), and weights for du
        u, w = np.polynomial.legendre.leggauss(n_theta)
        thetas = np.arccos(u)  # theta in [0, pi]

        phis = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
        dphi = 2 * np.pi / n_phi

        out = 0.0 + 0.0j
        for i, theta_bdy in enumerate(thetas):
            for phi_bdy in phis:
                bdy_pt = BoundaryPoint(t=bulk_point.t, theta=float(theta_bdy), phi=float(phi_bdy))
                K = self.smearing_kernel(bulk_point, bdy_pt)
                phi_val = float(boundary_field(float(theta_bdy), float(phi_bdy), float(bulk_point.t)))
                # measure: dΩ = sin(theta) dtheta dphi; with u=cos(theta), sin(theta)dtheta = -du
                out += K * phi_val * float(w[i]) * dphi
        return out

    def reconstruct_metric_perturbation(
        self,
        bulk_point: BulkPoint,
        boundary_field: Callable[[float, float, float], float],
        coupling_lambda: float,
        n_theta: int = 16,
        n_phi: int = 32,
        eps: float = 1e-3,
    ) -> np.ndarray:
        """
        Minimal Eq (2) enhancement:
          Δg_{μν} = λ (∂_μ Ψ)(∂_ν Ψ)

        Here we approximate gradients with finite differences in (t,r,theta,phi).
        """

        lam = float(coupling_lambda)
        if eps <= 0:
            raise ValueError("eps must be > 0")

        def Psi(pt: BulkPoint) -> complex:
            return self.reconstruct_bulk_field(pt, boundary_field, n_theta=n_theta, n_phi=n_phi)

        # gradients: [t, r, theta, phi] in these coordinates
        grads = np.zeros(4, dtype=complex)

        # t derivative
        ptp = BulkPoint(bulk_point.t + eps, bulk_point.r, bulk_point.theta, bulk_point.phi)
        ptm = BulkPoint(bulk_point.t - eps, bulk_point.r, bulk_point.theta, bulk_point.phi)
        grads[0] = (Psi(ptp) - Psi(ptm)) / (2 * eps)

        # r derivative
        r_p = bulk_point.r + eps
        r_m = max(1e-6, bulk_point.r - eps)
        prp = BulkPoint(bulk_point.t, r_p, bulk_point.theta, bulk_point.phi)
        prm = BulkPoint(bulk_point.t, r_m, bulk_point.theta, bulk_point.phi)
        grads[1] = (Psi(prp) - Psi(prm)) / (2 * eps)

        # theta derivative
        th_p = min(np.pi, bulk_point.theta + eps)
        th_m = max(0.0, bulk_point.theta - eps)
        ptp = BulkPoint(bulk_point.t, bulk_point.r, th_p, bulk_point.phi)
        ptm = BulkPoint(bulk_point.t, bulk_point.r, th_m, bulk_point.phi)
        grads[2] = (Psi(ptp) - Psi(ptm)) / (2 * eps)

        # phi derivative (wrap)
        ph_p = bulk_point.phi + eps
        ph_m = bulk_point.phi - eps
        ptp = BulkPoint(bulk_point.t, bulk_point.r, bulk_point.theta, ph_p)
        ptm = BulkPoint(bulk_point.t, bulk_point.r, bulk_point.theta, ph_m)
        grads[3] = (Psi(ptp) - Psi(ptm)) / (2 * eps)

        dg = np.zeros((4, 4), dtype=float)
        for mu in range(4):
            for nu in range(4):
                dg[mu, nu] = float(np.real(lam * grads[mu] * grads[nu]))
        return dg


class BoundaryToBulkProjection:
    """
    Projection tensor P^{ab}_{μν} scaffold for mapping boundary gradients to bulk components.
    """

    def __init__(self, boundary_radius: float, localization_width: float = 0.1):
        self.R = float(boundary_radius)
        self.sigma = float(localization_width)
        if self.R <= 0:
            raise ValueError("boundary_radius must be > 0")
        if self.sigma <= 0:
            raise ValueError("localization_width must be > 0")

    def embedding_jacobian(self, theta: float, phi: float) -> np.ndarray:
        R = self.R
        theta = float(theta)
        phi = float(phi)
        e = np.zeros((3, 2), dtype=float)
        # θ derivative
        e[0, 0] = R * np.cos(theta) * np.cos(phi)
        e[1, 0] = R * np.cos(theta) * np.sin(phi)
        e[2, 0] = -R * np.sin(theta)
        # φ derivative
        e[0, 1] = -R * np.sin(theta) * np.sin(phi)
        e[1, 1] = R * np.sin(theta) * np.cos(phi)
        e[2, 1] = 0.0
        return e

    def induced_metric(self, theta: float, phi: float) -> np.ndarray:
        e = self.embedding_jacobian(theta, phi)
        return e.T @ e

    def projection_tensor(self, theta: float, phi: float, r: float) -> np.ndarray:
        e = self.embedding_jacobian(theta, phi)
        r = float(r)
        loc = float(np.exp(-((r - self.R) ** 2) / (self.sigma**2)))
        P = np.zeros((4, 4, 2, 2), dtype=float)
        for mu in range(1, 4):
            for nu in range(1, 4):
                i, j = mu - 1, nu - 1
                for a in range(2):
                    for b in range(2):
                        P[mu, nu, a, b] = e[i, a] * e[j, b] * loc
        return P

    def contract_with_field_gradients(self, P: np.ndarray, grad_phi: np.ndarray) -> np.ndarray:
        grad_phi = np.asarray(grad_phi, dtype=float).reshape(2)
        return np.einsum("mnab,a,b->mn", P, grad_phi, grad_phi)


