"""
Functional Architecture of Reality
==========================================================

Identifies nine minimal mathematical operators required for internal
coherence of any multi-layer ontological system containing observers,
persistent structure, and cross-scale coupling.  Each operator is
grounded in the BPR–RPST–Cache formalism.

Key objects
-----------
* ``ResonanceKernel``            -- Constructive resonance K_r           (Eq 2)
* ``RealizedStateProjection``    -- ψ(t) = Π_r(Ψ)                       (Eq 3)
* ``IdentityWinding``            -- W[φ_mind] topological identity       (Def 3.2)
* ``PermissionField``            -- P(x,t) gated coupling               (Def 4.1)
* ``SemanticEncoding``           -- Φ: X → M optimal encoding           (Def 5.1)
* ``SalienceField``              -- A(x,t) softmax attention             (Def 6.1)
* ``TrajectoryEvaluation``       -- J(γ) = ∫ U(γ(t)) dt                 (Def 7.1)
* ``CoherenceStack``             -- 9-operator minimal stack             (Thm 8.1)
* ``InteroperatorConsistency``   -- Three consistency conditions         (Sec 9)

References: Al-Kahwati (2026), *Toward a Functional Architecture of Reality*,
StarDrive Research Group.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union, Callable


# ===================================================================
# Resonance Kernel (Eq 2)
# ===================================================================

@dataclass
class ResonanceKernel:
    r"""Constructive resonance kernel K_r (Eq 2).

    K_r(ψ_i, ψ_j) = exp(-α d_G(ψ_i, ψ_j))
                    · exp(β cos Δφ_{ij})
                    · exp(-η d_p(p_i, p_j))

    where:
    - d_G is graph distance on the substrate lattice
    - Δφ_{ij} is phase misalignment
    - d_p is compatibility distance in prime-index space

    Parameters
    ----------
    alpha : float
        Graph-distance decay rate.
    beta : float
        Phase-alignment coupling strength.
    eta : float
        Prime-index decay rate.
    """
    alpha: float = 1.0
    beta: float = 1.0
    eta: float = 1.0

    def evaluate(self, d_G: float, delta_phi: float,
                 d_p: float) -> float:
        r"""Evaluate resonance kernel K_r (Eq 2).

        Parameters
        ----------
        d_G : float
            Graph distance between states.
        delta_phi : float
            Phase misalignment Δφ_{ij}.
        d_p : float
            Prime-index compatibility distance.

        Returns
        -------
        float
            Kernel value K_r ≥ 0.
        """
        spatial = np.exp(-self.alpha * d_G)
        phase = np.exp(self.beta * np.cos(delta_phi))
        prime = np.exp(-self.eta * d_p)
        return float(spatial * phase * prime)

    def evaluate_batch(self, d_G: np.ndarray, delta_phi: np.ndarray,
                       d_p: np.ndarray) -> np.ndarray:
        r"""Evaluate resonance kernel on arrays.

        Parameters
        ----------
        d_G, delta_phi, d_p : ndarray
            Batch inputs.

        Returns
        -------
        ndarray
            Kernel values.
        """
        spatial = np.exp(-self.alpha * np.asarray(d_G, dtype=float))
        phase = np.exp(self.beta * np.cos(np.asarray(delta_phi, dtype=float)))
        prime = np.exp(-self.eta * np.asarray(d_p, dtype=float))
        return spatial * phase * prime

    def max_kernel_value(self) -> float:
        r"""Maximum kernel value (at d_G = 0, Δφ = 0, d_p = 0).

        K_max = exp(β)   (since cos(0) = 1, e^{-0} = 1)

        Returns
        -------
        float
            Maximum possible kernel value.
        """
        return float(np.exp(self.beta))

    def resonance_read(self, psi: np.ndarray, cache_states: np.ndarray,
                       d_G_values: np.ndarray, delta_phi_values: np.ndarray,
                       d_p_values: np.ndarray) -> np.ndarray:
        r"""Resonance-weighted read access C(ψ) (Eq 1).

        C(ψ) = Σ_Ψ K_r(ψ, Ψ) Ψ   (discrete form)

        Parameters
        ----------
        psi : ndarray
            Query state.
        cache_states : ndarray, shape (M, D)
            Cached states Ψ.
        d_G_values : ndarray, shape (M,)
            Graph distances to each cache state.
        delta_phi_values : ndarray, shape (M,)
            Phase misalignments.
        d_p_values : ndarray, shape (M,)
            Prime-index distances.

        Returns
        -------
        ndarray, shape (D,)
            Weighted superposition.
        """
        weights = self.evaluate_batch(d_G_values, delta_phi_values, d_p_values)
        # Normalize weights
        total = np.sum(weights)
        if total < 1e-15:
            return np.zeros(cache_states.shape[1])
        return np.sum(weights[:, None] * cache_states / total, axis=0)


# ===================================================================
# Realized State Projection (Eq 3)
# ===================================================================

@dataclass
class RealizedStateProjection:
    r"""Projection from Cache to realized state ψ(t) = Π_r(Ψ) (Eq 3).

    ψ(t) = ∫_Σ dΩ G(r|Ω) e^{iφ(Ω,t)}

    where G(r|Ω) is a boundary Green's function and Σ is the
    holographic boundary.

    This is the Born-rule analogue of the BPR framework.

    Parameters
    ----------
    n_modes : int
        Number of boundary modes to include.
    """
    n_modes: int = 10

    def project(self, amplitudes: np.ndarray,
                phases: np.ndarray) -> complex:
        r"""Compute realized state projection (Eq 3).

        ψ = Σ_n G_n exp(i φ_n)

        Parameters
        ----------
        amplitudes : ndarray, shape (n_modes,)
            Green's function amplitudes G_n = G(r|Ω_n).
        phases : ndarray, shape (n_modes,)
            Boundary phase values φ(Ω_n, t).

        Returns
        -------
        complex
            Realized state ψ(t).
        """
        return complex(np.sum(amplitudes * np.exp(1j * phases)))

    def project_field(self, amplitudes: np.ndarray,
                      phases: np.ndarray,
                      basis: np.ndarray) -> np.ndarray:
        r"""Project onto a spatial basis.

        ψ(r) = Σ_n G_n exp(i φ_n) e_n(r)

        Parameters
        ----------
        amplitudes : ndarray, shape (M,)
            Green's function amplitudes.
        phases : ndarray, shape (M,)
            Phase values.
        basis : ndarray, shape (M, D)
            Basis functions evaluated at D spatial points.

        Returns
        -------
        ndarray, shape (D,)
            Realized field ψ(r) (complex).
        """
        coeffs = amplitudes * np.exp(1j * phases)
        return np.sum(coeffs[:, None] * basis, axis=0)

    def projection_amplitude(self, amplitudes: np.ndarray,
                              phases: np.ndarray) -> float:
        r"""Projection amplitude |ψ|².

        Parameters
        ----------
        amplitudes : ndarray
            Green's function amplitudes.
        phases : ndarray
            Phase values.

        Returns
        -------
        float
            |ψ(t)|².
        """
        psi = self.project(amplitudes, phases)
        return float(abs(psi) ** 2)


# ===================================================================
# Identity as Topological Winding (Sec 3, Def 3.1-3.2, Thm 3.1)
# ===================================================================

@dataclass
class IdentityWinding:
    r"""Identity as topological winding number (Sec 3).

    W[φ_mind] = (1/2π) ∮_{∂O} ∇φ_mind · dℓ ∈ Z    (Eq 4)

    Topological protection (Thm 3.1): a configuration with W ≠ 0
    cannot be continuously deformed to W = 0 without passing through
    a singular (infinite-energy) configuration.

    Cache timescale: τ_m = τ_0 |W|^α

    Parameters
    ----------
    tau_0 : float
        Base memory timescale.
    alpha : float
        Winding protection exponent (α ≥ 1).
    """
    tau_0: float = 1.0
    alpha: float = 1.0

    @staticmethod
    def winding_number(phi: np.ndarray) -> int:
        r"""Compute consciousness winding number W[φ_mind] (Eq 4).

        W = (1/2π) ∮ ∇φ · dℓ

        Parameters
        ----------
        phi : ndarray
            Phase values around a closed loop on ∂O.

        Returns
        -------
        int
            Winding number W ∈ Z.
        """
        dphi = np.diff(phi)
        # Wrap to [-π, π] to handle branch cuts
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        return int(np.round(np.sum(dphi) / (2.0 * np.pi)))

    def memory_timescale(self, W: int) -> float:
        r"""Cache memory timescale τ_m = τ_0 |W|^α (Property 3).

        Parameters
        ----------
        W : int
            Winding number.

        Returns
        -------
        float
            Memory timescale.
        """
        if W == 0:
            return self.tau_0
        return self.tau_0 * abs(W) ** self.alpha

    @staticmethod
    def is_topologically_protected(W: int) -> bool:
        r"""Check if identity is topologically protected (Thm 3.1).

        W ≠ 0 → protected against small perturbations.

        Parameters
        ----------
        W : int
            Winding number.

        Returns
        -------
        bool
            True if W ≠ 0 (topologically protected).
        """
        return W != 0

    def identity_continuity(self, phi_trajectory: List[np.ndarray]) -> Dict[str, object]:
        r"""Check identity continuity along a trajectory.

        Identity is continuous iff W[φ_mind(γ(t))] is constant.

        Parameters
        ----------
        phi_trajectory : list of ndarray
            Phase field at successive time steps.

        Returns
        -------
        dict with:
            'winding_numbers' : list of int
            'is_continuous' : bool
            'discontinuity_times' : list of int (indices where W changes)
        """
        W_list = [self.winding_number(phi) for phi in phi_trajectory]
        disc_times = [i for i in range(len(W_list) - 1)
                      if W_list[i] != W_list[i + 1]]
        return {
            "winding_numbers": W_list,
            "is_continuous": len(disc_times) == 0,
            "discontinuity_times": disc_times,
        }

    @staticmethod
    def structural_similarity(phi1: np.ndarray, phi2: np.ndarray) -> float:
        r"""Phase overlap measure σ(γ(t), γ(t+Δt)).

        σ = |<exp(iφ₁) | exp(iφ₂)>|² / (N₁ N₂)

        Parameters
        ----------
        phi1, phi2 : ndarray
            Phase configurations at two times.

        Returns
        -------
        float
            Structural similarity ∈ [0, 1].
        """
        z1 = np.exp(1j * phi1)
        z2 = np.exp(1j * phi2)
        overlap = np.abs(np.sum(z1.conj() * z2)) ** 2
        norm = len(z1) * len(z2)
        return float(overlap / norm)

    @staticmethod
    def vortex_core_energy(W: int, system_size: float = 1.0,
                           core_size: float = 0.01) -> float:
        r"""Energy of a vortex core required to change W.

        E ~ |W| ∫ |∇φ|² d²x ~ |W| ln(R/a)

        where R is system size and a is core size.

        Parameters
        ----------
        W : int
            Winding number.
        system_size : float
            System size R.
        core_size : float
            Vortex core size a.

        Returns
        -------
        float
            Vortex core energy (proportional to topological barrier).
        """
        if core_size <= 0 or system_size <= core_size:
            return np.inf
        return abs(W) * np.log(system_size / core_size)


# ===================================================================
# Permission-Gated Boundary Coupling (Sec 4, Def 4.1)
# ===================================================================

@dataclass
class PermissionField:
    r"""Permission-gated boundary coupling (Sec 4, Def 4.1).

    The gated coupling strength:
        C(x, t) = P(x, t) ⟨x, Bx⟩                  (Eq 5)

    P = 0 → sealed boundary (maximum impedance mismatch)
    P = 1 → full resonance transmission

    P(x, t) = σ(E[ψ(x,t)] - E_min)

    where σ is a smooth sigmoid and E is the eligibility functional.

    Parameters
    ----------
    E_min : float
        Eligibility threshold.
    steepness : float
        Sigmoid steepness parameter.
    """
    E_min: float = 0.5
    steepness: float = 10.0

    def sigmoid(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Smooth sigmoid function σ(x) = 1/(1 + exp(-k·x)).

        Parameters
        ----------
        x : float or ndarray
            Input.

        Returns
        -------
        float or ndarray
            Sigmoid output ∈ (0, 1).
        """
        return 1.0 / (1.0 + np.exp(-self.steepness * np.asarray(x, dtype=float)))

    def permission(self, eligibility: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Compute permission field P(x, t) (Def 4.1).

        P = σ(E[ψ] - E_min)

        Parameters
        ----------
        eligibility : float or ndarray
            Eligibility functional value E[ψ(x, t)].

        Returns
        -------
        float or ndarray
            Permission value P ∈ (0, 1).
        """
        return self.sigmoid(np.asarray(eligibility, dtype=float) - self.E_min)

    def gated_coupling(self, eligibility: float,
                       inner_product: float) -> float:
        r"""Gated coupling strength C(x, t) (Eq 5).

        C = P(x, t) · ⟨x, Bx⟩

        Parameters
        ----------
        eligibility : float
            Eligibility functional value.
        inner_product : float
            Inner product ⟨x, Bx⟩.

        Returns
        -------
        float
            Gated coupling strength.
        """
        P = float(self.permission(eligibility))
        return P * inner_product

    def boundary_tension(self, grad_V: np.ndarray) -> float:
        r"""Boundary tension τ(x) = ‖∇_x V(x)‖.

        High tension → low permission (1 - P proxy).

        Parameters
        ----------
        grad_V : ndarray
            Gradient of coherence potential.

        Returns
        -------
        float
            Boundary tension.
        """
        return float(np.linalg.norm(grad_V))

    def is_sealed(self, eligibility: float,
                  threshold: float = 0.01) -> bool:
        r"""Check if boundary is effectively sealed (P ≈ 0).

        Parameters
        ----------
        eligibility : float
            Eligibility value.
        threshold : float
            Below this P value, boundary is considered sealed.

        Returns
        -------
        bool
            True if P < threshold.
        """
        return float(self.permission(eligibility)) < threshold

    def is_fully_open(self, eligibility: float,
                      threshold: float = 0.99) -> bool:
        r"""Check if boundary is fully transmitting (P ≈ 1).

        Parameters
        ----------
        eligibility : float
            Eligibility value.
        threshold : float
            Above this P value, boundary is considered fully open.

        Returns
        -------
        bool
            True if P > threshold.
        """
        return float(self.permission(eligibility)) > threshold


# ===================================================================
# Semantic Encoding Operator (Sec 5, Def 5.1)
# ===================================================================

@dataclass
class SemanticEncoding:
    r"""Semantic encoding operator Φ: X → M (Sec 5, Def 5.1).

    In the BPR framework:
        Φ(x) = φ_mind|_{∂O(x)}

    Optimal encoding minimizes:
        Φ* = arg min_Φ E[D(ψ, Φ(ψ))]                 (Eq 6)

    subject to the coherence regularizer:
        L_coh = E_{x,f} ‖Φ(f(g(x))) - Φ(g(f(x)))‖² = 0
    """

    @staticmethod
    def distortion(psi: np.ndarray, phi_encoded: np.ndarray) -> float:
        r"""Compute encoding distortion D(ψ, Φ(ψ)).

        D = ‖ψ - Φ(ψ)‖² / ‖ψ‖²

        Parameters
        ----------
        psi : ndarray
            Original state.
        phi_encoded : ndarray
            Encoded representation.

        Returns
        -------
        float
            Normalized distortion ∈ [0, ∞).
        """
        norm_sq = float(np.sum(np.abs(psi) ** 2))
        if norm_sq < 1e-15:
            return 0.0
        return float(np.sum(np.abs(psi - phi_encoded) ** 2) / norm_sq)

    @staticmethod
    def coherence_regularizer(phi_fg: np.ndarray,
                              phi_gf: np.ndarray) -> float:
        r"""Coherence regularizer L_coh.

        L_coh = ‖Φ(f(g(x))) - Φ(g(f(x)))‖²

        Measures equivariance: should be zero for semantic-preserving f, g.

        Parameters
        ----------
        phi_fg : ndarray
            Φ(f(g(x))) encoding.
        phi_gf : ndarray
            Φ(g(f(x))) encoding.

        Returns
        -------
        float
            Coherence loss L_coh ≥ 0.
        """
        return float(np.sum((phi_fg - phi_gf) ** 2))

    @staticmethod
    def phase_projection(x: np.ndarray) -> np.ndarray:
        r"""BPR phase projection Φ(x) = arg(x) (boundary phase field).

        Parameters
        ----------
        x : ndarray (complex)
            State vector.

        Returns
        -------
        ndarray
            Phase angles ∈ [-π, π].
        """
        return np.angle(x)


# ===================================================================
# Salience Field and Attention (Sec 6, Def 6.1)
# ===================================================================

@dataclass
class SalienceField:
    r"""Salience field and attention distribution (Sec 6, Def 6.1).

    Attention weight:
        A(x, t) = exp(β R(x, t)) / ∫ exp(β R(y, t)) dy    (Eq 7)

    In BPR: R(x, t) = S(γ|_x) = ∫_{∂S(x)} |∇φ · n̂|² dA

    High β → cold (mission-lock), concentrates on most stable attractors.
    Low β → warm (discovery), explores nearby resonant manifolds.

    Parameters
    ----------
    beta : float
        Inverse temperature (phase coupling strength).
    """
    beta: float = 1.0

    def attention_weights(self, relevance: np.ndarray) -> np.ndarray:
        r"""Compute softmax attention weights A(x, t) (Eq 7).

        A(x) = exp(β R(x)) / Σ exp(β R(y))

        Parameters
        ----------
        relevance : ndarray
            Relevance functional R(x, t) values.

        Returns
        -------
        ndarray
            Attention weights summing to 1.
        """
        R = np.asarray(relevance, dtype=float)
        # Numerically stable softmax
        R_shifted = R - np.max(R)
        exp_R = np.exp(self.beta * R_shifted)
        return exp_R / np.sum(exp_R)

    def attended_state(self, states: np.ndarray,
                       relevance: np.ndarray) -> np.ndarray:
        r"""Compute attention-weighted state.

        x_attended = Σ A(x_i) x_i

        Parameters
        ----------
        states : ndarray, shape (N, D)
            State vectors.
        relevance : ndarray, shape (N,)
            Relevance values.

        Returns
        -------
        ndarray, shape (D,)
            Attention-weighted state.
        """
        weights = self.attention_weights(relevance)
        return np.sum(weights[:, None] * states, axis=0)

    def entropy(self, relevance: np.ndarray) -> float:
        r"""Attention entropy H = -Σ A_i log A_i.

        Low entropy → concentrated (mission-lock).
        High entropy → diffuse (discovery mode).

        Parameters
        ----------
        relevance : ndarray
            Relevance values.

        Returns
        -------
        float
            Attention entropy.
        """
        weights = self.attention_weights(relevance)
        # Avoid log(0)
        w_safe = weights[weights > 1e-15]
        return float(-np.sum(w_safe * np.log(w_safe)))

    def max_entropy(self, n_states: int) -> float:
        r"""Maximum entropy (uniform attention).

        H_max = log(N)

        Parameters
        ----------
        n_states : int
            Number of states.

        Returns
        -------
        float
            Maximum possible entropy.
        """
        return float(np.log(n_states))

    def concentration_ratio(self, relevance: np.ndarray) -> float:
        r"""Concentration ratio: 1 - H/H_max.

        0 → uniform (discovery)
        1 → fully concentrated (mission-lock)

        Parameters
        ----------
        relevance : ndarray
            Relevance values.

        Returns
        -------
        float
            Concentration ratio ∈ [0, 1].
        """
        H = self.entropy(relevance)
        H_max = self.max_entropy(len(relevance))
        if H_max < 1e-15:
            return 1.0
        return 1.0 - H / H_max


# ===================================================================
# Trajectory Evaluation Functional (Sec 7, Def 7.1)
# ===================================================================

@dataclass
class TrajectoryEvaluation:
    r"""Trajectory evaluation functional J(γ) (Sec 7, Def 7.1).

    J(γ) = ∫_{t_0}^{t_1} U(γ(t)) dt                    (Eq 8)

    Preferred trajectories satisfy δJ = 0 (stationary action).

    With control input u(t):
        J(γ, u) = ∫ [U(γ(t)) - λ ‖u(t)‖²] dt

    where λ penalizes control effort.

    Parameters
    ----------
    lambda_control : float
        Control effort penalty (λ ≥ 0).
    """
    lambda_control: float = 0.1

    def evaluate(self, utility_values: np.ndarray,
                 dt: float = 1.0) -> float:
        r"""Evaluate trajectory functional J(γ) (Eq 8).

        J = Σ U(γ(t_i)) Δt

        Parameters
        ----------
        utility_values : ndarray
            Utility U(γ(t)) at each time step.
        dt : float
            Time step.

        Returns
        -------
        float
            Trajectory evaluation J(γ).
        """
        return float(np.sum(utility_values) * dt)

    def evaluate_with_control(self, utility_values: np.ndarray,
                              control_norms: np.ndarray,
                              dt: float = 1.0) -> float:
        r"""Evaluate trajectory with control cost.

        J(γ, u) = Σ [U(γ(t_i)) - λ ‖u(t_i)‖²] Δt

        Parameters
        ----------
        utility_values : ndarray
            Utility at each step.
        control_norms : ndarray
            ‖u(t)‖ at each step.
        dt : float
            Time step.

        Returns
        -------
        float
            Trajectory evaluation with control penalty.
        """
        control_cost = self.lambda_control * np.sum(control_norms ** 2)
        return float((np.sum(utility_values) - control_cost) * dt)

    @staticmethod
    def utility_from_coherence_potential(V: np.ndarray) -> np.ndarray:
        r"""Utility as negative coherence potential: U(x) = -V(x).

        Parameters
        ----------
        V : ndarray
            Coherence potential values.

        Returns
        -------
        ndarray
            Utility values.
        """
        return -np.asarray(V, dtype=float)

    def is_stationary(self, utility_values: np.ndarray,
                      dt: float = 1.0,
                      tolerance: float = 0.1) -> bool:
        r"""Check approximate stationarity δJ ≈ 0.

        A trajectory is approximately stationary if the utility
        gradient is small relative to mean utility.

        Parameters
        ----------
        utility_values : ndarray
            Utility at each step.
        dt : float
            Time step.
        tolerance : float
            Relative tolerance.

        Returns
        -------
        bool
            True if trajectory is approximately stationary.
        """
        if len(utility_values) < 3:
            return True
        grad_U = np.diff(utility_values) / dt
        mean_U = np.mean(np.abs(utility_values))
        if mean_U < 1e-15:
            return True
        return float(np.max(np.abs(grad_U))) / mean_U < tolerance


# ===================================================================
# Coherence Stack (Sec 8, Thm 8.1)
# ===================================================================

class OperatorType:
    """The nine minimal operators of the coherence stack."""
    STATE_SPACE = "state_space"
    STABILITY_MANIFOLDS = "stability_manifolds"
    PERSISTENCE = "persistence"
    COUPLING = "coupling"
    IDENTITY = "identity"
    PERMISSION = "permission"
    ENCODING = "encoding"
    SALIENCE = "salience"
    EVALUATION = "evaluation"


# Collapse modes if an operator is absent
COLLAPSE_MODES = {
    OperatorType.STATE_SPACE: "No substrate",
    OperatorType.STABILITY_MANIFOLDS: "No persistent structure",
    OperatorType.PERSISTENCE: "No memory",
    OperatorType.COUPLING: "No cross-layer interaction",
    OperatorType.IDENTITY: "Observer collapses",
    OperatorType.PERMISSION: "Uncontrolled coupling; causal chaos",
    OperatorType.ENCODING: "No semantic structure",
    OperatorType.SALIENCE: "No attentional selectivity",
    OperatorType.EVALUATION: "No preferred trajectories",
}


@dataclass
class CoherenceStack:
    r"""Minimal coherence stack for a reality architecture (Thm 8.1).

    Nine operators required for persistent observers, causal structure,
    and stable cross-scale coupling.  Absence of any single operator
    leads to a specific collapse mode.
    """
    operators_present: Dict[str, bool] = field(default_factory=lambda: {
        OperatorType.STATE_SPACE: True,
        OperatorType.STABILITY_MANIFOLDS: True,
        OperatorType.PERSISTENCE: True,
        OperatorType.COUPLING: True,
        OperatorType.IDENTITY: True,
        OperatorType.PERMISSION: True,
        OperatorType.ENCODING: True,
        OperatorType.SALIENCE: True,
        OperatorType.EVALUATION: True,
    })

    @property
    def is_coherent(self) -> bool:
        r"""Check if all nine operators are present (Thm 8.1).

        Returns
        -------
        bool
            True if the stack is fully coherent.
        """
        return all(self.operators_present.values())

    @property
    def n_operators(self) -> int:
        """Number of operators present."""
        return sum(1 for v in self.operators_present.values() if v)

    def collapse_modes(self) -> List[str]:
        r"""Return collapse modes for missing operators.

        Returns
        -------
        list of str
            Collapse mode descriptions for each missing operator.
        """
        modes = []
        for op, present in self.operators_present.items():
            if not present:
                modes.append(COLLAPSE_MODES.get(op, f"Unknown collapse: {op}"))
        return modes

    def remove_operator(self, operator: str) -> "CoherenceStack":
        r"""Return a new stack with one operator removed.

        Parameters
        ----------
        operator : str
            Operator to remove.

        Returns
        -------
        CoherenceStack
            New stack with the operator absent.
        """
        new_ops = dict(self.operators_present)
        new_ops[operator] = False
        return CoherenceStack(operators_present=new_ops)

    def missing_operators(self) -> List[str]:
        r"""Return list of missing operators.

        Returns
        -------
        list of str
            Names of absent operators.
        """
        return [op for op, present in self.operators_present.items()
                if not present]


# ===================================================================
# Interoperator Consistency Conditions (Sec 9)
# ===================================================================

@dataclass
class InteroperatorConsistency:
    r"""Interoperator consistency conditions (Sec 9).

    Three conditions for global coherence:

    Prop 9.1 (Salience–Stability): A(x,t) concentrates on S.
    Prop 9.2 (Permission–Eligibility): P > 0 only when E exceeds threshold.
    Prop 9.3 (Winding–Cache): τ_m = τ_0 |W|^α grows with W.
    """

    @staticmethod
    def salience_stability_consistency(
        attention_weights: np.ndarray,
        stability_mask: np.ndarray,
        epsilon: float = 0.1
    ) -> Dict[str, object]:
        r"""Check Salience–Stability Consistency (Prop 9.1).

        ∫_S A(x,t) dx ≥ 1 - ε

        Parameters
        ----------
        attention_weights : ndarray
            Attention weights A(x).
        stability_mask : ndarray of bool
            True for states in S.
        epsilon : float
            Consistency threshold.

        Returns
        -------
        dict with:
            'consistent' : bool
            'mass_on_S' : float
            'epsilon' : float
        """
        mass_on_S = float(np.sum(attention_weights[stability_mask]))
        return {
            "consistent": mass_on_S >= 1.0 - epsilon,
            "mass_on_S": mass_on_S,
            "epsilon": epsilon,
        }

    @staticmethod
    def permission_eligibility_consistency(
        permission_values: np.ndarray,
        eligibility_values: np.ndarray,
        E_threshold: float
    ) -> Dict[str, object]:
        r"""Check Permission–Eligibility Consistency (Prop 9.2).

        P(x,t) > 0 only when E[ψ] ≥ E_threshold.

        Parameters
        ----------
        permission_values : ndarray
            Permission field values P.
        eligibility_values : ndarray
            Eligibility functional values E.
        E_threshold : float
            Eligibility threshold.

        Returns
        -------
        dict with:
            'consistent' : bool
            'violations' : int (P > 0 when E < E_threshold)
        """
        # Check: wherever E < threshold, P should be ≈ 0
        below_threshold = eligibility_values < E_threshold
        violations = int(np.sum((permission_values > 0.01) & below_threshold))
        return {
            "consistent": violations == 0,
            "violations": violations,
        }

    @staticmethod
    def winding_cache_consistency(
        winding_numbers: np.ndarray,
        timescales: np.ndarray
    ) -> Dict[str, object]:
        r"""Check Winding–Cache Timescale Consistency (Prop 9.3).

        τ_m should increase with |W|.

        Parameters
        ----------
        winding_numbers : ndarray
            Winding numbers W.
        timescales : ndarray
            Corresponding memory timescales τ_m.

        Returns
        -------
        dict with:
            'consistent' : bool
            'correlation' : float (Spearman rank correlation)
        """
        abs_W = np.abs(winding_numbers)
        # Check monotonicity: higher |W| → higher τ_m
        if len(abs_W) < 2:
            return {"consistent": True, "correlation": 1.0}

        # Sort by |W| and check τ_m ordering
        order = np.argsort(abs_W)
        sorted_tau = timescales[order]
        sorted_W = abs_W[order]

        # Check: for distinct W values, τ should be non-decreasing
        consistent = True
        for i in range(len(sorted_W) - 1):
            if sorted_W[i] < sorted_W[i + 1]:
                if sorted_tau[i] > sorted_tau[i + 1]:
                    consistent = False
                    break

        # Simple correlation
        if np.std(abs_W) < 1e-10 or np.std(timescales) < 1e-10:
            corr = 1.0
        else:
            corr = float(np.corrcoef(abs_W, timescales)[0, 1])

        return {
            "consistent": consistent,
            "correlation": corr,
        }


# ===================================================================
# Stability Measure from BPR (bridging construct)
# ===================================================================

@dataclass
class BPRStabilityMeasure:
    r"""BPR stability measure S(γ) used by the salience field.

    S(γ|_x) = ∫_{∂S(x)} |∇φ · n̂|² dA

    States with stronger boundary phase gradients are more stable
    and attract more attention.
    """

    @staticmethod
    def compute(phi: np.ndarray, dx: float = 1.0) -> float:
        r"""Compute stability measure S from phase field.

        Parameters
        ----------
        phi : ndarray
            Phase field values on the boundary.
        dx : float
            Grid spacing.

        Returns
        -------
        float
            Stability measure S ≥ 0.
        """
        grad = np.diff(phi) / dx
        return float(np.sum(grad ** 2) * dx)

    @staticmethod
    def attractor_separability_loss(phi_ensemble: np.ndarray) -> float:
        r"""Attractor separability loss (Falsifiable Prediction 5).

        L_AS = ‖Cov(φ) - I‖²_F

        Should decrease with cognitive task performance.

        Parameters
        ----------
        phi_ensemble : ndarray, shape (N_samples, D)
            Ensemble of phase field samples.

        Returns
        -------
        float
            Attractor separability loss.
        """
        cov = np.cov(phi_ensemble, rowvar=False)
        I = np.eye(cov.shape[0])
        return float(np.sum((cov - I) ** 2))
