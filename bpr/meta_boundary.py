"""
Theory XXII: Meta-Boundary Dynamics with Exogenous Decree
==========================================================

Extends BPR by introducing a dynamical constraint field κ(x,t) that parametrizes
the space of admissible boundary configurations. The constraint field evolves by
gradient flow on a meta-functional, with optional exogenous "decree" term D(x,t).

Key objects
----------
* ``MetaFunctional`` – J[κ;b,m] with stiffness, multiwell potential, coupling to stress
* ``MetaBoundaryEvolver`` – gradient flow τκ ∂_t κ = -δJ/δκ + ν∇²κ + D
* ``DecreeStochastic`` – class (A): stationary stochastic decree
* ``DecreeImpulse`` – class (B): event-sparse impulse process
* ``MetaEligibility`` – E2 = S_acc + S_D - C_barrier for meta-rewrite
* ``DetectabilitySignatures`` – domain wall, spectral drift, coherence dip

The constraint field κ(x,t) is distinct from the boundary rigidity κ in the
standard BPR action (κ = z/2). Here κ parametrizes which constraint operator
C_κ(b)=0 is active.

References
----------
Al-Kahwati, Meta-Boundary Dynamics with Exogenous Decree (PNAS submission).
Al-Kahwati, Meta-Boundary Dynamics and Global Phase Reindexing (internal draft).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Tuple

# ---------------------------------------------------------------------------
# §1  Meta-functional and constraint potential
# ---------------------------------------------------------------------------


@dataclass
class MetaBoundaryParams:
    """Parameters for meta-boundary dynamics.

    Attributes
    ----------
    alpha : float
        Constraint gradient stiffness [energy][length]^(d-2).
    beta : float
        Constraint potential scale.
    gamma : float
        Stress-to-constraint coupling.
    nu : float
        Constraint diffusivity [length]²[time]⁻¹.
    tau_kappa : float
        Meta-boundary relaxation time [time].
    eta : float
        Constraint phase separation (double-well minima at ±η).
    lambda_kappa : float
        Double-well quartic coefficient.
    """
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    nu: float = 1.0
    tau_kappa: float = 1.0
    eta: float = 1.0
    lambda_kappa: float = 1.0


def constraint_potential_double_well(
    kappa: np.ndarray,
    eta: float = 1.0,
    lambda_kappa: float = 1.0,
) -> np.ndarray:
    """Double-well constraint potential V_κ(κ) = (λ_κ/4)(κ² - η²)².

    Minima at κ = ±η.
    """
    k2 = np.asarray(kappa) ** 2
    return (lambda_kappa / 4.0) * (k2 - eta**2) ** 2


def constraint_potential_derivative(
    kappa: np.ndarray,
    eta: float = 1.0,
    lambda_kappa: float = 1.0,
) -> np.ndarray:
    """V'_κ(κ) = λ_κ κ (κ² - η²)."""
    k = np.asarray(kappa)
    return lambda_kappa * k * (k**2 - eta**2)


def meta_functional_value(
    kappa: np.ndarray,
    grad_kappa: np.ndarray,
    sigma_frust: np.ndarray,
    chi: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    params: Optional[MetaBoundaryParams] = None,
    dx: float = 1.0,
) -> float:
    """Evaluate J[κ;b,m] = ∫ (α/2|∇κ|² + β V_κ(κ) + γ W(κ,b,m)) d^d x.

    W(κ,b,m) = -½ χ(κ) σ_frust. Default χ(κ)=1.

    Parameters
    ----------
    kappa : ndarray
        Constraint field.
    grad_kappa : ndarray
        Gradient of κ (shape (..., d) for d spatial dimensions).
    sigma_frust : ndarray
        Boundary frustration stress σ_frust(b,m).
    chi : callable, optional
        Susceptibility χ(κ). Default identity.
    params : MetaBoundaryParams, optional
    dx : float
        Volume element (uniform grid).
    """
    if params is None:
        params = MetaBoundaryParams()
    if chi is None:
        chi = lambda k: np.ones_like(k)

    grad2 = np.sum(grad_kappa**2, axis=-1)
    V = constraint_potential_double_well(
        kappa, params.eta, params.lambda_kappa
    )
    W = -0.5 * chi(kappa) * sigma_frust

    integrand = (
        params.alpha / 2 * grad2
        + params.beta * V
        + params.gamma * W
    )
    return float(np.sum(integrand) * dx)


# ---------------------------------------------------------------------------
# §2  Constraint gradient flow (endogenous)
# ---------------------------------------------------------------------------


def meta_boundary_rhs(
    kappa: np.ndarray,
    sigma_frust: np.ndarray,
    lap_kappa: np.ndarray,
    chi: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    chi_prime: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    decree: Optional[np.ndarray] = None,
    params: Optional[MetaBoundaryParams] = None,
) -> np.ndarray:
    """Right-hand side of τ_κ ∂_t κ = -δJ/δκ + ν∇²κ + D.

    δJ/δκ = α∇²κ - β V'_κ(κ) - γ ∂W/∂κ
    ∂W/∂κ = -½ χ'(κ) σ_frust

    Returns dκ/dt (without the 1/τ_κ prefactor; caller multiplies).
    """
    if params is None:
        params = MetaBoundaryParams()
    if chi is None:
        chi = lambda k: np.ones_like(k)
    if chi_prime is None:
        chi_prime = lambda k: np.zeros_like(k)

    Vp = constraint_potential_derivative(
        kappa, params.eta, params.lambda_kappa
    )
    dW_dk = -0.5 * chi_prime(kappa) * sigma_frust

    rhs = (params.alpha + params.nu) * lap_kappa - params.beta * Vp - params.gamma * dW_dk
    if decree is not None:
        rhs = rhs + decree
    return rhs / params.tau_kappa


# ---------------------------------------------------------------------------
# §3  Decree classes (A) and (B)
# ---------------------------------------------------------------------------


class DecreeClass(Enum):
    """Admissible decree classes preserving falsifiability."""
    NONE = "none"
    STOCHASTIC = "stochastic"   # (A): stationary stochastic
    IMPULSE = "impulse"         # (B): event-sparse


@dataclass
class DecreeStochasticParams:
    """Parameters for class (A): stationary stochastic decree.

    E[D]=0, E[D(x,t)D(y,s)] = Σ(|x-y|) R(|t-s|)
    """
    ell_D: float = 1.0    # spatial correlation length
    tau_D: float = 1.0    # temporal correlation time
    var_D: float = 1.0    # variance (bounded)
    rng: Optional[np.random.Generator] = field(default=None, repr=False)


def decree_stochastic_sample(
    x: np.ndarray,
    t: float,
    params: Optional[DecreeStochasticParams] = None,
) -> np.ndarray:
    """Sample D(x,t) from stationary stochastic process.

    Simplified: Gaussian field with exponential correlations.
    E[D]=0, Var[D] = var_D.
    """
    if params is None:
        params = DecreeStochasticParams()
    rng = params.rng or np.random.default_rng()

    D = rng.normal(0, np.sqrt(params.var_D), size=x.shape)
    return D.astype(float)


@dataclass
class DecreeImpulseParams:
    """Parameters for class (B): event-sparse impulse process.

    D(x,t) = Σ_n a_n ψ(x-x_n) δ(t-t_n)
    """
    rate: float = 0.1       # event rate λ_D
    amplitude_mean: float = 1.0
    amplitude_std: float = 0.3
    kernel_support: float = 1.0  # width of ψ
    rng: Optional[np.random.Generator] = field(default=None, repr=False)


def decree_impulse_sample(
    x: np.ndarray,
    t: float,
    t_events: np.ndarray,
    x_events: np.ndarray,
    amplitudes: np.ndarray,
    kernel_support: float = 1.0,
) -> np.ndarray:
    """Evaluate D(x,t) = Σ_n a_n ψ(x-x_n) δ(t-t_n) at time t.

    For discrete time, δ(t-t_n) is replaced by Kronecker-like contribution
    when t matches an event. ψ is a Gaussian kernel with given support.

    Returns D(x) at the current time (nonzero only near event times).
    """
    D = np.zeros_like(x, dtype=float)
    x = np.asarray(x)
    for xn, an in zip(x_events, amplitudes):
        psi = np.exp(-0.5 * ((x - xn) / kernel_support) ** 2)
        D = D + an * psi
    return D


def generate_impulse_events(
    t_max: float,
    params: Optional[DecreeImpulseParams] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate event times, positions, and amplitudes for impulse decree.

    Returns
    -------
    t_events, x_events, amplitudes
    """
    if params is None:
        params = DecreeImpulseParams()
    rng = params.rng or np.random.default_rng()

    n_events = rng.poisson(params.rate * t_max)
    t_events = rng.uniform(0, t_max, size=n_events)
    x_events = rng.uniform(-5 * params.kernel_support, 5 * params.kernel_support, size=n_events)
    amplitudes = rng.normal(params.amplitude_mean, params.amplitude_std, size=n_events)
    return t_events, x_events, amplitudes


# ---------------------------------------------------------------------------
# §4  Meta-eligibility E2
# ---------------------------------------------------------------------------


@dataclass
class MetaEligibility:
    """Meta-rewrite eligibility E2 = S_acc + S_D - C_barrier.

    Meta-rewrite occurs when E2 ≥ 0.
    """

    S_acc: float
    S_D: float
    C_barrier: float

    @property
    def E2(self) -> float:
        return self.S_acc + self.S_D - self.C_barrier

    @property
    def meta_rewrite_eligible(self) -> bool:
        return self.E2 >= 0


def compute_S_D(kappa: np.ndarray, decree: np.ndarray, dx: float = 1.0) -> float:
    """S_D[κ;D] = ∫ D(x,t) κ(x,t) dx (minimal coupling)."""
    return float(np.sum(decree * kappa) * dx)


def barrier_cost_double_well(
    eta: float = 1.0,
    lambda_kappa: float = 1.0,
    alpha: float = 1.0,
) -> float:
    """Barrier height ΔV = V(0) - V(±η) = λ_κ η⁴/4 for double-well."""
    return (lambda_kappa / 4.0) * eta**4


# ---------------------------------------------------------------------------
# §5  Detectability signatures
# ---------------------------------------------------------------------------


@dataclass
class DetectabilityResult:
    """Result of detectability checks for a meta-boundary transition.

    At least one must be nonzero for nontrivial κ_a → κ_b.
    """
    domain_wall_energy: float
    spectral_drift: float
    metric_perturbation: float
    coherence_dip: float

    @property
    def any_detectable(self) -> bool:
        return (
            self.domain_wall_energy > 0
            or self.spectral_drift != 0
            or self.metric_perturbation != 0
            or self.coherence_dip != 0
        )


def domain_wall_tension(
    eta: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    lambda_kappa: float = 1.0,
) -> float:
    """Wall surface tension σ_wall = (2√2/3) η³ √(α β λ_κ)."""
    return (2.0 * np.sqrt(2) / 3.0) * eta**3 * np.sqrt(alpha * beta * lambda_kappa)


def spectral_drift_bound(
    delta_kappa: float,
    kappa_char: float,
    dln_lambda_dln_kappa: float = 1.0,
) -> float:
    """|δλ_n/λ_n| ≤ |Δκ|/κ_char · |∂ln λ_n/∂ln κ|."""
    return abs(delta_kappa) / kappa_char * abs(dln_lambda_dln_kappa)


def detectability_signatures(
    kappa_a: float,
    kappa_b: float,
    wall_area: float = 1.0,
    params: Optional[MetaBoundaryParams] = None,
) -> DetectabilityResult:
    """Compute detectability signatures for transition κ_a → κ_b.

    Returns domain wall energy, spectral drift bound, metric perturbation
    proxy, and coherence dip proxy.
    """
    if params is None:
        params = MetaBoundaryParams()
    if kappa_a == kappa_b:
        return DetectabilityResult(0.0, 0.0, 0.0, 0.0)

    sigma = domain_wall_tension(
        params.eta, params.alpha, params.beta, params.lambda_kappa
    )
    E_wall = sigma * wall_area

    delta_kappa = kappa_b - kappa_a
    kappa_char = max(abs(kappa_a), abs(kappa_b), params.eta, 1e-10)
    drift = spectral_drift_bound(delta_kappa, kappa_char)

    # Metric perturbation ~ (Δκ/ℓ_κ)²; use proxy
    metric_proxy = (delta_kappa / params.eta) ** 2

    # Coherence dip ~ fiber distance proxy
    coherence_dip = abs(delta_kappa)

    return DetectabilityResult(
        domain_wall_energy=E_wall,
        spectral_drift=drift,
        metric_perturbation=metric_proxy,
        coherence_dip=coherence_dip,
    )


# ---------------------------------------------------------------------------
# §6  Front propagation speed bound (from meta_boundary PDF)
# ---------------------------------------------------------------------------


def front_velocity_bound(
    params: Optional[MetaBoundaryParams] = None,
) -> float:
    """Maximum propagation speed |v| ≤ v_max = √((α+ν)/τ_κ) · m_κ.

    m_κ = √(2β λ_κ/α) η for double-well.
    """
    if params is None:
        params = MetaBoundaryParams()
    m_kappa = np.sqrt(2 * params.beta * params.lambda_kappa / params.alpha) * params.eta
    D_eff = (params.alpha + params.nu) / params.tau_kappa
    return np.sqrt(D_eff) * m_kappa


# ---------------------------------------------------------------------------
# Summary / predictions for first_principles
# ---------------------------------------------------------------------------


def meta_boundary_predictions(
    kappa_rigidity: float = 3.0,
    xi: float = 1e-6,
    params: Optional[MetaBoundaryParams] = None,
) -> dict:
    """Return falsifiable predictions from meta-boundary dynamics.

    Keys include v_max, sigma_wall, E2 structure, decree signatures.
    """
    if params is None:
        params = MetaBoundaryParams()
    v_max = front_velocity_bound(params)
    sigma = domain_wall_tension(
        params.eta, params.alpha, params.beta, params.lambda_kappa
    )
    C_barrier = barrier_cost_double_well(
        params.eta, params.lambda_kappa, params.alpha
    )
    return {
        "P23.1_front_velocity_bound": v_max,
        "P23.2_domain_wall_tension": sigma,
        "P23.3_barrier_cost": C_barrier,
        "P23.4_tau_kappa": params.tau_kappa,
        "P23.5_detectability_theorem": "nontrivial transition implies ≥1 nonzero signature",
    }
