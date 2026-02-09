"""
Theory XXII: Invariant Structure, Boundary Dynamics, and Symbolic Meaning
==========================================================================

Constructs a formal translation layer between BPR substrate dynamics and
symbolic meaning spaces.  Religious eschatological concepts are mapped to
mathematically precise properties of dynamical systems:

    Truth        -> Dynamical invariance under Phi_t          (Def 4.1)
    Record       -> Information conservation H[Phi_t] = H     (Thm 5.1)
    Judgment     -> Asymptotic coherence evaluation J(H_i)    (Def 5.1)
    Deception    -> Local-global coherence inconsistency      (Def 6.1)
    Eschatology  -> Boundary-induced phase transition         (Sec 7)
    Death        -> Topological trichotomy of winding number  (Thm 8.1)

Key objects
-----------
* ``StainDynamics``           -- decoherence tracking variable s(t)   (Eq 6)
* ``HeartGainFunction``       -- coherence gain G(s)                  (Eq 12-13)
* ``JudgmentFunctional``      -- asymptotic coherence evaluator       (Def 5.1)
* ``DeceptionClassifier``     -- local vs global coherence detector   (Def 6.1)
* ``CollapseResetDynamics``   -- sawtooth Q_eff evolution             (Eq 15)
* ``DeathTrichotomy``         -- topological fate classification      (Thm 8.1)
* ``SymbolicProjection``      -- pi: S -> Sigma projection operator   (Def 3.1)
* ``CROSS_TRADITIONAL_MAP``   -- translation dictionary (Table 1)

References: Al-Kahwati (2026), *Invariant Structure, Boundary Dynamics,
and Symbolic Meaning*, StarDrive Research Group.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple, Callable
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_HBAR = 1.054571817e-34   # J s
_K_B = 1.380649e-23       # J/K


# ===================================================================
# Section 6 (Eq 6): Stain Dynamics
# ===================================================================

@dataclass
class StainDynamics:
    r"""Decoherence tracking variable s(t) in [0, 1].

    Equation (6) from the paper:

        ds/dt = alpha * u_minus(t) * (1 - s) - beta * u_plus(t) * s - gamma * s

    where:
        u_plus(t)  : coherence-restoring inputs (polishing)
        u_minus(t) : noise / decoherence inputs
        alpha      : decoherence accumulation rate
        beta       : coherence restoration rate
        gamma      : spontaneous decay rate of stain

    The stain variable tracks accumulated decoherence on [0, 1]:
        s = 0 : pristine coherence
        s = 1 : fully decohered

    Parameters
    ----------
    alpha : float
        Decoherence accumulation rate coefficient.
    beta : float
        Coherence restoration rate coefficient.
    gamma : float
        Spontaneous stain decay rate.
    s0 : float
        Initial stain value in [0, 1].
    """
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.01
    s0: float = 0.0

    def __post_init__(self):
        if not 0.0 <= self.s0 <= 1.0:
            raise ValueError(f"Initial stain s0 must be in [0, 1], got {self.s0}")

    def ds_dt(self, s: float, u_plus: float, u_minus: float) -> float:
        r"""Compute stain rate of change (Eq 6).

        Parameters
        ----------
        s : float
            Current stain value in [0, 1].
        u_plus : float
            Coherence-restoring input intensity (>= 0).
        u_minus : float
            Noise / decoherence input intensity (>= 0).

        Returns
        -------
        float
            Time derivative ds/dt.
        """
        return self.alpha * u_minus * (1.0 - s) - self.beta * u_plus * s - self.gamma * s

    def evolve(
        self,
        t_span: Tuple[float, float],
        u_plus: Callable[[float], float],
        u_minus: Callable[[float], float],
        n_points: int = 500,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Integrate stain dynamics over a time interval.

        Parameters
        ----------
        t_span : (t_start, t_end)
            Integration interval.
        u_plus : callable(t) -> float
            Coherence-restoring input as function of time.
        u_minus : callable(t) -> float
            Noise input as function of time.
        n_points : int
            Number of output time points.

        Returns
        -------
        t : ndarray, shape (n_points,)
            Time array.
        s : ndarray, shape (n_points,)
            Stain trajectory, clamped to [0, 1].
        """
        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        def rhs(t, y):
            s_val = np.clip(y[0], 0.0, 1.0)
            return [self.ds_dt(s_val, u_plus(t), u_minus(t))]

        sol = solve_ivp(
            rhs, t_span, [self.s0],
            t_eval=t_eval, method="RK45",
            max_step=(t_span[1] - t_span[0]) / 100,
        )
        s_out = np.clip(sol.y[0], 0.0, 1.0)
        return sol.t, s_out

    def steady_state(self, u_plus: float, u_minus: float) -> float:
        r"""Analytic steady-state stain for constant inputs.

        Setting ds/dt = 0:
            s* = alpha * u_minus / (alpha * u_minus + beta * u_plus + gamma)

        Parameters
        ----------
        u_plus : float
            Constant coherence-restoring input.
        u_minus : float
            Constant noise input.

        Returns
        -------
        float
            Steady-state stain s*.
        """
        numerator = self.alpha * u_minus
        denominator = self.alpha * u_minus + self.beta * u_plus + self.gamma
        if denominator <= 0:
            return 0.0
        return numerator / denominator


# ===================================================================
# Section 5.2 (Eq 12-13): Heart Gain Function & Coherence Evolution
# ===================================================================

@dataclass
class HeartGainFunction:
    r"""Heart gain function G(s) controlling coherence evolution.

    Equation (12):
        dK/dt = K_bar * G(s) * (1 - K) - nu * K

    where the gain function (beneath Eq 12) is:
        G(s) = exp(-kappa_s * s - 0.5 * sigma_s(s)^2)

    The asymptotic coherence (Eq 13):
        K* = K_bar * G(s*) / (K_bar * G(s*) + nu)

    High stain (s -> 1) drives G -> 0, hence K* -> 0.

    Parameters
    ----------
    K_bar : float
        Baseline coherence gain (maximum rate when s = 0).
    nu : float
        Coherence dissipation rate.
    kappa_s : float
        Stain-coherence coupling strength.
    sigma_scale : float
        Scale factor for stain-induced noise sigma_s(s) = sigma_scale * s.
    """
    K_bar: float = 1.0
    nu: float = 0.1
    kappa_s: float = 2.0
    sigma_scale: float = 1.0

    def sigma_s(self, s: float) -> float:
        """Stain-induced noise amplitude sigma_s(s) = sigma_scale * s."""
        return self.sigma_scale * s

    def G(self, s: float) -> float:
        r"""Heart gain function G(s) = exp(-kappa_s * s - 0.5 * sigma_s(s)^2).

        Parameters
        ----------
        s : float
            Stain value in [0, 1].

        Returns
        -------
        float
            Gain value G(s) in (0, 1].
        """
        return np.exp(-self.kappa_s * s - 0.5 * self.sigma_s(s) ** 2)

    def dK_dt(self, K: float, s: float) -> float:
        r"""Coherence rate of change (Eq 12).

        dK/dt = K_bar * G(s) * (1 - K) - nu * K

        Parameters
        ----------
        K : float
            Current coherence in [0, 1].
        s : float
            Current stain in [0, 1].

        Returns
        -------
        float
            Time derivative dK/dt.
        """
        return self.K_bar * self.G(s) * (1.0 - K) - self.nu * K

    def asymptotic_coherence(self, s_star: float) -> float:
        r"""Asymptotic coherence K* for a given steady-state stain (Eq 13).

        K* = K_bar * G(s*) / (K_bar * G(s*) + nu)

        Parameters
        ----------
        s_star : float
            Steady-state stain value.

        Returns
        -------
        float
            Asymptotic coherence K* in [0, 1].
        """
        g = self.G(s_star)
        kg = self.K_bar * g
        return kg / (kg + self.nu)

    def evolve_coherence(
        self,
        t_span: Tuple[float, float],
        K0: float,
        s_trajectory: Callable[[float], float],
        n_points: int = 500,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Integrate coherence evolution given a stain trajectory.

        Parameters
        ----------
        t_span : (t_start, t_end)
            Integration interval.
        K0 : float
            Initial coherence.
        s_trajectory : callable(t) -> float
            Stain as a function of time.
        n_points : int
            Number of output time points.

        Returns
        -------
        t : ndarray
            Time array.
        K : ndarray
            Coherence trajectory, clamped to [0, 1].
        """
        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        def rhs(t, y):
            K_val = np.clip(y[0], 0.0, 1.0)
            s_val = np.clip(s_trajectory(t), 0.0, 1.0)
            return [self.dK_dt(K_val, s_val)]

        sol = solve_ivp(
            rhs, t_span, [K0],
            t_eval=t_eval, method="RK45",
            max_step=(t_span[1] - t_span[0]) / 100,
        )
        K_out = np.clip(sol.y[0], 0.0, 1.0)
        return sol.t, K_out


# ===================================================================
# Section 5.2 (Def 5.1): Judgment Functional
# ===================================================================

@dataclass
class JudgmentFunctional:
    r"""Asymptotic coherence evaluation (Def 5.1).

    J(H_i) = lim_{t -> inf} K(S_t(H_i))

    Judgment is the dynamical process by which accumulated memory and
    stain force the system into a configuration consistent with its
    global invariants.

    Parameters
    ----------
    heart_gain : HeartGainFunction
        Controls coherence evolution.
    stain_dynamics : StainDynamics
        Controls stain evolution.
    """
    heart_gain: HeartGainFunction = field(default_factory=HeartGainFunction)
    stain_dynamics: StainDynamics = field(default_factory=StainDynamics)

    def evaluate(
        self,
        u_plus: Callable[[float], float],
        u_minus: Callable[[float], float],
        K0: float = 1.0,
        t_final: float = 100.0,
        n_points: int = 1000,
    ) -> Dict[str, float]:
        r"""Evaluate the judgment functional for a given input history.

        Integrates stain and coherence dynamics forward and returns the
        asymptotic coherence as the "judgment" outcome.

        Parameters
        ----------
        u_plus : callable(t) -> float
            Coherence-restoring input history.
        u_minus : callable(t) -> float
            Noise / decoherence input history.
        K0 : float
            Initial coherence.
        t_final : float
            Integration horizon (large enough for convergence).
        n_points : int
            Number of integration points.

        Returns
        -------
        dict with keys:
            'J'         : float -- judgment value (asymptotic coherence)
            'K_final'   : float -- coherence at t_final
            's_final'   : float -- stain at t_final
            'K_star'    : float -- analytic asymptotic coherence from steady-state stain
            's_star'    : float -- analytic steady-state stain
        """
        t_span = (0.0, t_final)

        # Evolve stain
        t_s, s_vals = self.stain_dynamics.evolve(t_span, u_plus, u_minus, n_points)
        s_final = s_vals[-1]

        # Build interpolating function for stain
        def s_interp(t):
            idx = np.searchsorted(t_s, t, side="right") - 1
            idx = np.clip(idx, 0, len(s_vals) - 1)
            return s_vals[idx]

        # Evolve coherence
        t_K, K_vals = self.heart_gain.evolve_coherence(t_span, K0, s_interp, n_points)
        K_final = K_vals[-1]

        # Analytic steady-state comparisons
        u_plus_final = u_plus(t_final)
        u_minus_final = u_minus(t_final)
        s_star = self.stain_dynamics.steady_state(u_plus_final, u_minus_final)
        K_star = self.heart_gain.asymptotic_coherence(s_star)

        return {
            "J": K_final,
            "K_final": K_final,
            "s_final": s_final,
            "K_star": K_star,
            "s_star": s_star,
        }

    def evaluate_analytic(self, u_plus_const: float, u_minus_const: float) -> float:
        r"""Quick analytic judgment for constant inputs.

        Computes s* then K* directly without numerical integration.

        Parameters
        ----------
        u_plus_const : float
            Constant coherence-restoring input.
        u_minus_const : float
            Constant noise input.

        Returns
        -------
        float
            Judgment value J = K*.
        """
        s_star = self.stain_dynamics.steady_state(u_plus_const, u_minus_const)
        return self.heart_gain.asymptotic_coherence(s_star)


# ===================================================================
# Section 6 (Def 6.1): Deception Classifier
# ===================================================================

@dataclass
class DeceptionClassifier:
    r"""Detect deceptive attractors via local-global coherence mismatch (Def 6.1).

    A state s is *deceptive* if:
        1. Local consistency:  K_local(s) > K_c   (appears stable locally)
        2. Global inconsistency:  K_global(s) < K_c  (unstable globally)

    The expected escape time from a deceptive attractor (Eq 14, Kramers):
        tau_escape ~ exp(Delta_V / epsilon)

    Parameters
    ----------
    K_c : float
        Critical coherence threshold separating stable from unstable.
    """
    K_c: float = 0.5

    def is_deceptive(self, K_local: float, K_global: float) -> bool:
        r"""Test whether a state is deceptive (Def 6.1).

        Parameters
        ----------
        K_local : float
            Coherence measured over a restricted observable set.
        K_global : float
            Coherence measured over the full observable set.

        Returns
        -------
        bool
            True if state is deceptive (locally stable, globally unstable).
        """
        return K_local > self.K_c and K_global < self.K_c

    def deception_degree(self, K_local: float, K_global: float) -> float:
        r"""Quantify the degree of deception as the coherence gap.

        D = max(0, K_local - K_global) when state is deceptive, else 0.

        Parameters
        ----------
        K_local : float
            Local coherence.
        K_global : float
            Global coherence.

        Returns
        -------
        float
            Deception degree in [0, 1]. Zero if state is not deceptive.
        """
        if self.is_deceptive(K_local, K_global):
            return max(0.0, K_local - K_global)
        return 0.0

    @staticmethod
    def kramers_escape_time(delta_V: float, epsilon: float) -> float:
        r"""Expected escape time from a deceptive (metastable) attractor (Eq 14).

        tau_escape ~ exp(Delta_V / epsilon)

        Parameters
        ----------
        delta_V : float
            Barrier height between local and global attractor (> 0).
        epsilon : float
            Noise intensity (> 0).

        Returns
        -------
        float
            Expected escape time. Returns np.inf if epsilon <= 0.
        """
        if epsilon <= 0:
            return np.inf
        if delta_V <= 0:
            return 0.0
        exponent = delta_V / epsilon
        if exponent > 700:
            return np.inf
        return np.exp(exponent)

    def classify_attractor_landscape(
        self,
        local_coherences: np.ndarray,
        global_coherences: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        r"""Classify an array of states as truthful, deceptive, or disordered.

        Parameters
        ----------
        local_coherences : ndarray
            Local coherence values for each state.
        global_coherences : ndarray
            Global coherence values for each state.

        Returns
        -------
        dict with boolean masks:
            'truthful'  : K_local > K_c AND K_global > K_c
            'deceptive' : K_local > K_c AND K_global < K_c
            'disordered': K_local < K_c AND K_global < K_c
            'hidden'    : K_local < K_c AND K_global > K_c (globally stable but locally obscured)
        """
        local_coherences = np.asarray(local_coherences)
        global_coherences = np.asarray(global_coherences)
        local_above = local_coherences > self.K_c
        global_above = global_coherences > self.K_c

        return {
            "truthful": local_above & global_above,
            "deceptive": local_above & ~global_above,
            "disordered": ~local_above & ~global_above,
            "hidden": ~local_above & global_above,
        }


# ===================================================================
# Section 7 (Eq 15): Collapse-Reset Dynamics
# ===================================================================

@dataclass
class CollapseResetDynamics:
    r"""Sawtooth collapse-reset dynamics of effective quality factor (Eq 15).

        Q_eff(t + dt) = Q_eff(t) - alpha_Q * Q_eff(t),   if Q_eff > Q_c
                       = Q_0,                              if Q_eff <= Q_c

    The system builds up quality (order) and then collapses when it
    drops below the critical threshold Q_c, resetting to baseline Q_0.

    Parameters
    ----------
    Q_0 : float
        Reset baseline quality factor (post-collapse).
    Q_c : float
        Collapse threshold (triggers reset when Q_eff drops to Q_c).
    alpha_Q : float
        Degradation rate per time step (fractional loss).
    """
    Q_0: float = 1.0
    Q_c: float = 0.1
    alpha_Q: float = 0.02

    def step(self, Q_current: float) -> float:
        r"""Advance Q_eff by one time step (Eq 15).

        Parameters
        ----------
        Q_current : float
            Current quality factor.

        Returns
        -------
        float
            Quality factor after one step.
        """
        Q_next = Q_current - self.alpha_Q * Q_current
        if Q_next <= self.Q_c:
            return self.Q_0
        return Q_next

    def evolve(self, n_steps: int, Q_init: Optional[float] = None) -> np.ndarray:
        r"""Evolve collapse-reset dynamics for n_steps.

        Parameters
        ----------
        n_steps : int
            Number of time steps to simulate.
        Q_init : float, optional
            Initial quality factor. Defaults to Q_0.

        Returns
        -------
        ndarray, shape (n_steps + 1,)
            Quality factor trajectory including initial value.
        """
        if Q_init is None:
            Q_init = self.Q_0
        Q = np.zeros(n_steps + 1)
        Q[0] = Q_init
        for i in range(n_steps):
            Q[i + 1] = self.step(Q[i])
        return Q

    def collapse_times(self, n_steps: int, Q_init: Optional[float] = None) -> List[int]:
        r"""Find time steps where collapse-reset events occur.

        Parameters
        ----------
        n_steps : int
            Number of time steps.
        Q_init : float, optional
            Initial quality factor.

        Returns
        -------
        list of int
            Indices where Q_eff was reset to Q_0.
        """
        Q_traj = self.evolve(n_steps, Q_init)
        resets = []
        for i in range(1, len(Q_traj)):
            # Detect reset: Q jumps upward to Q_0 from a decayed value
            if Q_traj[i] > Q_traj[i - 1] * 1.5:
                resets.append(i)
        return resets

    def collapse_period(self) -> float:
        r"""Analytic collapse period for exponential decay.

        Q(t) = Q_0 * (1 - alpha_Q)^t
        Collapse when Q(T) = Q_c:
            T = log(Q_c / Q_0) / log(1 - alpha_Q)

        Returns
        -------
        float
            Number of steps between consecutive collapses.
        """
        if self.alpha_Q <= 0 or self.alpha_Q >= 1:
            return np.inf
        ratio = self.Q_c / self.Q_0
        if ratio <= 0 or ratio >= 1:
            return np.inf
        return np.log(ratio) / np.log(1.0 - self.alpha_Q)


# ===================================================================
# Section 8 (Theorem 8.1): Death Trichotomy
# ===================================================================

class WindingFate(Enum):
    """Topologically allowed fates upon substrate decoupling (Thm 8.1)."""
    DISSOLUTION = "dissolution"
    MIGRATION = "migration"
    REINCORPORATION = "reincorporation"


@dataclass
class DeathTrichotomy:
    r"""Topological trichotomy of death transitions (Theorem 8.1).

    Upon complete substrate decoupling (chi -> 0), a consciousness
    winding number W != 0 must undergo exactly one of:

        1. Dissolution:       W -> 0 via annihilation with anti-winding
        2. Migration:         W transfers to higher-frequency boundary modes
        3. Re-incorporation:  W couples to an alternative substrate

    These exhaust the topologically allowed possibilities because
    W in Z is a homotopy invariant that cannot vanish under continuous
    evolution — it must annihilate, transfer, or re-couple.

    Parameters
    ----------
    W : int
        Consciousness winding number (must be nonzero for nontrivial fate).
    """
    W: int = 1

    def allowed_fates(self) -> List[WindingFate]:
        r"""Return the topologically allowed fates for this winding number.

        Returns
        -------
        list of WindingFate
            Always returns all three fates for W != 0;
            returns only DISSOLUTION for W = 0.
        """
        if self.W == 0:
            return [WindingFate.DISSOLUTION]
        return [
            WindingFate.DISSOLUTION,
            WindingFate.MIGRATION,
            WindingFate.REINCORPORATION,
        ]

    def dissolution_requires(self) -> int:
        r"""Anti-winding charge needed for dissolution.

        For dissolution, an equal and opposite winding -W must be present.

        Returns
        -------
        int
            Required anti-winding number (-W).
        """
        return -self.W

    def classify_fate(
        self,
        W_final: int,
        substrate_coupled: bool,
        mode_transferred: bool,
    ) -> WindingFate:
        r"""Classify the observed fate of a decoupling event.

        Parameters
        ----------
        W_final : int
            Winding number after decoupling.
        substrate_coupled : bool
            Whether W is coupled to an alternative substrate.
        mode_transferred : bool
            Whether W transferred to higher-frequency modes.

        Returns
        -------
        WindingFate
            The classified fate.
        """
        if W_final == 0:
            return WindingFate.DISSOLUTION
        if mode_transferred:
            return WindingFate.MIGRATION
        if substrate_coupled:
            return WindingFate.REINCORPORATION
        # Default: if W persists without clear migration or re-coupling,
        # classify as migration (winding is preserved somewhere)
        return WindingFate.MIGRATION

    @staticmethod
    def verify_conservation(W_before: int, W_after_components: List[int]) -> bool:
        r"""Verify that total winding is conserved across a transition.

        Parameters
        ----------
        W_before : int
            Total winding number before decoupling.
        W_after_components : list of int
            Winding numbers of all components after decoupling.

        Returns
        -------
        bool
            True if total winding is conserved.
        """
        return W_before == sum(W_after_components)


# ===================================================================
# Section 3 (Def 3.1-3.3): Symbolic Projection Operator
# ===================================================================

class AttractorType(Enum):
    """Source vs local attractor taxonomy (Def 3.2, 3.3)."""
    SOURCE = "source"   # universal dynamical principle
    LOCAL = "local"     # culture-specific symbolic encoding


@dataclass
class SymbolicElement:
    """An element of the symbolic meaning space Sigma.

    Parameters
    ----------
    name : str
        Symbolic concept name (e.g. 'truth', 'record', 'judgment').
    domain : str
        The tradition or domain (e.g. 'physics', 'Judaism', 'Christianity', 'Islam').
    attractor_type : AttractorType
        Whether this is a source (universal) or local (culture-specific) attractor.
    description : str
        Human-readable description of this symbolic element.
    """
    name: str
    domain: str
    attractor_type: AttractorType
    description: str = ""


@dataclass
class DynamicalInvariant:
    """A dynamical invariant in the substrate state space S.

    Parameters
    ----------
    name : str
        Invariant name (e.g. 'information_conservation', 'winding_number').
    mathematical_form : str
        LaTeX or text representation of the invariant.
    test_function : callable, optional
        Function I: S -> R that is preserved under evolution.
    """
    name: str
    mathematical_form: str
    test_function: Optional[Callable] = None


@dataclass
class SymbolicProjection:
    r"""Symbolic projection operator pi: S -> Sigma (Def 3.1).

    A surjective map that is many-to-one (lossy) but invariant-preserving:
    if I is an invariant of Phi_t, then pi(I) is well-defined in Sigma.

    The projection is analogous to coarse-graining in statistical mechanics.

    Methods
    -------
    project(invariant) -> list of SymbolicElement
        Map a dynamical invariant to its symbolic representations.
    inverse_image(element) -> DynamicalInvariant
        Map a symbolic element back to its underlying invariant.
    """
    _map: Dict[str, List[SymbolicElement]] = field(default_factory=dict)
    _inverse: Dict[Tuple[str, str], DynamicalInvariant] = field(default_factory=dict)

    def register(self, invariant: DynamicalInvariant, projections: List[SymbolicElement]):
        r"""Register a mapping from a dynamical invariant to symbolic elements.

        Parameters
        ----------
        invariant : DynamicalInvariant
            The dynamical invariant in S.
        projections : list of SymbolicElement
            Its projections in Sigma across different domains.
        """
        self._map[invariant.name] = projections
        for elem in projections:
            self._inverse[(elem.name, elem.domain)] = invariant

    def project(self, invariant_name: str) -> List[SymbolicElement]:
        r"""Project a dynamical invariant to its symbolic representations.

        Parameters
        ----------
        invariant_name : str
            Name of the dynamical invariant.

        Returns
        -------
        list of SymbolicElement
            All symbolic projections of this invariant.
        """
        return self._map.get(invariant_name, [])

    def inverse_image(self, element_name: str, domain: str) -> Optional[DynamicalInvariant]:
        r"""Map a symbolic element back to its dynamical invariant.

        Parameters
        ----------
        element_name : str
            Name of the symbolic element.
        domain : str
            Domain / tradition of the element.

        Returns
        -------
        DynamicalInvariant or None
            The underlying invariant, if registered.
        """
        return self._inverse.get((element_name, domain))

    def source_attractors(self) -> List[DynamicalInvariant]:
        r"""Return all registered invariants that have source-type projections.

        Source attractors are universal dynamical principles that appear
        in any substrate satisfying BPR/RPST axioms.

        Returns
        -------
        list of DynamicalInvariant
            Invariants with at least one SOURCE-type projection.
        """
        result = []
        seen = set()
        for inv_name, elems in self._map.items():
            if inv_name not in seen:
                if any(e.attractor_type == AttractorType.SOURCE for e in elems):
                    inv = self._inverse.get((elems[0].name, elems[0].domain))
                    if inv is not None:
                        result.append(inv)
                    seen.add(inv_name)
        return result

    def local_attractors(self, domain: str) -> List[SymbolicElement]:
        r"""Return all local-attractor elements for a specific tradition.

        Parameters
        ----------
        domain : str
            Tradition name (e.g. 'Judaism', 'Christianity', 'Islam').

        Returns
        -------
        list of SymbolicElement
            Local attractors in the specified domain.
        """
        result = []
        for elems in self._map.values():
            for e in elems:
                if e.domain == domain and e.attractor_type == AttractorType.LOCAL:
                    result.append(e)
        return result


# ===================================================================
# Section 9 (Table 1): Cross-Traditional Mapping
# ===================================================================

def build_default_projection() -> SymbolicProjection:
    r"""Construct the default cross-traditional projection (Table 1).

    Maps the core dynamical invariants to their symbolic representations
    across physics, Judaism, Christianity, and Islam.

    Returns
    -------
    SymbolicProjection
        Fully populated projection operator.
    """
    proj = SymbolicProjection()

    # --- Invariant truth: I(Phi_t(s)) = I(s) ---
    proj.register(
        DynamicalInvariant(
            "invariant_truth",
            r"I(\Phi_t(s)) = I(s) \; \forall t",
        ),
        [
            SymbolicElement("invariant_truth", "physics", AttractorType.SOURCE,
                            "Topological invariants preserved under evolution"),
            SymbolicElement("eternal_law", "Judaism", AttractorType.LOCAL,
                            "Torah as eternal law"),
            SymbolicElement("logos", "Christianity", AttractorType.LOCAL,
                            "Logos (John 1:1)"),
            SymbolicElement("preserved_tablet", "Islam", AttractorType.LOCAL,
                            "Qur'an as preserved tablet (al-Lawh al-Mahfuz)"),
        ],
    )

    # --- Information conservation: H[Phi_t(Psi)] = H[Psi] ---
    proj.register(
        DynamicalInvariant(
            "information_conservation",
            r"H[\Phi_t(\Psi)] = H[\Psi] \; \forall t",
        ),
        [
            SymbolicElement("information_conservation", "physics", AttractorType.SOURCE,
                            "Entropy preserved by bijective symplectic map"),
            SymbolicElement("book_of_life", "Judaism", AttractorType.LOCAL,
                            "Book of Life (Sefer HaChaim)"),
            SymbolicElement("book_of_revelation", "Christianity", AttractorType.LOCAL,
                            "Book of Revelation (Rev 20:12)"),
            SymbolicElement("kitab", "Islam", AttractorType.LOCAL,
                            "Kitab - recorded deeds"),
        ],
    )

    # --- Stain / decoherence ---
    proj.register(
        DynamicalInvariant(
            "stain_decoherence",
            r"\dot{s} = \alpha u^-(1-s) - \beta u^+ s - \gamma s",
        ),
        [
            SymbolicElement("stain_decoherence", "physics", AttractorType.SOURCE,
                            "Decoherence accumulation on boundary"),
            SymbolicElement("tumah", "Judaism", AttractorType.LOCAL,
                            "Impurity (tum'ah)"),
            SymbolicElement("sin_separation", "Christianity", AttractorType.LOCAL,
                            "Sin as separation"),
            SymbolicElement("black_spots", "Islam", AttractorType.LOCAL,
                            "Black spots on heart (ran)"),
        ],
    )

    # --- Coherence restoration ---
    proj.register(
        DynamicalInvariant(
            "coherence_restoration",
            r"u^+(t): \text{polishing inputs}",
        ),
        [
            SymbolicElement("coherence_restoration", "physics", AttractorType.SOURCE,
                            "Coherence-restoring inputs to boundary"),
            SymbolicElement("teshuvah", "Judaism", AttractorType.LOCAL,
                            "Teshuvah (repentance) and mitzvot"),
            SymbolicElement("repentance_grace", "Christianity", AttractorType.LOCAL,
                            "Repentance and grace"),
            SymbolicElement("tawbah", "Islam", AttractorType.LOCAL,
                            "Tawbah, dhikr, salat"),
        ],
    )

    # --- Deceptive attractor ---
    proj.register(
        DynamicalInvariant(
            "deceptive_attractor",
            r"K_{local} > K_c, \; K_{global} < K_c",
        ),
        [
            SymbolicElement("deceptive_attractor", "physics", AttractorType.SOURCE,
                            "Metastable attractor: locally stable, globally unstable"),
            SymbolicElement("false_prophets", "Judaism", AttractorType.LOCAL,
                            "False prophets"),
            SymbolicElement("antichrist", "Christianity", AttractorType.LOCAL,
                            "Antichrist"),
            SymbolicElement("dajjal", "Islam", AttractorType.LOCAL,
                            "Dajjal"),
        ],
    )

    # --- Collective synchronization ---
    proj.register(
        DynamicalInvariant(
            "collective_synchronization",
            r"R > R_c \text{ (Kuramoto)}",
        ),
        [
            SymbolicElement("collective_synchronization", "physics", AttractorType.SOURCE,
                            "Kuramoto synchronization above critical coupling"),
            SymbolicElement("sinai_revelation", "Judaism", AttractorType.LOCAL,
                            "Sinai revelation (collective receipt)"),
            SymbolicElement("pentecost", "Christianity", AttractorType.LOCAL,
                            "Pentecost"),
            SymbolicElement("ummah_coherence", "Islam", AttractorType.LOCAL,
                            "Ummah coherence"),
        ],
    )

    # --- Duty-cycle stability ---
    proj.register(
        DynamicalInvariant(
            "duty_cycle_stability",
            r"D^* \approx 6/7",
        ),
        [
            SymbolicElement("duty_cycle_stability", "physics", AttractorType.SOURCE,
                            "Optimal duty cycle near 6/7 for driven oscillators"),
            SymbolicElement("shabbat", "Judaism", AttractorType.LOCAL,
                            "Shabbat (6 days work, 1 day rest)"),
            SymbolicElement("sabbath_rest", "Christianity", AttractorType.LOCAL,
                            "Sabbath rest"),
            SymbolicElement("jumuah_prayer", "Islam", AttractorType.LOCAL,
                            "Jumu'ah / five daily prayer cycles"),
        ],
    )

    # --- Phase transition collapse ---
    proj.register(
        DynamicalInvariant(
            "phase_transition_collapse",
            r"Q_{eff} < Q_c",
        ),
        [
            SymbolicElement("phase_transition_collapse", "physics", AttractorType.SOURCE,
                            "Boundary-induced phase transition collapse"),
            SymbolicElement("temple_destruction", "Judaism", AttractorType.LOCAL,
                            "Destruction of Temple"),
            SymbolicElement("apocalypse", "Christianity", AttractorType.LOCAL,
                            "Apocalypse"),
            SymbolicElement("yawm_al_qiyamah", "Islam", AttractorType.LOCAL,
                            "Yawm al-Qiyamah (Day of Resurrection)"),
        ],
    )

    # --- Topological trichotomy ---
    proj.register(
        DynamicalInvariant(
            "topological_trichotomy",
            r"W: \text{dissolve, migrate, reincorporate}",
        ),
        [
            SymbolicElement("topological_trichotomy", "physics", AttractorType.SOURCE,
                            "Three topologically allowed fates for winding number"),
            SymbolicElement("sheol_olam_haba", "Judaism", AttractorType.LOCAL,
                            "Sheol / Olam HaBa"),
            SymbolicElement("heaven_hell_resurrection", "Christianity", AttractorType.LOCAL,
                            "Heaven / Hell / Resurrection"),
            SymbolicElement("jannah_jahannam_bath", "Islam", AttractorType.LOCAL,
                            "Jannah / Jahannam / Ba'th"),
        ],
    )

    return proj


# ===================================================================
# Convenience: summary dictionary matching Table 1
# ===================================================================

CROSS_TRADITIONAL_MAP: List[Dict[str, str]] = [
    {
        "dynamical_concept": "Invariant truth",
        "mathematical_form": "I(Phi_t(s)) = I(s)",
        "Judaism": "Torah as eternal law",
        "Christianity": "Logos (John 1:1)",
        "Islam": "Qur'an as preserved tablet",
    },
    {
        "dynamical_concept": "Information conservation",
        "mathematical_form": "H[Phi_t(Psi)] = H[Psi]",
        "Judaism": "Book of Life",
        "Christianity": "Book of Revelation",
        "Islam": "Kitab (recorded deeds)",
    },
    {
        "dynamical_concept": "Stain / decoherence",
        "mathematical_form": "ds/dt = alpha*u_minus*(1-s) - beta*u_plus*s",
        "Judaism": "Impurity (tum'ah)",
        "Christianity": "Sin as separation",
        "Islam": "Black spots on heart",
    },
    {
        "dynamical_concept": "Coherence restoration",
        "mathematical_form": "u_plus(t): polishing inputs",
        "Judaism": "Teshuvah, mitzvot",
        "Christianity": "Repentance, grace",
        "Islam": "Tawbah, dhikr, salat",
    },
    {
        "dynamical_concept": "Deceptive attractor",
        "mathematical_form": "K_local > K_c, K_global < K_c",
        "Judaism": "False prophets",
        "Christianity": "Antichrist",
        "Islam": "Dajjal",
    },
    {
        "dynamical_concept": "Collective synchronization",
        "mathematical_form": "Kuramoto R > R_c",
        "Judaism": "Sinai revelation",
        "Christianity": "Pentecost",
        "Islam": "Ummah coherence",
    },
    {
        "dynamical_concept": "Duty-cycle stability",
        "mathematical_form": "D* ~ 6/7",
        "Judaism": "Shabbat",
        "Christianity": "Sabbath rest",
        "Islam": "Jumu'ah / prayer cycles",
    },
    {
        "dynamical_concept": "Phase transition collapse",
        "mathematical_form": "Q_eff < Q_c",
        "Judaism": "Destruction of Temple",
        "Christianity": "Apocalypse",
        "Islam": "Yawm al-Qiyamah",
    },
    {
        "dynamical_concept": "Topological trichotomy",
        "mathematical_form": "W: dissolve, migrate, reincorporate",
        "Judaism": "Sheol / Olam HaBa",
        "Christianity": "Heaven / Hell / Resurrection",
        "Islam": "Jannah / Jahannam / Ba'th",
    },
]
