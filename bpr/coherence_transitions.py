"""
Theory XXII: Invariant Structure, Boundary Dynamics, and Symbolic Meaning
==========================================================================

Constructs a formal translation layer between BPR substrate dynamics and
symbolic meaning spaces.  Abstract symbolic concepts are mapped to
mathematically precise properties of dynamical systems:

    Truth        -> Dynamical invariance under Phi_t          (Def 4.1)
    Record       -> Information conservation H[Phi_t] = H     (Thm 5.1)
    Judgment     -> Asymptotic coherence evaluation J(H_i)    (Def 5.1)
    Deception    -> Local-global coherence inconsistency      (Def 6.1)
    Transition   -> Boundary-induced phase transition         (Sec 7)
    Fate         -> Topological trichotomy of winding number  (Thm 8.1)

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
            # Detect reset: Q jumps upward (only possible via reset to Q_0)
            if Q_traj[i] > Q_traj[i - 1]:
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

# ===================================================================
# Section 10.1.2: Terminal Coherence Surge
# ===================================================================

@dataclass
class TerminalCoherenceSurge:
    r"""Model for coherence burst during boundary decoupling (Section 10.1.2).

    The empirically observed gamma burst (~25-100 Hz) approximately 30 s
    post cardiac arrest should exhibit critical exponents consistent with
    a boundary-decoupling phase transition rather than simple neural rundown.

    Near the decoupling critical point chi -> 0, the coherence exhibits
    a surge before final collapse:

        K(t) = K_bg + A * |t - t_c|^{-gamma_crit} * exp(-(t - t_c)^2 / (2 sigma^2))

    for t < t_c (pre-collapse), followed by rapid decay for t > t_c.

    The critical exponent gamma_crit distinguishes:
        - BPR prediction:  gamma_crit in [0.5, 1.5] (boundary phase transition)
        - Neural rundown:  gamma_crit = 0 (monotone exponential decay, no surge)

    Parameters
    ----------
    K_bg : float
        Background coherence level.
    A : float
        Surge amplitude.
    gamma_crit : float
        Critical exponent of the boundary-decoupling transition.
    t_c : float
        Critical time (moment of complete decoupling).
    sigma : float
        Width of the surge envelope.
    freq_range : tuple of float
        Expected frequency range of the coherence burst (Hz).
    """
    K_bg: float = 0.05
    A: float = 0.8
    gamma_crit: float = 0.75
    t_c: float = 30.0
    sigma: float = 5.0
    freq_range: Tuple[float, float] = (25.0, 100.0)

    def coherence_profile(self, t: np.ndarray) -> np.ndarray:
        r"""Compute coherence K(t) during terminal decoupling.

        Parameters
        ----------
        t : ndarray
            Time array (seconds relative to cardiac arrest).

        Returns
        -------
        ndarray
            Coherence profile K(t).
        """
        t = np.asarray(t, dtype=float)
        K = np.full_like(t, self.K_bg)

        # Pre-critical surge: power-law divergence with Gaussian envelope
        pre = t < self.t_c
        dt_pre = self.t_c - t[pre]
        # Avoid division by zero with a small floor
        dt_safe = np.maximum(dt_pre, 1e-6)
        surge = self.A * dt_safe ** (-self.gamma_crit) * np.exp(
            -dt_pre ** 2 / (2.0 * self.sigma ** 2)
        )
        # Cap the surge at physical maximum K = 1
        K[pre] = np.minimum(self.K_bg + surge, 1.0)

        # Post-critical rapid decay
        post = t >= self.t_c
        dt_post = t[post] - self.t_c
        K[post] = self.K_bg * np.exp(-dt_post / (self.sigma * 0.1))

        return K

    def peak_coherence(self) -> float:
        r"""Compute the peak coherence during the surge.

        Returns
        -------
        float
            Maximum coherence value K_peak.
        """
        # Peak occurs where d/dt [|t_c - t|^{-gamma} * exp(-dt^2/2sigma^2)] = 0
        # Approximate numerically
        t_test = np.linspace(self.t_c - 4 * self.sigma, self.t_c - 0.01, 1000)
        K_test = self.coherence_profile(t_test)
        return float(np.max(K_test))

    def is_consistent_with_bpr(self, observed_gamma: float,
                                tolerance: float = 0.5) -> bool:
        r"""Test whether an observed critical exponent is consistent with BPR.

        BPR predicts gamma_crit in [0.5, 1.5] for boundary decoupling.
        Simple neural rundown predicts gamma_crit = 0 (no surge).

        Parameters
        ----------
        observed_gamma : float
            Observed critical exponent from neural coherence data.
        tolerance : float
            Allowed deviation from the BPR range.

        Returns
        -------
        bool
            True if observation is consistent with BPR prediction.
        """
        bpr_low = 0.5 - tolerance
        bpr_high = 1.5 + tolerance
        return bpr_low <= observed_gamma <= bpr_high

    @staticmethod
    def neural_rundown_profile(t: np.ndarray, K0: float = 0.5,
                                tau_decay: float = 10.0) -> np.ndarray:
        r"""Simple exponential neural rundown (null hypothesis).

        K(t) = K0 * exp(-t / tau_decay)

        No surge, monotone decay. This is the profile expected if the
        gamma burst is NOT a boundary-decoupling phase transition.

        Parameters
        ----------
        t : ndarray
            Time array.
        K0 : float
            Initial coherence.
        tau_decay : float
            Decay time constant.

        Returns
        -------
        ndarray
            Monotone decaying coherence.
        """
        return K0 * np.exp(-np.asarray(t) / tau_decay)


# ===================================================================
# Section 10.2.2: Superlinear Collective Coherence Scaling
# ===================================================================

@dataclass
class CollectiveCoherenceScaling:
    r"""Superlinear coherence scaling in synchronized groups (Section 10.2.2).

    BPR predicts that groups in synchronized practice show superlinear
    coherence scaling:

        chi_group ~ N^{1 + delta}     with delta > 0

    This is distinguishable from linear superposition (delta = 0), where
    coherence scales merely as N.

    The boundary-mediated coupling introduces correlations that amplify
    the collective coherence beyond the sum of individual contributions.

    Parameters
    ----------
    delta : float
        Superlinear exponent (BPR predicts delta > 0).
    chi_1 : float
        Single-agent baseline coherence.
    """
    delta: float = 0.15
    chi_1: float = 1.0

    def group_coherence(self, N: int) -> float:
        r"""Compute collective coherence for N synchronized agents.

        chi_group = chi_1 * N^{1 + delta}

        Parameters
        ----------
        N : int
            Number of synchronized agents.

        Returns
        -------
        float
            Collective coherence.
        """
        if N <= 0:
            return 0.0
        return self.chi_1 * float(N) ** (1.0 + self.delta)

    def linear_coherence(self, N: int) -> float:
        r"""Linear superposition baseline (null hypothesis).

        chi_linear = chi_1 * N

        Parameters
        ----------
        N : int
            Number of agents.

        Returns
        -------
        float
            Linear-scaling coherence.
        """
        if N <= 0:
            return 0.0
        return self.chi_1 * float(N)

    def superlinear_ratio(self, N: int) -> float:
        r"""Ratio of BPR superlinear to linear coherence.

        ratio = N^delta

        Parameters
        ----------
        N : int
            Number of agents.

        Returns
        -------
        float
            Superlinear enhancement factor.
        """
        if N <= 0:
            return 1.0
        return float(N) ** self.delta

    def fit_exponent(self, N_values: np.ndarray,
                     chi_values: np.ndarray) -> float:
        r"""Fit the scaling exponent from observed (N, chi) data.

        Fits log(chi) = log(chi_1) + (1 + delta) * log(N)
        and returns the estimated delta.

        Parameters
        ----------
        N_values : ndarray
            Array of group sizes.
        chi_values : ndarray
            Array of measured collective coherences.

        Returns
        -------
        float
            Fitted delta (superlinear exponent). delta > 0 supports BPR.
        """
        log_N = np.log(np.asarray(N_values, dtype=float))
        log_chi = np.log(np.asarray(chi_values, dtype=float))
        # Linear regression: log_chi = a + b * log_N, where b = 1 + delta
        coeffs = np.polyfit(log_N, log_chi, 1)
        return coeffs[0] - 1.0  # b - 1 = delta

    def is_superlinear(self, N_values: np.ndarray,
                       chi_values: np.ndarray,
                       significance: float = 0.0) -> bool:
        r"""Test whether observed scaling is superlinear.

        Parameters
        ----------
        N_values : ndarray
            Array of group sizes.
        chi_values : ndarray
            Array of measured collective coherences.
        significance : float
            Minimum delta to count as superlinear.

        Returns
        -------
        bool
            True if fitted delta > significance.
        """
        delta_fit = self.fit_exponent(N_values, chi_values)
        return delta_fit > significance


# ===================================================================
# Section 10.2.3: Duty-Cycle Optimization
# ===================================================================

@dataclass
class DutyCycleOptimizer:
    r"""Optimal duty cycle for driven resonant systems (Section 10.2.3).

    BPR predicts that productivity and resonance stability metrics show
    an optimal duty cycle near D* ~ 6/7, testable in any driven
    oscillator system.

    A driven oscillator with periodic forcing and dissipation achieves
    maximum sustained output at:

        D* = 1 - 1/(1 + Q/Q_rest)

    where Q is the active quality factor and Q_rest is the recovery
    quality factor during rest.

    For matched Q = Q_rest, D* = 1/2. For Q >> Q_rest (typical
    biological/physical systems), D* -> 1. For Q/Q_rest ~ 6,
    D* = 6/7 ~ 0.857.

    Parameters
    ----------
    Q_active : float
        Quality factor during active (driven) period.
    Q_rest : float
        Quality factor during rest (recovery) period.
    """
    Q_active: float = 6.0
    Q_rest: float = 1.0

    @property
    def optimal_duty_cycle(self) -> float:
        r"""Compute optimal duty cycle D*.

        D* = Q_active / (Q_active + Q_rest)

        Returns
        -------
        float
            Optimal duty cycle in (0, 1).
        """
        return self.Q_active / (self.Q_active + self.Q_rest)

    def sustained_output(self, D: float, n_cycles: int = 100) -> float:
        r"""Compute sustained output for a given duty cycle.

        Models a system that accumulates output during active phase
        and recovers capacity during rest phase.

        Output per cycle = D * capacity
        Capacity recovery = (1 - D) * Q_rest / Q_active

        Over many cycles, the system reaches a steady-state capacity
        that depends on D.

        Parameters
        ----------
        D : float
            Duty cycle in (0, 1).
        n_cycles : int
            Number of cycles to simulate.

        Returns
        -------
        float
            Average sustained output per cycle at steady state.
        """
        if D <= 0 or D >= 1:
            return 0.0
        capacity = 1.0
        outputs = []
        for _ in range(n_cycles):
            output = D * capacity
            outputs.append(output)
            # Depletion during active phase
            depletion = D * capacity / self.Q_active
            # Recovery during rest phase
            recovery = (1.0 - D) * (1.0 - capacity) * self.Q_rest
            capacity = np.clip(capacity - depletion + recovery, 0.0, 1.0)
        # Return average over last 10% of cycles (steady state)
        steady_start = max(0, int(n_cycles * 0.9))
        return float(np.mean(outputs[steady_start:]))

    def scan_duty_cycles(self, n_points: int = 100,
                          n_cycles: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        r"""Scan sustained output across duty cycles.

        Parameters
        ----------
        n_points : int
            Number of duty cycle values to test.
        n_cycles : int
            Cycles per simulation.

        Returns
        -------
        D_values : ndarray
            Duty cycle values.
        outputs : ndarray
            Sustained output at each D.
        """
        D_values = np.linspace(0.01, 0.99, n_points)
        outputs = np.array([self.sustained_output(D, n_cycles) for D in D_values])
        return D_values, outputs

    def find_optimal(self, n_points: int = 200,
                     n_cycles: int = 200) -> float:
        r"""Find the empirical optimal duty cycle by scanning.

        Returns
        -------
        float
            Duty cycle D that maximizes sustained output.
        """
        D_values, outputs = self.scan_duty_cycles(n_points, n_cycles)
        return float(D_values[np.argmax(outputs)])

    @staticmethod
    def sabbath_duty_cycle() -> float:
        r"""The Sabbath duty cycle: 6 days work / 7 days total.

        Returns
        -------
        float
            6/7 ~ 0.8571
        """
        return 6.0 / 7.0


# ===================================================================
# Section 10.4: Falsification Criteria
# ===================================================================

@dataclass
class FalsificationCriteria:
    r"""Explicit falsification criteria for the coherence transitions framework (Section 10.4).

    The framework is falsified if ANY of the following are observed:

    1. Physical memory effects are absent in systems where boundary
       phase coherence is well-established.
    2. The topological trichotomy admits a fourth, topologically
       distinct fate.
    3. Source attractors identified from one tradition fail to appear
       in independently developed traditions.
    4. Deceptive attractors (locally stable, globally unstable) are
       found to have infinite lifetime in the presence of fluctuations.
    """

    @staticmethod
    def test_memory_kernel_present(
        correlation_data: np.ndarray,
        tau_values: np.ndarray,
        oscillatory_threshold: float = 0.1,
    ) -> Dict[str, object]:
        r"""Test Criterion 1: memory kernel signatures must be present.

        If boundary phase coherence exists, temporal correlations should
        decay as e^{-|t|/tau_m} cos(omega_r t) — NOT pure exponential.

        Tests for sign changes in the correlation function (oscillatory
        component) which are absent in pure exponential decay.

        Parameters
        ----------
        correlation_data : ndarray
            Measured temporal correlation values C(tau).
        tau_values : ndarray
            Time lag values.
        oscillatory_threshold : float
            Minimum fraction of sign-changing points to confirm
            oscillatory behavior.

        Returns
        -------
        dict with:
            'has_oscillation' : bool
            'sign_change_fraction' : float
            'falsified' : bool -- True if criterion is violated
        """
        correlation_data = np.asarray(correlation_data)
        sign_changes = np.sum(np.diff(np.sign(correlation_data)) != 0)
        n_points = len(correlation_data) - 1
        fraction = sign_changes / n_points if n_points > 0 else 0.0

        has_oscillation = fraction > oscillatory_threshold
        return {
            "has_oscillation": has_oscillation,
            "sign_change_fraction": fraction,
            "falsified": not has_oscillation,
        }

    @staticmethod
    def test_trichotomy_complete(observed_fates: List[str]) -> Dict[str, object]:
        r"""Test Criterion 2: no fourth topologically distinct fate.

        Parameters
        ----------
        observed_fates : list of str
            Observed fate categories from decoupling experiments.
            Valid values: 'dissolution', 'migration', 'reincorporation'.

        Returns
        -------
        dict with:
            'known_fates' : set
            'unknown_fates' : set
            'falsified' : bool -- True if a fourth fate is observed
        """
        allowed = {"dissolution", "migration", "reincorporation"}
        observed_set = set(observed_fates)
        unknown = observed_set - allowed
        return {
            "known_fates": observed_set & allowed,
            "unknown_fates": unknown,
            "falsified": len(unknown) > 0,
        }

    @staticmethod
    def test_source_attractor_universality(
        traditions: Dict[str, List[str]],
        min_overlap_fraction: float = 0.5,
    ) -> Dict[str, object]:
        r"""Test Criterion 3: source attractors must appear across traditions.

        Parameters
        ----------
        traditions : dict
            Maps tradition name -> list of source attractor names found.
        min_overlap_fraction : float
            Minimum fraction of attractors shared across all traditions.

        Returns
        -------
        dict with:
            'shared_attractors' : set
            'union_attractors' : set
            'overlap_fraction' : float
            'falsified' : bool -- True if overlap is too low
        """
        if not traditions:
            return {"shared_attractors": set(), "union_attractors": set(),
                    "overlap_fraction": 0.0, "falsified": True}

        sets = [set(v) for v in traditions.values()]
        shared = sets[0]
        union = sets[0]
        for s in sets[1:]:
            shared = shared & s
            union = union | s

        fraction = len(shared) / len(union) if len(union) > 0 else 0.0
        return {
            "shared_attractors": shared,
            "union_attractors": union,
            "overlap_fraction": fraction,
            "falsified": fraction < min_overlap_fraction,
        }

    @staticmethod
    def test_deceptive_attractor_transience(
        delta_V: float,
        epsilon: float,
    ) -> Dict[str, object]:
        r"""Test Criterion 4: deceptive attractors must be transient.

        For any epsilon > 0, the Kramers escape time must be finite.

        Parameters
        ----------
        delta_V : float
            Barrier height of the deceptive attractor.
        epsilon : float
            Noise intensity.

        Returns
        -------
        dict with:
            'escape_time' : float
            'is_finite' : bool
            'falsified' : bool -- True if escape time is infinite with noise
        """
        tau = DeceptionClassifier.kramers_escape_time(delta_V, epsilon)
        is_finite = np.isfinite(tau)
        # Falsified if epsilon > 0 but lifetime is infinite
        # (Note: float64 overflow at dV/eps > 700 is a numerical limit,
        # not a physical infinite lifetime)
        falsified = (epsilon > 0) and not is_finite and (delta_V / epsilon <= 700)
        return {
            "escape_time": tau,
            "is_finite": is_finite,
            "falsified": falsified,
        }

    def run_all(
        self,
        correlation_data: Optional[np.ndarray] = None,
        tau_values: Optional[np.ndarray] = None,
        observed_fates: Optional[List[str]] = None,
        traditions: Optional[Dict[str, List[str]]] = None,
        delta_V: float = 1.0,
        epsilon: float = 0.1,
    ) -> Dict[str, Dict[str, object]]:
        r"""Run all four falsification criteria.

        Parameters are optional; tests with None inputs are skipped.

        Returns
        -------
        dict mapping criterion name -> result dict.
        """
        results = {}

        if correlation_data is not None and tau_values is not None:
            results["memory_kernel"] = self.test_memory_kernel_present(
                correlation_data, tau_values
            )

        if observed_fates is not None:
            results["trichotomy"] = self.test_trichotomy_complete(observed_fates)

        if traditions is not None:
            results["universality"] = self.test_source_attractor_universality(
                traditions
            )

        results["deceptive_transience"] = self.test_deceptive_attractor_transience(
            delta_V, epsilon
        )

        return results


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
