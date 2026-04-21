"""
EML Symbolic Regression via Master Formula
===========================================

Implements gradient-descent symbolic regression over the space of EML
expression trees, following Odrzywolek (2026) arXiv:2603.21852 §4.3.

The EML master formula is a fully parameterized binary tree of EML nodes
where every leaf or child input is a soft (softmax-weighted) mixture over
the available terminals.  Gradient descent recovers discrete structure
because argmax-like behaviour emerges as logits sharpen.

MASTER FORMULA PARAMETERISATION
────────────────────────────────
Tree is indexed BFS (root = 0; node i has children 2i+1 and 2i+2).

Leaf EML nodes (indices [2^(d-1)-1 … 2^d-2]):
    Each input (left, right) is  mix2(1, x) = softmax(α,β)·[1, x]
    → 2 logits per input, 4 logits per node.

Internal nodes (indices [0 … 2^(d-1)-2]):
    Each input (left, right) is  mix3(1, x, child) = softmax(α,β,γ)·[1, x, child]
    → 3 logits per input, 6 logits per node.

Parameter count:
    N = 6·(2^(d-1) - 1)  +  4·2^(d-1)  =  5·2^d - 6

Depth-2 check: 5·4 - 6 = 14.  Root (6) + 2 leaves (4+4=8) = 14. ✓

SNAP-AND-DECODE
───────────────
After continuous optimisation, `snap()` converts each logit vector to
one-hot (argmax → 10.0, rest → 0.0), making softmax a proper selector.
`to_symbolic()` then reads off the discrete EMLExpr tree.

References
----------
Odrzywolek, A. (2026). All elementary functions from a single operator.
  arXiv:2603.21852v2 [cs.SC], §4.3.
Al-Kahwati, J. (2026). BPR-Math-Spine. §22.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .eml import EMLNode, Const, Var, EMLExpr, LN_X
from .constants import P_DEFAULT, Z_DEFAULT
from .alpha_derivation import _EULER_GAMMA

# ── Optional PyTorch import ──────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    _ModuleBase = nn.Module
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    # Stub base so the class statement parses when torch is absent;
    # any actual call will raise via _require_torch().
    class _ModuleBase:  # type: ignore[no-redef]
        def __init__(self) -> None: ...  # noqa: E704


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for EML symbolic regression. "
            "Install it with: pip install torch"
        )


# ── BPR target functions (plain numpy) ──────────────────────────────────────

def target_ln(x: np.ndarray) -> np.ndarray:
    """ln(x) — the depth-3 EML sub-expression recoverable by regression."""
    return np.log(x)


def target_exp(x: np.ndarray) -> np.ndarray:
    """exp(x) — the depth-1 EML identity eml(x, 1)."""
    return np.exp(x)


def target_ln_sq(x: np.ndarray) -> np.ndarray:
    """[ln(x)]² — first composed EML benchmark; gate before BPR."""
    return np.log(x) ** 2


def target_bpr_screening(p: np.ndarray) -> np.ndarray:
    """[ln(p)]² — the dominant BPR fine-structure term (alias of target_ln_sq)."""
    return np.log(p) ** 2


def target_bpr_alpha(p: np.ndarray) -> np.ndarray:
    """Full BPR 1/alpha formula: [ln(p)]² + 3 + γ − 1/(2π)."""
    return np.log(p) ** 2 + 3.0 + _EULER_GAMMA - 1.0 / (2 * np.pi)


# ── EMLMasterFormula ─────────────────────────────────────────────────────────

class EMLMasterFormula(_ModuleBase):
    """Trainable EML master formula of fixed depth d.

    Each node i stores a (2, n_logits_i) parameter matrix where row 0
    governs the left input and row 1 governs the right input.

    - Leaf nodes: n_logits = 2  (mix over {1, x})
    - Internal nodes: n_logits = 3  (mix over {1, x, child_result})

    Parameters
    ----------
    depth : int
        Tree depth ≥ 1.  Root is at depth 0; leaf EML nodes are at
        depth d-1 (BFS level d-1, indices [2^(d-1)-1, 2^d-2]).
    """

    def __init__(self, depth: int) -> None:
        _require_torch()
        super().__init__()

        if depth < 1:
            raise ValueError(f"depth must be ≥ 1, got {depth}")

        self.depth = depth
        self._n_nodes = 2**depth - 1          # total nodes in full binary tree
        self._first_leaf = 2 ** (depth - 1) - 1
        self._last_leaf = 2**depth - 2

        params = []
        for i in range(self._n_nodes):
            is_leaf = self._first_leaf <= i <= self._last_leaf
            n_logits = 2 if is_leaf else 3
            # shape (2, n_logits): row 0 = left input, row 1 = right input
            p = nn.Parameter(torch.zeros(2, n_logits))
            params.append(p)

        self.node_params: nn.ParameterList = nn.ParameterList(params)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Evaluate master formula at x.

        Parameters
        ----------
        x : torch.Tensor
            Input values, shape (n,).  Cast to complex128 internally.
        temperature : float
            Softmax temperature.  1.0 = normal training.  Values < 1 sharpen
            the distribution toward one-hot (used during hardening phase).

        Returns
        -------
        torch.Tensor
            Complex128 output, shape (n,).
        """
        x = x.to(torch.complex128)
        one = torch.ones_like(x)

        # Bottom-up BFS evaluation; cache[i] = result at node i
        cache: dict[int, torch.Tensor] = {}

        # Process nodes in reverse BFS order (leaves first)
        for i in range(self._n_nodes - 1, -1, -1):
            is_leaf = self._first_leaf <= i <= self._last_leaf
            logits = self.node_params[i]   # (2, n_logits)

            scaled = logits / temperature
            left_weights = torch.softmax(scaled[0], dim=0)   # (n_logits,)
            right_weights = torch.softmax(scaled[1], dim=0)  # (n_logits,)

            if is_leaf:
                # mix2: {1, x}
                left_val = left_weights[0] * one + left_weights[1] * x
                right_val = right_weights[0] * one + right_weights[1] * x
            else:
                # mix3: {1, x, child_result}
                left_child = cache[2 * i + 1]
                right_child = cache[2 * i + 2]
                left_val = (
                    left_weights[0] * one
                    + left_weights[1] * x
                    + left_weights[2] * left_child
                )
                right_val = (
                    right_weights[0] * one
                    + right_weights[1] * x
                    + right_weights[2] * right_child
                )

            # Numerically stable EML: clamp exp input, guard log input
            left_clamped = torch.view_as_complex(
                torch.stack([
                    left_val.real.clamp(-50.0, 50.0),
                    left_val.imag,
                ], dim=-1)
            )
            exp_part = torch.exp(left_clamped)

            # Guard right_val away from zero before log
            abs_right = right_val.abs()
            guard = (abs_right < 1e-30).to(torch.float64)
            right_guarded = right_val + (1e-30j * guard)
            log_part = torch.log(right_guarded)

            cache[i] = exp_part - log_part

        return cache[0]

    # ── snap ─────────────────────────────────────────────────────────────────

    def snap(self) -> "EMLMasterFormula":
        """Return a copy with each logit vector replaced by its argmax one-hot.

        The returned model has the same architecture but each softmax
        collapses to a hard selector, making the formula fully discrete.
        """
        snapped = EMLMasterFormula(self.depth)
        with torch.no_grad():
            for i, param in enumerate(self.node_params):
                new_param = torch.zeros_like(param)
                for row in range(param.shape[0]):
                    idx = int(param[row].argmax())
                    new_param[row, idx] = 10.0
                snapped.node_params[i].copy_(new_param)
        return snapped

    # ── to_symbolic ──────────────────────────────────────────────────────────

    def to_symbolic(self) -> EMLExpr:
        """Convert (snapped) master formula to a discrete EMLExpr tree.

        Should be called on the output of `snap()`.  The dominant logit
        in each row selects Const(1), Var('x'), or the child subtree.

        Returns
        -------
        EMLExpr
            Root of the recovered symbolic expression.
        """
        _ONE = Const(1)
        _X = Var("x")

        symbolic: dict[int, EMLExpr] = {}

        for i in range(self._n_nodes - 1, -1, -1):
            is_leaf = self._first_leaf <= i <= self._last_leaf
            logits = self.node_params[i]

            def _pick(weights_row: torch.Tensor, is_leaf_node: bool, child_idx: int) -> EMLExpr:
                idx = int(weights_row.argmax())
                if idx == 0:
                    return _ONE
                if idx == 1:
                    return _X
                # idx == 2 only reachable for internal nodes
                return symbolic[child_idx]

            if is_leaf:
                left_expr = _pick(logits[0], True, 2 * i + 1)
                right_expr = _pick(logits[1], True, 2 * i + 2)
            else:
                left_expr = _pick(logits[0], False, 2 * i + 1)
                right_expr = _pick(logits[1], False, 2 * i + 2)

            symbolic[i] = EMLNode(left_expr, right_expr)

        return symbolic[0]

    # ── n_params property ────────────────────────────────────────────────────

    @property
    def n_params(self) -> int:
        """Total scalar parameter count.  Equals 5·2^depth − 6."""
        return sum(p.numel() for p in self.node_params)


# ── FitResult ────────────────────────────────────────────────────────────────

@dataclass
class FitResult:
    """Result of one symbolic regression run.

    Attributes
    ----------
    depth : int
        Depth of the master formula used.
    n_steps : int
        Number of gradient steps taken.
    final_loss : float
        Mean squared error of the continuous model at convergence.
    snapped_loss : float
        MSE after snapping to a discrete formula.
    recovered_expr : Optional[EMLExpr]
        The symbolic expression recovered after snapping, or None if
        to_symbolic raised an exception.
    success : bool
        True when snapped_loss < 1e-6.
    n_restarts_tried : int
        Number of random restarts attempted.
    """

    depth: int
    n_steps: int
    final_loss: float
    snapped_loss: float
    recovered_expr: Optional[EMLExpr]
    success: bool
    n_restarts_tried: int


# ── fit ──────────────────────────────────────────────────────────────────────

def fit(
    target_fn: Callable[[np.ndarray], np.ndarray],
    depth: int = 3,
    x_range: tuple[float, float] = (0.1, 10.0),
    n_points: int = 64,
    n_steps: int = 3000,
    n_restarts: int = 5,
    lr: float = 0.05,
    seed: int = 0,
    normalize_y: bool = False,
    harden_frac: float = 0.0,
    min_temp: float = 0.05,
    entropy_coeff: float = 0.1,
) -> FitResult:
    """Fit an EML master formula to a target function.

    Parameters
    ----------
    target_fn : callable
        Maps a 1-D numpy array of x values to a 1-D numpy array of
        target values.  Should return real-valued output.
    depth : int
        Depth of the master formula tree.
    x_range : (float, float)
        Log-uniform sampling range [a, b] for training points.
    n_points : int
        Number of training points.
    n_steps : int
        Gradient steps per restart.
    n_restarts : int
        Number of independent random restarts; best is kept.
    lr : float
        Adam learning rate.
    seed : int
        Base random seed (each restart uses seed + restart_index).
    harden_frac : float
        Fraction of n_steps to use as a hardening phase (0 = disabled).
        During hardening, softmax temperature anneals from 1.0 → min_temp
        and an entropy penalty pushes logits toward one-hot.
    min_temp : float
        Final softmax temperature at end of hardening phase.
    entropy_coeff : float
        Weight of the entropy penalty during hardening.

    Returns
    -------
    FitResult
    """
    _require_torch()

    # ── Training data ────────────────────────────────────────────────────────
    lo, hi = x_range
    rng = np.random.default_rng(seed)
    x_np = np.exp(rng.uniform(np.log(lo), np.log(hi), n_points))
    y_np = target_fn(x_np).astype(np.float64)

    # Optionally z-score targets so EML outputs stay O(1) during training.
    # We record scale/shift to convert snapped_loss back to original units.
    y_mean = 0.0
    y_std = 1.0
    if normalize_y:
        y_mean = float(y_np.mean())
        y_std = float(y_np.std()) or 1.0
        y_np = (y_np - y_mean) / y_std

    x_t = torch.tensor(x_np, dtype=torch.float64)
    y_t = torch.tensor(y_np, dtype=torch.float64)

    best_model: Optional[EMLMasterFormula] = None
    best_loss = float("inf")
    best_snap_loss = float("inf")
    best_snap_model: Optional[EMLMasterFormula] = None

    for restart in range(n_restarts):
        torch.manual_seed(seed + restart)
        model = EMLMasterFormula(depth)

        # Initialise logits so each node strongly prefers {1} over {x} or child.
        # This keeps initial exp() arguments near 1, preventing the cascade
        # saturation that occurs at depth ≥ 4 when x-leaves dominate.
        # Small noise (0.1) still differentiates restarts.
        with torch.no_grad():
            for p in model.node_params:
                noise = torch.randn_like(p) * 0.1
                # Strongly penalise the x-logit (col 1) and child-logit (col 2)
                # so softmax starts ~[0.98, 0.01] or ~[0.97, 0.01, 0.01]
                penalty = torch.zeros_like(p)
                penalty[:, 1:] = -5.0   # penalise x and child at init
                p.copy_(noise + penalty)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        last_loss = float("inf")
        stuck_count = 0
        prev_loss = float("inf")
        harden_start = int(n_steps * (1.0 - harden_frac)) if harden_frac > 0 else n_steps

        for step in range(n_steps):
            # ── Hardening phase: anneal temperature + entropy penalty ────────
            if step >= harden_start:
                frac = (step - harden_start) / max(n_steps - harden_start, 1)
                temperature = 1.0 - frac * (1.0 - min_temp)  # 1.0 → min_temp
            else:
                temperature = 1.0

            optimizer.zero_grad()
            pred = model(x_t, temperature=temperature)
            mse = ((pred.real - y_t) ** 2).mean()

            # Entropy penalty during hardening: minimise ambiguity in logits
            if step >= harden_start and entropy_coeff > 0:
                ent = torch.tensor(0.0, dtype=torch.float64)
                for param in model.node_params:
                    for row in range(param.shape[0]):
                        p_row = torch.softmax(param[row] / temperature, dim=0)
                        ent = ent - (p_row * torch.log(p_row + 1e-10)).sum()
                loss = mse + entropy_coeff * ent / model.n_params
            else:
                loss = mse

            if torch.isnan(loss) or torch.isinf(loss):
                # Reinitialise this restart from scratch with same seed offset
                with torch.no_grad():
                    for p in model.node_params:
                        noise = torch.randn_like(p) * 0.1
                        penalty = torch.zeros_like(p)
                        penalty[:, 1:] = -5.0
                        p.copy_(noise + penalty)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            last_loss = float(mse.item())  # track MSE, not loss+penalty

            # Stuck-restart detection: abandon if no improvement after 500 steps
            if step % 500 == 499:
                if last_loss > prev_loss * 0.99 and last_loss > 1.0:
                    stuck_count += 1
                    if stuck_count >= 2:  # stuck for 1000 consecutive steps
                        break
                else:
                    stuck_count = 0
                prev_loss = last_loss

        # Check snap quality for this restart
        snap_candidate = model.snap()
        with torch.no_grad():
            snap_pred = snap_candidate(x_t)
            snap_loss = float(((snap_pred.real - y_t) ** 2).mean().item())

        # Early exit on clean snap (threshold applied in whatever scale y_t is in)
        if snap_loss < 1e-6:
            recovered: Optional[EMLExpr] = None
            try:
                recovered = snap_candidate.to_symbolic()
            except Exception:
                pass
            orig_snap_loss = snap_loss * y_std ** 2 if normalize_y else snap_loss
            orig_last_loss = last_loss * y_std ** 2 if normalize_y else last_loss
            return FitResult(
                depth=depth,
                n_steps=n_steps,
                final_loss=orig_last_loss,
                snapped_loss=orig_snap_loss,
                recovered_expr=recovered,
                success=orig_snap_loss < 1e-6,
                n_restarts_tried=restart + 1,
            )

        if snap_loss < best_snap_loss:
            best_snap_loss = snap_loss
            best_snap_model = snap_candidate

        if last_loss < best_loss:
            best_loss = last_loss
            # Deep-copy by re-instantiating and copying state
            best_model = EMLMasterFormula(depth)
            best_model.load_state_dict(
                {k: v.clone() for k, v in model.state_dict().items()}
            )

    assert best_model is not None

    # ── Use best snap model if it beat best continuous model's snap ──────────
    final_snap = best_snap_model if best_snap_model is not None else best_model.snap()
    snapped_loss = best_snap_loss if best_snap_model is not None else float("inf")

    if best_snap_model is None or best_snap_loss >= float("inf"):
        snapped = best_model.snap()
        with torch.no_grad():
            snapped_pred = snapped(x_t)
            snapped_loss = float(((snapped_pred.real - y_t) ** 2).mean().item())
        final_snap = snapped

    # Convert losses back to original-scale units if we normalised
    if normalize_y:
        best_loss = best_loss * y_std ** 2
        snapped_loss = snapped_loss * y_std ** 2

    recovered_expr: Optional[EMLExpr] = None
    try:
        recovered_expr = final_snap.to_symbolic()
    except Exception:
        pass

    return FitResult(
        depth=depth,
        n_steps=n_steps,
        final_loss=best_loss,
        snapped_loss=snapped_loss,
        recovered_expr=recovered_expr,
        success=snapped_loss < 1e-6,
        n_restarts_tried=n_restarts,
    )


# ── demo_bpr_recovery ────────────────────────────────────────────────────────

def demo_bpr_recovery(depth: int = 3, n_restarts: int = 10) -> dict:
    """Demonstrate symbolic regression on ln(x), the core BPR sub-expression.

    Attempts to recover the LN_X formula
        eml(1, eml(eml(1, x), 1))
    from data using the EML master formula.

    Parameters
    ----------
    depth : int
        Depth of master formula (3 is the minimum needed for ln(x)).
    n_restarts : int
        Number of random restarts.

    Returns
    -------
    dict
        depth, n_params, final_loss, snapped_loss, success,
        recovered_expr (str), matches_LN_X, n_restarts_tried.
    """
    _require_torch()

    result = fit(
        target_fn=target_ln,
        depth=depth,
        x_range=(0.1, 5.0),
        n_points=64,
        n_steps=4000,
        n_restarts=n_restarts,
        lr=0.05,
        seed=42,
    )

    # Check whether the recovered expression evaluates identically to LN_X
    matches_ln_x = False
    if result.recovered_expr is not None and result.success:
        test_xs = np.array([0.5, 1.0, 2.0, 5.0, np.e])
        try:
            errors = [
                abs(result.recovered_expr.eval(x=xi).real - np.log(xi))
                for xi in test_xs
            ]
            matches_ln_x = all(e < 1e-6 for e in errors)
        except Exception:
            matches_ln_x = False

    model = EMLMasterFormula(depth)

    return {
        "depth": result.depth,
        "n_params": model.n_params,
        "final_loss": result.final_loss,
        "snapped_loss": result.snapped_loss,
        "success": result.success,
        "recovered_expr": repr(result.recovered_expr),
        "matches_LN_X": matches_ln_x,
        "LN_X_formula": repr(LN_X),
        "n_restarts_tried": result.n_restarts_tried,
    }


# ── module entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print(demo_bpr_recovery())
