"""
Tests for EML Symbolic Regression (bpr.eml_regression)

Run with:  pytest -v tests/test_eml_regression.py
Slow tests: pytest -v -m slow tests/test_eml_regression.py
"""

import numpy as np
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

requires_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="PyTorch not installed"
)

from bpr.eml_regression import (
    target_ln, target_exp, target_ln_sq, target_bpr_screening, target_bpr_alpha,
    FitResult,
)
from bpr.constants import P_DEFAULT


# ─────────────────────────────────────────────────────────────────────────────
# §1  BPR target functions (no torch needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestTargetFunctions:
    def test_target_ln(self):
        x = np.array([1.0, np.e, 10.0])
        np.testing.assert_allclose(target_ln(x), np.log(x))

    def test_target_exp(self):
        x = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(target_exp(x), np.exp(x))

    def test_target_ln_sq(self):
        x = np.array([1.0, np.e, 10.0])
        np.testing.assert_allclose(target_ln_sq(x), np.log(x) ** 2)

    def test_target_ln_sq_at_one(self):
        # [ln(1)]² = 0; important boundary case
        result = target_ln_sq(np.array([1.0]))
        np.testing.assert_allclose(result, [0.0])

    def test_target_bpr_screening(self):
        p = np.array([float(P_DEFAULT)])
        result = target_bpr_screening(p)
        np.testing.assert_allclose(result, np.log(P_DEFAULT) ** 2)

    def test_target_bpr_screening_matches_ln_sq(self):
        x = np.array([2.0, 10.0, 100.0])
        np.testing.assert_allclose(target_bpr_screening(x), target_ln_sq(x))

    def test_target_bpr_alpha_near_137(self):
        p = np.array([float(P_DEFAULT)])
        result = target_bpr_alpha(p)
        assert abs(result[0] - 137.031) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# §2  EMLMasterFormula — structure
# ─────────────────────────────────────────────────────────────────────────────

@requires_torch
class TestEMLMasterFormulaStructure:
    @pytest.mark.parametrize("depth,expected", [
        (1, 4),    # 5*2 - 6 = 4
        (2, 14),   # 5*4 - 6 = 14
        (3, 34),   # 5*8 - 6 = 34
        (4, 74),   # 5*16 - 6 = 74
    ])
    def test_param_count_matches_formula(self, depth, expected):
        from bpr.eml_regression import EMLMasterFormula
        model = EMLMasterFormula(depth)
        assert model.n_params == expected, (
            f"depth={depth}: expected {expected}, got {model.n_params}"
        )

    def test_invalid_depth_raises(self):
        from bpr.eml_regression import EMLMasterFormula
        with pytest.raises(ValueError):
            EMLMasterFormula(0)

    def test_node_count(self):
        from bpr.eml_regression import EMLMasterFormula
        for depth in [1, 2, 3]:
            model = EMLMasterFormula(depth)
            assert len(model.node_params) == 2**depth - 1

    def test_leaf_nodes_have_2_logits(self):
        from bpr.eml_regression import EMLMasterFormula
        model = EMLMasterFormula(2)
        # Leaf EML nodes: indices [1, 2] for depth=2
        for i in [1, 2]:
            assert model.node_params[i].shape == (2, 2)

    def test_internal_nodes_have_3_logits(self):
        from bpr.eml_regression import EMLMasterFormula
        model = EMLMasterFormula(2)
        # Root: index 0 for depth=2
        assert model.node_params[0].shape == (2, 3)


# ─────────────────────────────────────────────────────────────────────────────
# §3  EMLMasterFormula — forward pass
# ─────────────────────────────────────────────────────────────────────────────

@requires_torch
class TestEMLMasterFormulaForward:
    def test_forward_returns_complex(self):
        from bpr.eml_regression import EMLMasterFormula
        model = EMLMasterFormula(2)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        out = model(x)
        assert out.is_complex()

    def test_forward_shape_preserved(self):
        from bpr.eml_regression import EMLMasterFormula
        model = EMLMasterFormula(2)
        x = torch.linspace(0.1, 5.0, 20, dtype=torch.float64)
        out = model(x)
        assert out.shape == x.shape

    def test_forward_finite_values(self):
        from bpr.eml_regression import EMLMasterFormula
        torch.manual_seed(0)
        model = EMLMasterFormula(2)
        x = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
        out = model(x)
        assert torch.isfinite(out.real).all(), "forward() produced non-finite real values"

    def test_forward_depth3_finite(self):
        from bpr.eml_regression import EMLMasterFormula
        torch.manual_seed(1)
        model = EMLMasterFormula(3)
        x = torch.linspace(0.1, 10.0, 50, dtype=torch.float64)
        out = model(x)
        assert torch.isfinite(out.real).all()

    def test_loss_is_differentiable(self):
        from bpr.eml_regression import EMLMasterFormula
        model = EMLMasterFormula(2)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        y = torch.tensor([0.0, np.log(2.0), np.log(3.0)], dtype=torch.float64)
        out = model(x)
        loss = ((out.real - y) ** 2).mean()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# §4  snap() and to_symbolic()
# ─────────────────────────────────────────────────────────────────────────────

@requires_torch
class TestSnapAndSymbolic:
    def test_snap_returns_new_model(self):
        from bpr.eml_regression import EMLMasterFormula
        model = EMLMasterFormula(2)
        snapped = model.snap()
        assert snapped is not model

    def test_snap_makes_one_hot_rows(self):
        from bpr.eml_regression import EMLMasterFormula
        torch.manual_seed(42)
        model = EMLMasterFormula(2)
        with torch.no_grad():
            for p in model.node_params:
                p.copy_(torch.randn_like(p))
        snapped = model.snap()
        for param in snapped.node_params:
            for row in range(param.shape[0]):
                row_vals = param[row].detach()
                non_zero = (row_vals != 0).sum().item()
                assert non_zero == 1, f"Expected 1 non-zero, got {non_zero}"

    def test_to_symbolic_returns_eml_expr(self):
        from bpr.eml_regression import EMLMasterFormula
        from bpr.eml import EMLExpr
        model = EMLMasterFormula(2)
        snapped = model.snap()
        expr = snapped.to_symbolic()
        assert isinstance(expr, EMLExpr)

    def test_snapped_symbolic_evaluates(self):
        from bpr.eml_regression import EMLMasterFormula
        model = EMLMasterFormula(2)
        snapped = model.snap()
        expr = snapped.to_symbolic()
        # Should evaluate without error at x=1
        result = expr.eval(x=1.0)
        assert np.isfinite(result.real)


# ─────────────────────────────────────────────────────────────────────────────
# §5  fit() — basic checks (no slow recovery)
# ─────────────────────────────────────────────────────────────────────────────

@requires_torch
class TestFit:
    def test_fit_returns_fitresult(self):
        from bpr.eml_regression import fit
        result = fit(
            target_fn=target_ln,
            depth=2,
            n_points=20,
            n_steps=50,
            n_restarts=1,
            seed=0,
        )
        assert isinstance(result, FitResult)

    def test_fit_loss_is_finite(self):
        from bpr.eml_regression import fit
        result = fit(
            target_fn=target_exp,
            depth=2,
            n_points=20,
            n_steps=50,
            n_restarts=1,
            seed=0,
        )
        assert np.isfinite(result.final_loss)
        assert np.isfinite(result.snapped_loss)

    def test_fit_depth_recorded(self):
        from bpr.eml_regression import fit
        result = fit(target_fn=target_ln, depth=2, n_points=10, n_steps=10, n_restarts=1)
        assert result.depth == 2

    def test_fit_restarts_recorded(self):
        from bpr.eml_regression import fit
        result = fit(target_fn=target_ln, depth=2, n_points=10, n_steps=10, n_restarts=3)
        assert result.n_restarts_tried == 3

    def test_fit_exp_depth1_converges(self):
        """exp(x) = eml(x, 1) is depth-1; should be recoverable reliably."""
        from bpr.eml_regression import fit
        result = fit(
            target_fn=target_exp,
            depth=1,
            x_range=(0.1, 3.0),
            n_points=64,
            n_steps=2000,
            n_restarts=5,
            seed=7,
        )
        # exp(x) is depth-1 — final continuous loss should be very small
        assert result.final_loss < 0.1, (
            f"Expected small loss for exp(x) at depth 1, got {result.final_loss}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# §6  Slow recovery tests
# ─────────────────────────────────────────────────────────────────────────────

@requires_torch
@pytest.mark.slow
class TestSlowRecovery:
    def test_ln_recovery_depth3(self):
        """Attempt to recover ln(x) = eml(1, eml(eml(1,x),1)) at depth 3.

        Per Odrzywolek (2026): ~25% success rate from random init.
        We run 20 restarts to get a reliable pass.
        """
        from bpr.eml_regression import fit
        result = fit(
            target_fn=target_ln,
            depth=3,
            x_range=(0.1, 5.0),
            n_points=64,
            n_steps=5000,
            n_restarts=20,
            lr=0.05,
            seed=42,
        )
        assert result.success, (
            f"ln(x) recovery failed at depth 3 after 20 restarts. "
            f"snapped_loss={result.snapped_loss:.6f}, "
            f"recovered={result.recovered_expr!r}"
        )

    def test_ln_sq_continuous_loss_depth5(self):
        """[ln(x)]² at depth 5 should at least converge continuously.

        This is the gate benchmark before BPR.  We require the continuous
        optimizer to reach a small loss (proof that depth 5 has capacity);
        exact snap recovery is tested separately once minimum depth is known
        from scripts/benchmark_ln2.py.
        """
        from bpr.eml_regression import fit
        result = fit(
            target_fn=target_ln_sq,
            depth=5,
            x_range=(0.5, 5.0),
            n_points=64,
            n_steps=5000,
            n_restarts=20,
            lr=0.02,
            seed=42,
            normalize_y=True,
        )
        # Continuous loss < 0.01 in original units means the model found a
        # good approximation of [ln(x)]²; snap quality is tracked separately.
        assert result.final_loss < 0.01, (
            f"[ln(x)]² continuous loss too high at depth 5: {result.final_loss:.4e}. "
            f"Optimizer may be diverging — check overflow or lr."
        )
