"""
Tests for EML–BPR Bridge (bpr.eml)

Run with:  pytest -v tests/test_eml.py
"""

import numpy as np
import pytest

from bpr.eml import (
    eml,
    Const, Var, EMLNode,
    E_CONST, EXP_X, ZERO, LN_X, LN_P,
    X_SQ, LN_X_SQ,
    ln_p_via_eml,
    bpr_alpha_via_eml,
    verify_eml_identities,
)
from bpr.constants import P_DEFAULT, Z_DEFAULT


# ─────────────────────────────────────────────────────────────────────────────
# §1  EML operator
# ─────────────────────────────────────────────────────────────────────────────

class TestEMLOperator:
    def test_eml_zero_one(self):
        # eml(0, 1) = exp(0) − ln(1) = 1 − 0 = 1
        assert abs(eml(0, 1) - 1.0) < 1e-14

    def test_eml_one_one_gives_e(self):
        # eml(1, 1) = exp(1) − ln(1) = e − 0 = e
        assert abs(eml(1, 1) - np.e) < 1e-14

    def test_eml_x_one_gives_exp_x(self):
        for x in [-2.0, 0.0, 1.0, 3.5]:
            assert abs(eml(x, 1) - np.exp(x)) < 1e-14

    def test_eml_complex_input(self):
        # Should not raise for complex inputs
        result = eml(1j, 1)
        assert np.isfinite(result.real)


# ─────────────────────────────────────────────────────────────────────────────
# §2  Tree structure
# ─────────────────────────────────────────────────────────────────────────────

class TestTreeStructure:
    def test_const_depth_zero(self):
        assert Const(1).depth() == 0

    def test_var_depth_zero(self):
        assert Var("x").depth() == 0

    def test_const_leaf_count_one(self):
        assert Const(1).leaf_count() == 1

    def test_node_depth(self):
        # eml(1, 1) → depth 1
        assert EMLNode(Const(1), Const(1)).depth() == 1

    def test_node_leaf_count(self):
        assert EMLNode(Const(1), Const(1)).leaf_count() == 2

    def test_rpn_length_formula(self):
        # K = 2*leaves - 1
        for tree in [E_CONST, EXP_X, ZERO, LN_X]:
            assert tree.rpn_length() == 2 * tree.leaf_count() - 1

    def test_repr_eml_node(self):
        r = repr(EMLNode(Const(1), Const(1)))
        assert r == "eml(1, 1)"


# ─────────────────────────────────────────────────────────────────────────────
# §3  Pre-built trees — depths and K values match Odrzywolek 2026 Table 4
# ─────────────────────────────────────────────────────────────────────────────

class TestPrebuiltTreeMetrics:
    def test_e_const_depth(self):
        assert E_CONST.depth() == 1

    def test_e_const_K(self):
        assert E_CONST.rpn_length() == 3   # K=3 per Table 4

    def test_exp_x_depth(self):
        assert EXP_X.depth() == 1

    def test_exp_x_K(self):
        assert EXP_X.rpn_length() == 3     # K=3 per Table 4

    def test_zero_depth(self):
        assert ZERO.depth() == 3

    def test_zero_K(self):
        assert ZERO.rpn_length() == 7      # K=7 per Table 4

    def test_ln_x_depth(self):
        assert LN_X.depth() == 3

    def test_ln_x_K(self):
        assert LN_X.rpn_length() == 7     # K=7 per Table 4 (same as 0)

    def test_ln_p_same_shape_as_ln_x(self):
        assert LN_P.depth() == LN_X.depth()
        assert LN_P.rpn_length() == LN_X.rpn_length()


# ─────────────────────────────────────────────────────────────────────────────
# §4  Pre-built trees — numerical correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestPrebuiltTreeValues:
    def test_e_const_value(self):
        assert abs(E_CONST.eval() - np.e) < 1e-14

    def test_exp_x_values(self):
        for x in [-3.0, -1.0, 0.0, 1.0, 2.5]:
            assert abs(EXP_X.eval(x=x) - np.exp(x)) < 1e-13

    def test_zero_value(self):
        assert abs(ZERO.eval().real) < 1e-13

    def test_ln_x_values(self):
        for x in [0.01, 0.5, 1.0, np.e, 10.0, 100.0]:
            result = LN_X.eval(x=x)
            assert abs(result.real - np.log(x)) < 1e-12

    def test_ln_p_matches_numpy(self):
        result = LN_P.eval(p=float(P_DEFAULT))
        assert abs(result.real - np.log(P_DEFAULT)) < 1e-12

    def test_var_raises_on_missing_variable(self):
        with pytest.raises(KeyError):
            EXP_X.eval()   # 'x' not provided


# ─────────────────────────────────────────────────────────────────────────────
# §5  BPR fine-structure via EML
# ─────────────────────────────────────────────────────────────────────────────

class TestBPRAlphaViaEML:
    def setup_method(self):
        self.result = bpr_alpha_via_eml()

    def test_ln_p_machine_precision(self):
        assert self.result["ln_p_error"] < 1e-12

    def test_inv_alpha_matches_reference(self):
        assert self.result["inv_alpha_error"] < 1e-10

    def test_inv_alpha_close_to_137(self):
        assert abs(self.result["inv_alpha_eml"] - 137.036) / 137.036 < 0.001

    def test_alpha_eml_is_reciprocal(self):
        assert abs(self.result["alpha_eml"] - 1.0 / self.result["inv_alpha_eml"]) < 1e-14

    def test_tree_depth_reported(self):
        assert self.result["ln_p_tree_depth"] == 3

    def test_tree_K_reported(self):
        assert self.result["ln_p_tree_K"] == 7

    def test_custom_prime(self):
        r = bpr_alpha_via_eml(p=104761, z=6)
        assert r["ln_p_error"] < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# §6  Identity sweep
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifyIdentities:
    def test_all_pass(self):
        result = verify_eml_identities(n_samples=100)
        assert result["all_pass"], (
            f"EML identity failed: {result}"
        )

    def test_individual_errors_small(self):
        result = verify_eml_identities()
        assert result["e_constant_error"] < 1e-13
        assert result["exp_max_error"] < 1e-13
        assert result["zero_error"] < 1e-13
        assert result["ln_max_error"] < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# §7  X_SQ — exact EML tree for x²  (K=17, found by exhaustive enumeration)
# ─────────────────────────────────────────────────────────────────────────────

class TestXSqTree:
    def test_depth(self):
        assert X_SQ.depth() == 8

    def test_K(self):
        assert X_SQ.rpn_length() == 17

    def test_values_machine_precision(self):
        for x in [0.5, 1.0, 2.0, 3.0, np.e]:
            result = X_SQ.eval(x=x)
            assert abs(result.real - x ** 2) < 1e-12, (
                f"X_SQ({x}) = {result.real}, expected {x**2}"
            )

    def test_one_is_fixed_point(self):
        # x=1: x²=1
        assert abs(X_SQ.eval(x=1.0).real - 1.0) < 1e-12

    def test_leaf_count(self):
        # K = 2*leaves − 1  →  17 = 2*9 − 1
        assert X_SQ.leaf_count() == 9


# ─────────────────────────────────────────────────────────────────────────────
# §8  LN_X_SQ — exact EML tree for [ln(x)]²  (K=29, composed X_SQ ∘ LN_X)
# ─────────────────────────────────────────────────────────────────────────────

class TestLnXSqTree:
    def test_depth(self):
        assert LN_X_SQ.depth() == 11

    def test_K(self):
        assert LN_X_SQ.rpn_length() == 29

    def test_values_machine_precision(self):
        for x in [0.5, 1.5, 2.0, 3.0, np.e, np.pi]:
            result = LN_X_SQ.eval(x=x)
            expected = np.log(x) ** 2
            assert abs(result.real - expected) < 1e-12, (
                f"LN_X_SQ({x}) = {result.real}, expected {expected}"
            )

    def test_at_one_is_zero(self):
        # x=1: ln(1)=0, [ln(1)]²=0
        assert abs(LN_X_SQ.eval(x=1.0).real) < 1e-12

    def test_leaf_count(self):
        # K = 2*leaves − 1  →  29 = 2*15 − 1
        assert LN_X_SQ.leaf_count() == 15
