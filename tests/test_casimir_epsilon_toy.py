"""Tests for the TOY ε estimator (model-dependent; speculative branch)."""
import numpy as np
import pytest

from bpr.casimir_epsilon_toy import (
    epsilon, unsuppressed_epsilon, monte_carlo, classify, CASIMIR_BOUND,
)


def test_epsilon_is_product():
    assert epsilon(0.1, 0.5, 1e-10) == pytest.approx(0.1 * 0.5 * 1e-10)


def test_unsuppressed_epsilon_is_at_or_above_casimir_bound():
    # the toy's one real constraint: S~1 => epsilon near/above the bound
    lo, hi = unsuppressed_epsilon()
    assert hi >= CASIMIR_BOUND          # unsuppressed coupling would be excluded


def test_classify_fractions_sum_to_one():
    eps = monte_carlo(20_000, seed=1)
    c = classify(eps)
    total = c["excluded"] + c["liftable_window"] + c["no_lift"]
    assert total == pytest.approx(1.0, abs=1e-9)


def test_epsilon_spans_many_orders():
    # honest signature: dominated by the unknown S, so it spans a huge range
    eps = monte_carlo(50_000, seed=2)
    span = np.log10(eps.max()) - np.log10(eps.min())
    assert span > 20                     # not a prediction; an undetermined span
