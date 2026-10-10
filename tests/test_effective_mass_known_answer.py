"""Known-answer test for the effective-mass estimators used by the glueball and gauge Monte Carlo code.

The open classical O(3) (Heisenberg) chain with energy -beta * sum_i s_i . s_{i+1} is exactly solvable
(Fisher, Am. J. Phys. 32 (1964) 343): successive relative orientations are independent, and
<s_i . s_{i+r}> = u^r with u = coth(beta) - 1/beta (the Langevin function), so the mass is m = -ln u at every
separation, with no excited-state contamination. At beta = 2, m = 0.62118.

Configurations are drawn exactly (no Markov chain), so the only error is statistical.
"""

import numpy as np
import pytest

from bpr.gauge_mc_fast import gevp_effective_masses
from bpr.glueball_channels_mc import jackknife_effective_mass

BETA = 2.0
EXACT_MASS = -np.log(1.0 / np.tanh(BETA) - 1.0 / BETA)


def o3_chain(n_cfg, n_sites, beta, seed):
    """Exact samples of the open O(3) chain: each spin is drawn from exp(beta s . s_prev) around its neighbour."""
    rng = np.random.default_rng(seed)
    spins = np.empty((n_cfg, n_sites, 3))
    v = rng.standard_normal((n_cfg, 3))
    spins[:, 0] = v / np.linalg.norm(v, axis=1, keepdims=True)
    for i in range(1, n_sites):
        prev = spins[:, i - 1]
        # cos(theta) from p(w) ~ exp(beta w) on [-1, 1], by inverse CDF
        w = 1.0 + np.log(rng.random(n_cfg) * (1.0 - np.exp(-2.0 * beta)) + np.exp(-2.0 * beta)) / beta
        phi = rng.uniform(0.0, 2.0 * np.pi, n_cfg)
        a = np.where(np.abs(prev[:, :1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
        e1 = np.cross(prev, a)
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(prev, e1)
        sin_t = np.sqrt(np.clip(1.0 - w ** 2, 0.0, None))
        spins[:, i] = (w[:, None] * prev + (sin_t * np.cos(phi))[:, None] * e1
                       + (sin_t * np.sin(phi))[:, None] * e2)
    return spins


@pytest.fixture(scope="module")
def chain():
    return o3_chain(n_cfg=400, n_sites=64, beta=BETA, seed=20261010)


def test_exact_mass_value():
    assert EXACT_MASS == pytest.approx(0.62118, abs=1e-5)


def test_sampler_reproduces_the_langevin_function(chain):
    nn = np.einsum("cij,cij->ci", chain[:, :-1], chain[:, 1:]).mean()
    assert nn == pytest.approx(np.exp(-EXACT_MASS), abs=0.01)


def test_jackknife_effective_mass_known_answer(chain):
    n_t = 6
    c = np.stack([np.einsum("cij,cij->c", chain[:, :-t or None], chain[:, t:]) / (chain.shape[1] - t)
                  for t in range(n_t)], axis=1)
    out = jackknife_effective_mass(c)
    for row in out[:4]:
        assert row is not None
        assert abs(row["m_eff"] - EXACT_MASS) < 3 * row["err"]


def test_gevp_effective_masses_known_answer(chain):
    """3x3 correlator of the spin components: C_ab(t) = delta_ab u^t / 3, lowest mass -ln u. Only t <= 2 is
    checked: the three states are exactly degenerate, so taking the largest noisy eigenvalue biases m_eff low once
    the noise is comparable to the signal (t >= 3 here)."""
    n_t = 12
    C = np.zeros((chain.shape[0], 3, 3, n_t))
    for t in range(n_t):
        a, b = chain[:, :chain.shape[1] - t], chain[:, t:]
        C[..., t] = np.einsum("cia,cib->cab", a, b) / (chain.shape[1] - t)
    m = gevp_effective_masses(C, t0=1, n_blocks=40)
    for t in range(0, 3):
        assert abs(m["m_eff"][t] - EXACT_MASS) < 3 * m["err"][t]
