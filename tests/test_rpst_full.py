import numpy as np
import pytest


def test_rpst_prime_field_closure():
    from bpr.rpst import PrimeField

    p = 17
    field = PrimeField(p)
    a = np.array([15, 16, 0, 1])
    b = np.array([5, 5, 5, 5])
    out = field.add(a, b)
    assert np.all(out >= 0)
    assert np.all(out < p)


def test_rpst_symplectic_reversibility():
    from bpr.rpst import SubstrateState, SymplecticEvolution

    p = 101
    N = 20
    rng = np.random.default_rng(0)
    state = SubstrateState(q=rng.integers(0, p, size=N), pi=rng.integers(0, p, size=N), p=p)
    evo = SymplecticEvolution(p)
    assert evo.verify_reversibility(state, steps=50)


def test_rpst_winding_trivial_and_single():
    from bpr.rpst import compute_winding_number

    p = 31
    q = np.ones(10, dtype=int) * 5
    loop = list(range(10))
    assert compute_winding_number(q, loop, p).winding == 0

    q2 = np.arange(p, dtype=int) % p
    loop2 = list(range(p))
    assert compute_winding_number(q2, loop2, p).winding == 1


def test_rpst_hamiltonian_basic():
    from bpr.rpst import RPSTHamiltonian

    H = RPSTHamiltonian(p=31)
    M = H.build_hamiltonian()
    assert M.shape == (31, 31)
    assert np.allclose(M, M.T)
    eigs = H.eigenvalues()
    assert eigs.shape == (31,)


