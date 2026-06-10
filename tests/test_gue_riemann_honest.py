"""Honest assessment of the RPST GUE / Riemann-zeros conjecture.

Documents (and locks in) the finding that the RPST Hamiltonian as implemented
does NOT exhibit GUE level statistics, because it is rank-1 by construction.
This prevents the overclaim in doc/conjectures/riemann_connection.md from
silently persisting.
"""
import numpy as np
from bpr.rpst.hamiltonian import RPSTHamiltonian


def test_rpst_hamiltonian_is_rank_one():
    """H = outer(leg, leg) is rank-1: exactly one nonzero eigenvalue.

    A rank-1 Hamiltonian CANNOT exhibit GUE level-spacing statistics
    (those require a full spectrum of repelling levels). The GUE claim is
    therefore not supported by this operator.
    """
    H = RPSTHamiltonian(211)
    M = H.build_hamiltonian()
    rank = np.linalg.matrix_rank(M, tol=1e-9)
    assert rank == 1


def test_normalized_eigenvalues_too_few_for_gue():
    """Only one nonzero eigenvalue survives — far below the GUE sample need."""
    H = RPSTHamiltonian(211)
    eigs = H.normalized_eigenvalues()
    assert eigs.size < 10  # cannot do level-spacing statistics
