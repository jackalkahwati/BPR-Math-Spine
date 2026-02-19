"""
Test suite for BPR-Math-Spine

This module contains the three mathematical checkpoints:
1. Laplacian eigenvalues converge to l(l+1) within 0.1% for l≤10
2. Energy-momentum conservation ∇^μ T^φ_μν = 0 to tolerance 1e-8  
3. Recovery of standard Casimir force for λ→0

Run all tests with: pytest -q
Run specific checkpoint: python tests/test_boundary.py

Note: Checkpoint functions are lazily imported to avoid loading numpy/FEniCS
in test_boundary/test_metric/test_casimir during collection of other tests.
This prevents numpy _mac_os_check segfaults on some macOS + Python 3.9 setups.
"""

__all__ = [
    'test_mathematical_checkpoint_1',
    'test_mathematical_checkpoint_2',
    'test_mathematical_checkpoint_3',
    'test_equation_7_implementation',
]


def __getattr__(name: str):
    """Lazy import of checkpoint tests to avoid numpy segfault on package load."""
    if name == 'test_mathematical_checkpoint_1':
        from .test_boundary import test_mathematical_checkpoint_1
        return test_mathematical_checkpoint_1
    if name == 'test_mathematical_checkpoint_2':
        from .test_metric import test_mathematical_checkpoint_2
        return test_mathematical_checkpoint_2
    if name == 'test_mathematical_checkpoint_3':
        from .test_casimir import test_mathematical_checkpoint_3
        return test_mathematical_checkpoint_3
    if name == 'test_equation_7_implementation':
        from .test_casimir import test_equation_7_implementation
        return test_equation_7_implementation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")