"""
Test suite for BPR-Math-Spine

This module contains the three mathematical checkpoints:
1. Laplacian eigenvalues converge to l(l+1) within 0.1% for l≤10
2. Energy-momentum conservation ∇^μ T^φ_μν = 0 to tolerance 1e-8  
3. Recovery of standard Casimir force for λ→0

Run all tests with: pytest -q
Run specific checkpoint: python tests/test_boundary.py
"""

from .test_boundary import test_mathematical_checkpoint_1
from .test_metric import test_mathematical_checkpoint_2  
from .test_casimir import test_mathematical_checkpoint_3, test_equation_7_implementation

__all__ = [
    'test_mathematical_checkpoint_1',
    'test_mathematical_checkpoint_2', 
    'test_mathematical_checkpoint_3',
    'test_equation_7_implementation'
]