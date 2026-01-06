"""
BPR-Math-Spine: Minimal, Reproducible Maths for Boundary Phase Resonance

A laser-focused codebase that reproduces every numbered equation in the 
one-page synopsis and generates the falsifiable Casimir-deviation curve.

Based on the mathematical framework by Jack Al-Kahwati (July 2025)
Contact: jack@thestardrive.com

License: MIT
"""

__version__ = "0.1.0"
__author__ = "Jack Al-Kahwati / StarDrive Research Group"

from .geometry import make_boundary
from .boundary_field import solve_phase
from .metric import metric_perturbation
from .casimir import casimir_force, sweep_radius

# Import information integration and consciousness coupling
try:
    from .information import InformationIntegration, ConsciousnessCoupling, placeholder_consciousness_coupling
    INFORMATION_AVAILABLE = True
except ImportError:
    INFORMATION_AVAILABLE = False

__all__ = [
    "make_boundary",
    "solve_phase", 
    "metric_perturbation",
    "casimir_force",
    "sweep_radius"
]

if INFORMATION_AVAILABLE:
    __all__.extend(["InformationIntegration", "ConsciousnessCoupling", "placeholder_consciousness_coupling"])

# Optional: RPST / resonance / HKLL scaffolding
try:
    from . import rpst  # noqa: F401

    RPST_AVAILABLE = True
except Exception:
    RPST_AVAILABLE = False

try:
    from . import resonance  # noqa: F401

    RESONANCE_AVAILABLE = True
except Exception:
    RESONANCE_AVAILABLE = False

try:
    from . import hkll  # noqa: F401

    HKLL_AVAILABLE = True
except Exception:
    HKLL_AVAILABLE = False

__all__.extend(["RPST_AVAILABLE", "RESONANCE_AVAILABLE", "HKLL_AVAILABLE"])