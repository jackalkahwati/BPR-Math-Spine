"""
BPR-Math-Spine: Minimal, Reproducible Maths for Boundary Phase Resonance

A laser-focused codebase that reproduces every numbered equation in the
one-page synopsis, generates the falsifiable Casimir-deviation curve, and
now implements **21 theories** (Al-Kahwati, 2026):

    I     Boundary Memory Dynamics           (bpr.memory)
    II    Vacuum Impedance Mismatch          (bpr.impedance)
    III   Boundary-Induced Decoherence       (bpr.decoherence)
    IV    Universal Phase Transition Taxonomy (bpr.phase_transitions)
    V     Boundary-Mediated Neutrino Dynamics (bpr.neutrino)
    VI    Substrate Information Geometry      (bpr.info_geometry)
    VII   Gravitational Wave Phenomenology    (bpr.gravitational_waves)
    VIII  Substrate Complexity Theory         (bpr.complexity)
    IX    Bioelectric Substrate Coupling      (bpr.bioelectric)
    X     Resonant Collective Dynamics        (bpr.collective)
    XI    Cosmology & Early Universe          (bpr.cosmology)
    XII   QCD & Flavor Physics               (bpr.qcd_flavor)
    XIII  Emergent Spacetime & Holography     (bpr.emergent_spacetime)
    XIV   Topological Condensed Matter        (bpr.topological_matter)
    XV    Clifford Algebra Embedding          (bpr.clifford_bpr)
    XVI   Quantum Foundations                 (bpr.quantum_foundations)
    XVII  Gauge Unification & Hierarchy       (bpr.gauge_unification)
    XVIII Charged Lepton Masses               (bpr.charged_leptons)
    XIX   Nuclear Physics & Shell Structure   (bpr.nuclear_physics)
    XX    Quantum Gravity Phenomenology       (bpr.quantum_gravity_pheno)
    XXI   Quantum Chemistry & Periodic Table  (bpr.quantum_chemistry)

Based on the mathematical framework by Jack Al-Kahwati
Contact: jack@thestardrive.com

License: MIT
"""

__version__ = "0.7.0"
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
    "sweep_radius",
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

# ──────────────────────────────────────────────────────────────────────────
# Ten Adjacent Theories  (Feb 2026)
# ──────────────────────────────────────────────────────────────────────────

try:
    from . import memory            # Theory I    # noqa: F401
    from . import impedance         # Theory II   # noqa: F401
    from . import decoherence       # Theory III  # noqa: F401
    from . import phase_transitions # Theory IV   # noqa: F401
    from . import neutrino          # Theory V    # noqa: F401
    from . import info_geometry     # Theory VI   # noqa: F401
    from . import gravitational_waves  # Theory VII  # noqa: F401
    from . import complexity        # Theory VIII # noqa: F401
    from . import bioelectric       # Theory IX   # noqa: F401
    from . import collective        # Theory X    # noqa: F401
    from . import black_hole        # BH entropy  # noqa: F401
    from . import cosmology         # Theory XI   # noqa: F401
    from . import qcd_flavor        # Theory XII  # noqa: F401
    from . import emergent_spacetime  # Theory XIII  # noqa: F401
    from . import topological_matter  # Theory XIV  # noqa: F401
    from . import clifford_bpr      # Theory XV   # noqa: F401
    from . import quantum_foundations  # Theory XVI  # noqa: F401
    from . import gauge_unification   # Theory XVII # noqa: F401
    from . import charged_leptons     # Theory XVIII# noqa: F401
    from . import nuclear_physics     # Theory XIX  # noqa: F401
    from . import quantum_gravity_pheno  # Theory XX  # noqa: F401
    from . import quantum_chemistry   # Theory XXI  # noqa: F401

    ADJACENT_THEORIES_AVAILABLE = True
except Exception:
    ADJACENT_THEORIES_AVAILABLE = False

# First-principles coupling pipeline (requires adjacent theories)
try:
    from .first_principles import SubstrateDerivedTheories  # noqa: F401
    FIRST_PRINCIPLES_AVAILABLE = True
except Exception:
    FIRST_PRINCIPLES_AVAILABLE = False

__all__.extend(["ADJACENT_THEORIES_AVAILABLE", "FIRST_PRINCIPLES_AVAILABLE"])