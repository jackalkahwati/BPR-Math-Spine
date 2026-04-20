"""
BPR-Math-Spine: Minimal, Reproducible Maths for Boundary Phase Resonance

A laser-focused codebase that reproduces every numbered equation in the
one-page synopsis, generates the falsifiable Casimir-deviation curve, and
now implements **36 theories + cross-theory pipelines** (Al-Kahwati, 2026):

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
    XXII  Coherence Transitions & Symbolic Meaning  (bpr.coherence_transitions)
    XXIII Meta-Boundary Dynamics                    (bpr.meta_boundary)
    XXIV  RPST Extensions (BPR/RPST)                (bpr.rpst_extensions)
    XXV   RPST Stability Manifolds                  (bpr.stability_manifolds)
    XXVI  Functional Architecture of Reality        (bpr.functional_architecture)
    XXVII  TDGL BPR Solver                           (bpr.tdgl_bpr)
    XXVIII Hilbert Space BPR Operator                (bpr.hilbert_bpr)
    XXIX   Fractional Boundary Resonance Index       (bpr.fractional_boundary)
    XXX    Plasmoid Boundary-Phase Confinement       (bpr.plasmoid)
    XXXI   Resonance Families & Quasi-Integers       (bpr.resonance_families)
    XXXII  NP-Hard BPR Optimization                  (bpr.optimization)
    XXXIII BPR Fluid Dynamics                        (bpr.fluid_dynamics)
    XXXIV  Resonance Algebra PDE Rulebook            (bpr.resonance_algebra)
    XXXV   Electromechanical Coherence               (bpr.electromechanical)
    XXXVI  Conscious Agents Markov Bridge            (bpr.conscious_agents)

Based on the mathematical framework by Jack Al-Kahwati
Contact: jack@thestardrive.com

License: MIT
"""

__version__ = "1.0.0"
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
    from . import memory            # Boundary Memory Dynamics    # noqa: F401
    from . import impedance         # Vacuum Impedance Mismatch   # noqa: F401
    from . import decoherence       # Boundary-Induced Decoherence  # noqa: F401
    from . import phase_transitions # Universal Phase Transition Taxonomy   # noqa: F401
    from . import neutrino          # Boundary-Mediated Neutrino Dynamics    # noqa: F401
    from . import info_geometry     # Substrate Information Geometry   # noqa: F401
    from . import gravitational_waves  # Gravitational Wave Phenomenology  # noqa: F401
    from . import complexity        # Substrate Complexity # noqa: F401
    from . import bioelectric       # Bioelectric Substrate Coupling   # noqa: F401
    from . import collective        # Resonant Collective Dynamics    # noqa: F401
    from . import black_hole        # BH entropy  # noqa: F401
    from . import cosmology         # BPR Cosmology & Early Universe   # noqa: F401
    from . import qcd_flavor        # QCD & Flavor Physics  # noqa: F401
    from . import emergent_spacetime  # Emergent Spacetime & Holography  # noqa: F401
    from . import topological_matter  # Topological Condensed Matter  # noqa: F401
    from . import clifford_bpr      # Clifford Algebra Embedding   # noqa: F401
    from . import quantum_foundations  # Quantum Foundations  # noqa: F401
    from . import gauge_unification   # Gauge Unification & Hierarchy # noqa: F401
    from . import charged_leptons     # Charged Lepton Masses  # noqa: F401
    from . import nuclear_physics     # Nuclear Physics from Boundary Shell  # noqa: F401
    from . import quantum_gravity_pheno  # Quantum Gravity Phenomenology  # noqa: F401
    from . import quantum_chemistry   # Quantum Chemistry & Periodic Table  # noqa: F401
    from . import coherence_transitions  # Invariant Structure, Boundary Dynamics, and Symbolic Meaning  # noqa: F401
    from . import meta_boundary          # Meta-Boundary Dynamics  # noqa: F401
    from . import rpst_extensions        # Emergent Physics from Prime Substrates  # noqa: F401
    from . import stability_manifolds    # RPST Stability Manifolds  # noqa: F401
    from . import functional_architecture  # Functional Architecture of Reality  # noqa: F401

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

# Energy balance / thermodynamic accounting (gates all energy claims)
try:
    from .energy_balance import (  # noqa: F401
        BoundaryStateCycle,
        CycleEnergyBalance,
        FreeEnergy,
        generate_null_configurations,
        run_full_analysis,
    )
    ENERGY_BALANCE_AVAILABLE = True
except Exception:
    ENERGY_BALANCE_AVAILABLE = False

__all__.append("ENERGY_BALANCE_AVAILABLE")

# Substrate heat flow / steady-state thermal reservoir hypothesis
try:
    from .substrate_heat_flow import (  # noqa: F401
        SubstrateBath,
        BoundaryCoupling,
        HeatFlowResult,
        compute_Teff,
        compute_heat_flow,
        compute_all_methods,
        generate_nulls_for_heat_flow,
        generate_scaling_predictions,
        run_full_analysis as run_heat_flow_analysis,
    )
    SUBSTRATE_HEAT_FLOW_AVAILABLE = True
except Exception:
    SUBSTRATE_HEAT_FLOW_AVAILABLE = False

__all__.append("SUBSTRATE_HEAT_FLOW_AVAILABLE")

# ──────────────────────────────────────────────────────────────────────────
# Paper-Derived Extensions (April 2026)
# ──────────────────────────────────────────────────────────────────────────
# 10 new modules from gap analysis of 141 research papers

try:
    from . import tdgl_bpr              # TDGL BPR Solver (founding paper)       # noqa: F401
    from . import hilbert_bpr           # Hilbert Space BPR Operator             # noqa: F401
    from . import fractional_boundary   # Fractional Boundary Resonance Index    # noqa: F401
    from . import plasmoid              # Plasmoid Boundary-Phase Confinement    # noqa: F401
    from . import resonance_families    # Quasi-Integers & Farey Tree            # noqa: F401
    from . import optimization          # NP-Hard BPR Solver (Max-Cut)           # noqa: F401
    from . import fluid_dynamics        # BPR Stress in Fluids                   # noqa: F401
    from . import resonance_algebra     # Fused PDE Operator Rulebook            # noqa: F401
    from . import electromechanical     # Flexoelectric/Piezoelectric BPR        # noqa: F401
    from . import conscious_agents      # Hoffman-BPR Markov Bridge              # noqa: F401

    PAPER_EXTENSIONS_AVAILABLE = True
except Exception:
    PAPER_EXTENSIONS_AVAILABLE = False

__all__.append("PAPER_EXTENSIONS_AVAILABLE")

# EML–BPR synthesis (Odrzywolek 2026 × Al-Kahwati 2026)
try:
    from . import eml                   # All physics from one operator          # noqa: F401
    EML_AVAILABLE = True
except Exception:
    EML_AVAILABLE = False

__all__.append("EML_AVAILABLE")

# ──────────────────────────────────────────────────────────────────────────
# Cross-Theory Interpolation Layer (April 2026)
# ──────────────────────────────────────────────────────────────────────────

try:
    from . import constants             # Canonical physical constants          # noqa: F401
    from . import boundary_action       # Master Boundary Action (Rosetta Stone)  # noqa: F401
    from . import multiscale            # Cross-Scale Coherence Propagation     # noqa: F401
    from . import pipelines             # End-to-End Prediction Pipelines       # noqa: F401

    INTERPOLATION_AVAILABLE = True
except Exception:
    INTERPOLATION_AVAILABLE = False

__all__.append("INTERPOLATION_AVAILABLE")