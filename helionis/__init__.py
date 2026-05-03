"""Helionis D-He3 reactor architecture screening toolkit.

This package is an order-of-magnitude trade-study MVP. It is built for
transparent scenario comparison, not high-fidelity plasma simulation.
"""

from helionis.architecture import Scenario, TradeStudyResult, evaluate_scenario
from helionis.architecture_comparison_v07 import (
    ArchitectureV07Result,
    run_architecture_v07_comparison,
)
from helionis.bpr_coupled_v09 import BPRCoupledV09Result, run_bpr_coupled_v09_sweep
from helionis.cad_envelope_v11 import CADEnvelopeV11Result, run_cad_envelope_v11
from helionis.collector_nozzle_thermal_v13 import (
    CollectorNozzleThermalV13Result,
    run_collector_nozzle_thermal_v13,
)
from helionis.engineering_net import EngineeringNetResult, evaluate_engineering_net
from helionis.geometry_engineering import (
    GeometryEngineeringResult,
    run_geometry_engineering_rescore,
)
from helionis.geometry import GeometryCandidate, GeometryScore, rank_geometry_candidates
from helionis.margin_recovery_v08 import (
    MarginRecoveryV08Result,
    minimum_recovery_recipe,
    run_margin_recovery_v08_sweep,
)
from helionis.modulus_fusion_control import (
    ModulusFusionControlResult,
    run_modulus_fusion_control_twin,
)
from helionis.mirror_nozzle import MirrorNozzleResult, run_mirror_nozzle_sweep
from helionis.mirror_nozzle_v06 import (
    MirrorNozzleV06Result,
    run_mirror_nozzle_v06_sweep,
)
from helionis.reference_design import ReferenceDesignResult, best_reference_design
from helionis.reactions import D_D_AVERAGE, D_HE3, D_T, Reaction
from helionis.scale_sweep import ScaleSweepResult, minimum_viable_scale
from helionis.thermal_recovery_v12 import (
    ThermalRecoveryV12Result,
    run_thermal_recovery_v12,
)

__all__ = [
    "D_D_AVERAGE",
    "D_HE3",
    "D_T",
    "EngineeringNetResult",
    "ArchitectureV07Result",
    "BPRCoupledV09Result",
    "CADEnvelopeV11Result",
    "CollectorNozzleThermalV13Result",
    "GeometryCandidate",
    "GeometryEngineeringResult",
    "GeometryScore",
    "MarginRecoveryV08Result",
    "MirrorNozzleResult",
    "MirrorNozzleV06Result",
    "ModulusFusionControlResult",
    "ReferenceDesignResult",
    "Reaction",
    "ScaleSweepResult",
    "Scenario",
    "ThermalRecoveryV12Result",
    "TradeStudyResult",
    "evaluate_scenario",
    "best_reference_design",
    "evaluate_engineering_net",
    "minimum_viable_scale",
    "minimum_recovery_recipe",
    "rank_geometry_candidates",
    "run_architecture_v07_comparison",
    "run_bpr_coupled_v09_sweep",
    "run_cad_envelope_v11",
    "run_collector_nozzle_thermal_v13",
    "run_geometry_engineering_rescore",
    "run_mirror_nozzle_sweep",
    "run_mirror_nozzle_v06_sweep",
    "run_margin_recovery_v08_sweep",
    "run_modulus_fusion_control_twin",
    "run_thermal_recovery_v12",
]

__version__ = "0.1.0"
