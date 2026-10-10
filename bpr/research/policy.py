"""Autonomy policy and sandbox boundaries for the research loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath


@dataclass(frozen=True)
class SandboxTarget:
    """A narrow theory area eligible for autonomous iteration."""

    slug: str
    module_path: str
    description: str
    related_prediction_ids: tuple[str, ...]
    rationale: str
    acceptance_commands: tuple[str, ...] = ()


@dataclass(frozen=True)
class AutonomyPolicy:
    """Machine-readable safety boundary for the continuous loop."""

    evaluator_paths: tuple[str, ...]
    append_only_paths: tuple[str, ...]
    sandbox_targets: tuple[SandboxTarget, ...]
    flagship_prediction_ids: tuple[str, ...]
    query_hints: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def is_evaluator_path(self, path: str) -> bool:
        """Return True when a path is part of the frozen evaluation policy."""
        normalized = PurePosixPath(path).as_posix().lstrip("/")
        return normalized in {
            PurePosixPath(item).as_posix().lstrip("/") for item in self.evaluator_paths
        }

    def primary_sandbox(self) -> SandboxTarget:
        """Return the default first sandbox target."""
        return self.sandbox_targets[0]


DEFAULT_AUTONOMY_POLICY = AutonomyPolicy(
    evaluator_paths=(
        "bpr/experimental_data.py",
        "scripts/benchmark_predictions.py",
        "experiments/validate_all_theories.py",
        "tests/test_benchmark_regression.py",
        "tests/test_consistency_cross_validation.py",
        "VALIDATION_STATUS.md",
        "doc/LIMITATIONS_AND_FALSIFICATION.md",
        "doc/EXPERIMENTAL_ROADMAP.md",
    ),
    append_only_paths=(
        "doc/experiments/papers.md",
        "data/research/audit/research_events.jsonl",
    ),
    sandbox_targets=(
        SandboxTarget(
            slug="charged_leptons",
            module_path="bpr/charged_leptons.py",
            description=(
                "Improve charged-lepton mass derivation without touching evaluator "
                "policy or cross-theory scoring rules."
            ),
            related_prediction_ids=(
                "P18.1_m_electron_MeV",
                "P18.2_m_muon_MeV",
                "P18.3_m_tau_MeV",
                "P18.4_koide_parameter",
            ),
            rationale=(
                "This module has a clear local objective, existing regression "
                "coverage, and a visible derived discrepancy in the muon mass."
            ),
            acceptance_commands=(
                "pytest -q tests/test_benchmark_regression.py -k 'electron or muon or tau or koide'",
                "python3 scripts/benchmark_predictions.py",
            ),
        ),
    ),
    # P20.7_LI_delta_c_over_c was removed as a flagship (2026-10): the
    # exp(-p^{1/3}) speed-of-light variation is stipulated, superseded by the
    # BPR-6D architecture decision (section 4), and withdrawn as a prediction
    # in the 2026-10 review of unmerged branches (doc/CLOSED_AND_DEPRECATED.md).
    # Its registry row and query hints remain, so an explicit scan still works.
    flagship_prediction_ids=(
        "P4.7_Tc_niobium_K",
        "P4.9_Tc_MgB2_K",
        "P5.8_delta_m21_sq_eV2",
        "P18.2_m_muon_MeV",
        # P22.1 is a calibration, not a prediction (2026-10): p = 104761 is
        # chosen to reproduce alpha, and the formula then gives 1/alpha =
        # 137.0386 (+19.3 ppm), not the 137.031 quoted earlier (the p = 104729
        # arithmetic).  It stays in the list because the research loop and
        # data/research/evaluator_boundary.json pin this tuple; scans of it
        # monitor the calibration, they do not test BPR.  Found in the
        # 2026-10 review of unmerged branches.
        "P22.1_inv_alpha_0",
    ),
    query_hints={
        # Legacy, not a flagship (see above); kept for explicit scans.
        "P20.7_LI_delta_c_over_c": (
            "Lorentz invariance violation",
            "gamma ray burst timing",
            "GRB 221009A",
        ),
        "P4.9_Tc_MgB2_K": (
            "MgB2 superconductivity",
            "transition temperature",
            "magnesium diboride",
        ),
        "P4.7_Tc_niobium_K": (
            "niobium superconducting transition temperature",
            "Tc niobium",
        ),
        "P18.2_m_muon_MeV": (
            "muon mass precision",
            "charged lepton mass ratios",
        ),
    },
)
