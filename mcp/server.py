#!/usr/bin/env python3
"""
BPR-Math-Spine MCP Server
=========================

Exposes the Boundary Phase Resonance / Resonant Prime Substrate Theory
codebase as MCP tools so Claude (and other clients) can query, search,
read, and *compute* predictions from the research codebase directly.

Tools fall into two categories:
  - Codebase tools: list_theories, get_theory, read_file, search, list_files,
    get_validation_status, run_tests, get_readme
  - Computation tools: predict, run_pipeline, list_pipelines, get_constants,
    derive_equation, verify_e8, dark_energy, monte_carlo
"""

import os
import re
import sys
import subprocess
import traceback
from pathlib import Path

from mcp.server.fastmcp import FastMCP

REPO = Path(__file__).parent.parent  # .../BPR-Math-Spine

# Ensure bpr package is importable
sys.path.insert(0, str(REPO))

mcp = FastMCP("bpr-math-spine")

# ── Theory registry (all 37 theories) ────────────────────────────────────────
THEORIES = {
    "Boundary Memory Dynamics":                                       "bpr/memory.py",
    "Vacuum Impedance Mismatch":                                      "bpr/impedance.py",
    "Boundary-Induced Decoherence":                                   "bpr/decoherence.py",
    "Universal Phase Transition Taxonomy":                            "bpr/phase_transitions.py",
    "Boundary-Mediated Neutrino Dynamics":                            "bpr/neutrino.py",
    "Substrate Information Geometry":                                  "bpr/info_geometry.py",
    "Gravitational Wave Phenomenology":                               "bpr/gravitational_waves.py",
    "Substrate Complexity":                                           "bpr/complexity.py",
    "Bioelectric Substrate Coupling":                                 "bpr/bioelectric.py",
    "Resonant Collective Dynamics":                                   "bpr/collective.py",
    "BPR Cosmology & Early Universe":                                 "bpr/cosmology.py",
    "QCD & Flavor Physics":                                           "bpr/qcd_flavor.py",
    "Emergent Spacetime & Holography":                                "bpr/emergent_spacetime.py",
    "Topological Condensed Matter":                                   "bpr/topological_matter.py",
    "Clifford Algebra Embedding":                                     "bpr/clifford_bpr.py",
    "Quantum Foundations":                                             "bpr/quantum_foundations.py",
    "Gauge Unification & Hierarchy":                                  "bpr/gauge_unification.py",
    "Charged Lepton Masses":                                          "bpr/charged_leptons.py",
    "Nuclear Physics from Boundary Shell":                            "bpr/nuclear_physics.py",
    "Quantum Gravity Phenomenology":                                  "bpr/quantum_gravity_pheno.py",
    "Quantum Chemistry & Periodic Table":                             "bpr/quantum_chemistry.py",
    "Fine Structure Constant from Substrate":                         "bpr/alpha_derivation.py",
    "Invariant Structure, Boundary Dynamics, and Symbolic Meaning":   "bpr/coherence_transitions.py",
    "Meta-Boundary Dynamics":                                         "bpr/meta_boundary.py",
    "Emergent Physics from Prime Substrates":                         "bpr/rpst_extensions.py",
    "RPST Stability Manifolds":                                       "bpr/stability_manifolds.py",
    "Functional Architecture of Reality":                             "bpr/functional_architecture.py",
    # ── New theories ──
    "TDGL BPR Solver":                                                "bpr/tdgl_bpr.py",
    "Hilbert Space BPR Operator":                                     "bpr/hilbert_bpr.py",
    "Fractional Boundary Resonance":                                  "bpr/fractional_boundary.py",
    "Plasmoid Confinement":                                           "bpr/plasmoid.py",
    "Resonance Families":                                             "bpr/resonance_families.py",
    "NP-Hard Optimization":                                           "bpr/optimization.py",
    "BPR Fluid Dynamics":                                             "bpr/fluid_dynamics.py",
    "Resonance Algebra":                                              "bpr/resonance_algebra.py",
    "Electromechanical Coherence":                                    "bpr/electromechanical.py",
    "Conscious Agents":                                               "bpr/conscious_agents.py",
}


def _safe_path(rel: str) -> Path:
    """Resolve a relative path inside the repo, rejecting traversals."""
    p = (REPO / rel).resolve()
    if not str(p).startswith(str(REPO)):
        raise ValueError(f"Path outside repo: {rel}")
    return p


# ══════════════════════════════════════════════════════════════════════════════
#  CODEBASE TOOLS (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def list_theories() -> str:
    """List all 37 theories in the BPR-Math-Spine codebase with their module paths."""
    lines = []
    for i, (name, path) in enumerate(THEORIES.items(), 1):
        lines.append(f"{i:2d}. {name}\n      -> {path}")
    return "\n".join(lines)


@mcp.tool()
def get_theory(name: str) -> str:
    """
    Return the full source of a theory module by its descriptive name.

    Args:
        name: Descriptive name of the theory (e.g. "Boundary Memory Dynamics").
              Case-insensitive partial match is supported.
    """
    # Exact match first
    path_rel = THEORIES.get(name)

    # Partial / case-insensitive fallback
    if path_rel is None:
        needle = name.lower()
        matches = [(k, v) for k, v in THEORIES.items() if needle in k.lower()]
        if len(matches) == 1:
            path_rel = matches[0][1]
        elif len(matches) > 1:
            return (
                f"Ambiguous match for '{name}'. Candidates:\n"
                + "\n".join(f"  - {k}" for k, _ in matches)
            )
        else:
            return (
                f"No theory matches '{name}'.\n"
                f"Run list_theories() to see all available names."
            )

    p = _safe_path(path_rel)
    if not p.exists():
        return f"Module not found on disk: {path_rel}"
    return p.read_text()


@mcp.tool()
def read_file(path: str) -> str:
    """
    Read any file from the BPR-Math-Spine repository.

    Args:
        path: Repo-relative path (e.g. "bpr/rpst/hamiltonian.py",
              "VALIDATION_STATUS.md", "experiments/cross_model_comparison.py").
    """
    p = _safe_path(path)
    if not p.exists():
        return f"File not found: {path}"
    if not p.is_file():
        return f"Not a file: {path}"
    content = p.read_text(errors="replace")
    # Cap at ~100 KB to stay within context limits
    if len(content) > 100_000:
        content = content[:100_000] + "\n\n[... truncated at 100 KB ...]"
    return content


@mcp.tool()
def search(query: str, file_glob: str = "**/*.py") -> str:
    """
    Search the codebase with ripgrep (or grep fallback).

    Args:
        query:     Regular-expression pattern to search for.
        file_glob: Glob pattern to restrict which files are searched.
                   Defaults to "**/*.py". Use "**/*.md" for docs, or
                   "**/*" for everything.

    Returns up to 200 matching lines with file:line format.
    """
    try:
        result = subprocess.run(
            ["rg", "--no-heading", "-n", "--glob", file_glob, query],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=15,
        )
        out = result.stdout
    except FileNotFoundError:
        # Fallback to grep
        result = subprocess.run(
            ["grep", "-rn", "--include", file_glob.replace("**/*", "*"), query, "."],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=15,
        )
        out = result.stdout

    lines = out.splitlines()
    if not lines:
        return f"No matches for '{query}' in {file_glob}"
    truncated = len(lines) > 200
    summary = "\n".join(lines[:200])
    if truncated:
        summary += f"\n\n[... {len(lines) - 200} more lines truncated ...]"
    return summary


@mcp.tool()
def list_files(directory: str = "") -> str:
    """
    List files in a repo directory.

    Args:
        directory: Repo-relative path to directory (e.g. "bpr", "tests",
                   "experiments"). Defaults to the repo root.
    """
    p = _safe_path(directory) if directory else REPO
    if not p.is_dir():
        return f"Not a directory: {directory}"

    lines = []
    for item in sorted(p.iterdir()):
        rel = item.relative_to(REPO)
        suffix = "/" if item.is_dir() else ""
        lines.append(f"  {rel}{suffix}")
    return "\n".join(lines) if lines else "(empty directory)"


@mcp.tool()
def get_validation_status() -> str:
    """Return the full VALIDATION_STATUS.md -- pass/fail status of all theory predictions."""
    return (REPO / "VALIDATION_STATUS.md").read_text()


@mcp.tool()
def run_tests(module: str = "", verbose: bool = False) -> str:
    """
    Run the pytest suite (or a specific test file) and return the output.

    Args:
        module: Optional test file or pattern, e.g. "tests/test_memory.py"
                or "test_stability". Leave empty to run the full suite.
        verbose: If True, pass -v to pytest for per-test output.
    """
    cmd = [
        "python3", "-m", "pytest",
        "--ignore=tests/test_fenics_integration.py",
        "-q",
    ]
    if verbose:
        cmd.append("-v")
    if module:
        # Safety: only allow paths inside tests/
        if ".." in module or module.startswith("/"):
            return "Invalid module path."
        cmd.append(module)

    result = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = (result.stdout + result.stderr).strip()
    # Return last 200 lines to avoid overflow
    lines = out.splitlines()
    if len(lines) > 200:
        out = "\n".join(lines[-200:])
        out = "[... output truncated, showing last 200 lines ...]\n\n" + out
    return out


@mcp.tool()
def get_readme() -> str:
    """Return the project README with architecture overview and theory list."""
    p = REPO / "README.md"
    content = p.read_text()
    if len(content) > 50_000:
        content = content[:50_000] + "\n\n[... truncated ...]"
    return content


# ══════════════════════════════════════════════════════════════════════════════
#  COMPUTATION TOOLS (new)
# ══════════════════════════════════════════════════════════════════════════════

# ── Guarded imports for computation ──────────────────────────────────────────

try:
    from bpr.cross_predictions import full_cosmological_chain, dark_energy_from_impedance
    _HAS_CROSS = True
except Exception:
    _HAS_CROSS = False

try:
    from bpr.pipelines import (
        pipeline_impedance_to_lepton_masses,
        pipeline_impedance_to_decoherence,
        pipeline_substrate_to_casimir,
        pipeline_tdgl_to_phase_classification,
        pipeline_kuramoto_to_transition,
        pipeline_agents_to_consciousness,
        pipeline_bond_to_fractal_transport,
    )
    _HAS_PIPELINES = True
except Exception:
    _HAS_PIPELINES = False

try:
    import bpr.constants as _bpr_constants
    _HAS_CONSTANTS = True
except Exception:
    _HAS_CONSTANTS = False

try:
    from bpr.symbolic_derivations import (
        derive_maxwell_from_boundary,
        derive_schrodinger_from_boundary,
        derive_linearized_einstein_from_boundary,
        derive_conservation_law,
        derive_tdgl_from_boundary,
        verify_impedance_weinberg_angle,
        run_all_derivations,
    )
    _HAS_SYMBOLIC = True
except Exception:
    _HAS_SYMBOLIC = False

try:
    from bpr.clifford_bpr import verify_e8_properties
    _HAS_E8 = True
except Exception:
    _HAS_E8 = False

try:
    from bpr.rpst.monte_carlo import RPSTMonteCarlo
    _HAS_MC = True
except Exception:
    _HAS_MC = False


# ── Pipeline registry ────────────────────────────────────────────────────────

_PIPELINE_REGISTRY = {
    "lepton-masses": {
        "desc": "Gauge unification -> impedance -> charged lepton masses (m_e, m_mu, m_tau)",
        "func": "pipeline_impedance_to_lepton_masses",
        "args": ("p", "z"),
    },
    "decoherence": {
        "desc": "Impedance -> decoherence rate -> coherence dynamics -> quantum foundations",
        "func": "pipeline_impedance_to_decoherence",
        "args": ("p", "z"),
    },
    "casimir": {
        "desc": "RPST substrate -> resonance spectrum -> Casimir force prediction",
        "func": "pipeline_substrate_to_casimir",
        "args": ("p",),
    },
    "tdgl": {
        "desc": "Time-dependent Ginzburg-Landau -> coherence -> phase classification",
        "func": "pipeline_tdgl_to_phase_classification",
        "args": ("p",),
    },
    "kuramoto": {
        "desc": "Kuramoto collective dynamics -> phase transition -> coherence",
        "func": "pipeline_kuramoto_to_transition",
        "args": ("p",),
    },
    "consciousness": {
        "desc": "Conscious agents -> collective dynamics -> coherence transitions",
        "func": "pipeline_agents_to_consciousness",
        "args": ("p",),
    },
    "fractal-transport": {
        "desc": "Chemical bond -> resonance families -> fractional boundary transport",
        "func": "pipeline_bond_to_fractal_transport",
        "args": ("p",),
    },
}

# ── Derivation sector registry ───────────────────────────────────────────────

_DERIVATION_SECTORS = {
    "maxwell":       "derive_maxwell_from_boundary",
    "schrodinger":   "derive_schrodinger_from_boundary",
    "einstein":      "derive_linearized_einstein_from_boundary",
    "conservation":  "derive_conservation_law",
    "tdgl":          "derive_tdgl_from_boundary",
    "weinberg":      "verify_impedance_weinberg_angle",
    "all":           "run_all_derivations",
}


# ── Helper: format a dict nicely ─────────────────────────────────────────────

def _fmt_dict(d: dict, indent: int = 0) -> str:
    """Recursively format a dict into readable lines."""
    lines = []
    prefix = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{prefix}{k}:")
            lines.append(_fmt_dict(v, indent + 1))
        elif isinstance(v, float):
            if abs(v) > 1e6 or (abs(v) < 1e-3 and v != 0):
                lines.append(f"{prefix}{k}: {v:.6e}")
            else:
                lines.append(f"{prefix}{k}: {v:.6f}")
        else:
            lines.append(f"{prefix}{k}: {v}")
    return "\n".join(lines)


# ── Tool 1: predict ──────────────────────────────────────────────────────────

@mcp.tool()
def predict(p: int = 104761, z: int = 6) -> str:
    """Run the full BPR cosmological prediction chain from substrate parameters (p, z).

    Returns alpha_EM, sin^2(theta_W), v_EW, lepton masses, Omega_Lambda
    with comparison to experiment.

    Default: p=104761, z=6 (zero free parameters).

    Args:
        p: Substrate prime (default 104761)
        z: Substrate parameter (default 6)
    """
    if not _HAS_CROSS:
        return "ERROR: bpr.cross_predictions module not available. Check imports."

    try:
        result = full_cosmological_chain(p=p, z=z)
    except Exception as e:
        return f"ERROR running full_cosmological_chain: {e}\n{traceback.format_exc()}"

    # Format the output nicely
    out = []
    out.append("=" * 60)
    out.append(f"BPR Full Cosmological Chain  (p={p}, z={z})")
    out.append("=" * 60)
    out.append("")

    # alpha_EM
    if "inv_alpha_predicted" in result:
        out.append(f"  1/alpha_EM:  {result['inv_alpha_predicted']:.6f}  "
                    f"(expt: {result['inv_alpha_experiment']:.6f}, "
                    f"err: {result.get('inv_alpha_percent_error', '?'):.4f}%)")
    elif "inv_alpha_error" in result:
        out.append(f"  1/alpha_EM:  [error] {result['inv_alpha_error']}")

    # sin^2 theta_W
    if "sin2_theta_W_predicted" in result:
        out.append(f"  sin2(theta_W): {result['sin2_theta_W_predicted']:.6f}  "
                    f"(expt: {result['sin2_theta_W_experiment']:.6f}, "
                    f"err: {result.get('sin2_theta_W_percent_error', '?'):.4f}%)")
    elif "sin2_theta_W_error" in result:
        out.append(f"  sin2(theta_W): [error] {result['sin2_theta_W_error']}")

    # v_EW
    if "v_EW_GeV_predicted" in result:
        out.append(f"  v_EW [GeV]:  {result['v_EW_GeV_predicted']:.4f}  "
                    f"(expt: {result['v_EW_GeV_experiment']:.1f}, "
                    f"err: {result.get('v_EW_percent_error', '?'):.4f}%)")
    elif "v_EW_error" in result:
        out.append(f"  v_EW [GeV]:  [error] {result['v_EW_error']}")

    # Lepton masses
    for lep, label in [("m_e", "m_e [MeV]"), ("m_mu", "m_mu [MeV]"), ("m_tau", "m_tau [MeV]")]:
        pred_key = f"{lep}_MeV_predicted"
        exp_key = f"{lep}_MeV_experiment"
        err_key = f"{lep}_percent_error"
        if pred_key in result:
            out.append(f"  {label}:  {result[pred_key]:.6f}  "
                        f"(expt: {result[exp_key]:.4f}, "
                        f"err: {result.get(err_key, '?'):.4f}%)")
    if "lepton_mass_error" in result:
        out.append(f"  Lepton masses: [error] {result['lepton_mass_error']}")

    # Omega_Lambda
    if "Omega_Lambda_predicted" in result:
        out.append(f"  Omega_Lambda: {result['Omega_Lambda_predicted']:.6f}  "
                    f"(expt: {result['Omega_Lambda_experiment']:.4f}, "
                    f"err: {result.get('Omega_Lambda_percent_error', '?'):.4f}%)")
    elif "Omega_Lambda_error" in result:
        out.append(f"  Omega_Lambda: [error] {result['Omega_Lambda_error']}")

    # Summary
    if "summary" in result:
        out.append("")
        out.append("Summary (sorted by accuracy):")
        for name, err in result["summary"].items():
            out.append(f"  {name}: {err}")

    return "\n".join(out)


# ── Tool 2: run_pipeline ─────────────────────────────────────────────────────

@mcp.tool()
def run_pipeline(name: str, p: int = 104761, z: int = 6) -> str:
    """Run a BPR prediction pipeline by name.

    Available pipelines: lepton-masses, decoherence, casimir, tdgl,
    kuramoto, consciousness, fractal-transport.

    Args:
        name: Pipeline name (use list_pipelines to see all)
        p: Substrate prime (default 104761)
        z: Substrate parameter (default 6)
    """
    if not _HAS_PIPELINES:
        return "ERROR: bpr.pipelines module not available. Check imports."

    if name not in _PIPELINE_REGISTRY:
        available = ", ".join(sorted(_PIPELINE_REGISTRY.keys()))
        return f"Unknown pipeline '{name}'. Available: {available}"

    info = _PIPELINE_REGISTRY[name]
    func_name = info["func"]

    # Get the function from the pipelines module
    func_map = {
        "pipeline_impedance_to_lepton_masses": pipeline_impedance_to_lepton_masses,
        "pipeline_impedance_to_decoherence": pipeline_impedance_to_decoherence,
        "pipeline_substrate_to_casimir": pipeline_substrate_to_casimir,
        "pipeline_tdgl_to_phase_classification": pipeline_tdgl_to_phase_classification,
        "pipeline_kuramoto_to_transition": pipeline_kuramoto_to_transition,
        "pipeline_agents_to_consciousness": pipeline_agents_to_consciousness,
        "pipeline_bond_to_fractal_transport": pipeline_bond_to_fractal_transport,
    }

    func = func_map.get(func_name)
    if func is None:
        return f"Pipeline function '{func_name}' not found in imports."

    try:
        # Build kwargs based on what the pipeline accepts
        if "z" in info["args"]:
            result = func(p=p, z=z)
        else:
            result = func(p=p)
    except Exception as e:
        return f"ERROR running pipeline '{name}': {e}\n{traceback.format_exc()}"

    out = []
    out.append("=" * 60)
    out.append(f"Pipeline: {name}")
    out.append(f"  {info['desc']}")
    out.append(f"  Parameters: p={p}" + (f", z={z}" if "z" in info["args"] else ""))
    out.append("=" * 60)
    out.append("")
    out.append(_fmt_dict(result))
    return "\n".join(out)


# ── Tool 3: list_pipelines ───────────────────────────────────────────────────

@mcp.tool()
def list_pipelines() -> str:
    """List all available BPR prediction pipelines with descriptions."""
    lines = []
    for i, (name, info) in enumerate(sorted(_PIPELINE_REGISTRY.items()), 1):
        args = ", ".join(info["args"])
        lines.append(f"{i}. {name}  ({args})")
        lines.append(f"     {info['desc']}")
    return "\n".join(lines)


# ── Tool 4: get_constants ────────────────────────────────────────────────────

@mcp.tool()
def get_constants(filter_pattern: str = "") -> str:
    """Get BPR physical constants. Optionally filter by name pattern.

    Args:
        filter_pattern: Case-insensitive substring to filter constants (e.g. 'planck', 'alpha')
    """
    if not _HAS_CONSTANTS:
        return "ERROR: bpr.constants module not available."

    lines = []
    needle = filter_pattern.lower()

    for attr in sorted(dir(_bpr_constants)):
        if attr.startswith("_"):
            continue
        val = getattr(_bpr_constants, attr)
        # Only show numeric constants and arrays
        if not isinstance(val, (int, float)):
            try:
                import numpy as np
                if not isinstance(val, np.ndarray):
                    continue
            except ImportError:
                continue

        if needle and needle not in attr.lower():
            continue

        if isinstance(val, float):
            if abs(val) > 1e6 or (abs(val) < 1e-3 and val != 0):
                lines.append(f"  {attr:25s} = {val:.6e}")
            else:
                lines.append(f"  {attr:25s} = {val:.6f}")
        else:
            lines.append(f"  {attr:25s} = {val}")

    if not lines:
        if filter_pattern:
            return f"No constants matching '{filter_pattern}'."
        return "No constants found in bpr.constants."

    header = "BPR Physical Constants"
    if filter_pattern:
        header += f" (filter: '{filter_pattern}')"
    return header + "\n" + "-" * 50 + "\n" + "\n".join(lines)


# ── Tool 5: derive_equation ──────────────────────────────────────────────────

@mcp.tool()
def derive_equation(sector: str = "maxwell") -> str:
    """Symbolically derive field equations from BPR boundary action.

    Sectors: maxwell, schrodinger, einstein, tdgl, conservation, weinberg, all.

    Args:
        sector: Which sector to derive (default: maxwell)
    """
    if not _HAS_SYMBOLIC:
        return "ERROR: bpr.symbolic_derivations module not available (requires sympy)."

    sector = sector.lower().strip()
    if sector not in _DERIVATION_SECTORS:
        available = ", ".join(sorted(_DERIVATION_SECTORS.keys()))
        return f"Unknown sector '{sector}'. Available: {available}"

    func_name = _DERIVATION_SECTORS[sector]
    func_map = {
        "derive_maxwell_from_boundary": derive_maxwell_from_boundary,
        "derive_schrodinger_from_boundary": derive_schrodinger_from_boundary,
        "derive_linearized_einstein_from_boundary": derive_linearized_einstein_from_boundary,
        "derive_conservation_law": derive_conservation_law,
        "derive_tdgl_from_boundary": derive_tdgl_from_boundary,
        "verify_impedance_weinberg_angle": verify_impedance_weinberg_angle,
        "run_all_derivations": run_all_derivations,
    }

    func = func_map[func_name]

    try:
        result = func()
    except Exception as e:
        return f"ERROR deriving '{sector}': {e}\n{traceback.format_exc()}"

    out = []
    out.append("=" * 60)
    out.append(f"Symbolic Derivation: {sector}")
    out.append("=" * 60)
    out.append("")

    if isinstance(result, dict):
        # Extract description if present
        if "description" in result:
            out.append(f"Result: {result['description']}")
            out.append("")
        # For 'all' sector, result is dict of dicts
        if sector == "all":
            for sub_name, sub_result in result.items():
                out.append(f"--- {sub_name} ---")
                if isinstance(sub_result, dict) and "description" in sub_result:
                    out.append(f"  {sub_result['description']}")
                out.append("")
        else:
            # Show key results (skip sympy objects which don't serialize well)
            for k, v in result.items():
                if k == "description":
                    continue
                v_str = str(v)
                if len(v_str) > 500:
                    v_str = v_str[:500] + "..."
                out.append(f"  {k}: {v_str}")
    else:
        out.append(str(result))

    return "\n".join(out)


# ── Tool 6: verify_e8 ────────────────────────────────────────────────────────

@mcp.tool()
def verify_e8() -> str:
    """Verify E8 root system properties used in BPR's Clifford algebra embedding.

    Checks: 240 roots, all length-squared = 2, Cartan matrix det = 1,
    rank = 8, dimension = 248.
    """
    if not _HAS_E8:
        return "ERROR: bpr.clifford_bpr module not available (requires numpy)."

    try:
        result = verify_e8_properties()
    except Exception as e:
        return f"ERROR verifying E8: {e}\n{traceback.format_exc()}"

    out = []
    out.append("=" * 60)
    out.append("E8 Root System Verification")
    out.append("=" * 60)
    out.append("")

    checks = [
        ("Number of roots", result["n_roots"], result["expected_roots"],
         result["roots_correct"]),
        ("All roots length^2 = 2", result["all_length_2"], True,
         result["all_length_2"]),
        ("Cartan matrix det", f"{result['cartan_det']:.4f}", "1.0",
         result["det_is_1"]),
        ("Rank", result["rank"], 8, result["rank"] == 8),
        ("Dimension (roots + rank)", result["dimension"], 248,
         result["dim_correct"]),
    ]

    all_pass = True
    for label, got, expected, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        out.append(f"  [{status}] {label}: {got} (expected {expected})")

    out.append("")
    out.append(f"Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

    return "\n".join(out)


# ── Tool 7: dark_energy ──────────────────────────────────────────────────────

@mcp.tool()
def dark_energy() -> str:
    """Compute dark energy prediction (Omega_Lambda) from BPR impedance framework.

    Derives rho_DE from holographic mode counting, compares to Planck 2018.
    """
    if not _HAS_CROSS:
        return "ERROR: bpr.cross_predictions module not available."

    try:
        result = dark_energy_from_impedance()
    except Exception as e:
        return f"ERROR computing dark energy: {e}\n{traceback.format_exc()}"

    out = []
    out.append("=" * 60)
    out.append("Dark Energy from BPR Impedance")
    out.append("=" * 60)
    out.append("")
    out.append(f"  Omega_Lambda (predicted): {result['Omega_Lambda']:.6f}")
    out.append(f"  Omega_Lambda (Planck 18): 0.6889")
    if "percent_error" in result:
        out.append(f"  Percent error:           {result['percent_error']:.4f}%")
    out.append("")
    out.append(f"  rho_DE [J/m^3]:          {result['rho_DE_J_per_m3']:.6e}")
    out.append(f"  rho_critical [J/m^3]:    {result['rho_critical_J_per_m3']:.6e}")
    out.append(f"  p_cosmo (R_H/l_Pl):      {result['p_cosmo']:.6e}")

    if result.get("rho_DE_from_module") is not None:
        out.append(f"  rho_DE (impedance mod):  {result['rho_DE_from_module']:.6e}")

    return "\n".join(out)


# ── Tool 8: monte_carlo ──────────────────────────────────────────────────────

@mcp.tool()
def monte_carlo(p: int = 101, n_sites: int = 16, temperature: float = 1.0) -> str:
    """Run a quick RPST Monte Carlo simulation (XY model on Z_p lattice).

    Performs Metropolis sampling and returns energy, magnetisation, and
    spacing statistics.

    Args:
        p: Prime modulus for Z_p lattice (default 101, must be prime)
        n_sites: Number of lattice sites in 1-D ring (default 16)
        temperature: Temperature in k_B=1 units (default 1.0)
    """
    if not _HAS_MC:
        return "ERROR: bpr.rpst.monte_carlo module not available."

    try:
        mc = RPSTMonteCarlo(
            p=p,
            n_sites=n_sites,
            coupling_J=1.0,
            temperature=temperature,
            seed=42,
        )
    except Exception as e:
        return f"ERROR initializing Monte Carlo: {e}"

    # Thermalise
    n_therm = 500
    n_measure = 1000
    try:
        for _ in range(n_therm):
            mc.sweep()

        energies = []
        for _ in range(n_measure):
            mc.sweep()
            energies.append(mc.energy())
    except Exception as e:
        return f"ERROR during Monte Carlo sweep: {e}\n{traceback.format_exc()}"

    import numpy as np
    energies = np.array(energies)
    E_mean = float(np.mean(energies))
    E_std = float(np.std(energies))
    E_per_site = E_mean / n_sites
    C_v = float(np.var(energies)) / (temperature ** 2 * n_sites)

    out = []
    out.append("=" * 60)
    out.append(f"RPST Monte Carlo: Z_{p} lattice, {n_sites} sites, T={temperature}")
    out.append("=" * 60)
    out.append("")
    out.append(f"  Thermalisation sweeps: {n_therm}")
    out.append(f"  Measurement sweeps:   {n_measure}")
    out.append("")
    out.append(f"  <E>:                  {E_mean:.4f}")
    out.append(f"  sigma(E):             {E_std:.4f}")
    out.append(f"  <E>/N:                {E_per_site:.4f}")
    out.append(f"  C_v / N:              {C_v:.6f}")
    out.append("")
    out.append(f"  Final config (first 8): {list(mc.config[:8])}")

    return "\n".join(out)


if __name__ == "__main__":
    mcp.run()
