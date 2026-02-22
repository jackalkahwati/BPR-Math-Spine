#!/usr/bin/env python3
"""
BPR-Math-Spine MCP Server

Exposes the Boundary Phase Resonance / Resonant Prime Substrate Theory
codebase as MCP tools so Claude (and other clients) can query, search,
and read the research codebase directly.
"""

import os
import re
import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

REPO = Path(__file__).parent.parent  # /Users/jackal-kahwati/BPR-Math-Spine

mcp = FastMCP("bpr-math-spine")

# ── Theory registry ────────────────────────────────────────────────────────────
THEORIES = {
    "Boundary Memory Dynamics":                              "bpr/memory.py",
    "Vacuum Impedance Mismatch":                             "bpr/impedance.py",
    "Boundary-Induced Decoherence":                          "bpr/decoherence.py",
    "Universal Phase Transition Taxonomy":                   "bpr/phase_transitions.py",
    "Boundary-Mediated Neutrino Dynamics":                   "bpr/neutrino.py",
    "Substrate Information Geometry":                        "bpr/info_geometry.py",
    "Gravitational Wave Phenomenology":                      "bpr/gravitational_waves.py",
    "Substrate Complexity":                                  "bpr/complexity.py",
    "Bioelectric Substrate Coupling":                        "bpr/bioelectric.py",
    "Resonant Collective Dynamics":                          "bpr/collective.py",
    "BPR Cosmology & Early Universe":                        "bpr/cosmology.py",
    "QCD & Flavor Physics":                                  "bpr/qcd_flavor.py",
    "Emergent Spacetime & Holography":                       "bpr/emergent_spacetime.py",
    "Topological Condensed Matter":                          "bpr/topological_matter.py",
    "Clifford Algebra Embedding":                            "bpr/clifford_bpr.py",
    "Quantum Foundations":                                   "bpr/quantum_foundations.py",
    "Gauge Unification & Hierarchy":                         "bpr/gauge_unification.py",
    "Charged Lepton Masses":                                 "bpr/charged_leptons.py",
    "Nuclear Physics from Boundary Shell":                   "bpr/nuclear_physics.py",
    "Quantum Gravity Phenomenology":                         "bpr/quantum_gravity_pheno.py",
    "Quantum Chemistry & Periodic Table":                    "bpr/quantum_chemistry.py",
    "Fine Structure Constant from Substrate":                "bpr/alpha_derivation.py",
    "Invariant Structure, Boundary Dynamics, and Symbolic Meaning": "bpr/coherence_transitions.py",
    "Meta-Boundary Dynamics":                                "bpr/meta_boundary.py",
    "Emergent Physics from Prime Substrates":                "bpr/rpst_extensions.py",
    "RPST Stability Manifolds":                              "bpr/stability_manifolds.py",
    "Functional Architecture of Reality":                    "bpr/functional_architecture.py",
}


def _safe_path(rel: str) -> Path:
    """Resolve a relative path inside the repo, rejecting traversals."""
    p = (REPO / rel).resolve()
    if not str(p).startswith(str(REPO)):
        raise ValueError(f"Path outside repo: {rel}")
    return p


# ── Tools ──────────────────────────────────────────────────────────────────────

@mcp.tool()
def list_theories() -> str:
    """List all 27 theories in the BPR-Math-Spine codebase with their module paths."""
    lines = []
    for i, (name, path) in enumerate(THEORIES.items(), 1):
        lines.append(f"{i:2d}. {name}\n      → {path}")
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
                + "\n".join(f"  • {k}" for k, _ in matches)
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
    """Return the full VALIDATION_STATUS.md — pass/fail status of all theory predictions."""
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


if __name__ == "__main__":
    mcp.run()
