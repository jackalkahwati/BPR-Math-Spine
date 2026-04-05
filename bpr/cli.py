#!/usr/bin/env python3
"""
BPR-Math-Spine Command Line Interface
======================================

Usage:
    bpr predict [--p P] [--z Z]
    bpr pipeline <name> [--p P] [--z Z] [--N N]
    bpr pipeline --list
    bpr constants [--filter PATTERN]
    bpr verify [--sector SECTOR]
    bpr e8
    bpr status
    bpr table
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from typing import Any, Dict


# ── Formatting helpers ─────────────────────────────────────────────────

def _hrule(char="=", width=78):
    return char * width


def _fmt_value(val):
    """Format a numeric or boolean value for display."""
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, float):
        if abs(val) > 1e4 or (abs(val) < 1e-3 and val != 0):
            return f"{val:.6e}"
        return f"{val:.6f}"
    if isinstance(val, dict):
        return _fmt_dict(val, indent=4)
    if isinstance(val, (list, tuple)):
        items = ", ".join(str(x) for x in val[:8])
        if len(val) > 8:
            items += ", ..."
        return f"[{items}]"
    return str(val)


def _fmt_dict(d: Dict[str, Any], indent: int = 0) -> str:
    """Pretty-print a dict as an aligned key-value table."""
    if not d:
        return "(empty)"
    lines = []
    pad = " " * indent
    max_key = max(len(str(k)) for k in d) if d else 0
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{pad}{str(k):<{max_key}}:")
            lines.append(_fmt_dict(v, indent=indent + 4))
        else:
            lines.append(f"{pad}{str(k):<{max_key}}  {_fmt_value(v)}")
    return "\n".join(lines)


def _print_header(title: str):
    print()
    print(_hrule())
    print(f"  {title}")
    print(_hrule())


# ── Subcommand: predict ────────────────────────────────────────────────

def cmd_predict(args):
    """Run the full cosmological chain from (p, z)."""
    try:
        from bpr.cross_predictions import full_cosmological_chain
    except ImportError as e:
        print(f"Error: cannot import cross_predictions: {e}")
        print("Ensure bpr is installed: pip install -e .")
        return 1

    p = args.p
    z = args.z
    _print_header(f"BPR Full Cosmological Chain  (p={p}, z={z})")

    try:
        result = full_cosmological_chain(p=p, z=z)
    except Exception as e:
        print(f"\nPipeline error: {e}")
        return 1

    # Separate inputs from predictions
    inputs = result.pop("inputs", {"p": p, "z": z})
    print(f"\n  Substrate inputs: p = {inputs['p']},  z = {inputs['z']}\n")

    # Print predictions grouped by presence of percent_error
    predictions = {}
    errors = {}
    meta = {}
    for k, v in result.items():
        if k.endswith("_percent_error"):
            errors[k] = v
        elif k.endswith("_error"):
            meta[k] = v
        else:
            predictions[k] = v

    print(f"  {'Quantity':<35} {'Value':<24} {'% Error'}")
    print(f"  {'-'*35} {'-'*24} {'-'*12}")
    for k, v in predictions.items():
        err_key = k + "_percent_error"
        err = errors.get(err_key, "")
        err_str = f"{err:.4f}%" if isinstance(err, float) else str(err)
        print(f"  {k:<35} {_fmt_value(v):<24} {err_str}")

    if meta:
        print(f"\n  Notes:")
        for k, v in meta.items():
            print(f"    {k}: {v}")

    print()
    return 0


# ── Subcommand: pipeline ──────────────────────────────────────────────

PIPELINE_MAP = {
    "lepton-masses":     "pipeline_impedance_to_lepton_masses",
    "decoherence":       "pipeline_impedance_to_decoherence",
    "casimir":           "pipeline_substrate_to_casimir",
    "tdgl":              "pipeline_tdgl_to_phase_classification",
    "kuramoto":          "pipeline_kuramoto_to_transition",
    "consciousness":     "pipeline_agents_to_consciousness",
    "fractal-transport": "pipeline_bond_to_fractal_transport",
}

PIPELINE_DESCRIPTIONS = {
    "lepton-masses":     "Impedance -> Gauge Unification -> Lepton Masses",
    "decoherence":       "Impedance -> Decoherence -> Coherence -> QM",
    "casimir":           "RPST Substrate -> Resonance -> Casimir Force",
    "tdgl":              "TDGL -> Coherence -> Phase Transitions",
    "kuramoto":          "Collective Oscillators -> Phase Transitions",
    "consciousness":     "Conscious Agents -> Collective -> Coherence",
    "fractal-transport": "Chemistry -> Resonance Families -> Fractional Transport",
}


def cmd_pipeline(args):
    """Run a named prediction pipeline."""
    if args.list:
        _print_header("Available BPR Pipelines")
        for name, desc in PIPELINE_DESCRIPTIONS.items():
            print(f"  {name:<22} {desc}")
        print()
        return 0

    name = args.name
    if name not in PIPELINE_MAP:
        print(f"Error: unknown pipeline '{name}'")
        print(f"Available: {', '.join(PIPELINE_MAP)}")
        print("Use 'bpr pipeline --list' to see descriptions.")
        return 1

    func_name = PIPELINE_MAP[name]
    try:
        from bpr import pipelines
        func = getattr(pipelines, func_name)
    except (ImportError, AttributeError) as e:
        print(f"Error: cannot load pipeline '{name}': {e}")
        return 1

    _print_header(f"Pipeline: {name}  ({PIPELINE_DESCRIPTIONS[name]})")

    # Build kwargs from CLI overrides
    kwargs = {}
    if args.p is not None:
        kwargs["p"] = args.p
    if args.z is not None:
        kwargs["z"] = args.z
    if args.N is not None:
        # Map --N to the appropriate parameter per pipeline
        if name == "kuramoto":
            kwargs["N"] = args.N
        elif name == "consciousness":
            kwargs["n_agents"] = args.N

    try:
        result = func(**kwargs)
    except TypeError as e:
        # If kwargs don't match, fall back to defaults
        print(f"  (Note: some overrides not applicable to this pipeline: {e})")
        result = func()
    except Exception as e:
        print(f"\nPipeline error: {e}")
        return 1

    print()
    print(_fmt_dict(result, indent=2))
    print()
    return 0


# ── Subcommand: constants ─────────────────────────────────────────────

# (name, attribute, units) for display
_CONSTANTS_TABLE = [
    ("C",             "Speed of light",          "m/s"),
    ("HBAR",          "Reduced Planck constant",  "J*s"),
    ("H_PLANCK",      "Planck constant",          "J*s"),
    ("G",             "Gravitational constant",   "m^3/(kg*s^2)"),
    ("K_B",           "Boltzmann constant",       "J/K"),
    ("E_CHARGE",      "Elementary charge",        "C"),
    ("EPSILON_0",     "Vacuum permittivity",      "F/m"),
    ("MU_0",          "Vacuum permeability",      "H/m"),
    ("Z_0",           "Vacuum impedance",         "Ohm"),
    ("ALPHA_EM",      "Fine structure constant",  ""),
    ("ALPHA_EM_MZ",   "alpha_EM at M_Z",          ""),
    ("L_PLANCK",      "Planck length",            "m"),
    ("T_PLANCK",      "Planck time",              "s"),
    ("M_PLANCK",      "Planck mass",              "kg"),
    ("E_PLANCK",      "Planck energy",            "J"),
    ("M_PLANCK_GEV",  "Planck mass",              "GeV"),
    ("V_HIGGS",       "Electroweak VEV",          "GeV"),
    ("M_Z",           "Z boson mass",             "GeV"),
    ("M_W",           "W boson mass",             "GeV"),
    ("M_PROTON",      "Proton mass",              "MeV"),
    ("M_NEUTRON",     "Neutron mass",             "MeV"),
    ("M_ELECTRON",    "Electron mass",            "MeV"),
    ("HBAR_C",        "hbar*c",                   "MeV*fm"),
    ("LAMBDA_QCD",    "QCD scale",                "GeV"),
    ("M_SUN",         "Solar mass",               "kg"),
    ("R_SUN",         "Solar radius",             "m"),
    ("H_0",           "Hubble constant",          "km/s/Mpc"),
    ("T_CMB",         "CMB temperature",          "K"),
    ("OMEGA_LAMBDA",  "Dark energy fraction",     ""),
    ("R_NUCLEAR",     "Nuclear radius constant",  "fm"),
    ("P_DEFAULT",     "Default substrate prime",  ""),
    ("Z_DEFAULT",     "Default substrate z",      ""),
    ("PHI_GOLDEN",    "Golden ratio",             ""),
]


def cmd_constants(args):
    """Print all BPR physical constants."""
    try:
        from bpr import constants as C
    except ImportError as e:
        print(f"Error: cannot import bpr.constants: {e}")
        return 1

    _print_header("BPR Physical Constants")

    filt = args.filter.lower() if args.filter else None

    print(f"  {'Symbol':<16} {'Description':<28} {'Value':<20} {'Units'}")
    print(f"  {'-'*16} {'-'*28} {'-'*20} {'-'*12}")

    for sym, desc, units in _CONSTANTS_TABLE:
        if filt and filt not in sym.lower() and filt not in desc.lower():
            continue
        val = getattr(C, sym, None)
        if val is None:
            continue
        print(f"  {sym:<16} {desc:<28} {_fmt_value(val):<20} {units}")

    # Also show GAMMA_ZEROS
    if not filt or "gamma" in (filt or "") or "zero" in (filt or ""):
        zeros = getattr(C, "GAMMA_ZEROS", None)
        if zeros is not None:
            z_str = ", ".join(f"{g:.4f}" for g in zeros)
            print(f"  {'GAMMA_ZEROS':<16} {'Riemann zero Im parts':<28} [{z_str}]")

    print()
    return 0


# ── Subcommand: verify ────────────────────────────────────────────────

DERIVATION_MAP = {
    "maxwell":       "derive_maxwell_from_boundary",
    "schrodinger":   "derive_schrodinger_from_boundary",
    "einstein":      "derive_linearized_einstein_from_boundary",
    "conservation":  "derive_conservation_law",
    "tdgl":          "derive_tdgl_from_boundary",
    "weinberg":      "verify_impedance_weinberg_angle",
    "all":           "run_all_derivations",
}


def cmd_verify(args):
    """Run symbolic verification of field equations."""
    try:
        from bpr import symbolic_derivations as sd
    except ImportError as e:
        print(f"Error: cannot import symbolic_derivations: {e}")
        print("Ensure sympy is installed: pip install sympy")
        return 1

    sector = args.sector or "all"
    if sector not in DERIVATION_MAP:
        print(f"Error: unknown sector '{sector}'")
        print(f"Available: {', '.join(DERIVATION_MAP)}")
        return 1

    func_name = DERIVATION_MAP[sector]
    func = getattr(sd, func_name, None)
    if func is None:
        print(f"Error: function '{func_name}' not found in symbolic_derivations")
        return 1

    _print_header(f"Symbolic Verification: {sector}")

    try:
        result = func()
    except Exception as e:
        print(f"\nDerivation error: {e}")
        return 1

    print()
    if isinstance(result, dict):
        # For run_all, result is {sector: sub_dict}
        for key, val in result.items():
            if isinstance(val, dict):
                print(f"  [{key}]")
                for k2, v2 in val.items():
                    print(f"    {k2:<30} {_fmt_value(v2)}")
                print()
            else:
                print(f"  {key:<30} {_fmt_value(val)}")
    else:
        print(f"  Result: {result}")
    print()
    return 0


# ── Subcommand: e8 ────────────────────────────────────────────────────

def cmd_e8(args):
    """Verify E8 root system properties."""
    try:
        from bpr.clifford_bpr import verify_e8_properties
    except ImportError as e:
        print(f"Error: cannot import clifford_bpr: {e}")
        return 1

    _print_header("E8 Root System Verification")

    try:
        result = verify_e8_properties()
    except Exception as e:
        print(f"\nE8 verification error: {e}")
        return 1

    print()
    print(_fmt_dict(result, indent=2))
    print()
    return 0


# ── Subcommand: status ────────────────────────────────────────────────

def cmd_status(args):
    """Show codebase status."""
    _print_header("BPR-Math-Spine Status")

    # Version
    try:
        from bpr import __version__
        version = __version__
    except (ImportError, AttributeError):
        version = "(unknown)"
    print(f"  Version:    {version}")

    # Theory count from __init__.py docstring
    try:
        import bpr
        doc = bpr.__doc__ or ""
        theories = [l.strip() for l in doc.splitlines()
                    if l.strip() and l.strip()[0] in "IVX" and "(" in l]
        print(f"  Theories:   {len(theories)}")
    except Exception:
        print(f"  Theories:   (could not parse)")

    # Available pipelines
    print(f"  Pipelines:  {len(PIPELINE_MAP)}")
    for name in PIPELINE_MAP:
        print(f"    - {name}")

    # Derivation sectors
    sectors = [k for k in DERIVATION_MAP if k != "all"]
    print(f"  Derivations: {len(sectors)}")
    for s in sectors:
        print(f"    - {s}")

    # Test count
    try:
        import pathlib
        test_dir = pathlib.Path(__file__).resolve().parent.parent / "tests"
        if test_dir.is_dir():
            test_files = list(test_dir.glob("test_*.py"))
            print(f"  Test files: {len(test_files)}")
        else:
            print(f"  Test files: (tests/ not found)")
    except Exception:
        print(f"  Test files: (could not count)")

    # Module availability check
    print(f"\n  Module availability:")
    core_modules = [
        "bpr.constants", "bpr.impedance", "bpr.cross_predictions",
        "bpr.pipelines", "bpr.clifford_bpr", "bpr.symbolic_derivations",
        "bpr.cosmology", "bpr.charged_leptons", "bpr.gauge_unification",
    ]
    for mod in core_modules:
        try:
            __import__(mod)
            status = "OK"
        except ImportError:
            status = "MISSING"
        print(f"    {mod:<35} {status}")

    print()
    return 0


# ── Subcommand: table ─────────────────────────────────────────────────

def cmd_table(args):
    """Generate the full prediction table."""
    try:
        import importlib.util
        import pathlib

        # Try importing from scripts directory
        script_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "scripts" / "generate_prediction_table.py"
        )
        if not script_path.exists():
            print(f"Error: prediction table script not found at {script_path}")
            return 1

        spec = importlib.util.spec_from_file_location(
            "generate_prediction_table", str(script_path)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.run_all()
    except Exception as e:
        print(f"Error generating prediction table: {e}")
        return 1

    return 0


# ── Argument parser ───────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bpr",
        description="BPR-Math-Spine: Boundary Phase Resonance toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              bpr predict                     Full cosmological chain (default p=104729, z=6)
              bpr predict --p 7 --z 3         Chain with custom substrate
              bpr pipeline lepton-masses      Run the lepton mass pipeline
              bpr pipeline --list             List all pipelines
              bpr constants --filter planck   Show Planck-related constants
              bpr verify --sector maxwell     Derive Maxwell from boundary
              bpr verify                      Run all symbolic derivations
              bpr e8                          E8 root system check
              bpr status                      Codebase overview
              bpr table                       Full prediction table
        """),
    )

    sub = parser.add_subparsers(dest="command", help="subcommand")

    # predict
    p_pred = sub.add_parser("predict", help="Run full cosmological chain from (p, z)")
    p_pred.add_argument("--p", type=int, default=104729, help="Substrate prime (default: 104729)")
    p_pred.add_argument("--z", type=int, default=6, help="Substrate z (default: 6)")

    # pipeline
    p_pipe = sub.add_parser("pipeline", help="Run a named prediction pipeline")
    p_pipe.add_argument("name", nargs="?", default=None, help="Pipeline name")
    p_pipe.add_argument("--list", action="store_true", help="List available pipelines")
    p_pipe.add_argument("--p", type=int, default=None, help="Substrate prime override")
    p_pipe.add_argument("--z", type=int, default=None, help="Substrate z override")
    p_pipe.add_argument("--N", type=int, default=None, help="Number of agents/oscillators override")

    # constants
    p_const = sub.add_parser("constants", help="Print BPR physical constants")
    p_const.add_argument("--filter", type=str, default=None, help="Filter constants by pattern")

    # verify
    p_ver = sub.add_parser("verify", help="Symbolic verification of field equations")
    p_ver.add_argument(
        "--sector", type=str, default=None,
        help="Sector to verify (maxwell, schrodinger, einstein, conservation, tdgl, weinberg, all)",
    )

    # e8
    sub.add_parser("e8", help="Verify E8 root system properties")

    # status
    sub.add_parser("status", help="Show codebase status")

    # table
    sub.add_parser("table", help="Generate full prediction table")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    dispatch = {
        "predict":   cmd_predict,
        "pipeline":  cmd_pipeline,
        "constants": cmd_constants,
        "verify":    cmd_verify,
        "e8":        cmd_e8,
        "status":    cmd_status,
        "table":     cmd_table,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
