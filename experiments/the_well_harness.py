"""
BPR ↔ The Well Validation Harness
===================================

Cross-validates BPR predictions against PolymathicAI's "The Well"
physics simulation dataset collection (15 TB, 16+ datasets).

https://github.com/polymathicai/the_well

Usage
-----
    python experiments/the_well_harness.py            # run all validators
    python experiments/the_well_harness.py --dataset turing   # single
    python experiments/the_well_harness.py --json             # JSON output
    python experiments/the_well_harness.py --verbose          # debug detail
    python experiments/the_well_harness.py --list             # show mapping

Dataset → Validator mapping
----------------------------
turing        gray_scott_reaction_diffusion → INAPPLICABLE (GS ≠ Turing)
brusselator   brusselator                   → P23.1 Turing wavelength (DERIVED)
acoustic      acoustic_scattering_inclusions→ mode entropy (impedance mismatch)
convection    rayleigh_benard               → P4.x  β scaling (Class C)
active        active_matter                 → P10.x Kuramoto coherence
mhd           MHD_64 / MHD_256             → P7.1  energy spectral index
turb2d        turbulent_radiative_layer_2D  → P8.x  2-D enstrophy cascade k^{-3}
stratified    turbulence_gravity_cooling    → P8.y  Fr_c=1 stratification
rt            rayleigh_taylor_instability   → P9.x  RT mixing energy spectrum
supernova     supernova_explosion_64        → P10.x blast wave spectrum
turb3d        turbulent_radiative_layer_3D  → P11.x 3-D Kolmogorov k^{-5/3}
acoustic_maze acoustic_scattering_maze      → mode entropy (impedance mismatch)
shear         shear_flow                    → P13.x 2-D enstrophy cascade
planet        planetswe                     → P14.x geostrophic turbulence k^{-3}
helmholtz     helmholtz_staircase           → P15.x mode spacing regularity
viscoelastic  viscoelastic_instability      → P16.x steep spectrum (Class A+C)
euler         euler_multi_quadrants_openBC  → P17.x Burgers spectrum k^{-2}
rsg           convective_envelope_rsg       → P18.x stellar Kolmogorov k^{-5/3}
nsmerger      post_neutron_star_merger      → P19.x MHD energy spectrum

Data access
-----------
Install one of:
  pip install the_well          # official package (requires local download)
  pip install datasets          # HuggingFace streaming (no full download)

Without either package, validators report SKIPPED with installation hints.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

# Ensure repo root on sys.path
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

# ── ANSI colours (same palette as validate_all_theories.py) ──────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


# ---------------------------------------------------------------------------
# Validator registry
# ---------------------------------------------------------------------------

VALIDATORS = {
    "turing": {
        "module": "bpr.the_well.validators.turing",
        "dataset": "gray_scott_reaction_diffusion",
        "bpr_id": "P23.1",
        "description": "Gray-Scott pattern wavelength (CONJECTURAL — GS spots ≠ Turing)",
    },
    "brusselator": {
        "module": "bpr.the_well.validators.brusselator",
        "dataset": "brusselator",
        "bpr_id": "P23.1-B",
        "description": "Brusselator Turing wavelength (DERIVED — genuine Turing instability)",
    },
    "acoustic": {
        "module": "bpr.the_well.validators.acoustic",
        "dataset": "acoustic_scattering_inclusions",
        "bpr_id": "P-imp",
        "description": "Acoustic scattering mode entropy (impedance mismatch)",
    },
    "convection": {
        "module": "bpr.the_well.validators.convection",
        "dataset": "rayleigh_benard",
        "bpr_id": "P4.x",
        "description": "Rayleigh-Bénard Nu~Ra^β Class C scaling (CONSISTENT)",
    },
    "active": {
        "module": "bpr.the_well.validators.active_matter",
        "dataset": "active_matter",
        "bpr_id": "P10.x",
        "description": "Active matter flocking coherence (Kuramoto/Vicsek)",
    },
    "mhd": {
        "module": "bpr.the_well.validators.mhd",
        "dataset": "MHD_64",
        "bpr_id": "P7.1",
        "description": "MHD turbulence energy spectral index E∝k^α",
    },
    "turb2d": {
        "module": "bpr.the_well.validators.turbulence_2d",
        "dataset": "turbulent_radiative_layer_2D",
        "bpr_id": "P8.x",
        "description": "2-D turbulence enstrophy cascade E(k)∝k^{-3}",
    },
    "stratified": {
        "module": "bpr.the_well.validators.turbulence_stratified",
        "dataset": "turbulence_gravity_cooling",
        "bpr_id": "P8.y",
        "description": "Stratified turbulence Fr_c=1 transition (Brunt-Väisälä)",
    },
    "rt": {
        "module": "bpr.the_well.validators.rayleigh_taylor",
        "dataset": "rayleigh_taylor_instability",
        "bpr_id": "P9.x",
        "description": "Rayleigh-Taylor mixing zone energy spectrum (Class D)",
    },
    "supernova": {
        "module": "bpr.the_well.validators.supernova",
        "dataset": "supernova_explosion_64",
        "bpr_id": "P10.x",
        "description": "Supernova blast wave post-shock energy spectrum",
    },
    "turb3d": {
        "module": "bpr.the_well.validators.turbulence_3d",
        "dataset": "turbulent_radiative_layer_3D",
        "bpr_id": "P11.x",
        "description": "3-D Kolmogorov cascade E(k)∝k^{-5/3}",
    },
    "acoustic_maze": {
        "module": "bpr.the_well.validators.acoustic_maze",
        "dataset": "acoustic_scattering_maze",
        "bpr_id": "P-imp",
        "description": "Acoustic maze scattering mode entropy (impedance mismatch)",
    },
    "shear": {
        "module": "bpr.the_well.validators.shear_flow",
        "dataset": "shear_flow",
        "bpr_id": "P13.x",
        "description": "Shear flow 2-D enstrophy cascade E(k)∝k^{-3}",
    },
    "planet": {
        "module": "bpr.the_well.validators.planetswe",
        "dataset": "planetswe",
        "bpr_id": "P14.x",
        "description": "Planetary shallow water geostrophic turbulence k^{-3}",
    },
    "helmholtz": {
        "module": "bpr.the_well.validators.helmholtz",
        "dataset": "helmholtz_staircase",
        "bpr_id": "P15.x",
        "description": "Helmholtz staircase vertical mode regularity",
    },
    "viscoelastic": {
        "module": "bpr.the_well.validators.viscoelastic",
        "dataset": "viscoelastic_instability",
        "bpr_id": "P16.x",
        "description": "Viscoelastic instability steep spectrum (Class A+C)",
    },
    "euler": {
        "module": "bpr.the_well.validators.euler_compressible",
        "dataset": "euler_multi_quadrants_openBC",
        "bpr_id": "P17.x",
        "description": "Compressible Euler post-shock Burgers spectrum k^{-2}",
    },
    "rsg": {
        "module": "bpr.the_well.validators.convective_rsg",
        "dataset": "convective_envelope_rsg",
        "bpr_id": "P18.x",
        "description": "Red supergiant convective envelope Kolmogorov cascade",
    },
    "nsmerger": {
        "module": "bpr.the_well.validators.neutron_star",
        "dataset": "post_neutron_star_merger",
        "bpr_id": "P19.x",
        "description": "Post-neutron-star-merger MHD energy spectrum",
    },
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _sigma_str(r: dict) -> str:
    if r.get("skipped"):
        return f"{DIM}SKIPPED{RESET}"
    satisfies = r.get("satisfies")
    sigma = r.get("sigma")
    if satisfies is not None:
        colour = GREEN if satisfies else RED
        label = "✓ PASS" if satisfies else "✗ FAIL"
        return f"{colour}{label} (bound){RESET}"
    if sigma is None:
        return "—"
    if sigma < 1.0:
        return f"{GREEN}{sigma:.2f}σ{RESET}"
    if sigma < 2.0:
        return f"{YELLOW}{sigma:.2f}σ{RESET}"
    if sigma < 3.0:
        return f"{YELLOW}{BOLD}{sigma:.2f}σ{RESET}"
    return f"{RED}{BOLD}{sigma:.2f}σ{RESET}"


def _status_str(status: str) -> str:
    colours = {
        "DERIVED": GREEN,
        "CONSISTENT": CYAN,
        "FRAMEWORK": YELLOW,
        "CONJECTURAL": DIM,
    }
    c = colours.get(status, "")
    return f"{c}{status}{RESET}"


def _float_str(v) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    if abs(v) > 1e3 or (0 < abs(v) < 1e-3):
        return f"{v:.3e}"
    return f"{v:.4f}"


# ---------------------------------------------------------------------------
# Run a single validator
# ---------------------------------------------------------------------------

def run_validator(key: str, verbose: bool = False) -> dict:
    """Import and run one validator module."""
    import importlib
    info = VALIDATORS[key]
    try:
        mod = importlib.import_module(info["module"])
        result = mod.validate(verbose=verbose)
        result["_key"] = key
        result["_dataset"] = info["dataset"]
    except Exception as exc:
        result = dict(
            _key=key,
            _dataset=info["dataset"],
            pid=info["bpr_id"],
            name=info["description"],
            theory="—",
            predicted=float("nan"),
            observed=float("nan"),
            uncertainty=float("nan"),
            sigma=None,
            rel_err=None,
            satisfies=None,
            status="—",
            unit="—",
            skipped=True,
            skip_reason=f"Validator error: {exc}",
        )
    return result


# ---------------------------------------------------------------------------
# Print functions
# ---------------------------------------------------------------------------

def print_header():
    print()
    print(f"{BOLD}{'BPR ↔ The Well Validation Harness':^80}{RESET}")
    print(f"{'PolymathicAI The Well · 15TB physics simulations':^80}")
    print("─" * 80)


def print_result_row(r: dict):
    pid   = r.get("pid", "—")
    name  = r.get("name", "—")
    pred  = _float_str(r.get("predicted"))
    obs   = _float_str(r.get("observed"))
    unit  = r.get("unit", "—")
    sig   = _sigma_str(r)
    stat  = _status_str(r.get("status", "—"))
    ds    = r.get("_dataset", "—")

    print(f"\n  {BOLD}{pid}{RESET} [{stat}]  {name}")
    print(f"  {'Dataset':12s}: {ds}")
    print(f"  {'Predicted':12s}: {pred}  {unit}")
    print(f"  {'Observed':12s}: {obs}  {unit}")
    print(f"  {'Deviation':12s}: {sig}")
    if r.get("skipped") and r.get("skip_reason"):
        print(f"  {DIM}Skip reason : {r['skip_reason']}{RESET}")


def print_summary(results: list[dict]):
    print("\n" + "─" * 80)
    print(f"{BOLD}Summary{RESET}")
    total   = len(results)
    skipped = sum(1 for r in results if r.get("skipped"))
    ran     = total - skipped
    passed  = sum(1 for r in results
                  if not r.get("skipped")
                  and (
                      (r.get("satisfies") is True) or
                      (r.get("satisfies") is None and
                       r.get("sigma") is not None and r["sigma"] < 3.0)
                  ))
    failed  = ran - passed

    print(f"  Validators run    : {ran}/{total}  ({skipped} skipped — data not loaded)")
    print(f"  Passed (<3σ)      : {GREEN}{passed}{RESET}")
    print(f"  Failed (≥3σ)      : {RED if failed else ''}{failed}{RESET}")

    if skipped > 0:
        print()
        print(f"  {DIM}To load The Well data:")
        print(f"    pip install the_well        # official (needs disk space)")
        print(f"    pip install datasets        # HuggingFace streaming{RESET}")

    print()
    if failed > 0:
        sys.exit(1)


def print_listing():
    print(f"\n{BOLD}Validator → Dataset → BPR Prediction mapping:{RESET}\n")
    for key, info in VALIDATORS.items():
        print(f"  {CYAN}{key:12s}{RESET} {info['dataset']:40s} → {info['bpr_id']}")
        print(f"  {'':12s} {info['description']}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BPR ↔ The Well validation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", "-d",
        choices=list(VALIDATORS.keys()),
        help="Run only this validator (default: all)",
    )
    parser.add_argument("--json", action="store_true",
                        help="Output machine-readable JSON")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-frame debug detail")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List all validator → dataset mappings and exit")
    args = parser.parse_args()

    if args.list:
        print_listing()
        return

    keys = [args.dataset] if args.dataset else list(VALIDATORS.keys())

    if not args.json:
        print_header()

    results = []
    for key in keys:
        if not args.json:
            info = VALIDATORS[key]
            print(f"\n{BOLD}Running:{RESET} {key} ({info['dataset']})", flush=True)
        r = run_validator(key, verbose=args.verbose)
        results.append(r)
        if not args.json:
            print_result_row(r)

    if args.json:
        # Sanitise for JSON serialisation
        def _clean(v):
            if isinstance(v, float) and not math.isfinite(v):
                return None
            return v
        output = [{k: _clean(v) for k, v in r.items()} for r in results]
        print(json.dumps(output, indent=2))
        return

    print_summary(results)


if __name__ == "__main__":
    main()
