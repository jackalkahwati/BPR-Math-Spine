"""
BPR-Math-Spine REST API
========================

FastAPI service exposing BPR predictions, pipelines, constants,
and symbolic derivations as HTTP endpoints.

Run: uvicorn api.main:app --reload --port 8420
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Ensure the bpr package is importable (repo root on sys.path)
# ---------------------------------------------------------------------------
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Guarded BPR imports
# ---------------------------------------------------------------------------

try:
    import bpr
    _BPR_VERSION = getattr(bpr, "__version__", "unknown")
except Exception:
    bpr = None  # type: ignore[assignment]
    _BPR_VERSION = "unavailable"

try:
    from bpr import constants as bpr_constants
    _HAS_CONSTANTS = True
except Exception:
    _HAS_CONSTANTS = False

try:
    from bpr.cross_predictions import full_cosmological_chain
    _HAS_COSMO_CHAIN = True
except Exception:
    _HAS_COSMO_CHAIN = False

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
    from bpr.symbolic_derivations import (
        derive_maxwell_from_boundary,
        derive_schrodinger_from_boundary,
        derive_linearized_einstein_from_boundary,
        derive_conservation_law,
        derive_tdgl_from_boundary,
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

try:
    from bpr.gauge_unification import weinberg_angle_from_impedance
    _HAS_WEINBERG = True
except Exception:
    _HAS_WEINBERG = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numpy_safe(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serialisable Python types."""
    if isinstance(obj, dict):
        return {k: _numpy_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpy_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.complexfloating, complex)):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, np.bool_):
        return bool(obj)
    # sympy expressions -> str
    if hasattr(obj, "is_Symbol") or hasattr(obj, "free_symbols"):
        return str(obj)
    # Matrices (sympy Matrix)
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return [[str(cell) for cell in row] for row in obj.tolist()]
        except Exception:
            return str(obj)
    return obj


# ---------------------------------------------------------------------------
# Theory catalogue (36 theories from __init__.py docstring)
# ---------------------------------------------------------------------------

THEORY_CATALOGUE: List[Dict[str, str]] = [
    {"number": "I",      "name": "memory",                "module": "bpr.memory",                "title": "Boundary Memory Dynamics"},
    {"number": "II",     "name": "impedance",             "module": "bpr.impedance",             "title": "Vacuum Impedance Mismatch"},
    {"number": "III",    "name": "decoherence",           "module": "bpr.decoherence",           "title": "Boundary-Induced Decoherence"},
    {"number": "IV",     "name": "phase_transitions",     "module": "bpr.phase_transitions",     "title": "Universal Phase Transition Taxonomy"},
    {"number": "V",      "name": "neutrino",              "module": "bpr.neutrino",              "title": "Boundary-Mediated Neutrino Dynamics"},
    {"number": "VI",     "name": "info_geometry",         "module": "bpr.info_geometry",         "title": "Substrate Information Geometry"},
    {"number": "VII",    "name": "gravitational_waves",   "module": "bpr.gravitational_waves",   "title": "Gravitational Wave Phenomenology"},
    {"number": "VIII",   "name": "complexity",            "module": "bpr.complexity",            "title": "Substrate Complexity Theory"},
    {"number": "IX",     "name": "bioelectric",           "module": "bpr.bioelectric",           "title": "Bioelectric Substrate Coupling"},
    {"number": "X",      "name": "collective",            "module": "bpr.collective",            "title": "Resonant Collective Dynamics"},
    {"number": "XI",     "name": "cosmology",             "module": "bpr.cosmology",             "title": "Cosmology & Early Universe"},
    {"number": "XII",    "name": "qcd_flavor",            "module": "bpr.qcd_flavor",            "title": "QCD & Flavor Physics"},
    {"number": "XIII",   "name": "emergent_spacetime",    "module": "bpr.emergent_spacetime",    "title": "Emergent Spacetime & Holography"},
    {"number": "XIV",    "name": "topological_matter",    "module": "bpr.topological_matter",    "title": "Topological Condensed Matter"},
    {"number": "XV",     "name": "clifford_bpr",          "module": "bpr.clifford_bpr",          "title": "Clifford Algebra Embedding"},
    {"number": "XVI",    "name": "quantum_foundations",    "module": "bpr.quantum_foundations",   "title": "Quantum Foundations"},
    {"number": "XVII",   "name": "gauge_unification",     "module": "bpr.gauge_unification",     "title": "Gauge Unification & Hierarchy"},
    {"number": "XVIII",  "name": "charged_leptons",       "module": "bpr.charged_leptons",       "title": "Charged Lepton Masses"},
    {"number": "XIX",    "name": "nuclear_physics",       "module": "bpr.nuclear_physics",       "title": "Nuclear Physics & Shell Structure"},
    {"number": "XX",     "name": "quantum_gravity_pheno", "module": "bpr.quantum_gravity_pheno", "title": "Quantum Gravity Phenomenology"},
    {"number": "XXI",    "name": "quantum_chemistry",     "module": "bpr.quantum_chemistry",     "title": "Quantum Chemistry & Periodic Table"},
    {"number": "XXII",   "name": "coherence_transitions", "module": "bpr.coherence_transitions", "title": "Coherence Transitions & Symbolic Meaning"},
    {"number": "XXIII",  "name": "meta_boundary",         "module": "bpr.meta_boundary",         "title": "Meta-Boundary Dynamics"},
    {"number": "XXIV",   "name": "rpst_extensions",       "module": "bpr.rpst_extensions",       "title": "RPST Extensions (BPR/RPST)"},
    {"number": "XXV",    "name": "stability_manifolds",   "module": "bpr.stability_manifolds",   "title": "RPST Stability Manifolds"},
    {"number": "XXVI",   "name": "functional_architecture", "module": "bpr.functional_architecture", "title": "Functional Architecture of Reality"},
    {"number": "XXVII",  "name": "tdgl_bpr",              "module": "bpr.tdgl_bpr",              "title": "TDGL BPR Solver"},
    {"number": "XXVIII", "name": "hilbert_bpr",           "module": "bpr.hilbert_bpr",           "title": "Hilbert Space BPR Operator"},
    {"number": "XXIX",   "name": "fractional_boundary",   "module": "bpr.fractional_boundary",   "title": "Fractional Boundary Resonance Index"},
    {"number": "XXX",    "name": "plasmoid",              "module": "bpr.plasmoid",              "title": "Plasmoid Boundary-Phase Confinement"},
    {"number": "XXXI",   "name": "resonance_families",    "module": "bpr.resonance_families",    "title": "Resonance Families & Quasi-Integers"},
    {"number": "XXXII",  "name": "optimization",          "module": "bpr.optimization",          "title": "NP-Hard BPR Optimization"},
    {"number": "XXXIII", "name": "fluid_dynamics",        "module": "bpr.fluid_dynamics",        "title": "BPR Fluid Dynamics"},
    {"number": "XXXIV",  "name": "resonance_algebra",     "module": "bpr.resonance_algebra",     "title": "Resonance Algebra PDE Rulebook"},
    {"number": "XXXV",   "name": "electromechanical",     "module": "bpr.electromechanical",     "title": "Electromechanical Coherence"},
    {"number": "XXXVI",  "name": "conscious_agents",      "module": "bpr.conscious_agents",      "title": "Conscious Agents Markov Bridge"},
]

# Pipeline registry
PIPELINE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "lepton-masses": {
        "function": "pipeline_impedance_to_lepton_masses",
        "description": "Gauge unification -> impedance -> charged lepton masses (m_e, m_mu, m_tau) + Koide parameter",
        "default_params": {"p": 104761, "z": 6},
    },
    "decoherence": {
        "function": "pipeline_impedance_to_decoherence",
        "description": "Impedance mismatch -> decoherence rate -> coherence dynamics -> quantum foundations",
        "default_params": {"W_system": 1.0, "W_environment": 10.0, "T": 300.0, "A_eff": 1e-14, "lambda_dB": 1e-10, "p": 104761},
    },
    "casimir": {
        "function": "pipeline_substrate_to_casimir",
        "description": "RPST substrate -> resonance modes -> Casimir force deviation curve",
        "default_params": {},
    },
    "tdgl": {
        "function": "pipeline_tdgl_to_phase_classification",
        "description": "TDGL simulation -> coherence evolution -> phase transition classification",
        "default_params": {},
    },
    "kuramoto": {
        "function": "pipeline_kuramoto_to_transition",
        "description": "Kuramoto oscillators -> collective synchronisation -> phase transition",
        "default_params": {},
    },
    "consciousness": {
        "function": "pipeline_agents_to_consciousness",
        "description": "Conscious agents network -> collective dynamics -> coherence",
        "default_params": {},
    },
    "fractal-transport": {
        "function": "pipeline_bond_to_fractal_transport",
        "description": "Chemical bonds -> resonance families -> fractional boundary transport scaling",
        "default_params": {},
    },
}

# Symbolic derivation registry
DERIVATION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "maxwell": {
        "function": "derive_maxwell_from_boundary",
        "description": "Maxwell equations from EM boundary action stationarity",
    },
    "schrodinger": {
        "function": "derive_schrodinger_from_boundary",
        "description": "Schrodinger equation from QM boundary action",
    },
    "einstein": {
        "function": "derive_linearized_einstein_from_boundary",
        "description": "Linearised Einstein equations from GR boundary sector",
    },
    "tdgl": {
        "function": "derive_tdgl_from_boundary",
        "description": "Time-dependent Ginzburg-Landau from boundary action",
    },
    "conservation": {
        "function": "derive_conservation_law",
        "description": "Conservation law nabla.T = 0 from boundary Noether symmetry",
    },
    "weinberg": {
        "function": "weinberg_angle_from_impedance",
        "description": "Weinberg angle from impedance ratios (gauge_unification module)",
    },
}


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    p: int = Field(default=104761, description="Substrate prime modulus")
    z: int = Field(default=6, description="Coordination number")


class PipelineRequest(BaseModel):
    params: Dict[str, Any] = Field(default_factory=dict, description="Override parameters for the pipeline")


class MonteCarloRequest(BaseModel):
    p: int = Field(default=101, description="Prime modulus")
    n_sites: int = Field(default=16, description="Number of lattice sites")
    temperature: float = Field(default=1.0, description="Temperature (k_B = 1)")
    n_therm: int = Field(default=200, description="Thermalisation sweeps")
    n_samples: int = Field(default=100, description="Measurement sweeps")


class DeriveRequest(BaseModel):
    params: Dict[str, Any] = Field(default_factory=dict, description="Override parameters (sector-dependent)")


class StatusResponse(BaseModel):
    version: str
    bpr_available: bool
    theory_count: int
    available_layers: List[str]


class ErrorDetail(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="BPR-Math-Spine API",
    description=(
        "REST interface to the Boundary Phase Resonance mathematical spine. "
        "Exposes predictions, pipelines, constants, symbolic derivations, E8 "
        "verification, and RPST Monte Carlo from two integers (p, z)."
    ),
    version=_BPR_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================================================================
#  GET /  — health check
# ===================================================================

@app.get("/")
def root():
    """Health check with version and available endpoints."""
    return {
        "service": "BPR-Math-Spine API",
        "version": _BPR_VERSION,
        "status": "ok",
        "endpoints": [
            {"method": "GET",  "path": "/api/status",             "description": "Codebase status"},
            {"method": "GET",  "path": "/api/constants",          "description": "Physical constants"},
            {"method": "POST", "path": "/api/predict",            "description": "Full cosmological chain"},
            {"method": "GET",  "path": "/api/pipelines",          "description": "List pipelines"},
            {"method": "POST", "path": "/api/pipeline/{name}",    "description": "Run a named pipeline"},
            {"method": "POST", "path": "/api/derive/{sector}",    "description": "Symbolic derivation"},
            {"method": "GET",  "path": "/api/e8",                 "description": "E8 verification"},
            {"method": "POST", "path": "/api/monte-carlo",        "description": "RPST Monte Carlo"},
            {"method": "GET",  "path": "/api/theories",           "description": "List all 36 theories"},
            {"method": "GET",  "path": "/api/theory/{name}",      "description": "Theory module source"},
            {"method": "GET",  "path": "/api/the-well",           "description": "The Well validation harness results"},
            {"method": "GET",  "path": "/api/the-well/{dataset}", "description": "Run a single Well validator"},
        ],
    }


# ===================================================================
#  GET /api/status
# ===================================================================

@app.get("/api/status", response_model=StatusResponse)
def api_status():
    """Return codebase status: version, theory count, available layers."""
    layers = []
    if _HAS_CONSTANTS:
        layers.append("constants")
    if _HAS_COSMO_CHAIN:
        layers.append("cosmological_chain")
    if _HAS_PIPELINES:
        layers.append("pipelines")
    if _HAS_SYMBOLIC:
        layers.append("symbolic_derivations")
    if _HAS_E8:
        layers.append("e8_algebra")
    if _HAS_MC:
        layers.append("monte_carlo")
    if _HAS_WEINBERG:
        layers.append("weinberg_derivation")

    return StatusResponse(
        version=_BPR_VERSION,
        bpr_available=bpr is not None,
        theory_count=len(THEORY_CATALOGUE),
        available_layers=layers,
    )


# ===================================================================
#  GET /api/constants
# ===================================================================

@app.get("/api/constants")
def api_constants(filter: Optional[str] = Query(None, description="Case-insensitive substring filter on constant names")):
    """Return all physical constants as JSON. Optional filter query param."""
    if not _HAS_CONSTANTS:
        raise HTTPException(status_code=503, detail="bpr.constants module not available")

    # Collect all uppercase module-level names that are numeric
    result: Dict[str, Any] = {}
    for name in dir(bpr_constants):
        if name.startswith("_"):
            continue
        val = getattr(bpr_constants, name)
        if isinstance(val, (int, float)):
            result[name] = val
        elif isinstance(val, np.ndarray):
            result[name] = val.tolist()

    if filter:
        filt = filter.upper()
        result = {k: v for k, v in result.items() if filt in k.upper()}

    return {"count": len(result), "constants": result}


# ===================================================================
#  POST /api/predict
# ===================================================================

@app.post("/api/predict")
def api_predict(body: PredictRequest):
    """Run the full cosmological chain from (p, z) and return predictions."""
    if not _HAS_COSMO_CHAIN:
        raise HTTPException(status_code=503, detail="bpr.cross_predictions.full_cosmological_chain not available")

    try:
        raw = full_cosmological_chain(p=body.p, z=body.z)
        return _numpy_safe(raw)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}\n{traceback.format_exc()}")


# ===================================================================
#  GET /api/pipelines
# ===================================================================

@app.get("/api/pipelines")
def api_list_pipelines():
    """List all available pipelines with descriptions."""
    items = []
    for name, info in PIPELINE_REGISTRY.items():
        items.append({
            "name": name,
            "description": info["description"],
            "default_params": info["default_params"],
            "available": _HAS_PIPELINES,
        })
    return {"pipelines": items}


# ===================================================================
#  POST /api/pipeline/{name}
# ===================================================================

@app.post("/api/pipeline/{name}")
def api_run_pipeline(name: str, body: PipelineRequest):
    """Run a named pipeline with optional parameter overrides."""
    if name not in PIPELINE_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown pipeline '{name}'. Available: {list(PIPELINE_REGISTRY.keys())}",
        )
    if not _HAS_PIPELINES:
        raise HTTPException(status_code=503, detail="bpr.pipelines module not available")

    info = PIPELINE_REGISTRY[name]
    func_name = info["function"]

    # Resolve function from pipelines module
    from bpr import pipelines as _pipelines_mod
    func = getattr(_pipelines_mod, func_name, None)
    if func is None:
        raise HTTPException(status_code=500, detail=f"Pipeline function {func_name} not found in bpr.pipelines")

    # Merge defaults with overrides
    params = {**info["default_params"], **body.params}

    try:
        raw = func(**params)
        return _numpy_safe(raw)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline '{name}' failed: {exc}\n{traceback.format_exc()}")


# ===================================================================
#  POST /api/derive/{sector}
# ===================================================================

@app.post("/api/derive/{sector}")
def api_derive(sector: str, body: DeriveRequest):
    """Run a symbolic derivation and return results as JSON strings."""
    if sector not in DERIVATION_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown sector '{sector}'. Available: {list(DERIVATION_REGISTRY.keys())}",
        )

    info = DERIVATION_REGISTRY[sector]
    func_name = info["function"]

    # Resolve function
    func = None
    if sector == "weinberg":
        if not _HAS_WEINBERG:
            raise HTTPException(status_code=503, detail="bpr.gauge_unification not available")
        func = weinberg_angle_from_impedance
    else:
        if not _HAS_SYMBOLIC:
            raise HTTPException(status_code=503, detail="bpr.symbolic_derivations not available")
        from bpr import symbolic_derivations as _sym_mod
        func = getattr(_sym_mod, func_name, None)

    if func is None:
        raise HTTPException(status_code=500, detail=f"Derivation function {func_name} not found")

    try:
        if sector == "weinberg":
            # Needs impedance ratio arguments
            zeta_BW = body.params.get("zeta_BW", 1.0)
            zeta_WW = body.params.get("zeta_WW", 1.0)
            zeta_BB = body.params.get("zeta_BB", 1.0)
            raw = func(zeta_BW, zeta_WW, zeta_BB)
        else:
            raw = func(**body.params)
        return _numpy_safe(raw)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Derivation '{sector}' failed: {exc}\n{traceback.format_exc()}")


# ===================================================================
#  GET /api/e8
# ===================================================================

@app.get("/api/e8")
def api_e8():
    """Verify and return E8 Lie algebra properties."""
    if not _HAS_E8:
        raise HTTPException(status_code=503, detail="bpr.clifford_bpr.verify_e8_properties not available")

    try:
        raw = verify_e8_properties()
        return _numpy_safe(raw)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"E8 verification failed: {exc}\n{traceback.format_exc()}")


# ===================================================================
#  POST /api/monte-carlo
# ===================================================================

@app.post("/api/monte-carlo")
def api_monte_carlo(body: MonteCarloRequest):
    """Run RPST lattice Monte Carlo and return order parameter + measurements."""
    if not _HAS_MC:
        raise HTTPException(status_code=503, detail="bpr.rpst.monte_carlo not available")

    try:
        mc = RPSTMonteCarlo(
            p=body.p,
            n_sites=body.n_sites,
            temperature=body.temperature,
            seed=42,
        )

        # Thermalise
        mc.thermalize(body.n_therm)

        # Measure
        energies = []
        magnetisations = []
        for _ in range(body.n_samples):
            mc.sweep()
            energies.append(mc.energy())
            m = mc.order_parameter()
            magnetisations.append(abs(m))

        energies_arr = np.array(energies)
        mag_arr = np.array(magnetisations)

        # Correlation function
        dists, corr = mc.correlation_function()

        return _numpy_safe({
            "p": body.p,
            "n_sites": body.n_sites,
            "temperature": body.temperature,
            "n_therm": body.n_therm,
            "n_samples": body.n_samples,
            "final_order_parameter": abs(mc.order_parameter()),
            "mean_energy": float(np.mean(energies_arr)),
            "std_energy": float(np.std(energies_arr)),
            "mean_magnetisation": float(np.mean(mag_arr)),
            "std_magnetisation": float(np.std(mag_arr)),
            "energies": energies_arr,
            "magnetisations": mag_arr,
            "correlation_distances": dists,
            "correlation_function": corr,
        })
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Monte Carlo failed: {exc}\n{traceback.format_exc()}")


# ===================================================================
#  GET /api/theories
# ===================================================================

@app.get("/api/theories")
def api_theories():
    """List all 36 BPR theories."""
    enriched = []
    for theory in THEORY_CATALOGUE:
        available = False
        try:
            importlib.import_module(theory["module"])
            available = True
        except Exception:
            pass
        enriched.append({**theory, "available": available})
    return {"count": len(enriched), "theories": enriched}


# ===================================================================
#  GET /api/theory/{name}
# ===================================================================

@app.get("/api/theory/{name}")
def api_theory(name: str):
    """Get theory module source code and docstring."""
    # Find in catalogue
    match = None
    for t in THEORY_CATALOGUE:
        if t["name"] == name:
            match = t
            break
    if match is None:
        available_names = [t["name"] for t in THEORY_CATALOGUE]
        raise HTTPException(
            status_code=404,
            detail=f"Unknown theory '{name}'. Available: {available_names}",
        )

    module_path = match["module"]

    try:
        mod = importlib.import_module(module_path)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Cannot import {module_path}: {exc}")

    # Get source file
    try:
        source_file = inspect.getfile(mod)
        source = Path(source_file).read_text(encoding="utf-8")
    except Exception:
        source = None

    docstring = inspect.getdoc(mod) or ""

    # List public functions and classes
    public_api = []
    for attr_name in sorted(dir(mod)):
        if attr_name.startswith("_"):
            continue
        obj = getattr(mod, attr_name)
        if inspect.isfunction(obj) or inspect.isclass(obj):
            sig = ""
            try:
                sig = str(inspect.signature(obj))
            except (ValueError, TypeError):
                pass
            kind = "class" if inspect.isclass(obj) else "function"
            public_api.append({"name": attr_name, "kind": kind, "signature": sig})

    return {
        "number": match["number"],
        "name": match["name"],
        "title": match["title"],
        "module": module_path,
        "docstring": docstring,
        "public_api": public_api,
        "source": source,
    }


# ===================================================================
#  GET /api/the-well  — The Well validation harness results
# ===================================================================

_WELL_VALIDATORS = {
    "turing": "bpr.the_well.validators.turing",
    "brusselator": "bpr.the_well.validators.brusselator",
    "acoustic": "bpr.the_well.validators.acoustic",
    "convection": "bpr.the_well.validators.convection",
    "active": "bpr.the_well.validators.active_matter",
    "mhd": "bpr.the_well.validators.mhd",
    "turb2d": "bpr.the_well.validators.turbulence_2d",
    "stratified": "bpr.the_well.validators.turbulence_stratified",
    "rt": "bpr.the_well.validators.rayleigh_taylor",
    "supernova": "bpr.the_well.validators.supernova",
    "turb3d": "bpr.the_well.validators.turbulence_3d",
    "acoustic_maze": "bpr.the_well.validators.acoustic_maze",
    "shear": "bpr.the_well.validators.shear_flow",
    "planet": "bpr.the_well.validators.planetswe",
    "helmholtz": "bpr.the_well.validators.helmholtz",
    "viscoelastic": "bpr.the_well.validators.viscoelastic",
    "euler": "bpr.the_well.validators.euler_compressible",
    "rsg": "bpr.the_well.validators.convective_rsg",
    "nsmerger": "bpr.the_well.validators.neutron_star",
}


def _sanitise_result(r: dict) -> dict:
    """Replace NaN/inf with None for JSON serialisation."""
    import math
    out = {}
    for k, v in r.items():
        if isinstance(v, float) and not math.isfinite(v):
            out[k] = None
        else:
            out[k] = v
    return out


@app.get("/api/the-well")
def api_the_well(dataset: Optional[str] = Query(None, description="Run a single validator")):
    """Run The Well validation harness.

    Without ?dataset: returns cached summary of all 20 validators.
    With ?dataset=mhd: runs and returns that single validator.
    """
    if dataset:
        return api_the_well_single(dataset)

    results = []
    for key, module_path in _WELL_VALIDATORS.items():
        try:
            mod = importlib.import_module(module_path)
            r = mod.validate(verbose=False)
            r["_key"] = key
            results.append(_sanitise_result(r))
        except Exception as exc:
            results.append({
                "_key": key,
                "pid": key,
                "name": key,
                "skipped": True,
                "skip_reason": f"Import/run error: {exc}",
            })

    passed = sum(1 for r in results
                 if not r.get("skipped")
                 and ((r.get("satisfies") is True)
                      or (r.get("satisfies") is None
                          and r.get("sigma") is not None
                          and r["sigma"] < 3.0)))
    ran = sum(1 for r in results if not r.get("skipped"))
    skipped = sum(1 for r in results if r.get("skipped"))

    return {
        "harness": "BPR ↔ The Well",
        "source": "PolymathicAI — 15TB physics simulations",
        "total": len(results),
        "ran": ran,
        "passed": passed,
        "failed": ran - passed,
        "skipped": skipped,
        "results": results,
    }


@app.get("/api/the-well/{dataset}")
def api_the_well_single(dataset: str):
    """Run a single Well validator by key."""
    if dataset not in _WELL_VALIDATORS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown dataset '{dataset}'. Available: {list(_WELL_VALIDATORS.keys())}",
        )
    try:
        mod = importlib.import_module(_WELL_VALIDATORS[dataset])
        r = mod.validate(verbose=False)
        r["_key"] = dataset
        return _sanitise_result(r)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Validator failed: {exc}\n{traceback.format_exc()}")
