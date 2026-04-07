"""
The Well dataset loader
========================

Loads HDF5 files directly from HuggingFace Hub using ``huggingface_hub``
+ ``h5py``.  Falls back to the official ``the_well`` package if installed.

Each frame dict has flat field names mapped from the HDF5 hierarchy:
    "buoyancy"       ← t0_fields/buoyancy
    "velocity"       ← t1_fields/velocity
    "magnetic_field" ← t1_fields/magnetic_field
    "density"        ← t0_fields/density
    "A", "B"         ← t0_fields/A, t0_fields/B  (Gray-Scott)
    "concentration"  ← t0_fields/concentration
    "Ra", "Pr", "F", "k", "Ma", "Ms", "L", "alpha", "zeta"  ← scalars/*

Usage
-----
>>> frames = load_well_frames("rayleigh_benard",
...                           files=["data/test/rayleigh_benard_Rayleigh_1e6_Prandtl_1.hdf5"])
>>> frames[0]["buoyancy"].shape  # (5, 200, 512, 128)
>>> frames[0]["Ra"]              # 1000000.0
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class WellNotAvailable(RuntimeError):
    """Raised when The Well data cannot be loaded."""


# ---------------------------------------------------------------------------
# Known test-split file lists per dataset (curated from HF repo listings)
# ---------------------------------------------------------------------------

DATASET_FILES: dict[str, list[str]] = {
    "gray_scott_reaction_diffusion": [
        "data/test/gray_scott_reaction_diffusion_spots_F_0.03_k_0.062.hdf5",
        "data/test/gray_scott_reaction_diffusion_maze_F_0.029_k_0.057.hdf5",
        "data/test/gray_scott_reaction_diffusion_bubbles_F_0.098_k_0.057.hdf5",
        "data/test/gray_scott_reaction_diffusion_spirals_F_0.018_k_0.051.hdf5",
        "data/test/gray_scott_reaction_diffusion_gliders_F_0.014_k_0.054.hdf5",
    ],
    "rayleigh_benard": [
        "data/test/rayleigh_benard_Rayleigh_1e6_Prandtl_1.hdf5",
        "data/test/rayleigh_benard_Rayleigh_1e7_Prandtl_1.hdf5",
        "data/test/rayleigh_benard_Rayleigh_1e8_Prandtl_1.hdf5",
        "data/test/rayleigh_benard_Rayleigh_1e9_Prandtl_1.hdf5",
        "data/test/rayleigh_benard_Rayleigh_1e10_Prandtl_1.hdf5",
    ],
    "active_matter": [
        "data/test/active_matter_L_10.0_zeta_1.0_alpha_-1.0.hdf5",
        "data/test/active_matter_L_10.0_zeta_1.0_alpha_-2.0.hdf5",
        "data/test/active_matter_L_10.0_zeta_11.0_alpha_-1.0.hdf5",
        "data/test/active_matter_L_10.0_zeta_11.0_alpha_-2.0.hdf5",
    ],
    "MHD_64": [
        "data/test/MHD_Ma_0.7_Ms_0.5.hdf5",
        "data/test/MHD_Ma_0.7_Ms_1.5.hdf5",
        "data/test/MHD_Ma_2_Ms_0.5.hdf5",
    ],
    "acoustic_scattering_inclusions": [
        "data/test/acoustic_scattering_inclusions_chunk_36.hdf5",
        "data/test/acoustic_scattering_inclusions_chunk_37.hdf5",
    ],
    "acoustic_scattering_discontinuous": [
        "data/test/acoustic_scattering_inclusions_chunk_36.hdf5",  # fallback to inclusions
    ],
}


# ---------------------------------------------------------------------------
# HDF5 → flat dict extraction
# ---------------------------------------------------------------------------

def _hdf5_to_dict(path: str, max_samples: int = 2, max_timesteps: int = 5) -> dict:
    """Read an HDF5 file into a flat dict, slicing to avoid RAM overload.

    For arrays with shape (S, T, ...) loads at most max_samples × max_timesteps.
    """
    import h5py  # type: ignore
    frame: dict = {}
    with h5py.File(path, "r") as f:
        # Flatten t0/t1/t2 fields (strip group prefix)
        for group_name in ("t0_fields", "t1_fields", "t2_fields"):
            if group_name in f:
                for field_name, ds in f[group_name].items():
                    shape = ds.shape
                    if len(shape) >= 2:
                        s = min(shape[0], max_samples)
                        t = min(shape[1], max_timesteps)
                        frame[field_name] = ds[:s, -t:]  # last t timesteps
                    else:
                        frame[field_name] = ds[()]

        # Scalars — store with short keys
        SCALAR_MAP = {
            "Rayleigh": "Ra", "Prandtl": "Pr",
            "F": "F", "k": "k",
            "Ma": "Ma", "Ms": "Ms",
            "L": "L", "alpha": "alpha", "zeta": "zeta",
        }
        if "scalars" in f:
            for hdf_key, short_key in SCALAR_MAP.items():
                if hdf_key in f["scalars"]:
                    frame[short_key] = float(f["scalars"][hdf_key][()])

        # Dimension arrays
        if "dimensions" in f:
            for dim_name, ds in f["dimensions"].items():
                frame[f"dim_{dim_name}"] = ds[()]

    return frame


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _try_hf_hub(dataset_name: str, n: int,
                files: Optional[list[str]] = None,
                max_samples: int = 2,
                max_timesteps: int = 5) -> list[dict]:
    """Download specific HDF5 files via huggingface_hub and read with h5py."""
    from huggingface_hub import hf_hub_download  # type: ignore

    repo_id = f"polymathic-ai/{dataset_name}"
    target_files = files or DATASET_FILES.get(dataset_name, [])
    if not target_files:
        raise WellNotAvailable(f"No known test files for dataset '{dataset_name}'")

    frames = []
    for fname in target_files[:n]:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            repo_type="dataset",
        )
        frames.append(_hdf5_to_dict(local_path,
                                    max_samples=max_samples,
                                    max_timesteps=max_timesteps))
    return frames


def _try_the_well_package(dataset_name: str, n: int) -> list[dict]:
    """Load via the official ``the_well`` Python package (requires local data)."""
    from the_well.data import WellDataset  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore

    ds = WellDataset(
        well_base_path=None,
        well_dataset_name=dataset_name,
        well_split_name="test",
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    frames = []
    for i, batch in enumerate(loader):
        if i >= n:
            break
        frame = {k: v.squeeze(0).numpy() for k, v in batch.items()
                 if hasattr(v, "numpy")}
        frames.append(frame)
    return frames


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_well_frames(dataset_name: str, n: int = 1,
                     files: Optional[list[str]] = None,
                     backend: Optional[str] = None,
                     max_samples: int = 2,
                     max_timesteps: int = 5) -> list[dict]:
    """Return *n* sample frames (HDF5 dicts) from a Well dataset.

    Parameters
    ----------
    dataset_name : str
        Well dataset name, e.g. ``"rayleigh_benard"``.
    n : int
        Max number of files to load.
    files : list[str], optional
        Specific HDF5 paths within the HF repo.  Overrides the default list.
    backend : str, optional
        Force ``"hf_hub"`` or ``"the_well"``.

    Returns
    -------
    list[dict]
        Each dict maps field name → numpy array or scalar float.

    Raises
    ------
    WellNotAvailable
        If no backend can load the data.
    """
    errors = []

    if backend in (None, "the_well"):
        try:
            return _try_the_well_package(dataset_name, n)
        except Exception as e:
            errors.append(f"the_well package: {e}")

    if backend in (None, "hf_hub"):
        try:
            return _try_hf_hub(dataset_name, n, files=files,
                               max_samples=max_samples,
                               max_timesteps=max_timesteps)
        except Exception as e:
            errors.append(f"huggingface_hub: {e}")

    raise WellNotAvailable(
        f"Could not load '{dataset_name}'.\n"
        + "\n".join(f"  • {err}" for err in errors)
        + "\n\nInstall: pip install huggingface_hub h5py"
    )


def first_array(frame: dict, *field_names: str) -> np.ndarray:
    """Return the first matching field array from a frame dict."""
    for name in field_names:
        if name in frame and isinstance(frame[name], np.ndarray):
            return np.asarray(frame[name], dtype=float)
    # fallback to first large array
    for v in frame.values():
        if isinstance(v, np.ndarray) and v.ndim >= 2:
            return np.asarray(v, dtype=float)
    raise KeyError(f"No array field found (keys: {list(frame.keys())})")
