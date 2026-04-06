"""
The Well dataset loader
========================

Tries three backends in order:
  1. ``the_well`` package (local HDF5 download)
  2. HuggingFace ``datasets`` library (streaming, no full download)
  3. Raises ``WellNotAvailable`` so validators can skip gracefully.

Usage
-----
>>> frames = load_well_frames("gray_scott_reaction_diffusion", n=2)
>>> # frames is a list of dicts {field_name: np.ndarray, ...}
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class WellNotAvailable(RuntimeError):
    """Raised when The Well data cannot be loaded."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_the_well_package(dataset_name: str, n: int) -> list[dict]:
    """Load via the official ``the_well`` Python package."""
    from the_well.data import WellDataset  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore

    ds = WellDataset(
        well_base_path=None,           # uses default cache
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


def _try_huggingface(dataset_name: str, n: int) -> list[dict]:
    """Load via HuggingFace ``datasets`` with streaming (no full download)."""
    from datasets import load_dataset  # type: ignore

    hf_name = f"polymathic-ai/{dataset_name}"
    ds = load_dataset(hf_name, split="test", streaming=True,
                      trust_remote_code=True)
    frames = []
    for i, sample in enumerate(ds):
        if i >= n:
            break
        frame = {}
        for k, v in sample.items():
            arr = np.asarray(v, dtype=float)
            frame[k] = arr
        frames.append(frame)
    return frames


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_well_frames(dataset_name: str, n: int = 1,
                     backend: Optional[str] = None) -> list[dict]:
    """Return *n* sample frames from a Well dataset.

    Parameters
    ----------
    dataset_name : str
        One of the 16+ Well dataset names, e.g. ``"gray_scott_reaction_diffusion"``.
    n : int
        Number of frames to load (default 1 — enough for single-sample stats).
    backend : str, optional
        Force ``"the_well"`` or ``"huggingface"``.  Auto-detected if None.

    Returns
    -------
    list[dict[str, np.ndarray]]
        Each dict maps field name → numpy array (spatial dims last).

    Raises
    ------
    WellNotAvailable
        If neither backend is installed or data is unreachable.
    """
    errors = []

    if backend in (None, "the_well"):
        try:
            return _try_the_well_package(dataset_name, n)
        except Exception as e:
            errors.append(f"the_well package: {e}")

    if backend in (None, "huggingface"):
        try:
            return _try_huggingface(dataset_name, n)
        except Exception as e:
            errors.append(f"HuggingFace datasets: {e}")

    raise WellNotAvailable(
        f"Could not load '{dataset_name}' from any backend.\n"
        + "\n".join(f"  • {err}" for err in errors)
        + "\n\nInstall: pip install the_well  OR  pip install datasets"
    )


def first_array(frame: dict, *field_names: str) -> np.ndarray:
    """Return the first matching field from a frame dict.

    Tries each name in *field_names*; falls back to first available key.
    """
    for name in field_names:
        if name in frame:
            return np.asarray(frame[name], dtype=float)
    # fall back to first numeric array
    for v in frame.values():
        arr = np.asarray(v, dtype=float)
        if arr.ndim >= 1:
            return arr
    raise KeyError(f"No array field found in frame (keys: {list(frame.keys())})")
