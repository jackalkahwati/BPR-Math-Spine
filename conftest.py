"""
Pytest configuration for BPR-Math-Spine.

Sets BLAS/threading env vars BEFORE any test imports to avoid numpy segfaults
on macOS with Accelerate framework (numpy _mac_os_check / polyfit crash).
Must be first in the file.
"""
import importlib.util
import os
from pathlib import Path

# Set before any numpy import - prevents segfault on macOS + Python 3.9
# when numpy uses Accelerate BLAS for _mac_os_check
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# The helionis/lunarfire test modules import the `helionis` package, which
# lives on the `lunarfire` branch (see README §Repository Branches) and is
# not part of the BPR math spine on main. Skip their collection when the
# package is absent so `pytest -q` is green out of the box, as the README
# promises.
if importlib.util.find_spec("helionis") is None:
    _tests = Path(__file__).parent / "tests"
    collect_ignore = [
        str(p.relative_to(Path(__file__).parent))
        for p in sorted(_tests.glob("test_helionis_*.py"))
    ] + [
        str(p.relative_to(Path(__file__).parent))
        for p in sorted(_tests.glob("test_lunarfire_*.py"))
    ]
