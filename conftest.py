"""
Pytest configuration for BPR-Math-Spine.

Sets BLAS/threading env vars BEFORE any test imports to avoid numpy segfaults
on macOS with Accelerate framework (numpy _mac_os_check / polyfit crash).
Must be first in the file.
"""
import os

# Set before any numpy import - prevents segfault on macOS + Python 3.9
# when numpy uses Accelerate BLAS for _mac_os_check
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
