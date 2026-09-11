"""Lock the preregistered flavor-label scan and its negative outcome."""
import numpy as np
from pathlib import Path

from bpr.flavor_label_scan import (
    CANDIDATES, LABEL_INTS, EIGEN_INTS, N_COMPARISONS, run_scan, interesting,
    score, molien_allowed, octahedron_adjacency,
)

BPR_DIR = Path(__file__).resolve().parent.parent / "bpr"


def test_preregistration_is_fixed():
    assert len(CANDIDATES) == 8
    assert N_COMPARISONS == 2 * len(CANDIDATES)
    assert LABEL_INTS == [1, 4, 24, 30, 59, 283]
    assert EIGEN_INTS == [1, 4, 24, 30, 210, 283, 3481]


def test_molien_octahedral_degrees():
    allowed = molien_allowed((4, 6), cut=20)
    assert allowed == [4, 6, 8, 10, 12, 14, 16, 18, 20]
    assert 9 in molien_allowed((4, 6, 9), cut=20)


def test_octahedron_spectrum():
    A = octahedron_adjacency()
    lap = np.linalg.eigvalsh(np.diag(A.sum(1)) - A)
    assert np.allclose(sorted(lap), [0, 4, 4, 4, 6, 6])


def test_scoring_rule_is_exact_binomial():
    s = score([30, 210, 5], [30, 210, 283])
    assert s["hits"] == [30, 210]
    assert s["density"] == 3 / 300
    assert 0 < s["p_tail"] < 1


def test_scan_outcome_is_negative():
    """The recorded result: nothing survives the Bonferroni-corrected rule.
    If a later change makes a candidate pass, this test must be revisited
    deliberately together with doc/derivations/flavor_label_scan_2026-09.md."""
    scan = run_scan()
    assert interesting(scan) == []
    # the two ell(ell+1) coincidences are present but sub-threshold
    c1 = scan["C1 round S2 Laplacian"]["eigen_test"]
    assert c1["hits"] == [30, 210]
    assert c1["p_tail"] * N_COMPARISONS > 0.05


def test_scan_module_blind_to_glueball_targets():
    src = (BPR_DIR / "flavor_label_scan.py").read_text()
    for leaked in ("1730", "2400", "2590", "1.387", "1.497"):
        assert leaked not in src
