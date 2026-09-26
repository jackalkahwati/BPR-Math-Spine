"""Yukawa mechanisms for the three flux families of BPR-6D: brane-localized and bulk-vector Higgs fields.

See doc/derivations/yukawa_mechanisms_2026-09-26.md. The families are the spin-1
triplet of the sphere's SU(2) isometry (round 3). Two candidate sources of
Yukawa couplings, once a Higgs field is supplied:

1. A Higgs localized at a point of S^2 (a codimension-2 brane). The point is
   fixed by rotations about its axis (J_z), so a J_z-neutral brane Higgs couples
   only J_z-neutral family pairs. The invariant symmetric matrices are computed
   from Wigner matrices, and the vanishing orders of the zero modes at the
   point from the explicit spin-weighted harmonics.
2. A bulk internal one-form 10 / 126 of F-charge -6 (parent charge 3): the J=2
   channel of round 3, now as a non-gauge (Proca) field with gyromagnetic ratio
   g, whose lowest level must be tuned to the weak scale.
"""

from itertools import product

import numpy as np

try:
    from .sphere_family_structure import swsh, theta, phi, mass_spectrum, vev_for_spectrum
except ImportError:  # loaded as a top-level module by the demo script
    from sphere_family_structure import swsh, theta, phi, mass_spectrum, vev_for_spectrum

MODEL_ID = "bpr6d-yukawa-mechanisms-v1"

LIMITATIONS = [
    "The Higgs field is supplied in every mechanism: BPR-6D contains no field with the required quantum numbers.",
    "Brane couplings are treated through symmetry (J_z) and vanishing orders; brane dynamics, tension and backreaction are not modelled.",
    "The derivative-suppression scale epsilon = 1/(M_* r) of brane operators is a parameter, not derived.",
    "The bulk-vector route assumes a Proca field with free gyromagnetic ratio; its UV consistency is not addressed.",
]


# ---------------------------------------------------------------------------
# Spin-1 rotation matrices in the basis m = 1, 0, -1
# ---------------------------------------------------------------------------

def spin1_generators():
    Jz = np.diag([1.0, 0.0, -1.0]).astype(complex)
    Jp = np.zeros((3, 3), complex)
    Jp[0, 1] = Jp[1, 2] = np.sqrt(2.0)
    Jx = (Jp + Jp.conj().T) / 2
    Jy = (Jp - Jp.conj().T) / (2j)
    return Jx, Jy, Jz


def rotation(axis_angle):
    from scipy.linalg import expm
    Jx, Jy, Jz = spin1_generators()
    ax, ay, az = axis_angle
    return expm(-1j * (ax * Jx + ay * Jy + az * Jz))


def invariant_symmetric_matrices(generator_index=2, samples=(0.37, 1.1, 2.3)):
    """Basis of complex symmetric Y with R^T Y R = Y for all rotations about one axis (default z).

    Solved as a linear nullspace over the 6 symmetric entries, using explicit rotation matrices.
    """
    pairs = [(i, j) for i in range(3) for j in range(i, 3)]
    rows = []
    for angle in samples:
        vec = [0.0, 0.0, 0.0]
        vec[generator_index] = angle
        R = rotation(vec)
        for (i, j) in pairs:
            row = []
            for (k, l) in pairs:
                E = np.zeros((3, 3), complex)
                E[k, l] = E[l, k] = 1.0
                row.append((R.T @ E @ R - E)[i, j])
            rows.append(row)
    A = np.array(rows)
    _, sv, vh = np.linalg.svd(A)
    null = vh[sv.size - int(np.sum(sv < 1e-10)):] if np.sum(sv < 1e-10) else np.zeros((0, 6))
    basis = []
    for v in null:
        Y = np.zeros((3, 3), complex)
        for coeff, (k, l) in zip(v, pairs):
            Y[k, l] = Y[l, k] = coeff
        basis.append(Y)
    return basis


def twisted_invariant_symmetric_matrices(c, samples=(0.37, 1.1, 2.3)):
    """Symmetric Y with R_z(a)^T Y R_z(a) = exp(-i c a) Y: a brane field of J_z charge c at the pole.

    The J_z charge of a brane Higgs depends on its normal-bundle spin and on the lift of rotations to the
    U(1)_F bundle at the point; c is left free and every value is analysed.
    """
    pairs = [(i, j) for i in range(3) for j in range(i, 3)]
    rows = []
    for angle in samples:
        R = rotation([0.0, 0.0, angle])
        for (i, j) in pairs:
            row = []
            for (k, l) in pairs:
                E = np.zeros((3, 3), complex)
                E[k, l] = E[l, k] = 1.0
                row.append((R.T @ E @ R - np.exp(-1j * c * angle) * E)[i, j])
            rows.append(row)
    A = np.array(rows)
    _, sv, vh = np.linalg.svd(A)
    k_null = int(np.sum(sv < 1e-10))
    basis = []
    for v in (vh[sv.size - k_null:] if k_null else []):
        Y = np.zeros((3, 3), complex)
        for coeff, (k, l) in zip(v, pairs):
            Y[k, l] = Y[l, k] = coeff
        basis.append(Y)
    return basis


def single_brane_spectrum_pattern(c, rng, trials=20, tol=1e-9):
    """Classify single-brane spectra for J_z charge c: 'zero', 'rank1', 'pair+zero', 'pair+one', or 'distinct'."""
    basis = twisted_invariant_symmetric_matrices(c)
    if not basis:
        return "zero"
    patterns = set()
    for _ in range(trials):
        Y = sum((rng.normal() + 1j * rng.normal()) * B for B in basis)
        sv = sorted(np.linalg.svd(Y, compute_uv=False), reverse=True)
        nonzero = [x for x in sv if x > tol * max(sv)]
        degenerate = any(abs(a - b) < tol * max(sv) for a, b in zip(nonzero, nonzero[1:]))
        if len(nonzero) == 1:
            patterns.add("rank1")
        elif len(nonzero) == 2:
            patterns.add("pair+zero" if degenerate else "two_distinct")
        else:
            patterns.add("pair+one" if degenerate else "distinct")
    return "/".join(sorted(patterns))


def brane_spectrum(rng):
    """Singular values of a random J_z-invariant symmetric Yukawa (a single brane at the pole)."""
    basis = invariant_symmetric_matrices()
    Y = sum((rng.normal() + 1j * rng.normal()) * B for B in basis)
    return sorted(np.linalg.svd(Y, compute_uv=False), reverse=True), len(basis)


def vanishing_orders(k=3, small=(1e-3, 2e-3)):
    """Order n_m with |f_m(theta)| ~ theta^n near the north pole, for the zero modes f_m = (-j)Y_{j,m}."""
    import sympy as sp
    j = (k - 1) // 2
    out = {}
    for m in range(j, -j - 1, -1):
        f = sp.lambdify((theta, phi), swsh(sp.Integer(-j), sp.Integer(j), sp.Integer(m)), "numpy")
        a, b = (abs(complex(f(t, 0.4))) for t in small)
        out[m] = int(round(np.log(b / a) / np.log(small[1] / small[0]))) if a > 1e-14 else None
    return out


def brane_suppression_orders(k=3):
    """Each J_z-allowed entry (m, -m) needs n_m + n_{-m} derivatives at the brane: the same order for all."""
    orders = vanishing_orders(k)
    j = (k - 1) // 2
    return {"{},{}".format(m, -m): orders[m] + orders[-m] for m in range(j, -j - 1, -1)}


def two_brane_spectrum(rng, gamma=1.0):
    """Sum of two single-brane Yukawas at points separated by angle gamma (rotation about x)."""
    basis = invariant_symmetric_matrices()
    Y1 = sum((rng.normal() + 1j * rng.normal()) * B for B in basis)
    Y2 = sum((rng.normal() + 1j * rng.normal()) * B for B in basis)
    R = rotation([gamma, 0.0, 0.0])
    Y = Y1 + R.T @ Y2 @ R
    return sorted(np.linalg.svd(Y, compute_uv=False), reverse=True)


# ---------------------------------------------------------------------------
# Bulk internal-vector Higgs (Proca, gyromagnetic ratio g)
# ---------------------------------------------------------------------------

def proca_lowest_level(g_ratio, mass_r_squared, n=6):
    """m^2 r^2 of the aligned lowest level: |s_e| + 1 - g |n| / 2 + M^2 r^2 with |s_e| = |n|/2 - 1.

    For g = 2 and M = 0 this is the Yang-Mills value -|n|/2 of round 3 (tested there against Atiyah-Bott
    and a finite-difference Hessian). |n| = 6 for an F-charge -6 field in unit flux.
    """
    s_e = abs(n) / 2 - 1
    return s_e + 1 - g_ratio * abs(n) / 2 + mass_r_squared


def higgs_tuning(inverse_radius_gev=1.6e17, weak_gev=125.0):
    """Relative tuning of the bulk mass needed to put the J=2 level at the weak scale."""
    return (weak_gev / inverse_radius_gev) ** 2


def demonstration_report():
    rng = np.random.default_rng(0)
    single = [brane_spectrum(rng)[0] for _ in range(5)]
    double = [two_brane_spectrum(rng) for _ in range(5)]
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "status": "no_natural_yukawa_mechanism",
        "empirical_validation": False,
        "jz_invariant_dimension": len(invariant_symmetric_matrices()),
        "single_brane_patterns_by_c": {str(c): single_brane_spectrum_pattern(c, np.random.default_rng(c + 10))
                                       for c in range(-4, 5)},
        "single_brane_spectra": [[float(x) for x in s] for s in single],
        "vanishing_orders": vanishing_orders(),
        "brane_suppression_orders": brane_suppression_orders(),
        "two_brane_spectra": [[float(x) for x in s] for s in double],
        "proca_threshold_g2": 3.0, "proca_level_g2_M0": proca_lowest_level(2, 0),
        "higgs_mass_tuning": higgs_tuning(),
        "limitations": list(LIMITATIONS),
    }
