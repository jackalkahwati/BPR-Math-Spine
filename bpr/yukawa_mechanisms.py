"""Yukawa mechanisms for the three flux families of BPR-6D: brane-localized and bulk-vector Higgs fields.

See doc/derivations/yukawa_mechanisms_2026-09-26.md. The families are the spin-1
triplet of the sphere's SU(2) isometry (round 3). Two candidate sources of
Yukawa couplings, once a Higgs field is supplied:

1. A Higgs localized at a point of S^2 (a codimension-2 brane). The point is
   fixed by rotations about its axis (J_z), so a brane Higgs of J_z charge c
   couples only family pairs with m + m' = c. For an F-charge -6 brane field of
   normal-bundle spin weight s_h, c = s_h + 3 (the Wu-Yang shift of a charged
   field at a monopole pole): a brane scalar (c = 3) never couples, and a
   normal-bundle vector (s_h = -1, c = 2) gives rank 1. The covariant matrices
   are computed from Wigner matrices, and the pole behaviour of the zero modes
   and their eth-derivatives from the explicit spin-weighted harmonics.
2. A bulk internal one-form 10 / 126 of F-charge -6 (parent charge 3): the J=2
   channel of round 3, now as a non-gauge (Proca) field with gyromagnetic ratio
   g, whose lowest level must be tuned to the weak scale.
"""

import numpy as np

try:
    from .sphere_family_structure import swsh, eth, eth_bar, theta, phi
except ImportError:  # loaded as a top-level module by the demo script
    from sphere_family_structure import swsh, eth, eth_bar, theta, phi

MODEL_ID = "bpr6d-yukawa-mechanisms-v2"

LIMITATIONS = [
    "The Higgs field is supplied in every mechanism: BPR-6D contains no field with the required quantum numbers.",
    "Brane couplings are treated through symmetry (J_z) and vanishing orders; brane dynamics, tension and backreaction are not modelled.",
    "The derivative-suppression scale epsilon = 1/(M_* r), the J_z-breaking size and brane separations are parameters, not derived.",
    "Brane-localized (Aharonov-Bohm) flux would shift the J_z charges and vanishing orders by fractions; it is not included.",
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


def twisted_invariant_symmetric_matrices(c, samples=(0.37, 1.1, 2.3), antisymmetric=False):
    """Y with R_z(a)^T Y R_z(a) = exp(-i c a) Y: a brane field of J_z charge c at the pole.

    Symmetric for a 10 or 126 Higgs; antisymmetric=True gives the 120. See brane_jz_charge for c.
    """
    sign = -1.0 if antisymmetric else 1.0
    pairs = [(i, j) for i in range(3) for j in range(i + (1 if antisymmetric else 0), 3)]
    rows = []
    for angle in samples:
        R = rotation([0.0, 0.0, angle])
        for (i, j) in pairs:
            row = []
            for (k, l) in pairs:
                E = np.zeros((3, 3), complex)
                E[k, l] = 1.0
                E[l, k] = sign
                row.append((R.T @ E @ R - np.exp(-1j * c * angle) * E)[i, j])
            rows.append(row)
    A = np.array(rows)
    _, sv, vh = np.linalg.svd(A)
    k_null = int(np.sum(sv < 1e-10))
    basis = []
    for v in (vh[sv.size - k_null:] if k_null else []):
        Y = np.zeros((3, 3), complex)
        for coeff, (k, l) in zip(v, pairs):
            Y[k, l] = coeff
            Y[l, k] = sign * coeff
        basis.append(Y)
    return basis


def brane_jz_charge(s_h, higgs_f_charge=-6):
    """J_z charge c of a brane field with normal-bundle spin weight s_h and F-charge higgs_f_charge.

    In unit flux a field of F-charge Q picks up spin weight -Q/2 (the fermions, Q = 3, get -3/2 plus
    their spinor weight +1/2, total -1). Only the harmonic with m = -s_total is nonzero at the pole
    (pole_nonzero_m), and J_z invariance of the coupling forces m + m' = s_total of the Higgs.
    """
    return s_h - higgs_f_charge // 2


def pole_nonzero_m(s, l):
    """Azimuthal numbers m for which sY_l,m is nonzero at the north pole (expected: m = -s only)."""
    import sympy as sp
    return [mm for mm in range(-l, l + 1)
            if sp.limit(swsh(sp.Integer(s), sp.Integer(l), sp.Integer(mm)).subs(phi, sp.Rational(3, 10)), theta, 0) != 0]


def derivative_pole_table(k=3, max_a=2):
    """For each zero mode f_m and a = 0..max_a eth-derivatives, whether eth^a f_m is nonzero at the pole.

    Also checks eth-bar f_m = 0 (the zero modes are holomorphic sections), so eth is the only useful
    derivative. Expected: nonzero exactly when m = j - a.
    """
    import sympy as sp
    j = (k - 1) // 2
    table = {}
    for m in range(j, -j - 1, -1):
        f = swsh(sp.Integer(-j), sp.Integer(j), sp.Integer(m))
        assert sp.simplify(eth_bar(f, -j)) == 0
        g, s = f, -j
        for a in range(max_a + 1):
            # Exact limit: floating-point evaluation near theta = 0 suffers catastrophic cancellation.
            table["{},{}".format(m, a)] = sp.limit(g.subs(phi, sp.Rational(3, 10)), theta, 0) != 0
            g, s = eth(g, s), s + 1
    return table


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


def brane_suppression_orders(k=3, c=0):
    """Each J_z-allowed entry (m, c - m) needs n_m + n_{c-m} = 2 - c derivatives: the same order for all."""
    orders = vanishing_orders(k)
    j = (k - 1) // 2
    return {"{},{}".format(m, c - m): orders[m] + orders[c - m]
            for m in range(j, -j - 1, -1) if -j <= c - m <= j}


def branes_spectrum(rng, points, c=2):
    """Singular values of a sum of single-brane Yukawas of J_z charge c at points (polar, azimuth).

    A brane at the pole is moved to (polar, azimuth) by R = R_z(azimuth) R_x(polar); its Yukawa becomes
    D^T Y D with D the spin-1 matrix of R^{-1}.
    """
    basis = twisted_invariant_symmetric_matrices(c)
    Y = np.zeros((3, 3), complex)
    for polar, azimuth in points:
        Yi = sum((rng.normal() + 1j * rng.normal()) * B for B in basis)
        D = rotation([polar, 0.0, 0.0]) @ rotation([0.0, 0.0, azimuth])
        Y = Y + D.T @ Yi @ D
    return sorted(np.linalg.svd(Y, compute_uv=False), reverse=True)


def two_brane_spectrum(rng, gamma=1.0, c=0):
    """Two branes of J_z charge c at the pole and at polar angle gamma."""
    return branes_spectrum(rng, [(0.0, 0.0), (gamma, 0.0)], c=c)


def clustered_brane_scaling(rng, gammas=(0.3, 0.1, 0.03), samples=400, c=2):
    """Three c = 2 branes within angle gamma: median m2/m1 / gamma^2 and m3/m1 / gamma^4.

    Each c = 2 brane is a rank-1 projector onto a spin-1 coherent state v(z) ~ (1, sqrt2 z, z^2); the
    Vandermonde structure gives masses ~ (1, gamma^2, gamma^4), a Froggatt-Nielsen form with eps = gamma.
    """
    out = {}
    for g in gammas:
        sv = np.array([branes_spectrum(rng, [(0.0, 0.0), (g, 0.0), (g, 2 * np.pi / 3)], c=c)
                       for _ in range(samples)])
        out[str(g)] = [float(np.median(sv[:, 1] / sv[:, 0]) / g ** 2), float(np.median(sv[:, 2] / sv[:, 0]) / g ** 4)]
    return out


def fn_texture_spectrum(rng, eps, k=3):
    """J_z broken at the brane (Heckman-Vafa point Yukawa): Y_mm' = a_mm' eps^(n_m + n_m'), O(1) random a."""
    orders = vanishing_orders(k)
    ms = sorted(orders, reverse=True)
    A = rng.normal(size=(len(ms), len(ms))) + 1j * rng.normal(size=(len(ms), len(ms)))
    A = A + A.T
    Y = np.array([[A[i, j] * eps ** (orders[ms[i]] + orders[ms[j]]) for j in range(len(ms))]
                  for i in range(len(ms))])
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
    double_c2 = [two_brane_spectrum(rng, gamma=0.1, c=2) for _ in range(5)]
    fn_sv = [np.array(fn_texture_spectrum(rng, 0.1)) for _ in range(400)]
    fn = np.median([np.log10(sv / sv[0]) for sv in fn_sv], axis=0)
    return {
        "schema_version": 2,
        "model_id": MODEL_ID,
        "status": "yukawas_possible_no_hierarchy_predicted",
        "empirical_validation": False,
        "brane_jz_charge": {"scalar": brane_jz_charge(0), "normal_vector": brane_jz_charge(-1)},
        "jz_invariant_dimension": len(invariant_symmetric_matrices()),
        "single_brane_patterns_by_c": {str(c): single_brane_spectrum_pattern(c, np.random.default_rng(c + 10))
                                       for c in range(-4, 5)},
        "single_brane_spectra_c0": [[float(x) for x in s] for s in single],
        "vanishing_orders": vanishing_orders(),
        "brane_suppression_orders_by_c": {str(c): brane_suppression_orders(c=c) for c in (0, 1, 2)},
        "two_c2_branes_gamma_0.1": [[float(x) for x in s] for s in double_c2],
        "three_c2_branes_scaling": clustered_brane_scaling(rng, samples=200),
        "fn_texture_eps_0.1_median_log10_ratios": [float(x) for x in fn],
        "proca_M2r2_tuning_at_g2": -proca_lowest_level(2, 0), "proca_level_g2_M0": proca_lowest_level(2, 0),
        "higgs_mass_tuning": higgs_tuning(),
        "limitations": list(LIMITATIONS),
    }
