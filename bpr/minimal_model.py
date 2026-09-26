"""Phase 1a: a minimal complete specification of BPR-6D ("BPR-6D-M") and its structural checks.

See doc/derivations/minimal_model_2026-09-26.md. BPR-6D fixes gravity, Spin(10) x U(1)_F, the matter
16_+(3) + 16_-(0), the Green-Schwarz 2-form and the flux sphere. Rounds 7, 8 and 10 showed that a Higgs sector
must be supplied, and constrained it:
- every Higgs that gives Yukawas has F-charge -6 (so U(1)_F acts as a Peccei-Quinn symmetry);
- a brane Higgs couples only if its normal-bundle spin weight is s_h = -1 (J_z charge c = s_h + 3 = 2, rank 1);
- a viable QCD axion needs an F-charged singlet S linked to the doublets;
- geometric Spin(10) breaking is obstructed, so a breaking Higgs is needed.

This module writes down the smallest content meeting all of these, and checks:
1. every intended coupling is allowed and the dangerous ones are forbidden (F-charge, normal weight, SO(10));
2. the vevs break Spin(10) exactly to SU(3) x SU(2) x U(1)_Y (explicit 10-vector and 5-form representations);
3. c = 2 branes produce only the J = 2 part of a Yukawa matrix, and how many branes realize generic Yukawas
   modulo U(3) family redefinitions (Jacobian ranks; three branes obey one Bargmann-phase relation);
4. the branes break the SU(2) family isometry completely.
"""

from itertools import combinations

import numpy as np
from scipy.linalg import expm

try:
    from .yukawa_mechanisms import brane_jz_charge
except ImportError:  # loaded as a top-level module by the demo script
    from yukawa_mechanisms import brane_jz_charge

MODEL_ID = "bpr6d-minimal-model-v1"
N_BRANES = 4

LIMITATIONS = [
    "The Higgs content is chosen, not derived: it is the smallest content meeting the constraints of rounds 7, 8 and 10.",
    "The Spin(10)-breaking potential is not minimized here; for 45 + 126 a viable vacuum needs one-loop effects (cited).",
    "Fits of the charged-fermion and neutrino data with generic 10 + 126bar Yukawas are cited, not redone.",
    "Brane positions are classical moduli; their stabilization and the O(tension) distortion of the zero modes are not computed.",
    "SO(10) invariants are taken from standard tensor-product tables (Slansky 1981), not recomputed.",
]


# ---------------------------------------------------------------------------
# 1. Field content and coupling selection rules
# ---------------------------------------------------------------------------

FIELDS = {
    # name: (location, SO(10) representation, U(1)_F charge, normal-bundle spin weight, multiplicity, role)
    "16+": ("bulk", "16", 3, None, 1, "chiral matter; three zero-mode families per flux quantum"),
    "16-": ("bulk", "16", 0, None, 1, "anomaly completion; no zero modes"),
    "45H": ("bulk", "45", 0, 0, 1, "breaks Spin(10) to SU(3) x SU(2)_L x SU(2)_R x U(1)_B-L (zero mode exists: F = 0)"),
    "10H": ("brane", "10", -6, -1, N_BRANES, "complex; Higgs doublets and Yukawas (c = 2)"),
    "126barH": ("brane", "126bar", -6, -1, N_BRANES, "breaks B-L at M_I; Yukawas and nu^c masses (c = 2)"),
    "S": ("brane", "1", 6, 1, 1, "Peccei-Quinn singlet; links the doublet phases (DFSZ)"),
}

CONJUGATE = {"1": "1", "10": "10", "45": "45", "16": "16bar", "16bar": "16", "126": "126bar", "126bar": "126"}

# Products containing an SO(10) singlet, as sorted tuples (Slansky, Phys. Rep. 79 (1981) 1). Symmetric products of a
# single boson are listed only where the symmetric part contains the singlet.
SO10_INVARIANTS = {
    ("10", "16", "16"), ("126bar", "16", "16"),          # 16 x 16 = 10_s + 120_a + 126_s
    ("10", "10"), ("1", "1"), ("1", "10", "10"), ("1", "1", "10", "10"),  # Sym^2(10) = 1 + 54
    ("126", "126bar"), ("126", "126bar", "45"),          # 126 x 126bar = 1 + 45 + 210 + 770 + 3150 + 5940
    ("45", "45"),
}


def _term_rep_key(reps):
    return tuple(sorted(reps))


def coupling(term):
    """Check a term given as a list of (field, conjugated) pairs.

    Returns the F-charge sum, the normal-weight sum over brane fields, whether an SO(10) invariant exists, and
    whether the term is allowed. Terms containing the bulk fermion zero modes are Yukawas; for them the brane Higgs
    must have J_z charge c = s_h + 3 = 2 (round 8), i.e. s_h = -1, and the fermion weights are accounted for by c.
    """
    F = 0
    weight = 0
    reps = []
    fermions = [f for f, _ in term if f in ("16+", "16-")]
    for name, conj in term:
        loc, rep, q, w, _, _ = FIELDS[name]
        F += -q if conj else q
        reps.append(CONJUGATE[rep] if conj else rep)
        if loc == "brane":
            weight += -w if conj else w
    so10 = _term_rep_key(reps) in SO10_INVARIANTS
    if fermions:
        higgs = [(n, c) for n, c in term if n not in ("16+", "16-")]
        # Yukawa 16 16 H: the Higgs must be a c = 2 brane field (F-charge -6, s_h = -1), unconjugated.
        ok_c = (len(higgs) == 1 and not higgs[0][1]
                and brane_jz_charge(FIELDS[higgs[0][0]][3], FIELDS[higgs[0][0]][2]) == 2)
        return {"F": F, "normal_weight": None, "so10_invariant": so10, "allowed": bool(F == 0 and so10 and ok_c)}
    return {"F": F, "normal_weight": weight, "so10_invariant": so10, "allowed": bool(F == 0 and weight == 0 and so10)}


def coupling_table():
    """The intended couplings (allowed) and the dangerous ones (forbidden)."""
    wanted = {
        "Yukawa 16 16 10H": [("16+", False), ("16+", False), ("10H", False)],
        "Yukawa 16 16 126barH (incl. nu^c Majorana)": [("16+", False), ("16+", False), ("126barH", False)],
        "DFSZ link S^2 10H 10H": [("S", False), ("S", False), ("10H", False), ("10H", False)],
        "mass 10H^dag 10H": [("10H", True), ("10H", False)],
        "breaking 126barH^dag 45H 126barH": [("126barH", True), ("45H", False), ("126barH", False)],
    }
    dangerous = {
        "conjugate Yukawa 16 16 10H*": [("16+", False), ("16+", False), ("10H", True)],
        "mu-term H_u H_d = 10H 10H": [("10H", False), ("10H", False)],
        "S 10H 10H (needs F_S = +12)": [("S", False), ("10H", False), ("10H", False)],
    }
    return ({k: coupling(v) for k, v in wanted.items()}, {k: coupling(v) for k, v in dangerous.items()})


def brane_scalar_higgs_is_useless():
    """A brane scalar (s_h = 0) of F-charge -6 has c = 3: no Yukawa (round 8)."""
    return brane_jz_charge(0, -6) == 3


# ---------------------------------------------------------------------------
# 2. Spin(10) -> SU(3) x SU(2) x U(1)_Y from 45 + 126bar vevs
# ---------------------------------------------------------------------------

def so10_generators():
    """45 real antisymmetric 10 x 10 generators M_ij (i < j); plane k = (2k, 2k+1) carries the Cartan H_k."""
    gens = []
    for i, j in combinations(range(10), 2):
        M = np.zeros((10, 10))
        M[i, j], M[j, i] = -1.0, 1.0
        gens.append(M)
    return gens


def cartan(vector):
    """sum_k h_k H_k with H_k the rotation generator of plane (2k, 2k+1)."""
    A = np.zeros((10, 10))
    for k, h in enumerate(vector):
        A[2 * k, 2 * k + 1], A[2 * k + 1, 2 * k] = -h, h
    return A


def _five_form_basis():
    basis = list(combinations(range(10), 5))
    return basis, {S: n for n, S in enumerate(basis)}


def five_form_action(M):
    """Matrix of the derivation action of a 10 x 10 generator on Lambda^5(C^10) (252-dimensional)."""
    basis, index = _five_form_basis()
    R = np.zeros((len(basis), len(basis)))
    for col, S in enumerate(basis):
        for pos, i in enumerate(S):
            for j in range(10):
                if M[j, i] == 0 or (j in S and j != i):
                    continue
                new = list(S)
                new[pos] = j
                order = np.argsort(new)
                sign = np.linalg.det(np.eye(5)[order])  # permutation sign
                R[index[tuple(np.array(new)[order])], col] += M[j, i] * sign
    return R


def holomorphic_five_form():
    """Omega = wedge_k (e_{2k} + i e_{2k+1}): the SU(5)-singlet direction of the 126 / 126bar (weight +-X)."""
    basis, index = _five_form_basis()
    v = np.zeros(len(basis), complex)
    for choice in range(32):
        idx, coeff = [], 1.0 + 0j
        for k in range(5):
            if (choice >> k) & 1:
                idx.append(2 * k + 1)
                coeff *= 1j
            else:
                idx.append(2 * k)
        v[index[tuple(idx)]] += coeff
    return v


def stabilizer_dimension(adjoint_vev=None, five_form_vev=None):
    """dim of {x in so(10): [x, A] = 0 and x . Omega = 0}, by a real nullspace computation."""
    gens = so10_generators()
    rows = []
    if adjoint_vev is not None:
        cols = [(g @ adjoint_vev - adjoint_vev @ g).ravel() for g in gens]
        rows.append(np.array(cols).T)
    if five_form_vev is not None:
        cols = [five_form_action(g) @ five_form_vev for g in gens]
        C = np.array(cols).T
        rows.extend([C.real, C.imag])
    if not rows:
        return len(gens), None
    A = np.vstack(rows)
    _, s, vh = np.linalg.svd(A)
    rank = int(np.sum(s > 1e-9 * s[0]))
    return len(gens) - rank, vh[rank:]


def breaking_pattern(bl_vev=1.0, t3r_vev=0.0, with_126=True):
    """Unbroken dimension for 45H = bl (B-L direction) + t3r (T3R direction), with or without the 126bar vev.

    Expected: B-L alone 15 (3221), T3R alone 19 (421), both 13 (3211); with the 126bar vev 12 (321), and the
    unbroken Cartan direction is the hypercharge Y = (-1/3, -1/3, -1/3, 1/2, 1/2).
    """
    A = cartan([bl_vev, bl_vev, bl_vev, t3r_vev, t3r_vev])
    Omega = holomorphic_five_form() if with_126 else None
    dim, null = stabilizer_dimension(A, Omega)
    out = {"unbroken_dimension": dim}
    if with_126:
        # Which Cartan combinations survive: a (H1+H2+H3) + b (H4+H5) annihilates Omega iff 3a + 2b = 0.
        test = [cartan(v) for v in ([-1 / 3] * 3 + [1 / 2] * 2, [1 / 3] * 3 + [1 / 2] * 2, [1.0] * 5)]
        names = ["Y", "Y_flipped", "X"]
        out["cartan_unbroken"] = {n: bool(np.allclose(five_form_action(t) @ Omega, 0)) for n, t in zip(names, test)}
    return out


# ---------------------------------------------------------------------------
# 3. Brane Yukawas: J = 2 only, and how many branes realize generic Yukawas
# ---------------------------------------------------------------------------

def coherent_state(z):
    """Spin-1 coherent state at stereographic point z (m = 1, 0, -1): the c = 2 brane couples f_1 f_1 at the pole,
    and a brane at z gives Y proportional to u(z) u(z)^T."""
    return np.array([1.0, np.sqrt(2) * z, z * z]) / (1 + abs(z) ** 2)


def brane_matrix(z):
    u = coherent_state(z)
    return np.outer(u, u)


_SYM = [(i, j) for i in range(3) for j in range(i, 3)]


def _vec(Y):
    return np.array([Y[i, j] for i, j in _SYM])


def c2_brane_span_dimension(samples=12, seed=0):
    """Complex dimension of span{u(z) u(z)^T}: 5, i.e. only the J = 2 part of Sym^2(spin 1) = J0 + J2.

    The missing J = 0 direction is the SU(2)-invariant form: every brane matrix has j0_component(Y) = 0.
    """
    rng = np.random.default_rng(seed)
    zs = rng.normal(size=samples) + 1j * rng.normal(size=samples)
    return int(np.linalg.matrix_rank(np.array([_vec(brane_matrix(z)) for z in zs]), tol=1e-10))


def j0_component(Y):
    """SU(2)-invariant (J = 0) part of a symmetric Y in the m = (1, 0, -1) basis: 2 Y_{1,-1} - Y_{0,0}."""
    return 2 * Y[0, 2] - Y[1, 1]


_U3 = []
for _i in range(3):
    for _j in range(3):
        _E = np.zeros((3, 3), complex)
        if _i == _j:
            _E[_i, _i] = 1j
        elif _i < _j:
            _E[_i, _j], _E[_j, _i] = 1, -1
        else:
            _E[_i, _j], _E[_j, _i] = 1j, 1j
        _U3.append(_E)


def _pair(params, N):
    p = params[:6 * N].reshape(3, N, 2)
    z, c, d = (p[k, :, 0] + 1j * p[k, :, 1] for k in range(3))
    Y1 = sum(c[a] * brane_matrix(z[a]) for a in range(N))
    Y2 = sum(d[a] * brane_matrix(z[a]) for a in range(N))
    return Y1, Y2


def _pair_map(params, N, with_u3):
    Y1, Y2 = _pair(params, N)
    if with_u3:
        U = expm(sum(t * g for t, g in zip(params[6 * N:], _U3)))
        Y1, Y2 = U.T @ Y1 @ U, U.T @ Y2 @ U
    v = np.concatenate([_vec(Y1), _vec(Y2)])
    return np.concatenate([v.real, v.imag])


def realizability_rank(N, with_u3=True, trials=3, seed=1):
    """Real rank of the map (brane positions z_a, couplings c_a of 10H and d_a of 126barH [, U in U(3)]) -> (Y10, Y126).

    Generic pairs of complex symmetric matrices form a 24-dimensional real space. Full rank 24 means the branes,
    together with U(3) redefinitions of the three families (which leave all observables unchanged), reach an open
    set of generic pairs.
    """
    rng = np.random.default_rng(seed)
    n = 6 * N + (9 if with_u3 else 0)
    ranks = []
    for _ in range(trials):
        x = np.concatenate([rng.normal(size=6 * N), np.zeros(9) if with_u3 else []])
        f0 = _pair_map(x, N, with_u3)
        J = np.array([(_pair_map(x + 1e-6 * e, N, with_u3) - f0) / 1e-6 for e in np.eye(n)]).T
        s = np.linalg.svd(J, compute_uv=False)
        ranks.append(int(np.sum(s > 1e-6 * s[0])))
    return max(ranks)


def bargmann_relation(samples=20, seed=2):
    """Three c = 2 branes: the U(3)-invariant phase of the coherent-state Gram matrix is fixed by the distances.

    arg(<1|2><2|3><3|1>) for spin-1 coherent states equals -Omega (spin j = 1 times the solid angle of the
    geodesic triangle, up to orientation), and Omega follows from the three side lengths (l'Huilier). This is the
    one real relation left for three branes (rank 23 of 24 modulo U(3)). Returns the largest mismatch.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(samples):
        zs = rng.normal(size=3) + 1j * rng.normal(size=3)
        vs = [coherent_state(z) / np.linalg.norm(coherent_state(z)) for z in zs]
        phase = np.angle(np.vdot(vs[0], vs[1]) * np.vdot(vs[1], vs[2]) * np.vdot(vs[2], vs[0]))
        # Unit vectors on S^2 from stereographic z (north pole z = 0).
        pts = [np.array([2 * z.real, 2 * z.imag, 1 - abs(z) ** 2]) / (1 + abs(z) ** 2) for z in zs]
        a = np.arccos(np.clip(pts[1] @ pts[2], -1, 1))
        b = np.arccos(np.clip(pts[0] @ pts[2], -1, 1))
        c = np.arccos(np.clip(pts[0] @ pts[1], -1, 1))
        s = (a + b + c) / 2
        E = 4 * np.arctan(np.sqrt(max(np.tan(s / 2) * np.tan((s - a) / 2) * np.tan((s - b) / 2) * np.tan((s - c) / 2), 0)))
        worst = max(worst, min(abs(np.angle(np.exp(1j * (abs(phase) - E)))), abs(np.angle(np.exp(1j * (abs(phase) + E - 2 * np.pi))))))
    return worst


def _raw_brane_matrix(z):
    u = np.array([1.0, np.sqrt(2) * z, z * z])
    return np.outer(u, u)


def _quartic(functional):
    """Coefficients (ascending) of the quartic sum_k l_k [u(z) u(z)^T]_k in z."""
    base = [np.array([1.0]), np.array([0, np.sqrt(2)]), np.array([0, 0, 1.0])]
    coeff = np.zeros(5, complex)
    for k, (i, j) in enumerate(_SYM):
        p = np.convolve(base[i], base[j])
        coeff[:len(p)] += functional[k] * p
    return coeff


def realize_yukawa_pair(Y10, Y126, seed=0, starts=40):
    """Exact four-brane realization of a target pair (Y10, Y126), modulo a U(3) family redefinition.

    Stage A: find U in U(3) removing the J = 0 part of both U^T Y10 U and U^T Y126 U (4 real conditions on U(3)).
    Stage B (algebraic): any hyperplane of the J = 2 space containing both matrices meets the rational normal quartic
    {u(z) u(z)^T} in four points; the roots z_a of that quartic are the brane positions, and the couplings follow
    linearly. Returns positions (stereographic z, unit vectors), couplings and relative residuals.
    """
    from scipy.optimize import least_squares
    rng = np.random.default_rng(seed)
    n1, n2 = np.linalg.norm(Y10), np.linalg.norm(Y126)

    def unitary(t):
        return expm(sum(x * g for x, g in zip(t, _U3)))

    def residual(t):
        U = unitary(t)
        r = np.array([j0_component(U.T @ Y10 @ U) / n1, j0_component(U.T @ Y126 @ U) / n2])
        return np.concatenate([r.real, r.imag])

    best = None
    for _ in range(starts):
        sol = least_squares(residual, rng.normal(size=9), xtol=1e-15, ftol=1e-15, gtol=1e-15)
        if best is None or sol.cost < best.cost:
            best = sol
        if best.cost < 1e-28:
            break
    U = unitary(best.x)
    A, B = U.T @ Y10 @ U, U.T @ Y126 @ U
    _, _, vh = np.linalg.svd(np.array([_vec(A), _vec(B)]))
    null = vh[2:].conj()
    for _ in range(20):
        functional = (rng.normal(size=4) + 1j * rng.normal(size=4)) @ null
        z = np.roots(_quartic(functional)[::-1])
        if len(z) == 4 and min(abs(a - b) for a, b in combinations(z, 2)) > 1e-6:
            break
    Ps = np.array([_vec(_raw_brane_matrix(zz)) for zz in z]).T
    c, *_ = np.linalg.lstsq(Ps, _vec(A), rcond=None)
    d, *_ = np.linalg.lstsq(Ps, _vec(B), rcond=None)
    Ui = U.conj().T
    Y10_back = Ui.T @ sum(ca * _raw_brane_matrix(zz) for ca, zz in zip(c, z)) @ Ui
    Y126_back = Ui.T @ sum(da * _raw_brane_matrix(zz) for da, zz in zip(d, z)) @ Ui
    points = [np.array([2 * zz.real, 2 * zz.imag, 1 - abs(zz) ** 2]) / (1 + abs(zz) ** 2) for zz in z]
    return {"U": U, "stageA_cost": float(best.cost), "z": z, "points": points, "c": c, "d": d,
            "Y10_error": float(abs(Y10_back - Y10).max() / abs(Y10).max()),
            "Y126_error": float(abs(Y126_back - Y126).max() / abs(Y126).max()),
            "Y10_back": Y10_back, "Y126_back": Y126_back}


# ---------------------------------------------------------------------------
# 4. Family isometry breaking
# ---------------------------------------------------------------------------

def isometry_stabilizer_dimension(points):
    """dim of so(3) rotations fixing every brane point (unit vectors): omega x p = 0 for all p."""
    rows = []
    for p in points:
        cross = np.array([[0, p[2], -p[1]], [-p[2], 0, p[0]], [p[1], -p[0], 0]])  # omega -> omega x p
        rows.append(cross)
    return 3 - int(np.linalg.matrix_rank(np.vstack(rows), tol=1e-10))


def _hierarchical_example(seed=5):
    """Up-like Takagi values (1e-5, 3e-3, 1) for Y10 and a generic Y126 of size 0.02, realized by four branes."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    out = realize_yukawa_pair(np.diag([1e-5, 3e-3, 1.0]).astype(complex), 0.02 * (A + A.T))
    sv = np.sort(np.linalg.svd(out["Y10_back"], compute_uv=False))
    return {"Y10_error": out["Y10_error"], "Y126_error": out["Y126_error"], "Y10_takagi_back": sv.tolist(),
            "brane_polar_angles_deg": [float(np.degrees(np.arccos(np.clip(p[2], -1, 1)))) for p in out["points"]]}


def demonstration_report():
    wanted, dangerous = coupling_table()
    rng = np.random.default_rng(4)
    pts = [v / np.linalg.norm(v) for v in rng.normal(size=(N_BRANES, 3))]
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "status": "specified_minimal_model_structural_checks",
        "empirical_validation": False,
        "fields": {k: {"location": v[0], "so10": v[1], "F": v[2], "normal_weight": v[3], "copies": v[4], "role": v[5]}
                   for k, v in FIELDS.items()},
        "wanted_couplings": wanted,
        "forbidden_couplings": dangerous,
        "brane_scalar_useless": brane_scalar_higgs_is_useless(),
        "breaking": {"B-L only (3221)": breaking_pattern(1.0, 0.0, False),
                     "T3R only (421)": breaking_pattern(0.0, 1.0, False),
                     "both (3211)": breaking_pattern(1.0, 0.4, False),
                     "with 126bar (SM)": breaking_pattern(1.0, 0.4, True)},
        "c2_brane_span_dimension": c2_brane_span_dimension(),
        "realizability_rank_mod_U3": {str(N): realizability_rank(N) for N in (2, 3, 4)},
        "bargmann_relation_mismatch": bargmann_relation(),
        "hierarchical_example": _hierarchical_example(),
        "isometry_stabilizer": {"one brane": isometry_stabilizer_dimension(pts[:1]),
                                "two branes": isometry_stabilizer_dimension(pts[:2]),
                                "four branes": isometry_stabilizer_dimension(pts)},
        "limitations": list(LIMITATIONS),
    }
