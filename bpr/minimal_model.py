"""Phase 1a: a minimal complete specification of BPR-6D ("BPR-6D-M") and its structural checks.

See doc/derivations/minimal_model_2026-09-26.md. BPR-6D fixes gravity, Spin(10) x U(1)_F, the matter
16_+(3) + 16_-(0), the Green-Schwarz 2-form and the flux sphere. Rounds 7, 8 and 10 showed that a Higgs sector
must be supplied, and constrained it:
- every Higgs that gives Yukawas has F-charge -6 (so U(1)_F acts as a Peccei-Quinn symmetry);
- a brane-localized Higgs operator couples only with normal-bundle spin weight s_h = -1 (J_z charge c = 2, rank 1);
- a viable QCD axion needs an F-charged singlet S linked to the doublets;
- geometric Spin(10) breaking is obstructed, so a breaking Higgs is needed.

Two versions were tested.
A. Brane copies: an independent 10_H and 126bar_H on each of four branes. It fails (the independent review found
   this): fields on different branes do not mix at tree level, so one tuning leaves the light doublet on a single
   brane (rank-1 Yukawas), and a rank-3 Majorana matrix needs three condensing Delta_R, for which the one-loop
   running has no consistent intermediate scale (model_scales.multiplicity_scan).
B. Bulk Higgs (adopted): one bulk 10_H and one bulk 126bar_H of F-charge -6, coupled to the families by
   brane-localized operators 16 16 (dbar Phi)(z_a) at four branes. The lowest level of an F-charge -6 bulk scalar is
   a spin-3 multiplet at m^2 r^2 = 3 (annihilated by eth); dbar Phi has spin weight 2, so the operator has c = 2.
   Brane mass terms split the multiplet, one tuning makes one combination light, and that single combination feeds
   every brane.

Checks: couplings (F-charge, normal weight, SO(10)); the breaking pattern (explicit 10-vector and 5-form
representations); c = 2 couplings give only the J = 2 part of a Yukawa matrix, and four branes realize generic
Yukawas modulo U(3) family redefinitions (exact quartic-root construction; three branes obey a Bargmann relation);
the spin-3 lowest level and the uniqueness and brane weights of the light combination; SU(2) isometry breaking.
"""

from itertools import combinations

import numpy as np
from scipy.linalg import expm

try:
    from .yukawa_mechanisms import brane_jz_charge
except ImportError:  # loaded as a top-level module by the demo script
    from yukawa_mechanisms import brane_jz_charge

MODEL_ID = "bpr6d-minimal-model-v2"
N_BRANES = 4

LIMITATIONS = [
    "The Higgs content is chosen, not derived: it is the smallest content found that meets the constraints of rounds 7, 8 and 10.",
    "The Spin(10)-breaking potential is not minimized here; for 45 + 126 a viable vacuum needs one-loop effects and light 45 pseudo-Goldstones (cited).",
    "Fits of the charged-fermion and neutrino data with generic 10 + 126bar Yukawas are cited, not redone.",
    "Brane positions are classical moduli (five physical ones after the family gauge bosons eat three); they couple to fermions through position-dependent Yukawas, and their stabilization is not addressed: potentially fatal.",
    "Brane-localized fermion kinetic terms and the tension distortion of the zero modes add J = 0 pieces; Proposition 1 holds at leading order.",
    "SO(10) invariants are hand-coded from standard tensor-product tables (Slansky 1981), not recomputed.",
    "Forbidden terms are forbidden perturbatively only: e^{ib} carries F = 12, so terms such as 10H 10H e^{ib} are gauge invariant (the axion-quality question).",
]


# ---------------------------------------------------------------------------
# 1. Field content and coupling selection rules
# ---------------------------------------------------------------------------

FIELDS = {
    # name: (location, SO(10) representation, U(1)_F charge, normal-bundle spin weight, multiplicity, role)
    # Version B (adopted). Bulk fields have no normal weight; brane restrictions of bulk fields are listed as "@b".
    "16+": ("bulk", "16", 3, None, 1, "chiral matter; three zero-mode families per flux quantum"),
    "16-": ("bulk", "16", 0, None, 1, "anomaly completion; no zero modes"),
    "45H": ("bulk", "45", 0, None, 1, "breaks Spin(10) to SU(3) x SU(2)_L x SU(2)_R x U(1)_B-L (zero mode exists: F = 0)"),
    "10H": ("bulk", "10", -6, None, 1, "complex; lowest level a spin-3 multiplet at m^2 r^2 = 3, tuned light"),
    "126barH": ("bulk", "126bar", -6, None, 1, "breaks B-L at M_I; lowest level a spin-3 multiplet"),
    "10H@b": ("brane", "10", -6, 0, N_BRANES, "value of the bulk 10H at a brane (c = 3: no Yukawa)"),
    "dbar10H@b": ("brane", "10", -6, -1, N_BRANES, "eth-bar of the bulk 10H at a brane (c = 2: Yukawa)"),
    "126barH@b": ("brane", "126bar", -6, 0, N_BRANES, "value of the bulk 126barH at a brane"),
    "dbar126barH@b": ("brane", "126bar", -6, -1, N_BRANES, "eth-bar of the bulk 126barH at a brane (Yukawa, Majorana)"),
    "S": ("brane", "1", 6, 0, 1, "Peccei-Quinn singlet on brane 1; DFSZ link S^2 10H 10H there"),
}

FIELDS_BRANE_COPIES = {
    # Version A (rejected): independent Higgs copies on each brane.
    "10H": ("brane", "10", -6, -1, N_BRANES, "one independent copy per brane"),
    "126barH": ("brane", "126bar", -6, -1, N_BRANES, "one independent copy per brane"),
    "S": ("brane", "1", 6, 1, 1, "on one brane"),
}

CONJUGATE = {"1": "1", "10": "10", "45": "45", "16": "16bar", "16bar": "16", "126": "126bar", "126bar": "126"}

# Products containing an SO(10) singlet, as sorted tuples (Slansky, Phys. Rep. 79 (1981) 1). Symmetric products of a
# single boson are listed only where the symmetric part contains the singlet. Hand-coded, not exhaustive.
SO10_INVARIANTS = {
    ("10", "16", "16"), ("126bar", "16", "16"),          # 16 x 16 = 10_s + 120_a + 126_s
    ("10", "10"), ("1", "1"), ("1", "10", "10"), ("1", "1", "10", "10"),  # Sym^2(10) = 1 + 54
    ("126", "126bar"), ("126", "126bar", "45"),          # 126 x 126bar = 1 + 45 + 210 + 770 + 3150 + 5940
    ("45", "45"),
    ("10", "10", "126", "126"),                          # Sym^2(10) and Sym^2(126) share the 54
    ("10", "126bar", "45", "45"),                        # 10 x 126bar contains 210, as does Sym^2(45)
}


def _term_rep_key(reps):
    return tuple(sorted(reps))


def coupling(term, fields=None):
    """Check a term given as a list of (field, conjugated) pairs.

    Returns the F-charge sum, the normal-weight sum over brane fields, whether an SO(10) invariant exists (from the
    table), and whether the term is allowed. A term is either purely bulk (6D Lorentz invariant, no weight) or purely
    brane-localized (weights must cancel). Terms with the fermion zero modes are Yukawas: the Higgs operator at the
    brane must have J_z charge c = s_h + 3 = 2 (round 8), i.e. s_h = -1, unconjugated.
    In F-neutral brane terms the Wu-Yang parts of J_z cancel, so only the normal weights need to cancel.
    """
    fields = fields or FIELDS
    F = 0
    weight = 0
    reps = []
    fermions = [f for f, _ in term if f in ("16+", "16-")]
    for name, conj in term:
        loc, rep, q, w, _, _ = fields[name]
        F += -q if conj else q
        reps.append(CONJUGATE[rep] if conj else rep)
        if loc == "brane":
            weight += -w if conj else w
    so10 = _term_rep_key(reps) in SO10_INVARIANTS
    if fermions:
        higgs = [(n, c) for n, c in term if n not in ("16+", "16-")]
        ok_c = (len(higgs) == 1 and not higgs[0][1] and fields[higgs[0][0]][0] == "brane"
                and brane_jz_charge(fields[higgs[0][0]][3], fields[higgs[0][0]][2]) == 2)
        return {"F": F, "normal_weight": None, "so10_invariant": so10, "allowed": bool(F == 0 and so10 and ok_c)}
    return {"F": F, "normal_weight": weight, "so10_invariant": so10, "allowed": bool(F == 0 and weight == 0 and so10)}


def coupling_table():
    """The intended couplings (allowed) and the dangerous ones (forbidden) of version B."""
    wanted = {
        "Yukawa 16 16 (dbar 10H)@brane": [("16+", False), ("16+", False), ("dbar10H@b", False)],
        "Yukawa and nu^c Majorana 16 16 (dbar 126barH)@brane": [("16+", False), ("16+", False), ("dbar126barH@b", False)],
        "DFSZ link S^2 10H 10H at brane 1": [("S", False), ("S", False), ("10H@b", False), ("10H@b", False)],
        "bulk mass 10H^dag 10H": [("10H", True), ("10H", False)],
        "brane mass |10H|^2 (splits the spin-3 multiplet)": [("10H@b", True), ("10H@b", False)],
        "brane mass |dbar 10H|^2": [("dbar10H@b", True), ("dbar10H@b", False)],
        "breaking 126barH^dag 45H 126barH": [("126barH", True), ("45H", False), ("126barH", False)],
        "10-126 doublet mixing 10H^dag 126barH 45H 45H": [("10H", True), ("126barH", False), ("45H", False), ("45H", False)],
    }
    dangerous = {
        "Yukawa with the undifferentiated value 16 16 10H@brane (c = 3)": [("16+", False), ("16+", False), ("10H@b", False)],
        "conjugate Yukawa 16 16 (dbar 10H)*": [("16+", False), ("16+", False), ("dbar10H@b", True)],
        "mu-term 10H 10H": [("10H", False), ("10H", False)],
        "S 10H 10H (needs F_S = +12)": [("S", False), ("10H@b", False), ("10H@b", False)],
    }
    return ({k: coupling(v) for k, v in wanted.items()}, {k: coupling(v) for k, v in dangerous.items()})


def brane_scalar_higgs_is_useless():
    """A brane scalar (s_h = 0) of F-charge -6 has c = 3: no Yukawa (round 8)."""
    return brane_jz_charge(0, -6) == 3


def brane_copies_light_yukawa_rank(seed=0, trials=200):
    """Version A: with no tree-level mixing between branes the doublet mass matrix is diagonal in the brane index, so
    one tuning makes the light doublet live on a single brane and the tree-level Yukawa is rank 1 (largest rank found).
    """
    rng = np.random.default_rng(seed)
    worst = 0
    for _ in range(trials):
        masses = rng.uniform(0.5, 2.0, N_BRANES)
        masses[rng.integers(N_BRANES)] = 0.0  # the single tuning
        light = np.linalg.eigh(np.diag(masses))[1][:, 0]
        zs = rng.normal(size=N_BRANES) + 1j * rng.normal(size=N_BRANES)
        ys = rng.normal(size=N_BRANES) + 1j * rng.normal(size=N_BRANES)
        Y = sum(ys[a] * light[a] * brane_matrix(zs[a]) for a in range(N_BRANES))
        s = np.linalg.svd(Y, compute_uv=False)
        worst = max(worst, int(np.sum(s > 1e-12 * s[0])))
    return worst


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
    if n1 == 0 or n2 == 0:
        raise ValueError("both Yukawa matrices must be nonzero")

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
    if best is None or not best.cost <= 1e-20:  # also catches NaN
        raise RuntimeError("Step A did not remove the J = 0 parts ({}); unproven for this pair".format(
            "no start" if best is None else "cost {:.1e}".format(best.cost)))
    U = unitary(best.x)
    A, B = U.T @ Y10 @ U, U.T @ Y126 @ U
    _, _, vh = np.linalg.svd(np.array([_vec(A), _vec(B)]))
    null = vh[2:].conj()
    for _ in range(20):
        functional = (rng.normal(size=4) + 1j * rng.normal(size=4)) @ null
        z = np.roots(_quartic(functional)[::-1])
        if len(z) == 4 and min(abs(a - b) for a, b in combinations(z, 2)) > 1e-6:
            break
    else:
        raise RuntimeError("no hyperplane with four distinct brane positions found")
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
# 4. Version B: one bulk Higgs feeding every brane
# ---------------------------------------------------------------------------

def spin_matrices(j):
    d = int(2 * j + 1)
    ms = [j - k for k in range(d)]
    Jz = np.diag(ms).astype(complex)
    Jp = np.zeros((d, d), complex)
    for k in range(1, d):
        Jp[k - 1, k] = np.sqrt(j * (j + 1) - ms[k] * (ms[k] + 1))
    return (Jp + Jp.conj().T) / 2, (Jp - Jp.conj().T) / (2j), Jz


def wigner(j, theta, phi):
    """D(R) for R = R_z(phi) R_y(theta), which carries the north pole to (theta, phi)."""
    _, Jy, Jz = spin_matrices(j)
    return expm(-1j * phi * Jz) @ expm(-1j * theta * Jy)


def basis_vector(j, m):
    v = np.zeros(int(2 * j + 1), complex)
    v[int(j - m)] = 1.0
    return v


def bulk_scalar_levels(s=3, l_max=6):
    """Levels m^2 r^2 = l (l + 1) - s^2 (l >= |s|) of a charged scalar with spin weight s = -F/2 in unit flux."""
    return {l: l * (l + 1) - s * s for l in range(abs(s), l_max + 1)}


def lowest_level_checks():
    """Exact checks on the spin-weight-3 harmonics (F = -6): the l = 3 level is annihilated by eth, has
    -(eth eth-bar + eth-bar eth)/2 eigenvalue 3, its eth-bar has spin weight 2, and at the pole only m = -3
    (value) and m = -2 (eth-bar) survive, so the value has c = 3 and eth-bar has c = 2."""
    import sympy as sp
    try:
        from .sphere_family_structure import swsh, eth, eth_bar, _is_zero, _ratio
        from .yukawa_mechanisms import pole_nonzero_m
    except ImportError:
        from sphere_family_structure import swsh, eth, eth_bar, _is_zero, _ratio
        from yukawa_mechanisms import pole_nonzero_m
    samples = ((0.7, 0.3), (1.9, -1.1), (2.6, 2.2))
    ok_eth, ok_eig, ok_down = True, True, True
    for m in range(-3, 4):
        Y = swsh(sp.Integer(3), sp.Integer(3), sp.Integer(m))
        ok_eth &= _is_zero(eth(Y, 3), samples)
        lap = -(eth(eth_bar(Y, 3), 2) + eth_bar(eth(Y, 3), 4)) / 2
        ok_eig &= _is_zero(lap - 3 * Y, samples)
        # eth-bar of sY_{3m} is a constant (-sqrt 6) times (s-1)Y_{3m}: a spin-weight-2 section of the same l.
        ratios = _ratio(eth_bar(Y, 3), swsh(sp.Integer(2), sp.Integer(3), sp.Integer(m)), samples)
        ok_down &= all(abs(r - (-np.sqrt(6))) < 1e-9 for r in ratios)
    return {"eth_annihilates_lowest_level": bool(ok_eth), "laplacian_eigenvalue_3": bool(ok_eig),
            "eth_bar_gives_spin_weight_2": bool(ok_down),
            "value_pole_m": pole_nonzero_m(3, 3), "eth_bar_pole_m": pole_nonzero_m(2, 3),
            "value_c": brane_jz_charge(0, -6), "eth_bar_c": brane_jz_charge(-1, -6)}


def light_combination(thetas, phis, kappa, lam=None):
    """Brane mass terms kappa_a |Phi(z_a)|^2 + lam_a |eth-bar Phi(z_a)|^2 on the l = 3 multiplet.

    The value of the spin-weight-s component at brane a is <D(R_a) e_{-s}, phi>. Returns the light combination
    (lowest eigenvector: the one the single tuning makes light), the gap to the next level, and its eth-bar weights
    c_a at the branes, which multiply the rank-1 brane Yukawas.
    """
    lam = np.zeros(len(thetas)) if lam is None else lam
    Ds = [wigner(3, t, p) for t, p in zip(thetas, phis)]
    w = [D @ basis_vector(3, -3) for D in Ds]
    v = [D @ basis_vector(3, -2) for D in Ds]
    K = sum(k * np.outer(x, x.conj()) for k, x in zip(kappa, w)) + sum(l * np.outer(x, x.conj()) for l, x in zip(lam, v))
    ev, U = np.linalg.eigh(K)
    phi = U[:, 0]
    c = np.array([np.vdot(x, phi) for x in v])
    return {"phi": phi, "gap": float(ev[1] - ev[0]), "weights": c, "eigenvalues": ev}


def bulk_higgs_scan(trials=1000, seed=0):
    """Random brane positions and brane mass terms of both signs: how evenly the single light combination feeds the
    four branes, and the rank of the resulting Yukawa (and Majorana) matrices."""
    rng = np.random.default_rng(seed)
    weights, gaps, ratios = [], [], []
    for _ in range(trials):
        th = np.arccos(rng.uniform(-1, 1, N_BRANES))
        ph = rng.uniform(0, 2 * np.pi, N_BRANES)
        out = light_combination(th, ph, rng.normal(size=N_BRANES), rng.normal(size=N_BRANES))
        c = out["weights"]
        weights.append(np.sort(abs(c) / abs(c).max()))
        gaps.append(out["gap"])
        z = np.tan(th / 2) * np.exp(1j * ph)
        y = rng.normal(size=N_BRANES) + 1j * rng.normal(size=N_BRANES)
        Y = sum(y[a] * c[a] * brane_matrix(z[a]) for a in range(N_BRANES))
        sv = np.linalg.svd(Y, compute_uv=False)
        ratios.append(sv[-1] / sv[0])
    weights = np.array(weights)
    return {"median_sorted_weights": np.median(weights, axis=0).tolist(),
            "fraction_min_weight_above_0.05": float(np.mean(weights[:, 0] > 0.05)),
            "median_gap": float(np.median(gaps)), "min_gap": float(np.min(gaps)),
            "median_smallest_over_largest_singular_value": float(np.median(ratios)),
            "fraction_rank_deficient_1e-8": float(np.mean(np.array(ratios) < 1e-8))}


def positive_value_terms_degeneracy(seed=0):
    """Only positive brane value terms: they vanish on a (7 - 4)-dimensional subspace, so three combinations stay
    degenerate and light. A unique light combination needs a negative brane term or derivative terms."""
    rng = np.random.default_rng(seed)
    th = np.arccos(rng.uniform(-1, 1, N_BRANES))
    ph = rng.uniform(0, 2 * np.pi, N_BRANES)
    out = light_combination(th, ph, abs(rng.normal(size=N_BRANES)) + 0.1)
    return int(np.sum(abs(out["eigenvalues"]) < 1e-10))


# ---------------------------------------------------------------------------
# 5. Family isometry breaking
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
        "schema_version": 2,
        "model_id": MODEL_ID,
        "status": "version_A_fails_version_B_passes_structural_checks",
        "empirical_validation": False,
        "fields": {k: {"location": v[0], "so10": v[1], "F": v[2], "normal_weight": v[3], "copies": v[4], "role": v[5]}
                   for k, v in FIELDS.items()},
        "wanted_couplings": wanted,
        "forbidden_couplings": dangerous,
        "brane_scalar_useless": brane_scalar_higgs_is_useless(),
        "breaking": {"B-L only (3221)": breaking_pattern(1.0, 0.0, False),
                     "B-L with 126bar (SM)": breaking_pattern(1.0, 0.0, True),
                     "T3R only (421)": breaking_pattern(0.0, 1.0, False),
                     "both (3211)": breaking_pattern(1.0, 0.4, False),
                     "with 126bar (SM)": breaking_pattern(1.0, 0.4, True)},
        "c2_brane_span_dimension": c2_brane_span_dimension(),
        "realizability_rank_mod_U3": {str(N): realizability_rank(N) for N in (2, 3, 4)},
        "bargmann_relation_mismatch": bargmann_relation(),
        "hierarchical_example": _hierarchical_example(),
        "version_A_brane_copies_light_yukawa_rank": brane_copies_light_yukawa_rank(),
        "bulk_scalar_levels": {str(k): v for k, v in bulk_scalar_levels().items()},
        "lowest_level_checks": lowest_level_checks(),
        "bulk_higgs_scan": bulk_higgs_scan(trials=400),
        "positive_value_terms_degenerate_light_states": positive_value_terms_degeneracy(),
        "isometry_stabilizer": {"one brane": isometry_stabilizer_dimension(pts[:1]),
                                "two branes": isometry_stabilizer_dimension(pts[:2]),
                                "four branes": isometry_stabilizer_dimension(pts)},
        "limitations": list(LIMITATIONS),
    }
