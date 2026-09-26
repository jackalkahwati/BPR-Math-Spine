"""Family symmetry and Yukawa structure of flux-induced families on S^2 (BPR-6D).

See doc/derivations/family_symmetry_from_flux_2026-09-26.md. Spin-weighted
spherical harmonics (Goldberg formula) and the eth/eth-bar operators give the
monopole Dirac zero modes explicitly; the SU(2) isometry of the round sphere
acts on them as spin j = (k-1)/2, k = Q m. Six-dimensional gamma matrices give
the chirality selection rule for Yukawa couplings, and Clebsch-Gordan algebra
gives the allowed Yukawa channel and the mass spectra of vev orientations.
Everything is exact or numerically exact linear algebra for supplied fields.
"""

from functools import lru_cache

import numpy as np
import sympy as sp
from sympy.physics.quantum.cg import CG

MODEL_ID = "bpr6d-sphere-family-structure-v1"
theta, phi = sp.symbols("theta phi", real=True)

LIMITATIONS = [
    "The round S^2, the flux and the field content are supplied BPR-6D inputs.",
    "The Higgs sector is not specified; minimal BPR-6D has no 16x16 field, and the J=2 channel assumes an internal one-form 10 or 126 of F-charge -2.",
    "The charged-vector levels are checked against the Atiyah-Bott count and a finite-difference Yang-Mills Hessian, not by an analytic fluctuation calculation.",
    "The minimal SO(12) gauge-Higgs embedding gives 2k families paired by the Yukawa; a projection to three leaves one massless, and its tachyonic level destabilizes the flux vacuum.",
    "The SU(2) isometry is a 4D gauge symmetry that must be broken far above the electroweak scale; no mechanism is supplied.",
    "Selection rules assume perturbative U(1)_F; it is Stueckelberg-massive and broken by Spin(10) instantons to at most a discrete remnant.",
    "No fermion masses are predicted; only which structures are allowed.",
]


# ---------------------------------------------------------------------------
# Spin-weighted spherical harmonics and eth operators
# ---------------------------------------------------------------------------

def _half(value):
    value = sp.nsimplify(value)
    if (2 * value).is_integer is not True:
        raise ValueError("spin labels must be integers or half-integers")
    return value


@lru_cache(maxsize=None)
def swsh(s, l, mm):
    """Goldberg formula for the spin-weighted harmonic sY_l,m (integer or half-integer labels)."""
    s, l, mm = _half(s), _half(l), _half(mm)
    if l < abs(s) or l < abs(mm) or (l - s).is_integer is not True or (l - mm).is_integer is not True:
        raise ValueError("invalid (s, l, m)")
    pref = (-1) ** (mm) * sp.sqrt(sp.factorial(l + mm) * sp.factorial(l - mm) * (2 * l + 1)
                                  / (4 * sp.pi * sp.factorial(l + s) * sp.factorial(l - s)))
    total = 0
    for rr in range(0, int(l - s) + 1):
        k2 = rr + s - mm
        if k2 < 0 or k2 > l + s:
            continue
        total += (sp.binomial(l - s, rr) * sp.binomial(l + s, k2) * (-1) ** (l - rr - s)
                  * sp.cot(theta / 2) ** (2 * rr + s - mm))
    return sp.simplify(pref * sp.sin(theta / 2) ** (2 * l) * total * sp.exp(sp.I * mm * phi))


def eth(f, s):
    """eth on spin weight s: -(sin)^s (d_theta + i/sin d_phi) [(sin)^-s f]."""
    s = _half(s)
    g = sp.sin(theta) ** (-s) * f
    return sp.simplify(-sp.sin(theta) ** s * (sp.diff(g, theta) + sp.I / sp.sin(theta) * sp.diff(g, phi)))


def eth_bar(f, s):
    """eth-bar on spin weight s: -(sin)^-s (d_theta - i/sin d_phi) [(sin)^s f]."""
    s = _half(s)
    g = sp.sin(theta) ** s * f
    return sp.simplify(-sp.sin(theta) ** (-s) * (sp.diff(g, theta) - sp.I / sp.sin(theta) * sp.diff(g, phi)))


def _is_zero(expr, samples=((0.7, 0.3), (1.9, -1.1), (2.6, 2.2))):
    fn = sp.lambdify((theta, phi), expr, "numpy")
    return all(abs(complex(fn(a, b))) < 1e-10 for a, b in samples)


def _ratio(numer, denom, samples=((0.7, 0.3), (1.9, -1.1))):
    fn = sp.lambdify((theta, phi), numer, "numpy")
    fd = sp.lambdify((theta, phi), denom, "numpy")
    values = [complex(fn(a, b)) / complex(fd(a, b)) for a, b in samples]
    return values


def zero_mode_spin_weights(k):
    """Spinor components on S^2 in monopole charge k = Q m: s_plus = (1-k)/2, s_minus = s_plus - 1."""
    k = sp.Integer(k)
    return (1 - k) / 2, (1 - k) / 2 - 1


@lru_cache(maxsize=None)
def zero_modes(k):
    """Explicit Dirac zero modes on S^2 for monopole charge k >= 1: (-j)Y_{j,m}, j=(k-1)/2."""
    if type(k) is not int or k < 1 or k > 7:
        raise ValueError("k must be an int in 1..7")
    s_plus, _ = zero_mode_spin_weights(k)
    j = -s_plus
    return [(s_plus, j, mm, swsh(s_plus, j, mm)) for mm in [j - n for n in range(int(2 * j) + 1)]]


def zero_mode_report(k):
    """Kernel check: eth-bar annihilates each mode; J_z weights are -j..j, each once."""
    modes = zero_modes(k)
    s_plus, s_minus = zero_mode_spin_weights(k)
    annihilated = all(_is_zero(eth_bar(f, s)) for s, _, _, f in modes)
    weights = []
    for s, j, mm, f in modes:
        jz = _ratio(-sp.I * sp.diff(f, phi), f)
        weights.append(float(np.real(jz[0])))
    # No kernel for the other component: eth on s_minus never vanishes for l >= |s_minus|.
    other = []
    l = abs(s_minus)
    while l <= abs(s_minus) + 2:
        f = swsh(s_minus, l, l)
        other.append(not _is_zero(eth(f, s_minus)))
        l += 1
    return {"k": k, "count": len(modes), "spin": float(-s_plus), "annihilated": annihilated,
            "jz_weights": sorted(weights), "other_chirality_has_no_kernel": all(other)}


def kk_mass_squared(l, k):
    """-eth eth-bar eigenvalue on s_plus: (l+1/2)^2 - k^2/4 (units 1/r^2)."""
    return (sp.nsimplify(l) + sp.Rational(1, 2)) ** 2 - sp.Rational(k, 1) ** 2 / 4


def kk_check(k, max_extra=2):
    s_plus, _ = zero_mode_spin_weights(k)
    rows = []
    l = abs(s_plus)
    for _ in range(max_extra + 1):
        f = swsh(s_plus, l, l)
        lhs = -eth(eth_bar(f, s_plus), s_plus - 1)
        ratio = _ratio(lhs, f)[0] if not _is_zero(lhs) else 0.0
        rows.append({"l": float(l), "numeric": float(np.real(ratio)), "formula": float(kk_mass_squared(l, k)),
                     "degeneracy": int(2 * l + 1)})
        l += 1
    return rows


# ---------------------------------------------------------------------------
# Six-dimensional chirality lemma
# ---------------------------------------------------------------------------

def gamma6():
    """8x8 gamma matrices for signature (-,+,+,+,+,+) and the chirality matrix."""
    s0 = np.eye(2)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    def kron(*ms):
        out = np.array([[1.0 + 0j]])
        for mtx in ms:
            out = np.kron(out, mtx)
        return out

    euclid = [kron(sx, s0, s0), kron(sy, s0, s0), kron(sz, sx, s0), kron(sz, sy, s0),
              kron(sz, sz, sx), kron(sz, sz, sy)]
    G = [1j * euclid[0]] + euclid[1:]  # Gamma^0 squares to -1
    chir = G[0] @ G[1] @ G[2] @ G[3] @ G[4] @ G[5]
    # normalize so chir^2 = 1 and chir is Hermitian
    if np.allclose(chir @ chir, -np.eye(8)):
        chir = 1j * chir
    return G, chir


def charge_conjugations(G):
    """Matrices C with C Gamma^M C^-1 = eta (Gamma^M)^T for eta = +1 and -1 (each unique up to scale)."""
    basis = []
    for idx in range(64):
        mtx = np.zeros((8, 8), dtype=complex)
        mtx[idx // 8, idx % 8] = 1
        basis.append(mtx)
    found = {}
    for eta in (1, -1):
        # C Gamma - eta Gamma^T C = 0 is linear in the 64 entries of C.
        A = np.vstack([np.array([(b @ g - eta * g.T @ b).reshape(-1) for b in basis]).T for g in G])
        _, sv, vh = np.linalg.svd(A)
        null = vh[sv < 1e-10]
        if len(null) != 1:
            raise ArithmeticError("charge conjugation not unique")
        found[eta] = null[0].reshape(8, 8)
    return found


def chirality_lemma(samples=5, seed=0):
    """Max |bilinear| for random same-chirality spinors, by bilinear type."""
    G, chir = gamma6()
    Cs = charge_conjugations(G)
    w, vecs = np.linalg.eigh(chir)
    plus = vecs[:, w > 0]
    rng = np.random.default_rng(seed)
    out = {"dirac_scalar": 0.0, "majorana_scalar_C+": 0.0, "majorana_scalar_C-": 0.0,
           "dirac_vector": 0.0, "majorana_vector_C+": 0.0, "majorana_vector_C-": 0.0}
    for _ in range(samples):
        a = plus @ (rng.normal(size=4) + 1j * rng.normal(size=4))
        b = plus @ (rng.normal(size=4) + 1j * rng.normal(size=4))
        bar = a.conj() @ G[0]
        out["dirac_scalar"] = max(out["dirac_scalar"], abs(bar @ b))
        out["dirac_vector"] = max(out["dirac_vector"], max(abs(bar @ g @ b) for g in G))
        for eta, C in Cs.items():
            key = "C+" if eta == 1 else "C-"
            out["majorana_scalar_" + key] = max(out["majorana_scalar_" + key], abs(a @ C @ b))
            out["majorana_vector_" + key] = max(out["majorana_vector_" + key],
                                                max(abs(a @ C @ g @ b) for g in G))
    return {key: float(value) for key, value in out.items()}


# ---------------------------------------------------------------------------
# Selection rules, the J=2 channel and vev orientations
# ---------------------------------------------------------------------------

def yukawa_channels(k=3):
    """Which Higgs components give an integrable, SU(2)-allowed Yukawa among the zero modes (no derivatives).

    The zero modes have spin weight (1-k)/2 each; perturbative F-neutrality fixes the Higgs charge to -2.
    The triple overlap needs total spin weight zero and J_H in j (x) j with j=(k-1)/2.

    Robust form (any field, any number of derivatives): the zero-mode pair has total spin weight 1-k,
    so the Higgs mode needs spin weight k-1, hence J >= k-1 = 2j; and j (x) j caps J at 2j. So J = 2j
    exactly (J = 2 for three families). Derivatives (eth) raise spin weight at fixed J, so a field
    reaches the channel iff it has an l = 2j mode with effective spin weight <= 2j.
    """
    s_psi, _ = zero_mode_spin_weights(k)
    j_psi = -s_psi
    out = []
    for s_h in (-1, 0, 1):
        s_eff = s_h + k  # spin weight of a charge -2 field: s_h - (-2) k / 2
        weight_sum = 2 * s_psi + s_eff
        lowest = abs(s_eff)
        allowed = [J for J in range(int(lowest), int(lowest) + 3) if J <= 2 * j_psi]
        out.append({"higgs_spin_weight": s_h, "effective_spin_weight": float(s_eff),
                    "total_spin_weight": float(weight_sum), "lowest_isospin": float(lowest),
                    "allowed_isospins": allowed if weight_sum == 0 else []})
    return out


def overlap(k, j_h, mm, nodes=48):
    """Triple overlap of two zero modes with a (J_H, m3) mode of spin weight -2 s_psi.

    int sY_{j,m1} sY_{j,m2} s_hY_{J_H,m3} dOmega, total spin weight zero; Gauss-Legendre
    in cos(theta) (endpoints avoided) and an exact uniform rule in phi.
    """
    s_psi, _ = zero_mode_spin_weights(k)
    s_h = -2 * s_psi
    modes = {md[2]: md[3] for md in zero_modes(k)}
    integrand = modes[sp.nsimplify(mm[0])] * modes[sp.nsimplify(mm[1])] * swsh(s_h, j_h, mm[2])
    fn = sp.lambdify((theta, phi), integrand, "numpy")
    xs, ws = np.polynomial.legendre.leggauss(nodes)
    phs = np.linspace(0, 2 * np.pi, 4 * int(k + j_h) + 8, endpoint=False)
    T, P = np.meshgrid(np.arccos(xs), phs, indexing="ij")
    vals = np.broadcast_to(np.asarray(fn(T, P), dtype=complex), T.shape)
    return complex(np.sum(ws[:, None] * vals) * (2 * np.pi / len(phs)))


def yukawa_tensor(j=1, J=2):
    """Y[i, j, k] = <j m_i ; j m_j | J m_k> in the basis m = j, ..., -j."""
    ms = [j - n for n in range(2 * j + 1)]
    Ms = [J - n for n in range(2 * J + 1)]
    Y = np.zeros((len(ms), len(ms), len(Ms)))
    for a, ma in enumerate(ms):
        for b, mb in enumerate(ms):
            for c, mc in enumerate(Ms):
                if ma + mb == mc:
                    Y[a, b, c] = float(CG(j, ma, j, mb, J, mc).doit())
    return Y


def mass_spectrum(vev, j=1, J=2):
    """Singular values (descending) of M_ij = sum_k Y_ijk v_k for a J=2 vev vector (m = 2..-2)."""
    Y = yukawa_tensor(j, J)
    M = np.tensordot(Y, np.asarray(vev, dtype=complex), axes=([2], [0]))
    return sorted(np.linalg.svd(M, compute_uv=False).tolist(), reverse=True)


def real_vev(rng):
    """A random 'real' spin-2 vev: v_{-m} = (-1)^m conj(v_m) (a real Cartesian symmetric tensor)."""
    v = np.zeros(5, dtype=complex)
    v[2] = rng.normal()
    for mm in (1, 2):
        c = rng.normal() + 1j * rng.normal()
        v[2 - mm] = c
        v[2 + mm] = (-1) ** mm * np.conj(c)
    return v


SPHERICAL_BASIS = np.array([[-1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
                            [-1j / np.sqrt(2), 0, -1j / np.sqrt(2)],
                            [0, 1, 0]], dtype=complex)  # columns e_{+1}, e_0, e_{-1}


def cartesian_to_vev(M_cart):
    """J=2 vev whose mass matrix is the (symmetric, traceless) Cartesian matrix M_cart."""
    M_sph = SPHERICAL_BASIS.T @ np.asarray(M_cart, dtype=complex) @ SPHERICAL_BASIS
    Y = yukawa_tensor()
    vev = np.tensordot(Y, M_sph, axes=([0, 1], [0, 1]))
    residual = np.max(np.abs(np.tensordot(Y, vev, axes=([2], [0])) - M_sph))
    if residual > 1e-10:
        raise ValueError("matrix is not symmetric traceless (has a J=0 or J=1 part)")
    return vev


def vev_for_spectrum(sigmas):
    """A complex J=2 vev with prescribed mass spectrum: every ordered triple is reachable.

    M = U diag(a, b, c) U^T with U^T U = W symmetric unitary, W = [[i t, w, 0], [w, i t, 0], [0, 0, -i]],
    t = c / (a + b) <= 1, w = sqrt(1 - t^2): trace = i t (a + b) - i c = 0.
    """
    a, b, c = sorted((float(value) for value in sigmas), reverse=True)
    if c < 0 or a + b == 0:
        raise ValueError("need nonnegative values, not all zero")
    t = c / (a + b)
    w = np.sqrt(1 - t * t)
    O = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    U = np.zeros((3, 3), dtype=complex)
    U[:2, :2] = O @ np.diag(np.sqrt([1j * t + w, 1j * t - w])) @ O.T
    U[2, 2] = np.sqrt(-1j)
    return cartesian_to_vev(U @ np.diag([a, b, c]) @ U.T)


def charged_vector_lowest_level(n):
    """Lowest physical 4D mass^2 (units 1/r^2) of internal gauge components with monopole number n.

    Background-gauge operator: rough Laplacian l(l+1) - s_e^2 on effective spin weight s_e, plus Ricci
    (+1) and the gyromagnetic term (-|n| aligned, +|n| anti-aligned). Aligned: s_e = |n|/2 - 1.
    Pure-gauge directions D(lambda) have l >= |n|/2, so for |n| >= 2 the aligned level l = |n|/2 - 1 is
    physical; for |n| = 1 the aligned l = 1/2 mode is pure gauge and the lowest physical level is l = 3/2.
    """
    if type(n) is not int or n == 0:
        raise ValueError("n must be a nonzero int")
    a = abs(n)
    if a == 1:
        l = sp.Rational(3, 2)
        return {"n": n, "mass_squared": l * (l + 1) - sp.Rational(1, 4), "degeneracy": 4,
                "tachyonic": False, "isospin": l, "note": "l=1/2 aligned mode is pure gauge"}
    s_e = sp.Rational(a, 2) - 1
    return {"n": n, "mass_squared": s_e + 1 - a, "degeneracy": int(2 * s_e + 1),
            "tachyonic": True, "isospin": s_e}


def so12_family_pairing(k, rng=None):
    """SO(12) gauge-Higgs: A_a (in 10_{-2}) maps 16_{+1} to 16bar_{-1}, so the Yukawa pairs the k families
    from 16_{+1} with the k from 16bar_{-1}. The 2k x 2k Weyl mass matrix is [[0, M], [M^T, 0]]:
    each singular value of M appears twice, and any projection keeping n_A + n_B families has rank
    <= 2 min(n_A, n_B)."""
    rng = np.random.default_rng(0) if rng is None else rng
    Mk = rng.normal(size=(k, k)) + 1j * rng.normal(size=(k, k))
    full = np.block([[np.zeros((k, k)), Mk], [Mk.T, np.zeros((k, k))]])
    sv_full = sorted(np.linalg.svd(full, compute_uv=False).tolist(), reverse=True)
    sv_M = sorted(np.linalg.svd(Mk, compute_uv=False).tolist(), reverse=True)
    projections = {}
    for n_a in range(0, 4):
        n_b = 3 - n_a
        if n_a > k or n_b > k:
            continue
        keep = list(range(n_a)) + list(range(k, k + n_b))
        sub = full[np.ix_(keep, keep)]
        projections["{}+{}".format(n_a, n_b)] = int(np.linalg.matrix_rank(sub, tol=1e-9))
    return {"k": k, "singular_values_full": sv_full, "singular_values_M": sv_M,
            "three_family_projection_ranks": projections}


def higgs_representation_channels():
    """16 x 16 = 10 + 120 + 126 (SO(10)-symmetric: 10, 126; antisymmetric: 120).

    The same-chirality vector bilinear psi^T C Gamma^a psi is antisymmetric on the chiral subspace and
    fermions anticommute, so the coupling is symmetric in (SO(10) x family): symmetric reps need
    family-symmetric J (0, 2), the antisymmetric 120 needs J = 1. Spin weight forces J = 2.
    """
    G, chir = gamma6()
    w, vecs = np.linalg.eigh(chir)
    P = vecs[:, w > 0]
    antisym = all(np.allclose(P.T @ C @ g @ P, -(P.T @ C @ g @ P).T)
                  for C in charge_conjugations(G).values() for g in G)
    Y2, Y1 = yukawa_tensor(1, 2), yukawa_tensor(1, 1)
    family_symmetry = {"J2_symmetric": bool(np.allclose(Y2, Y2.transpose(1, 0, 2))),
                       "J1_antisymmetric": bool(np.allclose(Y1, -Y1.transpose(1, 0, 2)))}
    return {"vector_bilinear_antisymmetric_on_chiral_subspace": bool(antisym),
            "family_symmetry_of_channels": family_symmetry,
            "allowed": ["10", "126"], "excluded": ["120"],
            "ten_only_relations": ["M_d = M_e^T", "M_u = M_nu_Dirac"]}


def orientation_examples():
    ferro = mass_spectrum([1, 0, 0, 0, 0])
    uniaxial = mass_spectrum([0, 0, 1, 0, 0])
    rng = np.random.default_rng(5)
    real_rule = []
    for _ in range(200):
        s = mass_spectrum(real_vev(rng))
        real_rule.append(abs(s[0] - s[1] - s[2]) / max(s[0], 1e-300))
    target = [1.0, 7.3e-3, 1.3e-5]  # illustrative up-type-like hierarchy, not a fit
    reached = mass_spectrum(vev_for_spectrum(target))
    return {"ferromagnetic_m2": ferro, "uniaxial_m0": uniaxial,
            "real_vev_sum_rule_max_violation": float(max(real_rule)),
            "hierarchical_target": target, "hierarchical_reached": reached,
            "hierarchical_max_error": float(max(abs(u - v) for u, v in zip(target, reached)))}


def gauge_higgs_family_count(k):
    """SO(12) > SO(10) x U(1): the 32 gives 16 (charge +1) and 16bar (charge -1) in doubled units.

    Each component gives |k| left-handed 16s, so the family count is 2|k|: never 3.
    """
    try:
        from .chiral_parent_completion import zero_modes as flux_zero_modes
    except ImportError:  # loaded as a top-level module by the demo script
        from chiral_parent_completion import zero_modes as flux_zero_modes
    modes = flux_zero_modes([(1, "16", 1), (1, "16bar", -1)], k)
    return sum(md["multiplicity"] for md in modes if md["representation"] in ("16", "16bar"))


def minimal_yukawa_status(k=3):
    """Yukawa status of BPR-6D's flux families.

    (i) Minimal content: no field in 16 x 16 = 10 + 120 + 126 exists, so there is no Yukawa at all.
    (ii) Extensions (all orders in derivatives and flux insertions, perturbatively in U(1)_F): only an
    F-charge -2 internal one-form 10 or 126 in the J=2 channel couples; a scalar 10 or 126 of F-charge
    -2 has J >= 3 modes only and never couples; the 120 is excluded.
    """
    channels = {row["higgs_spin_weight"]: row for row in yukawa_channels(k)}
    return {
        "families_share_one_6d_weyl_field": True,
        "minimal_content_has_16x16_field": False,
        "scalar_higgs_channel": channels[0]["allowed_isospins"],
        "internal_vector_higgs_channel": channels[-1]["allowed_isospins"],
        "representations": higgs_representation_channels(),
        "spin10_gauge_fields_contain_a_10": False,
        "so12_gauge_higgs": {
            "families": gauge_higgs_family_count(k),
            "higgs_level": charged_vector_lowest_level(2 * k),
            "pairing": so12_family_pairing(k),
        },
        "status": "open_no_minimal_yukawa",
    }


def demonstration_report():
    zm = zero_mode_report(3)
    minimal = minimal_yukawa_status(3)
    minimal["so12_gauge_higgs"]["higgs_level"] = {
        key: (float(value) if isinstance(value, sp.Basic) else value)
        for key, value in minimal["so12_gauge_higgs"]["higgs_level"].items()}
    minimal["so12_gauge_higgs"]["pairing"] = {
        key: value for key, value in minimal["so12_gauge_higgs"]["pairing"].items() if key != "singular_values_M"}
    report = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "status": "exact_family_structure_yukawa_open",
        "empirical_validation": False,
        "zero_modes_k3": zm,
        "zero_mode_counts": {str(k): zero_mode_report(k)["count"] for k in (1, 2, 3, 4)},
        "kk_spectrum_k3": kk_check(3),
        "chirality_lemma": chirality_lemma(),
        "yukawa_channels_k3": yukawa_channels(3),
        "orientation_examples": orientation_examples(),
        "minimal_yukawa_status": minimal,
        "limitations": list(LIMITATIONS),
    }
    return report
