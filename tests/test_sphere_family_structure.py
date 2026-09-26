"""Checks for doc/derivations/family_symmetry_from_flux_2026-09-26.md.

Independent oracles: SymPy's ordinary spherical harmonics, the standard eth
ladder coefficients, numerical orthonormality, the analytic kernel of eth-bar
(holomorphic sections), the Clifford algebra, the Wigner 3j triple-integral
formula, eigenvalues of real traceless matrices, the Atiyah-Bott Morse index,
and a finite-difference Yang-Mills Hessian without gauge fixing.
"""

import json
import math

import numpy as np
import pytest
import sympy as sp
from sympy.physics.wigner import wigner_3j

import bpr.sphere_family_structure as f

TH, PH = f.theta, f.phi
POINT = (0.9, 0.4)


def value(expr, point=POINT):
    return complex(sp.lambdify((TH, PH), expr, "numpy")(*point))


def test_spin_zero_harmonics_match_sympy():
    for l, mm in [(0, 0), (1, -1), (2, 1), (3, 2)]:
        ours = value(f.swsh(0, l, mm))
        ref = complex(sp.Ynm(l, mm, TH, PH).expand(func=True).subs({TH: POINT[0], PH: POINT[1]}).evalf())
        assert ours == pytest.approx(ref, abs=1e-12)


@pytest.mark.parametrize("s,l,mm", [(0, 2, 1), (-1, 2, -1), (1, 3, 2), ("-1/2", "3/2", "1/2")])
def test_eth_ladder_coefficients(s, l, mm):
    s, l, mm = (sp.nsimplify(q) for q in (s, l, mm))
    Y = f.swsh(s, l, mm)
    up = value(f.eth(Y, s)) / value(f.swsh(s + 1, l, mm))
    down = value(f.eth_bar(Y, s)) / value(f.swsh(s - 1, l, mm))
    assert up == pytest.approx(float(sp.sqrt((l - s) * (l + s + 1))), abs=1e-10)
    assert down == pytest.approx(-float(sp.sqrt((l + s) * (l - s + 1))), abs=1e-10)


def test_harmonics_are_orthonormal():
    xs, ws = np.polynomial.legendre.leggauss(40)
    phs = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    T, P = np.meshgrid(np.arccos(xs), phs, indexing="ij")
    labels = [(-1, 1, 1), (-1, 1, 0), (-1, 2, 0), (-1, 2, 1)]
    vals = [np.broadcast_to(sp.lambdify((TH, PH), f.swsh(*lab), "numpy")(T, P), T.shape) for lab in labels]
    gram = np.array([[np.sum(ws[:, None] * np.conj(a) * b) * 2 * np.pi / len(phs) for b in vals] for a in vals])
    assert np.allclose(gram, np.eye(len(labels)), atol=1e-10)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_zero_modes_form_one_su2_multiplet(k):
    rep = f.zero_mode_report(k)
    j = (k - 1) / 2
    assert rep["count"] == k  # index theorem
    assert rep["annihilated"] and rep["other_chirality_has_no_kernel"]
    assert rep["spin"] == j
    assert rep["jz_weights"] == pytest.approx([j - n for n in range(k)][::-1])


@pytest.mark.parametrize("k", [3, 5])
def test_kernel_is_exactly_the_bounded_holomorphic_sections(k):
    # Independent of the harmonics: eth-bar f = 0 on spin weight -j means f = sin^j(theta) h(w),
    # w = exp(-i phi) tan(theta/2), h holomorphic on C*; boundedness at both poles leaves w^d, |d| <= j.
    j = (k - 1) // 2
    w = sp.exp(-sp.I * PH) * sp.tan(TH / 2)
    modes = [md[3] for md in f.zero_modes(k)]
    pts = [(a, b) for a in (0.4, 1.1, 1.9, 2.7) for b in (0.2, 1.3, 2.9)]
    basis = np.array([[value(md, pt) for md in modes] for pt in pts])
    for d in range(-j - 1, j + 2):
        cand = sp.sin(TH) ** j * w ** d
        assert f._is_zero(f.eth_bar(cand, -j))
        near_poles = max(abs(value(cand, (1e-5, 0.3))), abs(value(cand, (np.pi - 1e-5, 0.3))))
        if abs(d) <= j:
            target = np.array([value(cand, pt) for pt in pts])
            coef, *_ = np.linalg.lstsq(basis, target, rcond=None)
            assert np.linalg.norm(basis @ coef - target) < 1e-10 * np.linalg.norm(target)
            assert near_poles < 10
        else:
            assert near_poles > 1e4  # unbounded: not a section
    assert len(modes) == 2 * j + 1


def test_kk_spectrum_matches_monopole_dirac_formula():
    rows = f.kk_check(3, max_extra=3)
    for row in rows:
        l = row["l"]
        # Monopole-harmonic Dirac spectrum: lambda^2 r^2 = (l + 1/2)^2 - (k/2)^2, degeneracy 2l + 1.
        assert row["numeric"] == pytest.approx((l + 0.5) ** 2 - 9 / 4, abs=1e-9)
        assert row["numeric"] == pytest.approx(row["formula"], abs=1e-9)
    assert rows[0]["numeric"] == pytest.approx(0, abs=1e-12) and rows[1]["numeric"] == pytest.approx(4)


def test_gamma_matrices_and_charge_conjugation():
    G, chir = f.gamma6()
    eta = np.diag([-1, 1, 1, 1, 1, 1])
    for a in range(6):
        for b in range(6):
            assert np.allclose(G[a] @ G[b] + G[b] @ G[a], 2 * eta[a, b] * np.eye(8))
        assert np.allclose(chir @ G[a] + G[a] @ chir, 0)
    assert np.allclose(chir @ chir, np.eye(8)) and np.allclose(chir, chir.conj().T)
    for sign, C in f.charge_conjugations(G).items():
        for g in G:
            assert np.allclose(C @ g @ np.linalg.inv(C), sign * g.T)


def test_six_dimensional_chirality_lemma():
    lemma = f.chirality_lemma()
    for key in ("dirac_scalar", "majorana_scalar_C+", "majorana_scalar_C-"):
        assert lemma[key] < 1e-12
    for key in ("dirac_vector", "majorana_vector_C+", "majorana_vector_C-"):
        assert lemma[key] > 1e-3
    # Control: the scalar Majorana bilinear is nondegenerate between opposite chiralities.
    G, chir = f.gamma6()
    C = f.charge_conjugations(G)[1]
    w, vecs = np.linalg.eigh(chir)
    plus, minus = vecs[:, w > 0], vecs[:, w < 0]
    assert np.linalg.matrix_rank(plus.T @ C @ minus, tol=1e-10) == 4
    assert np.linalg.matrix_rank(plus.T @ C @ plus, tol=1e-10) == 0


def test_only_the_internal_vector_j2_channel_survives():
    rows = {row["higgs_spin_weight"]: row for row in f.yukawa_channels(3)}
    assert rows[0]["allowed_isospins"] == [] and rows[1]["allowed_isospins"] == []
    assert rows[-1]["total_spin_weight"] == 0 and rows[-1]["allowed_isospins"] == [2]
    # Integrals, not bookkeeping: spin-weight-2 Higgs modes with J=3 or J=4 never couple, J=2 does.
    for J in (3, 4):
        for m1 in (1, 0, -1):
            for m2 in (1, 0, -1):
                assert abs(f.overlap(3, J, (m1, m2, -(m1 + m2)))) < 1e-10
    assert abs(f.overlap(3, 2, (1, -1, 0))) > 0.1


def test_higgs_representations_10_and_126_but_not_120():
    rep = f.higgs_representation_channels()
    assert rep["vector_bilinear_antisymmetric_on_chiral_subspace"]
    assert rep["family_symmetry_of_channels"] == {"J2_symmetric": True, "J1_antisymmetric": True}
    assert rep["allowed"] == ["10", "126"] and rep["excluded"] == ["120"]


def test_triple_overlap_is_the_wigner_3j_formula():
    # int sY sY sY = sqrt(prod(2l+1)/4pi) 3j(l; m) 3j(l; -s), spins (-1, -1, 2).
    const = math.sqrt(3 * 3 * 5 / (4 * math.pi)) * float(wigner_3j(1, 1, 2, 1, 1, -2))
    for m1 in (1, 0, -1):
        for m2 in (1, 0, -1):
            m3 = -(m1 + m2)
            expected = const * float(wigner_3j(1, 1, 2, m1, m2, m3))
            assert f.overlap(3, 2, (m1, m2, m3)) == pytest.approx(expected, abs=1e-10)
    assert f.overlap(3, 2, (1, 1, 0)) == pytest.approx(0, abs=1e-12)  # J_z selection


def test_vev_orientations():
    ex = f.orientation_examples()
    assert ex["ferromagnetic_m2"] == pytest.approx([1, 0, 0], abs=1e-12)
    uni = ex["uniaxial_m0"]
    assert uni[0] == pytest.approx(2 * uni[1]) and uni[1] == pytest.approx(uni[2])
    assert ex["real_vev_sum_rule_max_violation"] < 1e-12
    assert ex["hierarchical_max_error"] < 1e-12


def test_real_traceless_matrix_gives_absolute_eigenvalues():
    rng = np.random.default_rng(11)
    for _ in range(20):
        A = rng.normal(size=(3, 3))
        T = A + A.T
        T -= np.trace(T) / 3 * np.eye(3)
        spectrum = f.mass_spectrum(f.cartesian_to_vev(T))
        assert spectrum == pytest.approx(sorted(np.abs(np.linalg.eigvalsh(T)), reverse=True), abs=1e-12)
    with pytest.raises(ValueError):
        f.cartesian_to_vev(np.eye(3))  # pure J=0: not a J=2 vev


def test_every_spectrum_is_reachable_with_a_complex_vev():
    rng = np.random.default_rng(3)
    for _ in range(50):
        target = sorted(rng.uniform(0, 1, size=3) ** 4, reverse=True)
        assert f.mass_spectrum(f.vev_for_spectrum(target)) == pytest.approx(target, abs=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 6, 10])
def test_charged_vector_level_matches_atiyah_bott_index(n):
    level = f.charged_vector_lowest_level(n)
    # Atiyah-Bott (1983): the U(2) (SO(3) for odd n) Yang-Mills critical point with W-degree n on S^2
    # has Morse index 2(n-1): n-1 complex negative directions.
    negative = level["degeneracy"] if level["tachyonic"] else 0
    assert negative == n - 1
    assert f.charged_vector_lowest_level(-n)["mass_squared"] == level["mass_squared"]


def _ym_hessian_levels(n, m, N=300):
    """Second variation of the SU(2) Yang-Mills energy on the unit S^2 around a W-degree-n monopole,
    no gauge fixing (gauge directions are exact zeros), staggered finite differences, phi-mode m."""
    h = np.pi / N
    tn = h * np.arange(1, N)
    tc = h * (np.arange(N) + 0.5)
    nu, nv = N, N - 1
    dim = nu + nv
    Fm = np.zeros((N, dim), complex)
    c = m - 0.5 * n * np.cos(tc)
    for i in range(N):
        Fm[i, i] = -1j * c[i]
        if i <= N - 2:
            Fm[i, nu + i] += np.sin(tn[i]) / h
        if i >= 1:
            Fm[i, nu + i - 1] -= np.sin(tn[i - 1]) / h
    Q = Fm.conj().T @ np.diag(h / np.sin(tc)) @ Fm
    U = np.zeros((nv, dim))
    V = np.zeros((nv, dim))
    for j in range(nv):
        U[j, j] = U[j, j + 1] = 0.5
        V[j, nu + j] = 1
    wt = np.diag(h * np.sin(tn))
    Q = Q - 1j * (n / 2) * (U.T @ wt @ V - V.T @ wt @ U)
    norm = np.diag(np.concatenate([h * np.sin(tc), h * np.sin(tn)]))
    import scipy.linalg as sl
    return sl.eigh((Q + Q.conj().T) / 2, norm, eigvals_only=True)


@pytest.mark.parametrize("n", [1, 2, 3, 6])
def test_charged_vector_level_matches_yang_mills_hessian(n):
    level = f.charged_vector_lowest_level(n)
    negatives, lowest_positive = [], []
    for m in np.arange(-(n / 2 + 1), n / 2 + 1 + 0.01, 1.0):
        ev = _ym_hessian_levels(n, m)
        negatives += [e for e in ev if e < -0.05]
        lowest_positive.append(min(e for e in ev if e > 0.05))
    if level["tachyonic"]:
        assert len(negatives) == level["degeneracy"]
        assert np.allclose(negatives, float(level["mass_squared"]), atol=2e-2)
    else:
        assert negatives == []
        assert min(lowest_positive) == pytest.approx(float(level["mass_squared"]), abs=2e-2)


def test_so12_gauge_higgs_family_count_and_pairing():
    assert [f.gauge_higgs_family_count(k) for k in (1, 2, 3, 4)] == [2, 4, 6, 8]
    pair = f.so12_family_pairing(3, np.random.default_rng(7))
    doubled = sorted(pair["singular_values_M"] * 2, reverse=True)
    assert pair["singular_values_full"] == pytest.approx(doubled, rel=1e-10)
    # Any three-family projection leaves at least one family massless at tree level.
    assert max(pair["three_family_projection_ranks"].values()) == 2
    level = f.minimal_yukawa_status(3)["so12_gauge_higgs"]["higgs_level"]
    assert level["degeneracy"] == 5 and level["isospin"] == 2 and level["tachyonic"]


def test_report_is_strict_json():
    report = f.demonstration_report()
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report["empirical_validation"] is False
    assert report["limitations"] == f.LIMITATIONS
