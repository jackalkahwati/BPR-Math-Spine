"""Flux vacuum M4 x S^2 of six-dimensional Einstein-Maxwell theory (BPR-6D background).

See doc/derivations/flux_compactification_2026-09-26.md. Everything here is
exact symbolic algebra (SymPy) on the stated action

    S = integral d^6x sqrt(-G) [ (M^4/2) R_6 - Lambda - (1/4) F_MN F^MN ],

with the U(1)_F field strength carrying m units of monopole flux on S^2 and
charge coupling D = partial - i e Q A (Q = 1 for the parent 16). The solution
is the Randjbar-Daemi-Salam-Strathdee (1983) vacuum; this module re-derives
its conditions, the radion (breathing-mode) potential and kinetic term, and the
four-dimensional scale relations. Only the breathing mode's stability is
derived; other fluctuations are cited, not computed.
"""

from functools import lru_cache
from itertools import combinations, permutations
from math import isfinite, pi, sqrt

import sympy as sp
from sympy.combinatorics import Permutation

MODEL_ID = "bpr6d-einstein-maxwell-flux-vacuum-v1"
REDUCED_PLANCK_GEV = 2.435e18  # reduced Planck mass, used only for illustration

t, x, y, z, theta, phi = sp.symbols("t x y z theta phi", real=True)
COORDS = (t, x, y, z, theta, phi)
M, Lam, e, r, r0, B = sp.symbols("M Lambda e r r0 B", positive=True)
m = sp.symbols("m", positive=True, integer=True)
psi = sp.Function("psi")(t)
Omega = sp.symbols("Omega", positive=True)

LIMITATIONS = [
    "Six-dimensional gravity, U(1)_F and the field content are supplied (fundamental), not derived from a substrate.",
    "Flat four-dimensional space requires one tuning of the six-dimensional cosmological constant.",
    "Only the breathing-mode stability is derived; other Einstein-Maxwell modes are cited (RSS 1983); Spin(10), fermion and 2-form fluctuations are not covered.",
    "Quantum corrections (Casimir energy of the sphere, Green-Schwarz axion dynamics) are not included.",
    "Numerical scales use an unknown U(1)_F coupling and are illustrative, not predictions.",
]


def _christoffel(g, ginv):
    n = len(COORDS)
    Gam = [[[0] * n for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                val = 0
                for d in range(n):
                    if ginv[a, d] == 0:
                        continue
                    val += ginv[a, d] * (sp.diff(g[d, b], COORDS[c]) + sp.diff(g[d, c], COORDS[b])
                                         - sp.diff(g[b, c], COORDS[d]))
                Gam[a][b][c] = sp.simplify(val / 2)
    return Gam


def _ricci(g):
    ginv = sp.simplify(g.inv())
    Gam = _christoffel(g, ginv)
    n = len(COORDS)
    Ric = sp.zeros(n, n)
    for b in range(n):
        for c in range(n):
            val = 0
            for a in range(n):
                val += sp.diff(Gam[a][b][c], COORDS[a]) - sp.diff(Gam[a][b][a], COORDS[c])
                for d in range(n):
                    val += Gam[a][a][d] * Gam[d][b][c] - Gam[a][c][d] * Gam[d][b][a]
            Ric[b, c] = sp.simplify(val)
    Rs = sp.simplify(sum(ginv[i, j] * Ric[i, j] for i in range(n) for j in range(n)))
    return ginv, Ric, Rs


def product_metric(radius=r, warp=1):
    """ds^2 = warp^2 (-dt^2 + dx^2 + dy^2 + dz^2) + radius^2 (dtheta^2 + sin^2 theta dphi^2)."""
    return sp.diag(-warp ** 2, warp ** 2, warp ** 2, warp ** 2, radius ** 2,
                   radius ** 2 * sp.sin(theta) ** 2)


def monopole_field(radius=r, field=B):
    """F = field * (area form of S^2): F_theta_phi = field radius^2 sin(theta)."""
    F = sp.zeros(6, 6)
    F[4, 5] = field * radius ** 2 * sp.sin(theta)
    F[5, 4] = -F[4, 5]
    return F


def einstein_residual(g, F):
    """E_MN = M^4 G_MN - (T_MN - Lambda g_MN) for metric g and Maxwell field F (module coordinates)."""
    ginv, Ric, Rs = _ricci(g)
    Fup = ginv * F * ginv
    F2 = sp.simplify(sum(F[i, j] * Fup[i, j] for i in range(6) for j in range(6)))
    T = sp.zeros(6, 6)
    for a in range(6):
        for b in range(6):
            T[a, b] = sp.simplify(sum(F[a, c] * F[b, d] * ginv[c, d] for c in range(6) for d in range(6))
                                  - g[a, b] * F2 / 4)
    return sp.simplify(M ** 4 * (Ric - g * Rs / 2) - (T - Lam * g)), Rs, F2


@lru_cache(maxsize=1)
def einstein_conditions():
    """Einstein equations M^4 G_MN = T_MN - Lambda g_MN on flat M4 x S^2 with flux B."""
    g = product_metric()
    E, Rs, F2 = einstein_residual(g, monopole_field())
    four_d = sp.simplify(E[1, 1] / g[1, 1])
    sphere = sp.simplify(E[4, 4] / g[4, 4])
    offdiag = all(sp.simplify(E[i, j]) == 0 for i in range(6) for j in range(6) if i != j)
    same_4d = all(sp.simplify(E[i, i] / g[i, i] - four_d) == 0 for i in range(4))
    same_sphere = sp.simplify(E[5, 5] / g[5, 5] - sphere) == 0
    return {"ricci_scalar": Rs, "F_squared": F2, "four_d_equation": four_d,
            "sphere_equation": sphere, "off_diagonal_vanish": offdiag,
            "four_d_isotropic": same_4d, "sphere_isotropic": same_sphere}


def flux_field(flux=m, radius=r, coupling=e):
    """Dirac quantization with unit charge: e * integral F = 2 pi m, so B = m / (2 e r^2)."""
    return flux / (2 * coupling * radius ** 2)


@lru_cache(maxsize=1)
def vacuum_solution():
    """Solve the Einstein conditions together with flux quantization."""
    cond = einstein_conditions()
    solB = sp.solve([cond["four_d_equation"], cond["sphere_equation"]], [Lam, B], dict=True)
    solB = [s for s in solB if s[B].is_positive is not False]
    if len(solB) != 1:
        raise ArithmeticError("unexpected Einstein solution branch structure")
    field_sol = sp.simplify(solB[0][B])
    lam_sol = sp.simplify(solB[0][Lam])
    radius = sp.solve(sp.Eq(field_sol, flux_field()), r)
    radius = [sp.simplify(x) for x in radius if x.is_positive is not False]
    if len(radius) != 1:
        raise ArithmeticError("unexpected radius branch structure")
    rad = radius[0]
    return {"B": sp.simplify(field_sol.subs(r, rad)), "Lambda": sp.simplify(lam_sol.subs(r, rad)),
            "radius": rad, "B_of_r": field_sol, "Lambda_of_r": lam_sol}


@lru_cache(maxsize=1)
def reduced_potential():
    """Einstein-frame 4D potential for the breathing mode at fixed flux (from the 6D action).

    Using the 4D warp Omega^2 = r0^2 / r^2 that keeps M_Pl^2 = 4 pi r0^2 M^4 fixed.
    """
    g = product_metric(radius=r, warp=Omega)
    _, _, Rs = _ricci(g)
    F = monopole_field(radius=r, field=flux_field())
    ginv = g.inv()
    Fup = ginv * F * ginv
    F2 = sp.simplify(sum(F[i, j] * Fup[i, j] for i in range(6) for j in range(6)))
    sqrtG = sp.sqrt(-g.det())
    lagr = sp.simplify(sqrtG * (M ** 4 / 2 * Rs - Lam - F2 / 4))
    integrated = sp.integrate(sp.integrate(lagr, (theta, 0, sp.pi)), (phi, 0, 2 * sp.pi))
    V = sp.simplify(-integrated.subs(Omega, r0 / r))
    return V


@lru_cache(maxsize=1)
def radion_kinetic_coefficient():
    """K in L_kin = -(1/2) K (partial psi)^2 for r = r0 e^psi, 4D metric e^{-2 psi} eta.

    Computed from sqrt(-G) (M^4/2) R_6 integrated over S^2, with psi = psi(t);
    terms f(psi) psi'' are integrated by parts.
    """
    g = product_metric(radius=r0 * sp.exp(psi), warp=sp.exp(-psi))
    _, _, Rs = _ricci(g)
    sqrtG = sp.sqrt(-g.det())
    dens = sp.simplify(sp.expand(sqrtG * M ** 4 / 2 * Rs))
    dens = sp.integrate(sp.integrate(dens, (theta, 0, sp.pi)), (phi, 0, 2 * sp.pi))
    dens = sp.expand(sp.simplify(dens))
    p1, p2 = sp.symbols("p1 p2")
    poly = dens.subs(sp.Derivative(psi, (t, 2)), p2).subs(sp.Derivative(psi, t), p1)
    poly = sp.expand(poly)
    coeff_p2 = poly.coeff(p2, 1)
    coeff_p1sq = poly.coeff(p2, 0).coeff(p1, 2)
    static = sp.simplify(poly.coeff(p2, 0).coeff(p1, 0))
    # f(psi) psi'' = d/dt(f psi') - f'(psi) psi'^2
    f = coeff_p2.subs(p1, 0)
    fprime = sp.diff(f.subs(psi, sp.Symbol("s")), sp.Symbol("s")).subs(sp.Symbol("s"), psi)
    total_p1sq = sp.simplify(coeff_p1sq - fprime)
    # For time dependence (partial psi)^2 = -psi'^2, so L = +(K/2) psi'^2.
    K = sp.simplify(2 * total_p1sq)
    return {"K": K, "static_part": static, "M_Pl_squared": sp.simplify(4 * sp.pi * r0 ** 2 * M ** 4)}


@lru_cache(maxsize=1)
def radion_stability():
    sol = vacuum_solution()
    V = reduced_potential()
    s = sp.Symbol("s", real=True)
    Vpsi = sp.simplify(V.subs(r, r0 * sp.exp(s)))
    at = {r0: sol["radius"], Lam: sol["Lambda"]}
    V0 = sp.simplify(Vpsi.subs(s, 0).subs(at))
    V1 = sp.simplify(sp.diff(Vpsi, s).subs(s, 0).subs(at))
    V2 = sp.simplify(sp.diff(Vpsi, s, 2).subs(s, 0).subs(at))
    K = radion_kinetic_coefficient()["K"].subs(psi, 0)
    K = sp.simplify(K.subs(r0, sol["radius"]))
    mass2 = sp.simplify(V2 / K)
    return {"V_at_vacuum": V0, "dV": V1, "d2V": V2, "kinetic_K": K,
            "radion_mass_squared": mass2,
            "radion_mass_squared_times_r0_squared": sp.simplify(mass2 * sol["radius"] ** 2)}


@lru_cache(maxsize=1)
def planck_normalization():
    """Reduce (M^4/2) sqrt(-G) R_6 for a curved 4D factor a(t)^2 eta: Z_g with M_Pl^2 = Z_g M^4.

    Checks R_6 = R_4 + 2/r^2 for the product, then integrates sqrt(-G)/sqrt(-g4) over S^2.
    """
    a = sp.Function("a", positive=True)(t)
    R4 = _ricci(sp.diag(-a ** 2, a ** 2, a ** 2, a ** 2, 1, 1))[2]
    g6 = product_metric(radius=r, warp=a)
    R6 = _ricci(g6)[2]
    if sp.simplify(R6 - R4 - 2 / r ** 2) != 0:
        raise ArithmeticError("product curvature does not split")
    density = sp.simplify(sp.sqrt(-g6.det()) / a ** 4)
    return sp.simplify(sp.integrate(sp.integrate(density, (theta, 0, sp.pi)), (phi, 0, 2 * sp.pi)))


@lru_cache(maxsize=1)
def gauge_normalization():
    """Reduce -(1/4) sqrt(-G) F^2 for a 4D field F_tx = E on top of the monopole, with a 4D warp Omega.

    Matching to -(Z/4) F_mu nu F^mu nu = Z E^2 / 2 gives Z; Omega drops out (4D Weyl invariance).
    """
    E = sp.Symbol("E", real=True)
    g = product_metric(radius=r, warp=Omega)
    ginv = g.inv()
    F = monopole_field()
    F[0, 1], F[1, 0] = E, -E
    Fup = ginv * F * ginv
    F2 = sum(F[i, j] * Fup[i, j] for i in range(6) for j in range(6))
    lagr = sp.expand(sp.simplify(-sp.sqrt(-g.det()) * F2 / 4))
    integrated = sp.integrate(sp.integrate(lagr, (theta, 0, sp.pi)), (phi, 0, 2 * sp.pi))
    return sp.simplify(2 * sp.expand(integrated).coeff(E, 2))


def four_d_relations():
    """M_Pl^2 = Z_g M^4 and g4^2 = e^2 / Z from the reductions above, at the vacuum radius."""
    sol = vacuum_solution()
    rad = sol["radius"]
    MPl2 = sp.simplify((planck_normalization() * M ** 4).subs(r, rad))
    g4sq = sp.simplify((e ** 2 / gauge_normalization()).subs(r, rad))
    inverse_radius_over_MPl = sp.simplify(1 / (rad * sp.sqrt(MPl2)))
    # Classical control needs r M >> 1. The O(1) threshold is convention-dependent:
    # 1/r < M, or r > l6 with the 6D Planck length l6 = G6^(1/4), G6 = 1/(8 pi M^4).
    inverse_radius_over_M = sp.simplify(1 / (rad * M))
    g4 = sp.sqrt(g4sq)
    return {"M_Pl_squared": MPl2, "g4_squared": g4sq,
            "inverse_radius_in_Planck_units": inverse_radius_over_MPl,
            "inverse_radius_in_terms_of_g4": sp.simplify(inverse_radius_over_MPl - 2 * g4 / m),
            "inverse_radius_over_M": inverse_radius_over_M,
            "inverse_radius_over_M_in_terms_of_g4": sp.simplify(
                inverse_radius_over_M - 2 * sp.pi ** sp.Rational(1, 4) * sp.sqrt(g4 / m)),
            "control_bound_on_g4": m / (4 * sp.sqrt(sp.pi)),
            "control_bound_on_g4_planck_length_convention": m / sp.sqrt(2)}


def wedge_two_forms(A, C):
    """Components (A ^ C)_{ijkl}, i<j<k<l, of two 2-forms given as antisymmetric 6x6 matrices (up to a constant)."""
    out = {}
    for quad in combinations(range(6), 4):
        val = 0
        for perm in permutations(range(4)):
            i, j, k, l = (quad[q] for q in perm)
            val += Permutation(list(perm)).signature() * A[i, j] * C[k, l]
        out[quad] = sp.simplify(val)
    return out


def wedge_one_two(A, C):
    """Components (A ^ C)_{ijk}, i<j<k, of a 1-form (length-6 list) and a 2-form (up to a constant)."""
    out = {}
    for trip in combinations(range(6), 3):
        val = 0
        for perm in permutations(range(3)):
            i, j, k = (trip[q] for q in perm)
            val += Permutation(list(perm)).signature() * A[i] * C[j, k]
        out[trip] = sp.simplify(val)
    return out


def curvature_two_forms(g):
    """Riemann curvature 2-forms (R^a_b)_{cd} = R^a_{bcd} of a metric in the module coordinates."""
    ginv = sp.simplify(g.inv())
    Gam = _christoffel(g, ginv)
    n = len(COORDS)
    forms = {}
    for a in range(n):
        for b in range(n):
            R = sp.zeros(n, n)
            for c in range(n):
                for d in range(n):
                    val = sp.diff(Gam[a][d][b], COORDS[c]) - sp.diff(Gam[a][c][b], COORDS[d])
                    for f in range(n):
                        val += Gam[a][c][f] * Gam[f][d][b] - Gam[a][d][f] * Gam[f][c][b]
                    R[c, d] = sp.simplify(val)
            forms[(a, b)] = R
    return forms


def trace_r_wedge_r(g):
    """Components of tr(R ^ R) = sum_ab R^a_b ^ R^b_a (proportional to the p1 density)."""
    forms = curvature_two_forms(g)
    total = {quad: 0 for quad in combinations(range(6), 4)}
    for a in range(6):
        for b in range(6):
            for quad, val in wedge_two_forms(forms[(a, b)], forms[(b, a)]).items():
                total[quad] += val
    return {quad: sp.simplify(val) for quad, val in total.items()}


@lru_cache(maxsize=1)
def green_schwarz_background():
    """Green-Schwarz sources evaluated on the background (1 = some component nonzero, 0 = all vanish).

    X is proportional to the U(1)_F field strength, so X^X ~ F^F; S2 = 0 with no Spin(10)
    background; p1 ~ tr R^R; the U(1)_F Chern-Simons 3-form ~ A^F with the monopole potential.
    """
    F = monopole_field()
    A = [0, 0, 0, 0, 0, B * r ** 2 * (1 - sp.cos(theta))]  # dA = F on the northern patch
    if sp.simplify(sp.diff(A[5], theta) - F[4, 5]) != 0:
        raise ArithmeticError("potential does not reproduce the monopole")
    FF = wedge_two_forms(F, F)
    trRR = trace_r_wedge_r(product_metric())
    AF = wedge_one_two(A, F)
    nonzero = lambda comps: int(any(value != 0 for value in comps.values()))  # noqa: E731
    return {"X_wedge_X": nonzero(FF), "S2": 0, "p1": nonzero(trRR),
            "chern_simons_A_wedge_F": nonzero(AF),
            "B_field_source": int(nonzero(FF) or nonzero(trRR))}


def flux_landscape(reference_flux=3, fluxes=(1, 2, 3, 4, 5, 6)):
    """Vacua of the other flux sectors when Lambda > 0 is tuned flat for reference_flux (units M = e = 1).

    Stationary points solve 2 Lambda u^2 - 4 M^4 u + (3/4) m^2/e^2 = 0 with u = r^2; the smaller root
    is the minimum. Minima exist only for m^2 < (8/3) M^8 e^2 / Lambda = (4/3) reference_flux^2
    (equality is an inflection point, unreachable for integer flux).
    """
    if type(reference_flux) is not int or reference_flux < 1:
        raise ValueError("reference_flux must be a positive int")
    lam = sp.Rational(2, reference_flux ** 2)
    V = reduced_potential().subs({M: 1, e: 1, Lam: lam, r0: 1})
    rows = []
    for flux in fluxes:
        disc = 16 - 6 * lam * flux ** 2
        if disc <= 0:
            rows.append({"flux": flux, "vacuum": "none", "radius": None})
            continue
        u = (4 - sp.sqrt(disc)) / (4 * lam)
        rad = sp.sqrt(u)
        energy = sp.nsimplify(sp.simplify(V.subs({m: flux, r: rad})))
        kind = "Minkowski" if energy == 0 else ("de Sitter" if energy > 0 else "anti-de Sitter")
        rows.append({"flux": flux, "vacuum": kind, "radius": float(rad)})
    return {"reference_flux": reference_flux, "Lambda": str(lam),
            "max_flux_squared": str(sp.Rational(4, 3) * reference_flux ** 2), "sectors": rows}


def illustrative_scales(flux=3, couplings=(0.1, 0.5, 1.0)):
    """Numbers only: 1/r = 2 g4 M_Pl / m and the 6D scale M for a few assumed U(1)_F couplings."""
    out = []
    ratio = float(radion_stability()["radion_mass_squared_times_r0_squared"])
    for g4 in couplings:
        inv_r = 2 * g4 * REDUCED_PLANCK_GEV / flux
        radius_gev = 1 / inv_r
        M6 = (REDUCED_PLANCK_GEV ** 2 / (4 * pi * radius_gev ** 2)) ** 0.25
        out.append({"g4": g4, "inverse_radius_GeV": inv_r, "M6_GeV": M6,
                    "radion_mass_GeV": sqrt(ratio) * inv_r, "inverse_radius_over_M6": inv_r / M6})
    for row in out:
        for value in row.values():
            if not isfinite(value):
                raise ValueError("numerical failure")
    return out


def demonstration_report():
    cond = einstein_conditions()
    sol = vacuum_solution()
    stab = radion_stability()
    rel = four_d_relations()
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "status": "exact_background_breathing_mode_stable",
        "empirical_validation": False,
        "einstein_conditions": {"four_d": str(cond["four_d_equation"]),
                                "sphere": str(cond["sphere_equation"]),
                                "off_diagonal_vanish": cond["off_diagonal_vanish"]},
        "vacuum": {key: str(value) for key, value in sol.items()},
        "radion": {key: str(value) for key, value in stab.items()},
        "four_d_relations": {key: str(value) for key, value in rel.items()},
        "green_schwarz_background": green_schwarz_background(),
        "illustrative_scales_flux_3": illustrative_scales(),
        "flux_landscape": flux_landscape(),
        "limitations": list(LIMITATIONS),
    }
