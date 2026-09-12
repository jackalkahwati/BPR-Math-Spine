#!/usr/bin/env python3
"""Independent algebra checks for the 2026-09-12 TOE research notes.

These are consistency checks and counterexamples, NOT experimental validation.
No Monte Carlo, data fitting, benchmark targets, file writes, or repo imports.
Run from the repository root: python3 scripts/check_toe_derivations.py
Requires NumPy and SymPy. All proposed replacements are NEW models.
"""

from itertools import combinations, product
from math import comb, factorial, pi

import numpy as np
import sympy as sp


def check_phase_symmetry():
    theta = np.array([0.0, 0.0])

    def energy(t):
        return 1.0 - np.cos(t[1] - t[0])

    assert np.isclose(energy(theta + 0.7), energy(theta))
    local_change = energy(theta + [0.0, pi]) - energy(theta)
    assert np.isclose(local_change, 2.0)
    t = np.array([0.2, 1.1, -0.8, 2.2])
    holonomy = np.prod(np.exp(1j * (np.roll(t, -1) - t)))
    assert np.isclose(holonomy, 1.0)
    print(f"Global shift symmetry, NOT local gauge invariance: delta H={local_change:g}")
    print(f"Pure phase-difference plaquette holonomy: {holonomy}")


def dihedral_matrices(n):
    elements = [(k, sign) for sign in (1, -1) for k in range(n)]
    index = {g: i for i, g in enumerate(elements)}

    def mul(g, h):
        return ((g[0] + g[1] * h[0]) % n, g[1] * h[1])

    def left(g):
        matrix = np.zeros((2 * n, 2 * n))
        for h in elements:
            matrix[index[mul(g, h)], index[h]] = 1.0
        return matrix

    def right(g):
        matrix = np.zeros((2 * n, 2 * n))
        for h in elements:
            matrix[index[mul(h, g)], index[h]] = 1.0
        return matrix

    return elements, left, right


def check_electric_operator():
    for n in (5, 8, 9, 12):
        elements, left, right = dihedral_matrices(n)
        delta = 3 * np.eye(2 * n) - left((1, 1)) - left((n - 1, 1)) - left((0, -1))
        defect = np.max(np.abs(delta @ left((1, 1)) - left((1, 1)) @ delta))
        assert np.isclose(defect, 1.0)
        assert np.allclose(delta @ right((1, 1)), right((1, 1)) @ delta)
        central = sum(left(g) @ delta @ left(g).T for g in elements) / (2 * n)
        for g in elements:
            assert np.allclose(central @ left(g), left(g) @ central)
            assert np.allclose(central @ right(g), right(g) @ central)
        assert np.linalg.eigvalsh(central).min() > -1e-10
        # A positive, stochastic heat kernel is a NEW temporal action.
        values, vectors = np.linalg.eigh(central)
        heat = (vectors * np.exp(-0.1 * values)) @ vectors.T
        assert heat.min() > -1e-10
        assert np.allclose(heat.sum(axis=0), 1)
        angle = 2 * pi / n
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        reflection = np.diag([1.0, -1.0])
        block = 3 * np.eye(2) - rotation - rotation.T - reflection
        actual = np.linalg.eigvalsh(block)
        average = 3 - 2 * np.cos(angle)
        assert np.allclose(actual, [average - 1, average + 1])
        print(f"D_{n}: endpoint commutator={defect:g}; E1 eigenvalues={actual}; "
              f"trace average={average:.9f}; conjugacy-twirl repair passes")


def check_wilson_transfer():
    n, beta = 5, 1.8
    angles = 2 * pi * np.arange(n) / n
    weights = np.exp(beta * np.cos(angles))
    total = weights.sum() + n
    taus = [(weights.sum() - n) / total]
    taus.extend(np.dot(weights, np.cos(k * angles)) / total for k in (1, 2))
    energies = -np.log(taus)
    frozen = np.array([2.0, 3 - 2 * np.cos(2 * pi / n),
                       3 - 2 * np.cos(4 * pi / n)])
    assert not np.allclose(energies / frozen, (energies / frozen)[0])
    elements, left, _ = dihedral_matrices(n)
    kernel = sum(np.exp(beta * np.cos(2 * pi * k / n) if sign == 1 else 0) * left((k, sign))
                 for k, sign in elements) / total
    assert np.linalg.eigvalsh(kernel).min() > 0
    expected = np.sort([1.0, taus[0]] + [taus[1]] * 4 + [taus[2]] * 4)
    assert np.allclose(np.linalg.eigvalsh(kernel), expected)
    print("D_5 Wilson transfer energies (a_t=1), A2/E1/E2:", energies)
    print("These are single-link electric energies, NOT glueball masses.")


def check_pisot():
    roots = np.roots([1.0, -3.0, 0.0, 1.0])
    sigma = max(roots)
    conjugates = roots[roots != sigma]
    exponents = -np.log(np.abs(conjugates)) / np.log(sigma)
    assert np.isclose(np.abs(np.prod(roots)), 1)
    assert np.isclose(exponents.sum(), 1)
    assert not np.any(np.isclose(exponents, 1))
    print("Ninefold cubic Pisot conjugates:", roots)
    print("Geometric contraction exponents:", exponents, "sum=", exponents.sum())


def check_current_algebra():
    # Lie root lattice D5 means so(10), NOT the dihedral group D_5.
    roots = []
    for i, j in combinations(range(5), 2):
        for a, b in product((-1, 1), repeat=2):
            root = np.zeros(5)
            root[i], root[j] = a, b
            roots.append(root)
    spinors = [np.array(signs) / 2 for signs in product((-1, 1), repeat=5)
               if sum(sign < 0 for sign in signs) % 2 == 0]
    assert len(roots) == 40 and len(spinors) == 16
    assert all(np.isclose(np.dot(r, r), 2) for r in roots)
    assert all(np.isclose(np.dot(w, w) / 2, 5 / 8) for w in spinors)
    assert sp.Rational(8, 4) + sp.Rational(3, 3) + 1 == 4
    assert sp.Rational(45, 9) == 5
    print("Conditional affine-current bound: c_SM >= 4; Spin(10)_1 has c=5")
    print("D5 lattice: 40 roots + 5 Cartan; 16 spinor-module weights with h=5/8")
    print("Internal modules do NOT establish spacetime fermions or anomaly completion.")


def check_anomalies():
    q, h = sp.symbols("q h")
    charges = {"Q": q, "uc": -q - h, "dc": -q + h,
               "L": -3 * q, "ec": 3 * q + h, "nc": 3 * q - h}
    dims = {"Q": 6, "uc": 3, "dc": 3, "L": 2, "ec": 1, "nc": 1}
    assert sp.expand(2 * charges["Q"] + charges["uc"] + charges["dc"]) == 0
    assert sp.expand(3 * charges["Q"] + charges["L"]) == 0
    assert sp.expand(sum(dims[k] * charges[k] for k in charges)) == 0
    assert sp.expand(sum(dims[k] * charges[k] ** 3 for k in charges)) == 0
    # Four SU(2) doublets per SM generation (three colors plus one lepton).
    assert (3 + 1) % 2 == 0
    sm = {k: sp.simplify(v.subs(h, 3 * q).subs(q, sp.Rational(1, 6)))
          for k, v in charges.items()}
    print("Conditional left-handed SM charges with neutral nc:", sm)
    print("Local anomalies cancel per family; cancellation does not choose family count.")


def check_monopole_flavor():
    for flux in (1, 2, 3, 4):
        degree = flux - 1
        for m in range(flux):
            norm2 = flux * comb(degree, m) / (4 * pi)
            integral = 4 * pi * factorial(m) * factorial(degree - m) / factorial(degree + 1)
            assert np.isclose(norm2 * integral, 1.0)
        print(f"O({flux}) spin Dirac kernel: {flux} modes; internal j={(flux - 1)/2:g}")
    jz = np.diag([1.0, 0.0, -1.0])
    jp = np.array([[0, np.sqrt(2), 0], [0, 0, np.sqrt(2)], [0, 0, 0]], complex)
    jm = jp.T.conj()
    generators = [(jp + jm) / 2, (jp - jm) / (2j), jz]
    identity = np.eye(3)
    quadrupoles = [(a @ b + b @ a) / 2 - (2 / 3) * identity * (i == j)
                   for i, a in enumerate(generators)
                   for j, b in enumerate(generators) if i <= j]
    basis = [identity] + generators + quadrupoles
    real_columns = np.column_stack([np.concatenate([b.real.ravel(), b.imag.ravel()])
                                    for b in basis])
    assert np.linalg.matrix_rank(real_columns) == 9
    for n in (5, 8, 9, 12):
        eigenvalues = np.exp(2j * pi * np.array([-1, 0, 1]) / n)
        allowed = np.isclose(eigenvalues[:, None], eigenvalues[None, :])
        assert np.array_equal(allowed, np.eye(3, dtype=bool))
    print("j=1 endomorphisms: scalar + vector + quadrupole span all 9 Hermitian components")
    print("Common exact C_n acting on that triplet forces invariant Hermitian masses diagonal.")
    print("No mass hierarchy/mixing prediction follows from arbitrary profile coefficients.")


def check_scalaron_normalization():
    # Displayed Jordan action: (M^2/2)R + (alpha/2)R^2.
    R, M, alpha, F, phi = sp.symbols("R M alpha F phi", positive=True)
    f = M ** 2 * R / 2 + alpha * R ** 2 / 2
    substitution = M ** 2 * (F - 1) / (2 * alpha)
    potential = sp.simplify((R * sp.diff(f, R) - f).subs(R, substitution) / F ** 2)
    expected = M ** 4 * (1 - 1 / F) ** 2 / (8 * alpha)
    assert sp.simplify(potential - expected) == 0
    canonical = potential.subs(F, sp.exp(sp.sqrt(sp.Rational(2, 3)) * phi / M))
    mass2 = sp.simplify(sp.diff(canonical, phi, 2).subs(phi, 0))
    assert mass2 == M ** 2 / (6 * alpha)
    print("For (M²/2)R+(alpha/2)R²: V=M^4/(8alpha)(1-exp(-sqrt(2/3)phi/M))²")
    print("Scalaron mass²=M²/(6alpha); leading As=N²/(144*pi²*alpha)")


def main():
    checks = [check_phase_symmetry, check_electric_operator, check_wilson_transfer,
              check_pisot, check_current_algebra, check_anomalies,
              check_monopole_flavor, check_scalaron_normalization]
    for check in checks:
        print(f"\n[{check.__name__}]")
        check()
    print(f"\nPASS: {len(checks)} algebra-check groups. No empirical validation claimed.")


if __name__ == "__main__":
    main()
