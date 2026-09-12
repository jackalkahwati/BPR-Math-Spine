"""Exact central gauge model, distinct from historical character-Wilson MC.

This is a new, dimensionless Hamiltonian specification, not a reinterpretation
of the noncentral frozen generator. On a single link L_g|h> = |gh> and
R_g|h> = |hg^-1>. The electric generator uses the conjugacy class containing
s=(0,-1), which for even n contains only half the reflections.

An isolated square has four distinct links and all four vertex Gauss laws.
Gauge fixing three links leaves holonomy modulo conjugation; its physical
Hilbert space is spanned by orthonormal characters with counting measure /|G|.
Only matrices of size <=24 (one link) or <=9 (physical square) are constructed.
The reported energies are single-link electric energies or isolated-square
toy energies, not glueball masses, spatial parity channels, or M4 results.
No physical matching, simulation, or measured targets enter this module.
"""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from .nonabelian_gauge_sector import (
    ALLOWED_CLASSES, character_table, conjugacy_classes, elements, inv, mul,
)

MODEL_ID = "central-class-heat-kernel-v1"
WILSON_MODEL_ID = "character-wilson-single-link-v1"


def _group(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, Integral) or n not in ALLOWED_CLASSES:
        raise ValueError(f"n must be one of {ALLOWED_CLASSES}")
    return int(n)


def _real(value: float, name: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def model_spec(n: int, lam: float) -> dict:
    """Return the assumption/provenance ledger; E0 is not determined here."""
    n, lam = _group(n), _real(lam, "lam")
    return {
        "model_id": MODEL_ID,
        "n": n,
        "group_order": 2 * n,
        "lam": lam,
        "units": "dimensionless",
        "physical_unit_restoration": "H_energy = E0 * H_dimensionless; dt = E0 * time (hbar=1)",
        "energy_scale_status": "E0 is an external input, not predicted",
        "electric_operator": "3I - L_r - L_r_inverse - average(L_g for g in C_s)",
        "reflection_class": reflection_class(n),
        "temporal_weight": "normalized character heat kernel",
        "square_geometry": "four distinct links, four vertex Gauss constraints, one holonomy",
        "physical_basis": "class functions, orthonormal irreducible characters",
        "energy_interpretation": "single-link electric energies or isolated-square toy energies",
        "wilson_mc_relation": "different action; existing character-Wilson MC is unchanged",
        "beta_lambda_mapping": "not assumed or calibrated",
        "limitations": [
            "This is a new central model, not the historical noncentral Hamiltonian.",
            "No glueball masses, spatial parity channels, or M4 results are inferred.",
            "No continuum limit, spatial-volume spectrum, or physical matching is established.",
        ],
    }


def _regular_action(n: int, g: tuple, *, right: bool) -> np.ndarray:
    n = _group(n)
    els = elements(n)
    if (not isinstance(g, tuple) or len(g) != 2
            or any(isinstance(x, bool) or not isinstance(x, Integral) for x in g)
            or g not in els):
        raise ValueError("g must be a canonical (k, +/-1) element of D_n")
    index = {h: i for i, h in enumerate(els)}
    action = np.zeros((len(els), len(els)))
    for j, h in enumerate(els):
        gh = mul(h, inv(g, n), n) if right else mul(g, h, n)
        action[index[gh], j] = 1.0
    return action


def left_regular(n: int, g: tuple) -> np.ndarray:
    """L_g|h> = |gh>, in the existing elements(n) ordering."""
    return _regular_action(n, g, right=False)


def right_regular(n: int, g: tuple) -> np.ndarray:
    """R_g|h> = |hg^-1>; the opposite endpoint action commutes with L."""
    return _regular_action(n, g, right=True)


def reflection_class(n: int) -> tuple:
    """Conjugacy class of s, not the union of reflection classes for even n."""
    n = _group(n)
    return next(tuple(cl) for cl in conjugacy_classes(n) if (0, -1) in cl)


def central_laplacian(n: int) -> np.ndarray:
    """Positive semidefinite central one-link electric generator Delta_c."""
    n = _group(n)
    cl = reflection_class(n)
    return (3 * np.eye(2 * n) - left_regular(n, (1, 1))
            - left_regular(n, inv((1, 1), n))
            - sum(left_regular(n, g) for g in cl) / len(cl))


def electric_spectrum(n: int) -> dict:
    """Irrep scalar epsilon of Delta_c, dimension and regular multiplicity d^2.

    Single-link electric energies are epsilon/lam; epsilon itself is independent
    of lam. Dictionary order and irrep names match character_table(n).
    """
    n = _group(n)
    cl = reflection_class(n)
    result = {}
    for name, chi in character_table(n).items():
        d = int(round(chi[(0, 1)]))
        epsilon = 3 - (chi[(1, 1)] + chi[inv((1, 1), n)]
                       + sum(chi[g] for g in cl) / len(cl)) / d
        result[name] = {"dimension": d, "epsilon": float(epsilon), "multiplicity": d * d}
    return result


def _heat_irrep_eigenvalues(spectrum: dict, lam: float, dt: float) -> np.ndarray:
    """Compute finite-time attenuation before forming possibly overflowing energies."""
    if dt == 0:
        return np.ones(len(spectrum))
    scaled_time = dt / lam
    return np.array([
        1.0 if data["epsilon"] == 0 else np.exp(-scaled_time * data["epsilon"])
        for data in spectrum.values()
    ])


def heat_kernel(n: int, lam: float, dt: float) -> np.ndarray:
    """Probability mass k_dt(g), not a density relative to Haar probability.

    k_dt(g) = sum_R d_R exp(-dt epsilon_R/lam) chi_R(g^-1) / |G|.
    Values follow elements(n). Zero time returns the exact identity mass.
    Roundoff-level negative entries at very small positive time are left visible;
    no clipping or post-hoc normalization is used.
    """
    n, lam, dt = _group(n), _real(lam, "lam"), _real(dt, "dt", positive=False)
    els = elements(n)
    if dt == 0:
        identity = np.zeros(len(els))
        identity[els.index((0, 1))] = 1.0
        return identity
    spectrum = electric_spectrum(n)
    table = character_table(n)
    weights = np.zeros(len(els))
    for (name, data), decay in zip(spectrum.items(), _heat_irrep_eigenvalues(spectrum, lam, dt)):
        weights += (data["dimension"] * decay
                    * np.array([table[name][inv(g, n)] for g in els]))
    return weights / len(els)


def heat_transfer(n: int, lam: float, dt: float) -> np.ndarray:
    """Single-link transfer sum_g k_dt(g)L_g = exp(-dt Delta_c/lam)."""
    weights = heat_kernel(n, lam, dt)
    return sum(w * left_regular(n, g) for w, g in zip(weights, elements(n)))


def _character_basis(n: int) -> tuple:
    """Character Gram and magnetic multiplication matrices, without energy scales."""
    table = character_table(n)
    els, names = elements(n), tuple(table)
    chars = np.array([[table[name][g] for name in names] for g in els])
    potential = np.array([1 - table["E1"][g] / 2 for g in els])
    gram = chars.conj().T @ chars / len(els)
    magnetic = chars.conj().T @ (potential[:, None] * chars) / len(els)
    return names, gram, magnetic


def isolated_square(n: int, lam: float) -> dict:
    """Exact physical character-basis H = 4 diag(epsilon)/lam + lam V.

    V_RS = <chi_R, (1-chi_E1/2) chi_S>_G is the magnetic multiplication
    operator. The factor four in H_E counts distinct links, not four independent
    holonomies. No four-link tensor product is formed. Extreme coupling scales
    whose ground energy is unresolved by dense float64 diagonalization raise
    FloatingPointError rather than report a spurious signed/zero ground energy.
    """
    n, lam = _group(n), _real(lam, "lam")
    names, gram, magnetic = _character_basis(n)
    epsilon = np.array([data["epsilon"] for data in electric_spectrum(n).values()])
    with np.errstate(over="ignore"):
        electric = np.diag(4 * (epsilon / lam))
    hamiltonian = electric + lam * magnetic
    if not np.all(np.isfinite(hamiltonian)):
        raise FloatingPointError("Hamiltonian exceeds floating-point range for this lam")
    # The constant character and identity-holonomy trial states bound E_ground
    # above by lam and 12/lam respectively. Reject scales where a conservative
    # dense-eigensolver error budget already reaches that bound. A rounded
    # negative/zero ground state is not evidence against the exact positive H.
    scale = float(np.linalg.norm(hamiltonian, ord=np.inf))
    spectral_resolution = 16 * np.finfo(float).eps * len(names) * scale
    if spectral_resolution >= min(lam, 12 / lam):
        raise FloatingPointError("isolated-square ground energy is numerically unresolved at this lam")
    return {
        "model_id": MODEL_ID,
        "energy_interpretation": "isolated-square toy energies",
        "n": n,
        "lam": lam,
        "irrep_names": names,
        "character_gram": gram,
        "electric": electric,
        "magnetic": magnetic,
        "hamiltonian": hamiltonian,
        "energies": np.linalg.eigvalsh(hamiltonian),
    }


def _magnetic_exponential(n: int, scale: float) -> np.ndarray:
    """Exponentiate known class potentials, never a rounded matrix zero mode."""
    table = character_table(n)
    classes = conjugacy_classes(n)
    transform = np.array([
        [np.sqrt(len(cl) / (2 * n)) * chi[cl[0]] for chi in table.values()]
        for cl in classes
    ])
    potential = np.array([1 - table["E1"][cl[0]] / 2 for cl in classes])
    # The identity-class potential is exactly zero. In particular, arbitrarily
    # long resolved magnetic steps retain it rather than exponentiating the
    # small spurious negative eigenvalue of a rounded character-basis matrix.
    with np.errstate(over="ignore"):
        attenuation = np.exp(-scale * potential)
    return transform.conj().T @ (attenuation[:, None] * transform)


def isolated_square_transfer(n: int, lam: float, dt: float) -> np.ndarray:
    """Symmetric (Strang) transfer, allowing dt=0; not exp(-dt H) at finite dt."""
    n, lam, dt = _group(n), _real(lam, "lam"), _real(dt, "dt", positive=False)
    if dt == 0:
        return np.eye(len(character_table(n)))
    magnetic_scale = (0.5 * dt) * lam
    if not np.isfinite(magnetic_scale):
        raise FloatingPointError("magnetic transfer exponent exceeds floating-point range")
    magnetic_half = _magnetic_exponential(n, magnetic_scale)
    # Four identical one-link eigenvalues avoid an overflowing unscaled H_E.
    electric_step = np.diag(_heat_irrep_eigenvalues(electric_spectrum(n), lam, dt) ** 4)
    transfer = magnetic_half @ electric_step @ magnetic_half
    if not np.all(np.isfinite(transfer)):
        raise FloatingPointError("Transfer exceeds floating-point range")
    return transfer


def effective_hamiltonian(transfer: np.ndarray, dt: float) -> np.ndarray:
    """Hermitian -log(T)/dt, rejecting nonpositive or unresolved eigenvalues.

    There is no eigenvalue floor. Highly ill-conditioned transfers require a
    smaller dt (or higher precision), rather than silently repairing a logarithm.
    """
    dt = _real(dt, "dt")
    transfer = np.asarray(transfer)
    if (transfer.ndim != 2 or transfer.shape[0] != transfer.shape[1]
            or not 1 <= transfer.shape[0] <= 24 or not np.all(np.isfinite(transfer))):
        raise ValueError("transfer must be a finite square matrix of size 1..24")
    scale = max(float(np.linalg.norm(transfer, ord=2)), np.finfo(float).tiny)
    if np.linalg.norm(transfer - transfer.conj().T, ord=2) > 1e-12 * scale:
        raise ValueError("transfer must be Hermitian")
    values, vectors = np.linalg.eigh(transfer)
    resolution = np.finfo(float).eps * len(values) * scale
    if np.min(values) <= resolution:
        raise FloatingPointError("transfer is nonpositive or numerically unresolved; reduce dt")
    with np.errstate(over="ignore"):
        energies = -np.log(values) / dt
    if not np.all(np.isfinite(energies)):
        raise FloatingPointError("effective Hamiltonian energies exceed floating-point range for this dt")
    result = (vectors * energies) @ vectors.conj().T
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("effective Hamiltonian exceeds floating-point range")
    return result


def isolated_square_diagnostics(n: int, lam: float, dt: float) -> dict:
    """Positivity/Hermiticity and H_eff -> H at dt, dt/2 and dt/4.

    In the resolved small-time regime, halving dt reduces the effective-H error
    by four (second order). Report observed errors, not an unconditional claim
    of asymptotic behavior at every input time.
    """
    dt = _real(dt, "dt")
    square = isolated_square(n, lam)
    hamiltonian = square["hamiltonian"]
    times = [dt, dt / 2, dt / 4]
    errors, minimums, hermiticity = [], [], []
    for time in times:
        transfer = isolated_square_transfer(n, lam, time)
        effective = effective_hamiltonian(transfer, time)
        errors.append(float(np.linalg.norm(effective - hamiltonian, ord=2)))
        minimums.append(float(np.min(np.linalg.eigvalsh(transfer))))
        hermiticity.append(float(np.linalg.norm(transfer - transfer.conj().T, ord=2)))
    orders = [float(np.log2(a / b)) if a > 0 and b > 0 else None
              for a, b in zip(errors[:-1], errors[1:])]
    return {
        "model_id": MODEL_ID,
        "energy_interpretation": "isolated-square toy energies",
        "times": times,
        "effective_hamiltonian_errors": errors,
        "observed_orders": orders,
        "transfer_min_eigenvalues": minimums,
        "transfer_hermiticity_errors": hermiticity,
        "hamiltonian_hermiticity_error": float(np.linalg.norm(hamiltonian - hamiltonian.conj().T)),
        "magnetic_min_eigenvalue": float(np.min(np.linalg.eigvalsh(square["magnetic"]))),
    }


def wilson_single_link(n: int, beta: float, dt: float = 1.0) -> dict:
    """Normalized character-Wilson convolution, separate from the heat model.

    w_beta(g) is proportional to exp(beta chi_E1(g)/2). Subtracting beta in
    the exponent avoids overflow and leaves the normalized weights unchanged.
    Its irrep transfer eigenvalues are sum_g w_beta(g) chi_R(g)/d_R.
    Energies are -log(t_R)/dt when the entire spectrum is resolved and positive.
    At beta=0 the transfer is a rank-one projector, so finite energies do not
    exist for all irreps; None plus a reason is returned instead of a floor.
    beta and dt are supplied dimensionless parameters, not a beta/lam map.
    """
    n, beta, dt = _group(n), _real(beta, "beta", positive=False), _real(dt, "dt")
    els, table = elements(n), character_table(n)
    weights = np.exp(beta * np.array([table["E1"][g] / 2 - 1 for g in els]))
    weights /= np.sum(weights)
    values = np.array([sum(w * chi[g] for w, g in zip(weights, els)) / chi[(0, 1)]
                       for chi in table.values()])
    transfer = sum(w * left_regular(n, g) for w, g in zip(weights, els))
    energies, reason = None, None
    resolution = np.finfo(float).eps * len(els)
    if np.min(values) > resolution:
        with np.errstate(over="ignore"):
            energies = -np.log(values) / dt
        if not np.all(np.isfinite(energies)):
            energies = None
            reason = "single-link electric energies exceed floating-point range for this dt"
    else:
        reason = "nonpositive or numerically unresolved transfer eigenvalue; no logarithmic floor"
    return {
        "model_id": WILSON_MODEL_ID,
        "energy_interpretation": "normalized Wilson single-link electric energies",
        "n": n, "beta": beta, "dt": dt,
        "irrep_names": tuple(table),
        "weights": weights,
        "transfer": transfer,
        "irrep_transfer_eigenvalues": values,
        "energies": energies,
        "undefined_reason": reason,
        "beta_lambda_mapping": "not assumed or calibrated",
    }


def wilson_comparison(n: int, beta: float, lam: float, dt: float) -> dict:
    """Compare two explicitly supplied models; do not fit or identify beta/lam."""
    n, lam, dt = _group(n), _real(lam, "lam"), _real(dt, "dt")
    wilson = wilson_single_link(n, beta, dt)
    heat = heat_transfer(n, lam, dt)
    spectrum = electric_spectrum(n)
    epsilon = np.array([data["epsilon"] for data in spectrum.values()])
    # Finite-time transfer can be resolved even when energies exceed float range.
    with np.errstate(over="ignore"):
        heat_energies = epsilon / lam
    energy_reason = None
    if not np.all(np.isfinite(heat_energies)):
        heat_energies = None
        energy_reason = "single-link electric energies exceed floating-point range for this lam"
    return {
        "heat_model_id": MODEL_ID,
        "wilson_model_id": WILSON_MODEL_ID,
        "scope": "single-link comparison, not a spatial lattice spectrum",
        "beta_lambda_mapping": "not assumed or calibrated",
        "lam": lam, "dt": dt,
        "irrep_names": wilson["irrep_names"],
        "heat_energies": heat_energies,
        "heat_energies_undefined_reason": energy_reason,
        "heat_irrep_transfer_eigenvalues": _heat_irrep_eigenvalues(spectrum, lam, dt),
        "transfer_difference_norm": float(np.linalg.norm(heat - wilson["transfer"], ord=2)),
        "wilson": wilson,
    }
