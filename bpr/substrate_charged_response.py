"""Complete finite-ring charged probes of the stipulated quantum Bose model.

Energies are signed, dimensionless model energies, not physical particle masses.
The numerical resolution screens below are heuristics, not error certificates.
"""
from fractions import Fraction
from numbers import Complex

import numpy as np

from bpr.substrate_fermionization import (
    MAX_DENSE_DIM, MAX_SITES, _EPS, _guard, _integer, _mul,
    _normal, _occupations, _real, _scale,
)
from bpr.substrate_neutral_response import grouped_spectral_measure
from bpr.substrate_vacuum_selection import all_number_model, capped_binomial

MODEL_ID = "conditional-substrate-charged-response-v1"
FROZEN_CONTROLS = tuple((L, 1.0, g) for L in (3, 4, 5) for g in (40.0, 0.7))
FROZEN_OFFSETS = (-3.0, 0.0, 2.5)
FROZEN_FREQUENCIES = (0.5j, 1 + 0.5j, 4 + 1j)
LIMITATIONS = (
    "Quantum Bose dynamics and modular neutrality are supplied assumptions.",
    "External charged insertions are diagnostic probes, not neutral-sector preparation.",
    "At weak coupling N=L is a reference sector, not a selected neutral minimum.",
    "Signed sector thresholds can be dark at a specified momentum.",
    "Upper-half-plane analyticity does not imply fixed-sector passivity.",
    "Resolution and eigensystem residuals are numerical heuristics, not certificates.",
    "Finite-ring poles are not physical particle masses or evidence for TOE status.",
)


def _components(value, name):
    """Unlike a complex modulus check, inspect both binary64 components."""
    result = np.asarray(value)
    _normal(result.real, name + " real component")
    if np.iscomplexobj(result):
        _normal(result.imag, name + " imaginary component")
    return result.item() if result.ndim == 0 else result


def _complex(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Complex):
        raise TypeError(name + " must be a numeric scalar")
    real = _real(value.real, name + " real component")
    imag = _real(value.imag, name + " imaginary component")
    return complex(real, imag)


def _dimensions(L, numbers):
    L = _integer(L, "L", 3, MAX_SITES)
    dimensions = []
    for N in numbers:
        N = _integer(N, "N", 0)
        dimension = capped_binomial(L + N - 1, N)
        if dimension > MAX_DENSE_DIM:
            raise ValueError("fixed-number dimension exceeds cap 512")
        dimensions.append(dimension)
    return L, dimensions


def _map_from_bases(lower, upper, coefficients):
    index = {state: row for row, state in enumerate(lower)}
    result = np.zeros((len(lower), len(upper)), dtype=complex)
    for col, state in enumerate(upper):
        for x, coefficient in enumerate(coefficients):
            if state[x] and coefficient != 0:
                moved = list(state)
                moved[x] -= 1
                result[index[tuple(moved)], col] += coefficient * np.sqrt(state[x])
    return _components(result, "annihilation map")


@_guard
def local_annihilation_map(L, N, x):
    """a_x: H_N -> H_(N-1), in inherited complete occupation-tuple order."""
    N = _integer(N, "N", 1)
    L, _ = _dimensions(L, (N - 1, N))
    x = _integer(x, "x", 0, L - 1)
    coefficients = np.zeros(L)
    coefficients[x] = 1.0
    return _map_from_bases(tuple(_occupations(L, N - 1)),
                           tuple(_occupations(L, N)), coefficients)


@_guard
def annihilation_map(L, N, m):
    """L^-1/2 sum_x exp(-2 pi i m x/L) a_x; canonical m in 0..L-1.

    Creation on H_N is annihilation_map(L,N+1,m).conj().T, not the
    transpose with un-conjugated Fourier coefficients.
    """
    N = _integer(N, "N", 1)
    L, _ = _dimensions(L, (N - 1, N))
    m = _integer(m, "m", 0, L - 1)
    coefficients = np.exp(-2j * np.pi * m * np.arange(L) / L) / np.sqrt(L)
    return _map_from_bases(tuple(_occupations(L, N - 1)),
                           tuple(_occupations(L, N)), coefficients)


def _difference(a, b, tolerance, name, structural_zero=False):
    value = _normal(np.subtract(a, b), name)
    unresolved = np.abs(value) <= tolerance
    if structural_zero:
        unresolved = unresolved & (value != 0)
    if np.any(unresolved):
        raise ValueError("numerically unresolved cancellation in " + name)
    return value


def _sector_spectrum(model, kappa, source_sector):
    """One eigendecomposition of the actual H+kappa*N, scaled for arithmetic."""
    offset = _mul(kappa, model.N, "number offset")
    base_scale = float(np.max(np.abs(model.H)))
    dimension = len(model.basis)
    relative = 128 * _EPS * dimension
    # A nonzero input must not disappear into a large identity offset or vice versa.
    if offset and abs(offset) <= _mul(relative, base_scale, "offset resolution"):
        raise ValueError("numerically unresolved number offset")
    if offset and base_scale <= _mul(relative, abs(offset), "Hamiltonian resolution"):
        raise ValueError("number offset obscures Hamiltonian resolution")
    H = model.H.copy()
    diagonal = np.diag_indices(dimension)
    H[diagonal] = _normal(H[diagonal] + offset, "shifted diagonal")
    if offset:
        original = np.diag(model.H)
        scale = np.maximum(np.abs(original), abs(offset))
        cancelled = (original != 0) & (np.abs(H[diagonal]) <= relative * scale)
        # Exact binary64 cancellation of specified diagonal terms is algebraic.
        if np.any(cancelled & (original != -offset)):
            raise ValueError("numerically unresolved shifted diagonal cancellation")
    scale = float(np.max(np.abs(H)))
    scaled = _normal(H / scale, "scaled shifted Hamiltonian")
    if np.any((H != 0) & (scaled == 0)):
        raise ValueError("Hamiltonian scaling underflow")
    values, vectors = np.linalg.eigh(scaled)
    _components(vectors, "eigenvectors")
    values = _normal(values, "scaled eigenvalues")
    radius = float(np.max(np.abs(values)))
    tolerance_scaled = _mul(relative, radius, "scaled spectral resolution")
    tolerance = _mul(tolerance_scaled, scale, "energy resolution")
    values = _scale(values, scale, "sector eigenvalues")
    multiplicity = int(np.count_nonzero(values - values[0] <= tolerance))
    gap = _normal(values[1] - values[0], "ground gap")
    if source_sector and (multiplicity != 1 or gap <= tolerance):
        raise ValueError("source ground state is degenerate or numerically unresolved")
    if model.g and abs(values[0]) <= tolerance:
        raise ValueError("numerically unresolved ground energy cancellation")
    reconstruction = float(np.max(np.abs(scaled - (vectors * (values / scale)) @ vectors.T)))
    orthonormality = float(np.max(np.abs(vectors.T @ vectors - np.eye(dimension))))
    hermiticity = float(np.max(np.abs(scaled - scaled.T)))
    if max(reconstruction, orthonormality, hermiticity) > tolerance_scaled:
        raise ValueError("numerically unresolved eigensystem diagnostics")
    return H, values, vectors, {
        "N": model.N, "dimension": dimension, "ground_energy": float(values[0]),
        "ground_gap": float(gap), "ground_multiplicity": multiplicity,
        "ground_status": "resolved" if multiplicity == 1 else "degenerate_or_unresolved",
        "energy_resolution": tolerance, "matrix_scale": scale,
        "scaled_reconstruction_residual": reconstruction,
        "orthonormality_residual": orthonormality,
        "scaled_hermiticity_residual": hermiticity,
        "offset": offset, "offset_method": "actual H + kappa*N*I before diagonalization",
    }


def _measure(energies, vectors, source, H, E0, tolerance):
    source = _components(source, "charged source")
    amplitudes = _components(vectors.conj().T @ source, "spectral amplitudes")
    weights = _normal(np.abs(amplitudes) ** 2, "spectral weights")
    if np.any((amplitudes != 0) & (weights == 0)):
        raise ValueError("spectral weight underflow")
    grouped = grouped_spectral_measure(energies, vectors, source, tolerance=tolerance)
    # Compute direct moments independently of spectral grouping, with scaled H.
    scale = max(float(np.max(np.abs(H))), abs(E0))
    shifted = H / scale - (E0 / scale) * np.eye(len(source))
    direct_scaled = _components(np.vdot(source, shifted @ source), "direct first moment")
    direct = _mul(float(direct_scaled.real), scale, "direct first moment")
    absolute_moment = float(np.sum(weights * np.abs(energies)))
    if absolute_moment and abs(grouped["first_moment"]) <= 128 * _EPS * len(energies) * absolute_moment:
        raise ValueError("numerically unresolved spectral first-moment cancellation")
    threshold_weight = float(np.sum(weights[energies - energies[0] <= tolerance]))
    weight_resolution = 128 * _EPS * len(energies) * grouped["total_weight"]
    grouped.update(energies=energies.tolist(), weights=weights.tolist(),
                   direct_first_moment=direct,
                   source_norm_squared=float(np.vdot(source, source).real),
                   sector_ground_group_weight=threshold_weight,
                   threshold_visibility="resolved_weight" if threshold_weight > weight_resolution else "dark_or_numerically_unresolved",
                   weight_resolution=weight_resolution,
                   representation="full numerical eigensystem; weights not clipped")
    return grouped


def _free_dispersion(L, m, C):
    if m == 0:
        return _mul(-2, C, "free dispersion")
    if 2 * m == L:
        return _mul(2, C, "free dispersion")
    if 4 * m in (L, 3 * L):
        return 0.0
    return _mul(-2 * np.cos(2 * np.pi * m / L), C, "free dispersion")


def _free_measure(energy, weight):
    first = _mul(energy, weight, "free first moment")
    return {"energies": [energy], "weights": [float(weight)],
            "groups": [{"energy": energy, "energy_min": energy, "energy_max": energy,
                        "energy_spread": 0.0, "weight": float(weight),
                        "first_moment": first, "multiplicity": 1}],
            "total_weight": float(weight), "first_moment": first,
            "direct_first_moment": first, "source_norm_squared": float(weight),
            "tolerance": 0.0, "grouping": "analytic single-line measure",
            "representation": "structural g=0 condensate oracle, not clipped numerical overlaps"}


@_guard
def spectroscopy(L, C=1.0, g=40.0, kappa=0.0):
    """All canonical momenta at N=L, L=3..5; return a strict-JSON-ready dict.

    Three sector caps are checked before allocation. Each shifted sector is
    diagonalized once. The source ground must be resolved; adjacent degeneracy
    is allowed. g=0 additionally uses labeled exact condensate spectral measures.
    No chemical potential is selected to make both insertion costs positive.
    """
    L = _integer(L, "L", 3, 5)
    _dimensions(L, (L - 1, L, L + 1))
    C = _real(C, "C", positive=True)
    g = _real(g, "g", lower=0)
    kappa = _real(kappa, "kappa")
    coupling_tolerance = 128 * _EPS * MAX_DENSE_DIM
    if g and min(C, g) <= _mul(coupling_tolerance, max(C, g), "coupling resolution"):
        raise ValueError("numerically unresolved hopping/interaction scale separation")
    models = [all_number_model(L, N, C, g) for N in (L - 1, L, L + 1)]
    spectra = [_sector_spectrum(model, kappa, i == 1) for i, model in enumerate(models)]
    E0 = spectra[1][1][0]
    ground = spectra[1][2][:, 0]
    resolution = [spectrum[3]["energy_resolution"] for spectrum in spectra]
    if g == 0:
        mu = _difference(kappa, _mul(2, C, "free hopping energy"),
                         max(resolution), "free threshold", structural_zero=True)
        mu_plus, mu_minus, gap = mu, mu, 0.0
    else:
        mu_plus = _difference(spectra[2][1][0], E0, resolution[2] + resolution[1], "mu_plus")
        mu_minus = _difference(E0, spectra[0][1][0], resolution[0] + resolution[1], "mu_minus")
        gap = _difference(mu_plus, mu_minus, resolution[0] + 2 * resolution[1] + resolution[2], "Delta_c")
    modes = []
    for m in range(L):
        if g == 0:
            dispersion = _free_dispersion(L, m, C)
            plus = _difference(dispersion, -kappa, max(resolution), "free addition energy", structural_zero=True)
            removal = _free_measure(-mu_minus, L if m == 0 else 0)
            addition = _free_measure(plus, L + 1 if m == 0 else 1)
            for measure in (removal, addition):
                measure.update(
                    sector_ground_group_weight=measure["total_weight"] if m == 0 else 0.0,
                    threshold_visibility="resolved_weight" if m == 0 else "structurally_dark",
                    weight_resolution=0.0)
        else:
            phases = np.exp(-2j * np.pi * m * np.arange(L) / L) / np.sqrt(L)
            down = _map_from_bases(models[0].basis, models[1].basis, phases)
            upper_down = _map_from_bases(models[1].basis, models[2].basis, phases)
            sources = (down @ ground, upper_down.conj().T @ ground)
            measures = []
            for i, source in zip((0, 2), sources):
                H, values, vectors, diagnostic = spectra[i]
                tolerance = resolution[i] + resolution[1]
                energies = _difference(values, E0, tolerance, "signed transition energies")
                measures.append(_measure(energies, vectors, source, H, E0, tolerance))
            removal, addition = measures
        coefficient = _normal(addition["first_moment"] + removal["first_moment"], "second coefficient")
        coefficient_scale = abs(addition["first_moment"]) + abs(removal["first_moment"])
        coefficient_resolved = abs(coefficient) > 128 * _EPS * MAX_DENSE_DIM * coefficient_scale
        # An unresolved interacting cancellation is unavailable, never a fake zero.
        if not coefficient_resolved and g != 0:
            coefficient = None
        modes.append({"m": m, "k": float(2 * np.pi * m / L),
                      "removal": removal, "addition": addition,
                      "n_k": removal["source_norm_squared"],
                      "commutator_residual": addition["total_weight"] - removal["total_weight"] - 1.0,
                      "high_frequency_second_coefficient": coefficient,
                      "second_coefficient_status": "analytic_free_control" if g == 0 else (
                          "resolved" if coefficient_resolved else "unresolved_cancellation")})
    return {"model_id": MODEL_ID, "parameters": {"L": L, "N": L, "C": C, "g": g, "kappa": kappa},
            "sectors": [spectrum[3] for spectrum in spectra],
            "thresholds": {"mu_plus": float(mu_plus), "mu_minus": float(mu_minus),
                           "Delta_c": float(gap), "removal_ground_cost": float(-mu_minus)},
            "modes": modes,
            "numerics": {"certified": False, "eigendecompositions": 3,
                         "resolution_rule": "128*eps*sector_dimension*spectral_radius, in physical model units",
                         "source_ground_required": "unique and numerically resolved",
                         "adjacent_degeneracy": "allowed; descriptive bounded-span groups",
                         "free_control": "analytic spectral measures after actual shifted sector diagonalization" if g == 0 else None},
            "limitations": list(LIMITATIONS)}


def _exact_green(channels, z):
    """Bounded rational fallback for cancellation of supplied binary64 data.

    Each input float is an exact rational here. This does not improve the
    original eigensolver accuracy; it only prevents response arithmetic from
    silently losing a normal real or imaginary component.
    """
    zr, zi = Fraction.from_float(z.real), Fraction.from_float(z.imag)
    real, imag = Fraction(0), Fraction(0)
    for energies, weights, sign in channels:
        for energy, weight in zip(energies, weights):
            if weight == 0:
                continue
            x = zr - sign * Fraction.from_float(float(energy))
            w = sign * Fraction.from_float(float(weight))
            denominator = x * x + zi * zi
            real += w * x / denominator
            imag -= w * zi / denominator
    components = []
    for value in (real, imag):
        converted = _normal(float(value), "exact response component")
        if value and converted == 0:
            raise ValueError("exact response component underflows binary64")
        components.append(converted)
    return complex(*components)


@_guard
def green_function(report, m, z):
    """Retarded commutator: sum W+/(z-d+) - sum W-/(z+d-), Im(z)>0.

    Only raw energies/weights enter denominators, never group representatives.
    """
    L = _integer(report["parameters"]["L"], "L", 3, 5)
    m = _integer(m, "m", 0, L - 1)
    z = _complex(z, "z")
    if z.imag <= 0:
        raise ValueError("retarded evaluation requires Im(z)>0")
    if not isinstance(report["modes"], (list, tuple)) or len(report["modes"]) != L:
        raise ValueError("report must contain exactly L modes")
    mode = report["modes"][m]
    terms = []
    channels = []
    exact_needed = False
    for name, sign in (("addition", 1), ("removal", -1)):
        data = mode[name]
        # Bound even caller-modified reports before invoking array conversion.
        for key in ("energies", "weights"):
            raw = data[key]
            if not isinstance(raw, (list, tuple, np.ndarray)):
                raise ValueError("raw spectral arrays must be bounded sequences")
            if isinstance(raw, np.ndarray) and raw.ndim != 1:
                raise ValueError("raw spectral arrays must be one dimensional")
            if not 1 <= len(raw) <= MAX_DENSE_DIM:
                raise ValueError("raw spectral array dimension exceeds cap 512")
        energies = np.array([_real(value, "response energy") for value in data["energies"]])
        weights = np.array([_real(value, "response weight", lower=0) for value in data["weights"]])
        if energies.ndim != 1 or weights.shape != energies.shape or np.any(weights < 0):
            raise ValueError("invalid raw spectral arrays")
        channels.append((energies, weights, sign))
        denominator = _components(z - sign * energies, "response denominator")
        # Detect an operand erased from the real denominator before division.
        active = weights != 0
        erased_z = (z.real != 0) & (energies != 0) & (denominator.real == -sign * energies)
        erased_energy = (energies != 0) & (denominator.real == z.real)
        exact_needed = exact_needed or bool(np.any(active & (erased_z | erased_energy)))
        values = _components(np.divide(sign * weights, denominator), "response terms")
        if np.any((weights != 0) & (values == 0)):
            raise ValueError("response term underflow")
        terms.extend(values.tolist())
    terms = np.asarray(terms, dtype=complex)
    result = _components(np.sum(terms), "Green function")
    for component in ("real", "imag"):
        scale = float(np.sum(np.abs(getattr(terms, component))))
        if scale and abs(getattr(result, component)) <= 128 * _EPS * len(terms) * scale:
            exact_needed = True
    if exact_needed:
        return _exact_green(channels, z)
    return complex(result)


@_guard
def demonstration_report():
    """Six frozen cases and actual number-offset checks; no file writes."""
    cases = [spectroscopy(L, C, g) for L, C, g in FROZEN_CONTROLS]
    offsets = []
    for base in cases:
        parameters = base["parameters"]
        L, C, g = (parameters[name] for name in ("L", "C", "g"))
        for kappa in FROZEN_OFFSETS:
            shifted = base if kappa == 0 else spectroscopy(L, C, g, kappa)
            green_residual = max(abs(green_function(shifted, m, z) - green_function(base, m, z - kappa))
                                 for m in range(L) for z in FROZEN_FREQUENCIES)
            offsets.append({"L": L, "g": g, "kappa": kappa,
                            "method": "actual H+kappa*N*I eigensystems; kappa=0 reuses base",
                            "thresholds": shifted["thresholds"],
                            "Delta_c_residual": shifted["thresholds"]["Delta_c"] - base["thresholds"]["Delta_c"],
                            "green_translation_max_residual": float(green_residual)})
        for mode in base["modes"]:
            m = mode["m"]
            samples = []
            for z in FROZEN_FREQUENCIES:
                response = green_function(base, m, z)
                samples.append({"z": [float(z.real), float(z.imag)],
                                "G": [float(response.real), float(response.imag)]})
            mode["green_samples"] = samples
            z = 10000j
            coefficient = mode["high_frequency_second_coefficient"]
            remainder_bound = sum(w * d * d / (abs(z) ** 2 * z.imag)
                                  for name in ("addition", "removal")
                                  for d, w in zip(mode[name]["energies"], mode[name]["weights"]))
            residual = None if coefficient is None else abs(
                green_function(base, m, z) - 1 / z - coefficient / z ** 2)
            mode["high_frequency_check"] = {
                "z": [0.0, 10000.0], "residual": residual,
                "algebraic_remainder_bound": remainder_bound,
                "roundoff_included": False,
                "bound_formula": "sum W*d^2 / (abs(z)^2*Im(z)) over both channels"}
    return {"model_id": MODEL_ID, "cases": cases, "number_offset_controls": offsets,
            "caps": {"sites": [3, 5], "sector_dimension": MAX_DENSE_DIM},
            "limitations": list(LIMITATIONS)}
