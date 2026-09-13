"""Bounded charge-resolved band/flux diagnostics of strict hard-core compression.

The frozen derivation is doc/derivations/substrate_chirality_2026-09-12.md.
Group velocity here is a 1+1D band diagnostic, not physical Weyl chirality.
Finite-g witnesses reuse the unchanged complete-basis Bose model. No new
Hamiltonian, population-selection mechanism, or physical prediction is added.
Only finite normal binary64 inputs/intermediates (and structural zero) are
supported. Numerical ties and residuals are diagnostics, not certified bounds.
"""
from itertools import combinations
from math import comb, pi

import numpy as np

from . import substrate_fermionization as sf

MODEL_ID = "conditional-substrate-chirality-v1"
MAX_ONE_BODY_SITES = 512
MAX_SLATER_DIM = 512
MAX_BINARY_DIM = 256
MAX_DENSE_DIM = 512
_TAU = 2 * pi
_EPS = np.finfo(float).eps

LIMITATIONS = (
    "Strict hard-core compression is not invariant finite-g Bose dynamics.",
    "Opposite group velocities are 1+1D band branches, not physical 3+1D Weyl chirality.",
    "Flux is an external diagnostic; a 2pi cycle permutes mode labels and has zero net signed flow.",
    "The filling reference equals the zero-flux symmetric ground-energy difference, not generally an addition/removal energy or flux-dependent occupation cut; it selects no population.",
    "Neutrality is independently imposed; Jordan-Wigner strings retain charge one.",
    "Finite-g second-order witnesses are algebraic negative controls, not an exact free-fermion Hamiltonian.",
    "Numerical tie tolerances and residuals do not certify exact degeneracies or roundoff bounds.",
    "No physical gauge sector, chiral particle, mass, family count, or theory of everything is derived.",
)


def _sector(L, N):
    L = sf._integer(L, "L", 3, MAX_ONE_BODY_SITES)
    N = sf._integer(N, "N", 0, L)
    return L, N


def _parameters(L, N, C, phi=0.0):
    L, N = _sector(L, N)
    C = sf._real(C, "C", positive=True)
    sf._mul(2.0, C, "band width")
    phi = sf._real(phi, "phi", lower=0)
    if phi > _TAU:
        raise ValueError("phi outside supported closed diagnostic cycle [0,2pi]")
    return L, N, C, phi


def _binary_cap(L):
    if L > 8 or (1 << L) > MAX_BINARY_DIM:
        raise ValueError(f"complete binary dimension exceeds cap {MAX_BINARY_DIM}")


def _slater_cap(L, N):
    dimension = comb(L, N)
    if dimension > MAX_SLATER_DIM:
        raise ValueError(f"Slater dimension {dimension} exceeds cap {MAX_SLATER_DIM}")
    return dimension


def boundary_phase(N):
    """theta_N=0 for odd N, pi for even N, including the empty control."""
    N = sf._integer(N, "N", 0, MAX_ONE_BODY_SITES)
    return 0.0 if N % 2 else pi


def _phase(phi):
    # Exact endpoints and quadrants prevent spurious imaginary closing links.
    special = {0.0: 1.0 + 0j, pi / 2: 1j, pi: -1.0 + 0j,
               3 * pi / 2: -1j, _TAU: 1.0 + 0j}
    if phi in special:
        return special[phi]
    return complex(sf._normal(np.cos(phi), "flux cosine"),
                   sf._normal(np.sin(phi), "flux sine"))


def _trig_quarters(numerator, denominator):
    """sin/cos of pi*numerator/(2*denominator), with exact quadrants."""
    numerator %= 4 * denominator
    if numerator % denominator == 0:
        quadrant = numerator // denominator
        return ((1., 0.), (0., 1.), (-1., 0.), (0., -1.))[quadrant]
    angle = (pi / 2) * (numerator / denominator)
    return float(np.cos(angle)), float(np.sin(angle))


@sf._guard
def one_body_hopping(L, N, C=1.0, phi=0.0):
    """h[x,x+1]=-C; h[L-1,0]=-C exp(i(theta_N+phi)).

    L<=512; 0<=phi<=2pi. Both endpoints are exact identical matrices.
    """
    L, N, C, phi = _parameters(L, N, C, phi)
    h = np.zeros((L, L), dtype=complex)
    for x in range(L - 1):
        h[x, x + 1] = h[x + 1, x] = -1.0
    h[L - 1, 0] = -(1 if N % 2 else -1) * _phase(phi)
    h[0, L - 1] = h[L - 1, 0].conjugate()
    return sf._scale(h, C, "one-body hopping")


@sf._guard
def one_body_spectrum(L, N, C=1.0, phi=0.0):
    """Mode-labelled arrays, NOT sorted: k_m=(2pi*m+theta_N+phi)/L.

    At phi=2pi, energies permute m->m+1 modulo L relative to phi=0.
    Exact quadrants use structural trig values, including exact zero levels.
    """
    L, N, C, phi = _parameters(L, N, C, phi)
    theta = boundary_phase(N)
    theta_quarters = 0 if N % 2 else 2
    phi_quarters = {0.: 0, pi / 2: 1, pi: 2, 3 * pi / 2: 3, _TAU: 4}.get(phi)
    cosines, sines = [], []
    if phi_quarters is None:
        delta = sf._div(phi, L, "flux angle per site")
        cp, sp = float(np.cos(delta)), float(np.sin(delta))
    for m in range(L):
        if phi_quarters is not None:
            co, si = _trig_quarters(4 * m + theta_quarters + phi_quarters, L)
        else:
            co, si = _trig_quarters(4 * m + theta_quarters, L)
            co, si = co * cp - si * sp, si * cp + co * sp
        cosines.append(co)
        sines.append(si)
    band = sf._mul(2., C, "band width")
    energies = sf._scale(np.asarray(cosines), -band, "one-body energies")
    velocities = sf._scale(np.asarray(sines), band, "mode velocities")
    slopes = sf._scale(velocities, 1. / L, "mode flux slopes")
    momenta = sf._normal((np.arange(L) * _TAU + theta + phi) / L, "mode momenta")
    return {"momenta": momenta, "energies": energies, "velocities": velocities,
            "flux_slopes": slopes, "theta": theta, "phi": phi}


@sf._guard
def slater_energies(L, N, C=1.0, phi=0.0):
    """Sorted actual N-body energies: all combinations of N DISTINCT modes.

    Check comb(L,N)<=512 before enumerating; this is not the one-body spectrum.
    """
    L, N, C, phi = _parameters(L, N, C, phi)
    dimension = _slater_cap(L, N)
    sf._mul(2 * N, C, "many-body spectral scale")
    if N in (0, L):
        return np.zeros(1)  # Trace(h)=0, exactly, as is the vacuum energy.
    levels = one_body_spectrum(L, N, C, phi)["energies"]
    result = np.fromiter((sum(float(levels[m]) for m in modes)
                          for modes in combinations(range(L), N)), float, count=dimension)
    return np.sort(sf._normal(result, "Slater energies"))


@sf._guard
def hard_core_hopping(L, N, C=1.0, phi=0.0):
    """Independent commuting hard-core hopping, ascending binary-bit order.

    Closing boson link is -C exp(i phi) b[L-1]^dagger b[0] + h.c.;
    no parity twist and no fermionic signs are inserted here. Full binary
    space is capped at256 before basis generation, even for vacuum/full N.
    """
    L, N, C, phi = _parameters(L, N, C, phi)
    _binary_cap(L)
    sf._mul(2 * N, C, "hard-core spectral scale")
    bits = [bit for bit in range(1 << L) if bin(bit).count("1") == N]
    index = {bit: i for i, bit in enumerate(bits)}
    hopping = np.zeros((len(bits), len(bits)), dtype=complex)
    closing = _phase(phi)
    for col, bit in enumerate(bits):
        for x in range(L):
            y = (x + 1) % L
            weight = closing if x == L - 1 else 1.0
            for target, source, amplitude in ((x, y, weight), (y, x, np.conjugate(weight))):
                if bit & (1 << source) and not bit & (1 << target):
                    row = bit ^ (1 << source) ^ (1 << target)
                    hopping[index[row], col] -= amplitude
    return sf._scale(hopping, C, "hard-core hopping")


@sf._guard
def ground_filling(L, N, C=1.0, phi=0.0):
    """Ordered levels and one ground occupation, with explicit numerical ties.

    The representative occupation is not a uniquely selected state. The
    binomial degeneracy counts a numerical tie cluster at the occupation cut.
    Empty/full sectors have no occupation cut, gap=None, and degeneracy1.
    """
    L, N, C, phi = _parameters(L, N, C, phi)
    sf._mul(2 * N, C, "ground energy scale")
    unit = one_body_spectrum(L, N, 1.0, phi)["energies"]
    order = np.argsort(unit, kind="stable")
    ordered = unit[order]
    tolerance_unit = 64 * _EPS * max(1, L)
    tolerance = sf._mul(tolerance_unit, C, "level tie tolerance")
    gap, cut_tie, degeneracy, tie_modes = None, False, 1, []
    if 0 < N < L:
        gap_unit = sf._normal(ordered[N] - ordered[N - 1], "occupation gap")
        gap = sf._mul(gap_unit, C, "occupation gap")
        cut_tie = bool(gap_unit <= tolerance_unit)
        if cut_tie:
            cluster = np.flatnonzero(np.abs(ordered - ordered[N - 1]) <= tolerance_unit)
            below = int(cluster[0])
            degeneracy = comb(len(cluster), N - below)
            tie_modes = [int(order[i]) for i in cluster]
    energy_unit = 0.0 if N in (0, L) else float(np.sum(ordered[:N]))
    energy = sf._mul(energy_unit, C, "ground energy")
    return {"ordered_levels": sf._scale(ordered, C, "ordered levels").tolist(),
            "ordered_modes": order.tolist(), "occupied_modes": order[:N].tolist(),
            "ground_energy": energy, "occupation_gap": gap,
            "cut_tie_numerical": cut_tie, "ground_degeneracy_numerical": degeneracy,
            "cut_tie_modes": tie_modes, "tie_tolerance": tolerance,
            "tie_classification": "numerical absolute level tolerance, not an exact degeneracy certificate",
            "representative_not_unique_when_tied": cut_tie}


@sf._guard
def _reference(L, N, C, mu):
    if mu is not None:
        return sf._real(mu, "mu"), "explicit band reference"
    if N in (0, L):
        return (-2 * C if N == 0 else 2 * C), "empty/full control"
    cosine, _ = _trig_quarters(2 * N, L)
    return sf._mul(-2 * C, cosine, "filling reference"), "thermodynamic filling convention; also the exact zero-flux symmetric ground-energy difference, not generally an addition/removal energy or flux-dependent cut"


@sf._guard
def branch_diagnostics(L, N, C=1.0, mu=None):
    """Band-reference branches. Default mu=-2C cos(pi N/L).

    Outside band: no events/branches; exact edges: one zero-velocity tangency.
    Near edges remain transverse; use relative distance to the edge to avoid
    loss in acos near +/-1, and compute speed independently of angle rounding.
    """
    L, N, C, _ = _parameters(L, N, C)
    mu, reference = _reference(L, N, C, mu)
    band = sf._mul(2., C, "band width")
    result = {"L": L, "N": N, "C": C, "mu": mu, "reference": reference,
              "status": "outside_band", "branches": [], "two_nonzero_velocity_branches": False}
    if abs(mu) > band:
        return result
    if abs(mu) == band:
        k = 0.0 if mu < 0 else pi
        result.update(status="band_edge", branches=[{"k": k, "velocity": 0.0,
                      "flux_slope": 0.0, "sign": 0, "kind": "tangency"}])
        return result
    if mu == 0:
        k, speed = pi / 2, band
    else:
        distance = sf._normal(band - abs(mu), "distance to band edge")
        gap = sf._div(distance, band, "relative band-edge distance")
        half_gap = sf._div(gap, 2., "half relative edge distance")
        angle = float(2 * np.arcsin(np.sqrt(half_gap)))
        sine = float(np.sqrt(sf._mul(gap, 2. - gap, "squared branch sine")))
        speed = sf._mul(band, sine, "branch speed")
        k = angle if mu < 0 else pi - angle
        if not 0 < k < pi or speed == 0:
            raise ValueError("numerically unresolved transverse branch")
    slope = sf._div(speed, L, "branch flux slope")
    result.update(status="interior", two_nonzero_velocity_branches=True,
                  branches=[{"k": k, "velocity": speed, "flux_slope": slope,
                             "sign": 1, "kind": "crossing"},
                            {"k": -k, "velocity": -speed, "flux_slope": -slope,
                             "sign": -1, "kind": "crossing"}])
    return result


@sf._guard
def flux_crossings(L, N, C=1.0, mu=0.0):
    """Signed events in half-open [0,2pi), retaining different tied branches.

    Exact mu=0 and band-edge roots use integer quarter-turn arithmetic;
    mu=+/-C uses exact sixth-turn arithmetic.
    Generic near-endpoint roots are NEVER snapped to zero: their numerical
    proximity is reported. If floating arithmetic itself lands an unproved
    endpoint at zero, reject it instead of inventing an exact crossing.
    """
    L, N, C, _ = _parameters(L, N, C)
    branches = branch_diagnostics(L, N, C, mu)
    reference_mu = branches["mu"]
    theta_quarters = 0 if N % 2 else 2
    events = []
    tolerance = 64 * _EPS * _TAU * L
    band = 2 * C
    for branch in branches["branches"]:
        k = branch["k"] % _TAU
        structural = reference_mu == 0 or abs(reference_mu) in (C, band)
        if structural:
            if abs(reference_mu) == C:
                # cos(k)=+/-1/2: exact signed sixth-turn roots.
                sixth = branch["sign"] * (1 if reference_mu < 0 else 2)
                mode, remainder = divmod(L * sixth - (0 if N % 2 else 3), 6)
                phi = remainder * (pi / 3)
            else:
                quarter = (1 if branch["sign"] > 0 else 3) if reference_mu == 0 else (0 if reference_mu < 0 else 2)
                mode, remainder = divmod(L * quarter - theta_quarters, 4)
                phi = remainder * (pi / 2)
            mode %= L
        else:
            # Keep signed beta until after solving: 2pi-k loses a small k.
            root = sf._normal(L * branch["k"] - boundary_phase(N), "flux root")
            mode = int(np.floor(root / _TAU))
            phi = sf._normal(root - mode * _TAU, "flux remainder")
            if not 0 < phi < _TAU:
                raise ValueError("numerically unresolved generic endpoint root; structural roots are handled separately")
            mode %= L
        exact_endpoint = bool(structural and phi == 0)
        numerical_endpoint = bool(not structural and min(phi, _TAU - phi) <= tolerance)
        events.append({"mode": mode, "phi": phi, "k": k,
                       "velocity": branch["velocity"], "flux_slope": branch["flux_slope"],
                       "sign": branch["sign"], "kind": branch["kind"],
                       "endpoint_exact": exact_endpoint, "endpoint_tie_numerical": numerical_endpoint})
    events.sort(key=lambda event: (event["phi"], event["mode"], event["sign"]))
    tied = any(abs(a["phi"] - b["phi"]) <= tolerance
               for i, a in enumerate(events) for b in events[i + 1:])
    return {"L": L, "N": N, "C": C, "mu": reference_mu,
            "interval": "[0,2pi)", "status": branches["status"], "events": events,
            "net_signed_flow": sum(event["sign"] for event in events),
            "event_count": len(events), "tied_event_fluxes_numerical": bool(tied),
            "endpoint_tolerance": tolerance,
            "endpoint_policy": "exact structural zero included; 2pi duplicate excluded; numerical proximity never snapped"}


@sf._guard
def charge_neutrality_diagnostics(L, q):
    """Complete-binary operator checks; no fixed-N creation compression trick.

    Uses the existing Jordan-Wigner operators. Report operator identities via
    maximum absolute matrix-entry residuals (explicitly NOT operator norms).
    Neutral offsite bilinears may survive for q<L but vanish at q=L, where
    only the one-dimensional empty/full sectors remain.
    """
    L, _ = _sector(L, 0)
    q = sf._integer(q, "q", 2)
    _binary_cap(L)
    ops = sf.binary_operators(L)
    number = np.diag(ops["number"])
    neutral = number.astype(int) % q == 0
    charge = bilinear_charge = projected_creation = projected_bilinear = string_charge = 0.0
    for x in range(L):
        creation = ops["c"][x].T
        commutator = (number[:, None] - number[None, :]) * creation
        charge = max(charge, float(np.max(np.abs(commutator - creation))))
        projected_creation = max(projected_creation, float(np.max(np.abs(creation[np.ix_(neutral, neutral)]))))
        string = np.array([(-1) ** bin(bit & ((1 << x) - 1)).count("1") for bit in range(1 << L)])
        # Diagonal JW string and N commute; compute the full diagonal-operator commutator.
        string_commutator = (number[:, None] - number[None, :]) * np.diag(string)
        string_charge = max(string_charge, float(np.max(np.abs(string_commutator))))
        for y in range(L):
            bilinear = creation @ ops["c"][y]
            commutator = (number[:, None] - number[None, :]) * bilinear
            bilinear_charge = max(bilinear_charge, float(np.max(np.abs(commutator))))
            if x != y:
                projected_bilinear = max(projected_bilinear,
                    float(np.max(np.abs(bilinear[np.ix_(neutral, neutral)]))))
    sectors = [{"N": n, "dimension": comb(L, n)} for n in range(L + 1) if n % q == 0]
    return {"L": L, "q": q, "binary_dimension": 1 << L,
            "neutral_dimension": int(np.count_nonzero(neutral)), "sectors": sectors,
            "assumed_neutrality": True, "rule": "N mod q = 0",
            "residual_metric": "maximum absolute matrix entry",
            "creation_charge_one_residual": charge,
            "bilinear_charge_zero_residual": bilinear_charge,
            "projected_creation_max_abs": projected_creation,
            "projected_offsite_bilinear_max_abs": projected_bilinear,
            "jw_string_number_commutator_residual": string_charge,
            "only_empty_full": q == L,
            "selected_full_sector_has_partial_filling": False,
            "dynamical_confinement_derived": False}


@sf._guard
def finite_g_witnesses(C=1.0, g=40.0):
    """Existing L5,N2 complete-Bose second-order adjacent/correlated witnesses."""
    C = sf._real(C, "C", positive=True)
    g = sf._real(g, "g", positive=True)
    # Check the inherited full-basis cap BEFORE model construction.
    if comb(5 + 2 - 1, 2) > MAX_DENSE_DIM:
        raise ValueError("complete Bose dimension exceeds cap")
    model = sf.fixed_number_model(5, 2, C, g)
    second = sf.second_order_effective(model)
    # Any number-conserving quadratic in these JW variables has diagonal
    # constant + sum(d_x n_x). Fit the COMPLETE fixed-N diagonal, not merely
    # the two selected entries. Normalize before fitting to avoid scale loss.
    occupations = np.asarray(model.hard_core_basis, dtype=float)
    design = np.column_stack((np.ones(len(occupations)), occupations))
    diagonal_unit = np.diag(second["correction_coefficient"])
    fitted, _, _, _ = np.linalg.lstsq(design, diagonal_unit, rcond=None)
    residual_unit = sf._normal(diagonal_unit - design @ fitted, "additive diagonal residual")
    residual = sf._scale(residual_unit, second["scale"], "additive diagonal residual")
    diagonal_fit = {"basis": [list(state) for state in model.hard_core_basis],
                    "diagonal": np.diag(second["correction"]).tolist(),
                    "design": "constant + sum_x d_x n_x",
                    "residual_max_abs": float(np.max(np.abs(residual))),
                    "residual_max_abs_in_C_squared_over_g": float(np.max(np.abs(residual_unit))),
                    "residuals": residual.tolist(),
                    "scope": "number-conserving quadratic in the inherited Jordan-Wigner variables; not arbitrary unitary redefinitions",
                    "numerical_fit_not_exact_certificate": True}
    return {"L": 5, "N": 2, "C": C, "g": g, "full_dimension": len(model.basis),
            "hard_core_dimension": len(model.p_indices),
            "scale_C_squared_over_g": second["scale"],
            "virtual_witnesses": sf._virtual_witnesses(model, second),
            "additive_diagonal_fit": diagonal_fit,
            "total_addition_resolved": second["total_addition_resolved"],
            "certificate": sf.validity_certificate(model),
            "finite_g_exact_free_fermion_interpretation": False,
            "other_emergent_fermion_mechanisms_excluded": False}


@sf._guard
def case_report(L, N, C=1.0, phi=0.0, mu=None, q=None):
    """Bounded JSON-safe numerical report, with independent many-body check."""
    L, N, C, phi = _parameters(L, N, C, phi)
    _binary_cap(L)
    _slater_cap(L, N)
    if q is not None:
        q = sf._integer(q, "q", 2)
    reference_mu, reference = _reference(L, N, C, mu)
    one = one_body_spectrum(L, N, C, phi)
    slater = slater_energies(L, N, C, phi)
    # Solve dimensionless matrices so eigensolver error scales with C rather
    # than depending on an arbitrary physical energy unit.
    hard_unit = np.linalg.eigvalsh(hard_core_hopping(L, N, 1.0, phi))
    hard = sf._scale(hard_unit, C, "hard-core eigenvalues")
    one_numeric = sf._scale(np.linalg.eigvalsh(one_body_hopping(L, N, 1.0, phi)), C, "one-body eigenvalues")
    hard_residual = sf._normal(float(np.max(np.abs(hard - slater))), "many-body spectral residual")
    one_residual = sf._normal(float(np.max(np.abs(one_numeric - np.sort(one["energies"])))), "one-body spectral residual")
    return {"parameters": {"L": L, "N": N, "C": C, "phi": phi, "mu": reference_mu, "q": q},
            "reference": reference,
            "dimensions": {"one_body": L, "hard_core": len(slater), "binary": 1 << L},
            "one_body": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in one.items()},
            "spectra": {"slater": slater.tolist(), "hard_core": hard.tolist(),
                        "one_body_numeric": one_numeric.tolist()},
            "ground_filling": ground_filling(L, N, C, phi),
            "branches": branch_diagnostics(L, N, C, mu),
            "flux_crossings": flux_crossings(L, N, C, reference_mu),
            "neutrality": None if q is None else charge_neutrality_diagnostics(L, q),
            "numerical_checks": {"hard_core_slater_max_abs_difference": hard_residual,
                                 "one_body_max_abs_difference": one_residual,
                                 "roundoff_certified": False}}


def demonstration_report():
    """Frozen controls; no fitting, writes, or finite-g retuning."""
    cases = [case_report(L, N, phi=phi, mu=mu)
             for L, N in ((5, 0), (5, 1), (5, 2), (5, 3), (5, 5), (6, 2), (6, 3))
             for phi in (0.0, pi / 3, _TAU) for mu in (0.0, None)]
    reference_controls = [flux_crossings(5, 2, mu=mu) for mu in (-3., -2., 0., 2., 3.)]
    return {"model_id": MODEL_ID, "controls_frozen_before_evaluation": True,
            "empirical_calibration": False,
            "caps": {"one_body_sites": MAX_ONE_BODY_SITES, "slater_dimension": MAX_SLATER_DIM,
                     "binary_dimension": MAX_BINARY_DIM, "full_bose_dimension": MAX_DENSE_DIM},
            "cases": cases, "reference_controls": reference_controls,
            "endpoint_controls": [flux_crossings(4, N, mu=0.) for N in (2, 1)],
            "neutrality_controls": [charge_neutrality_diagnostics(5, q) for q in (2, 5)],
            "finite_g_controls": [finite_g_witnesses(g=g) for g in (40., .7)],
            "limitations": list(LIMITATIONS)}
