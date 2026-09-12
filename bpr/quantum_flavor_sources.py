"""Conditional six-mode internal fermion diagnostic, not microscopic matching.

NEW instantaneous operator ansatz, NOT a derivation from the substrate or an
exact elimination of a dynamical scalar. Equal-time full squares and normal
ordered same-species quartics are distinct models unless their one-body
counterterm is compensated. A temporal kernel and controlled contact limit are
missing. Internal spin is not spacetime spin; physical masses/mixing are unset.

Fock basis |n_0,...,n_5> = (c_0†)^n0 ... (c_5†)^n5 |0>, integer
index sum_i 2**i*n_i. Modes 0:3 are u, modes 3:6 are d, in m=(1,0,-1)
order. All species anticommute. The 1+1 tensor basis |i>u |j>d maps to
index (1<<i)+(1<<(3+j)); it is not the sorted integer block order.
"""
from dataclasses import asdict
from functools import lru_cache
from math import comb
from numbers import Integral, Real

import numpy as np

from bpr.chiral_flavor_prototype import HARMONIC_DEGREES, HARMONIC_NORMALIZATIONS
from bpr.flavor_source_selection import (
    SelectionParameters, coherent_occupation, effective_energy, response_weights,
    validated_density_matrix,
)

MODEL_ID = "conditional-internal-fermion-flavor-sources-v1"
ORDERINGS = ("full_square", "normal_ordered")
DIMENSION = 64
# Bounds apply to normalized state validation, and relative operator residuals.
# No clipping, trace repair, implicit normalization, or eigenvector selection.
TOLERANCE = 512 * np.finfo(float).eps
CASIMIR_COEFFICIENTS = np.array([1/4, 3/8, 1/8]) / np.pi


def _finite(value, name):
    if not np.all(np.isfinite(value)):
        raise ValueError(f"numerically unresolved {name}: nonfinite result")
    return value


def _integer(value, name, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} must be in [0,{maximum}]")
    return int(value)


def _species(species):
    if species not in ("u", "d"):
        raise ValueError("species must be 'u' or 'd'")
    return 0 if species == "u" else 1


def _parameters(parameters):
    if parameters is None:
        return SelectionParameters()
    if not isinstance(parameters, SelectionParameters):
        raise TypeError("parameters must be SelectionParameters")
    return parameters


def _ordering(ordering):
    if ordering not in ORDERINGS:
        raise ValueError(f"ordering must be one of {ORDERINGS}")
    return ordering


def _scales(parameters):
    p = _parameters(parameters)
    a, b = response_weights(p)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        g = np.array([p.g_u, p.g_d])
        self_weights = g[:, None]**2 * a / (2 * p.R**2)
        cross_weights = (p.g_u * p.g_d / p.R**2) * b
        linear = p.h0 * g
    _finite(self_weights, "self-interaction weights")
    _finite(cross_weights, "cross-interaction weights")
    _finite(linear, "linear energies")
    if np.any(self_weights <= 0) or (p.eta > 0 and np.any(cross_weights <= 0)):
        raise ValueError("numerically unresolved positive interaction scale")
    if p.h0 > 0 and np.any(linear <= 0):
        raise ValueError("numerically unresolved linear scale")
    return p, self_weights, cross_weights, linear


def _check_normal_response_resolution(p, self_weights, ordering):
    if ordering == "normal_ordered" and p.kappa > 0:
        differences = self_weights[:, :1]-self_weights[:, 1:]
        if np.any(differences <= TOLERANCE*self_weights[:, :1]):
            raise ValueError("numerically unresolved normal-ordered response differences at positive kappa")


def spin_one_generators():
    """Conventional J with [Jx,Jy]=iJz; polynomial convention uses K=(Jx,-Jy,Jz)."""
    jx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], complex) / np.sqrt(2)
    jy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], complex) / np.sqrt(2)
    return np.array([jx, jy, np.diag([1., 0., -1.])])


def overlap_operators():
    """Nine algebraic P(Y_A), independently checkable against profile quadrature.

    P(x_i)=K_i/2, P(x_i x_j)=delta_ij I/5+{K_i,K_j}/10.
    The reflected y is essential to the prototype's complex density convention.
    """
    k = spin_one_generators()
    k[1] *= -1
    identity = np.eye(3)

    def quadratic(i, j):
        return (identity/5 if i == j else np.zeros((3, 3))) + (k[i]@k[j]+k[j]@k[i])/10

    polynomials = np.array([identity, k[0]/2, k[1]/2, k[2]/2,
                            quadratic(0, 1), quadratic(1, 2),
                            (3*quadratic(2, 2)-identity)/2,
                            quadratic(0, 2), quadratic(0, 0)-quadratic(1, 1)])
    return polynomials * HARMONIC_NORMALIZATIONS[:, None, None]


@lru_cache(maxsize=1)
def _annihilators():
    result = np.zeros((6, DIMENSION, DIMENSION), complex)
    for mode in range(6):
        for ket in range(DIMENSION):
            if ket & (1 << mode):
                sign = (-1)**(bin(ket & ((1 << mode)-1)).count("1"))
                result[mode, ket ^ (1 << mode), ket] = sign
    result.flags.writeable = False
    return result


def annihilation_operators():
    """Copy of six 64x64 CAR annihilators in documented bit order."""
    return _annihilators().copy()


@lru_cache(maxsize=1)
def _bilinears():
    c = _annihilators()
    result = np.array([[ [c[3*s+i].conj().T @ c[3*s+j] for j in range(3)]
                       for i in range(3)] for s in range(2)])
    result.flags.writeable = False
    return result


def second_quantization(matrix, species):
    """dGamma(T)=sum_ij c_i† T_ij c_j; finite matrix, not necessarily Hermitian."""
    s = _species(species)
    matrix = np.asarray(matrix, complex)
    if matrix.shape != (3, 3):
        raise ValueError("one-body matrix must have shape (3,3)")
    _finite(matrix, "one-body matrix")
    return _finite(np.einsum("ij,ijab->ab", matrix, _bilinears()[s]), "bilinear")


@lru_cache(maxsize=1)
def _densities():
    result = np.array([[second_quantization(t, s) for t in overlap_operators()]
                       for s in ("u", "d")])
    result.flags.writeable = False
    return result


def density_operators(R=1.0):
    """Physical harmonic operators Q_aA/R², shape (2,9,64,64)."""
    if isinstance(R, (bool, np.bool_)) or not isinstance(R, Real):
        raise TypeError("R must be positive real")
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        r2 = np.float64(R)**2
        inverse = 1/r2
    if not np.isfinite(R) or R <= 0 or not np.isfinite(r2) or not np.isfinite(inverse) or inverse <= 0:
        raise ValueError("numerically unresolved positive radius squared")
    return _finite(_densities()*inverse, "physical density harmonics")


def number_operators():
    return np.array([second_quantization(np.eye(3), species) for species in ("u", "d")])


def number_sector_indices(N_u, N_d):
    """Sorted integer Fock indices; dimensions binom(3,Nu)*binom(3,Nd)."""
    nu, nd = _integer(N_u, "N_u", 3), _integer(N_d, "N_d", 3)
    return np.array([i for i in range(DIMENSION)
                     if bin(i & 7).count("1") == nu and bin(i >> 3).count("1") == nd], int)


def one_plus_one_indices():
    """Tensor-product order, not sorted bit order."""
    return np.array([(1 << i)+(1 << (3+j)) for i in range(3) for j in range(3)])


def restrict_to_sector(operator, N_u, N_d):
    operator = np.asarray(operator, complex)
    if operator.shape != (DIMENSION, DIMENSION):
        raise ValueError("operator must have shape (64,64)")
    _finite(operator, "operator")
    indices = number_sector_indices(N_u, N_d)
    return operator[np.ix_(indices, indices)]


def normal_ordered_square(matrix, species):
    """Direct quartic -sum_ijkl T_ij T_kl c_i† c_k† c_j c_l.

    This independent CAR construction does NOT define ordering by subtraction.
    """
    s = _species(species)
    t = np.asarray(matrix, complex)
    if t.shape != (3, 3):
        raise ValueError("one-body matrix must have shape (3,3)")
    _finite(t, "one-body matrix")
    c = _annihilators()[3*s:3*s+3]
    result = np.zeros((64, 64), complex)
    for i, j, k, l in np.ndindex(3, 3, 3, 3):
        if t[i, j] != 0 and t[k, l] != 0:
            result -= t[i, j]*t[k, l] * (c[i].conj().T @ c[k].conj().T @ c[j] @ c[l])
    return _finite(result, "normal ordered quartic")


def ordering_counterterm(parameters=None):
    """H_normal-H_full=K(a)*(g_u² Nu+g_d² Nd)/(2R²).

    Positive one-body counterterm, generally NOT a number-independent constant.
    """
    _, self_weights, _, _ = _scales(parameters)
    coefficients = self_weights @ CASIMIR_COEFFICIENTS
    return _finite(np.einsum("s,sij->ij", coefficients, number_operators()), "counterterm")


def _angular_hamiltonian(parameters=None):
    """Cross l>0 only: all omitted terms are scalars within each number block."""
    _, _, cross, _ = _scales(parameters)
    q = _densities()
    result = np.zeros((64, 64), complex)
    for A, ell in enumerate(HARMONIC_DEGREES):
        if ell:
            result -= cross[ell] * (q[0, A] @ q[1, A])
    return _finite(result, "angular Hamiltonian")


def hamiltonian(parameters=None, ordering="full_square"):
    """NEW instantaneous equal-time ansatz, no exact dynamical mediator claim."""
    _ordering(ordering)
    p, self_weights, cross, linear = _scales(parameters)
    _check_normal_response_resolution(p, self_weights, ordering)
    q = _densities()
    result = -np.einsum("s,sij->ij", linear, number_operators())
    for A, ell in enumerate(HARMONIC_DEGREES):
        result -= cross[ell] * (q[0, A] @ q[1, A])
        for s, species in enumerate(("u", "d")):
            square = q[s, A] @ q[s, A]
            if ordering == "normal_ordered":
                t = overlap_operators()[A]
                square = square - second_quantization(t@t, species)
            # Completeness gives sum_A :Q_A²:=0 on this three-mode
            # space. Remove the common a0 weight before numerical summation;
            # otherwise kappa=0 leaves spurious tiny self energies. This is
            # independently checked against direct quartics in the tests.
            weight = (self_weights[s, ell]-self_weights[s, 0]
                      if ordering == "normal_ordered" else self_weights[s, ell])
            result -= weight * square
    return _finite(result, "Hamiltonian")


def spin_channel_projectors():
    """Analytic tensor-order projectors onto internal total J=0,1,2, ranks 1,3,5."""
    identity = np.eye(9)
    x = sum(np.kron(j, j) for j in spin_one_generators())
    return {0: (x+identity)@(x-identity)/3,
            1: -(x+2*identity)@(x-identity)/2,
            2: (x+2*identity)@(x+identity)/6}


def invariant_contractions():
    """Independent Casimir-polynomial expressions for sum_A T_A tensor T_A."""
    identity = np.eye(9)
    x = sum(np.kron(j, j) for j in spin_one_generators())
    return np.array([identity/(4*np.pi), 3*x/(16*np.pi),
                     3*(x@x+x/2-4*identity/3)/(40*np.pi)])


def analytic_sector_spectrum(N_u, N_d, parameters=None, ordering="full_square"):
    """Particle/hole oracle independent of Fock diagonalization.

    For N=1,2, self S(N)=a0*N²/(4pi)+(3a1+a2)/(8pi).
    Hole dipoles transform as spin one; quadrupoles reverse sign. The gap
    formulas remove all common self, monopole and h0 terms before arithmetic.
    """
    nu, nd = _integer(N_u, "N_u", 3), _integer(N_d, "N_d", 3)
    _ordering(ordering)
    p, sw, cross, linear = _scales(parameters)
    _check_normal_response_resolution(p, sw, ordering)
    numbers = (nu, nd)
    common = -float(np.dot(linear, numbers)) - cross[0]*nu*nd/(4*np.pi)
    for s, n in enumerate(numbers):
        if ordering == "normal_ordered":
            # Algebraic cancellation BEFORE combining independent energy scales.
            # In particular N=0,1 quartics vanish exactly and at kappa=0 all
            # three weights coincide. Subtracting S-NK after adding a tiny
            # cross term would erase a perfectly resolvable physical toy term.
            normal_self = n*(n-1)/2 * (3*(sw[s, 0]-sw[s, 1])
                                       +(sw[s, 0]-sw[s, 2]))/(8*np.pi)
            common -= normal_self
        else:
            common -= sw[s, 0]*n*n/(4*np.pi)
            if n in (1, 2):
                common -= (3*sw[s, 1]+sw[s, 2])/(8*np.pi)
    active = nu in (1, 2) and nd in (1, 2)
    dimension = comb(3, nu)*comb(3, nd)
    if active:
        epsilon = (1 if nu == 1 else -1)*(1 if nd == 1 else -1)
        # Dimensionless C_l eigenvalues at x=(-2,-1,1).
        x = np.array([-2., -1., 1.])
        angular = -cross[1]*3*x/(16*np.pi) - epsilon*cross[2]*3*(x*x+x/2-4/3)/(40*np.pi)
        gaps = np.array([9*(5*cross[1]-epsilon*cross[2])/(80*np.pi),
                         3*(5*cross[1]+epsilon*cross[2])/(40*np.pi), 0.])
        if p.eta > 0 and (np.any(gaps[:2] <= 0) or np.any(~np.isfinite(gaps[:2]))):
            raise ValueError("numerically unresolved positive channel gaps")
        channels = [{"J": j, "multiplicity": 2*j+1, "energy": float(common+angular[j]),
                     "angular_energy": float(angular[j]), "gap_from_ground": float(gaps[j])}
                    for j in range(3)]
        ground_channels = [2] if p.eta > 0 else [0, 1, 2]
        ground_dimension = 5 if p.eta > 0 else 9
        gap = float(min(gaps[:2])) if p.eta > 0 else None
        energy = float(common+angular[2])
    else:
        channels = [{"J": 1 if dimension == 3 else 0, "multiplicity": dimension,
                     "energy": float(common), "angular_energy": 0., "gap_from_ground": 0.}]
        ground_channels = [channels[0]["J"]]
        ground_dimension, gap, energy = dimension, None, float(common)
    _finite([common, energy]+[ch["energy"] for ch in channels], "sector energy")
    resolution = float(TOLERANCE * max(abs(ch["energy"]) for ch in channels))
    full_resolved = gap is None or gap > resolution
    return {"numbers": [nu, nd], "dimension": dimension, "energy": energy,
            "common_energy": float(common), "ground_dimension": ground_dimension,
            "analytic_ground_channels": ground_channels, "gap": gap,
            "numerical_status": ("exact_degeneracy" if gap is None else
                                 "resolved" if full_resolved else "full_energy_unresolved"),
            "resolution": resolution, "channels": channels}


def sector_spectrum(N_u, N_d, parameters=None, ordering="full_square"):
    """Validate analytic oracle against full CAR and constant-free block spectra.

    Residual tolerance is 512 eps times operator norm (no unit-sized floor).
    Exact channel labels survive a full-energy-unresolved result. Ground
    degeneracies come from symmetry, never from a preferred eigenvector.
    """
    result = analytic_sector_spectrum(N_u, N_d, parameters, ordering)
    block = restrict_to_sector(hamiltonian(parameters, ordering), N_u, N_d)
    angular = restrict_to_sector(_angular_hamiltonian(parameters), N_u, N_d)
    expected = np.sort(np.concatenate([np.repeat(ch["energy"], ch["multiplicity"])
                                     for ch in result["channels"]]))
    expected_angular = np.sort(np.concatenate([np.repeat(ch["angular_energy"], ch["multiplicity"])
                                             for ch in result["channels"]]))
    observed = np.linalg.eigvalsh(block)
    observed_angular = np.linalg.eigvalsh(angular)
    residual = float(np.max(abs(observed-expected)))
    angular_residual = float(np.max(abs(observed_angular-expected_angular)))
    full_scale = max(float(np.linalg.norm(block, 2)), float(np.max(abs(expected))))
    # In empty/filled blocks exact angular terms vanish by trace cancellation.
    # Use the pre-restriction angular norm to bound that cancellation error,
    # never the residual itself and never h0 or self/monopole constants.
    angular_scale = max(float(np.linalg.norm(_angular_hamiltonian(parameters), 2)),
                        float(np.max(abs(expected_angular))))
    full_tolerance = TOLERANCE*full_scale
    angular_tolerance = TOLERANCE*angular_scale
    if residual > full_tolerance or angular_residual > angular_tolerance:
        raise ArithmeticError("CAR block disagrees with independent analytic sector oracle")
    if result["gap"] is not None and (angular_tolerance == 0 or result["gap"] <= angular_tolerance):
        result["numerical_status"] = "angular_splitting_unresolved"
    result.update({"fock_residual": residual, "angular_residual": angular_residual,
                   "angular_resolution": float(angular_tolerance)})
    return result


def one_plus_one_solution(parameters=None, ordering="full_square"):
    p = _parameters(parameters)
    result = sector_spectrum(1, 1, p, ordering)
    projectors = spin_channel_projectors()
    ground = projectors[2] if p.eta > 0 else np.eye(9)
    indices = one_plus_one_indices()
    angular = _angular_hamiltonian(p)[np.ix_(indices, indices)]
    # Numerical projector is a whole multiplet, not an arbitrary eigensolver vector.
    eigenvalues, vectors = np.linalg.eigh(angular)
    rank = result["ground_dimension"]
    numerical = vectors[:, :rank] @ vectors[:, :rank].conj().T
    projector_residual = float(np.linalg.norm(numerical-ground, 2))
    scale = float(np.linalg.norm(angular, 2))
    gap = result["gap"]
    projector_tolerance = TOLERANCE if gap is None else TOLERANCE*max(1., scale/gap)
    resolved = result["numerical_status"] != "angular_splitting_unresolved"
    if resolved and projector_residual > projector_tolerance:
        raise ArithmeticError("ground projector disagrees with internal-spin symmetry")
    result.update({"stable_gaps": {"J0_minus_J2": result["channels"][0]["gap_from_ground"],
                                    "J1_minus_J2": result["channels"][1]["gap_from_ground"]},
                   "ground_projector": ground, "ground_ensemble": ground/rank,
                   "projector_residual": projector_residual if resolved else None,
                   "projector_tolerance": projector_tolerance,
                   "ground_status": "complete J=2 multiplet" if p.eta > 0 else "entire 1+1 block"})
    return result


def _state_matrix(state, dimension):
    array = np.asarray(state, complex)
    _finite(array, "state")
    if array.shape == (dimension,):
        if abs(np.vdot(array, array)-1) > TOLERANCE:
            raise ValueError("state vector must have norm one; no normalization applied")
        return np.outer(array, array.conj())
    if array.shape != (dimension, dimension):
        raise ValueError(f"state must be a vector or matrix of dimension {dimension}")
    if np.max(abs(array-array.conj().T)) > TOLERANCE or abs(np.trace(array)-1) > TOLERANCE:
        raise ValueError("state must be Hermitian with trace one; no repair applied")
    if np.linalg.eigvalsh(array).min() < -TOLERANCE:
        raise ValueError("state must be positive semidefinite")
    return array.copy()


def embed_one_plus_one(state):
    """Embed a normalized 9-dimensional tensor-order state into the Fock space."""
    state = _state_matrix(state, 9)
    result = np.zeros((64, 64), complex)
    indices = one_plus_one_indices()
    result[np.ix_(indices, indices)] = state
    return result


def partial_trace_one_plus_one(state, species):
    """Reduced 3x3 density operator in tensor order, without transpose/repair."""
    s = _species(species)
    state = _state_matrix(state, 9).reshape(3, 3, 3, 3)
    return np.trace(state, axis1=1, axis2=3) if s == 0 else np.trace(state, axis1=0, axis2=2)


def one_body_occupation(state, species):
    """Gamma_ij=<c_j† c_i>; rho=Gamma/<N>, or None at exact vacuum.

    For indefinite number, rho is normalized by mean number, not a declaration
    of a sharp population. Gamma's eigenvalues obey Pauli [0,1], not those of a
    bosonic substrate. Tiny unresolved nonzero populations are rejected.
    """
    s = _species(species)
    state = _state_matrix(state, 64)
    gamma = np.einsum("ab,ijba->ji", state, _bilinears()[s])
    number = np.trace(gamma)
    if abs(number.imag) > TOLERANCE:
        raise ValueError("numerically unresolved real occupation")
    number = float(number.real)
    eigenvalues = np.linalg.eigvalsh(gamma)
    if eigenvalues.min() < -TOLERANCE or eigenvalues.max() > 1+TOLERANCE:
        raise ValueError("occupation violates Pauli bounds")
    if number == 0 and np.max(abs(gamma)) == 0:
        rho = None
    elif number <= TOLERANCE:
        raise ValueError("numerically unresolved nonzero occupation normalization")
    else:
        rho = validated_density_matrix(gamma/number)
    return {"occupation": gamma, "number": number, "normalized_occupation": rho}


def density_expectation(state, species, R=1.0):
    state = _state_matrix(state, 64)
    values = np.einsum("ij,aji->a", state, density_operators(R)[_species(species)])
    if np.max(abs(values.imag)) > TOLERANCE*max(float(np.max(abs(values))), np.finfo(float).tiny):
        raise ValueError("numerically unresolved real density expectation")
    return _finite(values.real, "density expectation")


def ground_representatives(parameters=None):
    """Explicit illustrative states, not selected axes or unique pure vacua.

    Coherent |1,1> and entangled |2,0> lie in J=2 for eta>0. At eta=0
    both remain ground representatives, but all nine states are also ground.
    """
    solution = one_plus_one_solution(parameters)
    coherent = np.zeros(9, complex)
    coherent[0] = 1
    entangled = np.zeros(9, complex)
    entangled[[2, 4, 6]] = np.array([1., 2., 1.])/np.sqrt(6)
    states = {"ensemble": solution["ground_ensemble"],
              "chosen_coherent": np.outer(coherent, coherent.conj()),
              "chosen_entangled": np.outer(entangled, entangled.conj())}
    result = {}
    for name, state in states.items():
        result[name] = {"state": state,
                        "rho_u": partial_trace_one_plus_one(state, "u"),
                        "rho_d": partial_trace_one_plus_one(state, "d"),
                        "purity": float(np.trace(state@state).real),
                        "ground_residual": float(np.linalg.norm(solution["ground_projector"]@state-state))}
    return result


def product_variance_comparison(rho_u, rho_d, parameters=None):
    """Full-square product expectation equals old classical E minus variances.

    Cross densities factorize only for product states, not generic entangled
    ground representatives. The old classical purity theorem is unchanged.
    """
    p, sw, _, _ = _scales(parameters)
    rhos = [validated_density_matrix(rho_u), validated_density_matrix(rho_d)]
    t = overlap_operators()
    means = np.array([[np.trace(rho@matrix).real for matrix in t] for rho in rhos])
    variances = np.array([[np.trace(rho@matrix@matrix).real for matrix in t] for rho in rhos]) - means**2
    if np.any(variances < -TOLERANCE):
        raise ValueError("numerically unresolved negative density variance")
    correction = float(np.sum(sw[:, HARMONIC_DEGREES]*variances))
    state = embed_one_plus_one(np.kron(*rhos))
    quantum = np.trace(state@hamiltonian(p)).real
    classical = effective_energy(*rhos, parameters=p)
    residual = float(quantum-classical+correction)
    _finite([correction, quantum, classical, residual], "variance comparison")
    return {"classical_energy": float(classical), "quantum_energy": float(quantum),
            "weighted_variance_correction": correction, "identity_residual": residual,
            "variances": variances, "cross_factorization": "product states only"}


def matching_ledger():
    """Missing matching data are explicit; toy assumptions do not fill the gaps."""
    rows = (
        ("projection", "Conditional q=3 sphere profiles; classical fixed-norm ring dynamics",
         "Substrate-to-triplet projection/isometry and validity scale",
         "Three internal modes per species; q is not substrate p or charge Q"),
        ("statistics_and_source", "Bosonic truncated substrate examples; no fermion map",
         "Statistics and physical source bilinear with gauge/Lorentz quantum numbers",
         "Six CAR modes with number-density bilinears, not SM or Yukawa fields"),
        ("population", "Classical fixed trace and charge-mod-p bookkeeping",
         "Number sector, preparation or independently justified chemical potential",
         "All sixteen conserved number sectors; 1+1 is a conditional diagnostic"),
        ("temporal_kernel", "Previous scalar action is static",
         "Scalar temporal kernel and controlled instantaneous elimination regime",
         "NEW equal-time instantaneous ansatz; not exact dynamical mediator elimination"),
        ("couplings", "Previous conditional response weights and sign assumptions",
         "Independently matched coupling values and signs",
         "Inherited positive g and eta>=0; frozen toy inputs only"),
        ("ordering", "No microscopic ordering/renormalization prescription supplied",
         "Operator ordering, renormalization and compensating one-body matter terms",
         "Full squares and normal ordered same-sector control are different uncompensated models"),
    )
    return [{"id": key, "existing_evidence": evidence, "missing_input": missing,
             "toy_assumption": toy, "status": "missing microscopic matching"}
            for key, evidence, missing, toy in rows]


def _real_list(array):
    array = np.asarray(array)
    if np.iscomplexobj(array) and np.max(abs(array.imag)) > TOLERANCE:
        raise ValueError("complex matrix requires explicit serialization")
    return _finite(array.real, "report array").tolist()


def demonstration():
    """Frozen JSON-safe toy report, no physical masses, fits, or population claim."""
    p = SelectionParameters()
    orderings = {}
    for ordering in ORDERINGS:
        table = [sector_spectrum(nu, nd, p, ordering) for nu in range(4) for nd in range(4)]
        lowest = min(row["energy"] for row in table)
        tolerance = TOLERANCE*max(abs(row["energy"]) for row in table)
        coefficients = _scales(p)[1]@CASIMIR_COEFFICIENTS
        orderings[ordering] = {"counterterm_coefficient": _real_list(coefficients),
                               "sectors": table,
                               "lowest_sectors": [row["numbers"] for row in table
                                                  if abs(row["energy"]-lowest) <= tolerance]}
    solutions = {}
    for label, params, ordering in (("full_square", p, "full_square"),
                                    ("normal_ordered", p, "normal_ordered"),
                                    ("eta_zero_control", SelectionParameters(eta=0), "full_square")):
        solution = one_plus_one_solution(params, ordering)
        solutions[label] = {key: (_real_list(value) if isinstance(value, np.ndarray) else value)
                            for key, value in solution.items()
                            if key not in ("ground_projector", "ground_ensemble")}
    reps = ground_representatives(p)
    solutions["representatives"] = {name: {key: (_real_list(value) if isinstance(value, np.ndarray) else value)
                                         for key, value in rep.items() if key != "state"} for name, rep in reps.items()}
    filled = np.zeros(64)
    filled[63] = 1
    filled_occupation = one_body_occupation(filled, "u")
    q = _densities()
    commutator = q[0, 1]@q[0, 2]-q[0, 2]@q[0, 1]
    variance = product_variance_comparison(coherent_occupation(), coherent_occupation(), p)
    variance["variances"] = _real_list(variance["variances"])
    return {"model_id": MODEL_ID, "status": "conditional operator consistency diagnostic; microscopic origin not derived",
            "parameters": asdict(p), "assumptions": matching_ledger(),
            "operator_algebra": {"modes": 6, "dimension": 64,
                                 "basis": "bit index sum 2**i n_i; u[0:3], d[3:6]; m=(1,0,-1)",
                                 "density_commutator_norm": float(np.linalg.norm(commutator, 2)),
                                 "degree_casimirs": _real_list(CASIMIR_COEFFICIENTS)},
            "orderings": orderings, "one_plus_one": solutions,
            "filled_sector": {"numbers": [3, 3], "occupation": _real_list(filled_occupation["occupation"]),
                              "normalized_occupation": _real_list(filled_occupation["normalized_occupation"]),
                              "density_harmonics": _real_list(density_expectation(filled, "u", p.R)),
                              "constant_density_per_species": 3/(4*np.pi*p.R**2),
                              "full_square_energy_oracle": -12-3/np.pi,
                              "interpretation": "three occupied internal states, not three physical families"},
            "product_variance": variance,
            "conclusions": ["NEW instantaneous operator ansatz, not microscopic derivation or exact scalar elimination",
                            "eta>0: rank-five internal J=2 ground space, not a unique anisotropic source",
                            "eta=0: whole nine-dimensional 1+1 block is degenerate",
                            "Conserved numbers do not dynamically choose population; all-filled lowest only in this frozen table",
                            "Rotation-invariant ensemble is not the unique pure quantum vacuum",
                            "No mean-density scalar response or physical masses/mixing are inferred"],
            "physical_masses": None, "physical_mixing": None}
