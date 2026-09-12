"""Historical Path B Milestones 2-4 arithmetic and corrected provenance.

The original electric operator Delta_G = 3I - L_r - L_r_inverse - L_s is
noncentral: {r, r^-1, s} is not conjugacy invariant. It does not commute with
both endpoint gauge actions, so the historical H(lambda) is not a valid
Hamiltonian for the claimed gauge-invariant theory. The function names and
numerical formulas below are retained for reproducibility, not endorsement.

``electric_casimir`` returns normalized irrep trace averages of Delta_G, not
its scalar irrep eigenvalues. In E_k the original block has eigenvalues
2 - 2cos(2*pi*k/n) and 4 - 2cos(2*pi*k/n), whose average is the stored value.
The averages become scalar eigenvalues only for the NEW conjugacy-averaged
operator implemented in ``gauge_heat_kernel``. That repair is a distinct
model, not a retroactive validation of the historical calculations.

M3 checks a kinematic partition of an unchanged ring hopping spectrum by
rotation-charge labels. It does not add gauge links, impose all Gauss
constraints, or solve an interacting flavor Hamiltonian. Unchanged formula
inputs and mode-label arithmetic do not prove dynamical flavor survival.

The historical 4*eps_min/lambda expression is retained as arithmetic, not
an established glueball mass or proof of J^PC degeneracy. No spatial-channel
spectrum or substrate coupling has been derived here. The character-Wilson
simulations in ``gauge_phase_mc`` and ``gauge_mc_fast`` are another model;
neither equality to this H nor a beta/lambda mapping is established.

Benchmark v3 remains SEALED with its inherited targets untouched. No target
number appears here; the source-blindness regressions remain in force.
"""
from __future__ import annotations

import numpy as np

from .nonabelian_gauge_sector import ALLOWED_CLASSES, character_table

# The frozen quark-sector mode integers (bpr/qcd_flavor.py).
FLAVOR_INTEGERS = {"d": 1, "s": 4, "b": 30, "u": 1, "c": 24, "t": 283}


# ---------------------------------------------------------------------------
# M3 — flavor survival
# ---------------------------------------------------------------------------

def dn_rep_of_mode(l: int, n: int) -> str:
    """Kinematic D_n label from l mod n, with reflections pairing +-r.

    A/B denote rotation-charge families, not resolved reflection parities.
    This assignment alone does not establish a physical gauged excitation.
    """
    r = l % n
    if r == 0:
        return "A"                       # rotation-invariant
    if n % 2 == 0 and r == n // 2:
        return "B"
    return f"E{min(r, n - r)}"


def flavor_rep_table(n: int) -> dict:
    return {q: dn_rep_of_mode(l, n) for q, l in FLAVOR_INTEGERS.items()}


def spectrum_partition_check(n: int, N: int = 360) -> bool:
    """Kinematic ring identity, not a dynamical gauging calculation.

    For N divisible by n, the rotation-charge subsets k = q mod n partition
    the fixed hopping spectrum {2cos(2*pi*k/N)}. No Hamiltonian is changed.
    """
    assert N % n == 0
    ks = np.arange(N)
    full = np.sort(2 * np.cos(2 * np.pi * ks / N))
    union = np.sort(np.concatenate(
        [2 * np.cos(2 * np.pi * ks[ks % n == q] / N) for q in range(n)]))
    return bool(np.allclose(union, full))


def m3_report() -> dict:
    return {
        "mass_formula_inputs": ("mode integers l_i", "J", "p", "z", "n_gen"),
        "formula_inputs_changed_here": False,
        "any_input_gauged": None,
        "energies_shift_under_gauging": None,
        "dynamical_flavor_survival": "UNKNOWN — no interacting gauged calculation",
        "partition_verified": {n: spectrum_partition_check(n)
                               for n in ALLOWED_CLASSES},
        "flavor_reps": {n: flavor_rep_table(n) for n in ALLOWED_CLASSES},
        "lhcb_predictions_changed": None,
        "legacy_formula_values_changed_here": False,
        "caveats": (
            "lepton labels include sqrt(210): 'l mod n' undefined — open wrinkle",
            "gauge interactions and binding energies have not been computed",
            "A/B labels do not resolve reflection parity or Gauss constraints",
        ),
        "verdict": "KINEMATIC PARTITION VERIFIED — dynamical flavor survival UNKNOWN",
    }


# ---------------------------------------------------------------------------
# M2 — the frozen dynamics
# ---------------------------------------------------------------------------

GENERATING_SET_SIZE = 3      # S = {r, r^-1, s}


def frozen_hamiltonian_spec() -> dict:
    """Historical specification, invalid as the claimed gauge Hamiltonian."""
    return {
        "model_id": "legacy-noncentral-generator-hamiltonian-v0.1",
        "status": "INVALID as claimed gauge-invariant Hamiltonian — noncentral Delta_G",
        "electric_operator_central": False,
        "endpoint_gauge_invariant": False,
        "electric_casimir_interpretation": "normalized irrep trace averages",
        "central_replacement": "NEW model in bpr.gauge_heat_kernel",
        "wilson_equivalence": "NOT ESTABLISHED — character-Wilson transfer differs",
        "beta_lambda_mapping": "NOT ESTABLISHED",
        "form": "H(lambda) = (1/lambda) sum_links Delta_G "
                "+ lambda sum_plaq (1 - Re chi_F/d_F)",
        "gauge_group": "D_n, n in " + str(ALLOWED_CLASSES),
        "generating_set": "{r, r^-1, s} (symmetric)",
        "n_couplings": 1,
        "limits": {
            "lambda_large": "NOT ESTABLISHED for the claimed gauge theory",
            "lambda_small": "trace-average arithmetic only; no physical spectrum",
        },
        "phase_location": "OPEN — requires a valid model and physical matching",
    }


def electric_casimir(irrep: str, n: int) -> float:
    """Historical normalized irrep trace average, despite the legacy name.

    eps(R) = |S| - sum_{s in S} chi_R(s)/d_R for S = {r, r^-1, s}.
    Noncentral Delta_G is not scalar on E_k. These same values are scalar
    eigenvalues of the NEW conjugacy-averaged operator, not the original.
    """
    T = character_table(n)
    chi = T[irrep]
    d = chi[(0, 1)]
    return float(GENERATING_SET_SIZE - (2.0 * chi[(1, 1)] + chi[(0, -1)]) / d)


def casimir_table(n: int) -> dict:
    return {name: electric_casimir(name, n) for name in character_table(n)}


def lightest_charge_sector(n: int) -> tuple:
    """Smallest positive historical trace average, not original-block minimum."""
    tab = {k: v for k, v in casimir_table(n).items() if v > 1e-9}
    name = min(tab, key=tab.get)
    return name, tab[name]


def strong_coupling_leading_order(n: int, lam: float = 1.0) -> dict:
    """Retain historical 4*eps_min/lam arithmetic, not a physical mass.

    The legacy mass key is preserved for compatibility. Neither physical
    spatial J^PC channels nor their degeneracy are established here.
    """
    name, eps_min = lightest_charge_sector(n)
    return {
        "lightest_sector": name,
        "eps_min": eps_min,
        "glueball_LO_mass_units_invlam": 4.0 * eps_min / lam,
        "jpc_split_at_this_order": None,
        "interpretation": "historical trace-average arithmetic; not a glueball mass",
        "physical_jpc_status": "UNKNOWN — no spatial-channel calculation",
    }


# ---------------------------------------------------------------------------
# M4 — benchmark v3 status
# ---------------------------------------------------------------------------

def benchmark_v3_status() -> dict:
    return {
        "status": "SEALED",
        "reason": ("original Delta_G is noncentral; historical trace-average "
                   "arithmetic does not establish a J^PC spectrum. A valid "
                   "gauge Hamiltonian, controlled spatial-channel spectrum "
                   "and physical matching remain required; existing Wilson "
                   "MC is a distinct model"),
        "targets": "inherited unchanged from Benchmark v1 (sealed there)",
        "numbers_invented_here": False,
    }


def report() -> str:
    m3 = m3_report()
    spec = frozen_hamiltonian_spec()
    v3 = benchmark_v3_status()
    lines = [
        "Path B Milestones 2-4 — historical arithmetic and corrected provenance",
        "===================================================================",
        "",
        f"M3 (kill condition): {m3['verdict']}",
        f"  partition verified for n in {ALLOWED_CLASSES}: "
        f"{all(m3['partition_verified'].values())}",
        "  kinematic D_n label families (l mod n; A/B parity unresolved):",
    ]
    for n in ALLOWED_CLASSES:
        reps = m3["flavor_reps"][n]
        lines.append("    n=%2d: " % n +
                     "  ".join(f"{q}:{r}" for q, r in reps.items()))
    lines += ["  caveats: " + " | ".join(m3["caveats"]), "",
              f"M2: historical form: {spec['form']}",
              f"  status: {spec['status']}",
              f"  phase location: {spec['phase_location']}",
              "  normalized irrep trace averages eps(R), not original eigenvalues:"]
    for n in ALLOWED_CLASSES:
        tab = casimir_table(n)
        name, eps = lightest_charge_sector(n)
        lines.append("    n=%2d: " % n +
                     ", ".join(f"{k}={v:.3f}" for k, v in tab.items()) +
                     f"   smallest positive trace average: {name}")
    lines += [
        "",
        f"M4: Benchmark v3 {v3['status']} — {v3['reason']}",
        "",
        "No spectrum was invented. Dynamical flavor survival remains UNKNOWN.",
        "The original noncentral electric operator fails endpoint gauge invariance.",
        "Its central repair is a NEW model; Wilson MC is distinct from both.",
        "No beta/lambda mapping or physical spatial-channel spectrum is established.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
