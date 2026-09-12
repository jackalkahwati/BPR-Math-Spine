"""BPR 2.0 Path B, Milestone 1 — gauging the frozen point group (Postulate 0d, PROPOSED).

WHY THIS EXISTS (and why it is not a retrofit)
----------------------------------------------
The glueball benchmark closed with a structural diagnosis: BPR's boundary field
content is scalar and Abelian, and such content provably cannot produce the
parity-odd (0^-) sector or non-Abelian gauge-boson matter that reality contains.
The principled response is NOT to invent new fields for that sector, but to ask
whether the FROZEN theory already contains unused non-Abelian structure. It
does: Postulate 0c's quasicrystal classes n in {5, 8, 9, 12} carry dihedral
point groups D_n — non-Abelian discrete groups that the Abelian phase field
never used. Path B proposes to GAUGE that symmetry (Postulate 0d, version 0.1,
status PROPOSED — not merged into the frozen core).

Blindness discipline: no glueball mass, ratio, or experimental number appears
anywhere in this module. The motivation is the structural absence of parity-odd
content, not any measured value. The eventual spectrum faces a NEW sealed
benchmark (v3) under the same rules as v1.

WHAT MILESTONE 1 DELIVERS (all exactly computable, all verified here)
---------------------------------------------------------------------
At the quantum-double topological point, finite-group gauge theory has anyon
labels (conjugacy class C, irrep R of the centralizer of C), with quantum
dimension d = |C| * dim(R). This is conditional kinematics, not a determination
of the substrate's phase or physical particle content. Internal D_n reflection
charge is not by itself spatial parity.

1. INTERNAL SIGN CHARGE EXISTS. Every D_n has the 1-dim irrep A2:
   rotations -> +1, reflections -> -1. Its historical "pseudoscalar" name
   does not establish a spatial 0^- state after gauging.
2. INTERNAL DOUBLETS EXIST. The E_k 2-dim irreps carry rotation eigenvalues
   e^{+-2 pi i k/n}, up to k_max = (n-1)/2 (n odd) or n/2 - 1 (n even).
   Their identification with physical spatial angular momentum is unproved.
3. NON-ABELIAN CONTENT DOMINATES — the quantum double D(D_n) has mostly
   anyons with quantum dimension d > 1 (non-Abelian fusion/braiding): e.g.
   14 of 16 types for D_5. The sum rule sum(d^2) = |G|^2 is verified exactly.

WHAT MILESTONE 1 DOES **NOT** DELIVER (the honest gap list)
-----------------------------------------------------------
* NO PHYSICAL MASS SPECTRUM. The quantum-double labels do not fix propagating
  particle masses. Its topological ground-state degeneracy is not a claim
  that all excitation energies coincide. The original M2 electric operator
  is noncentral and invalid as the claimed gauge-invariant Hamiltonian;
  gauge_heat_kernel supplies a NEW central model, not a physical completion.
* FLAVOR SECTOR COMPATIBILITY UNPROVEN. The flavor integers came from boundary
  mode counting in the Abelian sector; whether they survive gauging must be
  SHOWN, not assumed (Milestone 3).
* CONTINUUM QUANTUM NUMBERS NOT ESTABLISHED. Internal A2/E_k labels do not
  establish spatial P or J. That requires a physical spatial-channel
  construction and spectrum (part of the sealed Benchmark v3, Milestone 4).

ROADMAP (frozen now):
  M1 (this module)  gauge-sector kinematics from the frozen point group   DONE
  M2                freeze a dynamics away from the topological point     OPEN
  M3                re-derive / verify the flavor sector under gauging    OPEN
  M4                sealed Benchmark v3 (same lattice targets, blind)     OPEN
"""
from __future__ import annotations

import numpy as np

# The frozen Postulate 0c symmetry classes (phason_sector.ALLOWED_INTERNAL_CLASSES).
ALLOWED_CLASSES = (5, 8, 9, 12)


# ---------------------------------------------------------------------------
# Dihedral group D_n as (rotation index k mod n, reflection flag eps = +-1)
# ---------------------------------------------------------------------------

def mul(a, b, n: int):
    """(k1,+)(k2,e) = (k1+k2, e);  (k1,-)(k2,e) = (k1-k2, -e)."""
    k1, e1 = a
    k2, e2 = b
    return ((k1 + k2) % n, e2) if e1 == 1 else ((k1 - k2) % n, -e2)


def inv(a, n: int):
    k, e = a
    return ((-k) % n, 1) if e == 1 else a      # reflections are involutions


def elements(n: int) -> list:
    return [(k, e) for e in (1, -1) for k in range(n)]


def conjugacy_classes(n: int) -> list:
    """Brute-force conjugacy classes of D_n."""
    els = elements(n)
    seen, classes = set(), []
    for g in els:
        if g in seen:
            continue
        cl = {mul(mul(h, g, n), inv(h, n), n) for h in els}
        classes.append(sorted(cl))
        seen |= cl
    return classes


def character_table(n: int) -> dict:
    """Dihedral character table (irrep name -> {element: character}).

    Irreps: A1 (trivial), A2 (sign: rotations +1, reflections -1); for n even
    additionally B1, B2; and 2-dim E_k with chi(r^m) = 2cos(2 pi k m / n),
    chi(reflection) = 0. Verified below by exact orthogonality + Burnside.
    """
    els = elements(n)
    T = {"A1_trivial": {g: 1.0 for g in els},
         "A2_sign": {g: (1.0 if g[1] == 1 else -1.0) for g in els}}
    if n % 2 == 0:
        T["B1"] = {g: (-1.0) ** g[0] for g in els}
        T["B2"] = {g: ((-1.0) ** g[0] if g[1] == 1 else -((-1.0) ** g[0]))
                   for g in els}
    kmax = (n - 1) // 2 if n % 2 else n // 2 - 1
    for k in range(1, kmax + 1):
        T[f"E{k}"] = {g: (2 * np.cos(2 * np.pi * k * g[0] / n)
                          if g[1] == 1 else 0.0) for g in els}
    return T


def table_checks(n: int) -> dict:
    """Exact verification: orthogonality of characters and Burnside sum rule."""
    els = elements(n)
    T = character_table(n)
    names = list(T)
    M = np.array([[T[nm][g] for g in els] for nm in names])
    gram = M @ M.T / len(els)
    dims = [T[nm][(0, 1)] for nm in names]
    return {
        "n_irreps": len(names),
        "n_classes": len(conjugacy_classes(n)),
        "orthogonal": bool(np.allclose(gram, np.eye(len(names)), atol=1e-12)),
        "burnside": bool(abs(sum(d * d for d in dims) - len(els)) < 1e-9),
    }


# ---------------------------------------------------------------------------
# The two structural results: pseudoscalar charge and spin-like doublets
# ---------------------------------------------------------------------------

def pseudoscalar_charge(n: int) -> dict:
    """Historical name for the internal A2 sign irrep, not spatial parity.

    Check rotations -> +1 and reflections -> -1 from the character table.
    This does not construct a physical spatial 0^- state.
    """
    T = character_table(n)
    A2 = T["A2_sign"]
    rot_ok = all(abs(A2[(k, 1)] - 1.0) < 1e-12 for k in range(n))
    refl_odd = all(abs(A2[(k, -1)] + 1.0) < 1e-12 for k in range(n))
    return {"exists": rot_ok and refl_odd,
            "rotation_invariant": rot_ok,       # internal group action
            "reflection_odd": refl_odd}         # not a spatial-parity proof


def spin_doublets(n: int) -> dict:
    """Internal E_k rotation eigenvalues e^{+-2 pi i k/n}, not physical J."""
    kmax = (n - 1) // 2 if n % 2 else n // 2 - 1
    return {"n_doublets": kmax, "k_values": tuple(range(1, kmax + 1))}


# ---------------------------------------------------------------------------
# Quantum double D(D_n): the full anyon content of the gauged theory
# ---------------------------------------------------------------------------

def _centralizer(g, n: int) -> list:
    return [h for h in elements(n) if mul(h, g, n) == mul(g, h, n)]


def _irrep_dims(sub: list, n: int) -> list:
    """Irrep dimensions of a centralizer subgroup: the whole group uses the
    dihedral list; every proper centralizer in D_n is abelian (all 1-dim)."""
    if len(sub) == 2 * n:
        T = character_table(n)
        return [int(round(T[nm][(0, 1)])) for nm in T]
    assert all(mul(a, b, n) == mul(b, a, n) for a in sub for b in sub), \
        "unexpected non-abelian proper centralizer in a dihedral group"
    return [1] * len(sub)


def anyon_content(n: int) -> dict:
    """Anyons of D(D_n): (class, centralizer-irrep) pairs, quantum dimension
    d = |C| * dim(R). Verified against the exact sum rule sum d^2 = |G|^2."""
    G = 2 * n
    dims = []
    for cl in conjugacy_classes(n):
        Z = _centralizer(cl[0], n)
        for dR in _irrep_dims(Z, n):
            dims.append(len(cl) * dR)
    return {
        "n_anyon_types": len(dims),
        "quantum_dimensions": tuple(sorted(dims)),
        "sum_d_squared": int(sum(d * d for d in dims)),
        "group_order_squared": G * G,
        "sum_rule_ok": sum(d * d for d in dims) == G * G,
        "n_nonabelian": sum(1 for d in dims if d > 1),
    }


# ---------------------------------------------------------------------------
# Milestone bookkeeping (honesty locked)
# ---------------------------------------------------------------------------

def milestone_status() -> dict:
    return {
        "M1_gauge_sector_kinematics": "DONE (this module)",
        "M2_dynamics_beyond_topological_point":
            "ORIGINAL INVALID — noncentral Delta_G; gauge_heat_kernel is a NEW "
            "central model. Physical matching OPEN; Wilson MC is distinct",
        "M3_flavor_sector_compatibility":
            "KINEMATIC PARTITION VERIFIED — dynamical flavor survival UNKNOWN",
        "M4_sealed_benchmark_v3":
            "SEALED — controlled physical spatial-channel spectrum and matching "
            "remain OPEN; trace averages do not establish glueball masses",
        "spectrum_claims": "NONE physical — group labels and isolated-square toy "
                           "energies are not a spatial glueball spectrum",
        "postulate_0d_status": "PROPOSED v0.1 — not merged into the frozen core",
    }


def report() -> str:
    lines = [
        "BPR 2.0 Path B, Milestone 1 — gauged point-group kinematics",
        "===========================================================",
        "Gauge group: the FROZEN Postulate 0c point groups D_n, n in "
        f"{ALLOWED_CLASSES} (non-Abelian, already in the theory — not invented).",
        "",
        " n | |G| | classes | anyons | non-Abelian | A2 sign charge | E_k doublets",
    ]
    for n in ALLOWED_CLASSES:
        ch = table_checks(n)
        ps = pseudoscalar_charge(n)
        sd = spin_doublets(n)
        an = anyon_content(n)
        assert ch["orthogonal"] and ch["burnside"] and an["sum_rule_ok"]
        lines.append(
            f" {n:2d}| {2*n:3d} |   {ch['n_classes']:2d}    |  {an['n_anyon_types']:3d}"
            f"  |   {an['n_nonabelian']:3d}      |      {str(ps['exists']):5s}     "
            f"|  k = 1..{sd['n_doublets']}")
    ms = milestone_status()
    lines += [
        "",
        "STRUCTURAL RESULTS (derived, verified exactly):",
        "  1. Internal A2 sign charge exists: rotation-invariant, reflection-ODD.",
        "     Internal reflection charge does not establish spatial parity.",
        "  2. E_k doublets carry internal rotation labels +-k, not proven physical J.",
        "  3. Quantum-double anyon dimensions are predominantly non-Abelian",
        "     (sum rule verified); applicability to the substrate phase is unknown.",
        "",
        "HONEST LIMITS:",
        f"  spectrum: {ms['spectrum_claims']}",
        f"  M2: {ms['M2_dynamics_beyond_topological_point']}",
        f"  M3: {ms['M3_flavor_sector_compatibility']}",
        f"  M4: {ms['M4_sealed_benchmark_v3']}",
        f"  status: Postulate 0d {ms['postulate_0d_status']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
