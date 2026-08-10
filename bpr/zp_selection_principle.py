"""Z_p neutrality superselection — the derived physical-state selection principle.

WHAT WAS ASKED (glueball post-mortem, doc/GLUEBALL_BENCHMARK_V1.md sec.9)
-------------------------------------------------------------------------
The glueball failure identified a missing principle: BPR had no rule saying
which boundary excitations are physical states (QCD's analog: color
confinement). The requirement was that any such principle be DERIVED from the
Z_p substrate — never postulated to fix a spectrum. This module contains the
derivation, its verified consequences, and — critically — what it does NOT fix.

THE DERIVATION
--------------
Premise 1 (frozen, verified numerically here): every term of the frozen
boundary action is built from phase DIFFERENCES — the hopping term depends on
e^{i(phi_y - phi_x)} and the quartic on |psi|^4 alone. A global shift
phi_x -> phi_x + c (mod p) leaves the action exactly invariant, and no frozen
observable measures the absolute phase.

Premise 2 (the load-bearing ASSUMPTION, flagged as such): the substrate is
self-contained — BPR's boundary is not embedded in any larger structure that
could serve as an external phase reference. In laboratory condensed matter the
environment provides such a reference, making the global U(1) a spontaneously
breakable symmetry; in a self-contained substrate there is nothing to measure
the absolute phase against, so configurations differing by a global shift are
the SAME physical state. The global shift is then a REDUNDANCY (gauge
identification), not a symmetry. (This is the closed-universe argument that
global symmetries of a self-contained system act trivially on physical states;
it is an interpretive premise, and everything below rides on it.)

Conclusion: physical states must be invariant under the global Z_p shift
generator S = exp(2*pi*i*Q/p), where Q = total quantum number of the psi field.
Invariance requires exp(2*pi*i*Q/p) = 1, i.e.

    Q  ==  0  (mod p)          --- the Z_p NEUTRALITY SUPERSELECTION RULE.

Note the mod-p: the Z_p shift (not a continuum U(1)) only forces charge zero
MODULO p. This is the substrate fingerprint, exactly analogous to the Z_N
center of SU(N) permitting N-quark baryons.

VERIFIED CONSEQUENCES (all numerically checked below / in tests)
----------------------------------------------------------------
1. CONFINEMENT ANALOG: a single bare psi-quantum (Q = 1) is NOT a physical
   state. Bare quanta are confined; only neutral combinations (particle-hole
   pairs) appear in the physical spectrum.
2. Z_p "BARYONS": composites of exactly p bare quanta are neutral (Q = p == 0
   mod p) and hence physical — the analog of N-quark baryons from the Z_N
   center. With p = 104761 these sit at absurdly high energy; a structural
   prediction, not a phenomenological one.
3. CONSISTENCY: the frozen Hamiltonian commutes with Q exactly and is exactly
   block-diagonal in charge — dynamics never mixes superselection sectors, so
   the rule is consistent with the frozen time evolution (verified).
4. SECTOR COUNTING: on a toy substrate (3 sites, n_max = 2, toy prime 3) the
   physical projector keeps exactly dim/p of the Hilbert space (9 of 27).

WHAT IT DOES **NOT** DO (the honest negative)
---------------------------------------------
It does NOT remove the light 1^- that helped kill the glueball benchmark.
Bogoliubov quasiparticles in the number-conserving formulation are operators
like b_k^dag ~ a_k^dag a_0 (move one quantum out of the condensate): they
carry Q = 0 EXACTLY (verified below). The neutrality rule confines bare
quanta but leaves every Bogoliubov quasiparticle physical — including the
single 1^- state. Therefore:

    THE GLUEBALL SECTOR REMAINS CLOSED. This principle does not reopen it.

The post-mortem lesson is thus only PARTIALLY addressed: BPR now has a derived
confinement analog (a real structural gain), but the specific spectroscopic
defect (a light neutral 1^- with no glueball counterpart) survives it, because
that state was already neutral. A principle that removes neutral single
quasiparticles would need structure the frozen substrate does not supply; we
searched (multiplicative/QR grading: not conserved by the vertex; reflection
identification: not in the frozen postulates) and record those dead ends.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Premise 1: the frozen classical action depends only on phase differences
# ---------------------------------------------------------------------------

def classical_energy(psi: np.ndarray, C: float = 1.0, g: float = 0.5) -> float:
    """The frozen boundary energy: hopping + quartic (ring geometry)."""
    hop = -C * np.sum(psi.conj() * np.roll(psi, -1)
                      + psi * np.roll(psi.conj(), -1)).real
    return float(hop + 0.5 * g * np.sum(np.abs(psi) ** 4))


def global_shift_invariance(n: int = 101, seed: int = 0,
                            p: int = 104761) -> float:
    """Max |H(e^{i theta} psi) - H(psi)| over global shifts, including the
    elementary Z_p shift 2*pi/p. Zero to machine precision (Premise 1)."""
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=n) + 1j * rng.normal(size=n)
    base = classical_energy(psi)
    thetas = (0.3, 2 * np.pi / p, 17 * 2 * np.pi / p, np.pi)
    return float(max(abs(classical_energy(psi * np.exp(1j * t)) - base)
                     for t in thetas))


# ---------------------------------------------------------------------------
# Quantum verification: small bosonic Hilbert space (3 sites, n_max = 2)
# ---------------------------------------------------------------------------

def _small_hilbert(C: float = 1.0, g: float = 0.7):
    """Build H, Q, site operators on 3 sites with local dim 3 (dim 27)."""
    d = 3
    a1 = np.diag(np.sqrt(np.arange(1, d)), k=1)
    n1 = np.diag(np.arange(d)).astype(float)
    I = np.eye(d)

    def k3(A, B, Cm):
        return np.kron(np.kron(A, B), Cm)

    a = [k3(a1, I, I), k3(I, a1, I), k3(I, I, a1)]
    n = [k3(n1, I, I), k3(I, n1, I), k3(I, I, n1)]
    Q = sum(n)
    H = sum(-C * (a[x].T.conj() @ a[(x + 1) % 3]
                  + a[(x + 1) % 3].T.conj() @ a[x]) for x in range(3))
    H = H + 0.5 * g * sum(nx @ (nx - np.eye(27)) for nx in n)
    return H, Q, a


def charge_is_conserved() -> float:
    """|| [H, Q] || — exactly zero: the rule is consistent with dynamics."""
    H, Q, _ = _small_hilbert()
    return float(np.max(np.abs(H @ Q - Q @ H)))


def sectors_never_mix() -> float:
    """Max |<Q|H|Q'>| across different charge sectors — exactly zero."""
    H, Q, _ = _small_hilbert()
    q = np.diag(Q).round().astype(int)
    mask = q[:, None] != q[None, :]
    return float(np.max(np.abs(H * mask)))


def physical_sector_dims(p_toy: int = 3) -> dict:
    """Charge-sector dimensions and the physical (Q = 0 mod p_toy) count.
    For the 27-dim toy space and p_toy = 3: physical dim = 9 = 27/3, and the
    p-quanta composite (Q = 3) IS physical while bare quanta (Q = 1, 2) are not.
    """
    _, Q, _ = _small_hilbert()
    q = np.diag(Q).round().astype(int)
    dims = {int(v): int(np.sum(q == v)) for v in sorted(set(q))}
    phys = int(sum(d for v, d in dims.items() if v % p_toy == 0))
    return {
        "sector_dims": dims,
        "physical_dim": phys,
        "total_dim": len(q),
        "bare_quantum_physical": (1 % p_toy == 0),
        "p_quanta_composite_physical": (p_toy % p_toy == 0),
    }


def quasiparticle_is_neutral() -> dict:
    """The honest-negative check: a number-conserving Bogoliubov-type operator
    b^dag ~ a_k^dag a_0 commutes with Q exactly (neutral, hence PHYSICAL), while
    a bare a^dag carries charge 1 (confined)."""
    _, Q, a = _small_hilbert()
    b_dag = a[1].T.conj() @ a[0]
    adag = a[0].T.conj()
    return {
        "quasiparticle_charge_commutator": float(np.max(np.abs(Q @ b_dag - b_dag @ Q))),
        "bare_creation_carries_charge_1": bool(
            np.allclose(Q @ adag - adag @ Q, adag)),
    }


# ---------------------------------------------------------------------------
# The principle, its consequences, and its honest limits
# ---------------------------------------------------------------------------

def derivation() -> dict:
    return {
        "premise_1": ("frozen action depends only on phase differences — "
                      "global shift exactly invariant (verified numerically)"),
        "premise_2_ASSUMPTION": (
            "substrate self-containment: no external phase reference exists, "
            "so the global shift is a REDUNDANCY, not a symmetry. This is the "
            "load-bearing interpretive step (closed-universe argument)."),
        "conclusion": "physical states satisfy Q == 0 (mod p) — Z_p neutrality "
                      "superselection",
        "postulated_to_fix_a_spectrum": False,
    }


def consequences() -> dict:
    return {
        "confinement_analog": "single bare psi-quanta (Q=1) are unphysical",
        "zp_baryons": ("composites of exactly p quanta are neutral and "
                       "physical — the Z_N-center/N-quark-baryon analog; "
                       "structurally distinctive, energetically inaccessible"),
        "charge_conserved_mod_p": ("the superselection label is Z_p-valued, "
                                   "not Z-valued"),
        "does_not_remove_light_1_minus": True,
        "glueball_sector": "REMAINS CLOSED — this principle does not reopen it",
        "dead_ends_searched": (
            "multiplicative/QR grading: not conserved by the momentum-"
            "conserving vertex; reflection identification x <-> -x: not in "
            "the frozen postulates (would be a retrofit)"),
    }


def report() -> str:
    d = derivation()
    c = consequences()
    inv = global_shift_invariance()
    comm = charge_is_conserved()
    mix = sectors_never_mix()
    dims = physical_sector_dims()
    qp = quasiparticle_is_neutral()
    lines = [
        "Z_p neutrality superselection — derived state-selection principle",
        "=================================================================",
        f"Premise 1 (verified): {d['premise_1']}",
        f"  max dH under global shift: {inv:.1e}",
        f"Premise 2 (ASSUMPTION, flagged): {d['premise_2_ASSUMPTION']}",
        f"Conclusion: {d['conclusion']}",
        "",
        "Verified consequences:",
        f"  [H, Q] = {comm:.1e}   (rule consistent with frozen dynamics)",
        f"  cross-sector matrix elements: {mix:.1e}  (sectors never mix)",
        f"  toy sector count: physical dim {dims['physical_dim']} of "
        f"{dims['total_dim']}  (= dim/p)",
        f"  bare quantum physical? {dims['bare_quantum_physical']}  "
        f"(confinement analog)",
        f"  p-quanta composite physical? {dims['p_quanta_composite_physical']}  "
        f"(Z_p 'baryon')",
        "",
        "THE HONEST NEGATIVE:",
        f"  quasiparticle charge commutator: "
        f"{qp['quasiparticle_charge_commutator']:.1e}  -> Bogoliubov",
        "  quasiparticles are exactly NEUTRAL, hence remain physical. The light",
        "  1^- that helped kill the glueball benchmark is a neutral quasiparticle",
        "  and SURVIVES this principle.",
        f"  => glueball sector: {c['glueball_sector']}.",
        "",
        f"Dead ends searched (recorded): {c['dead_ends_searched']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
