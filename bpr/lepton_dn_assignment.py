"""Path B M3 wrinkle: D_n charge assignment for the charged-lepton labels.

THE WRINKLE (recorded in gauge_dynamics_m2_m3.m3_report): the quark-sector
labels are integers, so their D_n gauge charge is fixed by l mod n. The lepton
labels are (l_e, l_mu, l_tau) = (1, sqrt(210), 59) — and "sqrt(210) mod n" is
undefined. The lepton masses are untouched by gauging either way (the mass
formula consumes only l^2, which is integer); the open question was purely the
CHARGE BOOKKEEPING.

THE PROPOSED RESOLUTION (and why it is not new physics)
-------------------------------------------------------
The l_mu derivation itself supplies the missing structure. l_mu = sqrt(z(z^2-1))
is the geometric mean of consecutive coordination shells — and the SAME
derivation was originally written as sqrt(14 x 15) (charged_leptons.py notes),
with z(z^2-1) = 210 = 14 x 15 exactly. A geometric-mean label denotes a
TWO-MODE COMPOSITE with integer constituents (14, 15). Composites carry the
TENSOR PRODUCT of their constituents' charges — standard representation
theory, no new postulate:

    charge(mu) = E_{14 mod n} (x) E_{15 mod n}   (decomposed via D_n fusion)

Fusion rules used (all standard dihedral):
    E_a (x) E_b   = E_{a+b} + E_{a-b}      (indices folded back into range;
                                            E_0 -> A1 + A2; E_{n/2} -> B1 + B2)
    E_a (x) A     = E_a,   E_a (x) B = E_{n/2 - a}
    A/B products: multiply signs.

VERDICTS BY CLASS (computed below, exactly):
    Determinate  — the tensor product is a single irrep: the muon has a
                   unique D_n charge and the wrinkle CLOSES for that n.
    Ambiguous    — the product decomposes into several irreps: the composite
                   picture alone does not pick one; recorded as remaining open
                   (selecting a component would require a dynamical input we
                   do not have).

WHAT IS PROVED REGARDLESS OF VERDICT
------------------------------------
* Lepton masses consume l^2 only; every candidate assignment is mass-inert.
* The M3 partition check is charge-blind, so M3's PASS is unaffected.

BLINDNESS: no glueball number appears here (grep-enforced in tests).
"""
from __future__ import annotations

import numpy as np

from .nonabelian_gauge_sector import ALLOWED_CLASSES

# The lepton labels as derived (charged_leptons.py): l_mu^2 = z(z^2-1) = 210.
LEPTON_LABELS = {"e": 1.0, "mu": float(np.sqrt(210.0)), "tau": 59.0}

# The integer factorization the l_mu derivation itself uses: 210 = 14 x 15
# (geometric mean of shells; sqrt(14 x 15) in the original derivation note).
MU_CONSTITUENTS = (14, 15)


# ---------------------------------------------------------------------------
# D_n irrep labels and fusion
# ---------------------------------------------------------------------------


def irrep_of_rotation_charge(r: int, n: int) -> str:
    """Single-mode D_n assignment (same convention as dn_rep_of_mode)."""
    r = r % n
    if r == 0:
        return "A"
    if n % 2 == 0 and r == n // 2:
        return "B"
    return f"E{min(r, n - r)}"


def fuse_e_e(a: int, b: int, n: int) -> list:
    """E_a (x) E_b decomposition in D_n (a, b are folded rotation charges)."""
    out = []
    for s in (a + b, a - b):
        s = s % n
        if s == 0:
            out.extend(["A1", "A2"])
        elif n % 2 == 0 and s == n // 2:
            out.extend(["B1", "B2"])
        else:
            out.append(f"E{min(s, n - s)}")
    # merge the two branches' duplicates (E_{a+b} = E_{a-b} can coincide)
    return sorted(set(out))


def composite_charge(constituents: tuple, n: int) -> dict:
    """Tensor-product charge of an integer-constituent composite in D_n."""
    reps = [irrep_of_rotation_charge(c, n) for c in constituents]
    r1, r2 = (constituents[0] % n), (constituents[1] % n)
    kind = tuple(sorted(r.rstrip("0123456789") if r[0] == "E" else r
                        for r in reps))
    if reps[0][0] == "E" and reps[1][0] == "E":
        a = min(r1, n - r1)
        b = min(r2, n - r2)
        comps = fuse_e_e(a, b, n)
    elif "A" in (reps[0][0], reps[1][0]):
        comps = [reps[0] if reps[1][0] == "A" else reps[1]]
    else:                                    # E (x) B or B (x) B (n even only)
        e = r1 if reps[0][0] == "E" else r2
        if reps[0][0] == "B" and reps[1][0] == "B":
            comps = ["A"]
        else:
            a = min(e, n - e)
            s = n // 2 - a
            comps = ["B"] if s == 0 else [f"E{min(s, n - s)}"]
    return {
        "constituent_reps": reps,
        "components": comps,
        "determinate": len(comps) == 1,
        "kind": kind,
    }


# ---------------------------------------------------------------------------
# The lepton table
# ---------------------------------------------------------------------------


def lepton_assignment(n: int) -> dict:
    """D_n charges for (e, mu, tau) under the composite resolution."""
    e = irrep_of_rotation_charge(1, n)
    tau = irrep_of_rotation_charge(59, n)
    mu = composite_charge(MU_CONSTITUENTS, n)
    return {
        "e": {"charge": e, "determinate": True, "basis": "integer l = 1"},
        "tau": {"charge": tau, "determinate": True, "basis": "integer l = 59"},
        "mu": {
            "charge": (mu["components"][0] if mu["determinate"]
                       else " + ".join(mu["components"])),
            "determinate": mu["determinate"],
            "basis": f"composite (14, 15): "
                     f"{mu['constituent_reps'][0]} x {mu['constituent_reps'][1]}",
            "components": mu["components"],
        },
    }


def masses_inert_check() -> bool:
    """The mass formula consumes l^2 only — integers for all three leptons —
    so every candidate charge assignment leaves the masses untouched."""
    l2 = {k: v * v for k, v in LEPTON_LABELS.items()}
    return all(abs(x - round(x)) < 1e-9 for x in l2.values())


def wrinkle_status() -> dict:
    per_class = {n: lepton_assignment(n) for n in ALLOWED_CLASSES}
    closed = [n for n in ALLOWED_CLASSES if per_class[n]["mu"]["determinate"]]
    open_ = [n for n in ALLOWED_CLASSES if not per_class[n]["mu"]["determinate"]]
    return {
        "resolution": "composite tensor-product charge from the derivation's "
                      "own factorization 210 = 14 x 15",
        "new_postulates": 0,
        "masses_affected": False,
        "m3_pass_affected": False,
        "assignments": per_class,
        "closed_for_n": closed,
        "still_ambiguous_for_n": open_,
        "verdict": ("CLOSED for n in %s; reduced to a component ambiguity "
                    "for n in %s (selection would need dynamical input)"
                    % (closed, open_)) if open_ else
                   f"CLOSED for all allowed classes {list(ALLOWED_CLASSES)}",
    }


def report() -> str:  # pragma: no cover
    st = wrinkle_status()
    lines = ["Lepton D_n assignment — the sqrt(210) wrinkle",
             "=" * 55,
             f"resolution: {st['resolution']}",
             f"masses affected: {st['masses_affected']}   "
             f"M3 PASS affected: {st['m3_pass_affected']}", ""]
    for n, tab in st["assignments"].items():
        lines.append(f"n={n:>2}: " + "  ".join(
            f"{lep}:{d['charge']}{'' if d['determinate'] else ' (ambiguous)'}"
            for lep, d in tab.items()))
    lines += ["", f"verdict: {st['verdict']}"]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
