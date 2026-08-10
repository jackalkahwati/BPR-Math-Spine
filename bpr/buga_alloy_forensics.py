"""Buga-sphere alloy forensics — what objective does the composition extremize?

SCOPE / HONESTY (read first)
----------------------------
This is a forensic test of the BUGA SPHERE's reported composition. It is NOT a
test of BPR. Two reasons, stated up front so the artifact can't be misread:

  1. BPR does not predict that this (or any) specific object IS its substrate.
     The Buga sphere is already FALSIFIED as a BPR artifact on visual grounds
     (doc/CLOSED_AND_DEPRECATED.md sec.2). So a metallurgy result here cannot
     confirm BPR.
  2. Even a POSITIVE quasicrystal result would be ordinary condensed matter
     (Shechtman's Nobel; Al-Cu-Fe / Al-Pd-Mn quasicrystals are commercial). A
     quasicrystalline alloy would not validate a theory of everything.

What this CAN do: reverse-engineer the objective function the composition sits
near, i.e. answer "what was it optimized for?" classically (no quantum computer
needed; this is cheap phase-field geometry, not DFT). The one BPR-adjacent
quantity is the quasicrystal-formation test (Hume-Rothery e/a + distance to
known QC phase fields), because BPR's substrate is quasicrystalline. We compute
it honestly and report the (negative) result.

Composition: Table 2 "Normalized Metals Results", Buga sphere conference slide
(parent column). Elemental (XRF-type) data — NOT isotopic. The discriminating
isotope-ratio measurement has never been publicly released.
"""
from __future__ import annotations

from dataclasses import dataclass

# --- Measured composition (parent column), weight % --------------------------
WT_PCT = {
    "Al": 94.68, "Cu": 2.26, "Fe": 1.25, "Zn": 1.08, "Mn": 0.23, "Mg": 0.17,
    "Ni": 0.14, "Pb": 0.098, "Ti": 0.046, "Cr": 0.035, "V": 0.0097,
    "Sc": 0.0025, "Ce": 0.0012, "La": 0.00068, "Nd": 0.00036, "Pr": 0.00011,
}
MOLAR = {
    "Al": 26.98, "Cu": 63.55, "Fe": 55.85, "Zn": 65.38, "Mn": 54.94, "Mg": 24.31,
    "Ni": 58.69, "Pb": 207.2, "Ti": 47.87, "Cr": 52.0, "V": 50.94, "Sc": 44.96,
    "Ce": 140.1, "La": 138.9, "Nd": 144.2, "Pr": 140.9,
}
# Hume-Rothery valences (transition metals NEGATIVE, Raynor convention) used for
# the electron-per-atom ratio that governs quasicrystal stabilization.
VALENCE = {
    "Al": 3, "Cu": 1, "Fe": -2.66, "Zn": 2, "Mn": -3.66, "Mg": 2, "Ni": 0,
    "Pb": 4, "Ti": 0, "Cr": 0, "V": 0, "Sc": 3, "Ce": 3, "La": 3, "Nd": 3, "Pr": 3,
}


def atomic_percent() -> dict:
    mol = {k: WT_PCT[k] / MOLAR[k] for k in WT_PCT}
    tot = sum(mol.values())
    return {k: 100 * mol[k] / tot for k in mol}


def electron_per_atom() -> float:
    """e/a — the Hume-Rothery electron concentration. Quasicrystal-forming Al-TM
    alloys cluster at e/a ~ 1.75-2.1; pure Al is 3.0."""
    at = atomic_percent()
    return sum(at[k] / 100 * VALENCE[k] for k in at)


# --- Known quasicrystal phase fields (atomic %, Al component) -----------------
QC_PHASE_FIELDS = {
    "Al-Cu-Fe icosahedral (Al62Cu25Fe13)": {"Al": 62, "ea": 1.76},
    "Al-Mn icosahedral (Al86Mn14)": {"Al": 86, "ea": 1.80},
    "Al-Pd-Mn icosahedral (Al70Pd21Mn9)": {"Al": 70, "ea": 1.75},
}


@dataclass(frozen=True)
class ObjectiveScore:
    objective: str
    measured: float
    target: float
    distance: float
    verdict: str


def quasicrystal_proximity() -> list[ObjectiveScore]:
    """Distance of the measured composition from each known QC phase field, in
    both Al-atomic-% and e/a. Large distance => NOT optimized for quasicrystal."""
    at = atomic_percent()
    ea = electron_per_atom()
    out = []
    for name, ref in QC_PHASE_FIELDS.items():
        d_al = abs(at["Al"] - ref["Al"])
        d_ea = abs(ea - ref["ea"])
        verdict = ("FAR — not quasicrystal-forming"
                   if (d_al > 10 or d_ea > 0.5) else "near")
        out.append(ObjectiveScore(name, at["Al"], ref["Al"], d_al, verdict))
    return out


def rare_earth_vs_mischmetal() -> ObjectiveScore:
    """Compare La:Ce:Nd:Pr to natural bastnaesite/mischmetal (~La5:Ce10:Nd3.6:Pr1).
    A match is a TERRESTRIAL ore fingerprint (engineered materials use purified
    single rare earths, not the natural blend)."""
    r_la = WT_PCT["La"] / WT_PCT["Pr"]
    r_ce = WT_PCT["Ce"] / WT_PCT["Pr"]
    r_nd = WT_PCT["Nd"] / WT_PCT["Pr"]
    # natural mischmetal reference ratios (La:Ce:Nd:Pr normalized to Pr=1)
    ref = {"La": 5.0, "Ce": 10.0, "Nd": 3.6}
    meas = {"La": r_la, "Ce": r_ce, "Nd": r_nd}
    dist = sum(abs(meas[k] - ref[k]) for k in ref) / 3.0
    verdict = ("MATCH — natural mischmetal (terrestrial ore fingerprint)"
               if dist < 2.0 else "does not match natural mischmetal")
    return ObjectiveScore("rare-earth blend vs natural mischmetal",
                          r_ce, ref["Ce"], dist, verdict)


def silicon_flag() -> str:
    """Standard cast/recycled Al alloys carry several % Si. This table lists NO
    silicon — consistent with a wrought/secondary metal, not a named cast grade,
    and notably NOT a clean aerospace spec."""
    return ("No Si reported. High Fe (1.25%) + Pb (0.1%) + Zn (1.08%) with no Si "
            "is a SECONDARY/SCRAP aluminium signature, not a clean alloy grade.")


def forensic_report() -> str:
    at = atomic_percent()
    ea = electron_per_atom()
    lines = [
        "Buga-sphere alloy forensics — 'what was it optimized for?'",
        "==========================================================",
        f"Bulk: {at['Al']:.1f} at% Al  (e/a = {ea:.2f}; pure Al = 3.0)",
        "",
        "Quasicrystal-formation test (the only BPR-adjacent objective):",
    ]
    for s in quasicrystal_proximity():
        lines.append(f"  {s.objective:<38} Δ(Al at%)={s.distance:5.1f}  -> {s.verdict}")
    lines += [
        f"  e/a measured {ea:.2f} vs QC window 1.75-2.10  -> "
        f"{'FAR' if ea > 2.1 else 'in window'}",
        "",
        "Rare-earth provenance:",
        f"  {rare_earth_vs_mischmetal().verdict}",
        "",
        "Alloy-grade check:",
        f"  {silicon_flag()}",
        "",
        "VERDICT (forensic, about the SPHERE — not about BPR):",
        "  The composition sits near pure recycled aluminium, tens of atomic-%",
        "  away from every known quasicrystal phase field, with a natural-",
        "  mischmetal rare-earth fingerprint. It is NOT optimized for quasi-",
        "  crystal formation. Consistent with terrestrial secondary aluminium.",
        "",
        "WHAT THIS DOES NOT DO: it does not test BPR. BPR's correctness rides on",
        "the flavor-sector pre-dictions (Xi_cc* hyperfine 85.9 MeV, etc.), not on",
        "this object. And the definitive provenance test — ISOTOPE ratios on the",
        "metal — has never been publicly released; elemental data cannot settle it.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(forensic_report())
