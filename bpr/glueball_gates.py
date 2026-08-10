"""Glueball Benchmark v1, Gates 2-4 — S^2 spectrum, emergent J^PC, envelope opened.

PROTOCOL (doc/GLUEBALL_BENCHMARK_V1.md)
---------------------------------------
Gate 1 (bpr/glueball_benchmark.py) established that the frozen equations
support discrete bound states. Gates 2-4 require the spectrum on the S^2
boundary with quantum numbers EMERGING from the solutions. Per the chain
already frozen in this repo (coherence lives on the k=0 condensate —
bpr/condensate_mechanism.py), the physical boundary state is the uniform
condensate, and the particle spectrum is its Bogoliubov excitation spectrum:

    omega_J = sqrt(eps_J (eps_J + gamma)),   eps_J = J(J+1)/R^2,

with gamma = 2 g n the single dimensionless interaction parameter (the overall
scale R is the ONE scale the protocol allows to be fixed; ratios are gamma's
job alone).

HOW QUANTUM NUMBERS EMERGE (nothing inserted by hand)
-----------------------------------------------------
* J   — a real-space conservative FD solver on S^2 (per azimuthal periodicity
        m, which is mere Fourier periodicity) produces eigenvalue clusters
        whose cross-m degeneracy is discovered numerically: 1, 3, 5, 7, ...
        => J = (deg-1)/2, with eps = J(J+1) as a cross-check.
* P   — the antipodal map (theta -> pi-theta, phi -> phi+pi) applied to the
        numerical eigenvectors gives P = (-1)^J per multiplet.
* C   — the U(1) charge conjugation psi -> psi* acts on Bogoliubov modes with
        real (u, v): quasiparticles are C = +1 (density-wave-like), matching
        the C-even lowest glueball sector.
* Composites — allowed total J of multi-quasiparticle states computed by SO(3)
        CHARACTER INTEGRALS (Haar measure), not hand-coded tables.

RESULT — THE ENVELOPE IS OPENED IN THIS MODULE (see open_envelope())
--------------------------------------------------------------------
The spectrum functions below never read the lattice targets; the targets enter
only in open_envelope(), after the spectrum exists. The verdicts are:

  Gate 2: PARTIAL FAIL. The 0^{++} and 2^{++} families DO emerge (pair of J=1
          quasiparticles: character integral gives exactly J=0 and J=2, both
          P=+, C=+), which is nontrivial. BUT (a) the lightest excitation is a
          single 1^- quasiparticle with no counterpart in the low glueball
          spectrum, and (b) a light 0^{-+} is FORBIDDEN: every 1- and
          2-quasiparticle J=0 state has P=+ (identical-pair selection rule),
          and character integrals kill J=0 for (1,1,1), (l,l,l') with the
          right parity, and (3,3,3). The lightest 0^{-+} is the distinct
          triple (2,3,4) — far too heavy.
  Gate 3: FAIL. Ordering is [1^-], then degenerate {0^{++}, 2^{++}}, with
          0^{-+} several times higher — not the QCD ordering
          0^{++} < 2^{++} < 0^{-+}.
  Gate 4: FAIL. 2^{++}/0^{++} = 1.00 exactly for gamma < 10 (degenerate pair
          states) or < 1 for gamma > 10 (single omega_2 undercuts the pair);
          never in the locked band [1.25, 1.55]. 0^{-+}/0^{++} ranges over
          [3.67, 9.5] across all gamma; never near 1.497. NO gamma rescues it.

Caveats (the only loopholes, recorded before anyone asks): (i) verdicts are
leading order in quasiparticle interactions — interactions split the 0/2
degeneracy but cannot plausibly bridge 1.00 -> 1.39 or 3.7 -> 1.5; (ii) the
self-bound-lump branch (focusing sign, broken-symmetry collective quantization)
and (iii) the phason sector's boundary parity content were not computed. These
are listed as future work, not used to soften the verdict.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Emergent S^2 spectrum (real-space, conservative; J/P discovered numerically)
# ---------------------------------------------------------------------------

def s2_multiplets(n_theta: int = 800, m_max: int = 6, k_per_m: int = 8,
                  n_clusters: int = 6) -> list[dict]:
    """Diagonalize the S^2 Laplacian per azimuthal periodicity m (Fourier
    periodicity only — no spherical harmonics inserted) and cluster across m.
    Returns multiplets with EMERGENT degeneracy, J, eps, and parity."""
    dth = np.pi / n_theta
    th = (np.arange(n_theta) + 0.5) * dth
    s = np.sin(th)
    sph = np.sin(th + dth / 2)
    smh = np.sin(th - dth / 2)
    smh[0] = 0.0
    sph[-1] = 0.0

    entries = []   # (eps, m, theta_reflection_sign)
    for m in range(0, m_max + 1):
        main = (sph + smh) / (s * dth * dth) + (m * m) / (s * s)
        off = -sph[:-1] / (np.sqrt(s[:-1] * s[1:]) * dth * dth)
        T = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
        w, V = np.linalg.eigh(T)
        for i in range(k_per_m):
            refl = float(V[:, i] @ V[::-1, i])
            entries.append((float(w[i]), m, int(np.sign(refl))))
    entries.sort()

    clusters, cur = [], [entries[0]]
    for e in entries[1:]:
        if e[0] - cur[-1][0] > 0.5:
            clusters.append(cur)
            cur = [e]
        else:
            cur.append(e)
    clusters.append(cur)

    out = []
    for cl in clusters[:n_clusters]:
        ms = sorted(c[1] for c in cl)
        deg = sum(2 if m > 0 else 1 for m in ms)      # +/-m both exist
        J = (deg - 1) // 2
        # parity per member: antipodal = (-1)^m x theta-reflection sign
        pars = {(-1) ** c[1] * c[2] for c in cl}
        out.append({
            "eps": float(np.mean([c[0] for c in cl])),
            "degeneracy": deg,
            "J_emergent": J,
            "parity": pars.pop() if len(pars) == 1 else 0,   # 0 = inconsistent
            "eps_expected_JJp1": float(J * (J + 1)),
        })
    return out


# ---------------------------------------------------------------------------
# Bogoliubov dispersion on the condensate (the frozen branch)
# ---------------------------------------------------------------------------

def omega(J: int, gamma: float) -> float:
    """Bogoliubov frequency of the J-multiplet: sqrt(eps(eps+gamma)), eps=J(J+1).
    gamma = 2 g n > 0 (defocusing/condensate branch, per condensate_mechanism)."""
    eps = float(J * (J + 1))
    return float(np.sqrt(eps * (eps + gamma)))


# ---------------------------------------------------------------------------
# Composite selection rules by SO(3) character integrals (computed, not coded)
# ---------------------------------------------------------------------------

_T = np.linspace(1e-6, np.pi, 100001)
_DT = _T[1] - _T[0]


def _chi(l: int, t: np.ndarray = _T) -> np.ndarray:
    return np.sin((2 * l + 1) * t / 2) / np.sin(t / 2)


def _mult(chi_prod: np.ndarray, J: int) -> int:
    """Multiplicity of spin J in a product rep, by Haar-measure projection."""
    val = np.sum((1 - np.cos(_T)) * chi_prod * _chi(J)) * _DT / np.pi
    return int(round(float(val)))


def _sym2(l: int) -> np.ndarray:
    return (_chi(l) ** 2 + np.sin((2 * l + 1) * _T) / np.sin(_T)) / 2


def _sym3(l: int) -> np.ndarray:
    chi2t = np.sin((2 * l + 1) * _T) / np.sin(_T)
    chi3t = np.sin((2 * l + 1) * 3 * _T / 2) / np.sin(3 * _T / 2)
    return (_chi(l) ** 3 + 3 * _chi(l) * chi2t + 2 * chi3t) / 6


def pair_J0_J2_from_two_J1() -> dict:
    """The nontrivial Gate-2 positive: two identical J=1 quasiparticles give
    exactly {J=0, J=2} (J=1 excluded by bose symmetry), both P=+, C=+."""
    return {"J0": _mult(_sym2(1), 0), "J1": _mult(_sym2(1), 1),
            "J2": _mult(_sym2(1), 2), "parity": +1}


def lightest_pseudoscalar(l_max: int = 6, gamma: float = 1.0) -> dict:
    """Search 1-, 2-, 3-quasiparticle states for the lightest J^P = 0^-.

    Selection rules (all computed):
      * single: J=l, P=(-1)^l -> 0^- impossible (l=0 has P=+).
      * pair identical l: P=+ always. pair distinct: J=0 needs l1=l2 -> P=+.
        => NO 0^- with <= 2 quasiparticles. (Proved by parity arithmetic.)
      * triples: need l1+l2+l3 odd AND J=0 allowed under bose symmetry
        (character integrals for identical subsets).
    Returns the lightest candidate and its energy in units where R=1.
    """
    best = None
    for l1 in range(1, l_max + 1):
        for l2 in range(l1, l_max + 1):
            for l3 in range(l2, l_max + 1):
                if (l1 + l2 + l3) % 2 == 0:
                    continue                       # parity must be -1
                if l1 == l2 == l3:
                    ok = _mult(_sym3(l1), 0) > 0
                elif l1 == l2:
                    ok = _mult(_sym2(l1) * _chi(l3), 0) > 0
                elif l2 == l3:
                    ok = _mult(_sym2(l2) * _chi(l1), 0) > 0
                else:
                    ok = _mult(_chi(l1) * _chi(l2) * _chi(l3), 0) > 0
                if not ok:
                    continue
                E = omega(l1, gamma) + omega(l2, gamma) + omega(l3, gamma)
                if best is None or E < best["E"]:
                    best = {"modes": (l1, l2, l3), "E": E}
    return best


# ---------------------------------------------------------------------------
# The spectrum and the gamma scan (no targets read anywhere above or here)
# ---------------------------------------------------------------------------

def spectrum_ratios(gamma: float) -> dict:
    """Mass ratios of the leading-order spectrum at interaction gamma.

      M(0++) = 2 omega_1              (pair of J=1, character-integral J=0)
      M(2++) = min(2 omega_1, omega_2) (pair J=2 degenerate with 0++, or the
                                        single J=2 quasiparticle if lighter)
      M(0-+) = lightest pseudoscalar triple (see lightest_pseudoscalar)
      lightest state overall: single 1^- at omega_1 (no glueball counterpart)
    """
    w1, w2 = omega(1, gamma), omega(2, gamma)
    m0pp = 2.0 * w1
    m2pp = min(m0pp, w2)
    ps = lightest_pseudoscalar(gamma=gamma)
    return {
        "gamma": gamma,
        "lightest_state": {"JPC": "1-+", "E": w1},
        "M_0pp": m0pp,
        "M_2pp": m2pp,
        "M_0mp": ps["E"],
        "pseudoscalar_modes": ps["modes"],
        "r_2pp": m2pp / m0pp,
        "r_0mp": ps["E"] / m0pp,
    }


def gamma_scan(gammas=None) -> list[dict]:
    if gammas is None:
        gammas = [0.1, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0]
    return [spectrum_ratios(g) for g in gammas]


# ---------------------------------------------------------------------------
#                     * * *  ENVELOPE OPENED BELOW  * * *
# The sealed targets (doc/GLUEBALL_BENCHMARK_V1.md sec.4) enter HERE and only
# here, AFTER the spectrum above is fully determined. Nothing above reads them.
# ---------------------------------------------------------------------------

SEALED_RATIO_2PP = 1.387          # Morningstar-Peardon 2400/1730
SEALED_RATIO_0MP = 1.497          # Morningstar-Peardon 2590/1730
GATE4_BAND_2PP = (1.25, 1.55)
GATE4_BAND_0MP = (1.35, 1.65)


def open_envelope() -> dict:
    """Compare the computed spectrum against the sealed targets. Verdicts."""
    scan = gamma_scan()
    any_2pp = any(GATE4_BAND_2PP[0] <= r["r_2pp"] <= GATE4_BAND_2PP[1]
                  for r in scan)
    any_0mp = any(GATE4_BAND_0MP[0] <= r["r_0mp"] <= GATE4_BAND_0MP[1]
                  for r in scan)
    pair = pair_J0_J2_from_two_J1()
    return {
        "gate2": {
            "families_0pp_2pp_emerge": pair["J0"] == 1 and pair["J2"] == 1
                                        and pair["J1"] == 0,
            "extra_light_1_minus": True,
            "light_0mp_forbidden": True,
            "verdict": "PARTIAL FAIL",
        },
        "gate3": {
            "computed_ordering": "1^- < {0++ = 2++} << 0-+",
            "required_ordering": "0++ < 2++ < 0-+",
            "verdict": "FAIL",
        },
        "gate4": {
            "r_2pp_range": (min(r["r_2pp"] for r in scan),
                            max(r["r_2pp"] for r in scan)),
            "r_0mp_range": (min(r["r_0mp"] for r in scan),
                            max(r["r_0mp"] for r in scan)),
            "target_2pp": SEALED_RATIO_2PP,
            "target_0mp": SEALED_RATIO_0MP,
            "any_gamma_passes_2pp": any_2pp,
            "any_gamma_passes_0mp": any_0mp,
            "verdict": "FAIL",
        },
        "loopholes": (
            "leading-order in quasiparticle interactions (splits 0/2 "
            "degeneracy; cannot plausibly bridge 1.00->1.39 or 3.7->1.5)",
            "self-bound-lump branch (focusing sign) not quantized",
            "phason-sector boundary parity content not computed",
        ),
        "overall": "BPR condensate branch FAILS the glueball benchmark",
    }


def report() -> str:
    mult = s2_multiplets(n_theta=400, m_max=4, k_per_m=6, n_clusters=5)
    env = open_envelope()
    scan = gamma_scan([0.1, 1.0, 10.0, 100.0])
    lines = [
        "Glueball Benchmark v1 — Gates 2-4 (envelope OPENED)",
        "===================================================",
        "",
        "Emergent S^2 multiplets (deg -> J, parity from antipodal map):",
    ]
    for m in mult:
        lines.append(f"  eps={m['eps']:8.4f}  deg={m['degeneracy']:2d}  "
                     f"J={m['J_emergent']}  P={m['parity']:+d}  "
                     f"[J(J+1)={m['eps_expected_JJp1']:.0f}]")
    lines += [
        "",
        "Composite families (character integrals): two J=1 quasiparticles ->",
        "  exactly {0++, 2++} (J=1 bose-forbidden). C=+ throughout. EMERGES.",
        "Pseudoscalar: no 0^- with <=2 quasiparticles (parity arithmetic);",
        f"  lightest triple: {spectrum_ratios(1.0)['pseudoscalar_modes']}.",
        "",
        "gamma scan (ratios to M(0++)):",
    ]
    for r in scan:
        lines.append(f"  gamma={r['gamma']:7.1f}:  2++/0++ = {r['r_2pp']:.3f}   "
                     f"0-+/0++ = {r['r_0mp']:.3f}   lightest = 1^- at "
                     f"{r['lightest_state']['E']/r['M_0pp']:.2f} x M(0++)")
    g4 = env["gate4"]
    lines += [
        "",
        f"SEALED TARGETS: 2++/0++ = {g4['target_2pp']}, 0-+/0++ = {g4['target_0mp']}",
        f"COMPUTED RANGES: 2++/0++ in [{g4['r_2pp_range'][0]:.2f}, "
        f"{g4['r_2pp_range'][1]:.2f}], 0-+/0++ in [{g4['r_0mp_range'][0]:.2f}, "
        f"{g4['r_0mp_range'][1]:.2f}]",
        "",
        f"GATE 2: {env['gate2']['verdict']}  (0++/2++ families emerge; extra "
        "light 1^-; light 0-+ forbidden)",
        f"GATE 3: {env['gate3']['verdict']}  ({env['gate3']['computed_ordering']}"
        f" vs required {env['gate3']['required_ordering']})",
        f"GATE 4: {env['gate4']['verdict']}  (no gamma reaches either band)",
        "",
        f"OVERALL: {env['overall']}.",
        "Loopholes (recorded, not used to soften the verdict):",
    ]
    lines += [f"  - {lp}" for lp in env["loopholes"]]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
