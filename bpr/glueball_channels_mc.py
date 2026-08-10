"""Path B M4 blocker machinery: J^PC-channel correlators for the frozen H.

THE BLOCKER (from gauge_dynamics_m2_m3): the leading-order strong-coupling
spectrum is J^PC-degenerate; splitting requires higher orders or Monte Carlo.
This module supplies the Monte Carlo route: zero-momentum loop-operator
correlators in distinct lattice-symmetry channels, measured in the confining
phase of the same 3D Euclidean Wilson theory used to locate the M2 transition
(`gauge_phase_mc`).

CHANNELS (square-lattice point group C_4v acting on the spatial plane;
time = the correlation axis):
    A1  symmetric plaquette sum            -> 0^{++} precursor
    B1  x-rectangle minus y-rectangle      -> 2^{++} precursor
    A2  L-loop minus its mirror image      -> parity-odd precursor
The A2 operator is built from bent (L-shaped) loops because for D_n all
plaquette characters are real, so square loops are reflection-even
identically; parity-odd content needs chiral loop shapes.

THE ENVELOPE STAYS SEALED UNTIL THE GATES PASS
----------------------------------------------
Benchmark v3 allows exactly one comparison against the locked targets.
Producing under-converged ratios and comparing would burn that shot on noise.
This module therefore hard-gates its output:

    GATE 1 (signal): effective-mass plateau exists (>= 2 consecutive t with
            overlapping jackknife errors) in every channel needed for a ratio.
    GATE 2 (error):  relative jackknife error on each plateau mass < 15%.
    GATE 3 (volume): A1 mass stable within errors across two lattice sizes.

`benchmark_v3_readiness()` reports gate status. Ratios are only exposed when
all gates pass, and even then the comparison against the sealed targets is a
separate, deliberate, one-time act — not performed by this module.

BLINDNESS: no glueball target number appears here (grep-enforced in tests).

WHAT THIS DELIVERS vs NOT
-------------------------
Delivers: the measurement machinery, validated operator symmetry properties,
and an honest readiness verdict at any given compute budget.
Does not deliver: a benchmark verdict. If gates fail at the available budget,
the envelope stays sealed and the blocker remains "insufficient statistics",
which is a compute problem, not a physics wall.
"""
from __future__ import annotations

import numpy as np

from .gauge_phase_mc import WilsonMC, dn_tables

# ---------------------------------------------------------------------------
# Loop operators (measured on the z=const time slices of the 3D lattice)
# ---------------------------------------------------------------------------


def _loop_char(mc: WilsonMC, sites_dirs, t: int) -> float:
    """Re chi_F(product of links)/d for a closed loop in the z=t slice.

    sites_dirs: sequence of ((x, y), mu, sign) steps; mu in {0, 1} spatial.
    sign=+1 walks the link forward, -1 backward (inverse element).
    """
    M, I, c, d = mc.M, mc.I, mc.c, mc.d
    g = 0                                      # identity index
    for (x, y), mu, sign in sites_dirs:
        u = mc.U[x % mc.L, y % mc.L, t % mc.L, mu]
        g = M[g, u if sign == 1 else I[u]]
    return float(c[g] / d)


def _plaquette_xy(x, y):
    return (((x, y), 0, 1), ((x + 1, y), 1, 1), ((x, y + 1), 0, -1),
            ((x, y), 1, -1))


def _rect(x, y, lx, ly):
    """lx x ly rectangle from (x, y), counterclockwise."""
    steps = []
    for i in range(lx):
        steps.append(((x + i, y), 0, 1))
    for j in range(ly):
        steps.append(((x + lx, y + j), 1, 1))
    for i in range(lx - 1, -1, -1):
        steps.append(((x + i, y + ly), 0, -1))
    for j in range(ly - 1, -1, -1):
        steps.append(((x, y + j), 1, -1))
    return tuple(steps)


def _l_loop(x, y):
    """Chiral L-shaped 6-link loop (an axis-asymmetric bent rectangle)."""
    return (((x, y), 0, 1), ((x + 1, y), 0, 1), ((x + 2, y), 1, 1),
            ((x + 2, y + 1), 0, -1), ((x + 1, y + 1), 1, 1),
            ((x + 1, y + 2), 0, -1), ((x, y + 2), 1, -1), ((x, y + 1), 1, -1))


def _l_loop_mirror(x, y):
    """The x-reflection of _l_loop (swap the roles so the shape is mirrored)."""
    return (((x, y), 1, 1), ((x, y + 1), 1, 1), ((x, y + 2), 0, 1),
            ((x + 1, y + 2), 1, -1), ((x + 1, y + 1), 0, 1),
            ((x + 2, y + 1), 1, -1), ((x + 2, y), 0, -1), ((x + 1, y), 0, -1))


def slice_operators(mc: WilsonMC, t: int) -> dict:
    """Zero-momentum channel operators on time slice t."""
    L = mc.L
    a1 = b1 = a2 = 0.0
    for x in range(L):
        for y in range(L):
            a1 += _loop_char(mc, _plaquette_xy(x, y), t)
            b1 += (_loop_char(mc, _rect(x, y, 2, 1), t)
                   - _loop_char(mc, _rect(x, y, 1, 2), t))
            a2 += (_loop_char(mc, _l_loop(x, y), t)
                   - _loop_char(mc, _l_loop_mirror(x, y), t))
    norm = L * L
    return {"A1": a1 / norm, "B1": b1 / norm, "A2": a2 / norm}


# ---------------------------------------------------------------------------
# Correlators, effective masses, jackknife
# ---------------------------------------------------------------------------


def measure_correlators(n: int, beta: float, L: int = 4, seed: int = 3,
                        n_equil: int = 200, n_meas: int = 400,
                        stride: int = 2) -> dict:
    """Time-slice correlator matrices C_ch(t) with per-configuration storage
    (for jackknife). Connected correlators; A1 is vacuum-subtracted."""
    mc = WilsonMC(dn_tables(n), L=L, beta=beta, seed=seed)
    for _ in range(n_equil):
        mc.sweep()
    slices = []          # (n_cfg, L, channels)
    for k in range(n_meas):
        mc.sweep()
        if k % stride:
            continue
        slices.append([slice_operators(mc, t) for t in range(L)])
    chans = ("A1", "B1", "A2")
    n_cfg = len(slices)
    O = {ch: np.array([[s[t][ch] for t in range(L)] for s in slices])
         for ch in chans}
    corr = {}
    for ch in chans:
        v = O[ch]                                    # (n_cfg, L)
        vac = v.mean() if ch == "A1" else 0.0        # only A1 overlaps vacuum
        c = np.zeros((n_cfg, L))
        for dt in range(L):
            c[:, dt] = ((v * np.roll(v, -dt, axis=1)).mean(axis=1) - vac**2)
        corr[ch] = c
    return {"n": n, "beta": beta, "L": L, "n_cfg": n_cfg, "corr": corr}


def jackknife_effective_mass(c: np.ndarray) -> list:
    """m_eff(t) = log(C(t)/C(t+1)) with delete-1 jackknife errors.
    Entries are None where the signal is unusable (non-positive C)."""
    n_cfg, L = c.shape
    out = []
    for t in range(L - 1):
        full_a, full_b = c[:, t].mean(), c[:, t + 1].mean()
        if full_a <= 0 or full_b <= 0 or full_b >= full_a:
            out.append(None)
            continue
        m = np.log(full_a / full_b)
        jk = []
        for i in range(n_cfg):
            a = np.delete(c[:, t], i).mean()
            b = np.delete(c[:, t + 1], i).mean()
            if a > 0 and b > 0 and b < a:
                jk.append(np.log(a / b))
        if len(jk) < n_cfg // 2:
            out.append(None)
            continue
        jk = np.array(jk)
        err = float(np.sqrt((len(jk) - 1) * jk.var()))
        out.append({"t": t, "m_eff": float(m), "err": err})
    return out


def plateau(meffs: list, rel_err_max: float = 0.15) -> dict | None:
    """First pair of consecutive usable m_eff values that agree within
    combined errors, with acceptable relative error. None if absent."""
    for a, b in zip(meffs, meffs[1:]):
        if a is None or b is None:
            continue
        if abs(a["m_eff"] - b["m_eff"]) < (a["err"] + b["err"]):
            m = 0.5 * (a["m_eff"] + b["m_eff"])
            e = 0.5 * np.hypot(a["err"], b["err"])
            if m > 0 and e / m < rel_err_max:
                return {"mass": float(m), "err": float(e), "t_start": a["t"]}
    return None


# ---------------------------------------------------------------------------
# The readiness verdict (the honest interface to Benchmark v3)
# ---------------------------------------------------------------------------


def benchmark_v3_readiness(n: int, beta: float, L: int = 4, seed: int = 3,
                           n_equil: int = 200, n_meas: int = 400,
                           L2: int | None = None) -> dict:
    """Run the machinery at the given budget and report gate status.

    Ratios appear in the output ONLY if all gates pass. Comparison to the
    sealed targets is intentionally NOT performed here.
    """
    run = measure_correlators(n, beta, L=L, seed=seed,
                              n_equil=n_equil, n_meas=n_meas)
    plats = {ch: plateau(jackknife_effective_mass(run["corr"][ch]))
             for ch in ("A1", "B1", "A2")}
    gate1 = {ch: p is not None for ch, p in plats.items()}
    gate2 = {ch: (p is not None and p["err"] / p["mass"] < 0.15)
             for ch, p in plats.items()}
    gate3 = None
    if L2 is not None and plats["A1"] is not None:
        run2 = measure_correlators(n, beta, L=L2, seed=seed + 1,
                                   n_equil=n_equil, n_meas=n_meas)
        p2 = plateau(jackknife_effective_mass(run2["corr"]["A1"]))
        gate3 = (p2 is not None and
                 abs(p2["mass"] - plats["A1"]["mass"])
                 < 2 * (p2["err"] + plats["A1"]["err"]))
    ready = (all(gate1.values()) and all(gate2.values())
             and (gate3 is True))
    out = {
        "n": n, "beta": beta, "L": L, "n_cfg": run["n_cfg"],
        "plateaus": plats,
        "gate1_plateau": gate1,
        "gate2_error": gate2,
        "gate3_volume": gate3,
        "ready_for_v3": bool(ready),
        "envelope": "SEALED",
        "note": ("ratios withheld until all gates pass; comparison to the "
                 "locked targets is a separate one-time act"),
    }
    if ready:                                        # pragma: no cover
        m0 = plats["A1"]["mass"]
        out["ratios_available"] = {
            "B1_over_A1": plats["B1"]["mass"] / m0,
            "A2_over_A1": plats["A2"]["mass"] / m0,
        }
    return out


def report(n: int = 5, beta: float = 1.0, L: int = 4) -> str:  # pragma: no cover
    r = benchmark_v3_readiness(n, beta, L=L)
    lines = [
        f"M4 machinery — D_{n} channel correlators at beta={beta}, L={L}",
        "=" * 60,
        f"configs: {r['n_cfg']}",
    ]
    for ch, p in r["plateaus"].items():
        lines.append(f"  {ch}: " + (
            f"m = {p['mass']:.3f} +- {p['err']:.3f} (t0={p['t_start']})"
            if p else "no plateau at this budget"))
    lines += [
        f"gates: plateau={r['gate1_plateau']} error={r['gate2_error']} "
        f"volume={r['gate3_volume']}",
        f"ready_for_v3: {r['ready_for_v3']}   envelope: {r['envelope']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
