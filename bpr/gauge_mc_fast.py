"""Vectorized checkerboard Metropolis for finite-group Wilson theories (Path B, M4).

Purpose: remove the compute blocker recorded in
doc/derivations/path_b_open_problems_2026-08.md §2 ("insufficient statistics").
Same frozen dynamics as `gauge_phase_mc.WilsonMC` (plaquette action in the
faithful irrep, beta(1 - Re chi/d)); same group tables; nothing new in the
physics. Differences are purely computational:

  * L_s x L_s x L_t lattice (time axis = correlation axis, may be longer),
  * optional anisotropy beta_t != beta_s (spatial vs temporal plaquettes),
  * checkerboard updates: for direction mu, links whose coordinates
    perpendicular to mu sum to the same parity share no plaquette, so all
    of them are updated in one vectorized Metropolis step (6 passes/sweep),
  * loop characters on time slices computed for all (x, y) at once.

Validated against WilsonMC (mean plaquette agrees within errors) in
tests/test_gauge_mc_fast.py. Blind to the sealed benchmark targets.
"""
from __future__ import annotations

import numpy as np

from .gauge_phase_mc import dn_tables, z2_tables  # noqa: F401  (re-export)


class FastWilsonMC:
    def __init__(self, tables, Ls: int = 8, Lt: int = 16, beta: float = 1.0,
                 beta_t: float | None = None, seed: int = 0):
        self.M, self.I, self.c, self.d = tables
        self.G = len(self.I)
        self.Ls, self.Lt = Ls, Lt
        self.beta_s = float(beta)
        self.beta_t = float(beta if beta_t is None else beta_t)
        self.rng = np.random.default_rng(seed)
        self.U = np.zeros((Ls, Ls, Lt, 3), dtype=np.int64)     # cold start
        cs = np.arange(Ls)[:, None, None]
        ys = np.arange(Ls)[None, :, None]
        ts = np.arange(Lt)[None, None, :]
        self._coord = (cs, ys, ts)
        # parity masks: for direction mu, parity of sum of the other two coords
        self._masks = []
        for mu in range(3):
            others = [self._coord[nu] for nu in range(3) if nu != mu]
            par = (others[0] + others[1]) % 2
            self._masks.append([par == 0, par == 1])

    # -- helpers -------------------------------------------------------------
    def _sh(self, A, mu, k):
        """Field A(site) shifted so that result[s] = A[s + k e_mu]."""
        return np.roll(A, -k, axis=mu)

    def _link(self, mu):
        return self.U[..., mu]

    def _staples(self, mu):
        """For every site s, the two cyclic staples per nu != mu, i.e. group
        elements S with chi(plaq) = chi(U_mu(s) * S). Returns list of
        (staple_array, beta_for_that_plaquette)."""
        M, I = self.M, self.I
        out = []
        for nu in range(3):
            if nu == mu:
                continue
            b = self.beta_t if (mu == 2 or nu == 2) else self.beta_s
            Umu, Unu = self._link(mu), self._link(nu)
            # plaquette (s, mu, nu): U_mu(s) U_nu(s+mu) U_mu(s+nu)^-1 U_nu(s)^-1
            s1 = M[M[self._sh(Unu, mu, 1), I[self._sh(Umu, nu, 1)]], I[Unu]]
            # plaquette (s-nu, nu, mu): U_nu(s-nu) U_mu(s) U_nu(s-nu+mu)^-1 U_mu(s-nu)^-1
            # cyclic after U_mu(s): U_nu(s-nu+mu)^-1 U_mu(s-nu)^-1 U_nu(s-nu)
            Unu_m = self._sh(Unu, nu, -1)
            s2 = M[M[I[self._sh(Unu_m, mu, 1)], I[self._sh(Umu, nu, -1)]], Unu_m]
            out.append((s1, b))
            out.append((s2, b))
        return out

    # -- Metropolis ----------------------------------------------------------
    def sweep(self):
        M, c, d = self.M, self.c, self.d
        shape = self.U.shape[:3]
        for mu in range(3):
            for mask in self._masks[mu]:
                staples = self._staples(mu)
                old = self.U[..., mu]
                new = self.rng.integers(0, self.G, size=shape)
                dS = np.zeros(shape)
                for S, b in staples:
                    dS -= (b / d) * (c[M[new, S]] - c[M[old, S]])
                acc = (dS <= 0) | (self.rng.random(shape) < np.exp(-np.clip(dS, 0, 700)))
                upd = acc & mask
                self.U[..., mu] = np.where(upd, new, old)

    def mean_plaquette(self) -> float:
        M, I, c, d = self.M, self.I, self.c, self.d
        tot = 0.0
        for mu in range(3):
            for nu in range(mu + 1, 3):
                Umu, Unu = self._link(mu), self._link(nu)
                f = M[M[M[Umu, self._sh(Unu, mu, 1)], I[self._sh(Umu, nu, 1)]], I[Unu]]
                tot += float((c[f] / d).mean())
        return tot / 3.0

    # -- loop operators on time slices --------------------------------------
    def loop_char(self, steps) -> np.ndarray:
        """chi(loop)/d for the loop started at every (x, y) on every time
        slice t. steps: sequence of ((dx, dy), mu, sign) with mu in {0, 1}.
        Returns array (Ls, Ls, Lt)."""
        M, I = self.M, self.I
        g = np.zeros(self.U.shape[:3], dtype=np.int64)          # identity
        for (dx, dy), mu, sign in steps:
            u = np.roll(np.roll(self.U[..., mu], -dx, axis=0), -dy, axis=1)
            g = M[g, u if sign == 1 else I[u]]
        return self.c[g] / self.d


# ---------------------------------------------------------------------------
# Loop shapes (same conventions as glueball_channels_mc, generalised in size)
# ---------------------------------------------------------------------------

def rect(lx: int, ly: int):
    steps = []
    for i in range(lx):
        steps.append(((i, 0), 0, 1))
    for j in range(ly):
        steps.append(((lx, j), 1, 1))
    for i in range(lx - 1, -1, -1):
        steps.append(((i, ly), 0, -1))
    for j in range(ly - 1, -1, -1):
        steps.append(((0, j), 1, -1))
    return tuple(steps)


def l_loop(s: int = 2, k: int = 1):
    """Chiral bent loop: an s x s square with its top-right k x k corner
    removed (s=2, k=1 is the 8-link L). Conventions as in rect(): a step
    ((x, y), mu, +1) walks link U_mu(x, y) forward from (x, y); a step
    ((x, y), mu, -1) walks it backward, arriving at (x, y). Every loop here
    is contiguous and closed, hence gauge invariant."""
    assert 0 < k < s
    steps = []
    for i in range(s):                       # right along the bottom
        steps.append(((i, 0), 0, 1))
    for j in range(s - k):                   # up the right side to the notch
        steps.append(((s, j), 1, 1))
    for i in range(s - 1, s - k - 1, -1):    # left along the notch bottom
        steps.append(((i, s - k), 0, -1))
    for j in range(s - k, s):                # up the notch's inner side
        steps.append(((s - k, j), 1, 1))
    for i in range(s - k - 1, -1, -1):       # left along the top
        steps.append(((i, s), 0, -1))
    for j in range(s - 1, -1, -1):           # down the left side
        steps.append(((0, j), 1, -1))
    return tuple(steps)


def polyomino_loop(cells):
    """Counter-clockwise boundary loop of a set of unit cells (x, y), in the
    step convention of rect(). Shared internal edges cancel. The cell set
    must be edge-connected and simply connected (no holes)."""
    cells = set(cells)
    edges = {}   # directed edge (start, end) -> count
    for (x, y) in cells:
        for a, b in (((x, y), (x + 1, y)), ((x + 1, y), (x + 1, y + 1)),
                     ((x + 1, y + 1), (x, y + 1)), ((x, y + 1), (x, y))):
            if (b, a) in edges:
                del edges[(b, a)]
            else:
                edges[(a, b)] = 1
    nxt = {a: b for (a, b) in edges}
    assert len(nxt) == len(edges), "boundary is not a simple loop"
    start = min(nxt)
    steps, cur = [], start
    while True:
        b = nxt[cur]
        (x, y), (x2, y2) = cur, b
        if x2 == x + 1:
            steps.append(((x, y), 0, 1))
        elif y2 == y + 1:
            steps.append(((x, y), 1, 1))
        elif x2 == x - 1:
            steps.append(((x2, y2), 0, -1))
        else:
            steps.append(((x2, y2), 1, -1))
        cur = b
        if cur == start:
            break
    assert len(steps) == len(edges), "boundary has more than one component"
    # translate so the loop starts at (0, 0)
    sx, sy = start
    return tuple((((dx - sx, dy - sy), mu, sign)) for (dx, dy), mu, sign in steps)


# Chiral polyominoes: their mirror images are not rotations of themselves,
# so a (loop - reflection) combination does not cancel after rotation
# symmetrisation. Achiral shapes (rectangles, the notched square) give an
# identically vanishing A2 operator when characters are real.
CHIRAL_SHAPES = {
    "L4": ((0, 0), (1, 0), (2, 0), (2, 1)),
    "S4": ((0, 0), (1, 0), (1, 1), (2, 1)),
    "P5": ((0, 0), (1, 0), (0, 1), (1, 1), (0, 2)),
}


def reflect_x(steps):
    """Spatial parity: x -> -x. An x-link based at (x, y) walked forward
    becomes the x-link based at (-x-1, y) walked backward."""
    out = []
    for (dx, dy), mu, sign in steps:
        if mu == 0:
            out.append(((-dx - 1, dy), 0, -sign))
        else:
            out.append(((-dx, dy), 1, sign))
    return tuple(out)


def rotate90(steps):
    """(x, y) -> (-y, x). x-links become y-links; y-links become backward
    x-links based one site to the left."""
    out = []
    for (dx, dy), mu, sign in steps:
        if mu == 0:
            out.append(((-dy, dx), 1, sign))
        else:
            out.append(((-dy - 1, dx), 0, -sign))
    return tuple(out)


def mirror(steps):
    """Kept for API compatibility: x <-> y swap. NOTE: for a loop symmetric
    under the diagonal this is orientation reversal, not parity; use
    reflect_x for parity-odd operators."""
    return tuple((((dy, dx), 1 - mu, sign)) for (dx, dy), mu, sign in steps)


def rotations(steps):
    r = [steps]
    for _ in range(3):
        r.append(rotate90(r[-1]))
    return r


def parity_odd_terms(loop):
    """Rotation-symmetrised (loop - reflect_x(loop)): the A2 combination of
    C_4v, even under 90-degree rotations, odd under reflections."""
    terms = []
    for r in rotations(loop):
        terms.append((1.0, r))
        terms.append((-1.0, reflect_x(r)))
    return terms


def channel_basis(n_ops: int = 3) -> dict:
    """Variational operator basis per C_4v channel. Each entry is a list of
    (weight, loop) terms; the zero-momentum operator is the sum over (x, y).
    A1: symmetric loops. B1: x-y antisymmetric rectangles.
    A2: rotation-symmetrised chiral polyomino loop minus its x-reflection
    (parity-odd). Two defects of the earlier glueball_channels_mc A2 are
    fixed here: its 8-link "L" was not a contiguous path (hence not gauge
    invariant), and the notched-square shape is achiral, so a (loop -
    mirror) combination cancels identically for real characters. See
    tests/test_gauge_mc_fast.py."""
    a1 = [[(1.0, rect(1, 1))], [(1.0, rect(2, 1)), (1.0, rect(1, 2))],
          [(1.0, rect(2, 2))], [(1.0, rect(3, 1)), (1.0, rect(1, 3))]]
    b1 = [[(1.0, rect(2, 1)), (-1.0, rect(1, 2))],
          [(1.0, rect(3, 1)), (-1.0, rect(1, 3))],
          [(1.0, rect(3, 2)), (-1.0, rect(2, 3))]]
    a2 = [parity_odd_terms(polyomino_loop(CHIRAL_SHAPES[k])) for k in ("L4", "S4", "P5")]
    return {"A1": a1[:n_ops], "B1": b1[:n_ops], "A2": a2[:n_ops]}


def slice_operators(mc: FastWilsonMC, basis: dict) -> dict:
    """Zero-momentum operators per channel: array (n_ops, Lt)."""
    out = {}
    for ch, ops in basis.items():
        rows = []
        for terms in ops:
            f = sum(w * mc.loop_char(loop) for w, loop in terms)
            rows.append(f.mean(axis=(0, 1)))
        out[ch] = np.array(rows)
    return out


# ---------------------------------------------------------------------------
# Production measurement with per-configuration storage
# ---------------------------------------------------------------------------

def measure(n: int, beta: float, Ls: int = 8, Lt: int = 16, beta_t=None,
            seed: int = 5, n_equil: int = 500, n_meas: int = 20000,
            stride: int = 4, n_ops: int = 3, tables=None, log=None) -> dict:
    tables = dn_tables(n) if tables is None else tables
    mc = FastWilsonMC(tables, Ls=Ls, Lt=Lt, beta=beta, beta_t=beta_t, seed=seed)
    basis = channel_basis(n_ops)
    for _ in range(n_equil):
        mc.sweep()
    store = {ch: [] for ch in basis}
    plaq = []
    for k in range(n_meas):
        mc.sweep()
        if k % stride:
            continue
        ops = slice_operators(mc, basis)
        for ch in basis:
            store[ch].append(ops[ch])
        plaq.append(mc.mean_plaquette())
        if log and (len(plaq) % 500 == 0):
            log(f"cfg {len(plaq)}")
    O = {ch: np.array(v) for ch, v in store.items()}         # (ncfg, nops, Lt)
    return {"n": n, "beta": beta, "beta_t": mc.beta_t, "Ls": Ls, "Lt": Lt,
            "n_cfg": len(plaq), "plaq": np.array(plaq), "ops": O}


def correlator_matrix(O: np.ndarray, vacuum_subtract: bool) -> np.ndarray:
    """C_ij(t) per configuration: shape (ncfg, nops, nops, Lt), time-averaged
    over the slice separation, connected if vacuum_subtract."""
    ncfg, nops, Lt = O.shape
    vac = O.mean(axis=(0, 2)) if vacuum_subtract else np.zeros(nops)
    C = np.zeros((ncfg, nops, nops, Lt))
    for dt in range(Lt):
        Os = np.roll(O, -dt, axis=2)
        C[..., dt] = np.einsum('cit,cjt->cij', O, Os) / Lt - np.outer(vac, vac)
    return C


def gevp_effective_masses(C: np.ndarray, t0: int = 1, n_blocks: int = 50) -> dict:
    """Generalised eigenvalue problem C(t) v = lambda C(t0) v on the
    block-jackknife mean; effective masses of the lowest state per t with
    jackknife errors. Returns {'t': [...], 'm_eff': [...], 'err': [...]}."""
    ncfg = C.shape[0]
    nb = min(n_blocks, ncfg)
    blocks = np.array_split(np.arange(ncfg), nb)

    def lowest_masses(Cm):
        Lt = Cm.shape[-1]
        lam = []
        C0 = Cm[..., t0]
        # symmetrise and regularise
        w, V = np.linalg.eigh(0.5 * (C0 + C0.T))
        keep = w > 1e-12 * w.max()
        P = V[:, keep] / np.sqrt(w[keep])
        for t in range(Lt // 2 + 1):
            A = P.T @ (0.5 * (Cm[..., t] + Cm[..., t].T)) @ P
            ev = np.linalg.eigvalsh(A)
            lam.append(ev.max())
        lam = np.array(lam)
        m = np.full(len(lam) - 1, np.nan)
        for t in range(len(lam) - 1):
            if lam[t] > 0 and lam[t + 1] > 0 and lam[t + 1] < lam[t]:
                m[t] = np.log(lam[t] / lam[t + 1])
        return m

    full = lowest_masses(C.mean(axis=0))
    jk = []
    for b in blocks:
        Cm = np.delete(C, b, axis=0).mean(axis=0)
        jk.append(lowest_masses(Cm))
    jk = np.array(jk)
    err = np.sqrt((nb - 1) * np.nanvar(jk, axis=0))
    return {"t": list(range(len(full))), "m_eff": full.tolist(), "err": err.tolist()}


def plateau(m: dict, rel_err_max: float = 0.15) -> dict | None:
    """First pair of consecutive finite m_eff agreeing within errors."""
    ts, ms, es = m["t"], m["m_eff"], m["err"]
    for i in range(len(ms) - 1):
        a, b, ea, eb = ms[i], ms[i + 1], es[i], es[i + 1]
        if any(np.isnan([a, b, ea, eb])):
            continue
        if abs(a - b) < (ea + eb):
            mm = 0.5 * (a + b)
            ee = 0.5 * float(np.hypot(ea, eb))
            if mm > 0 and ee / mm < rel_err_max:
                return {"mass": float(mm), "err": ee, "t_start": ts[i]}
    return None


def readiness(run: dict, run2: dict | None = None, t0: int = 1) -> dict:
    """Gate status in the same form as glueball_channels_mc.benchmark_v3_readiness.
    Ratios are exposed only if every gate passes; comparison to the sealed
    targets is not performed here."""
    plats, meffs = {}, {}
    for ch in ("A1", "B1", "A2"):
        C = correlator_matrix(run["ops"][ch], vacuum_subtract=(ch == "A1"))
        meffs[ch] = gevp_effective_masses(C, t0=t0)
        plats[ch] = plateau(meffs[ch])
    gate1 = {ch: p is not None for ch, p in plats.items()}
    gate2 = {ch: (p is not None and p["err"] / p["mass"] < 0.15) for ch, p in plats.items()}
    gate3 = None
    if run2 is not None and plats["A1"] is not None:
        C2 = correlator_matrix(run2["ops"]["A1"], vacuum_subtract=True)
        p2 = plateau(gevp_effective_masses(C2, t0=t0))
        gate3 = (p2 is not None and abs(p2["mass"] - plats["A1"]["mass"])
                 < 2 * (p2["err"] + plats["A1"]["err"]))
    ready = all(gate1.values()) and all(gate2.values()) and (gate3 is True)
    out = {"n": run["n"], "beta": run["beta"], "beta_t": run["beta_t"],
           "Ls": run["Ls"], "Lt": run["Lt"], "n_cfg": run["n_cfg"],
           "plaq_mean": float(run["plaq"].mean()),
           "m_eff": meffs, "plateaus": plats,
           "gate1_plateau": gate1, "gate2_error": gate2, "gate3_volume": gate3,
           "ready_for_v3": bool(ready), "envelope": "SEALED"}
    if ready:                                        # pragma: no cover
        m0 = plats["A1"]["mass"]
        out["ratios_available"] = {"B1_over_A1": plats["B1"]["mass"] / m0,
                                   "A2_over_A1": plats["A2"]["mass"] / m0}
    return out
