"""Character-Wilson finite-group Monte Carlo, distinct from historical M2 H.

The D_n tables use the faithful E1 character (d=2) in the Euclidean action
    S = beta * sum_p (1 - Re chi_E1(U_p)/2)
on a periodic 3D lattice. Generic supplied tables instead define their own
character/table action beta * sum_p (1 - c[U_p]/d). No sampler substitutes
heat-kernel weights for the character array used by actions or observables.

This is NOT simulation of the original noncentral generator Hamiltonian in
``gauge_dynamics_m2_m3``, nor of the new central heat-kernel model. The Wilson
transfer has a different electric spectrum. No equality of Hamiltonians or
beta/lambda mapping (including monotonicity) has been established.

The measured quantities are mean plaquette and n_plaq * Var(mean plaquette).
The susceptibility peak is a finite-volume pseudo-critical beta, not proof
of the number/order of transitions or a physical substrate phase. Neither
the substrate coupling nor continuum/spatial particle matching is derived.
Historical scan arithmetic and defaults are retained unchanged.

The Z_2 literature anchor is beta_c ~= 0.7613. This validation tests the
Wilson sampler, not its equivalence to a proposed substrate Hamiltonian.
No glueball benchmark target appears here; the benchmark remains sealed.
"""
from __future__ import annotations

import numpy as np

from .nonabelian_gauge_sector import ALLOWED_CLASSES, elements, mul, inv

# ---------------------------------------------------------------------------
# Group tables (integer-indexed, so the MC inner loop is table lookups)
# ---------------------------------------------------------------------------


def dn_tables(n: int):
    """Multiplication/inverse/character tables for D_n, F = E1 (faithful 2-dim).

    Returns (M, I, c, d) with M[g,h] = gh, I[g] = g^-1,
    c[g] = Re chi_E1(g) = 2 cos(2 pi k / n) on rotations, 0 on reflections.
    """
    els = elements(n)
    idx = {g: i for i, g in enumerate(els)}
    G = len(els)
    M = np.empty((G, G), dtype=np.int64)
    I = np.empty(G, dtype=np.int64)
    for g in els:
        I[idx[g]] = idx[inv(g, n)]
        for h in els:
            M[idx[g], idx[h]] = idx[mul(g, h, n)]
    c = np.array([2.0 * np.cos(2.0 * np.pi * g[0] / n) if g[1] == 1 else 0.0
                  for g in els])
    return M, I, c, 2.0


def z2_tables():
    """Z_2 = {+1, -1}: the validation group (3D Ising gauge)."""
    M = np.array([[0, 1], [1, 0]], dtype=np.int64)
    I = np.array([0, 1], dtype=np.int64)
    c = np.array([1.0, -1.0])
    return M, I, c, 1.0


# ---------------------------------------------------------------------------
# 3D lattice Metropolis
# ---------------------------------------------------------------------------


def wilson_model_metadata() -> dict:
    """Generic provenance: custom tables do not establish D_n/E1 identity."""
    return {
        "model_id": "finite-group-character-wilson-v1",
        "action": "S = sum_p beta_p * (1 - c[U_p]/d)",
        "group": "supplied group tables",
        "representation": "supplied character/table values c and normalization d",
        "units": "dimensionless lattice action and couplings",
        "electric_transfer": "Wilson character weight; not generator heat kernel",
        "beta_lambda_mapping": "NOT ESTABLISHED",
        "physical_matching": "NOT ESTABLISHED",
    }


class WilsonMC:
    """Metropolis for a finite-group Wilson action on an L^3 periodic lattice.

    Links: U[x, y, z, mu] = group-element index. A link in direction mu sits
    in 4 plaquettes (2 orientations x 2 sides), and the local action change
    is computed from those staples only.
    """

    def __init__(self, tables, L: int = 4, beta: float = 1.0, seed: int = 0):
        self.M, self.I, self.c, self.d = tables
        self.model_metadata = wilson_model_metadata()
        self.G = len(self.I)
        self.L, self.beta = L, beta
        self.rng = np.random.default_rng(seed)
        self.U = np.zeros((L, L, L, 3), dtype=np.int64)   # cold start (all id)
        # identity must be index of (0, 1) — element lists put it first
        self._sites = [(x, y, z) for x in range(L) for y in range(L)
                       for z in range(L)]

    # -- geometry helpers ---------------------------------------------------

    def _shift(self, s, mu, k=1):
        s = list(s)
        s[mu] = (s[mu] + k) % self.L
        return tuple(s)

    def _plaq_flux(self, s, mu, nu):
        """U_mu(s) U_nu(s+mu) U_mu(s+nu)^-1 U_nu(s)^-1 (element index)."""
        M, I, U = self.M, self.I, self.U
        a = U[s + (mu,)]
        b = U[self._shift(s, mu) + (nu,)]
        cc = I[U[self._shift(s, nu) + (mu,)]]
        dd = I[U[s + (nu,)]]
        return M[M[M[a, b], cc], dd]

    def _local_action(self, s, mu):
        """Sum over the 4 plaquettes containing link (s, mu) of the action
        density beta (1 - Re chi(U_p)/d)."""
        tot = 0.0
        for nu in range(3):
            if nu == mu:
                continue
            tot += 1.0 - self.c[self._plaq_flux(s, mu, nu)] / self.d
            sm = self._shift(s, nu, -1)
            tot += 1.0 - self.c[self._plaq_flux(sm, nu, mu)] / self.d
        return self.beta * tot

    # -- MC -------------------------------------------------------------

    def sweep(self):
        L, rng = self.L, self.rng
        props = rng.integers(0, self.G, size=3 * L**3)
        accepts = rng.random(size=3 * L**3)
        i = 0
        for s in self._sites:
            for mu in range(3):
                old = self.U[s + (mu,)]
                new = props[i]
                if new != old:
                    s_old = self._local_action(s, mu)
                    self.U[s + (mu,)] = new
                    s_new = self._local_action(s, mu)
                    if s_new > s_old and accepts[i] > np.exp(s_old - s_new):
                        self.U[s + (mu,)] = old       # reject
                i += 1

    def mean_plaquette(self) -> float:
        tot, cnt = 0.0, 0
        for s in self._sites:
            for mu in range(3):
                for nu in range(mu + 1, 3):
                    tot += self.c[self._plaq_flux(s, mu, nu)] / self.d
                    cnt += 1
        return tot / cnt

    def run(self, n_equil: int = 150, n_meas: int = 300, stride: int = 2):
        for _ in range(n_equil):
            self.sweep()
        vals = []
        for k in range(n_meas):
            self.sweep()
            if k % stride == 0:
                vals.append(self.mean_plaquette())
        v = np.array(vals)
        n_plaq = 3 * self.L**3
        return {
            "plaq_mean": float(v.mean()),
            "plaq_susc": float(n_plaq * v.var()),
            "n_samples": len(vals),
            "model_metadata": dict(self.model_metadata),
        }


# ---------------------------------------------------------------------------
# beta scans and the pseudo-critical point
# ---------------------------------------------------------------------------


def scan(tables, betas, L: int = 4, seed: int = 0,
         n_equil: int = 150, n_meas: int = 300) -> list:
    out = []
    for i, b in enumerate(betas):
        mc = WilsonMC(tables, L=L, beta=float(b), seed=seed + 1000 * i)
        r = mc.run(n_equil=n_equil, n_meas=n_meas)
        r["beta"] = float(b)
        out.append(r)
    return out


def pseudo_critical(scan_results: list) -> dict:
    """beta at the susceptibility peak, with the scan spacing as resolution."""
    b = np.array([r["beta"] for r in scan_results])
    s = np.array([r["plaq_susc"] for r in scan_results])
    j = int(np.argmax(s))
    db = float(np.diff(b).max()) if len(b) > 1 else float("nan")
    return {"beta_c": float(b[j]), "resolution": db,
            "peak_susc": float(s[j]),
            "at_edge": bool(j in (0, len(b) - 1))}


def validate_z2(L: int = 4, seed: int = 7) -> dict:
    """Anchor: 3D Z_2 gauge theory, known bulk transition beta_c ~= 0.7613.
    The pseudo-critical peak at small L must land near it (within the coarse
    scan resolution + finite-size shift)."""
    betas = np.arange(0.55, 1.00, 0.05)
    res = scan(z2_tables(), betas, L=L, seed=seed)
    pc = pseudo_critical(res)
    pc["literature_beta_c"] = 0.7613
    pc["consistent"] = bool(abs(pc["beta_c"] - 0.7613) < 0.15)
    return pc


def locate_dn_transition(n: int, L: int = 4, seed: int = 11,
                         betas=None) -> dict:
    """Finite-volume susceptibility peak of the D_n E1 character-Wilson model."""
    if betas is None:
        betas = np.arange(0.6, 2.2, 0.1)
    res = scan(dn_tables(n), betas, L=L, seed=seed)
    pc = pseudo_critical(res)
    pc["n"] = n
    pc["group"] = f"D_{n}"
    pc["model_metadata"] = {
        **wilson_model_metadata(), "group": f"D_{n}", "representation": "E1 (d=2)"}
    pc["scan"] = [(r["beta"], round(r["plaq_mean"], 4),
                   round(r["plaq_susc"], 4)) for r in res]
    return pc


def report(L: int = 4) -> str:  # pragma: no cover
    lines = ["Character-Wilson MC — finite-volume susceptibility peaks",
             "=============================================================",
             ""]
    v = validate_z2(L=L)
    lines.append(f"Z_2 validation: peak at beta = {v['beta_c']:.2f} "
                 f"(literature 0.7613) -> consistent: {v['consistent']}")
    lines.append("")
    for n in ALLOWED_CLASSES:
        r = locate_dn_transition(n, L=L)
        lines.append(f"D_{n:<2d}: pseudo-critical beta_c = {r['beta_c']:.2f} "
                     f"+- {r['resolution']:.2f} (L={L}, susceptibility peak"
                     f"{', AT SCAN EDGE' if r['at_edge'] else ''})")
    lines += ["",
              "These peaks characterize this Wilson model only, not the "
              "historical generator Hamiltonian or the NEW heat-kernel model.",
              "No beta/lambda mapping is established. Finite-volume peaks alone "
              "do not establish a full phase diagram or the substrate phase."]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
