"""Emergent U(1) gauge structure from hard-core link bosons (proposed amendment).

See doc/derivations/emergent_gauge_link_bosons_2026-09-25.md. Bosons live on
the links of a supplied simple graph, hop between links that share a vertex,
and pay U per unit squared deviation of each vertex charge from its target.
The low-energy sector is derived exactly to third order in t/U (triangle-free
graphs) and compared with exact diagonalization on small clusters. This is a
new model input, not a consequence of the site-boson substrate, and no
Coulomb (photon) phase is established by these finite checks.
"""

from itertools import combinations, product
from math import comb, isfinite

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

MODEL_ID = "hard-core-link-boson-emergent-u1-v1"
# Computational allocation caps, not physical bounds.
MAX_LINKS = 62
MAX_SECTOR_DIMENSION = 200000
MAX_ICE_DIMENSION = 20000
MAX_SEARCH_NODES = 5_000_000
DENSE_LIMIT = 800

LIMITATIONS = [
    "Bosons on links with a vertex charging term are a proposed amendment, not derived from the site-boson substrate.",
    "The effective gauge theory is derived exactly to second order (any simple graph) and third order (triangle-free graphs); higher orders are only checked numerically on small clusters.",
    "A deconfined Coulomb phase with a photon is not established here; on the cubic lattice it is literature-dependent and may require tuning toward the RK point.",
    "Only an Abelian U(1) gauge structure arises; no non-Abelian or Standard Model gauge group is supplied.",
    "Emergent charges are bosonic; no chiral fermions are supplied.",
    "Finite floating checks are not certified bounds or empirical validation.",
]


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

def _finite(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not isfinite(value):
            raise ValueError("numerical failure: nonfinite computed output")
        return
    if type(value) is list:
        for item in value:
            _finite(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("report keys must be strings")
            _finite(item)
        return
    raise TypeError("report contains a non-JSON-native value")


def make_graph(vertices, links, targets):
    """Validate a simple graph with per-vertex charge targets q_v."""
    vertices = list(vertices)
    index = {v: i for i, v in enumerate(vertices)}
    if len(index) != len(vertices):
        raise ValueError("duplicate vertices")
    seen = set()
    clean = []
    for a, b in links:
        if a == b or a not in index or b not in index:
            raise ValueError("links must join two distinct known vertices")
        key = frozenset((a, b))
        if key in seen:
            raise ValueError("multigraphs are not supported")
        seen.add(key)
        clean.append((index[a], index[b]))
    incident = [[] for _ in vertices]
    for l, (a, b) in enumerate(clean):
        incident[a].append(l)
        incident[b].append(l)
    if isinstance(targets, dict):
        q = [targets[v] for v in vertices]
    elif type(targets) is int:
        q = [targets] * len(vertices)
    else:
        raise TypeError("targets must be an int or a dict keyed by vertex")
    for qv, inc in zip(q, incident):
        if type(qv) is not int or not 0 <= qv <= len(inc):
            raise ValueError("charge targets must be ints between 0 and the degree")
    if sum(q) % 2:
        raise ValueError("the sum of charge targets must be even (each link has two ends)")
    graph = {"vertices": vertices, "links": clean, "incident": incident, "targets": q,
             "particles": sum(q) // 2}
    for component in _components(graph):
        sub = _two_coloring(graph, component)
        if sub is not None:
            side = [sum(q[v] for v in component if sub[v] == k) for k in (0, 1)]
            if side[0] != side[1]:
                raise ValueError("bipartite component: targets must sum equally on both sublattices")
        elif sum(q[v] for v in component) % 2:
            raise ValueError("each connected component needs an even target sum")
    return graph


def _components(graph):
    nbrs = [[] for _ in graph["vertices"]]
    for a, b in graph["links"]:
        nbrs[a].append(b)
        nbrs[b].append(a)
    seen = set()
    out = []
    for start in range(len(nbrs)):
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in nbrs[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        out.append(sorted(comp))
    return out


def _two_coloring(graph, restrict=None):
    color = {} if restrict is not None else [None] * len(graph["vertices"])
    nbrs = [[] for _ in graph["vertices"]]
    for a, b in graph["links"]:
        nbrs[a].append(b)
        nbrs[b].append(a)
    starts = restrict if restrict is not None else range(len(graph["vertices"]))
    if restrict is not None:
        for v in restrict:
            color[v] = None
    for start in starts:
        if color[start] is not None:
            continue
        color[start] = 0
        stack = [start]
        while stack:
            v = stack.pop()
            for w in nbrs[v]:
                if color[w] is None:
                    color[w] = 1 - color[v]
                    stack.append(w)
                elif color[w] == color[v]:
                    return None
    return color


def square_torus(n, q=2):
    if type(n) is not int or n < 3:
        raise ValueError("n must be an int >= 3")
    vertices = list(product(range(n), repeat=2))
    links = [((x, y), ((x + 1) % n, y)) for x, y in vertices] + \
            [((x, y), (x, (y + 1) % n)) for x, y in vertices]
    return make_graph(vertices, links, q)


def cubic_torus(n, q=3):
    if type(n) is not int or n < 3:
        raise ValueError("n must be an int >= 3")
    vertices = list(product(range(n), repeat=3))
    links = []
    for v in vertices:
        for axis in range(3):
            w = list(v)
            w[axis] = (w[axis] + 1) % n
            links.append((v, tuple(w)))
    return make_graph(vertices, links, q)


def open_cubes(count=1):
    """A row of `count` unit cubes (bipartite). Degree-3 vertices target 1, degree-4 target 2.

    On a bipartite graph the targets must sum equally over the two sublattices,
    because every link has one end on each; these choices satisfy that.
    """
    if type(count) is not int or count not in (1, 2):
        raise ValueError("count must be 1 or 2")
    vertices = list(product(range(count + 1), range(2), range(2)))
    vset = set(vertices)
    links = []
    for v in vertices:
        for axis in range(3):
            w = list(v)
            w[axis] += 1
            if tuple(w) in vset:
                links.append((v, tuple(w)))
    degree = {v: 0 for v in vertices}
    for a, b in links:
        degree[a] += 1
        degree[b] += 1
    targets = {v: 2 if degree[v] == 4 else 1 for v in vertices}
    return make_graph(vertices, links, targets)


def adamantane():
    """Diamond-lattice fragment: 4 bridgeheads, 6 bridges, 12 links, four hexagonal rings.

    Bridges target 1; bridgehead targets (2, 2, 1, 1) balance the two sublattices.
    """
    heads = [("B", i) for i in range(4)]
    bridges = [("M", i, j) for i in range(4) for j in range(i + 1, 4)]
    links = []
    for _, i, j in bridges:
        links.append((("B", i), ("M", i, j)))
        links.append((("B", j), ("M", i, j)))
    targets = {("B", 0): 2, ("B", 1): 2, ("B", 2): 1, ("B", 3): 1}
    targets.update({b: 1 for b in bridges})
    return make_graph(heads + bridges, links, targets)


def cycles_of_length(graph, length):
    """All simple cycles of the given length, as frozensets of link indices."""
    links = graph["links"]
    lookup = {frozenset(pair): l for l, pair in enumerate(links)}
    nbrs = [[] for _ in graph["vertices"]]
    for a, b in links:
        nbrs[a].append(b)
        nbrs[b].append(a)
    found = set()

    def walk(path):
        if len(path) == length:
            if path[0] in nbrs[path[-1]]:
                cyc = [lookup[frozenset((path[i], path[(i + 1) % length]))] for i in range(length)]
                found.add(frozenset(cyc))
            return
        for w in nbrs[path[-1]]:
            if w not in path and w > path[0]:
                walk(path + [w])

    for start in range(len(graph["vertices"])):
        walk([start])
    return sorted(found, key=lambda c: sorted(c))


def four_cycles(graph):
    """Each 4-cycle as a tuple of four link indices in cyclic order."""
    links = graph["links"]
    lookup = {frozenset(pair): l for l, pair in enumerate(links)}
    nbrs = [set() for _ in graph["vertices"]]
    for a, b in links:
        nbrs[a].add(b)
        nbrs[b].add(a)
    found = {}
    count = len(graph["vertices"])
    for v in range(count):
        for w in range(v + 1, count):
            common = sorted(nbrs[v] & nbrs[w])
            for a, b in combinations(common, 2):
                cycle = (lookup[frozenset((v, a))], lookup[frozenset((a, w))],
                         lookup[frozenset((w, b))], lookup[frozenset((b, v))])
                found[frozenset(cycle)] = cycle
    return [found[key] for key in sorted(found, key=lambda s: sorted(s))]


def is_bipartite(graph):
    return _two_coloring(graph) is not None


# ---------------------------------------------------------------------------
# Hilbert spaces and Hamiltonians
# ---------------------------------------------------------------------------

def _charges(mask, graph):
    return [sum((mask >> l) & 1 for l in inc) for inc in graph["incident"]]


def sector_basis(graph):
    """All hard-core configurations with the total particle number fixed by the targets."""
    L = len(graph["links"])
    N = graph["particles"]
    if L > MAX_LINKS:
        raise ValueError("too many links for the bitmask representation")
    if comb(L, N) > MAX_SECTOR_DIMENSION:
        raise ValueError("sector dimension exceeds the allocation cap")
    basis = []
    for occupied in combinations(range(L), N):
        mask = 0
        for l in occupied:
            mask |= 1 << l
        basis.append(mask)
    return basis


def ice_basis(graph):
    """Configurations with Q_v = q_v at every vertex (the constrained manifold)."""
    L = len(graph["links"])
    if L > MAX_LINKS:
        raise ValueError("too many links for the bitmask representation")
    incident = graph["incident"]
    targets = graph["targets"]
    order = list(range(L))
    ends = graph["links"]
    out = []
    remaining = [len(inc) for inc in incident]
    charge = [0] * len(incident)
    nodes = [0]

    def extend(position, mask):
        nodes[0] += 1
        if len(out) > MAX_ICE_DIMENSION or nodes[0] > MAX_SEARCH_NODES:
            raise ValueError("ice manifold search exceeds the allocation cap")
        if position == L:
            if charge == targets:
                out.append(mask)
            return
        l = order[position]
        a, b = ends[l]
        for occ in (0, 1):
            ok = True
            for v in (a, b):
                new = charge[v] + occ
                if new > targets[v] or new + remaining[v] - 1 < targets[v]:
                    ok = False
            if not ok:
                continue
            for v in (a, b):
                charge[v] += occ
                remaining[v] -= 1
            extend(position + 1, mask | (occ << l))
            for v in (a, b):
                charge[v] -= occ
                remaining[v] += 1

    extend(0, 0)
    return sorted(out)


def _hops(graph):
    """Ordered (from_link, to_link) pairs sharing a vertex; each appears once."""
    pairs = []
    for inc in graph["incident"]:
        for a in inc:
            for b in inc:
                if a != b:
                    pairs.append((a, b))
    return pairs


def _hopping_matrix(graph, basis):
    """Sparse V/(-t): sum over ordered link pairs sharing a vertex of b_to^dagger b_from."""
    masks = np.array(basis, dtype=np.int64)
    order = np.argsort(masks)
    sorted_masks = masks[order]
    rows, cols = [], []
    for src, dst in _hops(graph):
        sel = np.nonzero(((masks >> src) & 1 == 1) & ((masks >> dst) & 1 == 0))[0]
        new = masks[sel] ^ (np.int64(1) << src) ^ (np.int64(1) << dst)
        pos = np.searchsorted(sorted_masks, new)
        rows.append(order[pos])
        cols.append(sel)
    rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    cols = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    n = len(basis)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def _charge_array(graph, basis):
    masks = np.array(basis, dtype=np.int64)
    out = np.zeros((len(basis), len(graph["vertices"])), dtype=np.int64)
    for v, inc in enumerate(graph["incident"]):
        for l in inc:
            out[:, v] += (masks >> l) & 1
    return out


def full_hamiltonian(graph, t, U):
    """H = U sum_v (Q_v-q_v)^2 - t sum_v sum_{l != l' at v} b_l^dagger b_l' in the N sector."""
    t, U = float(t), float(U)
    basis = sector_basis(graph)
    charges = _charge_array(graph, basis)
    diag = U * np.sum((charges - np.array(graph["targets"])) ** 2, axis=1)
    H = sparse.diags(diag.astype(float)) - t * _hopping_matrix(graph, basis)
    return basis, H.tocsr()


def schrieffer_wolff(graph, t, U):
    """Numerical second- and third-order effective Hamiltonians on the ice manifold.

    With P0 V P0 = 0: H2 = P V R V P and H3 = P V R V R V P, R = -Q H0^{-1} Q.
    """
    t, U = float(t), float(U)
    basis = sector_basis(graph)
    charges = _charge_array(graph, basis)
    h0 = U * np.sum((charges - np.array(graph["targets"])) ** 2, axis=1).astype(float)
    ice = np.nonzero(h0 == 0)[0]
    if len(ice) == 0:
        raise ValueError("empty ice manifold: no configuration meets every vertex target")
    V = -t * _hopping_matrix(graph, basis)
    VP = V[:, ice]
    if abs(VP[ice, :]).max() != 0:
        raise ArithmeticError("P0 V P0 must vanish")
    inv = np.where(h0 > 0, -1.0 / np.where(h0 > 0, h0, 1.0), 0.0)
    R = sparse.diags(inv)
    RVP = R @ VP
    H2 = (VP.T @ RVP).toarray()
    H3 = (RVP.T @ (V @ RVP)).toarray()
    ice_masks = [basis[i] for i in ice]
    return ice_masks, H2, H3


def gauss_commutator_norms(graph, t, U):
    """Largest entry of [H, Q_v] for each vertex; nonzero for t != 0 (no exact Gauss law)."""
    basis, H = full_hamiltonian(graph, t, U)
    charges = _charge_array(graph, basis).astype(float)
    out = []
    for v in range(len(graph["vertices"])):
        Qv = sparse.diags(charges[:, v])
        diff = (H @ Qv - Qv @ H)
        out.append(float(abs(diff).max()) if diff.nnz else 0.0)
    return out


def effective_constant(graph, t, U):
    """Second-order diagonal: -(t^2/2U) sum_v q_v (d_v - q_v)."""
    return -(float(t) ** 2 / (2 * float(U))) * sum(
        q * (len(inc) - q) for q, inc in zip(graph["targets"], graph["incident"]))


def ring_exchange_coefficient(t, U):
    """K = 2 t^2 / U: two vertex pairs per plaquette, two time orderings each."""
    return 2 * float(t) ** 2 / float(U)


def is_triangle_free(graph):
    nbrs = [set() for _ in graph["vertices"]]
    for a, b in graph["links"]:
        nbrs[a].add(b)
        nbrs[b].add(a)
    return all(not (nbrs[a] & nbrs[b]) for a, b in graph["links"])


def third_order_constant(graph, t, U):
    """-(t^3/4U^2) sum_v q_v e_v (d_v - 2), e_v = d_v - q_v; valid on triangle-free graphs."""
    t, U = float(t), float(U)
    total = sum(q * (len(inc) - q) * (len(inc) - 2) for q, inc in zip(graph["targets"], graph["incident"]))
    return -(t ** 3 / (4 * U ** 2)) * total


def third_order_plaquette_shift(graph, cycle, t, U):
    """delta K = (3 t^3 / 4U^2) sum over the four corners of (d_c - 2) (detour and push paths)."""
    t, U = float(t), float(U)
    corners = set()
    for l in cycle:
        corners.update(graph["links"][l])
    return (3 * t ** 3 / (4 * U ** 2)) * sum(len(graph["incident"][c]) - 2 for c in corners)


def hexagon_coefficient(t, U):
    """K_6 = 3 t^3 / U^2: two pivot sets, 3! orderings, intermediate energies 2U."""
    return 3 * float(t) ** 3 / float(U) ** 2


def alternating(mask, cycle_links_in_order):
    bits = [(mask >> l) & 1 for l in cycle_links_in_order]
    return all(bits[i] != bits[(i + 1) % len(bits)] for i in range(len(bits)))


def _ordered_cycle(graph, cycle_set):
    """Order a simple cycle's links so consecutive links share a vertex.

    In a simple cycle every vertex lies on exactly two cycle links, so link
    adjacency inside the cycle is exactly cyclic succession.
    """
    links = list(cycle_set)
    ends = {l: set(graph["links"][l]) for l in links}
    ordered = [links[0]]
    previous = None
    while len(ordered) < len(links):
        current = ordered[-1]
        nxt = [l for l in links if l != current and l != previous and ends[l] & ends[current]]
        if not nxt:
            raise ArithmeticError("cycle ordering failed")
        previous = current
        ordered.append(nxt[0])
    return ordered


def analytic_third_order(graph, t, U):
    """Analytic third-order effective Hamiltonian on the ice manifold (triangle-free graphs)."""
    if not is_triangle_free(graph):
        raise ValueError("the analytic third-order formula assumes a triangle-free graph")
    basis = ice_basis(graph)
    index = {m: i for i, m in enumerate(basis)}
    squares = four_cycles(graph)
    hexagons = [_ordered_cycle(graph, h) for h in cycles_of_length(graph, 6)]
    C3 = third_order_constant(graph, t, U)
    K6 = hexagon_coefficient(t, U)
    H = np.zeros((len(basis), len(basis)))
    for col, mask in enumerate(basis):
        H[col, col] += C3
        for cycle in squares:
            if flippable(mask, cycle):
                flip = mask
                for l in cycle:
                    flip ^= 1 << l
                H[index[flip], col] -= third_order_plaquette_shift(graph, cycle, t, U)
        for cycle in hexagons:
            if alternating(mask, cycle):
                flip = mask
                for l in cycle:
                    flip ^= 1 << l
                H[index[flip], col] -= K6
    return basis, H


def flippable(mask, cycle):
    bits = [(mask >> l) & 1 for l in cycle]
    return bits in ([1, 0, 1, 0], [0, 1, 0, 1])


def effective_hamiltonian(graph, t, U, rk=0.0):
    """H_eff = C - K sum_p (F_p + F_p^dagger) + rk * sum_p P_flippable on the ice manifold."""
    basis = ice_basis(graph)
    if not basis:
        raise ValueError("empty ice manifold: no configuration meets every vertex target")
    index = {m: i for i, m in enumerate(basis)}
    cycles = four_cycles(graph)
    K = ring_exchange_coefficient(t, U)
    C = effective_constant(graph, t, U)
    rows, cols, vals = [], [], []
    for col, mask in enumerate(basis):
        diag = C
        for cycle in cycles:
            if flippable(mask, cycle):
                diag += float(rk)
                flip = mask
                for l in cycle:
                    flip ^= 1 << l
                rows.append(index[flip])
                cols.append(col)
                vals.append(-K)
        rows.append(col)
        cols.append(col)
        vals.append(diag)
    H = sparse.csr_matrix((vals, (rows, cols)), shape=(len(basis), len(basis)))
    return basis, H


def _lowest(H, count):
    n = H.shape[0]
    if n <= DENSE_LIMIT:
        return np.sort(np.linalg.eigvalsh(H.toarray()))[:count]
    w = sparse_linalg.eigsh(H, k=min(count, n - 1), which="SA", tol=1e-12, maxiter=200000)[0]
    return np.sort(w)


def _lowest_full(graph, t, U, count, extra=8, tol=1e-10, maxiter=500):
    """Lowest levels of the full Hamiltonian, robust to degenerate multiplets.

    Single-vector Lanczos (eigsh) can miss partners of a degenerate multiplet, and which one it
    misses depends on ARPACK's process-global start vector. Instead: block LOBPCG seeded with the
    third-order effective eigenvectors embedded in the full basis plus their first-order
    Schrieffer-Wolff correction (psi + R V psi), block size count + extra, diagonal preconditioner,
    and a residual check.
    """
    basis, H = full_hamiltonian(graph, t, U)
    n = H.shape[0]
    if n <= DENSE_LIMIT:
        return np.sort(np.linalg.eigvalsh(H.toarray()))[:count]
    t, U = float(t), float(U)
    charges = _charge_array(graph, basis)
    h0 = U * np.sum((charges - np.array(graph["targets"])) ** 2, axis=1).astype(float)
    ice = np.nonzero(h0 == 0)[0]
    _, H2, H3 = schrieffer_wolff(graph, t, U)
    m = min(len(ice), count + extra)
    _, vecs = np.linalg.eigh(H2 + H3)
    embed = np.zeros((n, m))
    embed[ice, :] = vecs[:, :m]
    V = -t * _hopping_matrix(graph, basis)
    inv = np.where(h0 > 0, -1.0 / np.where(h0 > 0, h0, 1.0), 0.0)
    X0 = embed + inv[:, None] * (V @ embed)
    d = H.diagonal()
    precond = sparse.diags(1.0 / (d - d.min() + 1.0))
    vals, X = sparse_linalg.lobpcg(H, X0, M=precond, largest=False, tol=tol, maxiter=maxiter)
    order = np.argsort(vals)
    vals, X = vals[order], X[:, order]
    residual = np.linalg.norm(H @ X - X * vals[None, :], axis=0) / np.maximum(np.abs(vals), 1.0)
    if residual[:count].max() > 1e-8:
        raise ArithmeticError("LOBPCG did not converge on the lowest levels")
    return vals[:count]


def compare_low_energy(graph, U=1.0, ratios=(0.01, 0.02), levels=4):
    """Full-model levels versus second-order (analytic) and second+third-order (numerical) models.

    Second-order check: Richardson extrapolation of E_full/t^2 to t=0 against the
    analytic second-order spectrum divided by t^2. Third-order check: residual
    after adding the numerical third-order term should scale as t^4.
    """
    if len(ratios) != 2 or not abs(ratios[1] - 2 * ratios[0]) < 1e-15 * max(1.0, ratios[1]):
        raise ValueError("ratios must be (r, 2r) for the Richardson and scaling checks")
    records = []
    scaled_full = []
    eff_scaled = None
    for ratio in ratios:
        t = ratio * U
        _, Heff = effective_hamiltonian(graph, t, U)
        _, H2, H3 = schrieffer_wolff(graph, t, U)
        k = min(levels, Heff.shape[0])
        full = _lowest_full(graph, t, U, k)
        eff = _lowest(Heff, k)
        numeric2 = np.sort(np.linalg.eigvalsh(H2))[:k]
        third = np.sort(np.linalg.eigvalsh(H2 + H3))[:k]
        records.append({
            "t_over_U": ratio, "full": [float(x) for x in full], "second_order": [float(x) for x in eff],
            "analytic_vs_numeric_second_order": float(np.max(np.abs(eff - numeric2))),
            "second_order_error": float(np.max(np.abs(full - eff))),
            "third_order_error": float(np.max(np.abs(full - third))),
        })
        scaled_full.append(full / t ** 2)
        eff_scaled = eff / t ** 2
    extrapolated = 2 * scaled_full[0] - scaled_full[1]
    summary = {
        "records": records,
        "richardson_second_order_gap": float(np.max(np.abs(extrapolated - eff_scaled))),
        "third_order_error_ratio": records[1]["third_order_error"] / max(records[0]["third_order_error"], 1e-300),
        "second_order_error_ratio": records[1]["second_order_error"] / max(records[0]["second_order_error"], 1e-300),
    }
    return summary


def rk_point_report(graph, K=1.0):
    """At rk = K the RK Hamiltonian is a sum of projectors; check its ground states."""
    t = (K / 2) ** 0.5  # so that 2 t^2 / U = K with U = 1
    basis, H = effective_hamiltonian(graph, t, 1.0, rk=K)
    C = effective_constant(graph, t, 1.0)
    H = H - sparse.identity(H.shape[0]) * C
    dense = H.toarray() if H.shape[0] <= 4000 else None
    if dense is None:
        raise ValueError("ice manifold too large for the dense RK check")
    w, v = np.linalg.eigh(dense)
    # flip-connected components
    cycles = four_cycles(graph)
    index = {m: i for i, m in enumerate(basis)}
    parent = list(range(len(basis)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, mask in enumerate(basis):
        for cycle in cycles:
            if flippable(mask, cycle):
                flip = mask
                for l in cycle:
                    flip ^= 1 << l
                a, b = find(i), find(index[flip])
                if a != b:
                    parent[a] = b
    components = {}
    for i in range(len(basis)):
        components.setdefault(find(i), []).append(i)
    zero_modes = int(np.sum(np.abs(w) < 1e-9))
    residual = 0.0
    for members in components.values():
        vec = np.zeros(len(basis))
        vec[members] = 1 / np.sqrt(len(members))
        residual = max(residual, float(np.linalg.norm(dense @ vec)))
    result = {"ice_dimension": len(basis), "components": len(components),
              "zero_energy_states": zero_modes, "min_eigenvalue": float(w[0]),
              "equal_superposition_residual": residual}
    _finite(result)
    return result


def cluster_summary(name, graph):
    cycles = four_cycles(graph)
    ice = ice_basis(graph)
    return {"name": name, "vertices": len(graph["vertices"]), "links": len(graph["links"]),
            "particles": graph["particles"], "sector_dimension": comb(len(graph["links"]), graph["particles"]),
            "ice_dimension": len(ice), "four_cycles": len(cycles), "bipartite": is_bipartite(graph),
            "flippable_counts": sorted({sum(flippable(m, c) for c in cycles) for m in ice})}


def demonstration_report():
    clusters = [("square_torus_3", square_torus(3)), ("open_cube", open_cubes(1)),
                ("two_open_cubes", open_cubes(2)), ("adamantane", adamantane())]
    summaries = []
    comparisons = []
    for name, graph in clusters:
        summaries.append(cluster_summary(name, graph))
        comparisons.append(dict(compare_low_energy(graph), name=name))
    commutator = gauss_commutator_norms(open_cubes(1), 0.1, 1.0)
    rk = [dict(rk_point_report(graph), name=name) for name, graph in clusters]
    report = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "status": "proposed_amendment_third_order_gauge_structure",
        "empirical_validation": False,
        "coulomb_phase_established": False,
        "controls": {"U": 1.0, "t_over_U": [0.01, 0.02], "levels": 4},
        "clusters": summaries,
        "low_energy_comparison": comparisons,
        "gauss_commutator_norms_open_cube_t_0_1": commutator,
        "rk_point": rk,
        "cubic_lattice_effective_couplings": {
            "plaquette_K_second_order_over_t2_U": 2.0,
            "plaquette_delta_K_third_order_over_t3_U2": 12.0,
            "hexagon_K6_third_order_over_t3_U2": 3.0,
            "constant_per_vertex_second_order_over_t2_U": -4.5,
            "constant_per_vertex_third_order_over_t3_U2": -9.0,
            "note": "q=3 of 6 links per vertex; all link pairs at a vertex hop.",
        },
        "diamond_lattice_effective_couplings": {
            "plaquette_K": 0.0,
            "hexagon_K6_third_order_over_t3_U2": 3.0,
            "constant_per_vertex_second_order_over_t2_U": -2.0,
            "constant_per_vertex_third_order_over_t3_U2": -2.0,
            "note": "q=2 of 4; no 4-cycles; equals pyrochlore hard-core bosons with NN repulsion V=2U.",
        },
        "limitations": list(LIMITATIONS),
    }
    _finite(report)
    return report
