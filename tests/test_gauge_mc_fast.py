"""Lock the vectorized Wilson MC against the reference implementation."""
import numpy as np
from pathlib import Path

from bpr.gauge_phase_mc import WilsonMC, dn_tables, z2_tables
from bpr.gauge_mc_fast import (
    FastWilsonMC, rect, l_loop, mirror, reflect_x, rotate90, channel_basis,
    polyomino_loop, CHIRAL_SHAPES, parity_odd_terms,
    slice_operators, correlator_matrix, gevp_effective_masses, plateau, measure,
)

BPR_DIR = Path(__file__).resolve().parent.parent / "bpr"


def _closed(steps):
    """Forward steps start at the link base; backward steps end there."""
    x = y = 0
    for (dx, dy), mu, sign in steps:
        if sign == 1:
            assert (dx, dy) == (x, y), "loop steps must be contiguous"
            x, y = (x + 1, y) if mu == 0 else (x, y + 1)
        else:
            x, y = (x - 1, y) if mu == 0 else (x, y - 1)
            assert (dx, dy) == (x, y), "loop steps must be contiguous"
    return (x, y) == (0, 0)


def test_loops_are_closed_and_contiguous():
    for lx, ly in ((1, 1), (2, 1), (1, 2), (2, 2), (3, 1)):
        assert _closed(rect(lx, ly))
    for a, b in ((2, 1), (3, 1), (3, 2)):
        assert _closed(l_loop(a, b))
        assert _closed(reflect_x(l_loop(a, b)))
        assert _closed(rotate90(l_loop(a, b)))
    assert len(l_loop(2, 1)) == 8
    for k, cells in CHIRAL_SHAPES.items():
        loop = polyomino_loop(cells)
        assert _closed(loop), k
        assert _closed(reflect_x(loop)), k
    assert len(polyomino_loop(CHIRAL_SHAPES["L4"])) == 10
    assert polyomino_loop(((0, 0),)) == rect(1, 1)


def test_achiral_shape_gives_vanishing_a2():
    """Documented pitfall: the notched square is achiral, so its parity-odd
    combination is identically zero for real characters."""
    mc = FastWilsonMC(dn_tables(5), Ls=6, Lt=4, beta=1.8, seed=9)
    for _ in range(60):
        mc.sweep()
    f = sum(w * mc.loop_char(loop) for w, loop in parity_odd_terms(l_loop(2, 1)))
    assert np.abs(f).max() < 1e-12


def test_reflection_and_rotation_are_symmetries():
    s = l_loop(2, 1)
    assert reflect_x(reflect_x(s)) == s
    r = s
    for _ in range(4):
        r = rotate90(r)
    assert r == s


def _gauge_transform(mc, rng):
    M, I = mc.M, mc.I
    g = rng.integers(0, mc.G, size=mc.U.shape[:3])
    U2 = mc.U.copy()
    for mu in range(3):
        g_shift = np.roll(g, -1, axis=mu)
        U2[..., mu] = M[M[g, mc.U[..., mu]], I[g_shift]]
    mc.U = U2


def _reflect_config_x(mc):
    """x -> -x on the link field."""
    M, I = mc.M, mc.I
    U = mc.U
    U2 = U.copy()
    # site x -> (-x) mod L. x-link based at x (x -> x+1) maps to the link
    # from -x to -x-1, i.e. the x-link based at (-x-1) mod L walked backward.
    U2[..., 0] = I[U[::-1, :, :, 0]]                 # index x -> L-1-x = (-x-1) mod L
    U2[..., 1] = np.roll(U[::-1, :, :, 1], 1, axis=0)  # index x -> (-x) mod L
    U2[..., 2] = np.roll(U[::-1, :, :, 2], 1, axis=0)
    mc.U = U2


def test_operators_gauge_invariant():
    mc = FastWilsonMC(dn_tables(5), Ls=6, Lt=4, beta=1.8, seed=9)
    for _ in range(100):
        mc.sweep()
    basis = channel_basis(3)
    before = slice_operators(mc, basis)
    _gauge_transform(mc, np.random.default_rng(1))
    after = slice_operators(mc, basis)
    for ch in basis:
        assert np.allclose(before[ch], after[ch], atol=1e-12), ch


def test_a2_is_parity_odd_and_nonzero():
    mc = FastWilsonMC(dn_tables(5), Ls=6, Lt=4, beta=1.8, seed=9)
    for _ in range(100):
        mc.sweep()
    basis = channel_basis(3)
    before = slice_operators(mc, basis)
    assert np.abs(before["A2"]).max() > 1e-6
    _reflect_config_x(mc)
    after = slice_operators(mc, basis)
    assert np.allclose(after["A2"], -before["A2"], atol=1e-12)
    assert np.allclose(after["A1"], before["A1"], atol=1e-12)
    assert np.allclose(after["B1"], before["B1"], atol=1e-12)


def test_cold_configuration_operator_symmetry():
    mc = FastWilsonMC(dn_tables(5), Ls=6, Lt=4, beta=1.0, seed=0)
    ops = slice_operators(mc, channel_basis(3))
    assert np.allclose(ops["A1"][0], 1.0)          # single plaquette, chi/d = 1
    assert np.allclose(ops["B1"], 0.0)
    assert np.allclose(ops["A2"], 0.0)


def test_matches_reference_plaquette():
    """Fast checkerboard MC reproduces WilsonMC's mean plaquette (Z2 and D5)."""
    for tab, beta in ((z2_tables(), 0.6), (dn_tables(5), 1.5)):
        ref = WilsonMC(tab, L=4, beta=beta, seed=2).run(n_equil=150, n_meas=400, stride=2)
        fast = FastWilsonMC(tab, Ls=4, Lt=4, beta=beta, seed=3)
        for _ in range(150):
            fast.sweep()
        vals = []
        for k in range(400):
            fast.sweep()
            if k % 2 == 0:
                vals.append(fast.mean_plaquette())
        v = np.array(vals)
        err = v.std() / np.sqrt(len(v) / 5)          # crude autocorrelation allowance
        assert abs(v.mean() - ref["plaq_mean"]) < 4 * err + 0.01


def test_gevp_pipeline_runs_and_gates_honest():
    mc = FastWilsonMC(dn_tables(5), Ls=4, Lt=6, beta=1.0, seed=1)
    for _ in range(50):
        mc.sweep()
    basis = channel_basis(2)
    O = np.array([slice_operators(mc, basis)["A1"] for _ in range(30) if not mc.sweep()])
    C = correlator_matrix(O, vacuum_subtract=True)
    m = gevp_effective_masses(C, t0=1, n_blocks=10)
    assert len(m["m_eff"]) == 6 // 2
    p = plateau(m)
    assert p is None or p["mass"] > 0


def test_fast_sampler_metadata_stays_generic_for_supplied_tables():
    for tables in (dn_tables(12), z2_tables()):
        mc = FastWilsonMC(tables, Ls=2, Lt=2, seed=0)
        metadata = mc.model_metadata
        assert metadata["model_id"] == "finite-group-character-wilson-v1"
        assert metadata["group"] == "supplied group tables"
        assert metadata["representation"] == "supplied character/table values c and normalization d"
        assert metadata["beta_lambda_mapping"] == "NOT ESTABLISHED"
        assert metadata["physical_matching"] == "NOT ESTABLISHED"
        np.testing.assert_array_equal(mc.c, tables[2])
        assert mc.d == tables[3]


def test_measure_metadata_preserves_legacy_schema_and_custom_identity():
    """The legacy n label must not misidentify explicitly supplied Z2 tables."""
    for tables in (None, z2_tables()):
        result = measure(12, beta=1.0, Ls=2, Lt=2, seed=0,
                         n_equil=0, n_meas=1, stride=1, n_ops=1, tables=tables)
        assert set(result) == {
            "n", "beta", "beta_t", "Ls", "Lt", "n_cfg", "plaq", "ops",
            "model_metadata",
        }
        assert result["n"] == 12
        assert result["beta"] == result["beta_t"] == 1.0
        assert result["Ls"] == result["Lt"] == 2
        assert result["n_cfg"] == 1
        assert result["plaq"].shape == (1,)
        assert np.isfinite(result["plaq"]).all()
        assert set(result["ops"]) == {"A1", "B1", "A2"}
        for values in result["ops"].values():
            assert values.shape == (1, 1, 2)
            assert np.isfinite(values).all()
        metadata = result["model_metadata"]
        assert metadata["model_id"] == "finite-group-character-wilson-v1"
        assert metadata["beta_lambda_mapping"] == "NOT ESTABLISHED"
        if tables is None:
            assert metadata["group"] == "D_12"
            assert metadata["representation"] == "E1 (d=2)"
        else:
            assert metadata["group"] == "supplied group tables"
            assert metadata["representation"] == "supplied character/table values c and normalization d"


def test_fast_mc_blind_to_glueball_targets():
    src = (BPR_DIR / "gauge_mc_fast.py").read_text()
    for leaked in ("1730", "2400", "2590", "1.387", "1.497", "2370", "2359",
                   "X(2370)", "Morningstar", "BESIII"):
        assert leaked not in src
