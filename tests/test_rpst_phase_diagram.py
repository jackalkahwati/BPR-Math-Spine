"""The 1-D Z_p clock ring has no finite-temperature transition, so phase_diagram reports its susceptibility
peak as a finite-size crossover (T_peak). T_c survives only as a backward-compatible alias."""

import numpy as np

from bpr.rpst.monte_carlo import phase_diagram


def test_phase_diagram_reports_a_crossover_not_a_critical_temperature():
    r = phase_diagram(p=7, n_sites=16, T_range=np.array([0.5, 1.0, 2.0]), n_samples=20, spacing=1, n_therm=20)
    assert r["T_peak"] in r["T_values"]
    assert r["T_c"] == r["T_peak"]
    assert "no finite-temperature transition" in r["note"] and "crossover" in r["note"]
    assert "finite-size crossover" in phase_diagram.__doc__
