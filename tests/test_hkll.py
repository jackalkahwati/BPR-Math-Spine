import numpy as np


def test_hkll_boundary_limit_smoke():
    from bpr.hkll import HKLLReconstruction, BulkPoint

    R = 1.0
    hkll = HKLLReconstruction(boundary_radius=R, l_max=6)

    def boundary_field(theta, phi, t):
        return float(np.cos(theta))

    bulk_near = BulkPoint(t=0.0, r=0.99 * R, theta=np.pi / 4, phi=0.0)
    Psi = hkll.reconstruct_bulk_field(bulk_near, boundary_field, n_theta=12, n_phi=24)
    target = boundary_field(np.pi / 4, 0.0, 0.0)
    assert np.isfinite(np.real(Psi))
    # Loose check: same order of magnitude
    assert abs(Psi) < 10.0
    assert abs(Psi - target) < 5.0


def test_hkll_metric_perturbation_symmetric():
    from bpr.hkll import HKLLReconstruction, BulkPoint

    R = 1.0
    hkll = HKLLReconstruction(boundary_radius=R, l_max=4)

    def boundary_field(theta, phi, t):
        return float(np.cos(theta))

    pt = BulkPoint(t=0.0, r=0.5 * R, theta=np.pi / 3, phi=np.pi / 4)
    dg = hkll.reconstruct_metric_perturbation(
        pt, boundary_field, coupling_lambda=1e-3, n_theta=8, n_phi=16, eps=5e-3
    )
    assert dg.shape == (4, 4)
    assert np.allclose(dg, dg.T)


def test_projection_induced_metric_sphere():
    from bpr.hkll import BoundaryToBulkProjection

    R = 2.0
    proj = BoundaryToBulkProjection(boundary_radius=R)
    theta = np.pi / 3
    h = proj.induced_metric(theta, 0.0)
    expected = np.array([[R**2, 0.0], [0.0, R**2 * np.sin(theta) ** 2]])
    assert np.allclose(h, expected, rtol=1e-10, atol=1e-10)


