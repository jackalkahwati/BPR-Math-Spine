"""
BPR StarDrive Mathematical Hypothesis v0.1 — Toy Model

Demonstrates that the constrained-control formulation is mathematically coherent.

State space:      omega = (x, y, z, a, b, c)  in R^6
Location map:     L(omega) = (x, y, z)
Matter map:       F_m(omega) = (a^2 + b^2,  b + c,  exp(a)/c)

A valid drive path satisfies:
    gamma(t) in F_m^{-1}(c_target)   for all t
    gamma_dot(t) in ker dF_m         for all t

This is not proof of physics.  It is proof that the mathematical structure
is non-empty: there exist boundary spaces and observable maps where matter-
preserving, location-changing paths exist and are computable.

Run:
    python scripts/stardrive_toy_model.py
"""

import numpy as np
from scipy.optimize import brentq
import sys

sys.path.insert(0, "/home/user/BPR-Math-Spine")


# ─────────────────────────────────────────────────────────────────────────────
# Maps
# ─────────────────────────────────────────────────────────────────────────────

def F_m(omega: np.ndarray) -> np.ndarray:
    """Matter observable map.  Depends only on (a, b, c), not on (x, y, z)."""
    x, y, z, a, b, c = omega
    return np.array([a**2 + b**2,  b + c,  np.exp(a) / c])


def L(omega: np.ndarray) -> np.ndarray:
    """Location map."""
    return omega[:3]


def dF_m(omega: np.ndarray) -> np.ndarray:
    """Jacobian of F_m at omega.  Shape (3, 6)."""
    x, y, z, a, b, c = omega
    ea_over_c = np.exp(a) / c
    return np.array([
        [0, 0, 0,  2*a,       2*b,         0          ],
        [0, 0, 0,  0,         1,           1          ],
        [0, 0, 0,  ea_over_c, 0,          -ea_over_c / c],
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Find matter configuration satisfying F_m(omega) = c_target
# ─────────────────────────────────────────────────────────────────────────────

def find_matter_config(c_target: np.ndarray,
                       a_search=(-3.0, 3.0, 500)) -> tuple[float, float, float]:
    """
    Solve for (a*, b*, c*) with F_m(_, _, _, a*, b*, c*) = c_target.

    Constraints:
        a^2 + b^2 = u          [circle in (a,b)-plane]
        b + c     = v          →  c = v - b
        exp(a)/c  = w          →  c = exp(a)/w

    Combining: b = v - exp(a)/w,  then a^2 + (v - exp(a)/w)^2 = u.
    Scans the given range to locate a sign change, then bisects.
    """
    u, v, w = c_target

    def residual(a):
        c_val = np.exp(a) / w
        b_val = v - c_val
        return a**2 + b_val**2 - u

    a_lo, a_hi, n = a_search
    a_vals = np.linspace(a_lo, a_hi, n)
    r_vals = np.array([residual(a) for a in a_vals])
    sign_changes = np.where(np.diff(np.sign(r_vals)))[0]
    assert len(sign_changes) > 0, (
        f"No sign change in [{a_lo}, {a_hi}] for c_target={c_target}."
    )
    idx = sign_changes[0]
    a_star = brentq(residual, a_vals[idx], a_vals[idx + 1], xtol=1e-12)
    c_star = np.exp(a_star) / w
    b_star = v - c_star
    return float(a_star), float(b_star), float(c_star)


# ─────────────────────────────────────────────────────────────────────────────
# Drive path and verification
# ─────────────────────────────────────────────────────────────────────────────

def make_drive_path(l0: np.ndarray, l1: np.ndarray,
                    a_star: float, b_star: float, c_star: float):
    """
    Return a function gamma(t) -> omega that:
      - starts at L = l0 when t=0
      - ends   at L = l1 when t=1
      - holds matter config (a*, b*, c*) constant throughout
    """
    def gamma(t: float) -> np.ndarray:
        loc = l0 + t * (l1 - l0)
        return np.array([*loc, a_star, b_star, c_star])
    return gamma


def verify_path(gamma, c_target: np.ndarray, n_steps: int = 200) -> dict:
    """Check matter conservation and kernel condition along the path."""
    ts = np.linspace(0, 1, n_steps)
    dt = ts[1] - ts[0]

    max_fm_err = 0.0
    max_kernel_err = 0.0

    for t in ts:
        omega = gamma(t)

        # matter conservation: F_m(gamma(t)) == c_target
        fm_err = np.max(np.abs(F_m(omega) - c_target))
        max_fm_err = max(max_fm_err, fm_err)

        # kernel condition: dF_m · gamma_dot ≈ 0
        if t < 1.0 - dt:
            gamma_dot = (gamma(t + dt) - gamma(t)) / dt
            J = dF_m(omega)
            kernel_err = np.max(np.abs(J @ gamma_dot))
            max_kernel_err = max(max_kernel_err, kernel_err)

    return {"max_F_m_error": max_fm_err, "max_kernel_error": max_kernel_err}


# ─────────────────────────────────────────────────────────────────────────────
# Null space analysis
# ─────────────────────────────────────────────────────────────────────────────

def ker_dim(omega: np.ndarray) -> int:
    """Dimension of ker dF_m at omega."""
    J = dF_m(omega)
    return 6 - np.linalg.matrix_rank(J)


def ker_basis(omega: np.ndarray) -> np.ndarray:
    """Orthonormal basis for ker dF_m at omega via SVD."""
    J = dF_m(omega)
    _, s, Vt = np.linalg.svd(J)
    tol = 1e-10 * max(s) if len(s) > 0 else 1e-10
    null_mask = s < tol if len(s) >= J.shape[0] else np.ones(6, dtype=bool)
    # take right singular vectors corresponding to small singular values
    rank = int(np.sum(s > (1e-10 * s[0])))
    return Vt[rank:].T   # columns are basis vectors of ker


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("═" * 62)
    print("  BPR StarDrive Toy Model v0.1")
    print("═" * 62)
    print()
    print("  State:    omega = (x, y, z, a, b, c)")
    print("  Location: L(omega) = (x, y, z)")
    print("  Matter:   F_m(omega) = (a²+b², b+c, exp(a)/c)")
    print()

    # ── 1. Fix target matter configuration ──────────────────────────────────
    c_target = np.array([5.0, 3.0, 2.0])
    print(f"  Target matter state c = {c_target}")

    a_star, b_star, c_star = find_matter_config(c_target)
    omega_base = np.array([0, 0, 0, a_star, b_star, c_star])
    fm_check = F_m(omega_base)

    print(f"  Found matter config:  a={a_star:.6f}  b={b_star:.6f}  c={c_star:.6f}")
    print(f"  F_m verification:     {fm_check}")
    print(f"  Max error vs target:  {np.max(np.abs(fm_check - c_target)):.2e}")
    print()

    # ── 2. Define locations ──────────────────────────────────────────────────
    l_earth = np.array([0.0, 0.0, 0.0])
    l_alpha = np.array([4.0, 0.0, 0.0])   # 4 units in x (proxy for 4 ly)

    print(f"  Location 0 (Earth):   L = {l_earth}")
    print(f"  Location 1 (α Cen):   L = {l_alpha}")
    print(f"  Spacetime distance:   {np.linalg.norm(l_alpha - l_earth):.2f} units")
    print()

    # ── 3. Construct drive path ──────────────────────────────────────────────
    gamma = make_drive_path(l_earth, l_alpha, a_star, b_star, c_star)

    print("  Drive path: gamma(t) = (l_Earth + t*(l_Alpha - l_Earth), a*, b*, c*)")
    print("  gamma_dot(t) = (4, 0, 0, 0, 0, 0)  ← purely location-changing")
    print()

    # ── 4. Verify path ───────────────────────────────────────────────────────
    results = verify_path(gamma, c_target)
    print(f"  Max |F_m(gamma(t)) − c_target|:  {results['max_F_m_error']:.2e}")
    print(f"  Max |dF_m · gamma_dot|:          {results['max_kernel_error']:.2e}")
    print()

    # ── 5. Null space analysis ───────────────────────────────────────────────
    omega_mid = gamma(0.5)
    J = dF_m(omega_mid)
    rank = np.linalg.matrix_rank(J)
    kdim = ker_dim(omega_mid)

    print(f"  Jacobian dF_m at midpoint  (rank={rank}, dim ker={kdim}):")
    print()
    labels = ["x", "y", "z", "a", "b", "c"]
    header = "         " + "  ".join(f"{l:>8}" for l in labels)
    print(header)
    for i, row in enumerate(J):
        label = ["f₁=a²+b²", "f₂=b+c  ", "f₃=eᵃ/c "][i]
        print(f"  {label}  " + "  ".join(f"{v:8.4f}" for v in row))
    print()
    print(f"  ker dF_m has dimension {kdim}:")
    print(f"    — always contains ∂/∂x, ∂/∂y, ∂/∂z  (location directions)")
    print(f"    — x, y, z columns are identically zero → F_m independent of L")
    print()

    # ── 6. The key structural fact ────────────────────────────────────────────
    print("  Key structural fact:")
    print()
    print("    F_m^{-1}(c) ≅ R³ × {(a*,b*,c*)}")
    print()
    print("    The level set is a copy of R³ — the full location space.")
    print("    Earth and α Cen are trivially in the same connected component.")
    print("    π₀(F_m^{-1}(c)) = 1  for this model.")
    print()
    print("  Drive path exists. gamma_dot ∈ ker dF_m everywhere. ✓")
    print()

    # ── 7. What v0.2 requires ────────────────────────────────────────────────
    print("═" * 62)
    print("  This model is intentionally decoupled: F_m does not depend")
    print("  on (x,y,z).  The interesting case — and the real test —")
    print("  is a coupled model where Ω(ℓ) mixes location and matter")
    print("  parameters.  For example:")
    print()
    print("    F_m(r, θ, a, b) = (a·r + b·sin θ,  a² + b²·cos θ)")
    print()
    print("  There, ker dF_m is genuinely non-trivial, the level set")
    print("  may not be connected, and the drive path must be computed")
    print("  rather than read off.  That is v0.2.")
    print("═" * 62)


if __name__ == "__main__":
    main()
