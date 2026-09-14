"""Conditional scalar elastic/diffusive response, not a microscopic BPR model.

Equations and controls: doc/derivations/phason_response_2026-09-12.md.
Fourier convention exp(iqx-iOmega*t); resonance means a complex pole, not a
response peak. All scalar inputs/outputs are finite normal binary64 or zero.
Numerically unavailable poles do not invalidate the general analytic theorem.
No fitting, scanning, mode participation ansatz, or simulation is performed.
"""
import math
import numbers
import sys
from fractions import Fraction

import numpy as np


_TINY = sys.float_info.min
_EPS = sys.float_info.epsilon


class NumericalUnavailable(ValueError):
    """The requested calculation is unresolved in the declared binary64 policy."""


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(name + " must be a real scalar (not a boolean or array)")
    original = value
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(name + " is not representable in binary64") from exc
    if value == 0.0 and original != 0:
        raise ValueError(name + " underflowed to zero during binary64 conversion")
    if not math.isfinite(value) or (value != 0.0 and abs(value) < _TINY):
        raise ValueError(name + " must be finite and normal, or zero")
    return value


def _checked(value, name):
    if not math.isfinite(value) or (value != 0.0 and abs(value) < _TINY):
        raise NumericalUnavailable(name + " overflowed or underflowed binary64")
    return float(value)


def _mul(a, b, name):
    out = _checked(a * b, name)
    if out == 0.0 and a != 0.0 and b != 0.0:
        raise NumericalUnavailable(name + " underflowed to zero")
    return out


def _div(a, b, name):
    if b == 0.0:
        raise NumericalUnavailable(name + " has a zero denominator")
    out = _checked(a / b, name)
    if out == 0.0 and a != 0.0:
        raise NumericalUnavailable(name + " underflowed to zero")
    return out


def _positive(value, name):
    value = _number(value, name)
    if value <= 0.0:
        raise ValueError(name + " must be positive; q=0 is a uniform-field degeneracy")
    return value


def _beta(value):
    value = _number(value, "beta")
    if not 0.0 <= value < 1.0:
        raise ValueError("positive strain energy requires 0 <= beta < 1")
    return value


def _scale(q, omega):
    q = _positive(q, "q")
    omega = _positive(omega, "Omega")
    return _div(_mul(q, q, "q squared"), omega, "q squared / Omega")


def _material(C, K, D, rho, Gamma):
    C, K = _positive(C, "C"), _positive(K, "K")
    rho, Gamma = _positive(rho, "rho"), _positive(Gamma, "Gamma")
    D = _number(D, "D")
    # Strict stability concerns supplied parameters, not a rounded square-root
    # normalization (which can turn exact D**2=C*K into beta just below one).
    exact_C, exact_K, exact_D = (Fraction.from_float(v) for v in (C, K, D))
    if exact_D * exact_D >= exact_C * exact_K:
        raise ValueError("positive strain energy requires D squared < C*K")
    # Square-root factors avoid gratuitous overflow of C*K or D*D.
    ratio = _div(_div(abs(D), math.sqrt(C), "D / sqrt(C)"),
                 math.sqrt(K), "D / sqrt(C*K)")
    beta = _beta(_mul(ratio, ratio, "beta"))
    tau = _div(Gamma, K, "Gamma / K")
    return C, K, D, rho, Gamma, beta, tau


def normalized_response(beta, u):
    """Return X,Y and modulus=[Re(Ceff/C),Im(Ceff/C)] at imposed u>=0."""
    beta, u = _beta(beta), _number(u, "u")
    if u < 0.0:
        raise ValueError("imposed dimensionless frequency u must be nonnegative")
    if beta == 0.0:
        X = Y = 0.0
    else:
        norm = math.hypot(1.0, u)
        X = _div(_div(beta, norm, "X numerator"), norm, "X")
        Y = _mul(X, u, "Y")
    real = _checked(1.0 - X, "normalized real modulus")
    if X != 0.0 and real == 1.0:
        raise NumericalUnavailable("softening is unresolved in the real modulus")
    return {"beta": beta, "u": u, "X": X, "Y": Y,
            "modulus": [real, -Y]}


def constitutive_response(C, K, D, rho, Gamma, q, Omega):
    """Physical modulus at an independently imposed Omega>=0, q>0.

    q=0 is rejected, including q=Omega=0. At D=0 phonon data cannot identify
    tau. rho is validated but does not enter the imposed-frequency modulus.
    """
    C, K, D, rho, Gamma, beta, tau = _material(C, K, D, rho, Gamma)
    q, Omega = _positive(q, "q"), _number(Omega, "Omega")
    if Omega < 0.0:
        raise ValueError("Omega must be nonnegative")
    q2 = _mul(q, q, "q squared")
    u = _mul(_div(Omega, q2, "Omega / q squared"), tau, "u")
    report = normalized_response(beta, u)
    report.update({"Ceff": [_mul(C, report["modulus"][0], "Re Ceff"),
                              _mul(C, report["modulus"][1], "Im Ceff")],
                   "q": q, "Omega": Omega, "tau": tau,
                   "relaxation_identifiable": beta > 0.0})
    return report


def _classification(beta, u0):
    # No powers of u0 are needed; inequalities apply even outside the oracle cap.
    result = {"regime": "oscillatory", "collision": False,
              "near_collision": False, "overdamped": False,
              "ancestry_available": beta < 8.0 / 9.0,
              "classification_available": True,
              "collision_u0_interval": None}
    if beta < 8.0 / 9.0:
        return result
    if beta == 8.0 / 9.0:
        lo = hi = 1.0 / math.sqrt(3.0)
    else:
        A = -8.0 + 36.0 * beta - 27.0 * beta * beta
        width = math.sqrt(beta * (9.0 * beta - 8.0) ** 3)
        upper = (A + width) / 8.0
        # Product of t roots is 1-beta; avoid lower-endpoint cancellation.
        lower = (1.0 - beta) / upper
        lo, hi = math.sqrt(lower), math.sqrt(upper)
        if hi - lo <= 128.0 * _EPS * max(lo, hi):
            # Near the cusp, rounded endpoints can collapse or even invert.
            # Sorting would disguise lost width as a resolved analytic interval.
            result.update(regime="numerically_unresolved", near_collision=True,
                          classification_available=False)
            return result
    result["collision_u0_interval"] = [lo, hi]
    result["near_collision"] = any(
        math.isclose(u0, endpoint, rel_tol=128.0 * _EPS, abs_tol=0.0)
        for endpoint in (lo, hi))
    # Literal floating comparisons are separate from proximity diagnostics;
    # an endpoint close in floating arithmetic is not a certified exact equality.
    if u0 == lo or u0 == hi:
        result.update(regime="collision", collision=True)
    elif lo < u0 < hi:
        result.update(regime="overdamped", overdamped=True)
    return result


def _bounds(beta, u0):
    if beta > 0.25:
        return None
    # hypot avoids gratuitous overflow/underflow of u0**2.
    norm = math.hypot(1.0, u0)
    softening = _div(_div(beta, norm, "leading softening factor"), norm, "leading softening")
    re = _div(softening, 2.0, "leading frequency shift")
    loss = _mul(softening, u0, "leading loss")
    imag = -_div(loss, 2.0, "leading pole imaginary part")
    leading_real = _checked(1.0 - re, "leading pole real part")
    if re != 0.0 and leading_real == 1.0:
        raise NumericalUnavailable("leading pole frequency shift is unresolved")
    beta2 = _mul(beta, beta, "beta squared bound")
    return {"pole_error_bound": _mul(13.0 / 8.0, beta2, "pole bound"),
            "Q_error_bound": _mul(5.0, beta2, "Q bound"),
            "leading_pole": [leading_real, imag],
            "leading_Q_inverse": loss,
            "certifies_roundoff": False,
            "theorem": "For every u0>0: |z-1+beta/(2(1-iu0))|<=13 beta^2/8; |Qinv-beta*u0/(1+u0^2)|<=5 beta^2."}


def pole_report(beta, u0):
    """Unique positive-frequency pole when numerically resolved, never arbitrary roots.

    beta<8/9 guarantees an oscillatory pair for every u0>0. At/above that
    threshold no acoustic ancestry through a collision is asserted. Roots near
    collisions and outside a conservative binary64 oracle window are withheld.
    The small-beta analytic bounds are not numerical error certificates.
    """
    beta, u0 = _beta(beta), _positive(u0, "u0")
    report = _classification(beta, u0)
    report.update({"beta": beta, "u0": u0, "positive_pole": None,
                   "Q_inverse": None, "numerical_available": False,
                   "bounds": _bounds(beta, u0), "reason": None,
                   "pole_kind": "unique_positive_frequency",
                   "phason_cancels_in_phonon_susceptibility": beta == 0.0,
                   "residual": None, "root_separation": None})
    if beta == 0.0:
        report.update(positive_pole=[1.0, 0.0], Q_inverse=0.0,
                      numerical_available=True, residual=0.0, regime="uncoupled")
        return report
    if not report["classification_available"]:
        report["reason"] = "Collision interval width is unresolved near the cusp; classification and pole withheld."
        return report
    if report["collision"] or report["near_collision"] or report["overdamped"]:
        report["reason"] = "No resolved oscillatory pole at a collision, near it, or in the overdamped interval."
        return report
    if not (1e-6 <= u0 <= 1e6 and beta >= 1e-10 and 1.0 - beta >= 1e-10):
        report["reason"] = "Outside conservative numerical pole window; analytic classification and theorem remain applicable."
        return report
    # One cubic, exactly three roots. No continuation scan or root-index heuristic.
    try:
        roots_r = np.roots([u0, 1.0, u0, 1.0 - beta])
    except np.linalg.LinAlgError:
        report["reason"] = "Cubic eigensolver did not converge."
        return report
    roots = [1j * complex(r) for r in roots_r]
    if any(not math.isfinite(z.real) or not math.isfinite(z.imag) for z in roots):
        report["reason"] = "Nonfinite cubic roots."
        return report
    positive = [z for z in roots if z.real > 1e-10]
    if len(positive) != 1 or any(z.imag >= 0.0 for z in roots):
        report["reason"] = "Unresolved root count or strict damping."
        return report
    z = positive[0]
    separation = min(abs(a - b) for i, a in enumerate(roots) for b in roots[i + 1:])
    residual = abs((1.0 - z * z) * (1.0 - 1j * u0 * z) - beta)
    tolerance = 2e-11 * max(1.0, abs((1.0 - z * z) * (1.0 - 1j * u0 * z)), beta)
    if (separation <= 1e-7 * max(1.0, abs(z)) or residual > tolerance
            or -z.imag <= 256.0 * _EPS * max(1.0, abs(z))):
        report["reason"] = "Pole residual, separation, or damping is numerically unresolved."
        return report
    if beta <= 0.25 and not abs(z - 1.0) < beta:
        report["reason"] = "Numerical pole did not resolve the analytically isolated disk."
        return report
    report.update(positive_pole=[_checked(z.real, "pole real"), _checked(z.imag, "pole imag")],
                  Q_inverse=_div(-2.0 * z.imag, z.real, "Q inverse"),
                  numerical_available=True, residual=float(residual),
                  root_separation=float(separation))
    return report


def resonance_report(C, K, D, rho, Gamma, q):
    """Link physical pole to q: omega0=q sqrt(C/rho), u0=tau*omega0/q**2."""
    C, K, D, rho, Gamma, beta, tau = _material(C, K, D, rho, Gamma)
    q = _positive(q, "q")
    speed = _div(math.sqrt(C), math.sqrt(rho), "bare wave speed")
    omega0 = _mul(q, speed, "omega0")
    u0 = _mul(tau, _div(speed, q, "speed / q"), "u0")
    report = pole_report(beta, u0)
    z = report["positive_pole"]
    report.update(q=q, omega0=omega0, tau=tau,
                  physical_pole=None if z is None else [
                      _mul(omega0, z[0], "physical pole real"),
                      _mul(omega0, z[1], "physical pole imag")])
    return report


def constitutive_inverse(X, Y, q, Omega):
    """Recover beta,tau from nonzero response, assuming exact known C,q,Omega.

    Zero coupling, static drive, unknown participation, or unknown backgrounds
    cannot identify intrinsic beta and tau with this inverse.
    """
    X, Y = _positive(X, "X"), _positive(Y, "Y")
    a = _scale(q, Omega)
    u = _div(Y, X, "Y / X")
    beta = _beta(_checked(X + _mul(Y, u, "Y squared / X"), "beta inverse"))
    tau = _mul(a, u, "tau inverse")
    return {"beta": beta, "u": u, "tau": tau}


def pole_inverse(z, q, omega0):
    """Exact pole inverse with independently known omega0; check the positive branch."""
    if isinstance(z, (bool, np.bool_)) or not isinstance(z, numbers.Complex):
        raise TypeError("z must be a complex scalar")
    x, y = _positive(z.real, "Re z"), _positive(-z.imag, "-Im z")
    a = _scale(q, omega0)
    x2, y2 = _mul(x, x, "x squared"), _mul(y, y, "y squared")
    H = _checked(math.fsum([1.0, -x2, _mul(3.0, y2, "3 y squared")]), "H")
    if H <= 64.0 * _EPS * max(1.0, x2, 3.0 * y2):
        raise NumericalUnavailable("H is nonpositive or cancellation-limited")
    u0 = _div(_mul(2.0, y, "2y"), H, "u0 inverse")
    real = _checked(math.fsum([1.0, -x2, y2]), "1-x squared+y squared")
    imag = _mul(_mul(2.0, x, "2x"), y, "2xy")
    numerator = _checked(_mul(real, real, "real squared") + _mul(imag, imag, "imag squared"), "pole inverse numerator")
    beta = _beta(_div(numerator, H, "beta inverse"))
    if beta == 0.0:
        raise ValueError("uncoupled phonon data do not identify relaxation")
    report = pole_report(beta, u0)
    if not report["numerical_available"]:
        raise NumericalUnavailable("pole inverse branch check unavailable: " + report["reason"])
    checked_z = complex(*report["positive_pole"])
    if abs(checked_z - complex(x, -y)) > 1e-8 * max(1.0, abs(checked_z)):
        raise NumericalUnavailable("input pole fails the positive-frequency branch check")
    residual = abs((1.0 - complex(x, -y) ** 2) * (1.0 - 1j * u0 * complex(x, -y)) - beta)
    if not math.isfinite(residual) or residual > 2e-11:
        raise NumericalUnavailable("input pole fails determinant residual check")
    return {"beta": beta, "u0": u0, "tau": _mul(a, u0, "tau inverse"),
            "q": float(q), "omega0": float(omega0), "residual": residual,
            "ancestry_available": report["ancestry_available"]}


def inverse_jacobian(X, Y, q, Omega):
    """Rows (beta,tau), columns (X,Y); C,q,Omega are independently exact."""
    fitted = constitutive_inverse(X, Y, q, Omega)
    X, u, a = float(X), fitted["u"], _scale(q, Omega)
    return np.array([[_mul(1.0 - u, 1.0 + u, "inverse J00"), _mul(2.0, u, "inverse J01")],
                     [-_div(_mul(a, u, "a*u"), X, "inverse J10"), _div(a, X, "inverse J11")]])


def prediction_jacobian(beta, tau, q, Omega):
    """Rows (X,Y), columns (beta,tau) for a held-out imposed response."""
    beta, tau = _beta(beta), _positive(tau, "tau")
    h = _div(1.0, _scale(q, Omega), "Omega / q squared")
    u = _mul(h, tau, "prediction u")
    norm = math.hypot(1.0, u)
    inv = _div(_div(1.0, norm, "inverse norm"), norm, "inverse norm squared")
    ui = _mul(u, inv, "u inverse norm squared")
    factor = _mul(_mul(beta, h, "beta h"), inv, "beta h inverse norm squared")
    # (1-u^2)/(1+u^2) is evaluated without forming large u^2.
    if u <= 1.0:
        contrast = _mul(_mul(1.0 - u, 1.0 + u, "1-u squared"), inv, "contrast")
    else:
        reciprocal = _div(1.0, u, "inverse u")
        contrast = (reciprocal - 1.0) * (reciprocal + 1.0) / (1.0 + reciprocal * reciprocal)
    return np.array([[inv, -_mul(_mul(2.0, factor, "2 factor"), ui, "prediction J01")],
                     [ui, _mul(factor, contrast, "prediction J11")]])


def _matrix2(value, name):
    # Shape/length cap BEFORE any conversion or allocation, including custom
    # array-like objects. Only actual ndarrays or bounded lists/tuples accepted.
    if isinstance(value, np.ndarray):
        if value.shape != (2, 2):
            raise ValueError(name + " must have shape (2,2)")
    elif isinstance(value, (list, tuple)):
        if len(value) != 2 or any(not isinstance(row, (list, tuple)) or len(row) != 2 for row in value):
            raise ValueError(name + " must have shape (2,2)")
    else:
        raise TypeError(name + " must be a 2x2 ndarray or nested list/tuple")
    return np.array([[_number(value[i][j], name) for j in range(2)] for i in range(2)], dtype=float)


def propagate_covariance(jacobian, covariance):
    """Local delta method J Sigma J.T, not nonlinear confidence or validation.

    Sigma must be exactly symmetric positive semidefinite in its supplied
    binary64 representation. Shape is capped before conversion. Shared parameter
    covariance correlates held-out predictions; C,q,Omega uncertainty is omitted.
    """
    J, S = _matrix2(jacobian, "jacobian"), _matrix2(covariance, "covariance")
    if S[0, 1] != S[1, 0] or S[0, 0] < 0.0 or S[1, 1] < 0.0:
        raise ValueError("covariance must be symmetric positive semidefinite")
    # Exact rational comparison avoids determinant overflow and tolerating a
    # genuinely negative variance eigenvalue merely because it is small.
    a, b, d = (Fraction.from_float(float(S[i, j])) for i, j in [(0, 0), (0, 1), (1, 1)])
    if a * d < b * b:
        raise ValueError("covariance must be positive semidefinite")
    # Fixed 2x2 cap makes exact products inexpensive. Summing rounded products
    # instead loses a normal variance such as (1-nextafter(1,2))**2 to zero.
    J_exact = [[Fraction.from_float(float(J[i, j])) for j in range(2)] for i in range(2)]
    S_exact = [[Fraction.from_float(float(S[i, j])) for j in range(2)] for i in range(2)]
    out = np.empty((2, 2), dtype=float)
    for i in range(2):
        for j in range(i, 2):
            exact = sum((J_exact[i][k] * S_exact[k][l] * J_exact[j][l]
                         for k in range(2) for l in range(2)), Fraction(0))
            try:
                entry = _checked(float(exact), "propagated covariance")
            except OverflowError as exc:
                raise NumericalUnavailable("covariance conversion overflow") from exc
            if entry == 0.0 and exact != 0:
                raise NumericalUnavailable("covariance underflowed to zero")
            out[i, j] = out[j, i] = entry
    rounded_a, rounded_b, rounded_d = (
        Fraction.from_float(float(out[i, j])) for i, j in [(0, 0), (0, 1), (1, 1)])
    if rounded_a * rounded_d < rounded_b * rounded_b:
        raise NumericalUnavailable("entrywise binary64 rounding does not preserve covariance PSD")
    return out
