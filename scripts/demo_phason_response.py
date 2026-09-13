#!/usr/bin/env python3
"""Frozen synthetic phason-response controls; stdout only, no target fitting."""
import argparse
import json
import math
from pathlib import Path
import sys

# Permit running by absolute path from an empty directory without package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from bpr.phason_response import (
    constitutive_inverse, constitutive_response, inverse_jacobian,
    normalized_response, pole_inverse, pole_report, prediction_jacobian,
    propagate_covariance, resonance_report,
)


def _vector(response):
    return [response["X"], response["Y"]]


def _residual(actual, predicted):
    return [actual[i] - predicted[i] for i in range(2)]


def _reject(function, *args):
    try:
        function(*args)
    except (ValueError, TypeError) as exc:
        return {"rejected": True, "reason": str(exc)}
    raise AssertionError("A frozen invalid control was not rejected")


def demonstration_report():
    """Evaluate only frozen dimensionless controls, with no retuning or files."""
    beta, tau = 761.0 / 4200.0, 20.0 / 21.0
    D = math.sqrt(beta)
    training = {"q": 1.0, "Omega": 21.0 / 20.0,
                "X": 761.0 / 8400.0, "Y": 761.0 / 8400.0}
    fitted = constitutive_inverse(training["X"], training["Y"],
                                  training["q"], training["Omega"])
    # ONLY the training constitutive observation determines these combinations.
    heldout = []
    for q, omega, expected in [
            (1.0, 21.0 / 10.0, [761.0 / 21000.0, 761.0 / 10500.0]),
            (2.0, 21.0 / 5.0, [761.0 / 8400.0, 761.0 / 8400.0])]:
        prediction = normalized_response(fitted["beta"], omega * fitted["tau"] / q ** 2)
        heldout.append({"q": q, "Omega": omega, "expected": expected,
                        "prediction": _vector(prediction),
                        "residual": _residual(expected, _vector(prediction))})
    exact_z = 19.0 / 20.0 - 1j / 20.0
    exact_pole = resonance_report(1.0, 1.0, D, 1.0, tau, 1.0)
    exact_pole.update(expected_positive_pole=[19.0 / 20.0, -1.0 / 20.0],
                      expected_Q_inverse=2.0 / 19.0,
                      independent_inverse=pole_inverse(exact_z, 1.0, 1.0))
    linked = []
    for q in [1.0, 2.0]:
        # At fixed C=rho=K=1, omega0=q and u0=tau/q, not an imposed Omega.
        report = resonance_report(1.0, 1.0, math.sqrt(fitted["beta"]),
                                  1.0, fitted["tau"], q)
        z = complex(*report["positive_pole"])
        report["independent_determinant_residual"] = abs(
            (1.0 - z * z) * (1.0 - 1j * fitted["tau"] / q * z) - fitted["beta"])
        linked.append(report)
    bound_grid = []
    for b in [0.0, 0.001, 0.02, 0.1, 0.25]:
        for u0 in [0.01, 0.1, 1.0, 10.0, 100.0]:
            pole = pole_report(b, u0)
            z = complex(*pole["positive_pole"])
            pole["observed_pole_approximation_error"] = abs(z - 1.0 + b / (2.0 * (1.0 - 1j * u0)))
            pole["observed_Q_approximation_error"] = abs(pole["Q_inverse"] - b * u0 / (1.0 + u0 * u0))
            bound_grid.append(pole)
    b = 0.95
    A = -8.0 + 36.0 * b - 27.0 * b * b
    width = math.sqrt(b * (9.0 * b - 8.0) ** 3)
    lo, hi = (A - width) / 8.0, (A + width) / 8.0
    collisions = [pole_report(8.0 / 9.0, 1.0 / math.sqrt(3.0))]
    collisions += [pole_report(b, u0) for u0 in [
        0.2, 0.4, 1.0, math.sqrt(lo), math.sqrt((lo + hi) / 2.0), math.sqrt(hi)]]
    inverse_J = inverse_jacobian(training["X"], training["Y"], 1.0, 21.0 / 20.0)
    measurement_covariance = np.diag([1e-10, 4e-10])
    parameter_covariance = propagate_covariance(inverse_J, measurement_covariance)
    uncertainty_heldout = []
    for case in heldout:
        J = prediction_jacobian(fitted["beta"], fitted["tau"], case["q"], case["Omega"])
        uncertainty_heldout.append({"q": case["q"], "Omega": case["Omega"],
                                    "jacobian": J.tolist(),
                                    "covariance": propagate_covariance(J, parameter_covariance).tolist()})
    # Predetermined competing channel sum; never adjust it to improve agreement.
    def mixture(omega):
        channels = [normalized_response(0.1, 0.5 * omega),
                    normalized_response(0.05, 2.0 * omega)]
        return [sum(c[key] for c in channels) for key in ["X", "Y"]]
    mixture_training = mixture(1.0)
    wrong_model = constitutive_inverse(mixture_training[0], mixture_training[1], 1.0, 1.0)
    mixture_prediction = normalized_response(wrong_model["beta"], 2.0 * wrong_model["tau"])
    mixture_actual = mixture(2.0)
    mixture_residual = _residual(mixture_actual, _vector(mixture_prediction))
    mismatch = math.hypot(*mixture_residual)
    if mismatch <= 1e-4:
        raise AssertionError("Frozen two-channel rejection failed")
    nuisance = {
        "participation_eta": 0.4,
        "participation_inverse": constitutive_inverse(0.4 * training["X"], 0.4 * training["Y"], 1.0, 21.0 / 20.0),
        "additive_Y_background": 0.01,
        "biased_inverse": constitutive_inverse(training["X"], training["Y"] + 0.01, 1.0, 21.0 / 20.0),
        "interpretation": "Unknown participation confounds beta with eta*beta. Unknown additive loss biases tau. Neither recovers microscopic parameters."}
    degeneracies = []
    base = constitutive_response(1.0, 1.0, D, 1.0, tau, 1.0, 21.0 / 20.0)
    for scale in [0.25, 4.0]:
        for sign in [-1.0, 1.0]:
            other = constitutive_response(1.0, scale, sign * math.sqrt(scale) * D,
                                          1.0, scale * tau, 1.0, 21.0 / 20.0)
            degeneracies.append({"scale": scale, "D_sign": sign,
                                 "response_residual": _residual(_vector(other), _vector(base))})
    return {
        "scope": "Conditional homogeneous elastic/diffusive phason model. Frozen synthetic controls, not experimental validation or a BPR-specific prediction.",
        "training": training, "frozen_inverse": fitted, "heldout": heldout,
        "exact_resonance": exact_pole, "linked_resonances": linked,
        "bound_grid": bound_grid, "collision_controls": collisions,
        "limits": {"imposed_controls": [normalized_response(beta, u) for u in [0.0, 1e-4, 1.0, 1e4]],
                   "fixed_q_then_Omega_to_zero_modulus": 1.0 - beta,
                   "fixed_positive_Omega_then_q_to_zero_modulus": 1.0,
                   "noncommuting_for_nonzero_beta": True,
                   "q_zero": _reject(constitutive_response, 1.0, 1.0, D, 1.0, tau, 0.0, 0.0),
                   "uniform_dynamics": "rho Uddot=0, Gamma Wdot=0: translations/free drift, not finite resonance",
                   "uncoupled_pole": pole_report(0.0, 1.0),
                   "uncoupled_inverse": _reject(constitutive_inverse, 0.0, 0.0, 1.0, 1.0)},
        "invalid_controls": {"beta_one": _reject(normalized_response, 1.0, 1.0),
                             "beta_above_one": _reject(normalized_response, 1.1, 1.0),
                             "negative_Gamma": _reject(constitutive_response, 1.0, 1.0, D, 1.0, -tau, 1.0, 1.0)},
        "sign_and_scaling_degeneracy": degeneracies,
        "uncertainty": {"measurement_covariance": measurement_covariance.tolist(),
                        "inverse_jacobian": inverse_J.tolist(),
                        "parameter_covariance": parameter_covariance.tolist(),
                        "heldout": uncertainty_heldout,
                        "interpretation": "Local first-order delta method only, not nonlinear confidence bounds. C,q,Omega independently exact. Shared training uncertainty correlates held-out predictions."},
        "two_channel_rejection": {"training": mixture_training, "frozen_single_channel_inverse": wrong_model,
                                  "heldout_actual": mixture_actual, "heldout_prediction": _vector(mixture_prediction),
                                  "actual_minus_prediction": mixture_residual,
                                  "expected_residual": [261.0 / 42500.0, -27.0 / 42500.0],
                                  "residual_norm": mismatch, "threshold": 1e-4,
                                  "rejected": mismatch > 1e-4},
        "nuisance_controls": nuisance,
        "honest_boundaries": [
            "Resonance is a complex pole, not a driven peak or an exact constitutive loss tangent.",
            "The exact constitutive circle differs from the leading-order pole circle.",
            "Numerical root diagnostics do not certify the analytic theorem or solver roundoff.",
            "Sign(D), absolute K/Gamma scale and unknown participation are not identified.",
            "A single complex pole without independently known omega0 has three unknowns and two real data.",
            "Arbitrary frequency-dependent nuisances can absorb discrepancies and make a single-channel test inconclusive.",
            "Finite-mode participation requires a derived model, not insertion into this cubic.",
            "Missing microscopic matching includes C,K,D,Gamma, tensors, channels, scale dependence, boundaries and mode coupling.",
            "No particle masses, spacetime chirality, gauge unification, quantum gravity or TOE is established."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit strict JSON to stdout")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print(report["scope"])
        print("Training-only inverse beta={:.12g}, tau={:.12g}".format(
            report["frozen_inverse"]["beta"], report["frozen_inverse"]["tau"]))
        for case in report["heldout"]:
            print("Held-out q={}, Omega={}: prediction={}, residual={}".format(
                case["q"], case["Omega"], case["prediction"], case["residual"]))
        print("Exact q=1 pole {} with Q_inverse={:.12g}; q=2 uses the same frozen parameters.".format(
            report["exact_resonance"]["positive_pole"], report["exact_resonance"]["Q_inverse"]))
        print("Frozen analytic bound controls: {}. Collision/overdamping controls: {}.".format(
            len(report["bound_grid"]), len(report["collision_controls"])))
        print("Two-channel held-out mismatch={:.12g} > 1e-4: rejected.".format(
            report["two_channel_rejection"]["residual_norm"]))
        print(report["nuisance_controls"]["interpretation"])
        print(report["uncertainty"]["interpretation"])
        print("Static and q-to-zero limits do not commute. Uniform q=0 and invalid stability/friction controls are rejected.")
        for boundary in report["honest_boundaries"]:
            print(boundary)


if __name__ == "__main__":
    main()
