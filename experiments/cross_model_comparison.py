"""
Statistical Cross-Model Comparison: RPST vs Standard Models
=============================================================

The adversarial suite tests whether RPST's threshold is robust.
This suite asks the different question:

    Does RPST predict something that standard models do NOT?

We run identical experiments on three models:
  1. RPST: discrete symplectic map on Z_p (sharp spectral threshold)
  2. Kuramoto: coupled phase oscillators (smooth sync transition)
  3. Linear diffusion: dθ/dt = J·L·θ (exponential relaxation, no threshold)

For each model, we measure the transition curve r(J) and fit:
  - Step function: r = {r_high, J < J_c; r_low, J > J_c}  (sharp)
  - Sigmoid: r = r_low + (r_high - r_low) / (1 + exp(k(J - J_c)))  (smooth)

Then compare fits using AIC/BIC. If RPST genuinely predicts a sharp
transition, the step function should win for RPST but the sigmoid
should win for Kuramoto.

This is the critical test. If both models are equally well-fit by the
same functional form, then RPST doesn't add explanatory power.

Usage:
    python experiments/cross_model_comparison.py
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bpr.stability_manifolds import DynamicalMatrix, StabilityManifold


# ===================================================================
# Model 1: RPST (discrete symplectic map)
# ===================================================================

def rpst_transition_curve(N=16, p=101, J_values=None, n_steps=2000):
    """Run RPST symplectic map and measure late-time order parameter r(J).

    Returns J_values, r_values (order parameter at each coupling).
    """
    if J_values is None:
        J_values = np.linspace(0.1, 2.5, 30)

    sm = StabilityManifold(p=p)
    rng = np.random.RandomState(42)

    r_values = []
    for J in J_values:
        # Ring coupling
        A = np.zeros((N, N))
        for i in range(N):
            A[i, (i + 1) % N] = J
            A[i, (i - 1) % N] = J

        # Start near constant with small perturbation
        q0 = np.full(N, p // 2, dtype=int)
        pi0 = np.zeros(N, dtype=int)
        pi0[0] = 1

        sim = sm.simulate(q0, pi0, A, n_steps=n_steps)

        # Convert to continuum phases and measure order parameter
        phi_traj = 2.0 * np.pi * sim["q_trajectory"].astype(float) / p
        order_param = np.array([
            abs(np.mean(np.exp(1j * phi_traj[t])))
            for t in range(len(phi_traj))
        ])

        # Late-time average (last 20%)
        n_late = max(1, len(order_param) // 5)
        r = float(np.mean(order_param[-n_late:]))
        r_values.append(r)

    return J_values, np.array(r_values)


# ===================================================================
# Model 2: Kuramoto
# ===================================================================

def kuramoto_transition_curve(N=16, K_values=None, n_steps=5000, dt=0.01):
    """Run Kuramoto model and measure steady-state order parameter r(K).

    dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j - θ_i)
    """
    if K_values is None:
        K_values = np.linspace(0.1, 5.0, 30)

    rng = np.random.RandomState(42)
    omega = rng.standard_cauchy(N) * 0.3  # Natural frequencies

    r_values = []
    for K in K_values:
        theta = rng.uniform(0, 2 * np.pi, N)

        for _ in range(n_steps):
            coupling = np.zeros(N)
            for i in range(N):
                coupling[i] = (K / N) * np.sum(np.sin(theta - theta[i]))
            theta = theta + (omega + coupling) * dt
            theta = theta % (2 * np.pi)

        r = abs(np.mean(np.exp(1j * theta)))
        r_values.append(float(r))

    return K_values, np.array(r_values)


# ===================================================================
# Model 3: Linear diffusion
# ===================================================================

def diffusion_transition_curve(N=16, J_values=None, n_steps=2000, dt=0.01):
    """Linear diffusion on ring: dθ/dt = -J·L·θ.

    No threshold at all — exponential relaxation for all J > 0.
    Measure "coherence" as 1 - variance(θ) / initial_variance.
    """
    if J_values is None:
        J_values = np.linspace(0.1, 2.5, 30)

    rng = np.random.RandomState(42)

    # Ring Laplacian
    L = np.zeros((N, N))
    for i in range(N):
        L[i, i] = 2.0
        L[i, (i + 1) % N] = -1.0
        L[i, (i - 1) % N] = -1.0

    r_values = []
    for J in J_values:
        theta = rng.uniform(0, 2 * np.pi, N)
        initial_var = np.var(theta)

        for _ in range(n_steps):
            theta = theta - J * L @ theta * dt

        r = abs(np.mean(np.exp(1j * theta)))
        r_values.append(float(r))

    return J_values, np.array(r_values)


# ===================================================================
# Fitting: step function vs sigmoid
# ===================================================================

def fit_step(J_values, r_values):
    """Fit step function: r = r_high if J < J_c, else r_low.

    Returns (J_c, r_high, r_low, residual_sum_sq, n_params).
    """
    best_rss = np.inf
    best_params = None

    for i in range(1, len(J_values) - 1):
        J_c = J_values[i]
        mask_low = J_values <= J_c
        mask_high = J_values > J_c

        if np.sum(mask_low) == 0 or np.sum(mask_high) == 0:
            continue

        r_high = np.mean(r_values[mask_low])
        r_low = np.mean(r_values[mask_high])

        predicted = np.where(mask_low, r_high, r_low)
        rss = float(np.sum((r_values - predicted) ** 2))

        if rss < best_rss:
            best_rss = rss
            best_params = (J_c, r_high, r_low)

    if best_params is None:
        return None

    J_c, r_high, r_low = best_params
    return {
        "J_c": J_c,
        "r_high": r_high,
        "r_low": r_low,
        "rss": best_rss,
        "n_params": 3,  # J_c, r_high, r_low
    }


def fit_sigmoid(J_values, r_values):
    """Fit sigmoid: r = r_low + (r_high - r_low) / (1 + exp(k(J - J_c))).

    Returns (J_c, k, r_high, r_low, residual_sum_sq, n_params).
    """
    def sigmoid(J, J_c, k, r_high, r_low):
        return r_low + (r_high - r_low) / (1.0 + np.exp(k * (J - J_c)))

    def objective(params):
        J_c, k, r_high, r_low = params
        predicted = sigmoid(J_values, J_c, k, r_high, r_low)
        return float(np.sum((r_values - predicted) ** 2))

    # Multi-start optimization
    best_result = None
    best_rss = np.inf

    J_mid = (J_values[0] + J_values[-1]) / 2
    for J_c_init in np.linspace(J_values[1], J_values[-2], 5):
        for k_init in [1.0, 5.0, 20.0]:
            r_hi = float(np.max(r_values))
            r_lo = float(np.min(r_values))
            x0 = [J_c_init, k_init, r_hi, r_lo]

            try:
                result = minimize(objective, x0, method="Nelder-Mead",
                                  options={"maxiter": 5000, "xatol": 1e-8})
                if result.fun < best_rss:
                    best_rss = result.fun
                    best_result = result.x
            except Exception:
                continue

    if best_result is None:
        return None

    J_c, k, r_high, r_low = best_result
    return {
        "J_c": J_c,
        "k": k,  # Steepness; large k ≈ step function
        "r_high": r_high,
        "r_low": r_low,
        "rss": best_rss,
        "n_params": 4,  # J_c, k, r_high, r_low
    }


# ===================================================================
# Model comparison: AIC and BIC
# ===================================================================

def compute_aic_bic(rss, n_data, n_params):
    """Compute AIC and BIC from residual sum of squares.

    AIC = n·ln(RSS/n) + 2k
    BIC = n·ln(RSS/n) + k·ln(n)

    Lower is better.
    """
    n = n_data
    k = n_params
    if rss <= 0 or n <= k:
        return float('inf'), float('inf')

    log_likelihood_term = n * np.log(rss / n)
    aic = log_likelihood_term + 2 * k
    bic = log_likelihood_term + k * np.log(n)
    return aic, bic


def compare_models(J_values, r_values, label=""):
    """Fit both step and sigmoid, compare via AIC/BIC."""
    n = len(J_values)

    step_fit = fit_step(J_values, r_values)
    sig_fit = fit_sigmoid(J_values, r_values)

    if step_fit is None or sig_fit is None:
        return {"error": "Fitting failed", "label": label}

    step_aic, step_bic = compute_aic_bic(step_fit["rss"], n, step_fit["n_params"])
    sig_aic, sig_bic = compute_aic_bic(sig_fit["rss"], n, sig_fit["n_params"])

    # Delta AIC: negative means step is better
    delta_aic = step_aic - sig_aic
    delta_bic = step_bic - sig_bic

    return {
        "label": label,
        "n_data": n,
        "step_fit": step_fit,
        "sigmoid_fit": sig_fit,
        "step_aic": step_aic,
        "step_bic": step_bic,
        "sigmoid_aic": sig_aic,
        "sigmoid_bic": sig_bic,
        "delta_aic": delta_aic,  # negative = step wins
        "delta_bic": delta_bic,
        "preferred_aic": "step" if delta_aic < 0 else "sigmoid",
        "preferred_bic": "step" if delta_bic < 0 else "sigmoid",
        "sigmoid_steepness": sig_fit["k"],
    }


# ===================================================================
# Bootstrap confidence intervals
# ===================================================================

def bootstrap_threshold(J_values, r_values, n_bootstrap=500, rng=None):
    """Bootstrap the threshold location J_c.

    For each bootstrap sample, fit a step function and record J_c.
    Returns 95% CI.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    n = len(J_values)
    J_c_samples = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        J_boot = J_values[idx]
        r_boot = r_values[idx]

        # Sort by J for step fitting
        order = np.argsort(J_boot)
        J_boot = J_boot[order]
        r_boot = r_boot[order]

        step = fit_step(J_boot, r_boot)
        if step is not None:
            J_c_samples.append(step["J_c"])

    if len(J_c_samples) < 10:
        return None

    J_c_samples = np.array(J_c_samples)
    return {
        "mean": float(np.mean(J_c_samples)),
        "std": float(np.std(J_c_samples)),
        "ci_95": (float(np.percentile(J_c_samples, 2.5)),
                  float(np.percentile(J_c_samples, 97.5))),
        "n_valid": len(J_c_samples),
    }


# ===================================================================
# Transition sharpness: derivative-based
# ===================================================================

def measure_sharpness(J_values, r_values):
    """Quantify transition sharpness via max |dr/dJ|.

    Sharp transition → large max derivative.
    Smooth transition → small, spread-out derivative.
    """
    dr = np.diff(r_values)
    dJ = np.diff(J_values)
    deriv = dr / dJ

    max_deriv = float(np.max(np.abs(deriv)))
    mean_deriv = float(np.mean(np.abs(deriv)))

    # Sharpness ratio: how concentrated is the transition
    sharpness = max_deriv / max(mean_deriv, 1e-15)

    # Width: J-range where |dr/dJ| > 50% of max
    threshold = 0.5 * np.max(np.abs(deriv))
    active = np.abs(deriv) > threshold
    if np.any(active):
        J_mid = (J_values[:-1] + J_values[1:]) / 2
        active_J = J_mid[active]
        width = float(np.max(active_J) - np.min(active_J))
    else:
        width = float(J_values[-1] - J_values[0])

    return {
        "max_derivative": max_deriv,
        "mean_derivative": mean_deriv,
        "sharpness_ratio": sharpness,
        "transition_width": width,
    }


# ===================================================================
# Kolmogorov-Smirnov test: are two transition curves different?
# ===================================================================

def ks_compare_curves(r_values_1, r_values_2, label_1="Model1", label_2="Model2"):
    """KS test between two r(J) curves (treated as distributions).

    Tests if the two curves represent significantly different distributions.
    """
    stat, p_value = stats.ks_2samp(r_values_1, r_values_2)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significantly_different": p_value < 0.05,
        "labels": (label_1, label_2),
    }


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("STATISTICAL CROSS-MODEL COMPARISON")
    print("=" * 70)
    print()
    print("Question: does RPST predict something standard models do NOT?")
    print()

    # --- Generate transition curves ---
    print("Generating transition curves...")
    print("  RPST symplectic map...", flush=True)
    rpst_J, rpst_r = rpst_transition_curve(N=16, p=101, n_steps=1500)
    print("  Kuramoto oscillators...", flush=True)
    kura_K, kura_r = kuramoto_transition_curve(N=16, n_steps=4000)
    print("  Linear diffusion...", flush=True)
    diff_J, diff_r = diffusion_transition_curve(N=16, n_steps=2000)

    # --- Measure transition sharpness ---
    print("\n--- Transition Sharpness ---")
    for label, J, r in [("RPST", rpst_J, rpst_r),
                          ("Kuramoto", kura_K, kura_r),
                          ("Diffusion", diff_J, diff_r)]:
        sharp = measure_sharpness(J, r)
        print(f"  {label:12s}: max|dr/dJ|={sharp['max_derivative']:.4f}  "
              f"ratio={sharp['sharpness_ratio']:.2f}  "
              f"width={sharp['transition_width']:.4f}")

    # --- Fit step vs sigmoid ---
    print("\n--- Step vs Sigmoid Fit (AIC/BIC) ---")
    comparisons = {}
    for label, J, r in [("RPST", rpst_J, rpst_r),
                          ("Kuramoto", kura_K, kura_r),
                          ("Diffusion", diff_J, diff_r)]:
        comp = compare_models(J, r, label)
        comparisons[label] = comp

        if "error" in comp:
            print(f"  {label}: fitting failed")
            continue

        print(f"\n  {label}:")
        print(f"    Step fit:    RSS={comp['step_fit']['rss']:.6f}  "
              f"J_c={comp['step_fit']['J_c']:.4f}  "
              f"AIC={comp['step_aic']:.2f}  BIC={comp['step_bic']:.2f}")
        print(f"    Sigmoid fit: RSS={comp['sigmoid_fit']['rss']:.6f}  "
              f"J_c={comp['sigmoid_fit']['J_c']:.4f}  k={comp['sigmoid_steepness']:.2f}  "
              f"AIC={comp['sigmoid_aic']:.2f}  BIC={comp['sigmoid_bic']:.2f}")
        print(f"    ΔAIC={comp['delta_aic']:.2f} → {comp['preferred_aic']} preferred")
        print(f"    ΔBIC={comp['delta_bic']:.2f} → {comp['preferred_bic']} preferred")
        if comp["preferred_aic"] == "step":
            print(f"    *** SHARP transition (step function wins) ***")
        else:
            print(f"    Smooth transition (sigmoid wins)")

    # --- Bootstrap confidence intervals on threshold ---
    print("\n--- Bootstrap Threshold CI (95%) ---")
    for label, J, r in [("RPST", rpst_J, rpst_r),
                          ("Kuramoto", kura_K, kura_r)]:
        bs = bootstrap_threshold(J, r)
        if bs is not None:
            print(f"  {label:12s}: J_c = {bs['mean']:.4f} ± {bs['std']:.4f}  "
                  f"95% CI = [{bs['ci_95'][0]:.4f}, {bs['ci_95'][1]:.4f}]  "
                  f"(n={bs['n_valid']})")
        else:
            print(f"  {label}: bootstrap failed")

    # --- KS tests ---
    print("\n--- Distribution Comparison (KS test) ---")
    ks_rk = ks_compare_curves(rpst_r, kura_r, "RPST", "Kuramoto")
    ks_rd = ks_compare_curves(rpst_r, diff_r, "RPST", "Diffusion")
    ks_kd = ks_compare_curves(kura_r, diff_r, "Kuramoto", "Diffusion")

    for ks in [ks_rk, ks_rd, ks_kd]:
        sig = "YES (p<0.05)" if ks["significantly_different"] else "NO"
        print(f"  {ks['labels'][0]} vs {ks['labels'][1]}: "
              f"D={ks['statistic']:.4f}  p={ks['p_value']:.4f}  "
              f"Different? {sig}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("CROSS-MODEL SUMMARY")
    print("=" * 70)

    rpst_comp = comparisons.get("RPST", {})
    kura_comp = comparisons.get("Kuramoto", {})

    rpst_prefers_step = rpst_comp.get("preferred_bic") == "step"
    kura_prefers_sigmoid = kura_comp.get("preferred_bic") == "sigmoid"
    rpst_kura_different = ks_rk["significantly_different"]

    print(f"  RPST transition best fit:      {rpst_comp.get('preferred_bic', 'N/A'):>10s}")
    print(f"  Kuramoto transition best fit:   {kura_comp.get('preferred_bic', 'N/A'):>10s}")
    print(f"  RPST ≠ Kuramoto (KS p<0.05):   {'YES' if rpst_kura_different else 'NO':>10s}")

    if rpst_prefers_step and kura_prefers_sigmoid:
        print("\n  CRITICAL RESULT: RPST and Kuramoto have QUALITATIVELY DIFFERENT")
        print("  transition types. RPST is sharp (step), Kuramoto is smooth (sigmoid).")
        print("  This is a genuine discriminating prediction.")
        discriminating = True
    elif rpst_prefers_step and not kura_prefers_sigmoid:
        print("\n  PARTIAL: RPST shows sharp transition, but Kuramoto fit is ambiguous.")
        discriminating = False
    elif not rpst_prefers_step:
        print("\n  WARNING: RPST transition is NOT best-fit by step function.")
        print("  This undermines the sharp-threshold claim.")
        discriminating = False
    else:
        print("\n  INCONCLUSIVE: Neither model shows clear preference.")
        discriminating = False

    if rpst_kura_different:
        print("  KS test confirms the curves are statistically distinguishable.")
    else:
        print("  KS test: curves are NOT statistically distinguishable.")
        print("  This is a problem — RPST should differ from Kuramoto.")

    print(f"\n  Overall: {'DISCRIMINATING PREDICTION CONFIRMED' if (discriminating and rpst_kura_different) else 'MORE WORK NEEDED'}")

    return {
        "rpst_prefers_step": rpst_prefers_step,
        "kura_prefers_sigmoid": kura_prefers_sigmoid,
        "ks_different": rpst_kura_different,
        "discriminating": discriminating and rpst_kura_different,
    }


if __name__ == "__main__":
    main()
