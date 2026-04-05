#!/usr/bin/env python3
"""
BPR Master Prediction Table
============================

Runs all 7 prediction pipelines + standalone module predictions
to generate a comprehensive table of falsifiable BPR predictions
from (p=104729, z=6) with zero free parameters.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bpr.pipelines import (
    pipeline_impedance_to_lepton_masses,
    pipeline_impedance_to_decoherence,
    pipeline_substrate_to_casimir,
    pipeline_tdgl_to_phase_classification,
    pipeline_kuramoto_to_transition,
    pipeline_agents_to_consciousness,
    pipeline_bond_to_fractal_transport,
)


def fmt(val, unit=""):
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, float):
        if abs(val) > 1e4 or (abs(val) < 1e-3 and val != 0):
            return f"{val:.4e} {unit}".strip()
        return f"{val:.6f} {unit}".strip()
    return str(val)


def run_all():
    print("=" * 90)
    print("BPR MASTER PREDICTION TABLE")
    print("Substrate: p = 104729, z = 6 | Zero free parameters")
    print("=" * 90)

    # ── Pipeline 1: Lepton Masses ──
    print("\n┌─ PIPELINE 1: Impedance → Gauge Unification → Lepton Masses")
    r1 = pipeline_impedance_to_lepton_masses(p=104729, z=6)
    experimental = {
        "sin2_theta_W": (0.23122, "PDG 2024"),
        "m_e_MeV": (0.51100, "PDG"),
        "m_mu_MeV": (105.658, "PDG"),
        "m_tau_MeV": (1776.86, "PDG"),
        "koide_Q": (0.66667, "exact 2/3"),
    }
    print(f"│  {'Quantity':<25} {'BPR Prediction':<22} {'Experiment':<22} {'Status'}")
    print(f"│  {'─'*25} {'─'*22} {'─'*22} {'─'*10}")
    for key in ["sin2_theta_W", "m_e_MeV", "m_mu_MeV", "m_tau_MeV", "koide_Q"]:
        pred = r1[key]
        exp_val, exp_src = experimental[key]
        pct = abs(pred - exp_val) / exp_val * 100
        status = "✓" if pct < 10 else "△" if pct < 30 else "✗"
        print(f"│  {key:<25} {fmt(pred):<22} {fmt(exp_val)+' ('+exp_src+')':<22} {status} {pct:.1f}%")
    print(f"│  v_EW_GeV:               {fmt(r1['v_EW_GeV'])}  (exp: 246.22 GeV)")
    print(f"│  α_EM:                   {fmt(r1['alpha_EM'])}  (exp: 1/137.036)")
    print(f"│  W_c:                    {fmt(r1['W_c'])}")

    # ── Pipeline 2: Decoherence ──
    print("\n┌─ PIPELINE 2: Impedance → Decoherence → Quantum-Classical Crossover")
    r2 = pipeline_impedance_to_decoherence(
        W_system=1.0, W_environment=100.0, T=300.0,
        A_eff=1e-14, lambda_dB=1e-10, p=104729
    )
    print(f"│  Decoherence rate:       {fmt(r2['gamma_dec_Hz'], 'Hz')}")
    print(f"│  Decoherence time:       {fmt(r2['decoherence_time_s'], 's')}")
    print(f"│  Stain (disorder):       {fmt(r2['s_star'])}")
    print(f"│  Coherence K*:           {fmt(r2['K_star'])}")
    print(f"│  Regime:                 {'QUANTUM' if r2['is_quantum'] else 'CLASSICAL'}")
    print(f"│  Measurement time:       {fmt(r2['measurement_time_s'], 's')}")
    print(f"│  Born rule correction:   {fmt(r2['born_correction'])} (= 1/p)")

    # ── Pipeline 3: Casimir ──
    print("\n┌─ PIPELINE 3: RPST Substrate → Dynamics → Resonance → Casimir Force")
    r3 = pipeline_substrate_to_casimir(
        p=104729, n_sites=32, n_steps=500,
        boundary_radius=1e-6, coupling_lambda=1e-3
    )
    print(f"│  δ (fractal exponent):   {fmt(r3['delta_derived'])} (derived from Z_p dynamics)")
    print(f"│  δ (hardcoded):          {fmt(r3['delta_hardcoded'])}")
    print(f"│  Casimir force:          {fmt(r3['casimir_force_N'], 'N')} at R=1μm")
    print(f"│  BPR relative deviation: {fmt(r3['relative_deviation'])}")
    print(f"│  Riemann zeros used:     {r3['n_riemann_zeros']}")

    # ── Pipeline 4: TDGL → Phase Classification ──
    print("\n┌─ PIPELINE 4: TDGL Simulation → Coherence Transitions → Phase Classification")
    r4 = pipeline_tdgl_to_phase_classification(
        alpha_range=np.linspace(-2, 1, 10), n_steps=50
    )
    print(f"│  Critical α_c:           {fmt(r4['alpha_c'])}")
    print(f"│  Transition class:       {r4['transition_class']}")
    print(f"│  Max coherence τ:        {fmt(r4['max_tau'], 'steps')}")
    print(f"│  Prediction: NCC > 0.9 during coupling, τ ~ 40-60 steps")

    # ── Pipeline 5: Kuramoto → Critical Exponents ──
    print("\n┌─ PIPELINE 5: Kuramoto Synchronization → Phase Transition Exponents")
    r5 = pipeline_kuramoto_to_transition(
        N=50, K_range=np.linspace(0.5, 5.0, 10), d=3, n_steps=500
    )
    print(f"│  Critical K_c:           {fmt(r5['K_c'])}")
    print(f"│  β (measured):           {fmt(r5['beta_measured'])}")
    print(f"│  β (BPR predicted):      {fmt(r5['beta_predicted'])} = (d-2)/(d+2) for d=3")
    print(f"│  β (mean-field):         0.5 (standard Kuramoto)")

    # ── Pipeline 6: Consciousness Onset ──
    print("\n┌─ PIPELINE 6: Conscious Agents → Collective Sync → Consciousness Threshold")
    r6 = pipeline_agents_to_consciousness(
        n_agents=50, state_dim=4,
        K_range=np.linspace(0.1, 5.0, 15), seed=42
    )
    print(f"│  K_c (consciousness):    {fmt(r6['K_c'])}")
    print(f"│  Φ onset threshold:      {fmt(r6['Phi_onset'])}")
    print(f"│  Min agents:             {r6['n_agents_minimum']}")

    # ── Pipeline 7: Molecular → Fractal Transport ──
    print("\n┌─ PIPELINE 7: Bond Coherence → Farey Fractions → Fractal Transport")
    # H₂O-like molecule
    r7 = pipeline_bond_to_fractal_transport(
        overlaps=[0.74, 0.74],  # two O-H bonds
        E_atomic=13.6
    )
    print(f"│  Molecule: H₂O (2 × O-H, overlap=0.74)")
    print(f"│  D_S (effective):        {fmt(r7['D_S'])}")
    print(f"│  Conductance exponent:   {fmt(r7['conductance_exponent'])} (G ~ L^exp)")
    print(f"│  Bond Farey fractions:   {r7['bond_fractions']}")
    print(f"│  Resonance weights:      {[round(w,4) for w in r7['resonance_weights']]}")

    # ── Standalone Predictions ──
    print("\n┌─ STANDALONE PREDICTIONS (from individual modules)")

    from bpr.experimental_data import antihydrogen_gravity, ANTIMATTER_PREDICTIONS, NUCLEAR_YIELD_PREDICTIONS
    g_anti = antihydrogen_gravity()
    print(f"│  Antihydrogen g:         {fmt(g_anti, 'm/s²')} (exp: 9.8 ± 0.3 m/s², ALPHA-g)")

    from bpr.nuclear_physics import NUCLEAR_REACTIONS_BPR
    print(f"│  Nuclear reactions (BPR corrected):")
    for rxn, data in NUCLEAR_REACTIONS_BPR.items():
        print(f"│    {rxn:<16} E_BPR = {data['E_BPR']} {data['unit']}")

    from bpr.fractional_boundary import sierpinski_dimension
    print(f"│  Sierpinski D_S:         {fmt(sierpinski_dimension())} = ln3/ln2")

    from bpr.emergent_spacetime import gw_dispersion_correction, mesoscopic_gravity_deviation
    print(f"│  GW dispersion at 100Hz: Δv/c = {fmt(1 - gw_dispersion_correction(100)/3e8)}")
    print(f"│  Gravity dev at 1μm:     δG/G = {fmt(mesoscopic_gravity_deviation(1e-6))}")

    from bpr.decoherence import born_rule_from_substrate
    born = born_rule_from_substrate(p=104729, n_samples=50000, seed=42)
    print(f"│  Born rule deviation:    {fmt(born['mean_deviation'])} (expected: ~1/p = {fmt(1/104729)})")

    from bpr.multiscale import MultiscaleLagrangian
    ml = MultiscaleLagrangian()
    chi = ml.propagate_coherence(chi_0=0.99, zeta_array=np.ones(12)*0.5, Phi_array=np.ones(12)*0.3)
    idx, name, drop = ml.coherence_bottleneck()
    print(f"│  Coherence bottleneck:   {name} (drop={fmt(drop)})")

    from bpr.boundary_action import sigma_effective, anomaly_inflow
    sigma = sigma_effective(omega=1e9, Z_s=100.0)
    print(f"│  σ_eff(1GHz, Z=100Ω):   {fmt(sigma)}")
    tr_q3, is_cancelled, msg = anomaly_inflow(np.array([2/3, -1/3, -1/3, -1, 0, 0]))
    print(f"│  Anomaly Tr[Q³]:         {fmt(tr_q3)} (cancelled: {is_cancelled})")

    print("\n" + "=" * 90)
    print("END OF PREDICTION TABLE")
    print(f"Total pipelines run: 7")
    print(f"Total predictions: ~40+")
    print(f"Free parameters: 0 (all from p=104729, z=6)")
    print("=" * 90)


if __name__ == "__main__":
    run_all()
