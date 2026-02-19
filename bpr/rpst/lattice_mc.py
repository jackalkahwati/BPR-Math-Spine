"""
Lattice Monte Carlo Validation of Z_p Coarse-Graining
======================================================

Validates that the discrete Z_p Hamiltonian (Eq. 1) produces the
predicted continuum coupling constants (kappa, xi) when simulated
directly via Monte Carlo.

This provides a non-trivial computational check: the analytic
coarse-graining used in boundary_energy.py should agree with
direct numerical simulation of the lattice model.

Usage
-----
>>> from bpr.rpst.lattice_mc import LatticeMCValidator
>>> validator = LatticeMCValidator(p=17, N=100, J=1.0)
>>> result = validator.validate(n_sweeps=5000)
>>> print(f"kappa_MC = {result['kappa_mc']:.3f}, kappa_theory = {result['kappa_theory']:.3f}")
>>> print(f"Agreement: {result['kappa_agrees']}")

Note: Uses small p (17-101) for computational feasibility.
The coarse-graining should work for ALL primes -- if it fails
for small p, the derivation has a problem.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class LatticeMCValidator:
    """Monte Carlo simulation of Z_p lattice Hamiltonian.

    Simulates the discrete model:
        H = -J sum_{<i,j>} cos(2*pi*(q_i - q_j) / p)

    on a 1D ring of N sites using Metropolis updates, then
    measures the effective kappa and xi from the two-point
    correlation function.

    Parameters
    ----------
    p : int
        Prime modulus (use small primes 17-101 for speed).
    N : int
        Number of lattice sites.
    J : float
        Coupling strength (in units of k_B T = 1).
    seed : int
        Random seed for reproducibility.
    """
    p: int = 17
    N: int = 100
    J: float = 1.0
    seed: int = 42

    def _energy(self, config: np.ndarray) -> float:
        """Total energy for a configuration q_i in {0, ..., p-1}."""
        diffs = np.diff(config, append=config[0])  # periodic BC
        phases = 2.0 * np.pi * diffs / self.p
        return -self.J * np.sum(np.cos(phases))

    def _single_site_update(self, config: np.ndarray, rng: np.random.Generator,
                             beta: float) -> np.ndarray:
        """One Metropolis sweep over all sites."""
        N = len(config)
        for i in range(N):
            old_q = config[i]
            new_q = rng.integers(0, self.p)
            # Compute local energy change (only neighbors matter)
            left = config[(i - 1) % N]
            right = config[(i + 1) % N]
            dE = -self.J * (
                np.cos(2 * np.pi * (new_q - left) / self.p)
                + np.cos(2 * np.pi * (new_q - right) / self.p)
                - np.cos(2 * np.pi * (old_q - left) / self.p)
                - np.cos(2 * np.pi * (old_q - right) / self.p)
            )
            if dE < 0 or rng.random() < np.exp(-beta * dE):
                config[i] = new_q
        return config

    def run_simulation(self, n_sweeps: int = 5000, n_thermalize: int = 1000,
                       beta: float = 1.0) -> Dict:
        """Run Monte Carlo simulation and measure observables.

        Parameters
        ----------
        n_sweeps : int
            Number of measurement sweeps after thermalization.
        n_thermalize : int
            Number of thermalization sweeps (discarded).
        beta : float
            Inverse temperature (1 / k_B T in units of J).

        Returns
        -------
        dict with 'energies', 'correlations', 'config_samples'
        """
        rng = np.random.default_rng(self.seed)
        config = rng.integers(0, self.p, size=self.N)

        # Thermalize
        for _ in range(n_thermalize):
            config = self._single_site_update(config, rng, beta)

        # Measure
        energies = []
        correlations = np.zeros(self.N // 2)
        n_corr_samples = 0

        for sweep in range(n_sweeps):
            config = self._single_site_update(config, rng, beta)
            energies.append(self._energy(config))

            # Measure two-point correlation function every 10 sweeps
            if sweep % 10 == 0:
                phases = 2.0 * np.pi * config / self.p
                for r in range(self.N // 2):
                    corr = np.mean(np.cos(phases - np.roll(phases, r)))
                    correlations[r] += corr
                n_corr_samples += 1

        correlations /= max(n_corr_samples, 1)

        return {
            'energies': np.array(energies),
            'correlations': correlations,
            'mean_energy': float(np.mean(energies)),
            'std_energy': float(np.std(energies)),
        }

    def measure_kappa(self, correlations: np.ndarray) -> float:
        """Extract effective kappa from correlation function.

        For the continuum model with energy density u = (kappa/2)|grad phi|^2,
        the correlation function is:

            C(r) = <cos(phi_i - phi_{i+r})> = exp(-r / xi)

        The gradient energy density is:
            <|grad phi|^2> = <(phi_{i+1} - phi_i)^2> ~ 1/kappa

        So kappa ~ 1 / (1 - C(1)) for small (1 - C(1)).
        """
        if len(correlations) < 2:
            return 0.0
        c1 = correlations[1]
        if c1 >= 1.0:
            return float('inf')
        if c1 <= 0:
            return 0.0
        return 1.0 / (1.0 - c1)

    def measure_xi(self, correlations: np.ndarray) -> float:
        """Extract correlation length xi from exponential fit.

        C(r) ~ exp(-r/xi) => xi = -1 / ln(C(1)/C(0))
        """
        if len(correlations) < 2:
            return 0.0
        c0 = max(correlations[0], 1e-10)
        c1 = max(correlations[1], 1e-10)
        ratio = c1 / c0
        if ratio <= 0 or ratio >= 1:
            return float('inf') if ratio >= 1 else 0.0
        return -1.0 / np.log(ratio)

    def theoretical_kappa(self) -> float:
        """Predicted kappa from analytic coarse-graining.

        For a 1D ring: z = 2, kappa = z/2 = 1.
        """
        z = 2  # 1D ring coordination number
        return z / 2.0

    def theoretical_xi(self) -> float:
        """Predicted correlation length from analytic formula.

        xi = a * sqrt(ln(p)) where a = 2*pi*R/N (lattice spacing for ring).
        For R = 1 (unit ring): a = 2*pi/N.
        """
        a = 2.0 * np.pi / self.N
        return a * np.sqrt(np.log(self.p))

    def validate(self, n_sweeps: int = 5000, n_thermalize: int = 1000,
                  beta: float = 1.0) -> Dict:
        """Full validation: simulate and compare to theory.

        Returns
        -------
        dict with:
            'kappa_mc' : measured kappa
            'kappa_theory' : predicted kappa
            'kappa_agrees' : bool (within 30% tolerance)
            'xi_mc' : measured correlation length
            'xi_theory' : predicted xi
            'xi_agrees' : bool (within 50% tolerance)
            'mean_energy' : average energy per site
        """
        sim = self.run_simulation(n_sweeps, n_thermalize, beta)
        kappa_mc = self.measure_kappa(sim['correlations'])
        xi_mc = self.measure_xi(sim['correlations'])
        kappa_th = self.theoretical_kappa()
        xi_th = self.theoretical_xi()

        kappa_ratio = kappa_mc / kappa_th if kappa_th > 0 else float('inf')
        xi_ratio = xi_mc / xi_th if xi_th > 0 else float('inf')

        return {
            'p': self.p,
            'N': self.N,
            'J': self.J,
            'kappa_mc': kappa_mc,
            'kappa_theory': kappa_th,
            'kappa_ratio': kappa_ratio,
            'kappa_agrees': 0.7 < kappa_ratio < 1.3,
            'xi_mc': xi_mc,
            'xi_theory': xi_th,
            'xi_ratio': xi_ratio,
            'xi_agrees': 0.5 < xi_ratio < 2.0,
            'mean_energy': sim['mean_energy'],
            'mean_energy_per_site': sim['mean_energy'] / self.N,
        }


def validate_coarse_graining(primes: tuple = (7, 11, 17, 23, 31, 41, 53),
                              N: int = 100, n_sweeps: int = 3000) -> Dict:
    """Validate coarse-graining across multiple primes.

    Tests that the derived kappa = z/2 and xi = a*sqrt(ln(p))
    hold for different values of p.

    Parameters
    ----------
    primes : tuple of int
        Prime moduli to test.
    N : int
        Lattice size.
    n_sweeps : int
        Monte Carlo sweeps per prime.

    Returns
    -------
    dict with 'results' (list of per-prime results) and 'all_agree' (bool).
    """
    results = []
    for p in primes:
        validator = LatticeMCValidator(p=p, N=N, J=1.0, seed=42 + p)
        result = validator.validate(n_sweeps=n_sweeps)
        results.append(result)

    all_kappa_agree = all(r['kappa_agrees'] for r in results)
    all_xi_agree = all(r['xi_agrees'] for r in results)

    return {
        'results': results,
        'all_kappa_agree': all_kappa_agree,
        'all_xi_agree': all_xi_agree,
        'all_agree': all_kappa_agree and all_xi_agree,
        'summary': {
            'primes_tested': list(primes),
            'kappa_ratios': [r['kappa_ratio'] for r in results],
            'xi_ratios': [r['xi_ratio'] for r in results],
        },
    }
