# Horizon Entropy from Induced Gravity

> **Status:** May 2026 — replaces the earlier heuristic "raw p-state
> counting divided by 4 ln p" explanation with the coefficient-level
> Sakharov/Wald derivation.

## Claim

BPR gets the Bekenstein-Hawking coefficient from the same induced
Einstein-Hilbert term that fixes the Planck-to-boundary cutoff hierarchy:

    S_BH = A / (4 l_P^2)

The important point is that the coefficient is not obtained by assigning
`p` independent labels to every Planck cell and then choosing a normalization.
It is obtained by computing the entropy of the induced gravitational action.

## Setup

From `planck_length_from_substrate.md`, integrating out the `p` boundary
sectors below the boundary cutoff `Lambda_b` gives:

    M_Pl^2 = p Lambda_b^2 / (48 pi^2)                                    (1)

This document uses the same unreduced Planck-mass convention as the codebase:

    l_P = hbar c / M_Pl                                                   (2)

so the Einstein-Hilbert action has the standard long-distance normalization
and the black-hole entropy is the Wald entropy of that action.

## Entropy

For an Einstein-Hilbert action in this convention, the Wald entropy of a
stationary horizon is:

    S_Wald = A M_Pl^2 / (4 (hbar c)^2)                                    (3)

Substituting (1):

    S_Wald = A p Lambda_b^2 / (192 pi^2 (hbar c)^2)                       (4)

The BPR boundary spacing is:

    a = hbar c / Lambda_b
    a / l_P = sqrt(p / (48 pi^2))                                         (5)

Using (5) in (4):

    S_Wald = A / (4 l_P^2)                                                (6)

So the `1/4` coefficient is fixed by the Einstein-Hilbert/Wald normalization
of the induced gravity term, not by an independent entropy normalization.

## Why raw p-state counting is not enough

The naive count of `p` labels per boundary cutoff cell gives:

    S_raw = (A / a^2) ln p                                                (7)

With the Sakharov spacing (5), this becomes:

    S_raw / S_BH = 192 pi^2 ln(p) / p                                     (8)

For `p = 104761`, this ratio is about `0.209`. It has the correct area
scaling, but the wrong coefficient. Therefore raw winding count is only a
microscopic heuristic unless supplemented by the full induced-gravity
normalization.

## Consequence for Jacobson's argument

Once (6) is established, Jacobson's local horizon derivation applies:

    delta Q = T dS
    T = hbar kappa / (2 pi)
    dS = delta A / (4 l_P^2)

Together with Raychaudhuri focusing of local Rindler horizons, this yields
the Einstein equation:

    R_ab - (1/2) R g_ab + Lambda g_ab = (8 pi G / c^4) T_ab

Thus BPR's clean bridge to classical gravity is:

    CS boundary sectors -> Sakharov induced EH action
    -> Wald horizon entropy A/(4l_P^2)
    -> Jacobson thermodynamic derivation of Einstein equations

## Remaining caveat

This closes the entropy coefficient inside the induced-gravity effective
description. It does not by itself compute the finite-`p` graviton propagator
or the full boundary-to-bulk holographic dictionary. Those remain the
quantum-gravity tasks described in `tier_B_attack_plans.md`.
