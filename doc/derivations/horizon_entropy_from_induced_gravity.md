# Horizon Entropy from Induced Gravity

> **Status:** May 2026 — replaces the earlier heuristic "raw p-state
> counting divided by 4 ln p" explanation with the coefficient-level
> Sakharov/Wald derivation.

## Claim

The stipulated induced Einstein-Hilbert action has the standard Bekenstein-Hawking coefficient when its Planck convention is handled consistently:

    S_BH = A / (4 l_P^2)

The important point is that the coefficient is not obtained by assigning
`p` independent labels to every Planck cell and then choosing a normalization.
It is obtained by computing the entropy of the induced gravitational action.

## Setup

From `planck_length_from_substrate.md`, integrating out the `p` boundary
sectors below the boundary cutoff `Lambda_b` gives:

    M_Pl^2 = p Lambda_b^2 / (48 pi^2)                                    (1)

Normalization correction2026-09-12: equation(1), when used in the action M_Pl² R/2, defines the **reduced** Planck energy, not the unreduced energy. Therefore

    l_P = hbar c / (sqrt(8pi) M_Pl)                                      (2)

The black-hole entropy below is conditional Wald entropy of that stipulated Einstein-Hilbert action, not an independently derived microscopic count. See [action normalization and identifiability](gravity_consistency_2026-09-12.md).

## Entropy

For the Einstein-Hilbert term in this convention, the Wald entropy of a stationary horizon is (A is Einstein-frame area if an R² sector has been transformed to that frame):

    S_Wald = 2 pi A M_Pl^2 / (hbar c)^2                                  (3)

Substituting (1):

    S_Wald = A p Lambda_b^2 / (24 pi (hbar c)^2)                          (4)

The BPR boundary spacing is:

    a = hbar c / Lambda_b
    a / l_P = sqrt(p / (6 pi))                                           (5)

Using (5) in (4):

    S_Wald = A / (4 l_P^2)                                                (6)

So the `1/4` coefficient is fixed by the Einstein-Hilbert/Wald normalization of the supplied induced gravity term, not by an independent entropy normalization. For the full Jordan-frame action M²R/2+alpha R²/2, S=2pi integral(M²+2alpha R)dA_J in natural units, not simply2pi A_J M² at nonzero curvature. Its area transformation dA_E=F dA_J recovers the Einstein-frame expression. The area-law calculation alone does not establish microscopic entanglement or the prerequisites of a holographic entropy dictionary.

## Why raw p-state counting is not enough

The naive count of `p` labels per boundary cutoff cell gives:

    S_raw = (A / a^2) ln p                                                (7)

With the Sakharov spacing (5), this becomes:

    S_raw / S_BH = 24 pi ln(p) / p                                       (8)

For `p = 104761`, this ratio is about `0.00832`, not the former `0.209` obtained from mixed conventions. It has the correct area
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

This gives the following conditional chain, provided local equilibrium, horizon thermodynamics and the effective curved-spacetime description are independently supplied:

    CS boundary sectors -> Sakharov induced EH action
    -> Wald horizon entropy A/(4l_P^2)
    -> Jacobson thermodynamic derivation of Einstein equations

## Remaining caveat

This checks the entropy coefficient inside the stipulated induced-gravity effective description. It does not establish the field-content/regulator assumptions, eliminate bare/counterterm freedom, select vacuum energy, or construct dynamical geometry from the quantum ring. It also does not compute the finite-`p` graviton propagator or the full boundary-to-bulk dictionary. The numerical entropy A/(4l_P²) is unchanged by this convention repair.
