# Scalaron Sector from Boundary R2

> **Status:** May 2026 — scalar-sector audit. The boundary-induced `R2` term
> derives the Starobinsky scalar shape, but the minimal coefficient does not
> match the observed scalar amplitude without an additional normalization.

## Boundary Input

The BPR boundary heat-kernel calculation gives the minimal curvature-squared
term:

    S_grav = int sqrt(-g) [(M_Pl^2 / 2) R + (alpha / 2) R^2 + ...]

with:

    alpha_min = (p / 384 pi^2) kappa^2
    kappa = z / 2

For `p = 104761`, `z = 6`, this gives:

    alpha_min ~= 248.8

## Scalaron Dual

In the repo convention, the `R2` term is equivalent to an Einstein-frame scalar
with:

    M / M_Pl = 1 / (2 sqrt(alpha))

and potential:

    V(phi) = (3 M_Pl^4 / 16 alpha)
             (1 - exp(-sqrt(2/3) phi / M_Pl))^2

So the minimal BPR boundary coefficient gives:

    M / M_Pl ~= 0.0317
    V0 / M_Pl^4 ~= 7.54e-4

The scalaron couples universally to trace stress-energy with the standard
Einstein-frame strength:

    g_trace = 1 / sqrt(6)   in 1/M_Pl units

## Inflation Observables

BPR's existing e-fold estimate is:

    N = p^(1/3) (1 + 1/d)

For `d = 3`, `N ~= 62.9`. The Starobinsky shape then gives:

    n_s = 1 - 2/N ~= 0.968
    r   = 12/N^2  ~= 0.003

These shape predictions remain the strong part of the scalar sector.

## Amplitude Audit

The scalar amplitude in this convention is:

    A_s = N^2 / (96 pi^2 alpha)

Using the minimal boundary `alpha_min ~= 248.8`:

    A_s,min ~= 0.0168

The observed value is:

    A_s,obs ~= 2.1e-9

So the minimal `R2` coefficient overpredicts the scalar amplitude by about:

    A_s,min / A_s,obs ~= 8.0e6

Equivalently, the coefficient needed to match the observed amplitude is:

    alpha_required = N^2 / (96 pi^2 A_s,obs) ~= 2.0e9

That is about:

    alpha_required / alpha_min ~= 8.0e6

## Winding/Anyon Normalization Diagnostic

The previous qualitative winding estimate used:

    F_old = 1 + W_c / W_bare
    W_c = sqrt(z/2)
    W_bare = sqrt(log p)

For `p = 104761`, `z = 6`:

    F_old ~= 1.51

This is nowhere near the required:

    F_required ~= 7.98e6

Other simple BPR-scale combinations are also only diagnostic, not derivations:

    p                 ~= 1.05e5
    p log p           ~= 1.21e6
    p (log p)^2       ~= 1.40e7
    p^(4/3)           ~= 4.94e6
    p^(3/2)           ~= 3.39e7

Some are within an order of magnitude, but none is an established
coefficient-level derivation of the scalar amplitude. Matching the observed
amplitude is equivalent to an effective boundary-sector count:

    p_eff = p * F_required ~= 8.36e11

The code implementation `scalaron_normalization_diagnostic(...)` records this
status as `open` unless an existing candidate lands within 10% of the required
gap.

## Interpretation

This closes one question and opens a sharper one:

- `R2` **does** derive the Starobinsky scalar potential shape.
- The resulting `n_s` and `r` predictions are still internally clean.
- The minimal one-loop boundary coefficient is far too small to set the
  observed amplitude.
- The proposed winding/anyon-loop enhancement must therefore supply a large
  normalization factor, not just an order-one correction.

The code implementation lives in `bpr/graviton_propagator.py` as
`scalaron_sector_from_boundary_r2(...)`.
