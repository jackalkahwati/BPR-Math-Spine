# Graviton Propagator from Boundary Stress Tensor

> **Status:** May 2026 — first-pass MVP. This derives the leading
> induced-gravity graviton propagator and derives the first finite-`p`
> boundary stress-tensor normalization correction from the Hopf/S² mode
> count. It does **not** claim the full finite-`p` holographic dictionary
> is solved.

## Target

The next bridge after horizon entropy is:

    boundary CFT stress tensor
    -> stress-tensor two-point function
    -> induced Einstein-Hilbert normalization
    -> transverse-traceless graviton propagator
    -> finite-p stress-tensor normalization correction

## Boundary Input

BPR's UV completion gives a `c = 1` compact boson boundary theory from
`U(1)_p` Chern-Simons on `S^3`. Its stress tensor has the standard CFT
two-point structure:

    <T T>_boundary ~ c / x^4

with `c = 1` per compact-boson sector and `p` sectors below the boundary
cutoff. Integrating those sectors out gives the Sakharov-induced
Einstein-Hilbert term:

    M_Pl^2 = p Lambda_b^2 / (48 pi^2)                                    (1)

This is the same normalization used in
`planck_length_from_substrate.md` and
`horizon_entropy_from_induced_gravity.md`.

## GR Limit

At long wavelengths, the induced effective action is:

    S_eff = (M_Pl^2 / 2) int d^4x sqrt(-g) R + ...

Linearizing around flat space and projecting onto the physical spatial
transverse-traceless sector gives:

    D_ij,kl(k) = P^TT_ij,kl(k) / (M_Pl^2 |k|^2)                           (2)

in natural units. The code API accepts wavevectors in `m^-1`, so it uses:

    |k|_energy = hbar c |k|_m^-1
    D_ij,kl(k) = P^TT_ij,kl(k) / (M_Pl^2 |k|_energy^2)

where:

    P_ij = delta_ij - k_i k_j / |k|^2

and:

    P^TT_ij,kl = 1/2 (P_ik P_jl + P_il P_jk - P_ij P_kl)                  (3)

This projector is symmetric, transverse, and traceless:

    k_i P^TT_ij,kl = 0
    delta_ij P^TT_ij,kl = 0

Equation (2) is the classical GR propagator normalization induced by the
BPR boundary sectors.

## Finite-p Stress-Tensor Correction

The stress-tensor two-point coefficient is additive in the number of retained
boundary compact-boson sectors. At finite CS level `p`, the Hopf map keeps
S² spherical harmonics up to:

    L_max = floor(sqrt(p))                                                (4)

so the finite S² mode count is:

    N_S2(p) = (L_max + 1)^2                                               (5)

The ideal induced-gravity normalization uses `p` sectors. The finite-cutoff
stress-tensor coefficient is therefore shifted by:

    <TT>_p / <TT>_ideal = N_S2(p) / p                                     (6)

Since the propagator is inverse to the induced Einstein-Hilbert coefficient,
the low-energy propagator correction is:

    D_p(k) / D_GR(k) = p / N_S2(p)                                        (7)

This correction approaches 1 as `p -> infinity`. For `p = 104761`,
`N_S2 = 104976`, giving:

    D_p / D_GR = 104761 / 104976 = 0.99795...

The separate EFT cutoff expansion was originally left as:

    D_p(k) = D_GR(k) * [p / N_S2(p)] * [1 + O(k^2/Lambda_b^2)]            (8)

The next sections separate two sources of curvature-squared physics:

1. pure `R²`, which affects only the scalar sector;
2. Weyl-squared/Ricci-squared, which affects the physical spin-2 sector.

## Curvature-Squared Term

The repo already contains a boundary-induced `R²` result:

    S_grav = int sqrt(-g) [(M_Pl²/2) R + (alpha/2) R² + ...]
    alpha_min = (p / 384 pi²) kappa²,   kappa = z/2                       (9)

This matters for the EFT question, but not in the naive way. Around flat
space, the pure `R²` term adds a scalar gravitational mode (the Starobinsky
scalar/scalaron). It does not shift the physical transverse-traceless spin-2
propagator. Therefore the derived spin-2 EFT coefficient from the currently
established `R²` sector is:

    eta_TT^(R²) = 0                                                       (10)

The scalar-sector mass scale in the same convention is:

    M_scalar / M_Pl = 1 / (2 sqrt(alpha_min))                             (11)

For `p = 104761`, `z = 6`, the minimal `R²` coefficient is about `249`, so
`M_scalar / M_Pl ≈ 0.0317`, matching the existing Starobinsky-sector note.

This means the current spin-2 propagator is:

    D_p^TT(k) = D_GR^TT(k) * [p / N_S2(p)]                                (12)

with no derived `k²/Lambda_b²` spin-2 correction from the known `R²` term.
To get a true energy-dependent spin-2 correction, BPR must derive induced
`R_mu_nu R^mu_nu` or Weyl-squared terms from the finite-`p` boundary
correlator.

## Weyl/Ricci-Squared Spin-2 Term

For the compact-boson boundary sectors, the universal real-scalar
heat-kernel / trace-anomaly coefficient contains:

    Gamma_1-loop includes [1 / (120 (4 pi)^2)] log(Lambda/mu) int C²       (13)

In the action convention used by the code,

    S_eff includes (beta_C / 2) int C²                                     (14)

so the finite-`p` Weyl-squared coefficient per logarithmic RG interval is:

    beta_C = N_S2(p) log(Lambda/mu) / (960 pi²)                            (15)

Using the four-dimensional identity

    C² = 2 R_mu_nu R^mu_nu - (2/3) R² + Euler                              (16)

the same TT-sector correction may be represented in a Ricci-squared basis as:

    beta_Ricci,TT = 2 beta_C                                               (17)

The low-energy spin-2 propagator has denominator

    M_Pl² k² + beta_C k^4 + ...

so, expanded below the cutoff:

    D_TT(k) = D_GR^TT(k) [1 - (beta_C Lambda_b² / M_Pl²)
                          (k²/Lambda_b²) + ...]                           (18)

and with `M_Pl² / Lambda_b² = p / (48 pi²)`:

    eta_TT^C = - beta_C (48 pi² / p)
             = - [N_S2(p) / (20 p)] log(Lambda/mu)                         (19)

For `p = 104761`, `N_S2 = 104976`, this is:

    eta_TT^C = -0.05010 * log(Lambda/mu)                                   (20)

This is the first actual derived energy-dependent spin-2 coefficient, but it
is a beta-function-like result: its absolute integrated value still depends on
the renormalization window `log(Lambda/mu)`. The code therefore keeps the
default public propagator at `eta = 0` and exposes this value through
`spin2_curvature_squared_correction(...).spin2_eft_coefficient` for callers
that want a specified RG interval.

## Physical RG Window

The abstract `log(Lambda/mu)` becomes concrete once the probe scale is chosen:

    Lambda = Lambda_b = hbar c / a
    mu     = E_probe                                                        (21)

so:

    log(Lambda/mu) = log(Lambda_b / E_probe)                                (22)

The resulting probe-specific fractional spin-2 shift is:

    delta_TT(E) = eta_TT^C(E) (E_probe / Lambda_b)^2                         (23)

with:

    eta_TT^C(E) = - [N_S2(p) / (20 p)] log(Lambda_b / E_probe)               (24)

For example, at `E_probe = 0.1 Lambda_b` and `p = 104761`:

    log(Lambda_b / E_probe) = log(10) = 2.3026
    eta_TT^C = -0.1153
    delta_TT = -0.001153                                                     (25)

So the correction is still small at one tenth of the cutoff. At ordinary
gravitational-wave energies it is fantastically smaller because of the
additional `(E_probe / Lambda_b)^2` suppression.

The helper `spin2_correction_for_probe_energy(...)` implements this convention.
It rejects probes above the boundary cutoff because the low-energy EFT
expansion no longer applies there.

For gravitational waves and black-hole ringdown signals, the natural probe
energy is the wave quantum:

    E_probe = hbar omega = hbar (2 pi f)                                  (26)

The helper `spin2_correction_for_gw_frequency(...)` implements this map. For a
detector-band signal at `f = 100 Hz`, the result is:

    E_probe = hbar 2 pi (100 Hz)
    |delta_TT| ~= 1.2e-78                                                  (27)

So this curvature-squared spin-2 correction is not an observable correction
for ordinary LIGO/Virgo/KAGRA band gravitational waves. It only becomes
numerically relevant near the BPR boundary cutoff, or in speculative early
universe / near-cutoff regimes.

The same formula also bounds the best-case size of this correction. Let:

    x = E_probe / Lambda_b                                                (28)

Then:

    delta_TT(x) = - [N_S2(p) / (20p)] x² log(1/x)                          (29)

The function `x² log(1/x)` peaks at:

    x_peak = exp(-1/2) = 0.6065                                            (30)

so:

    delta_TT,max = - N_S2(p) / (40 e p)                                    (31)

For `p = 104761`, this gives:

    delta_TT,max = -0.00922                                                (32)

So even the largest correction from this curvature-squared spin-2 term is
below one percent, and it occurs only for probes around `0.6 Lambda_b`. The
helper `spin2_max_fractional_shift(...)` returns this bound, and
`spin2_energy_ratio_for_fractional_shift(...)` returns the lower-energy branch
needed to reach a requested target shift.

## What Is Now Closed

This first-pass bridge closes the leading effective-field-theory piece:

    p boundary sectors -> M_Pl normalization -> spin-2 GR propagator

and connects it to the entropy result:

    same M_Pl normalization -> S_BH = A/(4l_P^2)

## What Remains Open

The following are still Tier-B research tasks:

1. Compute the exact finite-`p` holographic map from boundary CFT operators
   to bulk graviton modes.
2. Identify near-cutoff physical systems, if any, with probe energies around
   `0.1-0.6 Lambda_b`. Below that range the `(E_probe/Lambda_b)^2` suppression
   dominates; above it the low-energy EFT expansion becomes increasingly
   suspect.
3. Compute a full Lorentzian graviton propagator, including gauge-fixing and
   contact terms, instead of only the physical spatial TT sector.
4. Use the finite-`p` propagator to predict observationally meaningful
   corrections to gravitational waves, black-hole ringdown, or short-distance
   gravity.

The current code implementation lives in `bpr/graviton_propagator.py`.
