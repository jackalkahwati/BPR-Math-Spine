# Scalaron Sector from Boundary R2

> **Normalization correction, 2026-09-12:** This note uses the frozen action in
> `gravity_consistency_2026-09-12.md`, with reduced `M_Pl`: `M_Pl² R/2 + alpha R²/2`.
> The old mass, plateau and amplitude normalization were inconsistent with that
> action. At unchanged parameters, the required enhancement is `2/3` of its old
> value and every fixed candidate/required ratio is `3/2` of its old value.
> In particular, the historical `5/3` percent-level near-match and all numerical
> closure implications are superseded. No candidate is retuned or replaced.
> The scalar shape is conditional on the supplied induced-action assumptions;
> the amplitude and boundary-to-bulk coefficient derivation remain open.

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

For the stated action, the `R2` term is equivalent to an Einstein-frame scalar
with mass `m_s`:

    m_s / M_Pl = 1 / sqrt(6 alpha)

and potential:

    V(phi) = (M_Pl^4 / (8 alpha))
             (1 - exp(-sqrt(2/3) phi / M_Pl))^2

So the minimal BPR boundary coefficient gives:

    m_s / M_Pl ~= 0.02588
    V0 / M_Pl^4 ~= 5.025e-4

Here `M_Pl` is reduced. The auxiliary-field/Weyl calculation and the
small-field mass check are given in `gravity_consistency_2026-09-12.md`.

The scalaron couples universally to trace stress-energy with the standard
Einstein-frame strength:

    g_trace = 1 / sqrt(6)   in 1/M_Pl units

## Inflation Observables

BPR's existing e-fold estimate is:

    N = p^(1/3) (1 + 1/d)

For `d = 3`, `N ~= 62.9`. The Starobinsky shape then gives:

    n_s = 1 - 2/N ~= 0.968
    r   = 12/N^2  ~= 0.003

These leading shape relations are unchanged by the normalization repair;
they remain conditional on the pure-`R2` action and the supplied e-fold estimate.

## Amplitude Audit

At leading large-`N` slow roll, the scalar amplitude in this convention is:

    A_s = N^2 / (144 pi^2 alpha)

Using the minimal boundary `alpha_min ~= 248.8` and the unchanged `N` above:

    A_s,min ~= 0.01117

The existing comparison input is retained, not re-estimated:

    A_s,obs ~= 2.1e-9

So the minimal `R2` coefficient overpredicts that amplitude by about:

    A_s,min / A_s,obs ~= 5.321e6

Equivalently, the inverse coefficient needed to match this input is:

    alpha_required = N^2 / (144 pi^2 A_s,obs) ~= 1.324e9

That is about:

    alpha_required / alpha_min ~= 5.321e6

This inverse is a calibration, not a prediction of the observed amplitude.
The old `96 pi^2`, `0.0168`, `2.0e9` and `8.0e6` values are superseded.

## Winding/Anyon Normalization Diagnostic

The previous qualitative winding estimate used:

    F_old = 1 + W_c / W_bare
    W_c = sqrt(z/2)
    W_bare = sqrt(log p)

For `p = 104761`, `z = 6`:

    F_old ~= 1.51

This is nowhere near the required:

    F_required ~= 5.321e6

Other simple BPR-scale combinations are also only diagnostic, not derivations:

    p                 ~= 1.05e5
    p log p           ~= 1.21e6
    p (log p)^2       ~= 1.40e7
    p^(4/3)           ~= 4.94e6
    p^(3/2)           ~= 3.39e7

Some are within an order of magnitude, but none is an established
coefficient-level derivation of the scalar amplitude. Matching the observed
amplitude is equivalent to an effective boundary-sector count:

    p_eff = p * F_required ~= 5.574e11

The code implementation `scalaron_normalization_diagnostic(...)` compares the
existing fixed candidates with the corrected gap. Its unchanged candidate set
now has `p_four_thirds` as the closest candidate, with relative error about
`7.18568%`; this is the existing numerical diagnostic, not a newly proposed
normalization. A proximity status from a numerical threshold is not a
coefficient derivation or a closed physical normalization. No candidate is
retuned or added to recover the old match.

## Compact-Boson Mode Count

Using the `c = 1` compact boson spectrum,

    h(m,n) = m^2/R^2 + n^2 R^2
    R^2 = z/2 = 3

and the finite-p cutoff:

    L_max = floor(sqrt(p)) = 323

there are two useful diagnostic counts:

    full square lattice:      (2 L_max + 1)^2 - 1 = 418608
    h <= L_max ellipse:      1014

If the full square-lattice count is weighted by the topological log, it gives:

    418608 log(p) ~= 4.84e6

which is about:

    0.9094 * F_required

The stricter conformal-dimension ellipse gives:

    1014 log(p) ~= 1.17e4

which is far too small. The unchanged counts are diagnostics only. The
corrected residual after full-lattice counting is about `1.100`, not the
historical `1.65`; this arithmetic is not a prescription for a new loop weight.

The helper `compact_boson_mode_normalization_diagnostic(...)` records these
counts. The square-lattice ratio `0.9094098996` now falls within its unchanged
10% threshold and receives the numerical `candidate` status, superseding the
old `open` proximity status. This is not a coefficient-level
heat-kernel/anyon-loop calculation or a physical closure.

## Residual Loop-Weight Diagnostic

After the full finite lattice count, the corrected remaining factor is:

    F_residual = F_required / (418608 log(p)) ~= 1.0996

Keep the previously proposed radius factor fixed:

    F_R = 1 + 2/R^2 = 5/3 ~= 1.667

For `R^2 = 3`, this gives:

    (418608 log(p)) * (5/3) ~= 1.5157 * F_required

The historical `1.649` residual, `1.010` ratio and percent-level near-match
claim are superseded. The fixed `5/3` candidate now exceeds the required
coefficient by about 51.6% and is `open`, not a normalization closure or the
old `near_match_unproven` result. The code records the diagnostic through
`compact_boson_residual_loop_weight_diagnostic(...)`. Within its unchanged
candidate list, `self_dual_average` now has the smallest residual relative
error, `5.00961%`, but still reports `open`; it is not promoted as a replacement
physical weight. Physical meaning would require an independently derived
anyon/heat-kernel loop weight. No replacement near-match is sought here.

## Radius-Current Heat-Kernel Weight

The first coefficient-level ansatz is now explicit. Treat the scalar `R2`
coefficient as:

    alpha_full ~= alpha_min * N_lattice * log(p) * F_R

where `N_lattice = 418608` and the local compact-boson heat-kernel insertion is
the identity trace plus the two chiral current contractions:

    F_R = 1 + J_L weight + J_R weight
        = 1 + G^{theta theta} + G^{theta theta}
        = 1 + 2/R^2

For the BPR radius `R^2 = z/2 = 3`:

    F_R = 1 + 2/3 = 5/3 ~= 1.667

so:

    418608 log(p) * (5/3) ~= 8.06e6
    [418608 log(p) * (5/3)] / F_required ~= 1.5157

This is the unchanged current-insertion ansatz, not a closure of the corrected
numerical gap. The old percent-level closure statement is superseded.
Independently, a single `U(1)_p` Chern-Simons edge theory is chiral. The
two-current `J_L`, `J_R` factor requires either a doubled/non-chiral boundary
completion or an explicit derivation of how the bulk scalar loop pairs the
chiral edge with its conjugate.

The helper `compact_boson_heat_kernel_loop_weight(...)` evaluates this fixed
ansatz and now reports `open`. Its former `candidate_under_current_ansatz`
proximity implication is superseded by the corrected ratio; the CS dictionary
remains open.

## CS/WZW Compatibility Rule

The next step is to ask whether the boundary theory permits this insertion in
a constrained way, or whether it was merely chosen because it fits. Under the
abelian CS/WZW dictionary:

    U(1)_p Chern-Simons  ->  chiral U(1)_p WZW edge
    doubled/non-chiral completion  ->  c = 1 compact boson with J_L, J_R

the boundary operator content has:

    identity operator:       neutral, spinless
    chiral currents:         J_L, J_R after non-chiral completion
    vertex operators:        V_{m,n}, generally charged under momentum/winding
    descendants:             higher-dimension derivative insertions

The scalar `R2` loop insertion must be:

    neutral under the compact U(1)
    spinless on the boundary
    parity-even between left and right movers
    local and marginal at leading order

Those conditions exclude charged vertex operators, one-sided chiral currents,
and higher descendants. In the doubled/non-chiral compact-boson ansatz, the
surviving compatible leading insertion is:

    identity + radius-current pair

with the two allowed current contractions:

    J_L contribution:  G^{theta theta} = 1/R^2
    J_R contribution:  G^{theta theta} = 1/R^2

Therefore the compatibility rule motivates:

    F_R = 1 + 1/R^2 + 1/R^2 = 1 + 2/R^2

and for `R^2 = 3`:

    F_R = 5/3

This is a compatibility argument for the neutral scalar operator filters of
the abelian CS/WZW boundary theory, not a numerical near-match or a coefficient
proof. The normalization correction does not change these filters. Two things
remain open:

    1. chirality completion: why the scalar loop pairs the chiral CS edge with
       a conjugate edge or equivalent non-chiral completion;
    2. local and bulk normalization: why the identity/current mixing enters
       the induced 4D `R2` coefficient with exactly this unit normalization.

The helper `compact_boson_cs_wzw_selection_rule(...)` now reports `open` at
the default parameters because the fixed `5/3` coefficient/required ratio is
`1.515683166`, with
`dictionary_status = chirality_and_bulk_normalization_open`.

## Interpretation

The corrected interpretation is conditional, not a closed scalar-amplitude
prediction:

- The supplied pure-`R2` action has the Starobinsky scalar potential shape.
- Leading `n_s` and `r` relations are unchanged at the same e-fold estimate.
- The minimal one-loop coefficient is far too small for the existing amplitude
  comparison input; inverse matching is calibration.
- The proposed large enhancement remains unproved. The fixed `5/3` candidate
  no longer has the historical percent-level numerical match.

The code implementation lives in `bpr/graviton_propagator.py` as
`scalaron_sector_from_boundary_r2(...)`.
