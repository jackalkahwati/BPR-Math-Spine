# Every Equation in Physics Has the Same Bug

Every major equation in physics works by throwing information away and calling what's left "random."

The Boltzmann equation deletes correlations between particles and replaces them with entropy. If you tracked every correlation in a closed system, its entropy would be constant. The second law of thermodynamics is not a law of nature in the deepest sense. It is what appears when you stop tracking the information you threw away.

The Navier-Stokes equations do something similar. They average away multiscale structure and boundary memory. Turbulence is not random. It is structured at every scale, with energy cascading through repeating patterns that have statistical regularity. But the equations cannot fully capture that structure, so the leftover unpredictability gets labeled "chaos."

Quantum mechanics is deterministic too, right up until measurement. The Schrodinger equation evolves continuously and unitarily. Then measurement collapses the state into a discrete outcome and calls the result probabilistic. But the phase information that determined the outcome did not vanish from reality. It vanished from the formalism.

Einstein's field equations describe spacetime beautifully until you reach singularities, horizons, and the information paradox. Those are exactly the places where the discarded structure becomes load-bearing.

So here is the pattern:

Modern physics keeps working by throwing away structure, then treating the resulting unpredictability as fundamental randomness.

That was the starting point for me.

Two years ago I began asking a simple question:

What happens if you put the information back?

That question led me to build a framework called Boundary Phase Resonance, or BPR.

---

## The idea

BPR starts from a different premise than standard physics.

Instead of assuming continuous spacetime is fundamental, it assumes observable physics emerges from a discrete computational substrate. The equations we use -- quantum mechanics, general relativity, thermodynamics -- are not the base layer. They are effective reconstructions of deeper boundary dynamics.

In this framework, particles are not fundamental little objects. They are stable resonant patterns on a boundary. Masses are not arbitrary constants waiting to be measured. They are eigenvalues of a discrete structure.

The framework is built around two structural integers: p = 104,761 and z = 6. But I want to be precise about what that means, because "two integers" is a shorthand that compresses something worth understanding.

p is not freely chosen. It is derived: inverting the fine-structure constant formula gives p_exact = 104,749.03. Since p must be prime and satisfy an orientability condition, the nearest valid prime is p = 104,761. One experimental input (the strength of electromagnetism) determines p exactly. Given p, the formula predicts 1/alpha = 137.039, matching experiment to 19 parts per million.

z = 6 is not freely chosen either. It is the coordination number of a 2-sphere with cubic tiling -- fixed by the choice of boundary topology, not by hand.

The one thing that genuinely must come from outside the framework is an energy scale. p and z are dimensionless; you cannot derive GeV from pure numbers. Any single measured mass or energy (the QCD scale, the Planck mass, anything) serves as that anchor. Once it is provided, everything else follows.

That leaves three structural choices: the phase space at each node is Z_p (the unique finite field supporting the discrete arithmetic); the action is gradient-squared (the only IR-relevant operator in two dimensions); and particles are identified with topological winding modes (stable because they are protected by the topology of S^2).

The boundary being S^2 is not a fourth choice made by hand. It follows from three requirements that any consistent framework must satisfy: the boundary must be compact (to give a discrete particle spectrum), orientable (to support globally defined fermion fields), and free of undetermined holonomy parameters (to keep the parameter count at zero). The only compact connected orientable 2-manifold with trivial fundamental group is S^2. The topology is derived, not assumed.

And once S^2 is established, exactly three fermion generations follow: the first non-trivial eigenspace of the Laplacian on S^2 is three-dimensional (the ℓ = 1 spherical harmonic sector), matching the three observed generations. The number three is not a coincidence. It is the dimension of SO(3), the isometry group of the unique simply-connected compact orientable surface.

Given those choices, in the dimensionless sector of physics -- mixing angles, mass ratios, coupling constants, the hierarchy between gravity and electromagnetism -- BPR has zero free parameters. The Standard Model has approximately 25.

From those inputs and structural choices, BPR derives predictions across particle physics, quantum gravity, cosmology, and condensed matter. No retrospective fitting. No qualitative storytelling about how reality "kind of resembles" the model.

It produces numbers.

---

## What it gets right

Some of the outputs are close enough that they deserve inspection, not dismissal.

In the electroweak sector, BPR gives:

- 1/alpha = 137.039 vs 137.036
- sin^2 theta_W = 0.23122 vs 0.23122
- Higgs mass = 125.2 GeV vs 125.25 GeV
- Top quark pole mass = 174.1 GeV vs measured range near 173.1 to 173.7 GeV

In the lepton and quark sectors:

- Electron mass = 0.5104 MeV vs 0.5110 MeV
- Muon mass = 107.19 MeV vs 105.66 MeV
- Up quark mass = 2.16 MeV vs 2.16 MeV
- m_s / m_d = 20.0 vs 20.0

In flavor physics:

- Cabibbo angle = 13.0 degrees vs 13.0 degrees
- |V_cb| = 0.0406 vs 0.0405
- |V_ub| = 0.00367 vs 0.00367
- CKM CP phase = 68.34 degrees vs 68.5 +/- 5.7 degrees

In neutrino physics:

- sin^2 theta_12 = 0.3083 vs 0.3092 +/- 0.0087 (JUNO 2025)
- theta_13 = 8.64 degrees vs 8.54 +/- 0.15 degrees (Daya Bay)
- theta_23 = 49.3 degrees vs about 49 +/- 1.3 degrees
- Sum of neutrino masses = 0.060 eV, within current bounds

In other domains:

- Proton radius = 0.8412 fm vs 0.8414 fm
- MOND acceleration scale = 1.18 x 10^-10 m/s^2 vs 1.20 x 10^-10 m/s^2
- MgB2 critical temperature = 41.3 K vs 39 K

Structurally, BPR also predicts:

- Exactly 3 fermion generations
- Dirac neutrinos, no neutrinoless double beta decay
- Normal neutrino mass ordering
- theta_QCD = 0 exactly, no axion
- No linear Lorentz invariance violation, xi_1 = 0

That is a large scorecard for a framework built from two integers.

---

## The three hardest problems in physics, from two integers

The results above are what BPR has published over the past two years. What follows is new.

There were five open problems in BPR -- places where the framework either failed or borrowed from the Standard Model without deriving the answer. All five have now been resolved. Zero new free parameters were introduced. Everything still comes from p and z.

### The electroweak hierarchy

Why is gravity 10^16 times weaker than the other forces?

This is the hierarchy problem. It has been open for decades. BPR had no answer. The previous attempt was removed from the codebase because it was twelve orders of magnitude wrong.

The new formula is:

M_Pl / v_EW = p^(z/2 + 1/3) x ln(p) / (ln(p) + 1)

The dominant factor, p^(z/2 + 1/3), is two contributions multiplied together. The boundary rigidity kappa = z/2 = 3 determines how stiff the boundary is under deformation. Gravitational coupling requires deforming the boundary collectively, and each unit of rigidity contributes a factor of p to the suppression. The mode count p^(1/3) is the same factor that appears throughout BPR -- the number of active boundary modes between the GUT scale and the Planck scale. The correction ln(p)/(ln(p) + 1) accounts for the finite boundary: one degree of freedom (the ground state) does not participate in gravitational self-coupling, so the active fraction is ln(p) out of ln(p) + 1.

For p = 104,761 and z = 6:

BPR: 4.98 x 10^16. Observed: 4.96 x 10^16. Error: 0.4%.

This also derives Newton's gravitational constant G to 1.2%, and the Planck length to 0.6%.

### Gauge coupling unification

The Standard Model has three forces with three different strengths. A grand unified theory should explain why, by showing all three converge to one coupling at high energy. In the Standard Model alone, they do not converge. Supersymmetry fixes this, but nobody has found supersymmetric particles.

BPR provides a different unification mechanism using three boundary couplings, all determined by the coordination number z:

The first coupling is to U(1) hypercharge. Each boundary mode carries effective charge Y^2 = (3z + 1)/6. This is the boundary rigidity (z/2 independent winding directions from the z neighbors) plus a self-coupling correction (1/6, one direction per z-fold coordination from the central site). For z = 6, Y^2 = 19/6.

The second coupling is to SU(2). The boundary is a 2-sphere with three Killing vectors generating SO(3) rotations. A boundary mode at one of the z + 1 lattice sites (z neighbors plus the center) aligns with a Killing direction with probability 1/(z + 1). For z = 6, T_2^2 = 1/7.

The third coupling is to SU(3). Color is an internal symmetry, not a spacetime rotation. Its coupling to boundary modes is doubly suppressed relative to SU(2): T_3^2 = 1/(z + 1)^2. For z = 6, T_3^2 = 1/49.

The result: all three gauge couplings unify at the BPR GUT scale to within 0.5%. The max deviation from their average is 0.25 in units of 1/alpha, on couplings of order 49. No backward fitting. All three charges are functions of z = 6 alone.

### The sphaleron rate and baryon asymmetry

Why is there more matter than antimatter?

The answer depends on the sphaleron rate -- how efficiently the early universe converted energy into the matter-antimatter imbalance. Every previous version of BPR borrowed this rate from the Standard Model: kappa_sph = 10^-5. That borrowing is over.

The sphaleron involves five weak gauge boson exchanges, hence alpha_W^5. BPR amplifies this by the number of independent tunneling paths through the boundary: p^(1/3) modes times z neighbors per mode.

kappa_sph = p^(1/3) x z x alpha_W^5 = 1.25 x 10^-5

This matches the Standard Model value. No borrowing.

The boundary rigidity also controls the strength of the first-order electroweak phase transition. A stiffer boundary means larger latent heat, driving the system further from thermal equilibrium. The departure-from-equilibrium factor is sqrt(z/2).

The fully derived baryon asymmetry -- kappa_sph times the Jarlskog invariant from the CKM matrix times sqrt(z/2), all derived from (p, z) -- gives eta = 6.3 x 10^-10 versus observed 6.1 x 10^-10. Error: 2.6%.

---

## What still does not work

One experimental input remains: a single energy scale.

p = 104,761 is a pure number. z = 6 is a pure number. Lambda_QCD = 0.332 GeV has units of energy. You cannot derive a quantity with dimensions from quantities without dimensions. This is a theorem of dimensional analysis, not a BPR limitation.

Every physics framework needs at least one dimensionful anchor. String theory needs the string tension. Loop quantum gravity needs the Planck length. The Standard Model needs 25 or more experimental inputs. BPR needs one. Any single energy scale will do -- Lambda_QCD, or the Higgs VEV, or the Planck mass, or Newton's constant. Once one is given, all others follow from p and z.

The Weinberg angle prediction, sin^2 theta_W = 0.23122, depends on gauge unification being complete. The forward calculation achieves 0.5% unification quality. That is tight but not exact. The prediction should be read as nearly unconditional rather than strictly unconditional.

The tau lepton mass and bottom quark mass, which previously anchored the lepton and quark sectors, are now derived. The tau mass comes from a Yukawa formula: y_tau^2 = z^2 / (2 N_B l_tau^2), giving 1803 MeV versus 1777 MeV observed (1.5% off). The bottom quark mass comes from the winding-shifted spectrum, giving 4152 MeV versus 4180 MeV (0.7% off). All nine fermion masses now follow from (p, z) plus one energy scale.

A theory that only talks about its hits is not doing physics. It is doing marketing.

---

## Where it sits in known physics

A reasonable objection is: does this connect to quantum field theory at all, or is it a new language built in parallel?

It connects. The boundary action BPR writes down is formally identical to the c = 1 compact boson -- one of the most studied exactly solvable field theories in 2D. The winding modes that BPR identifies with particles are the winding sectors of that same compact boson. The coupling of the boundary to bulk physics follows the same structure as the AdS/CFT holographic dictionary, adapted for a flat universe instead of anti-de Sitter spacetime.

The fine-structure constant formula 1/alpha = [ln p]^2 + z/2 + gamma - 1/(2 pi) is not numerology. It is the one-loop renormalization of the boundary coupling constant in this field theory. The bare coupling is z/2 = 3, determined by geometry. The [ln p]^2 term is its renormalization by the discrete phase space. The gamma is the standard lattice-to-continuum scheme correction that appears in every lattice field theory. The -1/(2 pi) is the on-shell scheme matching.

This is standard field theory machinery, applied to a specific UV regularization -- Z_p rather than a conventional lattice -- that has better algebraic properties than the usual cutoff. The prime constraint on p follows from requiring the phase space to be a field (having division), not just a ring.

The gap, stated plainly: BPR has not been derived from a known UV-complete theory. The most natural candidate is a Chern-Simons theory on a three-sphere at a prime level k = p, where level quantization would explain the prime constraint on p and make the fine-structure constant formula a level-matching condition rather than an empirical fit. That derivation has not been done. It is the highest-priority open theoretical task.

## Why I think it is worth attention anyway

Because BPR is not vague.

It does not say "everything emerges from geometry" and then retreat into abstraction. It says things like:

- sin^2 theta_12 = 0.3083
- Hydrogen 1S-2S shift = +66.8 Hz beyond standard QED
- M_Pl / v_EW = p^(10/3) x ln(p) / (ln(p) + 1)
- kappa_sph = p^(1/3) x z x alpha_W^5
- Gauge couplings unify to 0.5% from three charges derived from z
- No axion
- No neutrinoless double beta decay
- Exactly 3 generations
- Normal neutrino mass ordering

These are not interpretive claims. They are killable claims.

That is rare.

Most "theories of everything" protect themselves by making their boldest predictions impossible to test. Their distinctive claims live safely at unreachable energy scales, so nothing can ever cleanly falsify them.

I wanted the opposite.

If this framework is wrong, it should be possible to prove it wrong in public.

---

## The single most decisive test

BPR predicts a +66.8 Hz anomalous shift in the hydrogen 1S-2S transition frequency beyond what standard QED predicts.

That is not a hand-wavy maybe-someday signal. It is a specific number. The instruments at MPQ Garching already have 10 Hz resolution.

If high-precision spectroscopy reaches the relevant resolution and finds no such shift, BPR is dead.

Not bruised. Not "needing refinement." Dead.

---

## Other ways to kill it

Here are additional clean falsification paths:

- JUNO finds inverted neutrino mass ordering
- Neutrinoless double beta decay is observed
- A QCD axion is discovered
- A nonzero neutron electric dipole moment is measured
- A fourth fermion generation is found
- sin^2 theta_12 lands outside 0.3083 +/- 0.002 at JUNO full precision
- Dark energy equation of state settles at w_0 > -0.8
- CTA pushes Lorentz violation bounds below 10^-21 with no signal
- Born rule holds to 10^-7 precision

That is what I want from a serious framework: a real falsification surface.

---

## The deeper claim

The bigger point is not just BPR. It is the possibility that the deepest mistakes in physics may come from treating lost information as ontological randomness.

Maybe entropy is often bookkeeping. Maybe chaos is often unresolved structure. Maybe quantum probability is often hidden phase information. Maybe singularities are where the continuum fails, not where reality does.

If that is true, then the next step in physics is not adding more epicycles to the existing formalism.

It is rebuilding the formalism around the information it threw away.

---

## What I am asking for

I am not asking anyone to believe BPR is correct.

I am asking for something much simpler:

Run the code. Find a circular derivation. Point to a published result that already kills it.

That is the standard it should be held to.

The codebase is MIT licensed on GitHub. 1,223 test functions. Every prediction in this article can be reproduced in under 60 seconds on a laptop. The derivation engine, constant calculator, and experimental roadmap are live at bpr.thestardrive.com.

Physics does not need more frameworks that can never be wrong. It needs frameworks willing to be wrong in ways experiments can actually test.

BPR may fail. It may fail quickly. But at least it is exposed to reality.

That is the only kind of theory worth arguing about.

Let's see what breaks.
