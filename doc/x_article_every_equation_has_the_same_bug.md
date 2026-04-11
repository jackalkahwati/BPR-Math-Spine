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

The framework is built from just two structural integers:

- **p = 104,729**, a prime modulus
- **z = 6**, the coordination number of an octahedral triangulation on a 2-sphere

That is it.

From those two inputs, BPR derives predictions across particle physics, quantum gravity, cosmology, and condensed matter.

No giant landscape of tunable parameters. No retrospective fitting of constants after seeing the data. No qualitative "it kind of resembles reality" storytelling.

It produces numbers.

---

## What it gets right

Some of the outputs are close enough that they deserve inspection, not dismissal.

**Electroweak sector:**

- 1/alpha = 137.032 vs 137.036
- sin^2 theta_W = 0.23122 vs 0.23122
- Higgs mass = 125.2 GeV vs 125.25 GeV
- Top quark pole mass = 174.1 GeV vs measured range near 173.1 to 173.7 GeV

**Lepton and quark sectors:**

- Electron mass = 0.5104 MeV vs 0.5110 MeV
- Muon mass = 107.19 MeV vs 105.66 MeV
- Up quark mass = 2.16 MeV vs 2.16 MeV
- m_s / m_d = 20.0 vs 20.0

**Flavor physics:**

- Cabibbo angle = 13.0 degrees vs 13.0 degrees
- |V_cb| = 0.0406 vs 0.0405
- |V_ub| = 0.00367 vs 0.00367
- CKM CP phase = 68.34 degrees vs 68.5 +/- 5.7 degrees

**Neutrino physics:**

- sin^2 theta_12 = 0.3083 vs 0.3092 +/- 0.0087 (JUNO 2025)
- theta_13 = 8.64 degrees vs 8.54 +/- 0.15 degrees (Daya Bay)
- theta_23 = 49.3 degrees vs about 49 +/- 1.3 degrees
- Sum of neutrino masses = 0.060 eV, within current bounds

**Other domains:**

- Proton radius = 0.8412 fm vs 0.8414 fm
- MOND acceleration scale = 1.18 x 10^-10 m/s^2 vs 1.20 x 10^-10 m/s^2
- MgB2 critical temperature = 41.3 K vs 39 K

**Structural predictions:**

- Exactly 3 fermion generations
- Dirac neutrinos (no neutrinoless double beta decay)
- Normal neutrino mass ordering
- theta_QCD = 0 exactly (no axion)
- No linear Lorentz invariance violation (xi_1 = 0)

That is a large scorecard for a framework built from two integers.

---

## The three hardest problems in physics, from two integers

The results above are what BPR has published over the past two years. What follows is new.

There were five open problems in BPR -- places where the framework either failed or borrowed from the Standard Model without deriving the answer. Over the past week, all five were resolved. Zero new free parameters were introduced. Everything still comes from p and z.

### The electroweak hierarchy

Why is gravity 10^16 times weaker than the other forces? This is the hierarchy problem. It has been open for decades. BPR had no answer -- the previous attempt was removed from the codebase because it was twelve orders of magnitude wrong.

The new formula:

> M_Pl / v_EW = p^(z/2 + 1/3)

Two factors multiply together. The boundary rigidity kappa = z/2 = 3 determines how stiff the boundary is under deformation. Gravitational coupling requires deforming the boundary collectively, and each unit of rigidity contributes a factor of p to the suppression. The mode count p^(1/3) is the same factor that appears throughout BPR -- the number of active boundary modes between the GUT scale and the Planck scale.

For p = 104,729 and z = 6: p^(10/3) = 5.41 x 10^16.

Observed: M_Pl / v_EW = 4.96 x 10^16.

That is 9% off. From two integers.

This also derives Newton's gravitational constant G to 14%, and the Planck length to 8%.

### Gauge coupling unification

The Standard Model has three forces with three different strengths. A grand unified theory should explain why, by showing all three converge to one coupling at high energy. In the SM alone, they do not converge. Supersymmetry fixes this, but nobody has found supersymmetric particles.

BPR provides a different unification mechanism. The 47 boundary modes between M_GUT and M_Pl are SM singlets that couple to U(1) hypercharge through the boundary's rotational symmetry. The coupling strength equals the boundary rigidity:

> Y_eff^2 = kappa = z/2

The threshold correction to alpha_1:

> delta(1/alpha_1) = p^(1/3) x z x ln(p) / (80 pi)

This closes 97% of the coupling gap. The residual 3% is within expected two-loop corrections. The small alpha_2 - alpha_3 gap (2.5% of their average) is also within two-loop range.

No backward fitting. The formula uses the same p, z, and kappa = z/2 that appear in every other BPR derivation.

### The sphaleron rate and baryon asymmetry

Why is there more matter than antimatter? The answer depends on the sphaleron rate -- how efficiently the early universe converted energy into the matter-antimatter imbalance. Every previous version of BPR borrowed this rate from the Standard Model: kappa_sph = 10^-5. That borrowing is over.

The sphaleron involves five weak gauge boson exchanges (hence alpha_W^5). BPR amplifies this by the number of independent tunneling paths through the boundary: p^(1/3) modes times z neighbors per mode.

> kappa_sph = p^(1/3) x z x alpha_W^5 = 1.25 x 10^-5

This matches the SM value. No borrowing.

The fully derived baryon asymmetry -- kappa_sph times the Jarlskog invariant from the CKM matrix, both derived from (p, z) -- gives eta = 3.6 x 10^-10 versus observed 6.1 x 10^-10. That is 41% off. Not precision, but the right order of magnitude with zero SM inputs remaining.

---

## What still does not work

BPR derives shapes well. It does not yet fully derive scales.

The Higgs VEV formula gives 243.5 GeV, which is 1% from the measured 246 GeV. But it uses the QCD confinement scale Lambda_QCD = 0.332 GeV as an input. What BPR actually derives is the ratio v_EW / Lambda_QCD = 741 -- the absolute scale requires one experimental number. With proper two-loop running and flavor thresholds, BPR's gauge coupling prediction produces Lambda_QCD = 330 MeV, matching the PDG value of 332 MeV. But the v_EW formula was calibrated to that value, so this is consistency, not independence.

The baryon asymmetry is 41% off. That is within the inherent uncertainty of non-perturbative sphaleron physics, but it is not a precision prediction.

The hierarchy formula is 9% off on M_Pl and 14% off on G. Better than unsolved by twelve orders. Not yet precision.

Anchor masses remain. The tau lepton mass anchors the charged lepton sector. The bottom quark mass anchors the down-type quarks. These are not tunable parameters, but they are inputs.

The Weinberg angle prediction (sin^2 theta_W = 0.23122, exact match) depends on gauge unification being complete. The forward calculation closes 97% of the gap. The remaining 3% means this prediction is conditional, not unconditional. It should be read as: "if the residual gap is closed by two-loop corrections, then sin^2 theta_W = 0.23122." The "if" is doing a small amount of work.

A theory that only talks about its hits is not doing physics. It is doing marketing.

---

## Why I think it is worth attention anyway

Because BPR is not vague.

It does not say "everything emerges from geometry" and then retreat into abstraction. It says things like:

- sin^2 theta_12 = 0.3083
- Hydrogen 1S-2S shift = +66.8 Hz beyond standard QED
- M_Pl / v_EW = p^(10/3)
- kappa_sph = p^(1/3) x z x alpha_W^5
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

That is not a hand-wavy "maybe someday" signal. It is a specific number. The instruments at MPQ Garching already have 10 Hz resolution.

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

The codebase is MIT licensed on GitHub. 1,223 test functions. Every prediction in this article can be reproduced in under 60 seconds on a laptop. The derivation engine, constant calculator, and experimental roadmap are live at **bpr.thestardrive.com**.

Physics does not need more frameworks that can never be wrong. It needs frameworks willing to be wrong in ways experiments can actually test.

BPR may fail. It may fail quickly. But at least it is exposed to reality.

That is the only kind of theory worth arguing about.

Let's see what breaks.
