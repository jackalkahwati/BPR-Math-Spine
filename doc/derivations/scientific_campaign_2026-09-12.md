# Six-module scientific campaign

2026-09-12. The user requests six sequential scientific calculations, each verified and published before proceeding. Completion of these modules does not mean resolution of the six corresponding foundational physics problems. Prior work is merged in main at 8b7dfd6 (PR23); the historical failed particle benchmark and withdrawn TOE claim remain unchanged.

## Status

| Module | Calculation | State | Publication |
|---|---|---|---|
| 1 | Consistently dressed neutral Hamiltonian and density response | Complete bounded calculation; math/code review passed after two scale fixes; 53 focused and 1,079 combined tests passed; ten demos/eight algebra groups passed | Committed/pushed 45f52c5; PR24 open; ordinary merge denied by environment's Merge Without Review rule, not merged |
| 2 | Full-model joint strong-coupling/large-volume bound and spatial scaling | Complete conditional calculation; math/code review passed after oracle resolution fix; 78 post-fix focused tests and 1,152 pre-fix combined tests passed (one intentional deselection, existing marker warning); eleven demos/eight algebra groups passed | Committed/pushed 41f0196; PR25 open, stacked on PR24; not merged; main integration blocked behind PR24 |
| 3 | Charge-resolved branch chirality and spectral flow | Complete bounded calculation; math/code review passed; 154 focused and 944 inherited substrate tests passed, eight algebra groups passed | Committed/pushed d86723e; PR26 open, stacked on PR25/PR24; not merged |
| 4 | Parity-aware quantum projection and symmetry matching | Complete bounded calculation; math/code review passed after four fixes; 119 focused and 1,085 post-fix substrate tests passed; twelve selected demos and eight algebra groups passed | Committed/pushed f53503f; PR27 open, stacked on PR26/25/24; not merged |
| 5 | Action-consistent gravity normalization and identifiability | Complete bounded calculation and narrow legacy repair; math/integration/code review passed after numerical guard fixes; 177 post-fix focused and 1,424 selected post-fix regression tests passed; isolated demo and eight algebra groups passed | Committed/pushed8705eef; PR28 open, stacked on PR27/26/25/24; not merged; main integration remains blocked |
| 6 | Causal phason response, resonance bounds and testable joint predictions | Complete bounded calculation; math/code review passed after five numerical fixes; 97 final focused and1,521 selected final regression tests passed;13 selected demos, grammar and eight algebra groups passed; one inherited last-bit equality failure occurred on the prior attempt and is documented | Committed/pushed5766a4a; PR29 open, stacked on PR28/27/26/25/24; not merged |

Each module requires derivation, bounded executable calculations, independent tests, stdout-only text/strict-JSON demo and independent mathematics/code review. Publish via normal branch/PR/merge; no instruction/config edits, permission bypass, fake approval or unrelated branch merge. Tests are conditional-model checks, not empirical validation. Preserve the inherited dense dimension-512 cap and frozen weak controls.

## Frozen module 1 controls

See [dressed neutral response](substrate_neutral_effective_2026-09-12.md). Use L=5,C=1,g=40 and .7,m=1/0. Order checks L=3,4 and lambda=1/40,1/80,1/160. No retrospective physical parameter tuning. Full H unchanged. Every virtual occupation is retained.

## Independent planning refinements to carry forward

- Module 1: all-D SW conjugation Htilde=UHU† and density transformed by the same U. Signed excited denominators. Hamiltonian residual is not an individual-residue certificate; diagonalized truncated results are partially resummed.
- Module 2: full normalized density measure coordinate s=(En-E0(L)-g_L)/C_L, where g_L and C_L are indexed sequences. Integer-D averaging proves ||S1||<=pi L and a first-order unitary generator remainder <=8pi*lambda*L² in C units, as independently reviewed in the module-2 derivation. With source norm a=4lambda|sin(k/2)| and inherited error delta=sqrt(L)*eta, the verified characteristic-function bound is 4delta/a+2pi*lambda*L+|t|[8pi*lambda*L²+4lambda*L/(1-4lambda*L)]. The derivation combines this with the existing compression theorem and Levy continuity theorem. Unbounded-moment and conditioned near-edge convergence are not inferred.
- Module 3: half-open boundary-flux interval, explicit endpoint crossing ties, interior-band reference, zero signed band-edge tangencies. Both parity sectors and charge1 creation operators retained. Partial filling is not selected neutral unit filling.
- Module 4: wedge dimensions binomial(3,N) vanish for neutral N=L>3; at L=N=3 the wedge is one-dimensional. Actual neutral D=1 excitations are outside hard core. Literal reflection lifts can mismatch but the allowed parity phase changes the lift; bilinear representations cancel the sign. Neither a chosen vector lift nor generated M3 algebra is a universal gauge obstruction or gauge derivation.
- Module 5: action M²R/2+alpha R²/2 implies exact mass²=M²/(6alpha), plateau=M⁴/(8alpha), leading large-Ne As=Ne²/(144pi²alpha). Legacy reduced-G conversion needs 1/(8pi), while numeric entropy A/(4lP²) stays unchanged. Inverse observational matching remains calibration.
- Module 6: exact constitutive circle differs from leading phonon-resonance circle. Resonance is a pole of (Cq²-rho Omega²)(Kq²-iGamma Omega)-D²q⁴=0, not a driven peak. The originally proposed bounds are now proved by Rouche isolation and an exact remainder identity for beta=D²/(CK)<=1/4: |Omega/Omega0-1+beta/[2(1-iu0)]|<=13beta²/8, and Qinv leading error<=5beta². The proof passed independent review; these are not numerical solver certificates. q=0, D=0, pole collisions and linked Omega/q resonance data require explicit treatment. Do not inherit unsupported D~1/p/core scaling.

## Scientific outcomes and unresolved foundational questions

1. **Neutral interactions:** the dressed Hamiltonian, density and vacuum reference are now consistent to the declared order, with signed virtual channels and explicit bounds. Neither a finite-ring level nor an unresolved separation establishes a physical bound particle.
2. **Limits:** a full normalized neutral spectral weak limit is proved on sufficient simultaneous strong-coupling/large-volume sequences. Fixed coupling and a conditioned spatial edge limit remain outside that theorem; absolute source weight vanishes. The quadratic support edge is not Lorentz invariance.
3. **Fermions/chirality:** the hard-core ring has charged Jordan–Wigner variables and paired one-dimensional branches with zero net spectral flow. Finite-repulsion virtual terms are nonquadratic. Physical 3+1D Weyl matter and mirror removal remain unconstructed.
4. **Substrate/gauge/flavor connection:** actual exterior embeddings expose the neutral candidate's one-dimensional/empty nature, even-sector leakage and the distinction between source span, generated algebra and conserved symmetry. No link/Gauss dictionary or physical flavor-sector identification is supplied.
5. **Vacuum/gravity:** the stipulated action and reduced Planck conversions are repaired; bare/counterterm/cutoff and normalization freedoms are explicit. Flat identity shifts cannot determine a covariant vacuum density. Geometry, absolute scales and vacuum-energy prediction remain open.
6. **Response tests:** the generic supplied elastic/diffusive model has an exact constitutive circle and proved weak-coupling pole-error bounds. Independent microscopic coefficients and experimental mode/background calibration are required before calling this a BPR-specific prediction. Implementation and held-out verification are recorded in the status table rather than inferred from the proof alone.

These are scoped calculations, not closure of six fundamental physics problems. The next scientific bottleneck is a specified microscopic-to-effective dictionary with independently determined parameters and an observable test that distinguishes BPR from generic models. More consistency checks or numerical near-matches cannot substitute for that derivation and empirical evidence. No historical failed benchmark is rerun or reinterpreted as a success.

## Verification baseline

Before this campaign: 1,026 bounded tests passed, one intentional phase-anchor deselection, one existing unregistered slow-marker warning. Nine demos and eight algebra-check groups passed. New module counts and failures will be recorded after execution; no full-repository or empirical-validation claim.
