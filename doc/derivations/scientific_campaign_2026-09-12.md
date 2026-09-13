# Six-module scientific campaign

2026-09-12. The user requests six sequential scientific calculations, each verified and published before proceeding. Completion of these modules does not mean resolution of the six corresponding foundational physics problems. Prior work is merged in main at 8b7dfd6 (PR23); the historical failed particle benchmark and withdrawn TOE claim remain unchanged.

## Status

| Module | Calculation | State | Publication |
|---|---|---|---|
| 1 | Consistently dressed neutral Hamiltonian and density response | Complete bounded calculation; math/code review passed after two scale fixes; 53 focused and 1,079 combined tests passed; ten demos/eight algebra groups passed | Committed/pushed 45f52c5; PR24 open; ordinary merge denied by environment's Merge Without Review rule, not merged |
| 2 | Full-model joint strong-coupling/large-volume bound and spatial scaling | Complete conditional calculation; math/code review passed after oracle resolution fix; 78 post-fix focused tests and 1,152 pre-fix combined tests passed (one intentional deselection, existing marker warning); eleven demos/eight algebra groups passed | Committed/pushed 41f0196; PR25 open, stacked on PR24; not merged; main integration blocked behind PR24 |
| 3 | Charge-resolved branch chirality and spectral flow | Complete bounded calculation; math/code review passed; 154 focused and 944 inherited substrate tests passed, eight algebra groups passed | Committed/pushed d86723e; PR26 open, stacked on PR25/PR24; not merged |
| 4 | Parity-aware quantum projection and symmetry matching | Complete bounded calculation; math/code review passed after four fixes; 119 focused and 1,085 post-fix substrate tests passed; twelve selected demos and eight algebra groups passed | Publication prepared on dependent branch science/substrate-quantum-matching based on d86723e; main integration remains blocked behind PR24 |
| 5 | Action-consistent gravity normalization and identifiability | Planned; not implemented | None |
| 6 | Causal phason response, resonance bounds and testable joint predictions | Planned; not implemented | None |

Each module requires derivation, bounded executable calculations, independent tests, stdout-only text/strict-JSON demo and independent mathematics/code review. Publish via normal branch/PR/merge; no instruction/config edits, permission bypass, fake approval or unrelated branch merge. Tests are conditional-model checks, not empirical validation. Preserve the inherited dense dimension-512 cap and frozen weak controls.

## Frozen module 1 controls

See [dressed neutral response](substrate_neutral_effective_2026-09-12.md). Use L=5,C=1,g=40 and .7,m=1/0. Order checks L=3,4 and lambda=1/40,1/80,1/160. No retrospective physical parameter tuning. Full H unchanged. Every virtual occupation is retained.

## Independent planning refinements to carry forward

- Module 1: all-D SW conjugation Htilde=UHU† and density transformed by the same U. Signed excited denominators. Hamiltonian residual is not an individual-residue certificate; diagonalized truncated results are partially resummed.
- Module 2: full normalized density measure coordinate s=(En-E0(L)-g_L)/C_L, where g_L and C_L are indexed sequences. Integer-D averaging proves ||S1||<=pi L and a first-order unitary generator remainder <=8pi*lambda*L² in C units, as independently reviewed in the module-2 derivation. With source norm a=4lambda|sin(k/2)| and inherited error delta=sqrt(L)*eta, the verified characteristic-function bound is 4delta/a+2pi*lambda*L+|t|[8pi*lambda*L²+4lambda*L/(1-4lambda*L)]. The derivation combines this with the existing compression theorem and Levy continuity theorem. Unbounded-moment and conditioned near-edge convergence are not inferred.
- Module 3: half-open boundary-flux interval, explicit endpoint crossing ties, interior-band reference, zero signed band-edge tangencies. Both parity sectors and charge1 creation operators retained. Partial filling is not selected neutral unit filling.
- Module 4: wedge dimensions binomial(3,N) vanish for neutral N=L>3; at L=N=3 the wedge is one-dimensional. Actual neutral D=1 excitations are outside hard core. Literal reflection lifts can mismatch but the allowed parity phase changes the lift; bilinear representations cancel the sign. Neither a chosen vector lift nor generated M3 algebra is a universal gauge obstruction or gauge derivation.
- Module 5: action M²R/2+alpha R²/2 implies exact mass²=M²/(6alpha), plateau=M⁴/(8alpha), leading large-Ne As=Ne²/(144pi²alpha). Legacy reduced-G conversion needs 1/(8pi), while numeric entropy A/(4lP²) stays unchanged. Inverse observational matching remains calibration.
- Module 6: exact constitutive circle differs from leading phonon-resonance circle. Resonance is a pole of (Cq²-rho Omega²)(Kq²-iGamma Omega)-D²q⁴=0, not a driven peak. Candidate contraction bound for beta=D²/(CK)<=1/4: |Omega/Omega0-1+beta/[2(1-iu0)]|<=13beta²/8, and Qinv leading error<=5beta². Independently prove and test. q=0, D=0, pole collisions and linked Omega/q resonance data require explicit treatment. Do not inherit unsupported D~1/p/core scaling.

## Verification baseline

Before this campaign: 1,026 bounded tests passed, one intentional phase-anchor deselection, one existing unregistered slow-marker warning. Nine demos and eight algebra-check groups passed. New module counts and failures will be recorded after execution; no full-repository or empirical-validation claim.
