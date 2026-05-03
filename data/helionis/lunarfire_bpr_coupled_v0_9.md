# LunarFire v0.9 BPR-Coupled Output

Best plant-net power: `-3.2 MW`.
BPR sources: `bpr.impedance.TopologicalImpedance; bpr.resonance.load_riemann_zeros`.

| Plant MW | Closes | R | Aspect | T keV | n m^-3 | tau s | BPR align | BPR Z match | BPR transport x | BPR direct x | Gross MW | Radiator m2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -3.2 | False | 4.0 | 8 | 200 | 3.5e+20 | 20 | 0.993 | 0.862 | 0.785 | 1.047 | 697 | 35284 |
| -3.2 | False | 4.0 | 10 | 200 | 3.5e+20 | 20 | 0.940 | 0.862 | 0.795 | 1.045 | 697 | 35313 |
| -3.4 | False | 4.0 | 8 | 200 | 3.5e+20 | 16 | 0.993 | 0.862 | 0.785 | 1.047 | 701 | 35489 |
| -3.5 | False | 4.0 | 10 | 200 | 3.5e+20 | 16 | 0.940 | 0.862 | 0.795 | 1.045 | 701 | 35521 |
| -4.2 | False | 5.0 | 8 | 200 | 3.5e+20 | 20 | 0.993 | 0.800 | 0.777 | 1.046 | 691 | 35047 |
| -4.2 | False | 5.0 | 10 | 200 | 3.5e+20 | 20 | 0.940 | 0.800 | 0.788 | 1.044 | 692 | 35072 |
| -4.3 | False | 5.0 | 12 | 200 | 3.5e+20 | 20 | 0.735 | 0.800 | 0.828 | 1.038 | 693 | 35167 |
| -4.3 | False | 4.0 | 12 | 200 | 3.5e+20 | 20 | 0.735 | 0.862 | 0.835 | 1.039 | 714 | 36225 |
| -4.4 | False | 4.0 | 8 | 200 | 2.5e+20 | 20 | 0.993 | 0.862 | 0.785 | 1.047 | 713 | 36180 |
| -4.4 | False | 4.0 | 10 | 200 | 2.5e+20 | 20 | 0.940 | 0.862 | 0.795 | 1.045 | 714 | 36223 |
| -4.4 | False | 5.0 | 8 | 200 | 3.5e+20 | 16 | 0.993 | 0.800 | 0.777 | 1.046 | 695 | 35247 |
| -4.4 | False | 5.0 | 10 | 200 | 3.5e+20 | 16 | 0.940 | 0.800 | 0.788 | 1.044 | 695 | 35275 |

Interpretation notes:

- v0.9 uses existing BPR Math Spine primitives as bounded correction factors.
- BPR factors are not treated as energy sources.
- Current BPR source modules: `bpr.impedance` and `bpr.resonance`.
