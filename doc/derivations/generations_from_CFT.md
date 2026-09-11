# Fermion generations: corrected CFT argument and open identification

> **Status: 2026-09-10 — previous closure withdrawn.** The compact-boson
> argument does not derive three fermion generations or exclude a fourth.
> The retained flavor model uses `n_gen = 3` as an empirical input.
> The companion [index calculation](color_bundle_index.md) gives a valid,
> conditional route through twisted Dirac zero modes, not a completed BPR model.

## 1. What the spherical spectrum establishes

For the **round metric** on S² of radius a, the scalar Laplacian has

    eigenvalue = l(l+1)/a²,     multiplicity = 2l+1,     l = 0,1,2,...

Thus l=1 has three states. This is a spatial spin-1 multiplet. It is not a
spin-1/2 multiplet and does not establish three copies of a fermion species.
Topology alone also does not select the round metric or its degeneracies.
A family interpretation needs an internal mode space and a map to spacetime
fields, with identical gauge representations and independently specified
spacetime spin. No such map was supplied by the previous argument.

S² has a unique spin structure: spin structures form a torsor for
H¹(S²; Z₂)=0. There is no additional Möbius choice. Spin structure alone does
not impose odd winding on a compact boson, and a Hopf fiber is not a
noncontractible loop in the base S².

## 2. Compact-boson weights: explicit convention

Use the nonchiral compact boson with alpha-prime = 1 and dimensionless R:

    p_L = m/R + n R,       p_R = m/R - n R,      m,n in Z
    h = p_L²/4,           hbar = p_R²/4
    x = h + hbar = (m²/R² + n² R²)/2
    s = h - hbar = mn.

These formulas are implemented with exact rational arithmetic in
`bpr/flavor_foundations.py`. Changing R does not change the integrality of s.
Descendants shift h and hbar by integers, so they do not supply a local
half-integer-spin operator in this untwisted bosonic operator lattice.
This statement concerns this lattice, not every possible fermionization or
spin-CFT extension of a bosonic theory.

The old note used h=p_L²/2 and hbar=p_R²/2 with the same momentum lattice.
Taken literally, those equations give s=2mn, still an integer; the later
table used mn instead. Neither convention justifies the claimed fermions.
The factor-of-two correction is local to this note and its checking module;
it does not rescale existing phenomenological mass formulas elsewhere.

For R²=3, the conventional weights are:

| (m,n) | h | hbar | x | s |
|-------|---|------|---|---|
| (1,0) | 1/12 | 1/12 | 1/6 | 0 |
| (0,1) | 3/4 | 3/4 | 3/2 | 0 |
| (1,1) | 4/3 | 1/3 | 5/3 | 1 |
| (1,-1) | 1/3 | 4/3 | 5/3 | -1 |
| (2,2) | 16/3 | 4/3 | 20/3 | 4 |

The use of R²=3 is conditional. Matching this nonchiral theory to a chosen
chiral U(1) Chern–Simons edge requires a separate normalization and operator
identification. Moreover S² as a Euclidean two-dimensional CFT surface is
not automatically a spatial S² supporting a 2+1-dimensional theory. The Hopf
base of a closed S³ is not its boundary; the CS edge correspondence alone
does not provide the missing identification.

## 3. Why the old selection rules fail

- Nonnegative h and hbar impose no upper dimension cutoff; they hold for
  every momentum/winding pair here.
- Adding an unspecified field of weight 1/16 is not a construction of a
  fermion. An orbifold/twist or spin-CFT extension needs its action, operator
  spectrum, locality, spin structure and projection. The old arithmetic
  also erred: 10/3 + 1/16 = 163/48, not 167/48.
- A vertex operator can be a Virasoro primary **and** appear in an operator
  product. The leading operator in V_(1,1) V_(1,1) is V_(2,2), which remains
  primary. Fusion is not a criterion for discarding it as a descendant or
  determining a spacetime single-particle spectrum.
- Four labels do not determine a representation. A 3+1 decomposition needs
  explicit generators acting on the states. A spin-1 triplet plus scalar
  would still not by itself supply three spin-1/2 families.
- Orbifolding can introduce twisted sectors. The previous blanket exclusion
  of additional generations from unspecified orbifolds is withdrawn.

## 4. A constructive replacement: Dirac zero modes

On S² = CP¹, the spin bundle is K^(1/2)=O(-1). Couple a Dirac field to
L=O(q), with integral degree q. Positive-chirality zero modes are holomorphic
sections of O(q-1); negative-chirality modes are counted by its H¹:

    n_plus  = max(q,0)
    n_minus = max(-q,0)
    index D_L = n_plus - n_minus = q.

This follows from the polynomial sections of O(k) and Serre duality; see
[color_bundle_index.md](color_bundle_index.md) for the full argument.
For q=3 it supplies three chiral internal zero modes. The same calculation
supplies four for q=4 on the same sphere: topology does not exclude four.
Here q is an additional specified flux/charge, **not derived from p or z**.
The zero-mode count is not yet a four-dimensional generation count.

To turn this into a BPR derivation one must specify the fermionic kinetic
operator and internal-to-spacetime map, derive the twisting bundle and q,
recover the gauge representations and chirality for every species, exclude
unwanted mirror modes, and check anomalies. The dynamics must select the
flux without using the observed family count as input. None is supplied by
merely choosing q=3. This extension is a candidate, not a new frozen postulate.

## 5. Consequences for code and claims

| Quantity | Corrected status |
|----------|------------------|
| Round S² scalar l=1 multiplicity = 3 | Mathematical result |
| Fermionic operators from the old CFT selection rules | Not established |
| Exactly three families / no fourth | OPEN |
| n_gen=3 in numerical flavor calculations | Empirical input |
| Chiral zero modes of O(q) on S² | Derived conditional on q and Dirac setup |
| l_t=283 and other physical flavor-mode assignments | CONJECTURAL |

`generation_count_from_topology` returns None unless the caller explicitly
requests the l=1-to-families assumption. `number_of_generations` retains the
legacy numerical ansatz for compatibility; its docstring and prediction
metadata identify it as an assumption. Passing a numerical regression test
does not restore the withdrawn proof.

## References

- David Tong, [String Theory, compactification and CFT](https://www.damtp.cam.ac.uk/user/tong/string/string.pdf), especially sections 4 and 8.1, for compact-boson weights and momentum/winding conventions.
- Deguchi and Kitsukawa, [Charge quantization conditions based on the Atiyah–Singer index theorem](https://arxiv.org/abs/hep-th/0512063), for Dirac zero modes on a monopole sphere.
