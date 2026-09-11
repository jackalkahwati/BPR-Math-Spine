# Color-bundle index on S²: corrected derivation

> **Status: 2026-09-10 — the pure-SU(3) index justification for l_t=283 is
> withdrawn.** The retained numerical formula is a conjectural mode assignment.
> The index and zero-mode calculations below are conditional mathematical
> results. No additional U(1) flux is inserted into the frozen BPR model.

## 1. The index for an ordinary SU(r) color bundle

Let E be the rank-r complex vector bundle associated to the fundamental
representation of SU(r), r>=2, over a smooth closed spin S². Use an
anti-Hermitian connection with curvature F. The twisted spin Dirac index is

    ind D_E = integral_S² [A-hat(TS²) ch(E)]_2
            = integral_S² c₁(E)
            = (i/2pi) integral_S² tr F.

The A-hat class has no degree-two term. Since the connection is su(r)-valued,
tr F=0. Equivalently det E is trivial, so c₁(E)=c₁(det E)=0. Therefore

    ind D_E = 0.

This holds independently of coordination number and any flavor ansatz.
A second Chern number cannot replace it on a two-dimensional base.
Principal SU(r) bundles on S² are classified by pi₁(SU(r))=0. This does not
make every connection flat, and index zero does not forbid paired zero modes:
it states n_plus=n_minus, not that both vanish for every gauge background.
For the untwisted Dirac operator with flat color connection on a round S²,
positive scalar curvature does imply no zero modes via D²=nabla* nabla+R/4.

Thus the former equation `ind D_color = n_gen = 3` is false for the stated
bundle. Neither a Cartan-rank count nor a holonomy label changes this index.

## 2. What an additional line-bundle twist actually gives

Specify L=O(q) on CP¹, q an integer degree. Then

    c₁(E tensor L) = c₁(E) + rank(E)c₁(L) = r q,
    ind D_(E tensor L) = r q.

This introduces U(1) twisting data. It is not a nonzero Chern class of the
original SU(r) color bundle. A Hopf line bundle with degree one is not
itself a justification for coupling physical fermions to it, or for their
charge under that bundle. Those are additional model choices.

For one line bundle, K=O(-2) and K^(1/2)=O(-1). On CP¹:

    ker D_plus  corresponds to H⁰(O(q-1)),
    ker D_minus corresponds to H¹(O(q-1)).

For k>=0, the homogeneous monomials u^j v^(k-j), 0<=j<=k, give k+1
independent sections of O(k); for k<0 there are none. Serre duality gives
h¹(O(k))=h⁰(O(-k-2)). Hence

    n_plus = max(q,0),    n_minus = max(-q,0),    index = q.

For q>0 these modes carry the rotational multiplet j=(q-1)/2, dimension q,
under the rotation action lifted to the monopole bundle. Its use as an
internal family space still requires a spacetime interpretation.
For a general nontrivial color connection only the total index r q is
fixed; extra paired zero modes and mixing require spectral analysis.

## 3. Color multiplicity is not family multiplicity

Take a trivial color connection so that the zero modes factor as
C^r tensor ker D_L. Then:

| Color rank r | Line degree q | Net zero-mode components | Copies of color representation |
|--------------|---------------|--------------------------|--------------------------------|
| 3 | 0 | 0 | 0 in this flat-color setup |
| 3 | 1 | 3 | 1 color triplet |
| 3 | 3 | 9 | 3 color triplets |
| 1 | 3 | 3 | 3 color singlets |

The tempting `3 colors x unit Hopf flux = 3 generations` counts color
components as families. In this factorized construction a degree-three twist
could yield three copies, but the flux and fermionic model are inputs.
It would need to work for the entire chiral matter content, not just color.
An internal Dirac index alone neither constructs four-dimensional particles
nor proves anomaly cancellation.

## 4. Why a nonzero index still would not derive l_t

An index counts a signed excess of zero modes. It does not add an integer
to a nonzero angular-momentum label or mass eigenvalue. Even a valid index
of three would not establish

    l_t = (z²-1)(z+n_gen+2-N_c) + n_gen = 283.

The multiplication by dim(su(z)), subtraction of Cartan rank from a mode
space, and conversion of an index into this additive shift each require an
explicit operator and spectrum. The existing formula remains unchanged as
phenomenology, labeled CONJECTURAL. Replacing +3 by +0 would not produce a
derived prediction of 280; the rest of the spectral identification is also
unproved. `n_gen=3` is an empirical input, not inferred from this index.

## 5. Reproducible checks and remaining task

`bpr/flavor_foundations.py` computes compact-boson weights, line-bundle
zero modes and the SU(r)-with-line-twist index. Its tests independently
check the determinant-one obstruction, the Chern-character tensor product,
and the monopole spectrum's zero-mode multiplicity. They also preserve the
legacy mass/mixing outputs while checking that their status is conjectural.

A replacement physical derivation needs a specified fermion action and
bundle, a dynamical selection of flux, a map from internal zero modes to
families, and a spectral calculation yielding the mass labels. These remain
open rather than being replaced by another counting analogy.

## References

- J. P. May, [Characteristic classes, chapter 3](https://www.math.uchicago.edu/~may/CHAR/charclasses.pdf), for the determinant/first-Chern-class obstruction to an SU(r) structure.
- Deguchi and Kitsukawa, [Charge quantization conditions based on the Atiyah–Singer index theorem](https://arxiv.org/abs/hep-th/0512063), for the monopole Dirac equation and index.
