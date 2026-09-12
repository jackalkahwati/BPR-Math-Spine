# Quantum source operators: a conditional diagnostic, not a microscopic derivation

**2026-09-12. Status: separately stipulated instantaneous internal-fermion model.**

The [classical source-selection model](flavor_source_selection_2026-09-12.md)
selects coherent occupations with aligned axes at positive inter-sector coupling.
Its occupation constraint, coupling signs and scalar action were added assumptions.
The next question is whether actual microscopic dynamics supplies those inputs.
A focused repository inspection found no substrate-to-monopole operator map or
scalar time kernel establishing that connection. This note records the missing
matching data and tests a conditional operator counterpart of the known geometry.

**It does not derive fermions, a physical occupation sector, a scalar mediator,
or realistic mixing from BPR.** The completed classical theorem remains valid
for its own functional. Replacing squared mean densities by operator squares is
a different model, not a correction to that theorem.

## 1. Microscopic matching inventory

| Required input | Existing evidence | Missing link / diagnostic assumption |
|---|---|---|
| Substrate-to-triplet projection and validity scale | `bpr/condensate_regime.py:95–110` implements fixed-norm ring dynamics; the monopole prototype implements three sections | No isometry or controlled low-energy projection connecting these Hilbert spaces. q=3 geometry is inherited, not derived. |
| Statistics and source quantum numbers | `bpr/zp_selection_principle.py:106–122` builds three truncated **bosonic sites** | Those sites are not the monopole triplet. Here six canonical fermion modes are stipulated. Their number density is not automatically a Lorentz-scalar chiral SM source. |
| Population / preparation | Ring norm and the small model's number are conserved | Conservation does not select the value. The assumed Z_p neutrality rule constrains charge modulo substrate p, not one particle in each monopole sector. No number-changing bath or chemical potential is supplied. |
| Scalar time kernel | `bpr/chiral_flavor_prototype.py:228–255` defines a static sphere action | No scalar momentum, time-kinetic coefficient or separation of scales. Instantaneous operator exchange below is a declared ansatz, not exact dynamical elimination. |
| Coupling signs and strengths | `bpr/flavor_source_selection.py` supplies positive g and eta with mu2>eta | No microscopic matching fixes those inputs. They are inherited unchanged; no frustrated interaction is added to generate mixing. |
| Ordering and one-body terms | The bosonic small-site example normal orders its quartic interaction | That does not fix ordering in a different effective theory. Full-square and normal-ordered prescriptions differ by an explicit one-body term. |

Substrate prime p, charge Q, monopole flux q and particle number N are different
objects. A three-state internal space is not a derivation of three physical
families. A blank entry in this table is an unresolved physical input, not
permission to substitute the demonstration default as a prediction.

## 2. Conditional finite operator model

Use modes `(u0,u1,u2,d0,d1,d2)`, satisfying the canonical anticommutation
relations on a 64-dimensional Fock space. The indices u,d label the two
conditional sectors, not a completed spatial SM field theory. The spin-one
representation below is **internal angular momentum**, not spacetime spin.

Let the existing monopole overlap be

\[
T_A=P(Y_A),\qquad P(h)_{ij}=\int dA\,\psi_i^*h\psi_j,
\qquad Q_{aA}=d\Gamma_a(T_A)=\sum_{ij}c_{ai}^\dagger(T_A)_{ij}c_{aj}.
\]

Harmonics are orthonormal in solid angle. The density harmonic operator is
\(Q_{aA}/R^2\). For a state with definite \(N_a>0\),

\[
(\gamma_a)_{ij}=\langle c_{aj}^\dagger c_{ai}\rangle,
\quad \rho_a=\gamma_a/N_a,
\quad \langle Q_{aA}\rangle=\operatorname{Tr}(\gamma_aT_A)
=N_a\operatorname{Tr}(\rho_aT_A).
\]

The transpose convention matters for complex occupations. The one-body
occupation obeys \(0\preceq\gamma\preceq I\) and \(\operatorname{Tr}\gamma=N\).
The vacuum has gamma=0 and **undefined** normalized rho. Full filling gives
gamma=I and rho=I/3, not a coherent rank-one occupation.

Retain the inherited parameters
\(R>0,\kappa\ge0,\mu^2>\eta\ge0,g_u,g_d>0,h_0\ge0\), and define

\[
d_\ell=\mu^2+\kappa\ell(\ell+1)/R^2,\quad
a_\ell=\frac{d_\ell}{d_\ell^2-\eta^2},\quad
b_\ell=\frac{\eta}{d_\ell^2-\eta^2}.
\]

The full-square instantaneous ansatz is

\[
H_{\rm full}=-h_0(g_uN_u+g_dN_d)
-\frac1{2R^2}\sum_A\left[
a_{\ell_A}(g_u^2Q_{uA}^2+g_d^2Q_{dA}^2)
+2b_{\ell_A}g_ug_dQ_{uA}Q_{dA}\right]. \tag{1}
\]

Each number is conserved. Different species' density bilinears commute even
though their individual creation/annihilation operators anticommute. All
sixteen number blocks are finite, with dimensions
\(\binom3{N_u}\binom3{N_d}\), summing to 64.

## 3. Overlap and ordering identities

Use conventional spin-one matrices J and the physical-axis convention
\(K=(J_x,-J_y,J_z)\) of the implemented monopole sections. Then

\[
P(1)=I,\qquad P(x_i)=K_i/2,\qquad
P(x_ix_j)=\delta_{ij}I/5+\{K_i,K_j\}/10.
\]

Multiplying the corresponding polynomials by the existing harmonic
normalizations gives all nine T matrices. Independent quadrature of each
harmonic must reproduce them, including the imaginary y and xy entries.

By irreducible tensor orthogonality,

\[
\operatorname{Tr}(T_AT_B)=t_\ell\delta_{AB},\quad
(t_0,t_1,t_2)=\left(\frac3{4\pi},\frac3{8\pi},\frac3{40\pi}\right),
\]
\[
\sum_{\ell_A=\ell}T_A^2=k_\ell I,\quad
(k_0,k_1,k_2)=\left(\frac1{4\pi},\frac3{8\pi},\frac1{8\pi}\right).
\]

The second identity follows from rotational invariance and tracing:
\(k_\ell=(2\ell+1)t_\ell/3\). Define \(K(a)=\sum_\ell a_\ell k_\ell\).

The canonical anticommutators imply

\[
Q_A^2={:Q_A^2:}+d\Gamma(T_A^2).
\]

Here normal ordering moves creation operators left, including fermionic signs;
it does not discard the contraction without changing the Hamiltonian. A
normal-ordered self-interaction control replaces each same-species square in
(1) by its normal-ordered quartic. The cross interaction is unchanged. Thus

\[
\boxed{H_{\rm normal}-H_{\rm full}
=\frac{K(a)}{2R^2}(g_u^2N_u+g_d^2N_d).} \tag{2}
\]

It is constant in a fixed-number block, but not across the Fock space. The two
prescriptions, without compensation, are different effective Hamiltonians.
They are not two equivalent calculations of a matched microscopic theory.
No arbitrary counterterm coefficient is introduced here.

## 4. Exact one-particle-per-sector solution

In the 1+1 block let

\[
X=\mathbf J_u\cdot\mathbf J_d,\qquad
x_J=[J(J+1)-4]/2\in\{-2,-1,1\},\quad J=0,1,2.
\]

A common y-sign reversal leaves this invariant dot product unchanged. The
degree contractions are

\[
C_0=\frac I{4\pi},\quad C_1=\frac{3X}{16\pi},\quad
C_2=\frac{6X^2+3X-8I}{80\pi}.
\]

For example, the last identity follows from the complete Hermitian tensor
basis and the spin-one swap polynomial \(\mathrm{Swap}=X^2+X-I\).

| Internal J | Dimension | C1 eigenvalue | C2 eigenvalue |
|---|---:|---:|---:|
| 0 | 1 | -3/(8pi) | 1/(8pi) |
| 1 | 3 | -3/(16pi) | -1/(16pi) |
| 2 | 5 | 3/(16pi) | 1/(80pi) |

The same-sector full-square reward is K(a) times identity on each triplet.
The same-sector normal-ordered quartic vanishes in a one-particle sector.
Accordingly,

\[
E_J^{\rm full}=-h_0(g_u+g_d)
-\frac{g_u^2+g_d^2}{2R^2}K(a)
-\frac{g_ug_d}{R^2}\left[\frac{b_0}{4\pi}+b_1C_{1,J}+b_2C_{2,J}\right].
\]

Normal ordering omits the K(a) term but does not change the channel gaps:

\[
E_0-E_2=\frac{9g_ug_d}{80\pi R^2}(5b_1-b_2),\qquad
E_1-E_2=\frac{3g_ug_d}{40\pi R^2}(5b_1+b_2). \tag{3}
\]

For eta>0, \(b_1\ge b_2>0\), so **the ground space is all of J=2, with
dimension five**. For eta=0, **all nine states are degenerate**. At kappa=0
and eta>0, the two excited channels coincide; the ground remains rank five.

Analytic projectors avoid choosing an arbitrary degenerate eigensolver basis:

\[
P_0=(X+I)(X-I)/3,\quad
P_1=-(X+2I)(X-I)/2,\quad
P_2=(X+2I)(X+I)/6.
\]

### Not a unique quantum source

The symmetry-invariant ground ensemble P2/5 has both one-body reduced states
I/3. This is a chosen ensemble over the ground subspace, not a unique pure
vacuum selected by the Hamiltonian. An aligned coherent product is also a
ground state, but does not exhaust the subspace. In particular,

\[
|2,0\rangle=\frac{|1,-1\rangle+|-1,1\rangle+2|0,0\rangle}{\sqrt6}
\]

is an entangled ground state whose reduced occupation is diag(1,4,1)/6.
Consequently, no unique scalar mean density follows from the quantum ground
space alone. A scalar response to a chosen mean density would be another
expectation-value prescription, not the full quantum scalar state. No physical
mass or mixing matrix is inferred from these internal ground representatives.

## 5. Why the classical and quantum selections differ

For a product state rho_u tensor rho_d in the 1+1 block, define

\[
m_{aA}=\operatorname{Tr}(\rho_aT_A),\qquad
V_{aA}=\operatorname{Tr}(\rho_aT_A^2)-m_{aA}^2.
\]

The exact identity is

\[
\boxed{\langle H_{\rm full}\rangle=E_{\rm classical}(\rho_u,\rho_d)
-\frac1{2R^2}\sum_Aa_{\ell_A}(g_u^2V_{uA}+g_d^2V_{dA}).} \tag{4}
\]

The old functional squared expectation values; the new Hamiltonian takes
expectations of squares. For entangled states the cross-sector expectation
also need not factorize, so (4) is explicitly a product-state identity.

For aligned coherent products, the quantum energy lies below the classical
minimum by

\[
\frac{g_u^2+g_d^2}{2R^2}
\left[\frac{3a_1}{16\pi}+\frac{9a_2}{80\pi}\right].
\]

At eta=0 the quantum self-terms are Casimir constants and no state within 1+1
is preferred. The classical self-term coherence selection therefore does not
carry over to this operator prescription. This does not contradict the
classical theorem: their variational functionals differ.

## 6. Number-sector comparison

A compact oracle for the same-sector reward is

\[
A(a)=\frac{3a_1+a_2}{8\pi},\qquad
S(N)=\frac{a_0N^2}{4\pi}+\mathbf1_{N\in\{1,2\}}A(a).
\]

Normal ordering replaces S(N) with S(N)-N K(a). For computation, simplify this
before adding baseline or cross-sector energies:

\[
S_{\rm normal}(N)=\binom N2\frac{3(a_0-a_1)+(a_0-a_2)}{8\pi}.
\]

It vanishes exactly at N=0,1 and at kappa=0. The corresponding operator
identity is \(\sum_A{:Q_A^2:}=0\), allowing the common a0 weight to be
removed before summing normal-ordered quartics. This avoids cancellation of
small physical terms against large self terms. For N=1 or 2 the sector is
an internal spin-one representation. Particle-hole conjugation reverses the
quadrupole sign and preserves the vector representation after the appropriate
basis change. Thus when both numbers are 1 or 2 the cross eigenvalues are

\[
W_J=\frac{b_0N_uN_d}{4\pi}+b_1C_{1,J}
+\epsilon_{N_u}\epsilon_{N_d}b_2C_{2,J},\quad
\epsilon_1=1,\ \epsilon_2=-1.
\]

If either number is 0 or 3 only the monopole cross term survives. Block energies
are

\[
E=-h_0(g_uN_u+g_dN_d)
-\frac{g_u^2S(N_u)+g_d^2S(N_d)}{2R^2}
-\frac{g_ug_d}{R^2}W,
\]

with the indicated normal-order substitution. The direct 64-dimensional CAR
construction independently checks these formulas, especially the particle-hole
sign. No numerical parameter scan is needed.

Freeze R=kappa=mu2=g_u=g_d=1, h0=2, eta=1/4. Then

\[
a=(16/15,48/143,112/783),\qquad b=(4/15,4/143,4/783).
\]

For these inputs the all-filled sector has the lowest energy under both
prescriptions. Its full-square energy is exactly \(-12-3/\pi\).
A short frozen-input certificate is available: the unweighted sum of the
squared density operators per species is
\([N^2/(4\pi)+\mathbf1_{N=1,2}/(2\pi)]I\le9I/(4\pi)\).
The largest default inverse-response eigenvalue is 4/3, so the total
full-square reward cannot exceed 3/pi. Every nonfilled sector has baseline
at least -10 and hence energy at least \(-10-3/\pi\). Normal ordering
only raises those energies. Its filled-sector shift is \(3K(a)<2\), with
\(K(a)=[4/15+18/143+14/783]/\pi\), so the filled sector is strictly lowest
for that prescription too. This proves the frozen comparison without a scan.

This is a property of the stipulated Hamiltonian with omitted bare matter
energies, not a derivation of the physical population. Number-conserving
unitary dynamics cannot transport an initially prepared 1+1 state to 3+3.
Ordering changes the inter-sector energy comparison through (2); a microscopic
matching calculation would also have to determine the associated one-body
terms. Full filling yields constant densities, not a flavor hierarchy.

## 7. Why square completion is not exact dynamical elimination

The existing flavor scalar action contains no time-kinetic coefficient.
Even if one hypothetically supplies scalar coordinates and momenta, the
formal completion of squares

\[
H=\tfrac12p^TM^{-1}p+
\tfrac12(q-K^{-1}J)^TK(q-K^{-1}J)-\tfrac12J^TK^{-1}J
\]

does not in general define independently displaced oscillators. Here J denotes
operator sources, not the internal angular momentum above. Projected source
components do not commute:

\[
[Q_A,Q_B]=d\Gamma([T_A,T_B])\ne0.
\]

The shifted coordinates therefore need not commute with one another. The
algebraic square identity is not a proof that the positive oscillator term
can simultaneously attain the naive displaced vacuum in every component.

With a specified temporal kernel, Gaussian mediator elimination generally
produces a time-nonlocal interaction. Replacing that kernel by its static
value needs a controlled regime and an ordering prescription. Neither has
been matched to the substrate here. Equation (1) is an explicit instantaneous
effective ansatz, **not** an exact finite-frequency scalar elimination.

## 8. Executable diagnostic and scope

- `bpr/quantum_flavor_sources.py`: finite operator construction, exact channels,
  ordering and number diagnostics, variance identity and matching ledger.
- `tests/test_quantum_flavor_sources.py`: independent operator and quadrature
  checks plus regression/integration tests.
- `python3 scripts/demo_quantum_flavor_sources.py`: frozen stdout report; add
  `--json` for strict structured output.

Numerical channel gaps must be evaluated after removing common energy shifts,
not by subtracting nearly equal total eigenvalues. Exact eta=0 degeneracy
must remain distinct from numerically unresolved positive-eta splitting.
Ground-space ensembles and explicit representatives are separately labeled;
no eigenvector returned by a diagonalizer is silently promoted to a vacuum.

Independent mathematical review found no substantive errors in the stated
conditional results. Separate overlap quadrature agreed within 1.1e-15;
explicit CAR normal-order contractions within 2.3e-16; independent 64-state
spectra matched all sixteen analytic number blocks under both prescriptions
within 1.8e-14. The variance identity and frozen full-filling certificate were
also verified. These checks concern the derivation, not experimental evidence.
Independent implementation review found and verified fixes for cancellation in
normal ordering. At eta=1e-20, h0=0, the 1+1 common energy now retains
-7.957747154594767e-22, with an independent CAR residual of 3.8e-37; at
eta=0, h0=1e-20 it retains -2e-20 exactly. The rewritten normal Hamiltonian
agreed with directly weighted CAR quartics within 4.4e-16 at nontrivial fixed
inputs. No substantive findings remained in the reviewed scope.

The new suite has **52 tests**. The final combined quantum, classical flavor,
gauge, foundation and historical regression selection passed **390 tests**.
One phase-scan test was intentionally deselected; the existing unregistered
`slow` marker warning remains. Eight independent algebra-check groups and both
modes of the previous two demos also passed. The new demo passes text and
strict JSON checks from outside the repository. The full repository suite
was not run; these checks do not constitute empirical validation.

The numerical implementation conservatively rejects unresolved intermediate
scales. In particular, for normal ordering at positive kappa, differences
between self-response weights must exceed the documented float64 resolution;
otherwise a ValueError distinguishes an unresolved positive gradient term from
exact kappa=0. These numerical restrictions do not narrow the mathematical
theorem. Tiny positive eta keeps its analytic rank-five label even when the
full-energy splitting is unresolved; angular projectors are checked separately.

This work supplies an operator-consistency diagnostic and a more precise
missing-physics inventory. It does not solve the substrate projection,
statistics, physical scalar bilinear, occupation preparation, temporal
mediator dynamics or coupling matching. The next genuine microscopic advance
requires one of those inputs to be derived, not another free interaction
chosen to rescue mixing. No fitting, benchmark access, optimization, production
Monte Carlo, remote calculation or deployment is part of this pass.

### Concrete substrate follow-up

The [ring-to-triplet projection test](substrate_triplet_projection_2026-09-12.md)
now derives a restricted three-mode Hamiltonian from the existing classical
ring, with explicit nonlinear leakage and a finite-time error bound. It does
not identify those bosonic modes with this module's stipulated fermions.
Direct local scalar sources span five real matrix directions, not the complete
nine-direction monopole source map. Their generated complex associative algebra is
nevertheless full M3(C), so the obstruction is confined to the direct linear
local-source prescription, not all possible effective constructions.
