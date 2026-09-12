# Toward a unified BPR extension: explicit constructions and unsolved steps

Date: 2026-09-12. **Status: NEW RESEARCH PROPOSAL, not the frozen theory, not a completed TOE.**

Companion: [Whole-theory audit](toe_audit_2026-09-12.md).

This note attempts a constructive continuation rather than only listing failures. It derives several useful conditional results, exposes where simple completions fail, and defines the calculations needed next. The constructions use established mathematical physics; their application here is a proposed research direction, not a claim of discovering a new established unification theory.

**Main recommendation:** keep BPR's geometry as a possible organizing ingredient, but stop treating internal labels as physical gauge bosons, fermions, or gravitons. Build and test the missing dynamical structures explicitly. The most conservative achievable target is a specified gauge–gravity effective theory with BPR-inspired flavor geometry. Deriving that target from a microscopic substrate is a separate, substantially harder goal.

Units are hbar=c=1 unless stated. `D_n` denotes the finite dihedral group. The root lattice `D5` used below denotes the Lie algebra so(10); these are NOT the same object.

## 1. An explicit gauge-invariant repair of Path B

### 1.1 Physical link Hilbert space and gauge condition

Take each oriented link to have Hilbert space C[D_n]. Local gauge transformations act by left multiplication at one end and right inverse multiplication at the other. Physical states obey Gauss constraints, implemented by a vertex-averaging projector P_G.

For an electric convolution operator to commute with both endpoint actions, its weights must be central. The frozen generating set {r,r^-1,s} is symmetric under inversion but not under conjugation. Symmetry under inversion guarantees Hermiticity, not gauge invariance.

### 1.2 Conjugacy averaging

Starting from the defective positive Cayley Laplacian, define the NEW operator

\[
\Delta_c=\frac1{|G|}\sum_{h\in G} L_h\Delta_S L_h^{-1}.
\]

Conjugacy averaging preserves positivity and Hermiticity. Since the rotation pair is already conjugacy invariant,

\[
\boxed{\Delta_c=3I-L_r-L_{r^{-1}}-
\frac1{|C_s|}\sum_{g\in C_s}L_g.}
\]

For odd n, C_s contains all n reflections; for even n it contains the n/2 reflections conjugate to the chosen s. Other class weights would define other models, so this is not a uniqueness theorem.

Schur's lemma now legitimately gives

\[
\Delta_c|_R=\epsilon_R I,
\quad \epsilon_R=3-
\frac{\chi_R(r)+\chi_R(r^{-1})+\chi_R(s)}{d_R}.
\]

Thus the original trace averages become actual scalar eigenvalues **of this new operator**. The independent script verifies positivity, both endpoint commutators, and a positive stochastic heat kernel for all four substrate classes.

### 1.3 A compatible temporal action

In this subsection use **dimensionless lattice energy and time units**, with a positive dimensionless coupling lambda. A physical energy scale E0 restores units as H_phys = E0 H_new and dimensionless a_t = E0 times physical time spacing. Natural units alone would not supply E0. Both Delta_c and V below are dimensionless; this convention also applies to the reported single-link transfer energies and heat-kernel script.

Choose, explicitly,

\[
H_{\rm new}=\frac1\lambda\sum_\ell\Delta_{c,\ell}
+\lambda\sum_p\left[1-\frac{\chi_{E_1}(U_p)}2\right],
\qquad H_{\rm phys}=H_{\rm new}|_{\operatorname{im}P_G}.
\]

Its electric time step is

\[
K_E(a_t)=\exp[-a_t\Delta_c/\lambda].
\]

A symmetric transfer step is

\[
T_{a_t}=P_Ge^{-a_t\lambda V/2}
\left(\bigotimes_\ell K_E(a_t)\right)
e^{-a_t\lambda V/2}P_G.
\]

On the physical subspace this is positive. As a_t tends to zero, its logarithm recovers the declared Hamiltonian, with controlled Trotter corrections. Character inversion supplies temporal plaquette weights:

\[
k_{a_t}(g)=\frac1{|G|}\sum_Rd_R
\exp[-a_t\epsilon_R/\lambda]\chi_R(g^{-1}).
\]

The normalization assumes normalized finite-group Fourier conventions. Equivalently, use the matrix exponential directly and read its convolution weights. This is **not** the existing temporal E1 Wilson weight.

### 1.4 What this repair does not establish

It gives a well-defined finite-group quantum gauge model. It does not select a physical coupling, preserve the old mass formulas dynamically, supply SU(3), change the number of spatial dimensions, or derive chirality/gravity. Existing MC data remain data for their original Wilson action. A new action needs a separately preregistered numerical program; do not silently reopen or retune a failed benchmark.

## 2. A real U(1) sector requires new link degrees of freedom

A conservative quantum rotor extension is

\[
H=\frac U2\sum_i n_i^2
-t\sum_{\langle ij\rangle}\cos(\theta_j-\theta_i-A_{ij})
+\frac{g^2}{2}\sum_\ell E_\ell^2
-K\sum_p\cos B_p,
\]

where theta,n are conjugate site variables, A,E independent link variables, and B_p is the oriented sum of A around a plaquette. Gauss's law couples electric divergence to site charge. Under

\[
\theta_i\mapsto\theta_i+\alpha_i,\quad
A_{ij}\mapsto A_{ij}+\alpha_j-\alpha_i,
\]

the hopping term is invariant and the plaquette flux is gauge invariant.

For weak fields, electric and magnetic terms yield a lattice Maxwell description. In **three spatial dimensions**, an appropriate deconfined Coulomb phase can support a photon. Two spatial dimensions have different compact-U(1) infrared physics; one cannot copy the conclusion across dimensions without additional matter or mechanisms.

This construction explains exactly what was missing from a pure phase-difference connection. It also adds U,t,g,K, quantum kinematics, a Gauss constraint, and a phase choice. It is a candidate model definition, not a derivation from the original four axioms or a prediction of alpha.

## 3. Enlarge the boundary current algebra honestly

### 3.1 A conditional lower bound

If the claimed gauge content is to be realized as **simultaneous commuting unitary holomorphic affine currents**,

\[
\widehat{su(3)}_k\oplus\widehat{su(2)}_l\oplus\widehat{u(1)},
\]

positive integer non-Abelian levels give Sugawara central charge

\[
c\ge\frac{8k}{k+3}+\frac{3l}{l+2}+1\ge 2+1+1=4.
\]

The nonnegative coset central charge gives the inequality [1]. Hence the original c=1 compact boson cannot supply this particular gauge-current realization. This does not rule out a different bulk realization or a theory with additional degrees of freedom.

### 3.2 A concrete Spin(10) current sector

Introduce five compact chiral bosons X^I normalized by

\[
X^I(z)X^J(0)\sim-\delta^{IJ}\ln z.
\]

Choose the even root lattice

\[
D_5=\{v\in\mathbb Z^5:\sum_Iv_I\text{ even}\}.
\]

Its roots alpha=±e_i±e_j have alpha^2=2. The currents

\[
H^I=i\partial X^I,\qquad E_\alpha=:\!e^{i\alpha\cdot X}\!:
\]

have conformal weight one and realize so(10) at level one, with the standard cocycle factors required in the vertex construction [2]. There are 40 root currents plus five Cartan currents, matching dim so(10)=45, and

\[
c=\frac{1\cdot45}{1+8}=5.
\]

One spinor module has the 16 weights

\[
w=\tfrac12(\pm1,\pm1,\pm1,\pm1,\pm1)
\]

with even minus-sign parity. Their weight is h=w^2/2=5/8. The independent script verifies the counts and norms.

### 3.3 Crucial limits of this result

- This is an **added internal current sector**, not emergent four-dimensional gauge dynamics.
- Spinor-module primary fields are not members of the vacuum root-lattice algebra; they cannot simply be added as mutually local bosonic vacuum operators.
- An internal Spin(10) spinor and conformal weight 5/8 are not four-dimensional fermionic spin/statistics.
- The chiral theory has gravitational/current anomalies. It requires an explicit opposite chirality or suitable anomaly inflow. Natural inflow for a 1+1D chiral theory is from 2+1D, not automatically the desired 3+1D spacetime.
- If the original independent c=1 sector is retained, the total chiral central charge is **six**, not five. A c=5 replacement must explicitly state that it replaces/reorganizes the original sector.
- Nothing about the finite dihedral group D5 implies this Lie root lattice. The notation is an accident, not a derivation.

The useful progress is an explicit possible algebra with adequate capacity, instead of assigning particle names to unrelated roots. Selecting it dynamically and constructing its bulk interpretation remain open.

## 4. A conventional anomaly-free four-dimensional target

Independently of the current algebra, choose the target field content

\[
G=\operatorname{Spin}(10),\qquad \Psi_i\in16_L.
\]

For now the number of i labels is a parameter, not three by theorem. On ordinary spin manifolds this chiral Spin(10) field content has no perturbative cubic gauge anomaly and no usual pi_4 global gauge anomaly. Additional global/non-spin formulations require their own checks [3,4].

The branching

\[
16\to10\oplus\overline5\oplus1
\]

contains one Standard Model family plus a neutral conjugate neutrino. In all-left-handed notation,

\[
Q:(3,2)_{1/6},\ u^c:(\bar3,1)_{-2/3},\ d^c:(\bar3,1)_{1/3},
\]
\[
L:(1,2)_{-1/2},\ e^c:(1,1)_1,\ \nu^c:(1,1)_0.
\]

### 4.1 Derive rather than merely assert anomaly cancellation

For one family,

\[
[SU(3)]^3:2-1-1=0,
\]
\[
[SU(3)]^2Y:2(1/6)(1/2)+(-2/3)(1/2)+(1/3)(1/2)=0,
\]
\[
[SU(2)]^2Y:3(1/6)(1/2)+(-1/2)(1/2)=0,
\]
\[
Y^3=6(1/6)^3+3(-2/3)^3+3(1/3)^3+2(-1/2)^3+1=0,
\]

and the gravitational-hypercharge sum also vanishes. There are four SU(2) doublets per family, so the usual SU(2) global anomaly cancels [4]. Cancellation works **per family** and therefore cannot select three. A neutral right-handed neutrino is not necessary for SM anomaly cancellation.

More generally, impose the usual Yukawa charge relations with Y_Q=q and Higgs charge h. Then

\[
(Y_Q,Y_{u^c},Y_{d^c},Y_L,Y_{e^c},Y_{\nu^c})
=(q,-q-h,-q+h,-3q,3q+h,3q-h)
\]

cancels those anomalies for arbitrary q,h. Requiring a neutral nu^c imposes h=3q; choosing the charge normalization q=1/6 gives the conventional values. This displays the assumptions needed, rather than falsely claiming anomalies alone select every charge.

### 4.2 A genuine unification normalization

Inside SU(5),

\[
Y=\operatorname{diag}(-1/3,-1/3,-1/3,1/2,1/2),\quad
\operatorname{tr}_5Y^2=5/6.
\]

With generators normalized by tr T_aT_b=delta_ab/2,

\[
T_Y=\sqrt{3/5}Y,\quad g_Y=\sqrt{3/5}\,g_U,
\quad\sin^2\theta_W(M_U)=3/8.
\]

This is a conditional representation-theoretic result. It is not the observed low-energy angle. Thresholds and RG evolution must be computed from an actual particle spectrum; measured low-energy couplings cannot be used to fit thresholds and then counted as predicted outputs.

## 5. A conditional chiral-family mechanism on an internal sphere

### 5.1 Specify the geometry being added

Assume a six-dimensional product background M^(3,1)×S2 of sphere radius R, a six-dimensional Weyl fermion, and an extra commuting U(1)_X line bundle L=O(q). This S2 is an **added internal compact space**, not automatically the original BPR spatial boundary. No such identification is proven here.

Absorb charge normalization into the integer

\[
q=\frac1{2\pi}\int_{S^2}F_X\in\mathbb Z.
\]

A flux inside Spin(10) generally breaks it to the centralizer and changes different component indices. The extra commuting U(1) avoids that immediate problem, at the cost of additional gauge structure and anomaly constraints.

### 5.2 The actual index calculation

Since K_(S2)=O(-2), its spin bundle is O(-1). For q>0,

\[
\ker D_+\simeq H^0(S^2,O(q-1)),\quad
\dim\ker D_+=q,\quad\ker D_-=0.
\]

For negative q the chirality reverses; the index is always q [5]. For flat color rank r, the kernel is C^r tensor ker D_L. Thus r=3,q=1 gives **one color triplet**, not three families. q=3 gives three copies of each supplied gauge multiplet.

On a round sphere the nonzero eigenvalues are

\[
\lambda_{n,\pm}=\pm R^{-1}\sqrt{n(n+|q|)},\quad
\text{multiplicity }|q|+2n,\quad n=1,2,\ldots.
\]

The zero modes form internal angular momentum j=(|q|-1)/2. The two-dimensional index counts internal chirality; relating it to four-dimensional chirality requires the six-dimensional Weyl condition, schematically Gamma_7=gamma_5 tensor sigma_3. A six-dimensional **Dirac** fermion contains both Weyl sectors and yields a vectorlike four-dimensional spectrum unless an additional mechanism removes the mirrors.

### 5.3 Explicit wavefunctions and family overlaps

For unit sphere and q>0, write N=q-1. In one gauge patch, including a common spin/gauge frame understood in all components, normalized profiles can be represented by

\[
\psi_m(\theta,\phi)=
\sqrt{\frac{q}{4\pi}{N\choose m}}
\left(\cos\frac\theta2\right)^{N-m}
\left(\sin\frac\theta2\right)^m e^{im\phi},\quad m=0,\ldots,N.
\]

Patch transformations are essential globally; these are sections, not globally scalar functions. For radius R insert 1/R. Orthogonality follows from the azimuthal integral and

\[
\int d\Omega\,
\cos^{2(N-m)}(\theta/2)\sin^{2m}(\theta/2)
=4\pi\frac{m!(N-m)!}{(N+1)!}.
\]

Hence q=3 gives three normalized profiles, equivalent to quadratic holomorphic sections (v^2,uv,u^2). This is a real construction of multiplicity. It still chooses q=3.

### 5.4 Flux selection and stabilization are not automatic

For a Maxwell term -F^2/(4g_6^2) and fixed radius,

\[
F_{\theta\phi}=\frac q2\sin\theta,\quad
F_{ab}F^{ab}=\frac{q^2}{2R^4},
\]

so its contribution to the four-dimensional energy density before gravitational frame/rescaling issues is

\[
V_{\rm flux}(R,q)=\frac{\pi q^2}{2g_6^2R^2}.
\]

At fixed R it favors q=0, not q=3. Stabilizing R changes the full effective potential and needs backreaction, curvature, cosmological terms, matter and possible quantization/tadpole constraints. Within a fixed topological sector q is conserved unless flux-changing processes are available; its energy ordering alone is not a transition mechanism. Either way no selection of three has been established [6].

A single six-dimensional Weyl 16 with U(1)_X charge is **not asserted to be a consistent six-dimensional theory**. Six-dimensional gauge, mixed, gravitational and global anomalies must cancel in a specified enlarged spectrum, potentially with explicit Green-Schwarz/inflow terms. Four-dimensional anomaly cancellation does not settle those requirements [7].

## 6. Deriving constraints on flavor rather than fitting labels

### 6.1 A mass matrix must be an operator overlap

If an actual higher-dimensional Yukawa coupling exists, dimensional reduction gives, schematically,

\[
(Y_f)_{ij}=y_{6,f}\int_{S^2}d^2y\sqrt g\,
\psi_{L,i}^\dagger(y)h_f(y)\psi_{R,j}(y).
\]

The Higgs profile normalization and dimension of y_6 must be supplied; their product gives a dimensionless four-dimensional Yukawa. Allowed contractions, gauge charges, and higher-dimensional chirality must be checked in a full model. This illustrative overlap is not a claim that the single Weyl construction above already supplies every needed Yukawa interaction.

In four-dimensional EFT,

\[
M_f=\frac v{\sqrt2}Y_f,
\quad U_{fL}^\dagger Y_fY_f^\dagger U_{fL}=\operatorname{diag}(y_{f,i}^2),
\quad V_{\rm CKM}=U_{uL}^\dagger U_{dL}.
\]

Physical comparisons require RG evolution and a specified mass scheme.

### 6.2 The symmetric construction fails to generate hierarchy

For identical orthonormal left/right internal profiles, a neutral constant h, and a family-independent coupling,

\[
Y_{ij}=y_0\delta_{ij}.
\]

The masses are degenerate. CKM is then unphysical within degenerate subspaces, rather than uniquely determined to be zero; it can be chosen to be the identity. The monopole index does not explain the observed hierarchy.

### 6.3 Arbitrary profiles merely reintroduce arbitrary flavor matrices

For the q=3 internal triplet,

\[
\operatorname{End}(V_{j=1})=V_0\oplus V_1\oplus V_2,
\quad 9=1+3+5.
\]

For a real scalar profile and identical left/right spaces, projected multiplication P h P is Hermitian. Its projections on the l=0,1,2 harmonics are all nonzero; these nine real profile coefficients span every Hermitian 3×3 matrix. Higher harmonics vanish in this simple direct-overlap projection.

Equivalently, any such matrix can be expanded as

\[
Y=aI+\sum_i b_iJ_i+
\sum_{ij}c_{ij}\left[\tfrac12\{J_i,J_j\}-\tfrac23\delta_{ij}I\right],
\]

with c symmetric traceless and real coefficients. The script verifies rank nine for this matrix basis. The surjectivity of projected multiplication additionally follows from the products of the quadratic holomorphic sections, or the nonzero spin-one coherent-state multipoles.

Independent unrestricted profiles h_u,h_d can therefore reproduce arbitrary positive mass eigenvalues and CKM mixing. That is a parameterization, not a prediction. Smoothness or locality alone does not cure it. A common frozen action must select the profiles, their coefficients, symmetry breaking, and vacuum before masses are compared.

### 6.4 Exact shared cyclic symmetry prevents nontrivial mixing in this ansatz

Assume the same left/right C_n lift, a neutral invariant Higgs, and the internal j=1 basis m=-1,0,1. Then

\[
U_n=\operatorname{diag}(e^{-2\pi i/n},1,e^{2\pi i/n}),\quad
U_n^\dagger Y_fU_n=Y_f.
\]

For n=5,8,9,12 the eigenvalues are distinct, so Y_f is diagonal. If both up and down sectors obey this in the same basis, they give no physical mixing for nondegenerate masses. Thus a literal unchanged cyclic geometry does not produce CKM in this simplest construction.

Different left/right lifts, charged Higgs profiles, spontaneous symmetry breaking, or additional tensor/derivative operators can change the result, but must be explicit new ingredients. A common real action with a CP-preserving vacuum also cannot generate the measured nonzero Jarlskog invariant; a specified CP-breaking mechanism is needed.

### 6.5 Dihedral geometry and flux need careful compatibility

An orientation-reversing map of S2 sends c1=q to -q. For q nonzero, it has no complex-linear lift to the same monopole line bundle. A parity operation may require doubled ±q bundles or a charge-conjugating/antilinear lift, with a fresh chirality analysis.

However, a proper pi rotation about an equatorial axis also exchanges m=+1 and -1 while preserving orientation and flux. It can generate a rotational dihedral subgroup of SO(3). If that common unitary symmetry is imposed along with the axial cyclic symmetry, it forces equality of those two diagonal entries. Do not conflate this valid degeneracy statement with the separate orientation-reversal obstruction.

**Constructive research target:** derive a small, shared symmetry-breaking vacuum that escapes degeneracy and zero mixing, generates CP violation, and predicts more independent observables than it introduces adjustable coefficients. Do not search for arithmetic expressions reproducing 24,283,210,59 after seeing those targets.

## 7. A gauge–gravity EFT target, with an honest assumption ledger

A conventional effective action could be

\[
S_{\rm EFT}=\int d^4x\sqrt{-g}\left[
\frac{\bar M_{\rm Pl}^2}{2}R-\rho_{\rm vac}
-\frac1{4g_{10}^2}\operatorname{Tr}F_{\mu\nu}F^{\mu\nu}
+\sum_i i\overline\Psi_i\slashed D\Psi_i
+\mathcal L_H+\mathcal L_Y
+\mathcal L_{\rm boundary/moduli}+\sum_a\frac{c_a\mathcal O_a}{\Lambda^{d_a-4}}
\right].
\]

The trace convention must be fixed with the gauge coupling. Symmetry breaking requires specified Higgs representations and a potential. For example 16×16=10_s+120_a+126_s organizes possible Yukawa channels, but adding those scalar channels without constraints introduces substantial flavor freedom. Proton decay, neutrino masses, extra light particles, and thresholds must all be calculated.

This is **GR plus a grand-unified EFT**, not quantum-gravity UV completion. Its gravity and gauge fields are postulates until obtained from a common substrate.

| Ingredient | Present status in this construction |
|---|---|
| Quantum Hilbert space and probability rule | Assumed, not emergent from deterministic RPST |
| 3+1D spacetime and spin structure | Target/background assumption |
| Spin(10) and chiral 16 | New field-content choice |
| Current-algebra sector | Explicit optional algebra, not a 4D dictionary |
| Internal S2 and commuting flux | Optional extra-dimensional hypothesis |
| Flux q=3 | Empirical family-count input |
| Flavor profiles and CP breaking | Not selected; must be dynamically derived |
| Higgs potential, v and gauge coupling | Additional matching data until derived |
| Einstein term and Planck scale | EFT input until microscopic matching exists |
| Vacuum energy and dark sector | Unsolved, not supplied by the index construction |

One dimensionful scale may be used to set units. The important missing predictions include dimensionless ratios, coupling matching, and the observed vacuum-energy hierarchy. Calling a parameter discrete does not remove its information content or observational calibration.

## 8. What would actually establish emergent gravity and quantum consistency

### 8.1 A continuum theory, not only a fitted lattice spectrum

Find a trajectory with xi_lattice=xi_physical/a tending to infinity while physical ratios stabilize. Massive channels require a m_i tending to zero, with finite-volume, anisotropy, and excited-state effects controlled. A finite-coupling second-order transition is not the only route; critical endpoints, asymptotically free limits, and other controlled scalings must be assessed on their actual equations.

Require reflection positivity or a positive physical transfer matrix. For positive-time observables F,

\[
\langle(\Theta F)F\rangle_E\ge0.
\]

Positive Boltzmann weights alone do not imply this. Together with the relevant additional Euclidean axioms and continuum covariance, this is part of reconstructing a unitary relativistic theory [8].

### 8.2 Relativity, fermions, and mirrors

Show a common limiting speed for every species, Lorentz-covariant interactions, and control of relevant/marginal Lorentz violations. Isotropic spatial dispersion alone is insufficient [9].

For fermions, demonstrate Lorentz spin, statistics, and physical operators or charged superselection sectors. Bosonic microscopic variables do not categorically forbid emergent fermions [10]. But the Nielsen–Ninomiya hypotheses obstruct simply retaining one desired Weyl cone while ignoring its partners. An explicit overlap/Ginsparg–Wilson, domain-wall, or interacting mirror-gapping construction changes the relevant hypotheses and must be analyzed in full [11].

### 8.3 A graviton must be a dynamical state

Calculate a gauge-invariant/dressed physical tensor sector whose Lorentzian correlators contain the appropriate massless helicity-two pole with positive residue, rather than inserting a TT propagator by definition. In an applicable flat-space scattering regime, the soft factor

\[
\mathcal M_{n+1}\simeq\left[
\sum_i\eta_i\kappa_i\frac{p_i^\mu p_i^\nu\epsilon_{\mu\nu}}{p_i\cdot q}
\right]\mathcal M_n
\]

must be gauge independent. This demands universal coupling for connected sectors under the usual assumptions [12]. Supply diffeomorphism Ward identities or equivalent constraints eliminating unphysical polarizations.

The Weinberg–Witten obstruction applies when the continuum theory possesses the required local Lorentz-covariant conserved stress tensor with suitable physical matrix elements. GR and some holographic constructions evade particular premises. Saying only that the microscopic model is a lattice does not identify a viable continuum loophole [13].

Induced curvature terms on an externally supplied metric do not prove any of these. Their coefficients depend on actual bulk species, nonminimal couplings, regulator, and bare counterterms:

\[
\bar M_{\rm Pl,ren}^2=\bar M_{\rm Pl,bare}^2+
\delta M^2(\Lambda,\mu,\text{scheme}).
\]

A fully specified UV model could fix matching. An unspecified cutoff integral cannot [14]. The cosmological constant remains an independent, radiatively sensitive matching problem.

## 9. An achievable near-term physical calculation

The repo's phonon–phason elastic functional can support a meaningful **ordinary condensed-matter test** without claiming a TOE signal. With

\[
f=\tfrac12Ce^2+\tfrac12Ks^2+Des,
\quad C>0,\ K>0,\ D^2<CK,
\]

and overdamped phason relaxation, eliminating a scalar Fourier component gives

\[
C_{\rm eff}(q,\Omega)=C-\frac{D^2}{K-i\Omega\Gamma_w/q^2}
=C\left[1-\frac{g^2}{1-i\Omega\tau_q}\right],
\]

where g^2=D^2/(CK) and tau_q=Gamma_w/(Kq^2). For weak coupling and unit participation,

\[
\frac{\Delta f}{f}\simeq-\frac{g^2}{2[1+(\Omega\tau_q)^2]},
\quad \Delta Q^{-1}\simeq\frac{g^2\Omega\tau_q}{1+(\Omega\tau_q)^2}.
\]

These predict linked softening and loss rather than an arbitrary Q multiplier on a frequency shift. A quasicrystal/periodic-approximant comparison should fit both observables with the same parameters and control ordinary defects, thermal stress, and surface loss. Success would initially support standard phonon–phason elasticity. BPR-specific evidence requires an additional preregistered prediction not already made by that theory.

## 10. Ordered research program and stop conditions

1. **Repair model identity.** Choose the central Hamiltonian with its heat kernel or the existing Wilson transfer model. Derive Gauss invariance and test transfer positivity before another production spectrum run.
2. **Establish scale separation.** Determine phase structure and continuum/controlled-effective-theory range, with physical spatial quantum numbers. Do not equate anyon labels with 4D spins.
3. **Freeze a chiral target.** Specify gauge group, all fermions and mirrors, scalar fields, and anomaly cancellation. The optional current algebra is not a substitute.
4. **Construct one dynamical flavor calculation.** Derive a shared profile-selection action and compute matrices, rather than independent mass labels. If symmetry forces degeneracy or zero mixing, record the failure before adding parameters.
5. **Predict before matching.** Lock parameters and scheme, then calculate unseen ratios, mixing relations, new states, or other discriminating observables. Use actual RG thresholds, not reverse-fit subtraction.
6. **Derive gravity or label it assumed.** Calculate its pole, coupling, constraints, and UV matching. If these remain absent, report a gauge/flavor EFT, not a TOE.
7. **Close cosmology separately.** Determine vacuum-energy stability, dark matter interactions and abundance, neutrino/radiation history, and conserved background/perturbation evolution; compare a single model jointly with observations.

An extension that requires arbitrary independent Yukawas, arbitrary thresholds, and an inserted Einstein action may remain a consistent EFT. It does not explain the Standard Model's free structure or complete BPR's original claim. The useful advance here is that the exact missing constructions and several immediate failure tests are now explicit.

## 11. Executable conditional models (2026-09-12 implementation)

The gauge implementation is `bpr/gauge_heat_kernel.py`. It keeps the new
central Hamiltonian separate from both the defective historical generator and
the Euclidean character-Wilson simulations. An isolated square with four
Gauss-constrained vertices has class-function wavefunctions of one holonomy.
In the orthonormal character basis its electric term is
`4 diag(epsilon_R)/lambda`; the magnetic term is multiplication by
`1-chi_E1/2`. The symmetric split transfer approaches this interacting
Hamiltonian under time refinement. This finite-system test supplies neither
bulk glueball masses nor a Benchmark v3 result.

### A specified source-response action, not a selected flavor vacuum

The flavor implementation is `bpr/chiral_flavor_prototype.py`. Its physical
scope is an **assumed four-dimensional SM chiral EFT**, not a demonstrated
six-dimensional Yukawa reduction or Spin(10) Yukawa channel. With the usual
Higgs `H:(1,2)_(1/2)`, the conventional four-component operators are
`-bar(Q_L) Y_u tilde(H) u_R - bar(Q_L) Y_d H d_R
 - bar(L_L) Y_e H e_R + h.c.`. The family-space overlap matrices are proposed
coefficients of those gauge-invariant operators. Choosing one sphere and q=3
is an input; internal chirality alone does not establish this four-dimensional
field content.

For each sector introduce an externally supplied real scalar source J and

\[
S[h]=\int_{S_R^2}dA\left[
\frac{\kappa}{2}|\nabla h|^2+
\frac{\mu^2}{2}(h-h_0)^2-Jh\right],
\qquad R>0,\quad\mu^2>0,\quad\kappa\geq0.
\]

Use real harmonics orthonormal with respect to unit-sphere solid angle.
Stationarity gives

\[
h_{\ell m}=h_0\sqrt{4\pi}\,\delta_{\ell0}\delta_{m0}
+\frac{J_{\ell m}}{\mu^2+\kappa\ell(\ell+1)/R^2}.
\]

The Hessian coefficients are `R² mu² + kappa l(l+1) > 0`, so this is a unique
minimum and a bounded-below functional, not necessarily a positive-valued
functional. Actual neutral overlaps `integral dA psi_i* h psi_j` determine
Yukawa matrices. Their singular values give masses only after an externally
specified factor `v*y_eff/sqrt(2)` and a matching scale/scheme are supplied.

The demonstration fixes `R=kappa=mu²=1`, `h0=2`,
`J_u=z+(3z²-1)/4`, and `J_d=x+z/3+xy/2`, with x,y,z unit-sphere coordinates,
**before calculation**. At unit effective Yukawa coupling, in the stated
profile order, the analytic projected matrices are

\[
Y_u=\operatorname{diag}\left(\frac{913}{420},\frac{139}{70},
\frac{773}{420}\right),\qquad
Y_d=\begin{pmatrix}
37/18 & 1/(6\sqrt2) & i/140\\
1/(6\sqrt2) & 2 & 1/(6\sqrt2)\\
-i/140 & 1/(6\sqrt2) & 35/18
\end{pmatrix}.
\]

These follow from `P(z)=diag(1/2,0,-1/2)`,
`P(P2(z))=diag(1,-2,1)/10`, nearest-neighbor entries
`P(x)_(01)=P(x)_(12)=1/(2 sqrt(2))`, and `P(xy)_(02)=i/10`.
The monopole-profile phase convention fixes the displayed imaginary signs.
They give dimensionless singular values approximately
`(1.84047619,1.98571429,2.17380952)` and
`(1.82417276,2,2.17582724)` in ascending order, not measured quark masses.
The quadrature implementation agrees with these analytic matrices. The forced
example gives

```text
|V| = [[0.65829681, 0.67026663, 0.34261927],
       [0.67026663, 0.31856757, 0.67026663],
       [0.34261927, 0.67026663, 0.65829681]]
J = Im(V00 V11 conjugate(V01) conjugate(V10)) = 0.00912538095
```

This CP-odd invariant belongs to the stipulated EFT matrices; it is not a
prediction of observed CKM CP violation or a demonstrated spontaneous CP
mechanism. The unforced example returns undefined mixing, while the cyclic
nondegenerate control returns the identity in absolute value, within roundoff.
These are target-independent toy sources, not an
optimization result or a BPR prediction. Unforced and common-cyclic-symmetry
controls must accompany the forced example. In a degenerate singular-value
subspace the left rotations are not unique; a physical CKM matrix must not be
reported from that arbitrary basis.

Every low-harmonic response denominator is nonzero. Consequently arbitrary
sources still span all nine Hermitian q=3 matrix directions. This construction
makes the action-to-profile-to-matrix calculation executable, but **does not
solve predictive source selection**. Four-dimensional anomaly cancellation
does not establish extra-U(1) or six-dimensional anomaly cancellation, flux or
radius selection, mirror removal, or an allowed higher-dimensional Yukawa
contraction. Real scalar sources do not by themselves guarantee real matrices
or CP conservation in the fixed monopole background.

## 12. Endogenous-source follow-up: a conditional alignment obstruction

The [source-selection derivation](flavor_source_selection_2026-09-12.md) adds
one new fixed-occupation mean-field model rather than altering the forced
prototype above. Positive trace-one occupation matrices generate monopole
densities, and the scalar fields and occupations minimize a shared bounded
action. An analytic energy bound classifies all global minima: coherent
densities with a common axis for attractive inter-sector coupling, or
independent coherent axes at zero coupling.

The model selects source shapes, but the common-axis minima have identity
absolute mixing and zero CP quartet. Removing the coupling leaves relative
mixing unselected; equal sector couplings also give equal spectra. These are
limitations, not observed flavor predictions. The action and fixed occupation
are added assumptions, not a substrate derivation or an exact quantum-vacuum
calculation. The independent stdout demo is
`python3 scripts/demo_flavor_source_selection.py` (optionally `--json`).
Previous gauge, fixed-source examples and historical benchmark status remain
unchanged.

## 13. Operator-level diagnostic and the remaining microscopic gap

The [quantum source diagnostic](quantum_flavor_sources_2026-09-12.md) compares
the classical functional with a separately stipulated six-mode internal
fermion Hamiltonian. Exact density-operator algebra reveals a rank-five
ground subspace in the 1+1 sector for positive exchange, including coherent
products and entangled states; zero exchange leaves all nine states degenerate.
A normalized ground projector is a diagnostic ensemble, not a selected vacuum.
Normal ordering changes number-sector energies by an explicit one-body term.

This is not an exact quantization of the static scalar action. The substrate
projection, statistics, population, physical scalar bilinear and temporal
mediator kernel remain missing matching inputs. No additional interaction was
introduced to generate mixing. Run the separate stdout-only report with
`python3 scripts/demo_quantum_flavor_sources.py` (optionally `--json`).

## Reproducibility

The executable models have separate tests in `tests/test_gauge_heat_kernel.py`
and `tests/test_chiral_flavor_prototype.py`. The stdout-only demonstration is
`python3 scripts/demo_conditional_gauge_flavor.py` (add `--json` for structured
matrices, inputs and assumption ledgers). It does not run MC or read benchmark
targets. Existing Wilson runs retain their original action and numbers.

Gauge verification on 2026-09-12: **110 tests passed**. At lambda=1.3 with
time steps 0.08, 0.04 and 0.02, the effective-Hamiltonian errors decrease with
observed orders 1.994–1.998 on the first refinement and 1.999 approximately
on the second, for all four groups. Transfer eigenvalues remain positive in
these diagnostics. Independent review verified endpoint invariance, the
four-link Gauss reduction and the numerical fixes. Extreme spectra outside
float64 resolution are explicitly rejected; no spurious negative ground
energy is interpreted physically. The historical gauge/foundation regression
selection passed 75 tests, with one phase-scan test deliberately deselected
and an existing unregistered `slow` marker warning. The combined gauge,
flavor and historical regression selection passed **256 tests**, with the
same one phase-scan deselection and marker warning. This includes 71 new
flavor tests and both demonstration output modes. Independent flavor review
confirmed the analytic matrices within 3.6e-15, the CP-invariant sign, radius
normalization, singular-degeneracy handling and four-dimensional anomaly
accounting, with no substantive correctness defect found. The full repository suite
was not run, and these algebraic/numerical tests are not physical validation.

`python3 scripts/check_toe_derivations.py` runs eight independent algebra-check groups without importing the BPR modules, fitting data, reading sealed targets, or running Monte Carlo. All eight passed in the review environment. They establish the stated algebraic examples, not the consistency of an unbuilt six-dimensional model or experimental validation.

## Sources

1. Goddard–Kent–Olive, *Virasoro Algebras and Coset Space Models*: https://doi.org/10.1016/0370-2693(85)91145-1
2. Frenkel–Kac, *Basic representations of affine Lie algebras and dual resonance models*: https://doi.org/10.1007/BF01391662 ; Witten, *Nonabelian Bosonization in Two Dimensions*: https://doi.org/10.1007/BF01215276
3. Fritzsch–Minkowski, *Unified Interactions of Leptons and Hadrons*: https://doi.org/10.1016/0003-4916(75)90211-0
4. Witten, *An SU(2) Anomaly*: https://doi.org/10.1016/0370-2693(82)90728-6 ; Wang–Wen–Witten, *A New SU(2) Anomaly*: https://arxiv.org/abs/1810.00844
5. Atiyah–Singer, *The Index of Elliptic Operators: I*: https://doi.org/10.2307/1970715
6. Randjbar-Daemi–Salam–Strathdee, *Spontaneous Compactification in Six-Dimensional Einstein-Maxwell Theory*: https://doi.org/10.1016/0550-3213(83)90247-X
7. Alvarez-Gaume–Witten, *Gravitational Anomalies*: https://doi.org/10.1016/0550-3213(84)90066-X
8. Osterwalder–Schrader, *Axioms for Euclidean Green's Functions*, I: https://doi.org/10.1007/BF01645738 ; II: https://doi.org/10.1007/BF01608978
9. Collins et al., *Lorentz invariance and quantum gravity: an additional fine-tuning problem?*: https://arxiv.org/abs/gr-qc/0403053
10. Levin–Wen, *Fermions, strings, and gauge fields in lattice spin models*: https://arxiv.org/abs/cond-mat/0302460
11. Nielsen–Ninomiya, *Absence of Neutrinos on a Lattice*: https://doi.org/10.1016/0550-3213(81)90361-8 ; Ginsparg–Wilson: https://doi.org/10.1103/PhysRevD.25.2649 ; Kaplan, domain walls: https://arxiv.org/abs/hep-lat/9206013 ; Luscher, exact abelian lattice chiral gauge theory: https://arxiv.org/abs/hep-lat/9811032
12. Weinberg, *Photons and Gravitons in S-Matrix Theory*: https://doi.org/10.1103/PhysRev.135.B1049
13. Weinberg–Witten, *Limits on Massless Particles*: https://doi.org/10.1016/0370-2693(80)90212-9
14. Sakharov, induced gravity: https://doi.org/10.1070/PU1991v034n05ABEH002498 ; Visser, technical review: https://arxiv.org/html/gr-qc/0204062

Source lookup used direct publication/arXiv records because the session's web-search service was unavailable. Literature establishes the cited mathematical tools and constraints, not the proposed BPR-to-bulk identifications.
