# Full-model neutral response in a joint strong-coupling/large-volume limit

2026-09-12. Module 2. Frozen derivation before implementation. Keep H=gD+CT at N=L=q with the complete occupation basis. The target is the exact normalized density-source spectral probability in s=(E_n-E_0(L)-g_L)/C_L, at nonzero lattice momentum. Explicitly, mu_full=sum_n |<n|rho_k|G>|²/||rho_k G||² times the point mass at s_n; every exact eigenstate is included. The sufficient source gate below makes the denominator nonzero. Here g_L and C_L denote L-indexed coupling sequences, NOT products g*L and C*L; at a fixed L the same expression is (E_n-E_0-g)/C. This is not a fixed-coupling thermodynamic theorem or a relativistic particle construction.

## Integer-D averaging

Let Td be the pinching of T onto equal-D blocks, To=T-Td. For integer D, Td=(1/(2pi)) integral_0^(2pi) exp(itD)T exp(-itD) dt. Thus ||Td||<=||T||<=v=2L and ||To||<=2v. Integer D and the norm2L bound are exact facts of each finite ring, not assumptions that its exponentially large matrix has been numerically diagonalized. The zero-diagonal inverse commutator S1_ab=T_ab/(Da-Db) for unequal D has an integral representation with kernel i(t-pi): its Fourier coefficient against exp(int) is 1/n for nonzero integer n. Therefore ||S1||<=pi*v/2<=pi L. S1 is anti-Hermitian and [S1,D]=-To.

Set lambda=C/g, U=exp(lambda S1). Instead of bounding ||D||, integrate its exact commutator:

U D U†-D = -lambda integral_0^1 exp(t lambda S1) To exp(-t lambda S1) dt.

Compare with Hbd=gD+CTd. Each unitary conjugation obeys ||exp(t lambda S1) A exp(-t lambda S1)-A||<=2t lambda||S1||||A||. Splitting the D-integral and hopping difference gives

||UHU†-Hbd||/C <= lambda||S1||||To||+2lambda||S1||||T|| <=4lambda||S1||v <=8pi lambda L² = B_H.

All these identities hold for each finite complete occupation space, however large. The scalar bounds require no Fock allocation.

## Density source, ground centering and characteristic function

Use the inherited unique ground G and choose its phase with positive Omega overlap. Define w=-lambda T Omega, x=rho_k G, y=rho_k w, a=||y||=4lambda|sin(k/2)|>0. For lambda<1/(4L), inherit the ground-source bound:

v_energy=2CL, b_energy=2C sqrt(L), delta_energy=g-4CL,
r=b_energy/delta_energy,
eta=(r²/2)*(b_energy/g)+2*v_energy*b_energy/(g*delta_energy),
delta_source=sqrt(L)*eta.

Equivalently with C=1 and g=1/lambda, r=2lambda sqrt(L)/(1-4lambda L), eta=r² lambda sqrt(L)+8lambda² L sqrt(L)/(1-4lambda L). Then delta_source is independent of common energy scale. If delta_source<a, x cannot vanish and

||Ux/||x||-y/a|| <=2delta_source/a+lambda||S1||.

This uses the normalization inequality ||x/||x||-y/||y||||<=2||x-y||/||y|| and ||U-I||<=lambda||S1||. The exact ground energy satisfies E0<=0 by the Omega trial. Ground-complement elimination and the inherited separation imply

|E0|/C <=4lambda L/(1-4lambda L) = B_0.

This is deliberately conservative. With Q=I-|Omega><Omega|, QHQ>=g-v_energy>0. Schur elimination gives -E0=b†(QHQ-E0)^(-1)b<=b_energy²/(g-v_energy)<=b_energy²/(g-2v_energy). Division by C gives the displayed bound. Thus centering uses a bound on the actual full ground energy, not an assumed perturbative value.

Let chi_full(t) be the characteristic function of s on normalized x; sign exp(it s). Let chi_comp(t) be that of the exact leading P1 compression in (E-g)/C coordinates on normalized y. U transforms both operator and source. Since Hbd preserves P1 and y is in P1, Duhamel's unitary inequality and the normalized-vector inequality give

|chi_full(t)-chi_comp(t)| <= 4delta_source/a+2pi lambda L+|t|(B_H+B_0).

The trivial bound is 2; at t=0 both characteristic functions equal 1 exactly. No individual spectral weights or unbounded moments are controlled by this bound.

## Semicircle comparison and sufficient sequences

The existing compression theorem has a=sqrt(5+4cos(k)) (call this shape parameter A to distinguish source norm) and semicircle support [-2A,2A]. Its real Lipschitz bound is K[32A/(3L)+2d_circle(k,k_target)]. Apply it separately to real and imaginary parts of exp(it s), each |t|-Lipschitz, giving safe complex bound

B_comp=2|t|[32A/(3L)+2d_circle(k,k_target)].

Thus total characteristic-function error <=min(2,4delta_source/a+2pi lambda L+|t|(B_H+B_0)+B_comp), under the normalization and separation gates. No claim of a certified floating-point error enclosure is made.

When lambda L is small, delta_source=O(lambda²L²). At a fixed limiting nonzero k, delta_source/a=O(lambda L²). At fixed nonzero integer m, |sin(pi m/L)| is order1/L, so the ratio is O(lambda L³). Consequently sufficient choices are lambda=L^(-2-epsilon) for fixed nonzero limiting k and lambda=L^(-3-epsilon) for fixed integer m, epsilon>0. The bound tends to zero uniformly on compact t intervals. The semicircle characteristic function is continuous at zero, so the Levy continuity theorem gives weak convergence of the full normalized probability measures. Fixed C/g is not covered; g merely proportional to L is insufficient for these estimates. These are sufficient, not necessary, conditions.

The normalized limiting measure has no atom. This does not exclude finite-g particles at fixed coupling, and does not derive uniform moment convergence. The absolute weight tends to zero in these joint strong-coupling sequences: ||x|| lies between a-delta_source and a+delta_source, both tending to zero.

## Spatial scaling is a different limit

Declare lattice spacing h, physical circumference ell=Lh, and C(h)=kappa/h² with kappa>0 an arbitrary scale coefficient, not a measured speed. For k=h p, the continuum compression support edge Eedge=g-2C sqrt(5+4cos(hp)) obeys

Eedge-(g-6C)=(2kappa/3)p²+O(kappa h² p^4).

An explicit error bound is |Eedge-(g-6C)-(2C/3)k²| <= (5/36)C k⁴ for |k|<=pi. To derive it put u=8sin²(k/2), A=sqrt(9-u)>=1. The identity 2(3-A)=u/3+u²/[3(3+A)²], together with |u-2k²|<=k⁴/6 and u<=2k², bounds the two errors by k⁴/18 and k⁴/12. The threshold shift can be evaluated without subtracting large energies as 16C sin²(k/2)/(3+A).

This is a support-edge shift, not a pole dispersion or the finite-chain minimum. The latter is g-2CA cos(pi/L), exceeding the support edge by 4CA sin²(pi/(2L)). At fixed ell, C=kappa L²/ell² and fixed m, this offset tends to 3kappa*pi²/ell² rather than zero. A density selection rule can also make the lowest eigenline dark, so this minimum is not automatically the first bright line. At fixed ell and integer m, p=2pi m/ell and k=2pi m/L. To make the coarse full/compression physical-generator error vanish in energy units requires C*lambda L² ->0, hence lambda L^4 ->0 when h=ell/L. Choosing lambda=L^(-5) is one explicit sufficient scaling for this generator error and the inherited source bound. This does not prove convergence of the near-edge spectral window: the normalized full response in centered bandwidth units tends to a semicircle, whereas the physical near-edge window has shrinking probability, so a further local/rescaled-source estimate would be required before conditioning on it. No common limiting speed or Lorentz covariance follows from a quadratic threshold, and no Planck spacing is inserted.

## Probability limit versus physical spectral window

The limiting characteristic function is that of the centered semicircle, namely J1(2A t)/(A t) for nonzero t, with continuous value 1 at t=0. Numerical verification can instead integrate (2/pi) sin²(u) exp(-2i A t cos(u)) over [0,pi), avoiding division at zero. The theorem compares the complete normalized exact source, including every D band; it does not discard weak higher-band tails before normalization. Weak convergence can coexist with nonconvergent high moments due to tiny spectral weight at very high energy.

If g grows with L, the uncentered excitation energies move to infinity. This statement concerns the explicit centered coordinates only. A near-threshold energy scaling would require a new normalization and cannot be inferred by changing the axis label of the bandwidth-scale probability theorem.

## Frozen controls and implementation gates

Scalar demos use L=16,64,256; fixed k=pi/2 via m=L/4 with lambda=L^-3; fixed m=1 with lambda=L^-4; compare against target k=pi/2 or0, at t=0,.5,1. Retain frozen original L=5,C=1,g=40,m=1 (expected failed relative-source gate) and g=.7, exact zero mode, and g=0 unavailable controls. Small complete-oracle tests L=3,4,5 use g=40,4000 with no adjustment after evaluation. Analytic arrays inherit the old continuum cap where used, while scalar bounds do not allocate full Fock spaces. Validate arbitrary momentum aliases, preallocation caps, nonfinite/subnormal values and loss of a nonzero mode; reject unrepresentable arithmetic rather than assign physical meaning to zero.

Acceptance: independent integral-kernel sign/norm, first-order unitary generator bound, exact-ground energy bound, normalized source comparison and exact characteristic functions against scalar bound; semicircle quadrature independent of the finite-line formula; fixed-k/fixed-m sequences, structural controls and symbolic threshold scaling. The sufficient full-model theorem passed independent mathematical review: the reviewer checked the integral kernel, generator/source/ground bounds, characteristic-function comparison, Levy sequences and quartic constant, and reproduced finite-matrix inequalities using independent complete-Fock matrices at L=3,4,5 and g=40,4000. Review required the explicit distinction above between continuum support edge and finite-chain minimum. The first focused executable run passed 66 tests with warnings as errors, including text/strict-JSON stdout demos; an expanded run passed 72 and the subsequent complete focused suite passed 73. The first bounded substrate regression passed 782 tests with warnings as errors. Independent implementation review found an extreme-coupling oracle defect: at L=3,g=1e18,t=1e-6 the eigensolver source was dominated by roundoff, yet the existing gap/phase screens allowed normalization and an order-one erroneous characteristic function. The scalar theorem was unaffected. The oracle now uses an explicit scaled spectral proxy 64*eps*dimension*maximum absolute row sum. It requires a factor 1e6 margin for the hopping bandwidth and source norm, and 1e3 for the reported ground-energy magnitude. The source proxy includes sqrt(L)/ground_gap. These numerical acceptance margins are declared, conservative heuristics, not physical parameters; no sign is forced and no comparison is made against the analytic truncation remainder. Regression tests require g=1e18 to be rejected at t=0,1e-6,1 while the scalar bound remains available. The post-fix focused suite passed 78 tests with warnings as errors. Independent implementation rechecking confirmed 24 unresolved extreme-coupling cases now reject and 84 ordinary oracle controls remain usable, with no remaining verified defects. The reviewer separately constructed full occupation matrices for 54 characteristic-function comparisons (maximum discrepancy 2.83e-12) and checked compression phases for all nonzero modes on L=3..6 (discrepancies below 8e-16). The combined bounded regression, launched before this oracle fix and five final test additions, passed 1,152 tests with one intentional phase-anchor deselection and one existing unregistered slow-marker warning. The final 78-test focused run covers the repaired code and all added regressions; no full-repository test claim is made. Such screens are heuristic safeguards, not rigorous roundoff enclosures or tests that numerical residuals must lie below analytic truncation bounds. Existing module1 remains committed/pushed in PR24 but merge was denied; this module is on the documented dependent branch science/neutral-joint-limit.

## Reproduction and scientific scope

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m pytest -q -W error tests/test_substrate_neutral_scaling.py
python3 scripts/demo_substrate_neutral_scaling.py
python3 scripts/demo_substrate_neutral_scaling.py --json
```

The scalar comparison and spatial audit allocate no occupation space. The optional exact oracle is restricted to L=3..6 before allocation, retains the full basis, and reports numerical diagnostics separately from analytic bounds. All demo output is stdout only. The independent full-model comparison is new; the compression semicircle theorem is reused without changing its semantics.

This result closes a specific mathematical joint-limit gap for the stipulated quantum model, not the fixed-coupling thermodynamic problem. It does not supply observed particles, physical fermions, a prepared vacuum, spacetime dynamics or empirical support for a theory of everything. No weak-case failure was repaired by changing the frozen inputs. The frozen original L=5,C=1,g=40,m=1 has delta_source/a approximately 4.35958539 and therefore no normalized-source certificate. At t=1,L=256, the full/semicircle total bounds are approximately 0.328867554 for fixed k and 0.358489582 for fixed m. Default spatial physical-generator bounds at L=16,64,256 are approximately 1.57079633, 0.392699082 and 0.0981747704. These are evaluated analytic bounds, not measured deviations or physical calibration.

All eleven scientific demos passed text and strict JSON from an empty temporary directory with no stderr or output files. All eight independent algebra-check groups passed. Python 3.8 grammar and compilation checks passed.
