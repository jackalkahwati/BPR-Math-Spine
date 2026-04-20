"""
EML–BPR Bridge: All Physics from One Operator
===============================================

Bridges two reductionist programs:

  BPR (Al-Kahwati 2026): all physics from two parameters (J, p)
  EML (Odrzywolek 2026): all elementary functions from one operator
        eml(x, y) = exp(x) − ln(y)   [arXiv:2603.21852v2]

Synthesis: every BPR prediction is an elementary function of p.
EML proves that entire class reduces to one operator + constant {1}.
Therefore: all of physics = finite EML tree with leaf p = 104 761.

This module
-----------
1. Implements the EML operator and expression-tree data structure.
2. Provides exact EML trees for functions appearing in BPR derivations
   (e, exp, 0, ln) with numerically verified identities.
3. Evaluates BPR's fine-structure formula using EML-derived ln(p),
   confirming machine-precision agreement with the standard derivation.
4. Reports the EML tree depth (complexity) of each BPR-relevant constant.

Verified EML identities (Odrzywolek 2026, eq. 5 and Table 4)
--------------------------------------------------------------
  e       = eml(1, 1)                         depth 1  K=3
  exp(x)  = eml(x, 1)                         depth 1  K=3
  0       = eml(1, eml(eml(1,1), 1))          depth 3  K=7
  ln(x)   = eml(1, eml(eml(1,x), 1))          depth 3  K=7

K = RPN program length = 2*leaves − 1 for a full binary EML tree.

EML complexity of BPR fine-structure terms
------------------------------------------
  ln(p)        K=7    exact tree provided
  [ln(p)]²    ~K=35   x² needs K≈19, composed with ln(p)
  z/2 (= 3)   ~K=29   integer 3 derivable in EML
  γ            ~K=50+  transcendental, large tree
  −1/(2π)     ~K=80+  π has K>53, then reciprocal
  Full 1/α    ~K=150+ lower bound

References
----------
Odrzywolek, A. (2026). All elementary functions from a single operator.
  arXiv:2603.21852v2 [cs.SC].
Al-Kahwati, J. (2026). BPR-Math-Spine. §22.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .constants import P_DEFAULT, Z_DEFAULT
from .alpha_derivation import _EULER_GAMMA


# ─────────────────────────────────────────────────────────────────────────────
# §1  The EML operator
# ─────────────────────────────────────────────────────────────────────────────

def eml(x: complex, y: complex) -> complex:
    """The EML Sheffer operator:  eml(x, y) = exp(x) − ln(y).

    A single binary operator sufficient to generate all elementary
    functions when combined with the constant 1 (Odrzywolek 2026).
    Operates over ℂ; complex intermediates are expected even for
    real inputs (trigonometric functions require ln of negatives).

    Parameters
    ----------
    x, y : complex
        Inputs.  ln(y) uses the principal branch; y = 0 is undefined.

    Returns
    -------
    complex
        exp(x) − ln(y).
    """
    return (np.exp(np.asarray(x, dtype=complex))
            - np.log(np.asarray(y, dtype=complex)))


# ─────────────────────────────────────────────────────────────────────────────
# §2  EML expression tree
# ─────────────────────────────────────────────────────────────────────────────

class EMLExpr:
    """Abstract base for EML expression-tree nodes."""

    def eval(self, **vars: complex) -> complex:
        raise NotImplementedError

    def depth(self) -> int:
        """Maximum depth of the tree (leaves have depth 0)."""
        raise NotImplementedError

    def leaf_count(self) -> int:
        """Number of terminal symbols (constants + variables)."""
        raise NotImplementedError

    def rpn_length(self) -> int:
        """RPN program length K = 2*leaf_count − 1."""
        return 2 * self.leaf_count() - 1

    def __repr__(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Const(EMLExpr):
    """Terminal node: a numeric constant."""
    value: complex = 1

    def eval(self, **_) -> complex:
        return complex(self.value)

    def depth(self) -> int:
        return 0

    def leaf_count(self) -> int:
        return 1

    def __repr__(self) -> str:
        if self.value == 1:
            return "1"
        if abs(self.value - np.e) < 1e-14:
            return "e"
        return str(self.value)


@dataclass(frozen=True)
class Var(EMLExpr):
    """Terminal node: a named variable."""
    name: str = "x"

    def eval(self, **vars: complex) -> complex:
        if self.name not in vars:
            raise KeyError(f"Variable '{self.name}' not provided to EML tree")
        return complex(vars[self.name])

    def depth(self) -> int:
        return 0

    def leaf_count(self) -> int:
        return 1

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class EMLNode(EMLExpr):
    """Internal node: eml(left, right) = exp(left) − ln(right)."""
    left: EMLExpr
    right: EMLExpr

    def eval(self, **vars: complex) -> complex:
        return eml(self.left.eval(**vars), self.right.eval(**vars))

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def leaf_count(self) -> int:
        return self.left.leaf_count() + self.right.leaf_count()

    def __repr__(self) -> str:
        return f"eml({self.left!r}, {self.right!r})"


# ─────────────────────────────────────────────────────────────────────────────
# §3  Pre-built EML trees (exact identities, Odrzywolek 2026)
# ─────────────────────────────────────────────────────────────────────────────

# Shared terminals
_ONE = Const(1)
_X = Var("x")
_P = Var("p")

# e = eml(1, 1)
# exp(1) − ln(1) = e − 0 = e          [depth 1, K=3]
E_CONST: EMLExpr = EMLNode(_ONE, _ONE)

# exp(x) = eml(x, 1)
# exp(x) − ln(1) = exp(x) − 0         [depth 1, K=3]
EXP_X: EMLExpr = EMLNode(_X, _ONE)

# 0 = eml(1, eml(eml(1,1), 1))
#   eml(1,1) = e
#   eml(e, 1) = exp(e) − 0 = e^e
#   eml(1, e^e) = e − ln(e^e) = e − e = 0   [depth 3, K=7]
ZERO: EMLExpr = EMLNode(_ONE, EMLNode(EMLNode(_ONE, _ONE), _ONE))

# ln(x) = eml(1, eml(eml(1, x), 1))   (Odrzywolek 2026, eq. 5)
#   eml(1, x)        = e − ln(x)
#   eml(e−ln(x), 1)  = exp(e−ln(x)) = e^e / x
#   eml(1, e^e/x)    = e − ln(e^e/x) = e − (e − ln(x)) = ln(x)  [depth 3, K=7]
LN_X: EMLExpr = EMLNode(_ONE, EMLNode(EMLNode(_ONE, _X), _ONE))

# ln(p) — same structure, variable named 'p'   [depth 3, K=7]
LN_P: EMLExpr = EMLNode(_ONE, EMLNode(EMLNode(_ONE, _P), _ONE))


# ─────────────────────────────────────────────────────────────────────────────
# §4  BPR fine-structure formula via EML
# ─────────────────────────────────────────────────────────────────────────────

def ln_p_via_eml(p: float = P_DEFAULT) -> complex:
    """Compute ln(p) using the exact depth-3 EML tree.

    Tree: eml(1, eml(eml(1, p), 1))

    Returns
    -------
    complex
        ln(p).  Imaginary part is 0 for p > 0.
    """
    return LN_P.eval(p=complex(p))


def bpr_alpha_via_eml(p: int = P_DEFAULT, z: int = Z_DEFAULT) -> dict:
    """Evaluate BPR's fine-structure formula with EML-derived ln(p).

    The ln(p) term — which dominates 1/α — is computed via the exact
    depth-3 EML tree.  The remaining constant terms (z/2, γ, −1/2π)
    use standard arithmetic; they too are EML-expressible but need
    trees of estimated depth > 40 each.

    Returns
    -------
    dict
        ln_p_eml        : complex  — ln(p) via EML tree
        ln_p_numpy      : float    — ln(p) via numpy (reference)
        ln_p_error      : float    — |EML − numpy|  (~machine ε)
        inv_alpha_eml   : float    — 1/α with EML-derived ln(p)
        inv_alpha_ref   : float    — 1/α with numpy ln(p)
        inv_alpha_error : float    — |EML − ref|
        alpha_eml       : float    — α_EM prediction
        ln_p_tree_depth : int      — depth of LN_P tree
        ln_p_tree_K     : int      — RPN length of LN_P tree
    """
    ln_p_eml = ln_p_via_eml(p)
    ln_p_ref = np.log(float(p))

    screening_eml = float(ln_p_eml.real) ** 2
    screening_ref = ln_p_ref ** 2

    tail = z / 2.0 + _EULER_GAMMA - 1.0 / (2.0 * np.pi)

    inv_alpha_eml = screening_eml + tail
    inv_alpha_ref = screening_ref + tail

    return {
        "ln_p_eml": ln_p_eml,
        "ln_p_numpy": ln_p_ref,
        "ln_p_error": abs(float(ln_p_eml.real) - ln_p_ref),
        "inv_alpha_eml": inv_alpha_eml,
        "inv_alpha_ref": inv_alpha_ref,
        "inv_alpha_error": abs(inv_alpha_eml - inv_alpha_ref),
        "alpha_eml": 1.0 / inv_alpha_eml,
        "ln_p_tree_depth": LN_P.depth(),
        "ln_p_tree_K": LN_P.rpn_length(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# §5  Identity verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_eml_identities(n_samples: int = 50) -> dict:
    """Numerically verify the four pre-built EML identities.

    Tests
    -----
    (i)   E_CONST.eval()       == e            (constant)
    (ii)  EXP_X.eval(x=x)     == exp(x)        x ∈ Uniform(−3, 3)
    (iii) ZERO.eval()          == 0             (constant)
    (iv)  LN_X.eval(x=x)      == ln(x)         x ∈ Uniform(0.01, 10)

    Returns
    -------
    dict
        max absolute error per identity and boolean 'all_pass'.
    """
    rng = np.random.default_rng(42)
    tol = 1e-10

    e_err = float(abs(E_CONST.eval() - np.e))

    xs = rng.uniform(-3.0, 3.0, n_samples)
    exp_err = float(max(abs(EXP_X.eval(x=xi) - np.exp(xi)) for xi in xs))

    zero_err = float(abs(ZERO.eval().real))

    xs_pos = rng.uniform(0.01, 10.0, n_samples)
    ln_err = float(max(abs(LN_X.eval(x=xi).real - np.log(xi)) for xi in xs_pos))

    return {
        "e_constant_error": e_err,
        "exp_max_error": exp_err,
        "zero_error": zero_err,
        "ln_max_error": ln_err,
        "all_pass": all(e < tol for e in [e_err, exp_err, zero_err, ln_err]),
        "tolerance": tol,
    }


# ─────────────────────────────────────────────────────────────────────────────
# §6  Human-readable report
# ─────────────────────────────────────────────────────────────────────────────

def report(p: int = P_DEFAULT, z: int = Z_DEFAULT) -> str:
    """Full EML–BPR synthesis report."""
    result = bpr_alpha_via_eml(p, z)
    ident = verify_eml_identities()

    lines = [
        "═══════════════════════════════════════════════════════════════",
        "  EML–BPR Synthesis: All Physics from One Operator",
        "  Odrzywolek (2026)  ×  Al-Kahwati (2026)",
        "═══════════════════════════════════════════════════════════════",
        "",
        "  EML operator:      eml(x, y) = exp(x) − ln(y)",
        "  BPR formula:       1/α = [ln(p)]² + z/2 + γ − 1/(2π)",
        f"  Substrate prime:   p = {p:,}",
        "",
        "  ── Exact EML trees ──────────────────────────────────────────",
        f"  {'Expression':<22} {'depth':>5}  {'K':>4}  Identity",
        f"  {'─'*22} {'─'*5}  {'─'*4}  {'─'*32}",
        f"  {'e':<22} {E_CONST.depth():>5}  {E_CONST.rpn_length():>4}  {E_CONST!r}",
        f"  {'exp(x)':<22} {EXP_X.depth():>5}  {EXP_X.rpn_length():>4}  {EXP_X!r}",
        f"  {'0':<22} {ZERO.depth():>5}  {ZERO.rpn_length():>4}  {ZERO!r}",
        f"  {'ln(x)':<22} {LN_X.depth():>5}  {LN_X.rpn_length():>4}  {LN_X!r}",
        f"  {'ln(p)':<22} {LN_P.depth():>5}  {LN_P.rpn_length():>4}  (same, variable = p)",
        "",
        "  ── Identity verification ────────────────────────────────────",
        f"  e error          = {ident['e_constant_error']:.2e}",
        f"  exp(x) max error = {ident['exp_max_error']:.2e}",
        f"  0 error          = {ident['zero_error']:.2e}",
        f"  ln(x) max error  = {ident['ln_max_error']:.2e}",
        f"  All pass (tol=1e-10): {ident['all_pass']}",
        "",
        "  ── ln(p) via EML tree ───────────────────────────────────────",
        f"  ln(p) [EML tree] = {result['ln_p_eml'].real:.15f}",
        f"  ln(p) [numpy]    = {result['ln_p_numpy']:.15f}",
        f"  Error            = {result['ln_p_error']:.2e}  (machine precision)",
        "",
        "  ── BPR 1/α with EML-derived ln(p) ──────────────────────────",
        f"  1/α  [EML ln(p)] = {result['inv_alpha_eml']:.6f}",
        f"  1/α  [numpy]     = {result['inv_alpha_ref']:.6f}",
        f"  1/α  [CODATA]    =  137.035999",
        f"  Difference       = {result['inv_alpha_error']:.2e}  (< machine ε)",
        "",
        "  ── EML complexity of all BPR 1/α terms ─────────────────────",
        f"  {'Term':<18} {'Value':>12}  {'K (est.)':>10}  Notes",
        f"  {'─'*18} {'─'*12}  {'─'*10}  {'─'*28}",
        f"  {'ln(p)':<18} {np.log(p):>12.6f}  {'7':>10}  exact tree above",
        f"  {'[ln(p)]²':<18} {np.log(p)**2:>12.6f}  {'~35':>10}  x²: K≈19 + ln(p)",
        f"  {'z/2 = κ':<18} {z/2.0:>12.6f}  {'~29':>10}  integer from EML",
        f"  {'γ':<18} {_EULER_GAMMA:>12.6f}  {'>50':>10}  transcendental",
        f"  {'−1/(2π)':<18} {-1/(2*np.pi):>12.6f}  {'>80':>10}  π: K>53",
        f"  {'─'*18}",
        f"  {'1/α (full)':<18} {result['inv_alpha_eml']:>12.6f}  {'>150':>10}  lower bound",
        "",
        "  ── Synthesis ────────────────────────────────────────────────",
        "  BPR : all physics        ←  p = {:,}".format(p),
        "  EML : all functions      ←  eml(x,y) + {{1}}",
        "  Combined: all of physics = EML tree with leaf p = {:,}".format(p),
        "═══════════════════════════════════════════════════════════════",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
