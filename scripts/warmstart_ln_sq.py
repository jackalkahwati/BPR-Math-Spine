"""
Warm-start regression experiment for [ln(x)]²
===============================================

Uses the exact verified LN_X_SQ tree (depth=11, K=29) from bpr/eml.py
as a scaffold to test whether gradient training can track the known
symbolic structure.

LN_X_SQ BFS layout at depth=11 (first_leaf=1023):
  pos  0: eml(A2, 1)           root
  pos  1: eml(1, B2)           A2
  pos  4: eml(C2, 1)           B2
  pos  9: eml(D2, LX2)         C2
  pos 19: eml(1, E2)           D2
  pos 20: eml(1, inner_LX2)    LX2   ← ln(x) subtree #2
  pos 40: eml(F2, 1)           E2
  pos 42: eml(eml(1,x), 1)     inner_LX2
  pos 81: eml(1, G2)           F2
  pos 85: eml(1, x)            eml(1,x) in LX2  [internal, 3-logit]
  pos164: eml(1, LX1)          G2
  pos330: eml(1, inner_LX1)    LX1   ← ln(x) subtree #1
  pos662: eml(eml(1,x), 1)     inner_LX1
 pos1325: eml(1, x)            eml(1,x) in LX1  [leaf, 2-logit]

Experiments
-----------
  1. EXACT  : init at LN_X_SQ + tiny noise → immediate snap + short train
  2. PARTIAL: LN_X subtrees warm+locked; x²-structure cold+trainable
  3. COLD   : standard bias-to-1 cold start (baseline)
  4. ABLATION: noise sweep — find snapping threshold

Run:
    python scripts/warmstart_ln_sq.py
"""

import sys, time
import numpy as np
import torch

sys.path.insert(0, "/home/user/BPR-Math-Spine")

from bpr.eml import EMLNode, Const, Var, LN_X_SQ
from bpr.eml_regression import EMLMasterFormula

# ── Config ────────────────────────────────────────────────────────────────────

DEPTH       = 11          # depth required by LN_X_SQ
SCALE       = 7.0         # logit magnitude for warm-start assignments
X_RANGE     = (1.1, 5.0)  # log-uniform; away from ln(x)=0 singularity
N_POINTS    = 32
LR_WARM     = 0.005       # small lr: we're testing stability, not convergence
LR_COLD     = 0.02        # normal lr for cold start
N_STEPS_WARM = 15         # short: warm init is already near 0 loss
N_STEPS_COLD = 25         # give cold start a fair but bounded budget
EXACT_THR   = 1e-6

# ── Data ─────────────────────────────────────────────────────────────────────

rng    = np.random.default_rng(42)
x_np   = np.exp(rng.uniform(np.log(X_RANGE[0]), np.log(X_RANGE[1]), N_POINTS))
y_np   = np.log(x_np) ** 2
x_t    = torch.tensor(x_np, dtype=torch.float64)
y_t    = torch.tensor(y_np, dtype=torch.float64)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _assign_node(sym, pos: int, params: list, first_leaf: int, scale: float):
    """Recursively set params[pos] so master-formula node at `pos` computes `sym`."""
    if not isinstance(sym, EMLNode) or pos >= len(params):
        return
    n_logits = 2 if pos >= first_leaf else 3
    with torch.no_grad():
        for row_idx, (child_sym, child_pos) in enumerate(
            [(sym.left, 2*pos+1), (sym.right, 2*pos+2)]
        ):
            logits = torch.full((n_logits,), -scale)
            if isinstance(child_sym, Const):
                logits[0] = scale
            elif isinstance(child_sym, Var):
                logits[1] = scale
            else:  # EMLNode — recurse into child
                if n_logits < 3:
                    raise ValueError(
                        f"Symbolic tree is deeper than master formula at pos={pos}"
                    )
                logits[2] = scale
                _assign_node(child_sym, child_pos, params, first_leaf, scale)
            params[pos].data[row_idx] = logits


def make_model_warm(sym_tree, scale: float = SCALE, noise: float = 0.01,
                    seed: int = 0) -> EMLMasterFormula:
    """Init all nodes to const(1) bias, then override with symbolic tree logits."""
    torch.manual_seed(seed)
    model = EMLMasterFormula(DEPTH)
    params = list(model.node_params)
    first_leaf = model._first_leaf
    # Step 1: all → strongly prefer const(1)
    with torch.no_grad():
        for p in params:
            p.zero_()
            p[:, 1:] = -scale
    # Step 2: override with symbolic assignments
    _assign_node(sym_tree, 0, params, first_leaf, scale)
    # Step 3: small noise so different restarts are distinguishable
    with torch.no_grad():
        for p in params:
            p.add_(torch.randn_like(p) * noise)
    return model


def make_model_cold(seed: int = 0) -> EMLMasterFormula:
    """Standard bias-to-1 cold start (identical to fit() default init)."""
    torch.manual_seed(seed)
    model = EMLMasterFormula(DEPTH)
    with torch.no_grad():
        for p in model.node_params:
            noise = torch.randn_like(p) * 0.1
            penalty = torch.zeros_like(p)
            penalty[:, 1:] = -5.0
            p.copy_(noise + penalty)
    return model


def snap_and_eval(model: EMLMasterFormula):
    """Snap model, compute snap_loss, check functional match to [ln(x)]²."""
    snapped = model.snap()
    with torch.no_grad():
        pred = snapped(x_t)
        snap_loss = float(((pred.real - y_t) ** 2).mean())
    expr = None
    try:
        expr = snapped.to_symbolic()
    except Exception:
        pass
    is_match = False
    if expr is not None:
        test_xs = np.array([1.5, 2.0, 3.0, np.e, np.pi, 4.5])
        try:
            errs = [abs(expr.eval(x=xi).real - np.log(xi)**2) for xi in test_xs]
            is_match = all(e < 1e-8 for e in errs)
        except Exception:
            pass
    return snap_loss, expr, is_match


def train(model: EMLMasterFormula, n_steps: int, lr: float,
          locked_positions: set | None = None) -> float:
    """Gradient descent; return final continuous MSE."""
    if locked_positions:
        for pos in locked_positions:
            model.node_params[pos].requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    last_loss = float("inf")
    for step in range(n_steps):
        optimizer.zero_grad()
        pred = model(x_t)
        loss = ((pred.real - y_t) ** 2).mean()
        if torch.isnan(loss) or torch.isinf(loss):
            return float("inf")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        last_loss = float(loss.item())
    return last_loss


def print_result(name: str, t0: float,
                 sl_init: float, match_init: bool,
                 cont_loss: float,
                 sl_final: float, match_final: bool,
                 expr_final):
    bar = "─" * 60
    print(f"\n  {bar}")
    print(f"  {name}")
    print(f"  {bar}")
    print(f"  snap@init : {sl_init:.3e}  {'MATCH' if match_init  else 'miss'}")
    print(f"  cont loss : {cont_loss:.3e}  (after {N_STEPS_WARM if 'COLD' not in name else N_STEPS_COLD} steps)")
    print(f"  snap@final: {sl_final:.3e}  {'MATCH ✓' if match_final else 'miss  ✗'}")
    print(f"  expr      : {repr(expr_final)[:120]}")
    print(f"  runtime   : {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 1 — EXACT WARM START (tiny noise)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("  WARM-START EXPERIMENT: [ln(x)]²  (depth=11, K=29)")
print(f"  n_params = {EMLMasterFormula(DEPTH).n_params:,}")
print("═"*60)

t0 = time.time()
model_exact = make_model_warm(LN_X_SQ, scale=SCALE, noise=0.01)
sl_init, expr_init, match_init = snap_and_eval(model_exact)
cont_loss = train(model_exact, N_STEPS_WARM, LR_WARM)
sl_final, expr_final, match_final = snap_and_eval(model_exact)
print_result("EXP 1: EXACT warm start (noise=0.01)", t0,
             sl_init, match_init, cont_loss, sl_final, match_final, expr_final)

# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 2 — PARTIAL WARM START (LN_X locked, x²-structure cold)
# ═══════════════════════════════════════════════════════════════════════════════
# LN_X BFS positions within LN_X_SQ at depth=11:
#   LX1 subtree: {330, 662, 1325}
#   LX2 subtree: {20, 42, 85}
LN_X_POSITIONS = {330, 662, 1325, 20, 42, 85}

t0 = time.time()
# Start from full warm init (gives correct logits everywhere),
# then re-init the outer x²-structure nodes to cold (bias-to-1).
model_partial = make_model_warm(LN_X_SQ, scale=SCALE, noise=0.01, seed=1)
OUTER_X2_POSITIONS = {0, 1, 4, 9, 19, 40, 81, 164}
with torch.no_grad():
    for pos in OUTER_X2_POSITIONS:
        p = model_partial.node_params[pos]
        noise = torch.randn_like(p) * 0.1
        penalty = torch.zeros_like(p)
        penalty[:, 1:] = -5.0
        p.copy_(noise + penalty)

sl_init_p, expr_init_p, match_init_p = snap_and_eval(model_partial)
cont_loss_p = train(model_partial, N_STEPS_WARM, LR_WARM,
                    locked_positions=LN_X_POSITIONS)
sl_final_p, expr_final_p, match_final_p = snap_and_eval(model_partial)
print_result("EXP 2: PARTIAL warm start (LN_X locked, x²-structure cold)", t0,
             sl_init_p, match_init_p, cont_loss_p,
             sl_final_p, match_final_p, expr_final_p)

# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 3 — COLD START BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

t0 = time.time()
model_cold = make_model_cold(seed=0)
sl_init_c, expr_init_c, match_init_c = snap_and_eval(model_cold)
cont_loss_c = train(model_cold, N_STEPS_COLD, LR_COLD)
sl_final_c, expr_final_c, match_final_c = snap_and_eval(model_cold)
print_result("EXP 3: COLD start baseline", t0,
             sl_init_c, match_init_c, cont_loss_c,
             sl_final_c, match_final_c, expr_final_c)

# ═══════════════════════════════════════════════════════════════════════════════
#  ABLATION — NOISE SWEEP (warm init only; no training)
#  Find the noise level at which the snap leaves the correct basin.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n  " + "─"*60)
print("  ABLATION: noise sweep — snap immediately after warm init")
print(f"  {'noise':>8}  {'snap_loss':>12}  {'match':>6}")
print("  " + "─"*34)

noise_levels = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
for noise in noise_levels:
    m = make_model_warm(LN_X_SQ, scale=SCALE, noise=noise, seed=7)
    sl, _, match = snap_and_eval(m)
    print(f"  {noise:>8.3f}  {sl:>12.3e}  {'MATCH' if match else 'miss'}")

# ═══════════════════════════════════════════════════════════════════════════════
#  ABLATION — UNLOCK ONE SUBTREE AT A TIME (only if EXP 1 matched)
# ═══════════════════════════════════════════════════════════════════════════════

if match_final:
    print("\n  " + "─"*60)
    print("  ABLATION: unlock one ln(x) subtree at a time from warm init")
    print(f"  (base: exact warm init + {N_STEPS_WARM} training steps)")
    print(f"  {'unlocked':>20}  {'snap_loss':>12}  {'match':>6}")
    print("  " + "─"*44)

    subtrees = {
        "LX1 ({330,662,1325})": {330, 662, 1325},
        "LX2 ({20,42,85})":     {20, 42, 85},
        "outer-x² ({0,1,4,...})": OUTER_X2_POSITIONS,
        "all unlocked":         set(),
    }

    for label, unlock_set in subtrees.items():
        locked = LN_X_POSITIONS - unlock_set
        m = make_model_warm(LN_X_SQ, scale=SCALE, noise=0.01, seed=2)
        train(m, N_STEPS_WARM, LR_WARM, locked_positions=locked if locked else None)
        sl_a, _, match_a = snap_and_eval(m)
        print(f"  {label:>20}  {sl_a:>12.3e}  {'MATCH' if match_a else 'miss'}")

# ═══════════════════════════════════════════════════════════════════════════════
#  CONCLUSION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("  SUMMARY")
print("═"*60)
print(f"  EXP 1 exact warm-start    snap@init={sl_init:.2e}  snap@final={sl_final:.2e}  {'MATCH' if match_final  else 'miss'}")
print(f"  EXP 2 partial warm-start  snap@init={sl_init_p:.2e}  snap@final={sl_final_p:.2e}  {'MATCH' if match_final_p else 'miss'}")
print(f"  EXP 3 cold start          snap@init={sl_init_c:.2e}  snap@final={sl_final_c:.2e}  {'MATCH' if match_final_c else 'miss'}")
print()
if match_final and sl_init < EXACT_THR:
    print("  Basin: STABLE.  Warm init places us exactly in the correct snap basin.")
    print("  Gradient descent maintains the tree structure.")
elif match_final and sl_init >= EXACT_THR:
    print("  Basin: gradient CORRECTED the snap (snap@init missed, snap@final matched).")
elif not match_final and sl_init < EXACT_THR:
    print("  Basin: FRAGILE.  Correct snap at init but training DRIFTED away.")
else:
    print("  Basin: INCORRECT.  Neither init nor training reached the correct snap.")

if match_final_p:
    print("  Composition: LN_X scaffold is SUFFICIENT — gradient found x² structure.")
else:
    print("  Composition: LN_X scaffold is NOT sufficient — x² structure not findable by gradient.")

if not match_final_c:
    print("  Cold start: FAILS as expected — confirms composition requires scaffold or enumeration.")
else:
    print("  Cold start: surprisingly SUCCEEDED — LN_X_SQ reachable without warm start!")
print()
