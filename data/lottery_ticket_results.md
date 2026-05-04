# Lottery Ticket × BPR — Cheap-Test Result Log

This directory records the actual run output of the BPR-LTH variance-slope
falsification test described in `experiments/lottery_ticket_variance.py`
and `experiments/lottery_ticket_imp_runner.py`.

## Run 1 — MNIST + SmallCNN smoke test (2026-05-04)

**Configuration**
- Dataset: MNIST
- Model: SmallCNN (~50K params; Conv-Conv-FC)
- Seeds: 8
- Pruning rounds: 6 (= 7 evaluations including round 0)
- Epochs per round: 1
- Optimizer: SGD lr=0.01, momentum=0.9, weight decay 1e-4
- Batch size: 128
- Conditions: winning_ticket, random_mask, rerandomized_init
- Wall clock: ~25 min on 4-core CPU (3 conditions in parallel)

**Raw CSV:** `imp_mnist.csv` (168 rows; force-committed because `data/*.csv` is
in `.gitignore`).

**Slope-test verdict**

```
ACCURACY                    slope ± stderr     R²    verdict
  winning_ticket:           -0.39  ± 0.48     0.14   WEAK
  random_mask:              +0.86  ± 0.33     0.63   FALSIFY (variance grows)
  rerandomized_init:        +0.03  ± 0.41     0.00   FALSIFY (flat)
  → CONTRAST FALSIFY

LOSS                        slope ± stderr     R²    verdict
  winning_ticket:           -0.33  ± 0.42     0.13   WEAK
  random_mask:              +0.90  ± 0.34     0.63   FALSIFY (variance grows)
  rerandomized_init:        +0.48  ± 0.51     0.18   FALSIFY (flat-ish)
  → CONTRAST FALSIFY
```

**Read of the result**

The cheap test does *not* produce a BPR-shaped variance collapse on
this configuration. The winning-ticket arm has a negative slope in
both metrics (directionally consistent with BPR), but the magnitude
(~ -0.4) is far from the predicted -2 and the error bars overlap
zero.

The contrast against baselines *is* partly visible (winner goes
negative; baselines do not), but not in a way that earns a HIT.

**Known limitations of this run**

These limitations would individually disqualify this as the final
falsification, and collectively they mean a negative result here is
not strong evidence against the BPR-LTH analogy:

1. *Ceiling-compressed metrics.* Test accuracy is 96-98% across every
   condition and every round; loss is similarly saturated at 0.07-0.12.
   Almost no dynamic range to express variance.
2. *Short lever arm.* Only 6 log-spaced points per fit. The winner's
   R² = 0.14 indicates the data does not constrain the slope tightly.
3. *Tiny architecture.* SmallCNN is 50K parameters. LTH effects are
   most pronounced in over-parameterized networks (ResNet-20 is ~270K
   params on CIFAR-10).
4. *Short training.* 1 epoch per round may not let the differential
   stability of winning-ticket weights manifest.
5. *Few seeds.* 8 seeds gives wide variance estimates; the original
   plan called for 30+.

**What would change the verdict**

The faithful version of the test:

```
python experiments/lottery_ticket_imp_runner.py \
    --dataset cifar10 --seeds 30 --rounds 12 --epochs 10 \
    --out data/imp_cifar10.csv
python experiments/lottery_ticket_variance.py data/imp_cifar10.csv
python experiments/lottery_ticket_variance.py data/imp_cifar10.csv --metric loss
```

A faithful run requires GPU access. A negative result there would
genuinely retire the BPR-LTH framing; a positive result would justify
moving to the money test (`bpr/stability_score.py` plus AUC-against-
SNIP/GraSP/SynFlow benchmarking).

## Provenance

Branch: `claude/analyze-lottery-ticket-lKCYK`
Scripts: see commit log of `experiments/lottery_ticket_*.py`.
