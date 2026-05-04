"""
IMP runner for the lottery-ticket variance-slope test
=======================================================

Trains a small CNN under iterative magnitude pruning (IMP) and emits a
CSV in the format expected by `experiments/lottery_ticket_variance.py`.

Three condition arms are supported and run as separate sweeps:

    winning_ticket    - standard IMP: train, prune by magnitude, reset
                        surviving weights to original init, retrain.
    random_mask       - matched-sparsity control: at each round drop the
                        same fraction at random; reset to original init.
    rerandomized_init - same IMP-derived mask as winning_ticket, but
                        reset surviving weights to a fresh random init
                        each round. The "lottery destroyer" baseline.

A real HIT in the slope test requires the winning_ticket arm to show
BPR-shaped variance collapse AND both baselines to *not* show it.

Defaults are sized for a fast first run on CPU/MNIST. For the real
test crank --dataset cifar10 --seeds 30 --rounds 12 --epochs 10 and
run on a GPU.

Usage
-----
    # Fast smoke run (MNIST, ~10–30 min on CPU):
    python experiments/lottery_ticket_imp_runner.py

    # Real run (CIFAR-10, GPU recommended):
    python experiments/lottery_ticket_imp_runner.py \\
        --dataset cifar10 --seeds 30 --rounds 12 --epochs 10 \\
        --out data/imp_cifar10.csv

    # Then feed CSV to the slope test:
    python experiments/lottery_ticket_variance.py data/imp_cifar10.csv
    python experiments/lottery_ticket_variance.py data/imp_cifar10.csv --metric loss
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SmallCNN(nn.Module):
    """Conv-Conv-FC, ~50K params on MNIST, ~80K on CIFAR-10."""

    def __init__(self, in_ch: int = 3, im_size: int = 32, n_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        feat = 32 * (im_size // 4) ** 2
        self.fc1 = nn.Linear(feat, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Mask handling
# ---------------------------------------------------------------------------

def prunable(model: nn.Module) -> List[Tuple[str, torch.Tensor]]:
    return [(n, p) for n, p in model.named_parameters()
            if p.requires_grad and p.dim() > 1]


def init_masks(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {n: torch.ones_like(p) for n, p in prunable(model)}


def apply_masks(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for n, p in prunable(model):
            p.mul_(masks[n])


def density(masks: Dict[str, torch.Tensor]) -> float:
    total = sum(m.numel() for m in masks.values())
    kept = sum(m.sum().item() for m in masks.values())
    return kept / total


def magnitude_prune(model: nn.Module, masks: Dict[str, torch.Tensor],
                    rate: float) -> Dict[str, torch.Tensor]:
    """Global magnitude prune: drop bottom `rate` of currently-active weights."""
    active_mags = []
    for n, p in prunable(model):
        active_mags.append(p.data.abs()[masks[n] > 0].flatten())
    flat = torch.cat(active_mags)
    n_drop = int(rate * flat.numel())
    if n_drop == 0:
        return {n: m.clone() for n, m in masks.items()}
    threshold = torch.kthvalue(flat, n_drop).values.item()
    new = {}
    for n, p in prunable(model):
        keep = (p.data.abs() > threshold) & (masks[n] > 0)
        new[n] = keep.to(masks[n].dtype)
    return new


def random_prune(masks: Dict[str, torch.Tensor], rate: float,
                 generator: torch.Generator) -> Dict[str, torch.Tensor]:
    """Drop `rate` of currently-active weights uniformly at random (global)."""
    sizes = [(n, m.numel()) for n, m in masks.items()]
    flat_active = torch.cat([m.flatten() for _, m in masks.items()])
    active_idx = flat_active.nonzero(as_tuple=True)[0]
    n_drop = int(rate * active_idx.numel())
    if n_drop == 0:
        return {n: m.clone() for n, m in masks.items()}
    perm = active_idx[torch.randperm(active_idx.numel(), generator=generator)]
    drop = perm[:n_drop]
    new_flat = flat_active.clone()
    new_flat[drop] = 0.0
    new = {}
    offset = 0
    for n, sz in sizes:
        new[n] = new_flat[offset:offset + sz].view_as(masks[n])
        offset += sz
    return new


# ---------------------------------------------------------------------------
# Init handling
# ---------------------------------------------------------------------------

def clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def reinit_weights(model: nn.Module) -> None:
    """Re-randomize prunable weights using the layers' default init."""
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.reset_parameters()


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_epoch(model, loader, opt, masks, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        for n, p in prunable(model):
            if p.grad is not None:
                p.grad.mul_(masks[n])
        opt.step()
        apply_masks(model, masks)


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    n, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()
    return correct / n, loss_sum / n


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_loaders(dataset: str, batch_size: int, root: str):
    if dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        tfm = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tfm)
        test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tfm)
        in_ch, im_size = 3, 32
    elif dataset == "mnist":
        tfm = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        train = torchvision.datasets.MNIST(root, train=True, download=True, transform=tfm)
        test = torchvision.datasets.MNIST(root, train=False, download=True, transform=tfm)
        in_ch, im_size = 1, 28
    else:
        raise SystemExit(f"unknown dataset {dataset!r}")
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(test, batch_size=512, shuffle=False, num_workers=0),
        in_ch, im_size,
    )


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_seed(seed: int, condition: str, args, device, writer) -> None:
    torch.manual_seed(seed)
    train_loader, test_loader, in_ch, im_size = make_loaders(
        args.dataset, args.batch_size, args.data_root,
    )
    model = SmallCNN(in_ch=in_ch, im_size=im_size).to(device)
    init_state = clone_state(model)
    masks = init_masks(model)
    rmg = torch.Generator(device="cpu").manual_seed(seed * 31 + 7)

    for k in range(args.rounds + 1):
        if k > 0:
            if condition == "rerandomized_init":
                reinit_weights(model)
            else:
                model.load_state_dict(init_state)
            apply_masks(model, masks)

        opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=1e-4)
        for _ in range(args.epochs):
            train_epoch(model, train_loader, opt, masks, device)
        acc, loss = evaluate(model, test_loader, device)
        d = density(masks)
        writer({
            "round": k, "seed": seed,
            "accuracy": acc, "loss": loss,
            "density": d, "condition": condition,
        })

        if condition == "random_mask":
            masks = random_prune(masks, args.prune_rate, rmg)
        else:
            masks = magnitude_prune(model, masks, args.prune_rate)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    ap.add_argument("--data-root", default="./data/torchvision")
    ap.add_argument("--out", default="data/imp_results.csv")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--prune-rate", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--conditions", nargs="+",
                    default=["winning_ticket", "random_mask", "rerandomized_init"],
                    choices=["winning_ticket", "random_mask", "rerandomized_init"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["round", "seed", "accuracy", "loss", "density", "condition"]
    device = torch.device(args.device)

    print(f"# device {device}, dataset {args.dataset}, "
          f"{args.seeds} seeds × {args.rounds + 1} rounds × {len(args.conditions)} conditions")
    print(f"# writing to {out}")

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        f.flush()

        def row(d):
            w.writerow(d)
            f.flush()
            print(f"  {d['condition']:<18s}  seed {d['seed']:3d}  round {d['round']:2d}  "
                  f"density {d['density']:.4f}  acc {d['accuracy']:.4f}  loss {d['loss']:.4f}",
                  flush=True)

        t0 = time.time()
        for cond in args.conditions:
            for seed in range(args.seeds):
                run_seed(seed, cond, args, device, row)
        print(f"\n# done in {time.time() - t0:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
