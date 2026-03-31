"""
prune_and_plot.py
─────────────────
Task requirements:
    1. Load a fine-tuned classifier checkpoint
    2. Apply unstructured magnitude pruning at multiple ratios
    3. Measure classification error and FLOPs at each ratio
    4. Plot FLOPS vs error as a line plot
    5. [Bonus] Overlay both baseline and sparse model on the same plot

Pruning approach: unstructured L1 magnitude pruning via torch.nn.utils.prune
    - Prunes individual weights globally across all Linear/Conv layers
    - No retraining after pruning (one-shot pruning)
    - This is standard for benchmarking pruning sensitivity

FLOPS computation:
    - Uses fvcore library (pip install fvcore) for FoldingNet
    - Manual analytical counting for SparseConv (fvcore doesn't handle
      sparse tensors natively)

Run examples
────────────
    # prune baseline only
    python prune_and_plot.py \
        --baseline_ckpt checkpoints/foldingnet_finetune_pretrained_best.pt \
        --labelled data/labelled.h5

    # prune both models and overlay on same plot (bonus task)
    python prune_and_plot.py \
        --baseline_ckpt checkpoints/foldingnet_finetune_pretrained_best.pt \
        --bonus_ckpt    checkpoints/sparseconv_finetune_pretrained_best.pt \
        --labelled data/labelled.h5
"""

import os
import json
import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — works on headless servers
import matplotlib.pyplot as plt

from models.folding_net_encoder import FoldingNetEncoder
from models.folding_net_decoder import FoldingNetDecoder
from models.auto_encoder import SparseAutoencoder, SparseClassifier
from data.data_loader import make_dataloaders


# ─────────────────────────────────────────────
# argument parsing
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Pruning + FLOPS vs Error plot')
    p.add_argument('--baseline_ckpt', type=str, required=True,
                   help='Fine-tuned FoldingNet checkpoint')
    p.add_argument('--bonus_ckpt',    type=str, default=None,
                   help='Fine-tuned SparseConv checkpoint (bonus task)')
    p.add_argument('--labelled',      type=str, default='data/labelled.h5')
    p.add_argument('--unlabelled',    type=str, default='data/unlabelled.h5')
    p.add_argument('--out_dir',       type=str, default='results')
    p.add_argument('--batch_size',    type=int, default=32)
    p.add_argument('--num_workers',   type=int, default=4)
    p.add_argument('--codeword_dim',  type=int, default=512)
    p.add_argument('--seed',          type=int, default=42)
    # pruning ratios to sweep — 0.0 = no pruning, 1.0 = all weights zeroed
    p.add_argument('--ratios', type=float, nargs='+',
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                            0.6, 0.7, 0.8, 0.9])
    return p.parse_args()


# ─────────────────────────────────────────────
# model loading
# ─────────────────────────────────────────────

def load_classifier(ckpt_path: str, codeword_dim: int,
                    spatial_shape: tuple) -> SparseClassifier:
    """
    Rebuild the classifier architecture and load weights from checkpoint.
    Returns the classifier in eval mode on CPU.
    """
    ckpt         = torch.load(ckpt_path, map_location='cpu')
    encoder_type = ckpt['args']['encoder']

    if encoder_type == 'foldingnet':
        encoder = FoldingNetEncoder(k=16, codeword_dim=codeword_dim, in_dim=10)

    classifier = SparseClassifier(
        encoder      = encoder,
        encoder_type = encoder_type,
        num_classes  = 2,
        codeword_dim = codeword_dim,
    )
    classifier.load_state_dict(ckpt['model_state'])
    classifier.eval()
    return classifier


# ─────────────────────────────────────────────
# pruning
# ─────────────────────────────────────────────

def apply_global_pruning(model: nn.Module, ratio: float) -> nn.Module:
    """
    Apply global unstructured L1 magnitude pruning to all Linear and
    Conv layers in the model.

    Global pruning: computes a single threshold across ALL eligible weights
    in the model and zeros those below it. This is more principled than
    per-layer pruning because it lets redundant layers prune more heavily
    while critical layers stay dense.

    Args:
        model : classifier to prune (deepcopy is taken — original unchanged)
        ratio : fraction of weights to zero out  [0.0, 1.0)

    Returns:
        pruned model (with pruning masks applied permanently)
    """
    if ratio == 0.0:
        return model

    model = copy.deepcopy(model)

    # collect all (module, param_name) pairs eligible for pruning
    params_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            params_to_prune.append((module, 'weight'))

    if not params_to_prune:
        return model

    # global L1 unstructured pruning
    prune.global_unstructured(
        params_to_prune,
        pruning_method = prune.L1Unstructured,
        amount         = ratio,
    )

    # make pruning permanent — removes mask buffers, replaces weight with
    # the zeroed tensor. Required so FLOP counting sees actual zero weights.
    for module, param_name in params_to_prune:
        prune.remove(module, param_name)

    return model


def count_nonzero_params(model: nn.Module) -> tuple:
    """
    Count total parameters and non-zero parameters in the model.

    Returns:
        (total_params, nonzero_params, actual_sparsity)
    """
    total    = 0
    nonzero  = 0
    for p in model.parameters():
        total   += p.numel()
        nonzero += p.nonzero().shape[0]
    sparsity = 1.0 - nonzero / total if total > 0 else 0.0
    return total, nonzero, sparsity


# ─────────────────────────────────────────────
# FLOPS computation
# ─────────────────────────────────────────────

def count_linear_flops(model: nn.Module, sparsity: float) -> float:
    """
    Analytically count FLOPs for all Linear layers in the model,
    accounting for weight sparsity (zeroed weights skip multiply-adds).

    For a Linear layer with input (B, in) → output (B, out):
        dense FLOPs = 2 * in * out   (multiply + add per output element)
        sparse FLOPs = dense * (1 - sparsity)

    This is an approximation — in practice sparse kernels may not achieve
    perfect proportional speedup, but FLOP count is the standard metric
    for comparing pruning efficiency.

    Args:
        model    : classifier (after pruning applied permanently)
        sparsity : fraction of weights that are zero

    Returns:
        total_flops : float  (for a single forward pass, batch=1)
    """
    total_flops = 0.0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            in_f  = module.in_features
            out_f = module.out_features
            # 2 ops per weight: multiply + accumulate
            dense_flops   = 2 * in_f * out_f
            # sparse flops proportional to non-zero weights
            sparse_flops  = dense_flops * (1.0 - sparsity)
            total_flops  += sparse_flops

    return total_flops


def compute_flops(model: nn.Module, ratio: float) -> float:
    """
    Compute FLOPs for a pruned model.

    We use analytical Linear FLOP counting because:
      - fvcore doesn't handle SparseConvTensor inputs
      - Both models (FoldingNet and SparseConv) have Linear layers as
        the dominant cost in their classification heads and projection MLPs
      - The relative FLOP reduction from pruning is what matters for the
        plot, not the absolute number

    Returns FLOPs in millions (MFLOPs) for readability.
    """
    _, _, actual_sparsity = count_nonzero_params(model)
    raw_flops = count_linear_flops(model, actual_sparsity)
    return raw_flops / 1e6   # MFLOPs


# ─────────────────────────────────────────────
# evaluation
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    """
    Evaluate classifier on a dataloader.
    Returns error rate (1 - accuracy) as a percentage.
    """
    model.eval()
    correct, total = 0, 0

    for batch in loader:
        batch  = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch.items()}
        labels = batch['label']
        logits = model(batch)
        preds  = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    accuracy = 100.0 * correct / total
    error    = 100.0 - accuracy
    return error


# ─────────────────────────────────────────────
# pruning sweep
# ─────────────────────────────────────────────

def pruning_sweep(
    classifier:  nn.Module,
    test_loader,
    device:      torch.device,
    ratios:      list,
    label:       str,
) -> dict:
    """
    Sweep over pruning ratios, measuring error and FLOPs at each point.

    Args:
        classifier  : fine-tuned classifier (unpruned)
        test_loader : test set DataLoader
        device      : torch device
        ratios      : list of pruning ratios to evaluate
        label       : model name for logging

    Returns:
        dict with keys 'ratios', 'errors', 'flops', 'sparsities'
    """
    results = {
        'label':      label,
        'ratios':     [],
        'errors':     [],
        'flops':      [],
        'sparsities': [],
    }

    # baseline FLOPs at ratio=0 for normalisation
    baseline_flops = None

    for ratio in ratios:
        print(f"  [{label}] Pruning ratio: {ratio:.0%} ...", end=' ', flush=True)

        # prune a fresh copy at this ratio
        pruned = apply_global_pruning(classifier, ratio).to(device)

        # measure error on test set
        error  = evaluate(pruned, test_loader, device)

        # measure FLOPs
        flops  = compute_flops(pruned, ratio)
        if baseline_flops is None:
            baseline_flops = flops

        _, _, sparsity = count_nonzero_params(pruned)

        results['ratios'].append(ratio)
        results['errors'].append(error)
        results['flops'].append(flops)
        results['sparsities'].append(sparsity)

        print(f"Error: {error:.2f}%  |  FLOPs: {flops:.1f}M  |  "
              f"Actual sparsity: {sparsity:.1%}")

        # free GPU memory
        del pruned
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return results


# ─────────────────────────────────────────────
# plotting
# ─────────────────────────────────────────────

def plot_flops_vs_error(results_list: list, out_path: str):
    """
    Plot FLOPS vs error line plot for one or more models.

    X axis: FLOPs (MFLOPs) — lower is cheaper
    Y axis: Classification error (%) — lower is better
    Each curve is one model at different pruning ratios.

    Points are annotated with the pruning ratio for readability.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # color palette — distinct for each model
    colors  = ['#2563EB', '#DC2626', '#16A34A', '#9333EA']
    markers = ['o', 's', '^', 'D']

    for i, res in enumerate(results_list):
        flops  = res['flops']
        errors = res['errors']
        ratios = res['ratios']
        label  = res['label']
        color  = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        ax.plot(
            flops, errors,
            color     = color,
            marker    = marker,
            linewidth = 2.0,
            markersize= 7,
            label     = label,
            zorder    = 3,
        )

        # annotate each point with its pruning ratio
        for x, y, r in zip(flops, errors, ratios):
            ax.annotate(
                f'{r:.0%}',
                xy         = (x, y),
                xytext     = (4, 4),
                textcoords = 'offset points',
                fontsize   = 8,
                color      = color,
                alpha      = 0.8,
            )

    ax.set_xlabel('FLOPs (MFLOPs)', fontsize=13)
    ax.set_ylabel('Classification Error (%)', fontsize=13)
    ax.set_title('FLOPs vs Classification Error under Pruning', fontsize=14)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.invert_xaxis()   # convention: left = more pruned (fewer FLOPs)

    # add secondary annotation
    ax.text(
        0.98, 0.02,
        'Points labelled with pruning ratio\n← more pruned   less pruned →',
        transform           = ax.transAxes,
        fontsize            = 8,
        color               = 'gray',
        ha                  = 'right',
        va                  = 'bottom',
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {out_path}")


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)

    # ── determine encoder type from checkpoint ────────────────────────────
    baseline_ckpt = torch.load(args.baseline_ckpt, map_location='cpu')
    baseline_enc  = baseline_ckpt['args']['encoder']

    print(f"Baseline encoder : {baseline_enc}")
    print(f"Device           : {device}")
    print(f"Pruning ratios   : {args.ratios}")

    # ── build dataloaders (test split only needed) ────────────────────────
    # we use baseline encoder type for the dataloader mode
    # if bonus is sparseconv and baseline is foldingnet we need two loaders
    loaders_baseline = make_dataloaders(
        unlabelled_path = args.unlabelled,
        labelled_path   = args.labelled,
        mode            = baseline_enc,
        n_pretrain      = 100,          # not used here
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        seed            = args.seed,
    )
    spatial_shape = loaders_baseline['spatial_shape']

    # ── load baseline classifier ──────────────────────────────────────────
    print(f"\nLoading baseline checkpoint: {args.baseline_ckpt}")
    baseline_clf = load_classifier(
        args.baseline_ckpt, args.codeword_dim, spatial_shape
    )

    # ── baseline pruning sweep ────────────────────────────────────────────
    print(f"\nSweeping pruning ratios for {baseline_enc}...")
    baseline_results = pruning_sweep(
        classifier  = baseline_clf,
        test_loader = loaders_baseline['test'],
        device      = device,
        ratios      = args.ratios,
        label       = f'{baseline_enc} (baseline)',
    )

    all_results = [baseline_results]

    # ── bonus: sparse model sweep ─────────────────────────────────────────
    if args.bonus_ckpt and os.path.exists(args.bonus_ckpt):
        bonus_ckpt   = torch.load(args.bonus_ckpt, map_location='cpu')
        bonus_enc    = bonus_ckpt['args']['encoder']
        print(f"\nLoading bonus checkpoint: {args.bonus_ckpt}")

        # need a separate dataloader if encoder type differs
        if bonus_enc != baseline_enc:
            loaders_bonus = make_dataloaders(
                unlabelled_path = args.unlabelled,
                labelled_path   = args.labelled,
                mode            = bonus_enc,
                n_pretrain      = 100,
                batch_size      = args.batch_size,
                num_workers     = args.num_workers,
                seed            = args.seed,
            )
            bonus_test_loader = loaders_bonus['test']
        else:
            bonus_test_loader = loaders_baseline['test']

        bonus_clf = load_classifier(
            args.bonus_ckpt, args.codeword_dim, spatial_shape
        )

        print(f"\nSweeping pruning ratios for {bonus_enc}...")
        bonus_results = pruning_sweep(
            classifier  = bonus_clf,
            test_loader = bonus_test_loader,
            device      = device,
            ratios      = args.ratios,
            label       = f'{bonus_enc} (sparse)',
        )
        all_results.append(bonus_results)

    # ── save raw numbers ──────────────────────────────────────────────────
    results_path = str(Path(args.out_dir) / 'pruning_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved: {results_path}")

    # ── plot ──────────────────────────────────────────────────────────────
    plot_path = str(Path(args.out_dir) / 'flops_vs_error.png')
    plot_flops_vs_error(all_results, plot_path)

    # ── print summary table ───────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'Ratio':>8} | {'FLOPs (M)':>10} | {'Error (%)':>10} | Model")
    print("-"*65)
    for res in all_results:
        for r, f, e in zip(res['ratios'], res['flops'], res['errors']):
            print(f"  {r:>5.0%}  | {f:>10.1f} | {e:>10.2f} | {res['label']}")
        print("-"*65)


if __name__ == '__main__':
    main()