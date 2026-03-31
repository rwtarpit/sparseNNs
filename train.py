import os
import sys
import time
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.folding_net_encoder import FoldingNetEncoder
from models.folding_net_decoder import FoldingNetDecoder
from models.auto_encoder import SparseAutoencoder, SparseClassifier
from data.data_loader import make_dataloaders


def parse_args():
    p = argparse.ArgumentParser(description='Sparse Autoencoder Training')

    # paths
    p.add_argument('--unlabelled',    type=str, default='data/Dataset_Specific_Unlabelled.h5')
    p.add_argument('--labelled',      type=str, default='data/Dataset_Specific_labelled.h5')
    p.add_argument('--pretrain_ckpt', type=str, default=None,
                   help='Path to pretrained autoencoder checkpoint for fine-tuning')
    p.add_argument('--out_dir',       type=str, default='/data/checkpoints')

    # training phase
    p.add_argument('--phase',   type=str, choices=['pretrain', 'finetune', 'both'],
                   default='both')
    p.add_argument('--encoder', type=str, choices=['foldingnet', 'sparseconv'],
                   default='foldingnet')

    # pretraining hyperparams
    p.add_argument('--pretrain_epochs', type=int,   default=5)
    p.add_argument('--pretrain_lr',     type=float, default=1e-3)
    p.add_argument('--n_pretrain',      type=int,   default=10_000)

    # fine-tuning hyperparams
    p.add_argument('--finetune_epochs',  type=int,   default=50)
    p.add_argument('--finetune_lr',      type=float, default=1e-3)
    p.add_argument('--unfreeze_epoch',   type=int,   default=10,
                   help='Epoch at which to unfreeze encoder during fine-tuning')
    p.add_argument('--scratch',          action='store_true',
                   help='Fine-tune without pretrained weights (ablation)')

    # shared
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--codeword_dim', type=int,   default=512)
    p.add_argument('--n_max',        type=int,   default=2048)
    p.add_argument('--seed',         type=int,   default=42)

    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device('cpu')
        print("No GPU found — using CPU")
    return dev


def make_encoder(encoder_type: str, codeword_dim: int, spatial_shape: tuple):
    """Construct the right encoder from the type string."""
    if encoder_type == 'foldingnet':
        return FoldingNetEncoder(k=16, codeword_dim=codeword_dim, in_dim=10)


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  Saved checkpoint: {path}")


def log(metrics: dict, log_path: str):
    """Append a metrics dict as a JSON line to a log file."""
    with open(log_path, 'a') as f:
        f.write(json.dumps(metrics) + '\n')


def run_pretrain(args, device: torch.device, loaders: dict) -> str:
    """
    Pretrain the autoencoder on unlabelled data.
    Returns path to best checkpoint.
    """
    print("\n" + "="*55)
    print(f"PHASE 1 — PRETRAINING  ({args.encoder})")
    print("="*55)

    spatial_shape = loaders['spatial_shape']
    encoder = make_encoder(args.encoder, args.codeword_dim, spatial_shape)
    decoder = FoldingNetDecoder(m=2025, codeword_dim=args.codeword_dim)
    model   = SparseAutoencoder(encoder=encoder, decoder=decoder,
                                codeword_dim=args.codeword_dim)
    model   = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.pretrain_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1,args.pretrain_epochs), eta_min=1e-5)

    best_loss = float('inf')
    ckpt_path = str(Path(args.out_dir) / f'{args.encoder}_pretrain_best.pt')
    log_path  = str(Path(args.out_dir) / f'{args.encoder}_pretrain_log.jsonl')

    for epoch in range(1, args.pretrain_epochs + 1):
        t0 = time.time()

        # train
        model.train()
        train_loss = 0.0
        n_batches  = 0

        for batch in loaders['pretrain']:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            optimizer.zero_grad()
            loss = model.pretraining_step(batch)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            train_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_train = train_loss / n_batches
        elapsed   = time.time() - t0

        metrics = {
            'phase': 'pretrain',
            'epoch': epoch,
            'train_chamfer': avg_train,
            'lr': scheduler.get_last_lr()[0],
            'elapsed_s': round(elapsed, 1),
        }
        log(metrics, log_path)

        is_best = avg_train < best_loss
        if is_best:
            best_loss = avg_train
            save_checkpoint({
                'epoch':        epoch,
                'encoder_type': args.encoder,
                'model_state':  model.state_dict(),
                'optim_state':  optimizer.state_dict(),
                'loss':         best_loss,
                'args':         vars(args),
            }, ckpt_path)

        print(
            f"  Epoch {epoch:>3}/{args.pretrain_epochs} | "
            f"Chamfer: {avg_train:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {elapsed:.1f}s"
            + (" *" if is_best else "")
        )

    print(f"\nPretraining done. Best Chamfer: {best_loss:.4f}")
    return ckpt_path


def run_finetune(args, device: torch.device, loaders: dict,
                 pretrain_ckpt: str = None) -> str:
    """
    Fine-tune a classifier on labelled data.
    Loads pretrained encoder if ckpt provided.
    Returns path to best checkpoint.
    """
    print("\n" + "="*55)
    suffix = "scratch" if args.scratch else "pretrained"
    print(f"PHASE 2 — FINE-TUNING  ({args.encoder}, {suffix})")
    print("="*55)

    spatial_shape = loaders['spatial_shape']
    encoder = make_encoder(args.encoder, args.codeword_dim, spatial_shape)
    decoder = FoldingNetDecoder(m=2025, codeword_dim=args.codeword_dim)
    model   = SparseAutoencoder(encoder=encoder, decoder=decoder,
                                codeword_dim=args.codeword_dim)

    # load pretrained weights
    if not args.scratch and pretrain_ckpt and os.path.exists(pretrain_ckpt):
        ckpt = torch.load(pretrain_ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model_state'])
        print(f"  Loaded pretrained weights from {pretrain_ckpt}")
    elif args.scratch:
        print("  Scratch mode — no pretrained weights loaded")
    else:
        print("  WARNING: no pretrain checkpoint found — training from scratch")

    # build classifier
    classifier = model.get_classifier(
        num_classes    = 2,
        hidden_dim     = 256,
        dropout        = 0.3,
        freeze_encoder = True,
    )
    classifier = classifier.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, classifier.parameters()),
        lr=args.finetune_lr,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.finetune_epochs, eta_min=1e-6)

    best_acc  = 0.0
    suffix_str = 'scratch' if args.scratch else 'pretrained'
    ckpt_path  = str(Path(args.out_dir) /
                     f'{args.encoder}_finetune_{suffix_str}_best.pt')
    log_path   = str(Path(args.out_dir) /
                     f'{args.encoder}_finetune_{suffix_str}_log.jsonl')

    for epoch in range(1, args.finetune_epochs + 1):
        t0 = time.time()

        if epoch == args.unfreeze_epoch:
            print(f"\n  [Epoch {epoch}] Unfreezing encoder for full fine-tuning")
            classifier.unfreeze_encoder()
            optimizer = AdamW([
                {'params': classifier.head.parameters(),
                 'lr': args.finetune_lr},
                {'params': classifier.encoder.parameters(),
                 'lr': args.finetune_lr * 0.1},   
            ], weight_decay=1e-4)
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max = args.finetune_epochs - epoch,
                eta_min = 1e-6,
            )

        # train
        classifier.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch in loaders['train']:
            batch  = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items()}
            labels = batch['label']                        # (B,)

            optimizer.zero_grad()
            logits = classifier(batch)                     # (B, 2)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=5.0)
            optimizer.step()

            preds = logits.argmax(dim=1)
            train_loss    += loss.item()
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        scheduler.step()

        # validate 
        classifier.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for batch in loaders['val']:
                batch  = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch.items()}
                labels = batch['label']
                logits = classifier(batch)
                loss   = criterion(logits, labels)

                preds = logits.argmax(dim=1)
                val_loss    += loss.item()
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        train_acc = 100.0 * train_correct / train_total
        val_acc   = 100.0 * val_correct   / val_total
        avg_train = train_loss / len(loaders['train'])
        avg_val   = val_loss   / len(loaders['val'])
        elapsed   = time.time() - t0

        metrics = {
            'phase':       'finetune',
            'encoder':     args.encoder,
            'scratch':     args.scratch,
            'epoch':       epoch,
            'train_loss':  avg_train,
            'train_acc':   round(train_acc, 3),
            'val_loss':    avg_val,
            'val_acc':     round(val_acc, 3),
            'elapsed_s':   round(elapsed, 1),
        }
        log(metrics, log_path)

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            save_checkpoint({
                'epoch':         epoch,
                'encoder_type':  args.encoder,
                'scratch':       args.scratch,
                'model_state':   classifier.state_dict(),
                'val_acc':       best_acc,
                'args':          vars(args),
            }, ckpt_path)

        print(
            f"  Epoch {epoch:>3}/{args.finetune_epochs} | "
            f"Train: {avg_train:.3f} ({train_acc:.1f}%) | "
            f"Val: {avg_val:.3f} ({val_acc:.1f}%) | "
            f"Time: {elapsed:.1f}s"
            + (" *" if is_best else "")
        )

    print(f"\nFine-tuning done. Best val acc: {best_acc:.2f}%")
    return ckpt_path


def evaluate_test(args, device: torch.device, loaders: dict,
                  finetune_ckpt: str) -> dict:
    """
    Load best fine-tuned checkpoint and evaluate on held-out test set.
    Returns dict of metrics.
    """
    print("\n" + "="*55)
    print("TEST SET EVALUATION")
    print("="*55)

    spatial_shape = loaders['spatial_shape']
    encoder    = make_encoder(args.encoder, args.codeword_dim, spatial_shape)
    classifier = SparseClassifier(
        encoder      = encoder,
        encoder_type = args.encoder,
        num_classes  = 2,
        codeword_dim = args.codeword_dim,
    )

    ckpt = torch.load(finetune_ckpt, map_location='cpu')
    classifier.load_state_dict(ckpt['model_state'])
    classifier = classifier.to(device)
    classifier.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loaders['test']:
            batch  = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items()}
            labels = batch['label']
            logits = classifier(batch)
            loss   = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            test_loss += loss.item()
            correct   += (preds == labels).sum().item()
            total     += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc      = 100.0 * correct / total
    avg_loss = test_loss / len(loaders['test'])

    # per-class accuracy
    for cls in [0, 1]:
        mask     = [l == cls for l in all_labels]
        cls_preds = [p for p, m in zip(all_preds, mask) if m]
        cls_labs  = [l for l, m in zip(all_labels, mask) if m]
        cls_acc   = 100.0 * sum(p == l for p, l in zip(cls_preds, cls_labs)) \
                    / len(cls_labs)
        print(f"  Class {cls} accuracy: {cls_acc:.2f}%  ({len(cls_labs)} samples)")

    print(f"  Overall accuracy  : {acc:.2f}%")
    print(f"  Test loss         : {avg_loss:.4f}")

    return {
        'encoder':    args.encoder,
        'scratch':    args.scratch,
        'test_acc':   round(acc, 3),
        'test_loss':  round(avg_loss, 4),
        'n_samples':  total,
    }


def main():
    args   = parse_args()
    device = get_device()
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nEncoder : {args.encoder}")
    print(f"Phase   : {args.phase}")
    print(f"Out dir : {args.out_dir}")

    # build dataloaders
    print("\nBuilding dataloaders...")
    loaders = make_dataloaders(
        unlabelled_path = args.unlabelled,
        labelled_path   = args.labelled,
        mode            = args.encoder,
        n_pretrain      = args.n_pretrain,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        seed            = args.seed,
    )

    pretrain_ckpt  = args.pretrain_ckpt
    finetune_ckpt  = None

    if args.phase in ('pretrain', 'both'):
        pretrain_ckpt = run_pretrain(args, device, loaders)

    if args.phase in ('finetune', 'both'):
        finetune_ckpt = run_finetune(
            args, device, loaders,
            pretrain_ckpt = pretrain_ckpt,
        )

    if finetune_ckpt and os.path.exists(finetune_ckpt):
        results = evaluate_test(args, device, loaders, finetune_ckpt)

        results_path = str(Path(args.out_dir) /
                           f'{args.encoder}_test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}") 
