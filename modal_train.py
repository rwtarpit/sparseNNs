import os
import modal
from pathlib import Path

app = modal.App("sparseNN-training")

data_vol = modal.Volume.from_name("sparseNN")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(
        Path(__file__).parent,
        remote_path="/root",
        ignore=["data/*.h5", "__pycache__/", "*.pyc", ".git/", ".venv/"],
    )
)


def _train_worker(
    phase: str,
    encoder: str,
    pretrain_epochs: int,
    finetune_epochs: int,
    batch_size: int,
    seed: int,
):
    import sys
    sys.path.insert(0, "/root")

    import torch
    from data.data_loader import make_dataloaders
    from train import run_pretrain, run_finetune, evaluate_test, get_device, set_seed

    import argparse
    args = argparse.Namespace(
        unlabelled      = "/data/data/Dataset_Specific_Unlabelled.h5",
        labelled        = "/data/data/Dataset_Specific_labelled.h5",
        out_dir         = "/data/checkpoints",
        phase           = phase,
        encoder         = encoder,
        pretrain_ckpt   = None,
        pretrain_epochs = pretrain_epochs,
        pretrain_lr     = 1e-3,
        n_pretrain      = 50000,
        finetune_epochs = finetune_epochs,
        finetune_lr     = 1e-3,
        unfreeze_epoch  = 10,
        scratch         = False,
        batch_size      = batch_size,
        num_workers     = 4,
        codeword_dim    = 512,
        n_max           = 2048,
        seed            = seed,
    )

    device = get_device()
    set_seed(seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Building dataloaders...")
    loaders = make_dataloaders(
        unlabelled_path = args.unlabelled,
        labelled_path   = args.labelled,
        mode            = args.encoder,
        n_pretrain      = args.n_pretrain,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        seed            = args.seed,
    )

    pretrain_ckpt = None
    finetune_ckpt = None

    if phase in ("pretrain", "both"):
        pretrain_ckpt = run_pretrain(args, device, loaders)

    if phase in ("finetune", "both"):
        finetune_ckpt = run_finetune(args, device, loaders, pretrain_ckpt=pretrain_ckpt)

    if finetune_ckpt and os.path.exists(finetune_ckpt):
        evaluate_test(args, device, loaders, finetune_ckpt)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 12,
    volumes={"/data": data_vol},
)
def train(
    phase:           str = "both",
    encoder:         str = "foldingnet",
    pretrain_epochs: int = 30,
    finetune_epochs: int = 50,
    batch_size:      int = 128,
    seed:            int = 42,
):
    print("Starting training from Modal", flush=True)
    try:
        _train_worker(phase, encoder, pretrain_epochs, finetune_epochs, batch_size, seed)
    finally:
        data_vol.commit()


@app.local_entrypoint()
def main():
    train.remote(phase="both", encoder="foldingnet")