"""
dataset.py
──────────
Dataset classes for both pretraining (unlabelled) and fine-tuning (labelled).

Data format (both files):
    jet : (N, 125, 125, 8)  float32   — 8-channel sparse detector images
    Y   : (N, 1)            float32   — binary labels 0/1 (labelled only)

Active pixel definition:
    A pixel at (r, c) is active if ANY of its 8 channel values is nonzero.
    Mean ~1220 active pixels per image (7.81% sparsity).

Two output modes controlled by `mode` argument:
    'foldingnet'  — returns point cloud  (N_active, 10)
                    each point: (row_norm, col_norm, ch0, ..., ch7)
                    rows/cols normalised to [-0.5, 0.5] to match the
                    fixed 2D grid in the folding decoder

    'sparseconv'  — returns sparse tensors
                    coords : (N_active, 3)  int32  [batch_idx, row, col]
                    feats  : (N_active, 8)  float32

Both modes pad/truncate to a fixed N_max points per sample so DataLoader
can collate variable-length point clouds into a single batch tensor.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, Dict


H, W, C      = 125, 125, 8     # image spatial dims and channels
N_MAX        = 2048             # pad/truncate all clouds to this size
                                # chosen > observed max (2173 would need 2200,
                                # we use 2048 for efficiency — rare overflow
                                # gets truncated, which is fine)
ROW_SCALE    = 1.0 / (H - 1)   # maps row index [0, 124] → [0, 1]
COL_SCALE    = 1.0 / (W - 1)   # maps col index [0, 124] → [0, 1]



def image_to_pointcloud(image: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    """
    Convert a single (125, 125, 8) image to a point cloud.
    Each active pixel becomes one point with 10 dimensions

    row_norm, col_norm are in [-0.5, 0.5]

    """
    # find active pixels — any channel nonzero
    active_mask = np.any(image != 0, axis=-1)          # (H, W)
    rows, cols  = np.where(active_mask)                # each (n_active,)

    if len(rows) == 0:
        # degenerate case — empty image
        return np.zeros((n_max, 10), dtype=np.float32)

    # normalise coordinates to [-0.5, 0.5]
    rows_norm = rows * ROW_SCALE - 0.5                 # (n_active,)
    cols_norm = cols * COL_SCALE - 0.5                 # (n_active,)

    # gather channel values at active pixels
    ch_vals   = image[rows, cols, :]                   # (n_active, 8)

    # assemble point cloud — (n_active, 10)
    pc = np.concatenate([
        rows_norm[:, None],
        cols_norm[:, None],
        ch_vals,
    ], axis=1).astype(np.float32)

    if len(pc) > n_max:
        idx = np.random.choice(len(pc), n_max, replace=False)
        pc  = pc[idx]

    # pad with zeros if under n_max
    if len(pc) < n_max:
        pad = np.zeros((n_max - len(pc), 10), dtype=np.float32)
        pc  = np.concatenate([pc, pad], axis=0)

    return pc                                          # (n_max, 10)


class UnlabelledDataset(Dataset):
    """
    Unlabelled dataset for pretraining
    """

    def __init__(
        self,
        h5_path:  str,
        indices:  Optional[np.ndarray] = None,
        mode:     str = 'foldingnet',
        n_max:    int = N_MAX,
    ):
        assert mode in ('foldingnet'), \
            f"mode must be 'foldingnet', got '{mode}'"

        self.h5_path = h5_path
        self.mode    = mode
        self.n_max   = n_max
        self.h5_file = None   

        # determine which indices to use
        with h5py.File(h5_path, 'r') as f:
            total = f['jet'].shape[0]

        if indices is None:
            self.indices = np.arange(total)
        else:
            self.indices = np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        real_idx = int(self.indices[idx])
        image    = self.h5_file['jet'][real_idx]       # (125, 125, 8)

        if self.mode == 'foldingnet':
            pc = image_to_pointcloud(image, self.n_max)
            return {'points': torch.from_numpy(pc)}    # (n_max, 10)

    def __del__(self):
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass


class LabelledDataset(Dataset):
    """
    Labelled dataset for fine-tuning.
    """

    def __init__(
        self,
        h5_path:  str,
        indices:  np.ndarray,
        mode:     str = 'foldingnet',
        n_max:    int = N_MAX,
    ):
        assert mode in ('foldingnet'), \
            f"mode must be 'foldingnet' or, got '{mode}'"

        self.h5_path = h5_path
        self.mode    = mode
        self.n_max   = n_max
        self.indices = np.asarray(indices)
        self.h5_file = None

        with h5py.File(h5_path, 'r') as f:
            self.labels = f['Y'][:].squeeze(1).astype(np.int64)  # (N,)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        real_idx = int(self.indices[idx])
        image    = self.h5_file['jet'][real_idx]       #
        label    = self.labels[real_idx]               

        if self.mode == 'foldingnet':
            pc = image_to_pointcloud(image, self.n_max)
            return {
                'points': torch.from_numpy(pc),       
                'label':  torch.tensor(label),         
            }

    def __del__(self):
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass



def collate_foldingnet(batch: list) -> Dict:
    """
    Standard collate for foldingnet model.
    """
    points = torch.stack([b['points'] for b in batch], dim=0) 
    out    = {'points': points}

    if 'label' in batch[0]:
        out['label'] = torch.stack([b['label'] for b in batch])  

    return out


def make_dataloaders(
    unlabelled_path: str,
    labelled_path:   str,
    mode:            str   = 'foldingnet',
    n_pretrain:      int   = 10_000,
    batch_size:      int   = 32,
    num_workers:     int   = 4,
    val_ratio:       float = 0.1,
    test_ratio:      float = 0.1,
    seed:            int   = 42,
) -> Dict:
    assert mode in ('foldingnet', 'sparseconv')
    collate_fn = collate_foldingnet if mode == 'foldingnet' else None
    rng        = np.random.default_rng(seed)

    with h5py.File(unlabelled_path, 'r') as f:
        total_unlabelled = f['jet'].shape[0]

    pretrain_idx = rng.choice(total_unlabelled, size=n_pretrain, replace=False)
    pretrain_ds  = UnlabelledDataset(unlabelled_path, pretrain_idx, mode=mode)
    pretrain_loader = DataLoader(
        pretrain_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        collate_fn  = collate_fn,
        pin_memory  = True,
        drop_last   = True,     
    )

    with h5py.File(labelled_path, 'r') as f:
        all_labels = f['Y'][:].squeeze(1).astype(np.int64)

    all_idx = np.arange(len(all_labels))

    # stratified split 
    train_idx, temp_idx = train_test_split(
        all_idx,
        test_size = val_ratio + test_ratio,
        stratify  = all_labels,
        random_state = seed,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size = test_ratio / (val_ratio + test_ratio),
        stratify  = all_labels[temp_idx],
        random_state = seed,
    )

    print(f"Split sizes — train: {len(train_idx)} | "
          f"val: {len(val_idx)} | test: {len(test_idx)}")

    def make_loader(indices, shuffle):
        ds = LabelledDataset(labelled_path, indices, mode=mode)
        return DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
            collate_fn  = collate_fn,
            pin_memory  = True,
            drop_last   = False,
        )

    return {
        'pretrain':     pretrain_loader,
        'train':        make_loader(train_idx, shuffle=True),
        'val':          make_loader(val_idx,   shuffle=False),
        'test':         make_loader(test_idx,  shuffle=False),
        'spatial_shape': (H, W),
    }
