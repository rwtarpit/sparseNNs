"""
autoencoder.py
──────────────
Full sparse autoencoder combining either encoder with the shared
FoldingNet decoder.

Two encoder modes, selected by passing the encoder at construction time:
    - 'foldingnet' : FoldingNetEncoder   — graph-based, point cloud input
    - 'sparseconv' : SparseConvEncoder   — submanifold sparse ResNet, voxel input

The decoder is identical in both cases. The codeword interface (B, 512)
is the only contract between encoder and decoder.

Usage
─────
    # baseline
    model = SparseAutoencoder(encoder=FoldingNetEncoder(), decoder=FoldingNetDecoder())

    # bonus
    model = SparseAutoencoder(encoder=SparseConvEncoder(), decoder=FoldingNetDecoder())

    # pretraining step
    loss = model.pretraining_step(batch)

    # after pretraining — extract encoder only for fine-tuning
    classifier = model.get_classifier(num_classes=2)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Tuple, Union

from models.folding_net_encoder import FoldingNetEncoder
from models.folding_net_decoder import FoldingNetDecoder, chamfer_distance


# ─────────────────────────────────────────────
# encoder type sentinel
# ─────────────────────────────────────────────

FOLDINGNET  = 'foldingnet'


# ─────────────────────────────────────────────
# classification head
# ─────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Lightweight MLP head attached to frozen/unfrozen encoder for fine-tuning.

    Input  : codeword theta  (B, codeword_dim)
    Output : logits          (B, num_classes)

    A single hidden layer with dropout is enough given the small
    labelled dataset size typical in these physics tasks.
    """

    def __init__(self, codeword_dim: int = 512, num_classes: int = 2,
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(codeword_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return self.net(theta)


# ─────────────────────────────────────────────
# main autoencoder
# ─────────────────────────────────────────────

class SparseAutoencoder(nn.Module):
    """
    Unified sparse autoencoder — encoder swappable, decoder fixed.

    Args:
        encoder      : FoldingNetEncoder or SparseConvEncoder instance.
                       Detected automatically from type.
        decoder      : FoldingNetDecoder instance (shared between both modes).
        codeword_dim : latent dimension — must match both encoder output
                       and decoder expectation (default 512).

    Forward input format depends on encoder type:

        FoldingNet mode:
            batch = {
                'points': Tensor(B, N, 3)   # point cloud (row, col, intensity)
            }

        SparseConv mode:
            batch = {
                'coords': Tensor(num_active, 3)   # [batch_idx, row, col]
                'feats':  Tensor(num_active, 1)   # pixel intensities
                'batch_size': int
            }

    Both modes return the same output dict:
        {
            'theta':   Tensor(B, 512)     # codeword
            'pc_out':  Tensor(B, m, 3)    # reconstructed point cloud
            'loss':    scalar Tensor      # chamfer distance
        }
    """

    def __init__(
        self,
        encoder: FoldingNetEncoder,
        decoder: FoldingNetDecoder,
        codeword_dim: int = 512,
    ):
        super().__init__()

        self.encoder      = encoder
        self.decoder      = decoder
        self.codeword_dim = codeword_dim

        # detect encoder type once at construction — avoids isinstance checks
        # in the hot path of forward()
        if isinstance(encoder, FoldingNetEncoder):
            self.encoder_type = FOLDINGNET
        else:
            raise TypeError(
                f"encoder must be FoldingNetEncoder"
                f"got {type(encoder).__name__}"
            )

    # ── core forward ──────────────────────────────────────────────────────────

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Full autoencoder forward pass: encode → decode → loss.

        Args:
            batch : dict with keys depending on encoder_type (see class docstring)

        Returns:
            dict with 'theta', 'pc_out', 'loss'
        """
        theta  = self._encode(batch)                   # (B, 512)
        pc_out = self.decoder(theta)                   # (B, m, 3)
        loss   = self._reconstruction_loss(batch, pc_out)

        return {
            'theta':  theta,
            'pc_out': pc_out,
            'loss':   loss,
        }

    # ── encode ────────────────────────────────────────────────────────────────

    def _encode(self, batch: Dict) -> torch.Tensor:
        """
        Route to the correct encoder based on encoder_type.
        Returns codeword theta: (B, 512).
        """
        # FoldingNet expects a point cloud tensor directly
        return self.encoder(batch['points'])           # (B, N, 3) → (B, 512)

    # ── loss ──────────────────────────────────────────────────────────────────

    def _reconstruction_loss(
        self,
        batch:  Dict,
        pc_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Chamfer distance between input point cloud and reconstruction.

        For FoldingNet mode: 'points' is already a point cloud (B, N, 3).
        For SparseConv mode: we reconstruct the point cloud from
            coords/feats before computing the loss.

        Both modes compare against a (B, N, 3) point cloud, so the
        decoder always has a fair reconstruction target regardless of
        which encoder was used.
        """
        pts   = batch['points']                      # (B, N, 10)
        energy = pts[:, :, 2:].sum(dim=-1, keepdim=True)  # (B, N, 1)
        pc_input = torch.cat([pts[:, :, :2], energy], dim=-1)  # (B, N, 3)                                             # (B, N, 3)

        return chamfer_distance(pc_input, pc_out)

    
    # ── pretraining convenience ───────────────────────────────────────────────

    def pretraining_step(self, batch: Dict) -> torch.Tensor:
        """
        Single pretraining step — returns the scalar loss.
        Caller handles optimizer.zero_grad(), loss.backward(), optimizer.step().

        Example training loop:
            for batch in unlabelled_loader:
                optimizer.zero_grad()
                loss = model.pretraining_step(batch)
                loss.backward()
                optimizer.step()
        """
        out = self.forward(batch)
        return out['loss']

    # ── fine-tuning helpers ───────────────────────────────────────────────────

    def encode(self, batch: Dict) -> torch.Tensor:
        """
        Encode only — no decoder. Used during fine-tuning and inference.
        Returns codeword theta: (B, 512).
        """
        return self._encode(batch)

    def freeze_encoder(self) -> None:
        """
        Freeze all encoder parameters.
        Call before fine-tuning if you want the encoder fully frozen.
        Linear probing mode: only the classification head trains.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """
        Unfreeze encoder parameters for full fine-tuning.
        Call after a few warm-up epochs of linear probing.
        """
        for param in self.encoder.parameters():
            param.requires_grad = True

    def get_classifier(
        self,
        num_classes: int = 2,
        hidden_dim:  int = 256,
        dropout:     float = 0.3,
        freeze_encoder: bool = True,
    ) -> 'SparseClassifier':
        """
        Build a classifier that wraps this autoencoder's encoder.

        Discards the decoder — only the encoder is kept.
        Returns a SparseClassifier instance ready for fine-tuning.

        Args:
            num_classes    : output classes (2 for binary task)
            hidden_dim     : hidden dim of the classification head
            dropout        : dropout rate in the head
            freeze_encoder : if True, encoder weights are frozen initially
                             (linear probing). Set to False for full fine-tuning.
        """
        if freeze_encoder:
            self.freeze_encoder()

        return SparseClassifier(
            encoder      = self.encoder,
            encoder_type = self.encoder_type,
            num_classes  = num_classes,
            codeword_dim = self.codeword_dim,
            hidden_dim   = hidden_dim,
            dropout      = dropout,
        )


# ─────────────────────────────────────────────
# classifier (encoder + head, no decoder)
# ─────────────────────────────────────────────

class SparseClassifier(nn.Module):
    """
    Encoder + classification head for the fine-tuning phase.
    Built from a pretrained SparseAutoencoder via .get_classifier().

    The decoder is gone — only the encoder and head remain.
    The encoder's weights are pretrained; only the head starts from scratch.

    Forward input format is the same as SparseAutoencoder
    (depends on encoder_type).

    Returns:
        logits : (B, num_classes)
    """

    def __init__(
        self,
        encoder:      FoldingNetEncoder,
        encoder_type: str,
        num_classes:  int   = 2,
        codeword_dim: int   = 512,
        hidden_dim:   int   = 256,
        dropout:      float = 0.3,
    ):
        super().__init__()
        self.encoder      = encoder
        self.encoder_type = encoder_type
        self.head         = ClassificationHead(
            codeword_dim = codeword_dim,
            num_classes  = num_classes,
            hidden_dim   = hidden_dim,
            dropout      = dropout,
        )

    def forward(self, batch: Dict) -> torch.Tensor:
        """
        Args:
            batch : same format as SparseAutoencoder.forward()

        Returns:
            logits : (B, num_classes)
        """
        theta = self.encoder(batch['points'])

        return self.head(theta)                            # (B, num_classes)

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True


# ─────────────────────────────────────────────
# quick shape tests
# ─────────────────────────────────────────────

if __name__ == '__main__':
    torch.manual_seed(0)
    B, N = 2, 1024
    H, W = 224, 224

    print("=" * 55)
    print("TEST 1 — FoldingNet encoder + FoldingNet decoder")
    print("=" * 55)

    model_fn = SparseAutoencoder(
        encoder = FoldingNetEncoder(k=16, codeword_dim=512),
        decoder = FoldingNetDecoder(m=2025, codeword_dim=512),
    )
    model_fn.eval()

    batch_fn = {'points': torch.randn(B, N, 3)}

    with torch.no_grad():
        out = model_fn(batch_fn)

    print(f"  theta  : {out['theta'].shape}")
    print(f"  pc_out : {out['pc_out'].shape}")
    print(f"  loss   : {out['loss'].item():.4f}")
    assert out['theta'].shape  == (B, 512)
    assert out['pc_out'].shape == (B, 2025, 3)
    print("  PASSED\n")

    # ── fine-tuning path ──────────────────────────────────────────────────
    print("=" * 55)
    print("TEST 3 — get_classifier() for fine-tuning  (FoldingNet)")
    print("=" * 55)

    classifier = model_fn.get_classifier(num_classes=2, freeze_encoder=True)
    classifier.eval()

    with torch.no_grad():
        logits = classifier(batch_fn)

    print(f"  logits : {logits.shape}")
    assert logits.shape == (B, 2)

    # confirm encoder is frozen
    frozen = all(not p.requires_grad for p in classifier.encoder.parameters())
    print(f"  encoder frozen : {frozen}")
    assert frozen
    print("  PASSED\n")

    # ── param counts ──────────────────────────────────────────────────────
    print("=" * 55)
    print("PARAMETER COUNTS")
    print("=" * 55)

    def count(m):
        return sum(p.numel() for p in m.parameters())

    print(f"  FoldingNet encoder : {count(model_fn.encoder):>12,}")
    print(f"  decoder     : {count(model_fn.decoder):>12,}")
    print(f"  Classification head: {count(classifier.head):>12,}")