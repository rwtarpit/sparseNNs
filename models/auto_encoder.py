from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict

from models.folding_net_encoder import FoldingNetEncoder
from models.folding_net_decoder import FoldingNetDecoder, chamfer_distance

FOLDINGNET  = 'foldingnet'


class ClassificationHead(nn.Module):
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


class SparseAutoencoder(nn.Module):
    """
    Unified sparse autoencoder
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

        if isinstance(encoder, FoldingNetEncoder):
            self.encoder_type = FOLDINGNET
        else:
            raise TypeError(
                f"encoder must be FoldingNetEncoder"
                f"got {type(encoder).__name__}"
            )

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        theta  = self._encode(batch)                   # (B, 512)
        pc_out = self.decoder(theta)                   # (B, m, 3)
        loss   = self._reconstruction_loss(batch, pc_out)

        return {
            'theta':  theta,
            'pc_out': pc_out,
            'loss':   loss,
        }

    def _encode(self, batch: Dict) -> torch.Tensor:
        """
        Returns codeword theta: (B, 512).
        """
        return self.encoder(batch['points'])           # (B, N, 3) -> (B, 512)

    def _reconstruction_loss(
        self,
        batch:  Dict,
        pc_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Chamfer distance between input point cloud and reconstruction.
        """
        pts   = batch['points']                      # (B, N, 10)
        energy = pts[:, :, 2:].sum(dim=-1, keepdim=True)  # (B, N, 1)
        pc_input = torch.cat([pts[:, :, :2], energy], dim=-1)  # (B, N, 3)                                             # (B, N, 3)

        return chamfer_distance(pc_input, pc_out)

    def pretraining_step(self, batch: Dict) -> torch.Tensor:
        out = self.forward(batch)
        return out['loss']

    def encode(self, batch: Dict) -> torch.Tensor:
        return self._encode(batch)

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True

    def get_classifier(
        self,
        num_classes: int = 2,
        hidden_dim:  int = 256,
        dropout:     float = 0.3,
        freeze_encoder: bool = True,
    ) -> 'SparseClassifier':
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


class SparseClassifier(nn.Module):
    """
    Encoder + classification head for the fine-tuning phase.
    Built from a pretrained SparseAutoencoder via .get_classifier().
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
        theta = self.encoder(batch['points'])

        return self.head(theta)                            # (B, num_classes)

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True
