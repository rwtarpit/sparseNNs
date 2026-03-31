import torch
import torch.nn as nn
import math


def build_grid(m: int, device: torch.device) -> torch.Tensor:
    """
    Build a fixed 2D grid of m points on a unit square centred at origin.
    """
    side  = int(math.isqrt(m))
    assert side * side == m, \
        f"m={m} must be a perfect square for a square grid (got side={side})"

    lin   = torch.linspace(-0.5, 0.5, side, device=device)
    gy, gx = torch.meshgrid(lin, lin, indexing='ij')   # each (side, side)
    grid  = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (m, 2)
    return grid


class FoldingLayer(nn.Module):
    """
    One folding operation from the paper
    """

    def __init__(self, in_channels: int, out_channels: int = 3):
        super().__init__()
        hidden = 512
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_channels, bias=True),
        )

    def forward(self, grid_pts: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        B, m, _ = grid_pts.shape

        theta_rep = theta.unsqueeze(1).expand(-1, m, -1)       # (B, m, 512)

        x = torch.cat([grid_pts, theta_rep], dim=-1)           # (B, m, D+512)

        B, m, C = x.shape
        out = self.mlp(x.reshape(B * m, C))                    # (B*m, 3)
        out = out.reshape(B, m, 3)                             # (B, m, 3)

        return out


class FoldingNetDecoder(nn.Module):
    """
    Folding-based decoder from FoldingNet (Yang et al., CVPR 2018).
    """

    def __init__(self, m: int = 2025, codeword_dim: int = 512):
        super().__init__()
        self.m            = m
        self.codeword_dim = codeword_dim

        # 1st fold: input is [2D grid point (2), codeword (512)] = 514
        self.fold1 = FoldingLayer(in_channels=2 + codeword_dim, out_channels=3)

        # 2nd fold: input is [3D intermediate point (3), codeword (512)] = 515
        self.fold2 = FoldingLayer(in_channels=3 + codeword_dim, out_channels=3)

        grid = build_grid(m, device=torch.device('cpu'))   # (m, 2)
        self.register_buffer('grid', grid)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        B = theta.shape[0]

        grid = self.grid.unsqueeze(0).expand(B, -1, -1)       # (B, m, 2)

        pc_intermediate = self.fold1(grid, theta)              # (B, m, 3)

        pc_out = self.fold2(pc_intermediate, theta)            # (B, m, 3)

        return pc_out


def chamfer_distance(s: torch.Tensor, s_hat: torch.Tensor) -> torch.Tensor:
    """
    Chamfer distance

        d_CH(S, Ŝ) = max(
            (1/|S|)  Σ_{x∈S}   min_{x̂∈Ŝ} ||x - x̂||²,
            (1/|Ŝ|)  Σ_{x̂∈Ŝ} min_{x∈S}  ||x̂ - x||²
        )
    """
    # pairwise squared distances  (B, N, m)
    dist2 = torch.cdist(s, s_hat, p=2) ** 2                           # (B, N, m)

    d_fwd = dist2.min(dim=2).values.mean(dim=1)               # (B,)

    d_bwd = dist2.min(dim=1).values.mean(dim=1)               # (B,)

    loss  = torch.max(d_fwd, d_bwd).mean()                    # scalar

    return loss
