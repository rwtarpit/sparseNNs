import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────
# 2D grid builder
# ─────────────────────────────────────────────

def build_grid(m: int, device: torch.device) -> torch.Tensor:
    """
    Build a fixed 2D grid of m points on a unit square centred at origin.

    From the paper:
        "we replicate it m times and concatenate the m×512 matrix with an
         m×2 matrix that contains the m grid points on a square centred
         at the origin."
        "m is chosen as 2025 which is the closest square number to 2048."

    The grid is fixed — it does not change during training.
    The decoder learns to *fold* this grid into the shape of the input.

    Args:
        m      : number of grid points (should be a perfect square)
        device : torch device

    Returns:
        grid : (m, 2)  grid point coordinates in [-0.5, 0.5]²
    """
    side  = int(math.isqrt(m))
    assert side * side == m, \
        f"m={m} must be a perfect square for a square grid (got side={side})"

    # linspace from -0.5 to 0.5 inclusive on each axis
    lin   = torch.linspace(-0.5, 0.5, side, device=device)
    gy, gx = torch.meshgrid(lin, lin, indexing='ij')   # each (side, side)
    grid  = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (m, 2)
    return grid


# ─────────────────────────────────────────────
# single folding operation
# ─────────────────────────────────────────────

class FoldingLayer(nn.Module):
    """
    One folding operation from the paper (Definition 1):

        "The concatenation of replicated codewords to low-dimensional
         grid points, followed by a pointwise MLP."

    The i-th output row is  f([u_i, θ])  where:
        u_i  = i-th grid point  (2D on 1st fold, 3D on 2nd fold)
        θ    = codeword (replicated m times)
        f    = 3-layer MLP applied independently to each row

    Args:
        in_channels  : dim of [u_i, θ] concatenation
                       1st fold: 2 + 512 = 514
                       2nd fold: 3 + 512 = 515
        out_channels : output dim — always 3 (x, y, z positions)
    """

    def __init__(self, in_channels: int, out_channels: int = 3):
        super().__init__()

        # 3-layer MLP, paper uses same hidden dim throughout
        # hidden dim set to 512 following common FoldingNet implementations
        hidden = 512
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_channels, bias=True),
        )

    def forward(self, grid_pts: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid_pts : (B, m, D)    D=2 for 1st fold, D=3 for 2nd fold
            theta    : (B, 512)     codeword

        Returns:
            out : (B, m, 3)
        """
        B, m, _ = grid_pts.shape

        # replicate codeword m times  →  (B, m, 512)
        theta_rep = theta.unsqueeze(1).expand(-1, m, -1)       # (B, m, 512)

        # concatenate  →  (B, m, D+512)
        x = torch.cat([grid_pts, theta_rep], dim=-1)           # (B, m, D+512)

        # apply MLP row-wise  →  (B, m, 3)
        B, m, C = x.shape
        out = self.mlp(x.reshape(B * m, C))                    # (B*m, 3)
        out = out.reshape(B, m, 3)                             # (B, m, 3)

        return out


# ─────────────────────────────────────────────
# FoldingNet Decoder
# ─────────────────────────────────────────────

class FoldingNetDecoder(nn.Module):
    """
    Folding-based decoder from FoldingNet (Yang et al., CVPR 2018).

    Pipeline (exactly as described in Section 2.2):
        1. Take codeword θ  (B, 512)
        2. Build fixed 2D grid  →  (m, 2)
        3. Replicate θ m times, concat with grid  →  (B, m, 514)
        4. 1st folding MLP  →  intermediate point cloud  (B, m, 3)
        5. Concat θ again  →  (B, m, 515)
        6. 2nd folding MLP  →  output point cloud  (B, m, 3)

    The decoder is shared between both encoder architectures.
    It receives only the codeword θ — it has no knowledge of
    which encoder produced it.

    Args:
        m            : number of output points (paper: 2025)
        codeword_dim : must match encoder output (paper: 512)
    """

    def __init__(self, m: int = 2025, codeword_dim: int = 512):
        super().__init__()
        self.m            = m
        self.codeword_dim = codeword_dim

        # 1st fold: input is [2D grid point (2), codeword (512)] = 514
        self.fold1 = FoldingLayer(in_channels=2 + codeword_dim, out_channels=3)

        # 2nd fold: input is [3D intermediate point (3), codeword (512)] = 515
        self.fold2 = FoldingLayer(in_channels=3 + codeword_dim, out_channels=3)

        # grid is fixed — register as buffer so it moves with .to(device)
        # but is NOT a learnable parameter
        grid = build_grid(m, device=torch.device('cpu'))   # (m, 2)
        self.register_buffer('grid', grid)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            theta : (B, 512)  codeword from either encoder

        Returns:
            pc_out : (B, m, 3)  reconstructed point cloud
        """
        B = theta.shape[0]

        # expand grid to batch  →  (B, m, 2)
        grid = self.grid.unsqueeze(0).expand(B, -1, -1)       # (B, m, 2)

        # ── 1st folding: 2D grid → 3D intermediate cloud
        # input:  [grid (B,m,2), θ (B,512)]
        # output: (B, m, 3)
        pc_intermediate = self.fold1(grid, theta)              # (B, m, 3)

        # ── 2nd folding: 3D intermediate → final 3D cloud
        # input:  [intermediate (B,m,3), θ (B,512)]
        # output: (B, m, 3)
        pc_out = self.fold2(pc_intermediate, theta)            # (B, m, 3)

        return pc_out


# ─────────────────────────────────────────────
# Chamfer distance loss
# ─────────────────────────────────────────────

def chamfer_distance(s: torch.Tensor, s_hat: torch.Tensor) -> torch.Tensor:
    """
    Extended Chamfer distance from Equation 1 of the paper:

        d_CH(S, Ŝ) = max(
            (1/|S|)  Σ_{x∈S}   min_{x̂∈Ŝ} ||x - x̂||²,
            (1/|Ŝ|)  Σ_{x̂∈Ŝ} min_{x∈S}  ||x̂ - x||²
        )

    The max enforces both directions simultaneously:
        - every input point has a close match in the reconstruction
        - every reconstructed point has a close match in the input

    Args:
        s     : (B, N, 3)  input point cloud
        s_hat : (B, m, 3)  reconstructed point cloud

    Returns:
        loss : scalar  mean over batch
    """
    # pairwise squared distances  (B, N, m)
    dist2 = torch.cdist(s, s_hat, p=2) ** 2                           # (B, N, m)

    # forward:  for each input point → closest reconstructed point
    d_fwd = dist2.min(dim=2).values.mean(dim=1)               # (B,)

    # backward: for each reconstructed point → closest input point
    d_bwd = dist2.min(dim=1).values.mean(dim=1)               # (B,)

    # max of the two directions (paper Eq. 1)
    loss  = torch.max(d_fwd, d_bwd).mean()                    # scalar

    return loss


# ─────────────────────────────────────────────
# quick shape test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    B, N  = 2, 1024
    m     = 2025    # closest perfect square to 2048 (paper choice)

    decoder = FoldingNetDecoder(m=m, codeword_dim=512)
    decoder.eval()

    # simulate a codeword coming from the encoder
    theta   = torch.randn(B, 512)

    with torch.no_grad():
        pc_out = decoder(theta)

    print(f"Codeword shape      : {theta.shape}")
    print(f"Output cloud shape  : {pc_out.shape}")
    assert pc_out.shape == (B, m, 3), "output shape mismatch"
    print("Shape test passed.")

    # test chamfer loss
    pc_input = torch.randn(B, N, 3)
    loss = chamfer_distance(pc_input, pc_out)
    print(f"Chamfer loss (random): {loss.item():.4f}")

    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"Total parameters: {total_params:,}")