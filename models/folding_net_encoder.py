import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    pos = x[:, :, :2].detach()                              # (B, N, 2)
    with torch.no_grad():
        dist = torch.cdist(pos, pos, p=2)                   # (B, N, N)
        B, N, _ = dist.shape
        diag = torch.eye(N, dtype=torch.bool, device=x.device).unsqueeze(0)
        dist = dist.masked_fill(diag, float('inf'))
    return dist.topk(k, dim=-1, largest=False).indices      # (B, N, K)


def local_covariance(x, idx):
    B, N, K = idx.shape
    energy = x[:, :, 2:].sum(dim=-1, keepdim=True)
    pos    = torch.cat([x[:, :, :2], energy], dim=-1).detach()  # (B, N, 3)

    # gather neighbours: source must be (B, N, 3), index over dim=1
    # idx: (B, N, K) → expand to (B, N, K, 3) for gathering from (B, N, 3)
    pos_exp  = pos.unsqueeze(2).expand(-1, -1, K, -1)           # (B, N, K, 3)  — not the gather source
    idx_exp  = idx.unsqueeze(-1).expand(-1, -1, -1, 3)          # (B, N, K, 3)

    # gather from (B, N, 3) along dim=1 using indices of shape (B, N*K, 3)
    # reshape idx to (B, N*K, 3), gather, reshape back
    idx_flat = idx.reshape(B, N * K)                            # (B, N*K)
    idx_flat = idx_flat.unsqueeze(-1).expand(-1, -1, 3)         # (B, N*K, 3)
    pos_nbrs = torch.gather(pos, 1, idx_flat)                   # (B, N*K, 3)
    pos_nbrs = pos_nbrs.reshape(B, N, K, 3)                     # (B, N, K, 3)

    pts   = torch.cat([pos.unsqueeze(2), pos_nbrs], dim=2)      # (B, N, K+1, 3)
    mu    = pts.mean(dim=2, keepdim=True)
    pts_c = pts - mu
    cov   = torch.einsum('bnkd,bnke->bnde', pts_c, pts_c) / (K + 1)
    return cov.reshape(B, N, 9)


class GraphMaxPool(nn.Module):
    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()
        self.k  = k
        self.fc = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, idx):
        B, N, C = x.shape
        K = idx.shape[-1]

        # gather neighbours without expanding to (B, N, N, C)
        idx_flat  = idx.reshape(B, N * K)                        # (B, N*K)
        idx_flat  = idx_flat.unsqueeze(-1).expand(-1, -1, C)     # (B, N*K, C)
        nbr_feats = torch.gather(x, 1, idx_flat)                 # (B, N*K, C)
        nbr_feats = nbr_feats.reshape(B, N, K, C)                # (B, N, K, C)

        agg = F.relu(nbr_feats.max(dim=2).values)                # (B, N, C)
        y   = self.fc(agg.reshape(B * N, C))
        return self.bn(y).reshape(B, N, -1)


def make_mlp(dims, last_relu=True):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1], bias=False))
        layers.append(nn.BatchNorm1d(dims[i+1]))
        if i < len(dims) - 2 or last_relu:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class FoldingNetEncoder(nn.Module):
    """
    FoldingNet encoder adapted for 10-dim particle detector input.

    Input : (B, N, 10)  — (row_norm, col_norm, ch0..ch7)
    Output: (B, 512)    — codeword theta

    Changes from original paper (3-dim XYZ):
        - KNN uses 2D spatial coords (row, col) only
        - Local covariance on pseudo-3D (row, col, sum_channels) -> 9 dims
        - mlp1 input: 10 + 9 = 19  [original: 3 + 9 = 12]
        - Everything else identical
    """

    def __init__(self, k=16, codeword_dim=512, in_dim=10):
        super().__init__()
        self.k      = k
        self.in_dim = in_dim
        mlp1_in     = in_dim + 9                          # 19

        self.mlp1   = make_mlp([mlp1_in, 64, 64, 64], last_relu=True)
        self.graph1 = GraphMaxPool(64,   512,  k=k)
        self.graph2 = GraphMaxPool(512,  1024, k=k)
        self.mlp2   = make_mlp([1024, codeword_dim], last_relu=False)

    def forward(self, x):
        """x : (B, N, 10)  ->  theta : (B, 512)"""
        B, N, D = x.shape

        idx  = knn(x, self.k)                             # (B, N, K)
        cov  = local_covariance(x, idx)                   # (B, N, 9)
        feat = torch.cat([x, cov], dim=-1)                # (B, N, 19)
        feat = self.mlp1(feat.reshape(B * N, -1))         # (B*N, 64)
        feat = feat.reshape(B, N, 64)
        feat = self.graph1(feat, idx)                     # (B, N, 512)
        feat = self.graph2(feat, idx)                     # (B, N, 1024)
        feat = feat.max(dim=1).values                     # (B, 1024)
        return self.mlp2(feat)                            # (B, 512)


if __name__ == '__main__':
    B, N = 2, 1024
    encoder = FoldingNetEncoder(k=16, codeword_dim=512, in_dim=10)
    encoder.eval()
    dummy = torch.randn(B, N, 10)
    with torch.no_grad():
        theta = encoder(dummy)
    print(f"Input  : {dummy.shape}")
    print(f"Output : {theta.shape}")
    assert theta.shape == (B, 512)
    print("Shape test passed.")
    print(f"Parameters: {sum(p.numel() for p in encoder.parameters()):,}")