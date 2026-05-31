"""
NeuralTailor-based model for garment panel segmentation.
Architecture: EdgeConv (DGCNN backbone) + Global Context + Point Segmentation Head.

Based on "NeuralTailor: Reconstructing Sewing Pattern Structures from 3D Point Clouds
of Garments" (Korosteleva & Lee, SIGGRAPH 2022).

All operations use plain PyTorch (no torch_geometric) for RTX 5090 compatibility.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsemax import Sparsemax


# ====================== Distance helper (matmul-based, works on RTX 5090) ======================

def pairwise_dist(x1, x2):
    """Compute pairwise Euclidean distance using matmul (avoids torch.cdist which
    uses a CUDA kernel not compiled for sm_120 on older PyTorch)."""
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    x1_sq = (x1 ** 2).sum(dim=-1, keepdim=True)   # (B, N1, 1)
    x2_sq = (x2 ** 2).sum(dim=-1, keepdim=True)   # (B, N2, 1)
    cross = x1 @ x2.transpose(1, 2)                # (B, N1, N2)
    dist_sq = x1_sq + x2_sq.transpose(1, 2) - 2 * cross
    dist_sq = torch.clamp(dist_sq, min=0.0)
    return torch.sqrt(dist_sq)


# ====================== MLP helper ======================

def MLP(channels, batch_norm=True):
    return nn.Sequential(*[
        nn.Sequential(
            nn.Linear(channels[i - 1], channels[i]),
            nn.ReLU(),
            nn.BatchNorm1d(channels[i]) if batch_norm else nn.Identity(),
        )
        for i in range(1, len(channels))
    ])


# ====================== EdgeConv Layer ======================

class EdgeConv(nn.Module):
    """
    Single EdgeConv layer with manual k-NN graph construction.
    Same implementation as Baseline DGCNN's get_graph_feature — uses
    torch.cdist + topk which work on RTX 5090.
    """
    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()
        self.k = k
        self.conv = nn.Conv2d(in_channels * 2, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        x: (B, D, N) point features
        Returns: (B, out_channels, N)
        """
        B, D, N = x.shape
        # Pairwise distance (manual matmul implementation for RTX 5090 compat)
        xx = x.transpose(2, 1)                     # (B, N, D)
        dist = pairwise_dist(xx, xx)                # (B, N, N)
        idx = dist.topk(self.k, dim=-1, largest=False)[1]  # (B, N, k)

        # Gather k-NN features
        idx_flat = idx.reshape(B, N * self.k)
        feat = xx.gather(1, idx_flat.unsqueeze(-1).repeat(1, 1, D))  # (B, N*k, D)
        feat = feat.reshape(B, N, self.k, D)                         # (B, N, k, D)

        # Edge features: [center, neighbor - center]
        x_expand = xx.unsqueeze(2).repeat(1, 1, self.k, 1)           # (B, N, k, D)
        edge_feat = torch.cat([x_expand, feat - x_expand], dim=-1)   # (B, N, k, 2D)
        edge_feat = edge_feat.permute(0, 3, 1, 2)                    # (B, 2D, N, k)

        out = self.conv(edge_feat)                                    # (B, out, N, k)
        out = self.bn(out)
        out = F.leaky_relu(out, 0.2)
        out = out.max(dim=-1)[0]                                      # (B, out, N)
        return out


# ====================== EdgeConv Backbone (NeuralTailor-style) ======================

class EdgeConvBackbone(nn.Module):
    """
    DGCNN-based feature extractor with NeuralTailor configuration:
      - 2 EdgeConv layers (128-dim features)
      - Skip connections with raw XYZ
      - Global mean pooling → global encoding
    """
    def __init__(self, out_size=256, feat_dim=128, k=16):
        super().__init__()
        self.feat_dim = feat_dim

        self.conv1 = EdgeConv(3, feat_dim, k=k)         # 3 → 128
        self.conv2 = EdgeConv(feat_dim, feat_dim, k=k)   # 128 → 128

        # Global feature: concat local features → linear
        local_dim = feat_dim * 2  # 256
        self.global_conv = nn.Conv1d(local_dim, out_size, 1, bias=False)
        self.bn_global = nn.BatchNorm1d(out_size)

    def forward(self, positions):
        """
        Args:
            positions: (B, N, 3) point cloud
        Returns:
            global_enc: (B, out_size) global feature
            point_feat: (B, feat_dim+3, N) per-point features (with skip XYZ)
            xyz: (B, 3, N) raw coordinates
        """
        B, N, _3 = positions.shape
        xyz = positions.transpose(1, 2)                 # (B, 3, N)

        x1 = self.conv1(xyz)                            # (B, 128, N)
        x2 = self.conv2(x1)                             # (B, 128, N)

        local = torch.cat([x1, x2], dim=1)              # (B, 256, N)

        # Global encoding (NeuralTailor: mean pool + linear)
        g = F.leaky_relu(self.bn_global(self.global_conv(local)), 0.2)
        g = g.mean(dim=-1)                               # (B, out_size)

        # Per-point features (with skip connection)
        point_feat = torch.cat([local, xyz], dim=1)      # (B, 256+3, N)

        return g, point_feat, xyz


# ====================== Point Segmentation Head ======================

class PointSegHead(nn.Module):
    """
    Point-level segmentation head (NeuralTailor-style).
    Takes per-point features + global encoding + XYZ → per-point panel scores.

    Uses Sparsemax for sparse, attention-like predictions.
    """
    def __init__(self, point_feat_dim, global_feat_dim, num_classes, hidden_dim=256):
        super().__init__()
        in_dim = point_feat_dim + global_feat_dim + 3  # local + global + xyz
        self.mlp = nn.Sequential(
            MLP([in_dim, hidden_dim, hidden_dim, hidden_dim, num_classes]),
            Sparsemax(dim=1),
        )

    def forward(self, point_feat, global_enc, xyz):
        """
        Args:
            point_feat: (B, point_feat_dim, N) per-point features
            global_enc: (B, global_feat_dim) global features
            xyz: (B, 3, N) raw coordinates
        Returns:
            seg: (B, num_classes, N) per-point panel scores
        """
        B, _, N = point_feat.shape
        # Broadcast global encoding
        global_per_point = global_enc.unsqueeze(-1).repeat(1, 1, N)  # (B, global_dim, N)
        # Concatenate
        feat = torch.cat([point_feat, global_per_point, xyz], dim=1)  # (B, in_dim, N)
        feat = feat.transpose(1, 2).reshape(-1, feat.shape[1])         # (B*N, in_dim)
        seg = self.mlp(feat)                                            # (B*N, num_classes)
        seg = seg.view(B, N, -1).transpose(1, 2)                       # (B, num_classes, N)
        return seg


# ====================== Full NeuralTailorSeg Model ======================

class NeuralTailorSeg(nn.Module):
    """
    NeuralTailor-based garment panel segmentation model.

    Architecture:
      1. EdgeConvBackbone: 2-layer EdgeConv + global mean pool
      2. PointSegHead: per-point classification with global context

    Based on the segmentation mechanism from NeuralTailor's GarmentSegmentPattern3D,
    adapted for fixed-category panel classification.
    """
    def __init__(self, num_classes=41, feat_dim=128, global_dim=256, hidden_dim=256, k=16):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = EdgeConvBackbone(out_size=global_dim, feat_dim=feat_dim, k=k)

        point_feat_dim = feat_dim * 2 + 3  # local (256) + xyz (3)
        self.seg_head = PointSegHead(
            point_feat_dim=point_feat_dim,
            global_feat_dim=global_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
        )

    def forward(self, positions):
        """
        Args:
            positions: (B, N, 3) point cloud
        Returns:
            seg_logits: (B, N, num_classes) per-point segmentation logits
        """
        global_enc, point_feat, xyz = self.backbone(positions)
        seg = self.seg_head(point_feat, global_enc, xyz)        # (B, num_classes, N)
        seg = seg.transpose(1, 2)                                 # (B, N, num_classes)
        return seg
