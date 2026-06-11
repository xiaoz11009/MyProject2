"""PhysUnfolder: GCN + FiLM + dual-head with curvature-gated relaxation.

Architecture (from DOCX):
  1. GCN encoder (3 layers, PyG GCNConv) — local geometry on mesh edges
  2. Material FiLM — modulate features by fabric params [E, B, S]
  3. Geo Head — regress base 2D coordinates U_geo
  4. Relax Head + Curvature Gate — curvature-gated compensation delta
  5. Output: U_geo + delta

Input:  (N,3) vertices, (2,E) edge_index, (N,2) curvature, (3,) material
Output: (N,2) 2D coordinates, (N,2) delta (for visualization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class PhysUnfolder(nn.Module):
    """GCN-based physics-constrained panel unfolding network."""

    def __init__(self, in_dim=3, dim=256):
        super().__init__()
        # GCN encoder: 3 layers with skip connections to prevent over-smoothing
        self.conv1 = GCNConv(in_dim, 128)
        self.conv2 = GCNConv(128, dim)
        self.conv3 = GCNConv(dim, dim)
        self.skip_proj = nn.Linear(in_dim, dim)  # project input for skip

        # Material encoder: [E, B, S] → FiLM (gamma, beta)
        self.mat_encoder = nn.Sequential(
            nn.Linear(3, dim), nn.ReLU(),
            nn.Linear(dim, dim * 2),
        )

        # Geo head: features → base 2D coordinates
        self.geo_head = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )

        # Relax head: features → compensation delta
        self.relax_head = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )

        # Curvature gate: learnable sigmoid parameters
        # Curvature gate: fixed (non-learnable) to prevent model from cheating
        # by opening the gate everywhere. Values chosen so:
        #   K_norm=0 (flat) → gate≈0.08 (92% membrane)  K_norm=1 (peak) → gate≈0.92 (8% membrane)
        self.register_buffer('curv_gamma', torch.tensor(5.0))
        self.register_buffer('curv_beta', torch.tensor(-2.5))

    def forward(self, x, edge_index, curv, mat):
        # 1. GCN encoding with skip connections (prevent over-smoothing)
        x_proj = self.skip_proj(x)               # (N, 3) → (N, dim)
        h = F.relu(self.conv1(x, edge_index))    # (N, 128)
        h = F.relu(self.conv2(h, edge_index))    # (N, dim)
        h = h + x_proj                           # skip: raw position info preserved
        h = self.conv3(h, edge_index) + h        # residual: dim→dim

        # 2. Material FiLM modulation
        mat_feat = self.mat_encoder(mat)         # (dim*2,)
        gamma, beta = mat_feat.chunk(2, dim=-1)  # (dim,), (dim,)
        h = gamma * h + beta                     # (N, dim)

        # 3. Geo head
        U_geo = self.geo_head(h)  # (N, 2)

        # 4. Relax head + curvature gate (panel-normalized so gate actually discriminates)
        delta_raw = self.relax_head(h)           # (N, 2)
        K_abs = curv[:, 0:1].abs()               # (N, 1) |gaussian|
        K_min, K_max = K_abs.min(), K_abs.max()
        K_norm = (K_abs - K_min) / (K_max - K_min).clamp(min=1e-6)
        gate = torch.sigmoid(self.curv_gamma * K_norm + self.curv_beta)
        delta = delta_raw * gate

        U = U_geo + delta
        return U, delta


# ====================== Inference helper ======================

def unfold_panels_fast(model, panels, device='cuda', default_mat=None):
    """Fast inference with PhysUnfolder.

    Args:
        model: PhysUnfolder
        panels: list of dicts from cut_mesh_into_panels
        device: 'cuda' or 'cpu'
        default_mat: (3,) default material [E, B, S] if panel has none

    Returns:
        list of (N, 2) numpy arrays — 2D coordinates per panel
    """
    from curvature_utils import compute_curvature, mesh_faces_to_edges

    if default_mat is None:
        default_mat = torch.tensor([1.0, 0.5, 0.3])

    model.eval()
    results = []
    with torch.no_grad():
        for p in panels:
            v = p['vertices'].to(device)
            f = p['faces'].to(device)
            if v.shape[0] < 3 or f.shape[0] < 1:
                results.append(v[:, :2].cpu().numpy())
                continue

            # Build edge_index from faces
            edge_index = mesh_faces_to_edges(f)

            # Compute curvature on original vertices
            curv = compute_curvature(v, f)

            # Material (use panel's if stored, else default)
            mat = p.get('material', default_mat)
            if isinstance(mat, (list, tuple)):
                mat = torch.tensor(mat, dtype=torch.float32)
            mat = mat.to(device)

            # Normalize input
            v_center = v.mean(dim=0, keepdim=True)
            v_norm = v - v_center
            scale = v_norm.norm(dim=-1).max().clamp(min=1e-6)

            U, _ = model(v_norm / (scale + 1e-8), edge_index, curv, mat)

            # Model output is in normalized PCA-aligned 2D frame (zero-centered, unit scale).
            # Denormalize: scale by 3D extent (≈ 2D extent for developable panels).
            # Do NOT add v_center[:,:2] — model output frame ≠ world XY frame.
            U = U * (scale + 1e-8)
            results.append(U.cpu().numpy())
    return results
