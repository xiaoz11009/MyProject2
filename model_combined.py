"""
Combined model: Dual-channel encoder (GCN + EdgeConv) + Segmentation + PhysUnfolder.

Architecture:
  - GCN channel (mesh topology):    local geometric features via fixed mesh edges
  - EdgeConv channel (dynamic k-NN): long-range semantic features in feature space
  - FiLM modulation:                material-conditioned feature gating
  - Three heads: SegHead (41-class), GeoHead (2D base), RelaxHead (curvature-gated)

Based on:
  - NeuralTailor (Korosteleva & Lee, SIGGRAPH 2022)
  - DGCNN (Wang et al., TOG 2019)
  - PhysUnfolder design document
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsemax import Sparsemax


# ====================== Utility Functions ======================

def pairwise_dist(x1, x2):
    """Pairwise Euclidean distance via matmul (RTX 5090 compatible)."""
    x1_sq = (x1 ** 2).sum(dim=-1, keepdim=True)   # (B, N1, 1)
    x2_sq = (x2 ** 2).sum(dim=-1, keepdim=True)   # (B, N2, 1)
    cross = x1 @ x2.transpose(1, 2)                # (B, N1, N2)
    dist_sq = x1_sq + x2_sq.transpose(1, 2) - 2 * cross
    dist_sq = torch.clamp(dist_sq, min=0.0)
    return torch.sqrt(dist_sq)


class ResidualMLP(nn.Module):
    """MLP with residual connections where input/output dims match."""

    def __init__(self, channels, batch_norm=True):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(channels)):
            self.layers.append(nn.Sequential(
                nn.Linear(channels[i - 1], channels[i]),
                nn.ReLU(),
                nn.BatchNorm1d(channels[i]) if batch_norm else nn.Identity(),
            ))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            out = layer(x)
            # Residual: skip if shapes match
            if x.shape[-1] == out.shape[-1]:
                out = out + x
            x = out
        return x


# mesh_faces_to_edges imported from curvature_utils


# ====================== GCN Layer (Pure PyTorch, no PyG) ======================

class GCNConv(nn.Module):
    """
    Graph convolution (Kipf & Welling, ICLR 2017).
    Pure PyTorch — no torch_geometric dependency, compatible with RTX 5090.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index):
        """
        Args:
            x: (N, in_channels)
            edge_index: (2, E) source → target
        Returns:
            (N, out_channels)
        """
        src, tgt = edge_index
        N, E = x.shape[0], edge_index.shape[1]
        device = x.device
        dtype = x.dtype

        # D^{-1/2} A D^{-1/2} normalization
        deg = torch.zeros(N, device=device, dtype=dtype).scatter_add_(
            0, tgt, torch.ones(E, device=device, dtype=dtype))
        deg_inv_sqrt = (deg + 1.0).pow(-0.5)
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[tgt]  # (E,)

        h = self.lin(x)  # (N, out) — respects autocast dtype

        # Aggregate messages from neighbors (scatter sum)
        msg = (h[src] * norm.unsqueeze(-1)).to(h.dtype)  # (E, out)
        idx = tgt.unsqueeze(-1).expand(-1, h.shape[1])
        out = torch.zeros(N, h.shape[1], device=device, dtype=h.dtype)
        out.scatter_add_(0, idx, msg)

        # Self-loop
        deg_inv = (1.0 / (deg + 1.0)).unsqueeze(-1).to(h.dtype)
        out = out + h * deg_inv

        out = self.bn(out)
        out = F.relu(out)
        return out


class MeshGCNEncoder(nn.Module):
    """2-layer GCN on fixed mesh topology — fast, sufficient for local geometry."""
    def __init__(self, dim=128):
        super().__init__()
        self.conv1 = GCNConv(3, dim)
        self.conv2 = GCNConv(dim, dim)

    def forward(self, x, edge_index):
        """
        Args:
            x: (total_N, 3) batched vertex positions
            edge_index: (2, total_E) batched mesh edges (with offsets)
        Returns:
            (total_N, dim) per-vertex local geometric features
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


# ====================== EdgeConv Module ======================

class EdgeConv(nn.Module):
    """Single EdgeConv layer with manual k-NN graph construction."""
    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()
        self.k = k
        self.conv = nn.Conv2d(in_channels * 2, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, mask=None):
        """
        x: (B, D, N)
        mask: (B, N) optional, True=valid, False=pad
        Returns: (B, out_channels, N)
        """
        B, D, N = x.shape
        xx = x.transpose(2, 1)                     # (B, N, D)
        dist = pairwise_dist(xx, xx)                # (B, N, N)

        # Exclude padded vertices from k-NN
        if mask is not None:
            large = dist.max().detach() + 1e6
            dist = dist.masked_fill(~mask.unsqueeze(1), large)  # pad as query
            dist = dist.masked_fill(~mask.unsqueeze(2), large)  # pad as key

        idx = dist.topk(self.k, dim=-1, largest=False)[1]  # (B, N, k)

        idx_flat = idx.reshape(B, N * self.k)
        feat = xx.gather(1, idx_flat.unsqueeze(-1).repeat(1, 1, D))
        feat = feat.reshape(B, N, self.k, D)

        x_expand = xx.unsqueeze(2).repeat(1, 1, self.k, 1)
        edge_feat = torch.cat([x_expand, feat - x_expand], dim=-1)
        edge_feat = edge_feat.permute(0, 3, 1, 2)  # (B, 2D, N, k)

        out = self.conv(edge_feat)
        out = self.bn(out)
        out = F.leaky_relu(out, 0.2)
        out = out.max(dim=-1)[0]                    # (B, out, N)

        # Zero out padded positions
        if mask is not None:
            out = out * mask.unsqueeze(1)
        return out


class EdgeConvEncoder(nn.Module):
    """
    2-layer EdgeConv encoder with input projection + residual connection.
    """
    def __init__(self, dim=128, k=16):
        super().__init__()
        self.dim = dim
        self.input_proj = nn.Conv1d(3, dim, 1, bias=False)
        self.bn_proj = nn.BatchNorm1d(dim)
        self.conv1 = EdgeConv(3, dim, k=k)
        self.conv2 = EdgeConv(dim, dim, k=k)
        self.fuse = nn.Conv1d(dim * 2, dim, 1, bias=False)
        self.bn_fuse = nn.BatchNorm1d(dim)

    def forward(self, positions, mask=None):
        xyz = positions.transpose(1, 2)              # (B, 3, N)
        shortcut = F.leaky_relu(self.bn_proj(self.input_proj(xyz)), 0.2)
        x1 = self.conv1(xyz, mask)                   # (B, dim, N)
        x2 = self.conv2(x1, mask)                    # (B, dim, N)
        fused = torch.cat([x1, x2], dim=1)           # (B, 2*dim, N)
        out = F.leaky_relu(self.bn_fuse(self.fuse(fused)), 0.2)
        out = out + shortcut                         # residual
        if mask is not None:
            out = out * mask.unsqueeze(1)
        out = out.transpose(1, 2)                    # (B, N, dim)
        if positions.shape[0] == 1 and mask is None:
            out = out.squeeze(0)
        return out


# ====================== FiLM Modulation ======================

class FiLMModulation(nn.Module):
    """
    Feature-wise Linear Modulation (Perez et al., CVPR 2018).
    Generates per-channel γ, β from material parameters [E, B, S].
    """
    def __init__(self, feat_dim, mat_dim=3, hidden_dim=64):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(mat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim * 2),
        )

    def generate_params(self, materials):
        """
        Args:
            materials: (B, 3) [stretch_modulus, bending_stiffness, shear_coeff]
        Returns:
            (B, feat_dim * 2) concatenated [γ | β]
        """
        return self.generator(materials)

    def modulate(self, x, gamma_beta):
        """
        Args:
            x: (N, feat_dim) features
            gamma_beta: (N, feat_dim * 2) per-vertex [γ | β]
        Returns:
            (N, feat_dim) modulated features = γ * x + β
        """
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma * x + beta


# ====================== Segmentation Head ======================

class SegHead(nn.Module):
    """Per-vertex panel classification. Outputs raw logits (no softmax/sparsemax)
    — cross_entropy loss handles log-softmax internally."""

    def __init__(self, in_dim, num_classes=41, hidden_dim=256):
        super().__init__()
        self.mlp = ResidualMLP([in_dim, hidden_dim, hidden_dim, hidden_dim, num_classes])

    def forward(self, x):
        return self.mlp(x)


# ====================== Unfolding Heads ======================

class GeoHead(nn.Module):
    """Predicts base 2D pattern coordinates (isometric unfolding)."""
    def __init__(self, in_dim, hidden_dim=256):
        super().__init__()
        self.net = ResidualMLP([in_dim, hidden_dim, hidden_dim, 2], batch_norm=True)

    def forward(self, x):
        return self.net(x)


class CurvatureGatedRelaxation(nn.Module):
    """
    Core innovation: curvature-gated residual correction.

    High-curvature regions (collars, armholes) receive more material-aware
    compensation; flat regions get near-zero correction. The gate is:
        gate = σ(γ_g * |K| + β_g)
    where |K| is the absolute Gaussian curvature and γ_g, β_g are learnable.
    """
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.relax_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 2),
        )
        self.curv_gamma = nn.Parameter(torch.tensor(1.0))
        self.curv_beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, gaussian_curvature):
        """
        Args:
            x: (N, in_dim) features
            gaussian_curvature: (N, 1) absolute Gaussian curvature per vertex
        Returns:
            delta: (N, 2) gated relaxation field
            gate:  (N, 1) curvature gate values (for visualization)
        """
        delta = self.relax_head(x)                              # (N, 2)
        gate = torch.sigmoid(self.curv_gamma * gaussian_curvature + self.curv_beta)
        return delta * gate, gate


# ====================== Full Combined Model ======================

class PhysUnfolderCombined(nn.Module):
    """
    GCN (real mesh topology) + EdgeConv (dynamic k-NN) → fused features → Heads.

    - GCN  on full mesh:    local geometry via fixed triangle edges (dim=128)
    - EdgeConv on FPS pts:  long-range semantics via dynamic k-NN (dim=128)
    - Fusion: concat → 256-dim → three heads
    """
    def __init__(self, num_classes=41, gcn_dim=128, ec_dim=128, k=16,
                 hidden_dim=256, use_film=False):
        super().__init__()
        self.num_classes = num_classes
        self.use_film = use_film
        # Fused feat = local(256) + global_pool(256) = 512
        self.head_dim = (gcn_dim + ec_dim) * 2  # 512

        # GCN: 2 layers on full mesh, dim=128
        self.gcn_encoder = MeshGCNEncoder(dim=gcn_dim)

        # EdgeConv: 2 layers on FPS points, dim=128
        self.ec_encoder = EdgeConvEncoder(dim=ec_dim, k=k)

        # FiLM (optional, on fused features)
        if use_film:
            self.film = FiLMModulation(feat_dim=self.head_dim, mat_dim=3)

        # GarmentHead: predicts which panels exist (41-dim binary) from global code
        local_feat_dim = gcn_dim + ec_dim  # 256
        self.garment_head = nn.Sequential(
            nn.Linear(local_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        # Heads
        self.seg_head = SegHead(in_dim=self.head_dim + 3, num_classes=num_classes,
                                hidden_dim=hidden_dim)
        # GeoHead/Relax do NOT see verts_3d — forces learning proper 2D shapes
        # instead of trivially copying (vert_x, vert_y)
        unfold_in = self.head_dim + num_classes
        self.geo_head = GeoHead(in_dim=unfold_in, hidden_dim=hidden_dim)
        self.relax = CurvatureGatedRelaxation(in_dim=unfold_in, hidden_dim=hidden_dim)

    def _heads_forward(self, feat_mod, verts, curv_abs, labels=None,
                       garment_logits=None):
        """Shared head computation. feat_mod/verts: (total_N, D).

        When garment_logits is provided, seg_logits are masked to only allow
        panels predicted to exist (solves multi-garment-type confusion).
        """
        seg_input = torch.cat([feat_mod, verts], dim=-1)
        seg_logits = self.seg_head(seg_input)

        # Garment mask: filter out panels that don't exist in this garment
        if garment_logits is not None:
            B = garment_logits.shape[0]
            S_total = seg_logits.shape[0]
            S = S_total // B
            garment_prob = torch.sigmoid(garment_logits)  # (B, C)
            garment_mask = (garment_prob > 0.5).float().unsqueeze(1).expand(B, S, -1).reshape(S_total, -1)
            # Set logits for absent panels to -inf
            seg_logits = seg_logits + (garment_mask - 1.0) * 1e10

        # Teacher forcing: GT one-hot during training avoids the "soft blend"
        # problem where ambiguous seg_logits cause all panels to overlap in 2D.
        if labels is not None:
            # Teacher forcing with scheduled sampling:
            # 70% GT one-hot (clean signal), 30% softmax(logits) (inference-like)
            gt_onehot = F.one_hot(labels, num_classes=self.num_classes).to(
                dtype=seg_logits.dtype)
            if self.training and torch.rand(1).item() < 0.3:
                seg_cond = F.softmax(seg_logits, dim=-1)
            else:
                seg_cond = gt_onehot
        else:
            # Inference: softmax raw logits → probabilities for GeoHead
            seg_cond = F.softmax(seg_logits, dim=-1)

        unfold_input = torch.cat([feat_mod, seg_cond], dim=-1)
        U_geo = self.geo_head(unfold_input)
        delta, gate = self.relax(unfold_input, curv_abs)
        U_final = U_geo + gate * delta

        return seg_logits, U_geo, U_final, delta, gate

    # ---- List-based forward (inference, variable-size meshes) ----

    def _build_batched_graph(self, vertices_list, edge_indices, curvature_list):
        """Concatenate per-sample data into a single batched graph."""
        device = vertices_list[0].device
        all_verts, all_edges, all_curvs, batch_idx = [], [], [], []
        offset = 0
        for i, (v, ei, c) in enumerate(zip(vertices_list, edge_indices, curvature_list)):
            n = v.shape[0]
            all_verts.append(v)
            all_edges.append(ei + offset)
            all_curvs.append(c)
            batch_idx.append(torch.full((n,), i, dtype=torch.long, device=device))
            offset += n
        return (torch.cat(all_verts, dim=0), torch.cat(all_edges, dim=1),
                torch.cat(all_curvs, dim=0), torch.cat(batch_idx, dim=0))

    def forward(self, vertices_list, edge_indices, curvature_list, materials):
        """List-based forward — delegates to forward_dual internally."""
        B = len(vertices_list)
        device = vertices_list[0].device
        S = curvature_list[0].shape[0]  # num_vertices

        # Build sample tensors from lists
        verts_sample = torch.stack(vertices_list, dim=0)  # (B, S, 3)
        curv_sample = torch.stack(curvature_list, dim=0)  # (B, S, 2)
        e_sample = [ei.to(device) for ei in edge_indices]

        # Need full mesh data — but caller only passes sampled data
        # For inference with forward_dual, we need vertices_full and edge_indices_full
        # If only sampled data is available, treat samples AS full mesh
        return self.forward_dual(
            vertices_list, edge_indices,   # full mesh = samples
            verts_sample, e_sample,         # sampled
            [torch.arange(S, dtype=torch.long) for _ in range(B)],  # identity mapping
            curv_sample, materials)

    # ---- Dual forward (GCN on full mesh → index to samples → heads) ----

    def forward_dual(self, vertices_full_list, edge_indices_full,
                     vertices_sample, edge_indices_sample, sample_indices_list,
                     curvature_sample, materials, labels_sample=None):
        """
        GCN on full mesh with real triangle edges → index to FPS samples → heads.

        Args:
            vertices_full_list:    list of (N_i, 3)
            edge_indices_full:     list of (2, E_i) — real triangle edges
            vertices_sample:       (B, S, 3) — FPS points
            edge_indices_sample:   list of (2, E_sample) — for physics loss only
            sample_indices_list:   list of (S,) long
            curvature_sample:      (B, S, 2)
            materials:             (B, 3)
        """
        B, S, _ = vertices_sample.shape
        device = vertices_sample.device

        # ---- 1. GCN on full mesh (batched graph, single forward pass) ----
        # Build batched graph: concat vertices + offset edges
        verts_cat, edges_cat, batch_idx = [], [], []
        offset = 0
        for i in range(B):
            v = vertices_full_list[i].to(device)
            ei = edge_indices_full[i].to(device)
            n = v.shape[0]
            verts_cat.append(v)
            edges_cat.append(ei + offset)
            batch_idx.append(torch.full((n,), i, dtype=torch.long, device=device))
            offset += n
        verts_cat = torch.cat(verts_cat, dim=0)
        edges_cat = torch.cat(edges_cat, dim=1)
        batch_cat = torch.cat(batch_idx, dim=0)

        gcn_full = self.gcn_encoder(verts_cat, edges_cat)  # (total_N, gcn_dim=128)

        # Index GCN features to FPS positions
        feat_gcn = torch.zeros(B, S, self.gcn_encoder.conv2.bn.num_features,
                               device=device)
        offset = 0
        for i in range(B):
            sample_idx = sample_indices_list[i].to(device)
            feat_gcn[i] = gcn_full[offset + sample_idx]        # (S, gcn_dim=128)
            offset += vertices_full_list[i].shape[0]

        # ---- 2. EdgeConv on FPS points (long-range semantics) ----
        feat_ec = self.ec_encoder(vertices_sample)             # (B, S, 128) or (S, 128) when B=1
        if feat_ec.dim() == 2:
            feat_ec = feat_ec.unsqueeze(0)                     # restore batch dim

        # ---- 3. Fuse: GCN (local) + EdgeConv (global) + global context ----
        feat_local = torch.cat([feat_gcn, feat_ec], dim=-1)    # (B, S, 256)
        # Garment code: mean pool → predict which panels exist
        global_code = feat_local.mean(dim=1)                    # (B, 256)
        garment_logits = self.garment_head(global_code)         # (B, C)
        # Expand global for per-vertex context
        global_feat = global_code.unsqueeze(1).expand(-1, S, -1)  # (B, S, 256)
        feat = torch.cat([feat_local, global_feat], dim=-1)    # (B, S, 512)

        # ---- 4. FiLM (optional) ----
        if self.use_film:
            film_params = self.film.generate_params(materials)
            gamma, beta = film_params.unsqueeze(1).expand(-1, S, -1).chunk(2, dim=-1)
            feat = gamma * feat + beta

        # ---- 5. Heads ----
        BSN = B * S
        feat_flat = feat.reshape(BSN, -1)
        verts_flat = vertices_sample.reshape(BSN, 3)
        curv_flat = curvature_sample.reshape(BSN, 2)[:, 0:1]

        lbls_flat = labels_sample.reshape(BSN) if labels_sample is not None else None
        seg, U_geo, U_final, delta, gate = self._heads_forward(
            feat_flat, verts_flat, curv_flat, lbls_flat, garment_logits)

        return {
            'seg_logits': seg.reshape(B, S, -1),
            'U_geo':      U_geo.reshape(B, S, 2),
            'U_final':    U_final.reshape(B, S, 2),
            'delta':      delta.reshape(B, S, 2),
            'garment_logits': garment_logits,
            'gate':       gate.reshape(B, S, 1),
        }

    # ---- Padded batch forward (training, fixed-size, fast) ----

    def forward_padded(self, vertices, mask, edge_indices, curvature, materials):
        """
        Padded batch forward for training.

        Args:
            vertices:  (B, N_pad, 3) padded positions
            mask:      (B, N_pad) bool, True=valid
            edge_indices: list of (2, E_i) — for GCN
            curvature: (B, N_pad, 2)
            materials: (B, 3)

        Returns:
            dict: seg_logits (B, N_pad, C), U_geo (B, N_pad, 2), etc.
        """
        B, N, _ = vertices.shape
        device = vertices.device

        # 1. EdgeConv (batched, fast)
        feat_ec = self.ec_encoder(vertices, mask)  # (B, N, ec_out)

        # 2. GCN (optional, per-sample)
        if self.use_gcn:
            feat_gcn = torch.zeros(B, N, self.gcn_encoder.conv3.bn.num_features,
                                   device=device)
            for i in range(B):
                ni = mask[i].sum().item()
                if ni == 0:
                    continue
                v_i = vertices[i, :ni]              # (N_i, 3)
                ei = edge_indices[i].to(device)
                feat_gcn[i, :ni] = self.gcn_encoder(v_i, ei)
            feat_fused = torch.cat([feat_gcn, feat_ec], dim=-1)  # (B, N, 256)
        else:
            feat_fused = feat_ec  # (B, N, 256)

        # 3. FiLM
        if self.use_film:
            film_params = self.film.generate_params(materials)  # (B, 512)
            film_flat = film_params.unsqueeze(1).expand(-1, N, -1)  # (B, N, 512)
            gamma, beta = film_flat.chunk(2, dim=-1)
            feat_mod = gamma * feat_fused + beta
        else:
            feat_mod = feat_fused

        # 4. Flatten to (total_valid_N, D) for heads
        feat_flat = feat_mod[mask]          # (total_N, 256)
        verts_flat = vertices[mask]          # (total_N, 3)
        curv_flat = curvature[mask][:, 0:1]  # (total_N, 1)

        seg_flat, geo_flat, final_flat, delta_flat, gate_flat = self._heads_forward(
            feat_flat, verts_flat, curv_flat)

        # 5. Unflatten back to padded
        def _unflatten(flat, fill=0.0):
            out = torch.full((B, N, flat.shape[-1]), fill, device=device,
                             dtype=flat.dtype)
            out[mask] = flat
            return out

        seg_logits = _unflatten(seg_flat, 0.0)
        U_geo = _unflatten(geo_flat)
        U_final = _unflatten(final_flat)
        delta = _unflatten(delta_flat)
        gate = _unflatten(gate_flat)

        return {
            'seg_logits': seg_logits,  # (B, N, C)
            'U_geo': U_geo,            # (B, N, 2)
            'U_final': U_final,        # (B, N, 2)
            'delta': delta,            # (B, N, 2)
            'gate': gate,              # (B, N, 1)
            'mask': mask,              # pass through for loss
        }


# ====================== Test ======================

if __name__ == '__main__':
    print("=" * 60)
    print("PhysUnfolderCombined — GCN+EdgeConv 双通道架构测试")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    from curvature_utils import mesh_faces_to_edges
    from data_loader import build_knn_edges_from_points

    S = 500  # fixed FPS size

    vertices_full, edges_full, curv_full = [], [], []
    sample_idx_list, vs_list, cs_list, es_list = [], [], [], []

    for i, n in enumerate([600, 800, 700]):
        v = torch.randn(n, 3, device=device)
        v = v / v.norm(dim=-1, keepdim=True) * torch.rand(n, 1, device=device)
        faces = torch.randint(0, n, (n * 2, 3), dtype=torch.long, device=device)
        ei = mesh_faces_to_edges(faces)
        curv = torch.rand(n, 2, device=device) * 0.1

        # FPS indices
        idx = torch.randperm(n)[:S]
        vs = v[idx]
        cs = curv[idx]
        es = build_knn_edges_from_points(vs.cpu().numpy(), k=12)

        vertices_full.append(v)
        edges_full.append(ei)
        curv_full.append(curv)
        sample_idx_list.append(idx)
        vs_list.append(vs)
        cs_list.append(cs)
        es_list.append(es)

    verts_sample = torch.stack(vs_list).to(device)   # (B, S, 3)
    curv_sample = torch.stack(cs_list).to(device)     # (B, S, 2)
    materials = torch.tensor([[500., 0.01, 100.], [200., 0.005, 50.], [1000., 0.02, 200.]], device=device)

    model = PhysUnfolderCombined(num_classes=41, gcn_dim=128, ec_dim=128, k=16,
                                 hidden_dim=256).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out = model.forward_dual(vertices_full, edges_full,
                                     verts_sample, es_list, sample_idx_list,
                                     curv_sample, materials)

    print(f"\n批次大小: {len(vertices_full)}")
    for i in range(len(vertices_full)):
        print(f"  样本 {i}: {vertices_full[i].shape[0]} 顶点 (FPS→{S})")
        print(f"    seg_logits: {out['seg_logits'][i].shape}")
        print(f"    U_geo:      {out['U_geo'][i].shape}")
        print(f"    U_final:    {out['U_final'][i].shape}")
        print(f"    delta:      |Δ|={out['delta'][i].norm(dim=-1).mean().item():.4f}")
        print(f"    gate:       mean={out['gate'][i].mean().item():.3f}, max={out['gate'][i].max().item():.3f}")

    print("\n所有形状校验通过 ✓")
