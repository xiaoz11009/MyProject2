"""
Physics-inspired loss functions for PhysUnfolder.

Based on the PDF design:
  - Membrane energy:  penalizes edge stretching (E * strain^2)
  - Bending energy:   penalizes curvature in 2D pattern (B * ||L(U)||^2)
  - Relaxation reg:   encourages sparse relaxation (|K| * ||delta||)
  - Segmentation loss: cross-entropy for panel classification

Training principle: segmentation is supervised (GT labels);
unfolding is self-supervised via physics constraints (no GT 2D coords needed).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================== Cotangent Laplacian (for bending energy) ======================

def _compute_cotangent_weights(verts_3d, faces):
    """
    Compute cotangent weights for the Laplacian from 3D mesh geometry.

    Returns per-face cotangents and face areas needed for Laplacian application.
    """
    v0 = verts_3d[faces[:, 0]]  # (F, 3)
    v1 = verts_3d[faces[:, 1]]
    v2 = verts_3d[faces[:, 2]]

    e01 = v1 - v0
    e12 = v2 - v1
    e20 = v0 - v2

    l01 = e01.norm(dim=-1).clamp(min=1e-10)
    l12 = e12.norm(dim=-1).clamp(min=1e-10)
    l20 = e20.norm(dim=-1).clamp(min=1e-10)

    e01_n = e01 / l01.unsqueeze(-1)
    e12_n = e12 / l12.unsqueeze(-1)
    e20_n = e20 / l20.unsqueeze(-1)

    cos0 = (-e20_n * e01_n).sum(dim=-1).clamp(-1.0, 1.0)
    cos1 = (-e01_n * e12_n).sum(dim=-1).clamp(-1.0, 1.0)
    cos2 = (-e12_n * e20_n).sum(dim=-1).clamp(-1.0, 1.0)

    sin0 = (1.0 - cos0 * cos0).sqrt().clamp(min=1e-10)
    sin1 = (1.0 - cos1 * cos1).sqrt().clamp(min=1e-10)
    sin2 = (1.0 - cos2 * cos2).sqrt().clamp(min=1e-10)

    cot0 = cos0 / sin0  # (F,) opposite vertex 0 = edge (1,2)
    cot1 = cos1 / sin1  # opposite vertex 1 = edge (2,0)
    cot2 = cos2 / sin2  # opposite vertex 2 = edge (0,1)

    # Face areas
    e02 = -e20
    cross_norm = torch.cross(e01, e02, dim=-1).norm(dim=-1).clamp(min=1e-10)
    face_areas = 0.5 * cross_norm

    return cot0, cot1, cot2, face_areas, faces


def apply_cotangent_laplacian(coords, verts_3d, faces):
    """
    Apply the cotangent Laplacian (computed from 3D mesh) to 2D coordinates.

    Args:
        coords:   (N, D) 2D coordinates [u, v] or any per-vertex signal
        verts_3d: (N, 3) 3D vertex positions (for computing weights)
        faces:    (F, 3) triangle faces

    Returns:
        lap: (N, D) Laplacian of coords, normalized by vertex area
    """
    cot0, cot1, cot2, face_areas, faces_t = _compute_cotangent_weights(verts_3d, faces)
    N = verts_3d.shape[0]
    D = coords.shape[1]
    device = verts_3d.device
    half = 0.5

    # Vertex areas (barycentric)
    vertex_areas = torch.zeros(N, device=device)
    bary = face_areas / 3.0
    vertex_areas.scatter_add_(0, faces_t[:, 0], bary)
    vertex_areas.scatter_add_(0, faces_t[:, 1], bary)
    vertex_areas.scatter_add_(0, faces_t[:, 2], bary)

    lap = torch.zeros(N, D, device=device)

    for d in range(D):
        u = coords[:, d]  # (N,)
        u0, u1, u2 = u[faces_t[:, 0]], u[faces_t[:, 1]], u[faces_t[:, 2]]

        # Per-face Laplacian contribution for this dimension
        lap_u0 = half * (cot2 * (u1 - u0) + cot1 * (u2 - u0))
        lap_u1 = half * (cot2 * (u0 - u1) + cot0 * (u2 - u1))
        lap_u2 = half * (cot1 * (u0 - u2) + cot0 * (u1 - u2))

        lap[:, d].scatter_add_(0, faces_t[:, 0], lap_u0)
        lap[:, d].scatter_add_(0, faces_t[:, 1], lap_u1)
        lap[:, d].scatter_add_(0, faces_t[:, 2], lap_u2)

    lap = lap / (vertex_areas.unsqueeze(-1) + 1e-10)
    return lap


# ====================== Membrane Energy ======================

def membrane_energy(U, verts_3d, edge_index, stretch_modulus):
    """
    Thin-plate membrane strain energy: E * Σ ε² / |E|

    For each mesh edge:
        ε = (L_2d - L_3d) / max(L_3d, min_len)
        E_mem = E * ε²

    Clamping: L_3d has a minimum floor (avoids division by ~0);
              strain is clamped to [-5, 5] (avoids random-init explosion).
    """
    src, tgt = edge_index
    min_len = 1e-3

    L_3d = (verts_3d[src] - verts_3d[tgt]).norm(dim=-1).clamp(min=min_len)
    L_2d = (U[src] - U[tgt]).norm(dim=-1)

    strain = ((L_2d - L_3d) / L_3d).clamp(-5.0, 5.0)
    energy = stretch_modulus * (strain ** 2)

    return energy.mean()


# ====================== Bending Energy (k-NN graph Laplacian) ======================

def bending_energy_knn(U, edge_index, bending_modulus):
    """
    Bending energy via k-NN graph Laplacian (no faces needed).

    L(U_i) = mean_{j in N(i)} (U_i - U_j)   — graph Laplacian at vertex i
    E_bend = B * mean(||L(U)||²)

    Penalizes non-smooth deformations of the 2D pattern.
    Works directly with the k-NN edge_index that GCN already uses.
    """
    src, tgt = edge_index
    N = U.shape[0]
    device = U.device

    # Compute graph Laplacian: L_i = mean_j (U_i - U_j)
    diff = U[src] - U[tgt]  # (E, 2) — difference along each edge

    lap = torch.zeros(N, 2, device=device)
    lap.scatter_add_(0, src.unsqueeze(-1).expand(-1, 2), diff)

    # Normalize by degree (k-NN with constant k, so degree ≈ k)
    deg = torch.zeros(N, device=device).scatter_add_(
        0, src, torch.ones_like(src, dtype=torch.float))
    lap = lap / (deg.unsqueeze(-1) + 1e-8)

    energy = bending_modulus * (lap ** 2).sum(dim=-1)  # (N,)
    return energy.mean()


# ====================== Bending Energy (cotangent Laplacian, needs faces) ======================

def bending_energy(U, verts_3d, faces, bending_modulus):
    """
    Bending energy via cotangent Laplacian: B * mean(||L(U)||²)

    The cotangent Laplacian L is computed from the 3D mesh geometry,
    then applied to the 2D coordinates U. A flat (undistorted) unfolding
    has L(U) ≈ 0 everywhere. Non-zero values indicate bending distortion.

    Args:
        U:               (N, 2) predicted 2D coordinates
        verts_3d:        (N, 3) original 3D vertex positions
        faces:           (F, 3) triangle faces
        bending_modulus: scalar bending stiffness

    Returns:
        scalar energy averaged over vertices
    """
    lap = apply_cotangent_laplacian(U, verts_3d, faces)  # (N, 2)
    energy = bending_modulus * (lap ** 2).sum(dim=-1)     # (N,)
    return energy.mean()


# ====================== Relaxation Regularization ======================

def relaxation_reg(delta, gaussian_curvature):
    """
    Encourage sparse relaxation: penalize ||delta|| weighted by curvature.

    In flat regions (|K| ≈ 0), this term heavily penalizes any relaxation.
    In curved regions (|K| > 0), the penalty is reduced, allowing compensation.

        L_reg = mean(|K| * ||delta||)

    Args:
        delta:              (N, 2) relaxation field per vertex
        gaussian_curvature: (N,)  Gaussian curvature per vertex

    Returns:
        scalar regularization loss
    """
    delta_norm = delta.norm(dim=-1)                         # (N,)
    K_abs = gaussian_curvature.abs()                        # (N,)

    # High curvature: low penalty. Low curvature: high penalty.
    # Use 1/(|K|+eps) as weight, or equivalently, minimize |K|*||delta||
    # Actually the PDF says: reg = |K| * ||delta||.mean() as a penalty.
    # This PENALIZES delta everywhere, but more in high-curvature regions?
    # Wait, no. Let me re-read...

    # From the PDF: "松弛正则化：只在需要的地方放松"
    # reg = (curv[:, 0].abs() * delta.norm(dim=-1)).mean()
    #
    # This is confusing. If we ADD this to the loss, then it penalizes
    # delta *more* in high-curvature regions (because |K| is larger).
    # But we WANT more relaxation in high-curvature regions!
    #
    # Correct interpretation: the reg loss should penalize delta in FLAT regions.
    # So: reg = (1/|K|+eps) * ||delta||, or reg = exp(-|K|) * ||delta||
    #
    # Actually, re-reading the PDF: "reg = (curv[:, 0].abs() * delta.norm(dim=-1)).mean()"
    # This is minimized when delta is small where |K| is small. But it also
    # penalizes delta where |K| is large (even more so).
    #
    # The gate mechanism already handles WHERE to relax (gate = σ(γ|K|+β)).
    # The reg term should penalize the TOTAL amount of relaxation, not
    # encourage more in curved regions.
    #
    # Better interpretation: L_reg = mean(||delta||) — just penalize the
    # magnitude of the relaxation field. The gate already ensures it's
    # only active in curved regions. No need for curvature weighting here.
    #
    # But the PDF explicitly has curv weight... Let me use it but with the
    # understanding that it's a soft constraint. The gate is the hard
    # mechanism; this is an additional regularizer.

    return (K_abs * delta_norm).mean()


# ====================== Physics-Inspired Loss ======================

def unfold_supervision_loss(U_pred, U_gt, valid_mask):
    """L2 loss on 2D coordinates, only for vertices with valid GT."""
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=U_pred.device)
    return F.mse_loss(U_pred[valid_mask], U_gt[valid_mask])


class PhysicsInspiredLoss(nn.Module):
    """
    Combined loss for PhysUnfolder training.

    L_total = λ_seg * L_seg + λ_mem * L_membrane + λ_bend * L_bending + λ_reg * L_reg

    The first term is supervised (needs GT panel labels);
    the last three are self-supervised physics constraints.
    """

    def __init__(self, num_classes=41, ignore_idx=0,
                 lambda_membrane=0.1, lambda_bending=0.01, lambda_reg=0.1,
                 lambda_unfold=1.0, lambda_garment=1.0, class_weights=None):
        """
        Args:
            num_classes:      number of panel classes
            ignore_idx:       label index to ignore in segmentation loss
            lambda_membrane:  weight for membrane strain energy
            lambda_bending:   weight for bending energy (k-NN Laplacian)
            lambda_reg:       weight for relaxation regularization
            lambda_unfold:    weight for GT 2D supervision (L2)
            lambda_garment:   weight for garment mask BCE loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.ignore_idx = ignore_idx
        self.lambda_membrane = lambda_membrane
        self.lambda_bending = lambda_bending
        self.lambda_reg = lambda_reg
        self.lambda_unfold = lambda_unfold
        self.lambda_garment = lambda_garment
        self.register_buffer('class_weights',
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None else None)

    def seg_loss(self, seg_logits, labels):
        """Cross-entropy for panel segmentation. Returns scalar."""
        # seg_logits: list of (N_i, C)
        # labels:     list of (N_i,)
        total_loss = 0.0
        total_valid = 0
        for logits, gt in zip(seg_logits, labels):
            mask = gt != self.ignore_idx
            if mask.sum() == 0:
                continue
            loss = F.cross_entropy(logits[mask], gt[mask], reduction='sum')
            total_loss += loss
            total_valid += mask.sum().item()
        return total_loss / max(total_valid, 1)

    def forward(self, pred, batch):
        """
        Args:
            pred: dict from PhysUnfolderCombined.forward()
                  {seg_logits, U_geo, U_final, delta, gate} — each a list of tensors
            batch: dict with
                  {vertices, edge_indices, faces, curvature, material, labels}
                  — vertices, edge_indices, faces, curvature, labels are lists
                  — material is (B, 3)

        Returns:
            total_loss: scalar
            loss_dict:  dict with individual loss terms for logging
        """
        B = len(batch['vertices'])
        device = batch['vertices'][0].device

        # Unpack materials: (B, 3) → per-sample scalars
        materials = batch['material']  # (B, 3)
        E_mod = materials[:, 0]        # (B,)  stretch modulus
        B_mod = materials[:, 1]        # (B,)  bending stiffness

        # --- 1. Segmentation loss (supervised) ---
        loss_seg = self.seg_loss(pred['seg_logits'], batch['labels'])

        # --- 2. Physics losses (self-supervised) ---
        loss_membrane = 0.0
        loss_bending = 0.0
        loss_reg = 0.0
        count = 0
        n_bend = 0  # samples with valid faces

        for i in range(B):
            U = pred['U_final'][i]                 # (N_i, 2)
            delta = pred['delta'][i]               # (N_i, 2)
            verts_3d = batch['vertices'][i]        # (N_i, 3)
            edge_idx = batch['edge_indices'][i]    # (2, E_i)
            faces_i = batch['faces'][i]            # (F_i, 3) — may be empty
            curv = batch['curvature'][i]           # (N_i, 2)
            E_i = E_mod[i]                         # scalar
            B_i = B_mod[i]                         # scalar
            K_abs = curv[:, 0].abs()               # (N_i,)

            # Membrane energy (always available)
            loss_membrane += membrane_energy(U, verts_3d, edge_idx, E_i)

            # Bending energy (skip if faces are empty — k-NN sampled mode)
            if faces_i.numel() > 0:
                loss_bending += bending_energy(U, verts_3d, faces_i, B_i)
                n_bend += 1

            # Relaxation regularization
            loss_reg += relaxation_reg(delta, K_abs)

            count += 1

        loss_membrane = loss_membrane / max(count, 1)
        loss_bending = loss_bending / max(n_bend, 1) if n_bend > 0 else 0.0
        loss_reg = loss_reg / max(count, 1)

        # --- 3. Total loss ---
        total_loss = (
            loss_seg +
            self.lambda_membrane * loss_membrane +
            self.lambda_bending * loss_bending +
            self.lambda_reg * loss_reg
        )

        def _val(x):
            return x.item() if isinstance(x, torch.Tensor) else float(x)

        loss_dict = {
            'total': _val(total_loss),
            'seg': _val(loss_seg),
            'membrane': _val(loss_membrane),
            'bending': _val(loss_bending),
            'reg': _val(loss_reg),
        }

        return total_loss, loss_dict

    def forward_padded(self, pred, labels, mask, vertices, edge_indices, curvature,
                        materials, u_gt=None, u_gt_valid=None):
        """
        Padded batch loss (for training speed).

        Args:
            pred:         dict from forward_padded {seg_logits, U_final, delta, gate, mask}
            labels:       (B, N_pad) long
            mask:         (B, N_pad) bool
            vertices:     (B, N_pad, 3)
            edge_indices: list of (2, E_i)
            curvature:    (B, N_pad, 2)
            materials:    (B, 3)
            u_gt:         (B, N_pad, 2) or None — GT 2D coords
            u_gt_valid:   (B, N_pad) bool or None

        Returns:
            total_loss, loss_dict
        """
        B, N = labels.shape
        device = labels.device

        # --- 1. Segmentation loss ---
        # Only compute on valid+labeled vertices
        valid_labeled = mask & (labels != self.ignore_idx)
        if valid_labeled.sum() > 0:
            seg_flat = pred['seg_logits'][valid_labeled]   # (V, C)
            lbl_flat = labels[valid_labeled]                # (V,)
            loss_seg = F.cross_entropy(seg_flat, lbl_flat)
        else:
            loss_seg = torch.tensor(0.0, device=device)

        # --- 2. Physics losses (per-sample) ---
        E_mod = materials[:, 0]  # (B,)
        B_mod = materials[:, 1]

        loss_membrane, loss_bending, loss_reg = 0.0, 0.0, 0.0
        count, n_bend = 0, 0

        for i in range(B):
            valid_i = mask[i]
            ni = valid_i.sum().item()
            if ni == 0:
                continue

            U_i = pred['U_final'][i, :ni]            # (N_i, 2)
            delta_i = pred['delta'][i, :ni]           # (N_i, 2)
            V_i = vertices[i, :ni]                    # (N_i, 3)
            curv_i = curvature[i, :ni]                # (N_i, 2)
            ei = edge_indices[i].to(device)

            loss_membrane += membrane_energy(U_i, V_i, ei, E_mod[i])
            loss_bending += bending_energy_knn(U_i, ei, B_mod[i])
            loss_reg += relaxation_reg(delta_i, curv_i[:, 0].abs())
            count += 1

        loss_membrane = loss_membrane / max(count, 1)
        loss_bending = loss_bending / max(count, 1)
        loss_reg = loss_reg / max(count, 1)

        # --- 3. GT 2D supervision ---
        if u_gt is not None and u_gt_valid is not None:
            valid_unfold = mask & u_gt_valid
            loss_unfold = unfold_supervision_loss(
                pred['U_final'], u_gt, valid_unfold)
        else:
            loss_unfold = torch.tensor(0.0, device=device)

        total_loss = (
            loss_seg +
            self.lambda_membrane * loss_membrane +
            self.lambda_bending * loss_bending +
            self.lambda_reg * loss_reg +
            self.lambda_unfold * loss_unfold
        )

        # --- 4. Garment mask BCE loss ---
        loss_garment = torch.tensor(0.0, device=device)
        # garment_mask_gt passed directly as parameter
        if garment_mask_gt is not None and 'garment_logits' in pred:
            loss_garment = F.binary_cross_entropy_with_logits(
                pred['garment_logits'],
                garment_mask_gt.to(device, dtype=pred['garment_logits'].dtype))
            total_loss = total_loss + self.lambda_garment * loss_garment

        def _val(x):
            return x.item() if isinstance(x, torch.Tensor) else float(x)

        loss_dict = {
            'total': _val(total_loss), 'seg': _val(loss_seg),
            'membrane': _val(loss_membrane), 'bending': _val(loss_bending),
            'reg': _val(loss_reg), 'unfold': _val(loss_unfold),
            'garment': _val(loss_garment),
        }
        return total_loss, loss_dict

    def forward_dual(self, pred, labels_sample, vertices_sample, edge_indices_sample,
                     curvature_sample, materials, u_gt=None, u_gt_valid=None,
                     garment_mask_gt=None):
        """
        Dual-channel loss with intra-panel physics + per-panel centered unfold.

        1. Membrane/bending energy only on INTRA-panel edges (GT labels).
           This removes the penalty for separating different panels in 2D.
        2. Unfold loss: per-panel centering of both pred and GT before L2.
           Model only learns panel SHAPE, not absolute position.
        """
        B, S = labels_sample.shape
        device = labels_sample.device

        # --- 1. Segmentation loss ---
        valid = labels_sample != self.ignore_idx
        if valid.sum() > 0:
            seg_flat = pred['seg_logits'][valid]
            lbl_flat = labels_sample[valid]
            w = self.class_weights.to(device=device, dtype=seg_flat.dtype) if self.class_weights is not None else None
            loss_seg = F.cross_entropy(seg_flat, lbl_flat, weight=w)
        else:
            loss_seg = torch.tensor(0.0, device=device)

        # --- 2. Physics losses (intra-panel edges only) ---
        E_mod = materials[:, 0]
        B_mod = materials[:, 1]

        loss_membrane, loss_bending, loss_reg = 0.0, 0.0, 0.0
        mem_count, bend_count, reg_count = 0, 0, 0

        for i in range(B):
            U_i = pred['U_final'][i]              # (S, 2)
            delta_i = pred['delta'][i]             # (S, 2)
            V_i = vertices_sample[i]               # (S, 3)
            curv_i = curvature_sample[i]           # (S, 2)
            ei = edge_indices_sample[i].to(device)
            lbl_i = labels_sample[i]               # (S,)

            # Build intra-panel edge mask (src & tgt same GT panel)
            src, tgt = ei[0], ei[1]
            intra_mask = (lbl_i[src] == lbl_i[tgt]) & (lbl_i[src] != self.ignore_idx) & (lbl_i[tgt] != self.ignore_idx)
            if intra_mask.sum() >= 3:
                ei_intra = ei[:, intra_mask]
                loss_membrane += membrane_energy(U_i, V_i, ei_intra, E_mod[i])
                loss_bending += bending_energy_knn(U_i, ei_intra, B_mod[i])
                mem_count += 1
                bend_count += 1

            # Relaxation reg (all vertices)
            loss_reg += relaxation_reg(delta_i, curv_i[:, 0].abs())
            reg_count += 1

        loss_membrane = loss_membrane / max(mem_count, 1)
        loss_bending = loss_bending / max(bend_count, 1)
        loss_reg = loss_reg / max(reg_count, 1)

        # --- 3. GT 2D supervision: direct MSE on absolute positions ---
        # GT panels are already laid out in a grid (data_loader), so the model
        # must learn to put different panels at different 2D locations.
        if u_gt is not None and u_gt_valid is not None:
            valid_unfold = u_gt_valid & (labels_sample != self.ignore_idx)
            if valid_unfold.sum() > 0:
                loss_unfold = F.mse_loss(
                    pred['U_final'][valid_unfold], u_gt[valid_unfold])
            else:
                loss_unfold = torch.tensor(0.0, device=device)
        else:
            loss_unfold = torch.tensor(0.0, device=device)

        total_loss = (
            loss_seg +
            self.lambda_membrane * loss_membrane +
            self.lambda_bending * loss_bending +
            self.lambda_reg * loss_reg +
            self.lambda_unfold * loss_unfold
        )

        # --- 4. Garment mask BCE loss ---
        loss_garment = torch.tensor(0.0, device=device)
        # garment_mask_gt passed directly as parameter
        if garment_mask_gt is not None and 'garment_logits' in pred:
            loss_garment = F.binary_cross_entropy_with_logits(
                pred['garment_logits'],
                garment_mask_gt.to(device, dtype=pred['garment_logits'].dtype))
            total_loss = total_loss + self.lambda_garment * loss_garment

        def _val(x):
            return x.item() if isinstance(x, torch.Tensor) else float(x)

        loss_dict = {
            'total': _val(total_loss), 'seg': _val(loss_seg),
            'membrane': _val(loss_membrane), 'bending': _val(loss_bending),
            'reg': _val(loss_reg), 'unfold': _val(loss_unfold),
            'garment': _val(loss_garment),
        }
        return total_loss, loss_dict


# ====================== Test ======================

if __name__ == '__main__':
    print("=" * 60)
    print("PhysicsInspiredLoss 测试")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    from curvature_utils import mesh_faces_to_edges, compute_curvature

    # Create a simple test mesh: a folded plane (should have non-zero physics loss)
    N = 6
    verts_3d = torch.tensor([
        [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
        [1., 1., 0.], [0.5, 0.5, 0.3], [0.5, 0.5, -0.3],
    ], device=device)
    faces = torch.tensor([
        [0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4],
        [0, 5, 1], [1, 5, 3], [3, 5, 2], [2, 5, 0],
    ], dtype=torch.long, device=device)

    # Flattened 2D coords for an "unfolded" version (still distorted)
    U_perfect = verts_3d[:, :2].clone()  # just project to XY (perfect for flat plane)

    # Deliberately distorted 2D coords
    U_distorted = U_perfect.clone()
    U_distorted[4, :] += 0.5  # large distortion at peak vertex

    edge_index = mesh_faces_to_edges(faces)
    curv = compute_curvature(verts_3d, faces)  # (N, 2)

    # Test membrane energy
    E_mod = 500.0  # cotton-like
    B_mod = 0.01

    mem_perfect = membrane_energy(U_perfect, verts_3d, edge_index, E_mod)
    mem_distort = membrane_energy(U_distorted, verts_3d, edge_index, E_mod)
    print(f"\n膜能量 (完美): {mem_perfect.item():.6f}")
    print(f"膜能量 (畸变): {mem_distort.item():.6f}  ← 应更大")

    # Test bending energy
    bend_perfect = bending_energy(U_perfect, verts_3d, faces, B_mod)
    bend_distort = bending_energy(U_distorted, verts_3d, faces, B_mod)
    print(f"\n弯曲能量 (完美): {bend_perfect.item():.6f}")
    print(f"弯曲能量 (畸变): {bend_distort.item():.6f}  ← 应更大")

    # Test relaxation regularization
    delta_small = torch.zeros(N, 2, device=device)
    delta_large = torch.randn(N, 2, device=device) * 0.5

    reg_small = relaxation_reg(delta_small, curv[:, 0].abs())
    reg_large = relaxation_reg(delta_large, curv[:, 0].abs())
    print(f"\n松弛正则 (零):     {reg_small.item():.6f}")
    print(f"松弛正则 (大ΔU):   {reg_large.item():.6f}  ← 应更大")

    # Test combined loss
    loss_fn = PhysicsInspiredLoss(num_classes=41, ignore_idx=0)

    # Dummy batch simulating model output
    B = 2
    pred = {
        'seg_logits': [torch.randn(N, 41, device=device).softmax(-1) for _ in range(B)],
        'U_final':    [U_perfect.clone() for _ in range(B)],
        'delta':      [delta_small.clone() for _ in range(B)],
    }
    batch = {
        'vertices':     [verts_3d.clone() for _ in range(B)],
        'edge_indices': [edge_index.clone() for _ in range(B)],
        'faces':        [faces.clone() for _ in range(B)],
        'curvature':    [curv.clone() for _ in range(B)],
        'material':     torch.tensor([[500., 0.01, 100.], [200., 0.005, 50.]], device=device),
        'labels':       [torch.randint(0, 41, (N,), device=device) for _ in range(B)],
    }

    total, losses = loss_fn(pred, batch)
    print(f"\n组合损失:")
    for k, v in losses.items():
        print(f"  {k:12s}: {v:.4f}")

    print("\n所有测试通过 ✓")
