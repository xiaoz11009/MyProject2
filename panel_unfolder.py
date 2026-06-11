"""
Physics-based 2D panel unfolding via energy minimization.

Given a 3D fabric panel (triangle mesh), find 2D vertex positions that:
  - Preserve edge lengths (membrane energy)
  - Preserve local smoothness (bending energy)

The optimization is purely geometric — no learning, no GT 2D needed.
Uses L-BFGS for fast convergence.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from curvature_utils import mesh_faces_to_edges


def compute_panel_edges(faces):
    """Get undirected edges from a single panel's faces."""
    edges = torch.cat([
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]],
    ], dim=0).t().contiguous()
    edges = torch.unique(edges, dim=1)
    return edges


def init_2d_from_pca(vertices, scale=1.0):
    """Initialize 2D positions using PCA projection (best planar fit).

    Args:
        vertices: (N, 3) tensor — 3D vertex positions
        scale: scaling factor for initial positions

    Returns:
        u_init: (N, 2) tensor — initial 2D positions
    """
    v = vertices - vertices.mean(dim=0, keepdim=True)
    if v.shape[0] < 3:
        return v[:, :2] * scale
    # SVD for best-fitting plane
    U, S, Vh = torch.linalg.svd(v, full_matrices=False)
    principal = v @ Vh[:2].t()  # project onto first 2 principal components
    # Scale to roughly match 3D extent
    extent_3d = (v.max(dim=0).values - v.min(dim=0).values).norm()
    extent_2d = (principal.max(dim=0).values - principal.min(dim=0).values).norm()
    if extent_2d > 1e-8:
        principal = principal * (extent_3d / extent_2d) * scale
    return principal


def unfold_panel(vertices, faces=None, edges=None, n_iter=500, lr=0.01,
                 lambda_bend=0.01, verbose=False):
    """Unfold a single 3D panel to 2D via Adam optimization.

    Uses Adam (low memory, stable) instead of L-BFGS (high memory, O(N²) Hessian).
    For panels > 5000 vertices, downsamples via FPS first, then interpolates back.

    Args:
        vertices: (N, 3) tensor — 3D vertex positions of the panel
        faces: (F, 3) long tensor — triangle faces (optional, if edges not given)
        edges: (2, E) long tensor — undirected edges (optional, computed from faces)
        n_iter: max Adam iterations
        lr: Adam learning rate
        lambda_bend: weight of bending regularizer (small → more flexible)
        verbose: print loss at each iteration

    Returns:
        u_2d: (N, 2) numpy array — optimized 2D positions
        info: dict with convergence metrics
    """
    if edges is None and faces is not None:
        edges = compute_panel_edges(faces)
    elif edges is None:
        raise ValueError("Must provide either faces or edges")

    N = vertices.shape[0]
    device = vertices.device
    src, tgt = edges[0], edges[1]

    # Downsample large panels for efficiency
    max_v = 5000
    if N > max_v:
        from scipy.spatial import cKDTree as KDTree2
        idx = np.random.choice(N, min(max_v, N), replace=False)
        idx = np.sort(idx)
        sub_verts = vertices[idx]
        sub_faces = None
        if faces is not None and faces.shape[0] > 0:
            sub_faces_list = []
            face_arr = faces.cpu().numpy()
            old2new = {old: new for new, old in enumerate(idx)}
            for fi in face_arr:
                a, b, c = int(fi[0]), int(fi[1]), int(fi[2])
                if a in old2new and b in old2new and c in old2new:
                    sub_faces_list.append([old2new[a], old2new[b], old2new[c]])
            sub_faces = torch.tensor(sub_faces_list, dtype=torch.long, device=device) if sub_faces_list else None
        # Run on subset
        u_sub, info = unfold_panel(sub_verts, faces=sub_faces,
                                   n_iter=n_iter, lr=lr, lambda_bend=lambda_bend,
                                   verbose=verbose)
        # Interpolate back via nearest neighbor in 3D
        tree_3d = KDTree2(sub_verts.cpu().numpy())
        _, nn_idx = tree_3d.query(vertices.cpu().numpy())
        u_full = u_sub[nn_idx]
        info['downsampled'] = True
        return u_full, info

    # Normalize: center at origin, scale by max edge length for stable optimization
    v_centroid = vertices.mean(dim=0, keepdim=True)
    v_centered = vertices - v_centroid
    scale = (v_centered[src] - v_centered[tgt]).norm(dim=-1).max().clamp(min=1e-6)
    v_normed = v_centered / scale

    # Store 3D reference lengths in normalized units
    L3d = (v_normed[src] - v_normed[tgt]).norm(dim=-1).clamp(min=1e-8)

    # Initialize 2D positions via PCA on normalized vertices
    u_init = init_2d_from_pca(v_normed)
    u = nn.Parameter(u_init.clone().detach().to(device).requires_grad_(True))

    # Compute per-vertex area for bending normalization (in normalized units)
    if faces is not None and faces.shape[0] > 0:
        v0 = v_normed[faces[:, 0]]
        v1 = v_normed[faces[:, 1]]
        v2 = v_normed[faces[:, 2]]
        e01 = v1 - v0
        e02 = v2 - v0
        cross = torch.cross(e01, e02, dim=-1)
        face_areas = 0.5 * cross.norm(dim=-1).clamp(min=1e-10)
        bary_area = face_areas / 3.0
        vertex_area = torch.zeros(N, device=device)
        vertex_area.scatter_add_(0, faces[:, 0], bary_area)
        vertex_area.scatter_add_(0, faces[:, 1], bary_area)
        vertex_area.scatter_add_(0, faces[:, 2], bary_area)
        vertex_area = vertex_area.clamp(min=1e-10)
    else:
        vertex_area = torch.ones(N, device=device)

    # Build topological neighbors from mesh edges (not spatial k-NN)
    adj = {i: set() for i in range(N)}
    for s, t in zip(src.tolist(), tgt.tolist()):
        adj[s].add(t)
        adj[t].add(s)
    max_neighbors = 16
    knn_idx = []
    for v in range(N):
        nbs = list(adj[v])[:max_neighbors]
        if len(nbs) < max_neighbors:
            nbs += [v] * (max_neighbors - len(nbs))  # pad with self
        knn_idx.append(nbs)
    knn_idx_t = torch.tensor(knn_idx, dtype=torch.long, device=device)

    optimizer = torch.optim.Adam([u], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iter)

    loss_history = []
    best_loss = float('inf')

    for step in range(n_iter):
        optimizer.zero_grad()

        # Membrane energy: edge length preservation
        L2d = (u[src] - u[tgt]).norm(dim=-1)
        strain = (L2d - L3d) / L3d
        loss_membrane = (strain ** 2).mean()

        # Bending energy: graph Laplacian smoothness in 2D
        u_nb = u[knn_idx_t]  # (N, k, 2)
        u_self = u.unsqueeze(1)  # (N, 1, 2)
        lap = (u_self - u_nb).square().sum(dim=-1).mean(dim=-1)  # (N,)
        loss_bending = (lap / vertex_area).mean()

        loss = loss_membrane + lambda_bend * loss_bending
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(float(loss))

        if loss < best_loss:
            best_loss = float(loss)
        if step > 50 and abs(loss_history[-1] - loss_history[-2]) < 1e-7:
            break

    # Denormalize back to original world scale
    u_np = u.detach().cpu().numpy() * scale.item()
    u_np = u_np + v_centroid[:, :2].cpu().numpy()  # preserve XY offset, discard Z

    avg_strain = float((L2d - L3d).abs().div(L3d).mean())

    info = {
        'initial_loss': float(loss_history[0]) if loss_history else 0,
        'final_loss': float(loss_history[-1]) if loss_history else 0,
        'n_iter': len(loss_history),
        'avg_strain': avg_strain,
    }
    if verbose:
        print(f"  {len(loss_history)} iters, loss {info['initial_loss']:.4f} → {info['final_loss']:.4f}, "
              f"avg_strain={avg_strain:.4f}")

    return u_np, info


def unfold_all_panels(panels, device='cuda', n_iter=500, lambda_bend=0.01,
                      verbose=False):
    """Unfold all extracted panels.

    Args:
        panels: list of dicts from `cut_mesh_into_panels`
        device: 'cuda' or 'cpu'
        n_iter: L-BFGS iterations per panel
        lambda_bend: bending regularization weight

    Returns:
        results: list of dicts with keys 'u_2d', 'info', 'panel'
    """
    results = []
    device = torch.device(device)

    for i, panel in enumerate(panels):
        verts = panel['vertices'].to(device)
        faces = panel['faces'].to(device)

        if verts.shape[0] < 3 or faces.shape[0] < 1:
            u_2d = verts[:, :2].cpu().numpy()
            results.append({'u_2d': u_2d, 'info': {}, 'panel': panel})
            continue

        u_2d, info = unfold_panel(
            verts, faces=faces, n_iter=n_iter, lambda_bend=lambda_bend,
            verbose=(verbose and i < 2))
        results.append({'u_2d': u_2d, 'info': info, 'panel': panel})

    return results


def refine_with_seams(panels, results, seam_correspondences,
                      device='cuda', n_iter=150, lr=0.003, lambda_seam=2.0):
    """Global seam refinement.

    Identifies boundary vertices in each panel (vertices adjacent to
    seam edges) and re-optimizes with higher membrane weight on boundary
    edges.  This ensures seam lengths are accurately preserved across
    adjacent panels.

    Args:
        panels, results: from individual unfolding
        seam_correspondences: from build_seam_correspondences
        device, n_iter, lr: optimization parameters
        lambda_seam: extra weight multiplier for boundary edge membrane energy
    """
    device = torch.device(device)
    n_panels = len(results)

    # Mark boundary vertices per panel
    boundary_sets = [set() for _ in range(n_panels)]
    for sc in seam_correspondences:
        pa, pb = sc['panel_A'], sc['panel_B']
        if pa < n_panels:
            boundary_sets[pa].add(sc['vtx_A'])
        if pb < n_panels:
            boundary_sets[pb].add(sc['vtx_B'])

    # Initialize parameters
    u_params = []
    panel_data = []  # (src, tgt, L3d, boundary_mask)

    for i, r in enumerate(results):
        u_init = torch.from_numpy(r['u_2d']).float().to(device)
        u = nn.Parameter(u_init.clone().detach().requires_grad_(True))
        u_params.append(u)

        faces = panels[i]['faces'].to(device)
        if faces.shape[0] == 0:
            panel_data.append(None)
            continue

        edges = compute_panel_edges(faces)
        src, tgt = edges[0], edges[1]
        verts_3d = panels[i]['vertices'].to(device)
        L3d = (verts_3d[src] - verts_3d[tgt]).norm(dim=-1).clamp(min=1e-6)

        # Boundary edge mask
        bset = boundary_sets[i]
        is_boundary = torch.tensor(
            [(s.item() in bset) or (t.item() in bset)
             for s, t in zip(src, tgt)], device=device, dtype=torch.float32)
        panel_data.append((src, tgt, L3d, is_boundary))

    if all(pd is None for pd in panel_data):
        return [r['u_2d'] for r in results]

    optimizer = torch.optim.Adam(u_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iter)

    for _ in range(n_iter):
        optimizer.zero_grad()
        total_loss = 0.0

        for i in range(n_panels):
            pd = panel_data[i]
            if pd is None:
                continue
            src, tgt, L3d, is_boundary = pd
            L2d = (u_params[i][src] - u_params[i][tgt]).norm(dim=-1)
            strain = (L2d - L3d) / L3d
            weights = 1.0 + lambda_seam * is_boundary
            total_loss += (weights * strain ** 2).mean()

        total_loss.backward()
        optimizer.step()
        scheduler.step()

    return [u.detach().cpu().numpy() for u in u_params]
