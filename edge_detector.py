"""
Panel boundary detection: find edges where two garment panels meet.

Primary method: uses segmentation labels (when available) — 100% accurate.
The panel ID changes across a seam edge, which is the gold standard for
boundary detection in this dataset.

A purely geometric fallback (dihedral angle + curvature gradient) is
provided for use when segmentation data is not available.
"""

import numpy as np
import torch

from curvature_utils import mesh_faces_to_edges


def detect_seam_edges_from_labels(faces, vertex_labels):
    """Find edges that cross panel boundaries using GT per-vertex labels.

    An undirected edge (i, j) is a seam edge if vertex i and vertex j have
    different (non-zero) panel labels.

    Args:
        faces: (F, 3) long tensor of triangle faces
        vertex_labels: (N,) int array/tensor of panel IDs (0 = unlabeled/background)

    Returns:
        seam_edges: (2, E_seam) long tensor — seam edges
        panel_edges: (2, E_panel) long tensor — interior edges
        seam_mask: (E_all,) bool — True for seam edges
    """
    N = len(vertex_labels)
    labels = np.asarray(vertex_labels)

    all_edges = mesh_faces_to_edges(faces)  # (2, E_all) long tensor
    src = all_edges[0].numpy()
    tgt = all_edges[1].numpy()

    is_seam = labels[src] != labels[tgt]
    is_seam = is_seam & (labels[src] != 0) & (labels[tgt] != 0)

    seam_edges = all_edges[:, is_seam]
    panel_edges = all_edges[:, ~is_seam]
    return seam_edges, panel_edges, is_seam


def detect_seam_edges_geometric(vertices, faces, dihedral_threshold=30.0,
                                curvature_gradient_threshold=0.5):
    """Find seam edges using purely geometric signals (no labels).

    Computes two signals for each edge:
      1. Dihedral angle: angle between face normals. Large angle → likely seam.
      2. Curvature gradient: difference in mean curvature across the edge.

    Edges exceeding either threshold are classified as seams.

    NOTE: This is less reliable than label-based detection. Internal folds
    also have high dihedral angles, and flat seams may be missed.  Use
    `detect_seam_edges_from_labels` when segmentation data is available.

    Args:
        vertices: (N, 3) tensor of vertex positions
        faces: (F, 3) long tensor
        dihedral_threshold: degrees, edges with larger angle → seam
        curvature_gradient_threshold: curvature discontinuity threshold

    Returns:
        seam_edges, panel_edges, seam_mask — same format as label-based
    """
    from curvature_utils import compute_curvature
    N = vertices.shape[0]
    device = vertices.device

    all_edges = mesh_faces_to_edges(faces)
    E = all_edges.shape[1]

    # 1. Dihedral angle per edge
    # Build edge → face mapping
    edge_to_faces = {}
    for fi in range(faces.shape[0]):
        a, b, c = faces[fi, 0].item(), faces[fi, 1].item(), faces[fi, 2].item()
        for e in [(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(c, a), max(c, a))]:
            edge_to_faces.setdefault(e, []).append(fi)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e01 = v1 - v0
    e02 = v2 - v0
    cross = torch.cross(e01, e02, dim=-1)
    cross_norm = cross.norm(dim=-1).clamp(min=1e-10)
    face_normals = cross / cross_norm.unsqueeze(-1)

    dihedral = torch.zeros(E, device=device)
    edge_list = all_edges.t().tolist()
    for ei in range(E):
        a, b = edge_list[ei]
        key = (min(a, b), max(a, b))
        f_indices = edge_to_faces.get(key, [])
        if len(f_indices) >= 2:
            n1 = face_normals[f_indices[0]]
            n2 = face_normals[f_indices[1]]
            cos_angle = (n1 * n2).sum().clamp(-1.0, 1.0)
            dihedral[ei] = torch.acos(cos_angle).abs().rad2deg()

    # 2. Curvature gradient
    curv = compute_curvature(vertices, faces)
    H = curv[:, 1]  # mean curvature
    src = all_edges[0]
    tgt = all_edges[1]
    curv_grad = (H[src] - H[tgt]).abs()

    # Normalize curvature gradient
    h_range = H.max() - H.min()
    curv_grad = curv_grad / max(h_range.item(), 1e-6)

    # 3. Combine signals
    is_seam = (dihedral > dihedral_threshold) | (curv_grad > curvature_gradient_threshold)

    seam_edges = all_edges[:, is_seam]
    panel_edges = all_edges[:, ~is_seam]
    return seam_edges, panel_edges, is_seam


def build_seam_correspondences(seam_edges, panels, vertex_labels_np):
    """Map seam edges to local indices within each panel after cutting.

    Returns list of dicts: {panel_A, local_A, panel_B, local_B, L3d}
    """
    N = len(vertex_labels_np)
    vtx_to_panel = np.full(N, -1, dtype=np.int32)
    for pi, p in enumerate(panels):
        for gi in p['global_indices'].tolist():
            vtx_to_panel[gi] = pi

    global_to_local = {}
    for pi, p in enumerate(panels):
        global_to_local[pi] = {int(gi): li for li, gi in enumerate(p['global_indices'].tolist())}

    correspondences = []
    for ei in range(seam_edges.shape[1]):
        a = seam_edges[0, ei].item()
        b = seam_edges[1, ei].item()
        pa = vtx_to_panel[a]
        pb = vtx_to_panel[b]
        if pa < 0 or pb < 0 or pa == pb:
            continue
        g2la = global_to_local.get(pa, {})
        g2lb = global_to_local.get(pb, {})
        if a not in g2la or b not in g2lb:
            continue
        # Verify both vertices exist in their respective panels
        try:
            va_local = g2la[a]
            vb_local = g2lb[b]
        except KeyError:
            continue
        va = panels[pa]['vertices'][va_local]
        vb = panels[pb]['vertices'][vb_local]
        L3d = float((va - vb).norm())
        correspondences.append({
            'panel_A': pa, 'vtx_A': va_local,
            'panel_B': pb, 'vtx_B': vb_local,
            'L3d': L3d,
        })
    return correspondences


def cut_mesh_into_panels(vertices, faces, panel_edges):
    """Cut the mesh along panel-internal edges to extract connected components.

    Each connected component (using only panel_edges) is an individual panel.

    Args:
        vertices: (N, 3) tensor
        faces: (F, 3) long tensor
        panel_edges: (2, Ep) long tensor — edges within panels (NOT seam edges)

    Returns:
        panels: list of dicts with keys 'vertices' (Np, 3), 'faces' (Fp, 3),
                'global_indices' (Np,) — mapping to original vertex indices
    """
    if panel_edges.shape[1] == 0:
        return [{'vertices': vertices, 'faces': faces,
                 'global_indices': torch.arange(vertices.shape[0])}]

    N = vertices.shape[0]
    device = vertices.device

    # Build adjacency using only panel-internal edges
    adj = {i: [] for i in range(N)}
    src = panel_edges[0].tolist()
    tgt = panel_edges[1].tolist()
    for s, t in zip(src, tgt):
        adj[s].append(t)
        adj[t].append(s)

    # Find connected components via BFS
    visited = [False] * N
    panels = []
    for start in range(N):
        if visited[start]:
            continue
        # BFS
        queue = [start]
        visited[start] = True
        component = []
        while queue:
            v = queue.pop(0)
            component.append(v)
            for nb in adj[v]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)

        if len(component) < 3:
            continue  # skip degenerate components

        comp_set = set(component)
        comp_list = sorted(component)
        old2new = {old: new for new, old in enumerate(comp_list)}

        # Filter faces where all 3 vertices are in this component
        face_mask = torch.tensor([all(fi.item() in comp_set
                                      for fi in faces[f, :])
                                  for f in range(faces.shape[0])], device=device)
        panel_faces = faces[face_mask]
        # Remap to local indices
        panel_faces_remapped = torch.tensor(
            [[old2new[fi.item()] for fi in panel_faces[f, :]]
             for f in range(panel_faces.shape[0])], dtype=torch.long, device=device)

        panel_verts = vertices[comp_list]

        panels.append({
            'vertices': panel_verts,
            'faces': panel_faces_remapped,
            'global_indices': torch.tensor(comp_list, dtype=torch.long),
        })

    return panels
