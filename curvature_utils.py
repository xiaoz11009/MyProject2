"""
Curvature computation utilities for triangle meshes.

Gaussian curvature: angle deficit method (Meyer et al., 2003)
Mean curvature:      cotangent Laplacian method

Pure PyTorch, no external dependencies (no libigl).
"""

import torch
import torch.nn.functional as F


# ====================== Mesh Utility ======================

def mesh_faces_to_edges(faces):
    """Convert mesh faces (F, 3) to undirected edge_index (2, E)."""
    device = faces.device
    edges = torch.cat([
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]],
        faces[:, [1, 0]], faces[:, [2, 1]], faces[:, [0, 2]],
    ], dim=0).t()
    edges = torch.unique(edges, dim=1)
    return edges


# ====================== Internal: Face Geometry ======================

def _compute_face_geometry(vertices, faces):
    """
    Compute per-face angles, cotangents, areas, and normals.

    Args:
        vertices: (N, 3)
        faces:    (F, 3) long tensor

    Returns:
        angles:     (F, 3) interior angle at each vertex [a0, a1, a2]
        cot_angles: (F, 3) cot(angle) at each vertex
        areas:      (F,)   face area
        normals:    (F, 3) face unit normal
    """
    # Vertex positions per face
    v0 = vertices[faces[:, 0]]  # (F, 3)
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Edge vectors
    e01 = v1 - v0   # (F, 3)
    e12 = v2 - v1
    e20 = v0 - v2

    # Edge lengths
    l01 = e01.norm(dim=-1).clamp(min=1e-10)  # (F,)
    l12 = e12.norm(dim=-1).clamp(min=1e-10)
    l20 = e20.norm(dim=-1).clamp(min=1e-10)

    # Face normals (before normalization for area)
    e02 = -e20
    cross = torch.cross(e01, e02, dim=-1)      # (F, 3)
    cross_norm = cross.norm(dim=-1).clamp(min=1e-10)
    face_normals = cross / cross_norm.unsqueeze(-1)
    face_areas = 0.5 * cross_norm              # (F,)

    # Unit edge directions
    e01_n = e01 / l01.unsqueeze(-1)
    e12_n = e12 / l12.unsqueeze(-1)
    e20_n = e20 / l20.unsqueeze(-1)

    # Cosine of interior angles via dot products of -edge_in, +edge_out
    cos0 = (-e20_n * e01_n).sum(dim=-1).clamp(-1.0, 1.0)  # at v0
    cos1 = (-e01_n * e12_n).sum(dim=-1).clamp(-1.0, 1.0)  # at v1
    cos2 = (-e12_n * e20_n).sum(dim=-1).clamp(-1.0, 1.0)  # at v2

    angle0 = torch.acos(cos0)
    angle1 = torch.acos(cos1)
    angle2 = torch.acos(cos2)

    angles = torch.stack([angle0, angle1, angle2], dim=-1)  # (F, 3)

    # Cotangents: cot(θ) = cos(θ) / sin(θ)
    sin = torch.sin(angles).clamp(min=1e-10)
    cot_angles = cos0.cos() / sin  # wait, cos0.cos() is wrong
    # cos0 is from above, stored in cos0, cos1, cos2
    cos_all = torch.stack([cos0, cos1, cos2], dim=-1)
    cot_angles = cos_all / sin  # (F, 3)

    return angles, cot_angles, face_areas, face_normals


# ====================== Vertex Normals ======================

def compute_vertex_normals(vertices, faces):
    """
    Area-weighted average of face normals around each vertex.

    Args:
        vertices: (N, 3)
        faces:    (F, 3)

    Returns:
        (N, 3) unit vertex normals
    """
    N = vertices.shape[0]
    device = vertices.device

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e01 = v1 - v0
    e02 = v2 - v0
    cross = torch.cross(e01, e02, dim=-1)       # (num_faces, 3), unnormalized
    areas = 0.5 * cross.norm(dim=-1).clamp(min=1e-10)  # (num_faces,)

    weighted = cross / (2.0 * areas.unsqueeze(-1) + 1e-10)  # (num_faces, 3) unit face normals
    weighted = weighted * areas.unsqueeze(-1)  # weight by area

    # Scatter sum to vertices
    norm = torch.zeros(N, 3, device=device, dtype=vertices.dtype)
    for j in range(3):
        norm.scatter_add_(0, faces[:, j].unsqueeze(-1).expand(-1, 3), weighted)

    return torch.nn.functional.normalize(norm, dim=-1)


# ====================== Gaussian Curvature ======================

def compute_gaussian_curvature(vertices, faces):
    """
    Gaussian curvature via angle deficit method.

    K(v_i) = (2π - Σθ_i) / A_i   for interior vertices
    K(v_i) = (π  - Σθ_i) / A_i   for boundary vertices

    where Σθ_i is the sum of angles at vertex i across all incident faces,
    and A_i is the barycentric area (1/3 of each incident face area).

    Args:
        vertices: (N, 3)
        faces:    (F, 3)

    Returns:
        K: (N,) Gaussian curvature at each vertex
    """
    N = vertices.shape[0]
    num_f = faces.shape[0]
    device = vertices.device

    # Compute per-face geometry
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e01 = v1 - v0
    e12 = v2 - v1
    e20 = v0 - v2

    l01 = e01.norm(dim=-1).clamp(min=1e-10)
    l12 = e12.norm(dim=-1).clamp(min=1e-10)
    l20 = e20.norm(dim=-1).clamp(min=1e-10)

    e01_n = e01 / l01.unsqueeze(-1)
    e12_n = e12 / l12.unsqueeze(-1)
    e20_n = e20 / l20.unsqueeze(-1)

    # Angles
    cos0 = (-e20_n * e01_n).sum(dim=-1).clamp(-1.0, 1.0)
    cos1 = (-e01_n * e12_n).sum(dim=-1).clamp(-1.0, 1.0)
    cos2 = (-e12_n * e20_n).sum(dim=-1).clamp(-1.0, 1.0)

    angle0 = torch.acos(cos0)  # (F,)
    angle1 = torch.acos(cos1)
    angle2 = torch.acos(cos2)

    # Face areas
    e02 = -e20
    cross_norm = torch.cross(e01, e02, dim=-1).norm(dim=-1).clamp(min=1e-10)
    face_areas = 0.5 * cross_norm  # (F,)

    # Scatter angles and barycentric areas to vertices
    angle_sum = torch.zeros(N, device=device)
    area_sum = torch.zeros(N, device=device)

    angle_sum.scatter_add_(0, faces[:, 0], angle0)
    angle_sum.scatter_add_(0, faces[:, 1], angle1)
    angle_sum.scatter_add_(0, faces[:, 2], angle2)

    bary_area = face_areas / 3.0
    area_sum.scatter_add_(0, faces[:, 0], bary_area)
    area_sum.scatter_add_(0, faces[:, 1], bary_area)
    area_sum.scatter_add_(0, faces[:, 2], bary_area)

    # Detect boundary vertices
    is_boundary = _detect_boundary_vertices(vertices, faces, N, device)

    # Gaussian curvature
    expected_sum = torch.where(is_boundary, torch.pi, 2.0 * torch.pi)
    K = (expected_sum - angle_sum) / (area_sum + 1e-10)

    return K


# ====================== Mean Curvature ======================

def compute_mean_curvature(vertices, faces):
    """
    Mean curvature via cotangent Laplacian (Meyer et al., 2003).

    H(v_i) = -0.5 * ⟨Δv_i, n_i⟩

    where Δv_i is the discrete Laplace-Beltrami operator applied to vertex
    position, and n_i is the vertex normal.

    Positive H → convex (outward), negative H → concave (inward).

    Args:
        vertices: (N, 3)
        faces:    (F, 3)

    Returns:
        H_signed: (N,) signed mean curvature
        H_abs:    (N,) absolute mean curvature magnitude
    """
    N = vertices.shape[0]
    num_f = faces.shape[0]
    device = vertices.device

    # Per-face vertex positions
    v0 = vertices[faces[:, 0]]  # (num_f, 3)
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Edge vectors and lengths
    e01 = v1 - v0
    e12 = v2 - v1
    e20 = v0 - v2

    l01 = e01.norm(dim=-1).clamp(min=1e-10)
    l12 = e12.norm(dim=-1).clamp(min=1e-10)
    l20 = e20.norm(dim=-1).clamp(min=1e-10)

    # Unit edge directions
    e01_n = e01 / l01.unsqueeze(-1)
    e12_n = e12 / l12.unsqueeze(-1)
    e20_n = e20 / l20.unsqueeze(-1)

    # Cosine of interior angles
    cos0 = (-e20_n * e01_n).sum(dim=-1).clamp(-1.0, 1.0)
    cos1 = (-e01_n * e12_n).sum(dim=-1).clamp(-1.0, 1.0)
    cos2 = (-e12_n * e20_n).sum(dim=-1).clamp(-1.0, 1.0)

    # Cotangents: cot(θ) = cos(θ) / sqrt(1 - cos²(θ))
    sin0 = torch.sqrt(1.0 - cos0 * cos0).clamp(min=1e-10)
    sin1 = torch.sqrt(1.0 - cos1 * cos1).clamp(min=1e-10)
    sin2 = torch.sqrt(1.0 - cos2 * cos2).clamp(min=1e-10)

    cot0 = cos0 / sin0  # (F,)  cot(angle at v0) = cot(angle opposite edge (v1, v2))
    cot1 = cos1 / sin1  # cot(angle at v1)
    cot2 = cos2 / sin2  # cot(angle at v2)

    # Face areas for barycentric vertex areas
    e02 = -e20
    cross_norm = torch.cross(e01, e02, dim=-1).norm(dim=-1).clamp(min=1e-10)
    face_areas = 0.5 * cross_norm  # (F,)
    bary_area = face_areas / 3.0

    # Per-vertex Laplacian contribution from each face
    # L(v_a) = 0.5 * [cot(γ) * (v_b - v_a) + cot(β) * (v_c - v_a)]
    #   where γ is angle opposite edge (a,b) and β opposite (a,c)
    #
    # For vertex 0: opposite edges are (1,2) [angle at v0→cot0] and (0,2) [angle at v1→cot1], (0,1) [angle at v2→cot2]
    #   L(v0) = 0.5 * cot(∠v2) * (v1 - v0) + 0.5 * cot(∠v1) * (v2 - v0)
    #         = 0.5 * cot2 * e01 + 0.5 * cot1 * (-e20) ... wait let me redo this
    #
    # Opposite edges in face (v0, v1, v2):
    #   Edge (v0, v1): opposite vertex v2, weight = cot(∠v2) = cot2
    #   Edge (v1, v2): opposite vertex v0, weight = cot(∠v0) = cot0
    #   Edge (v2, v0): opposite vertex v1, weight = cot(∠v1) = cot1
    #
    # Contribution to L(v0): 0.5 * cot2 * (v1 - v0) + 0.5 * cot1 * (v2 - v0)
    # Contribution to L(v1): 0.5 * cot2 * (v0 - v1) + 0.5 * cot0 * (v2 - v1)
    # Contribution to L(v2): 0.5 * cot1 * (v0 - v2) + 0.5 * cot0 * (v1 - v2)

    half = 0.5
    lap_v0 = half * (cot2.unsqueeze(-1) * e01 + cot1.unsqueeze(-1) * (-e20))  # (F, 3)
    lap_v1 = half * (cot2.unsqueeze(-1) * (-e01) + cot0.unsqueeze(-1) * e12)
    lap_v2 = half * (cot1.unsqueeze(-1) * e20 + cot0.unsqueeze(-1) * (-e12))

    # Scatter sum to vertices
    lap = torch.zeros(N, 3, device=device)
    lap.scatter_add_(0, faces[:, 0].unsqueeze(-1).expand(-1, 3), lap_v0)
    lap.scatter_add_(0, faces[:, 1].unsqueeze(-1).expand(-1, 3), lap_v1)
    lap.scatter_add_(0, faces[:, 2].unsqueeze(-1).expand(-1, 3), lap_v2)

    # Vertex areas
    vertex_areas = torch.zeros(N, device=device)
    vertex_areas.scatter_add_(0, faces[:, 0], bary_area)
    vertex_areas.scatter_add_(0, faces[:, 1], bary_area)
    vertex_areas.scatter_add_(0, faces[:, 2], bary_area)

    # Normalize Laplacian by area
    lap = lap / (vertex_areas.unsqueeze(-1) + 1e-10)  # (N, 3)

    # Vertex normals for sign
    normals = compute_vertex_normals(vertices, faces)  # (N, 3)

    # Signed mean curvature: H = -0.5 * ⟨Δv, n⟩
    H_signed = -0.5 * (lap * normals).sum(dim=-1)  # (N,)

    # Absolute mean curvature
    H_abs = 0.5 * lap.norm(dim=-1)  # (N,)

    return H_signed, H_abs


# ====================== Boundary Detection ======================

def _detect_boundary_vertices(vertices, faces, N, device):
    """
    Detect boundary vertices using edge occurrence counting.

    An undirected edge appearing in exactly 1 face is a boundary edge.
    Any vertex incident to such an edge is a boundary vertex.
    """
    num_f = faces.shape[0]

    # Build all 3 undirected edges per face as (min, max) pairs
    e0 = faces[:, [0, 1]]
    e1 = faces[:, [1, 2]]
    e2 = faces[:, [2, 0]]

    e0_sorted = torch.stack([e0.min(dim=1).values, e0.max(dim=1).values], dim=1)
    e1_sorted = torch.stack([e1.min(dim=1).values, e1.max(dim=1).values], dim=1)
    e2_sorted = torch.stack([e2.min(dim=1).values, e2.max(dim=1).values], dim=1)

    # Hash: min_vtx * N + max_vtx
    all_hashes = torch.cat([
        e0_sorted[:, 0].long() * N + e0_sorted[:, 1].long(),
        e1_sorted[:, 0].long() * N + e1_sorted[:, 1].long(),
        e2_sorted[:, 0].long() * N + e2_sorted[:, 1].long(),
    ], dim=0)  # (3F,)

    # Count occurrences
    unique, counts = torch.unique(all_hashes, return_counts=True)
    boundary_hashes = unique[counts == 1]

    if boundary_hashes.numel() == 0:
        return torch.zeros(N, dtype=torch.bool, device=device)

    # Extract boundary vertex indices
    v1 = boundary_hashes // N
    v2 = boundary_hashes % N
    boundary_verts = torch.cat([v1, v2]).unique()

    is_boundary = torch.zeros(N, dtype=torch.bool, device=device)
    is_boundary[boundary_verts] = True
    return is_boundary


# ====================== Combined API ======================

def compute_curvature(vertices, faces):
    """
    Compute Gaussian and mean curvature for all mesh vertices.

    Args:
        vertices: (N, 3) mesh vertex positions
        faces:    (F, 3) triangle faces as vertex indices

    Returns:
        curvature: (N, 2) tensor with columns [gaussian_curvature, mean_curvature]
    """
    N = vertices.shape[0]
    num_faces = faces.shape[0]
    device = vertices.device

    # --- Shared geometry ---
    v0 = vertices[faces[:, 0]]  # (num_faces, 3)
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e01 = v1 - v0
    e12 = v2 - v1
    e20 = v0 - v2

    l01 = e01.norm(dim=-1).clamp(min=1e-10)
    l12 = e12.norm(dim=-1).clamp(min=1e-10)
    l20 = e20.norm(dim=-1).clamp(min=1e-10)

    e01_n = e01 / l01.unsqueeze(-1)
    e12_n = e12 / l12.unsqueeze(-1)
    e20_n = e20 / l20.unsqueeze(-1)

    # Interior angles
    cos0 = (-e20_n * e01_n).sum(dim=-1).clamp(-1.0, 1.0)
    cos1 = (-e01_n * e12_n).sum(dim=-1).clamp(-1.0, 1.0)
    cos2 = (-e12_n * e20_n).sum(dim=-1).clamp(-1.0, 1.0)

    angle0 = torch.acos(cos0)  # (F,)
    angle1 = torch.acos(cos1)
    angle2 = torch.acos(cos2)

    # Face areas
    e02 = -e20
    cross = torch.cross(e01, e02, dim=-1)
    cross_norm = cross.norm(dim=-1).clamp(min=1e-10)
    face_areas = 0.5 * cross_norm
    bary_area = face_areas / 3.0

    # Face normals (for vertex normals → mean curvature sign)
    face_normals = cross / cross_norm.unsqueeze(-1)  # (F, 3)

    # --- Vertex normals (area-weighted) ---
    vn_weighted = face_normals * face_areas.unsqueeze(-1)
    vert_normals = torch.zeros(N, 3, device=device)
    vert_normals.scatter_add_(0, faces[:, 0].unsqueeze(-1).expand(-1, 3), vn_weighted)
    vert_normals.scatter_add_(0, faces[:, 1].unsqueeze(-1).expand(-1, 3), vn_weighted)
    vert_normals.scatter_add_(0, faces[:, 2].unsqueeze(-1).expand(-1, 3), vn_weighted)
    vert_normals = F.normalize(vert_normals, dim=-1)

    # --- Vertex areas ---
    vertex_areas = torch.zeros(N, device=device)
    vertex_areas.scatter_add_(0, faces[:, 0], bary_area)
    vertex_areas.scatter_add_(0, faces[:, 1], bary_area)
    vertex_areas.scatter_add_(0, faces[:, 2], bary_area)

    # --- Gaussian curvature (angle deficit) ---
    angle_sum = torch.zeros(N, device=device)
    angle_sum.scatter_add_(0, faces[:, 0], angle0)
    angle_sum.scatter_add_(0, faces[:, 1], angle1)
    angle_sum.scatter_add_(0, faces[:, 2], angle2)

    is_boundary = _detect_boundary_vertices(vertices, faces, N, device)
    expected_sum = torch.where(is_boundary, torch.pi, 2.0 * torch.pi)
    gaussian_curv = (expected_sum - angle_sum) / (vertex_areas + 1e-10)

    # --- Mean curvature (cotangent Laplacian) ---
    sin0 = (1.0 - cos0 * cos0).sqrt().clamp(min=1e-10)
    sin1 = (1.0 - cos1 * cos1).sqrt().clamp(min=1e-10)
    sin2 = (1.0 - cos2 * cos2).sqrt().clamp(min=1e-10)

    cot0 = cos0 / sin0  # cot(angle opposite edge (v1,v2))
    cot1 = cos1 / sin1  # cot(angle opposite edge (v2,v0))
    cot2 = cos2 / sin2  # cot(angle opposite edge (v0,v1))

    lap = torch.zeros(N, 3, device=device)
    half = 0.5
    lap.scatter_add_(0, faces[:, 0].unsqueeze(-1).expand(-1, 3),
                     half * (cot2.unsqueeze(-1) * e01 + cot1.unsqueeze(-1) * (-e20)))
    lap.scatter_add_(0, faces[:, 1].unsqueeze(-1).expand(-1, 3),
                     half * (cot2.unsqueeze(-1) * (-e01) + cot0.unsqueeze(-1) * e12))
    lap.scatter_add_(0, faces[:, 2].unsqueeze(-1).expand(-1, 3),
                     half * (cot1.unsqueeze(-1) * e20 + cot0.unsqueeze(-1) * (-e12)))

    lap = lap / (vertex_areas.unsqueeze(-1) + 1e-10)
    mean_curv = -0.5 * (lap * vert_normals).sum(dim=-1)  # signed

    curvature = torch.stack([gaussian_curv, mean_curv], dim=-1)  # (N, 2)
    return curvature


# ====================== Test ======================

def _subdivide_octahedron(vertices, faces, steps=3):
    """Subdivide an octahedron to create a uniform closed sphere mesh."""
    for _ in range(steps):
        edge_midpoint = {}
        new_faces = []
        for f in faces:
            v0, v1, v2 = f[0].item(), f[1].item(), f[2].item()
            edges = [(min(v0, v1), max(v0, v1)),
                     (min(v1, v2), max(v1, v2)),
                     (min(v2, v0), max(v2, v0))]
            mids = []
            for e in edges:
                if e not in edge_midpoint:
                    mid = (vertices[e[0]] + vertices[e[1]]) / 2.0
                    edge_midpoint[e] = len(vertices)
                    vertices = torch.cat([vertices, mid.unsqueeze(0)], dim=0)
                mids.append(edge_midpoint[e])
            m01, m12, m20 = mids
            new_faces.extend([
                [v0, m01, m20],
                [v1, m12, m01],
                [v2, m20, m12],
                [m01, m12, m20],
            ])
        faces = torch.tensor(new_faces, dtype=torch.long, device=vertices.device)
    # Project to unit sphere
    vertices = torch.nn.functional.normalize(vertices, dim=-1)
    return vertices, faces


if __name__ == '__main__':
    print("=" * 60)
    print("曲率计算测试")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    eps = 1e-5

    # ---- Test 1: Closed sphere (analytical: K ≈ 1, H ≈ 1) ----
    print("\n--- 测试 1: 闭合球面 (八面体细分) ---")
    verts = torch.tensor([
        [1., 0., 0.], [-1., 0., 0.], [0., 1., 0.],
        [0., -1., 0.], [0., 0., 1.], [0., 0., -1.],
    ], device=device)
    faces = torch.tensor([
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
    ], dtype=torch.long, device=device)

    verts_s, faces_s = _subdivide_octahedron(verts, faces, steps=3)
    curv_s = compute_curvature(verts_s, faces_s)

    gauss_ok = curv_s[:, 0].abs() > eps
    interior = gauss_ok  # closed sphere has no boundary
    K_mean = curv_s[:, 0][interior].mean().item()
    H_mean = curv_s[:, 1][interior].mean().item()
    print(f"顶点数: {verts_s.shape[0]}, 面数: {faces_s.shape[0]}")
    print(f"高斯曲率均值: {K_mean:.4f}  (理论: 1.0)")
    print(f"平均曲率均值: {H_mean:.4f}  (理论: 1.0)")
    print(f"  K 标准差: {curv_s[:, 0][interior].std().item():.4f}")
    print(f"  H 标准差: {curv_s[:, 1][interior].std().item():.4f}")

    # ---- Test 2: Cylinder (analytical: K ≈ 0, H = 0.5) ----
    print("\n--- 测试 2: 闭合圆柱 (无边界) ---")
    import numpy as np
    n_t, n_z = 80, 16
    theta = torch.linspace(0, 2 * np.pi, n_t, device=device)
    z_vals = torch.linspace(-1, 1, n_z, device=device)
    TH, Z = torch.meshgrid(theta, z_vals, indexing='ij')

    verts_cyl = torch.stack([
        torch.cos(TH).flatten(),
        torch.sin(TH).flatten(),
        Z.flatten(),
    ], dim=-1)

    faces_cyl_list = []
    for i in range(n_t):
        for j in range(n_z - 1):
            a = i * n_z + j
            b = ((i + 1) % n_t) * n_z + j
            c = ((i + 1) % n_t) * n_z + j + 1
            d = i * n_z + j + 1
            faces_cyl_list.extend([[a, b, c], [a, c, d]])
    faces_cyl = torch.tensor(faces_cyl_list, dtype=torch.long, device=device)

    curv_cyl = compute_curvature(verts_cyl, faces_cyl)

    K_abs_mean = curv_cyl[:, 0].abs().mean().item()
    H_median = curv_cyl[:, 1].median().item()
    print(f"顶点数: {verts_cyl.shape[0]}, 面数: {faces_cyl.shape[0]}")
    print(f"|K| 均值: {K_abs_mean:.6f}  (理论: 0.0)")
    print(f"H 中位数: {H_median:.4f}   (理论: 0.5)")

    # ---- Test 3: Plane (4 vertices, has boundary) ----
    print("\n--- 测试 3: 平面 (含边界) ---")
    verts_p = torch.tensor([
        [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.],
    ], device=device)
    faces_p = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long, device=device)
    curv_p = compute_curvature(verts_p, faces_p)
    print(f"H: {curv_p[:, 1].abs().max().item():.6f}  (理论: 0.0)")
    print(f"  注: 边界顶点 K 非零是正常的 (角度亏格含边界测地曲率)")
    # Interior vertex 1,2 on a flat mesh: K should be 0
    print(f"  内点 v1 H={curv_p[1, 1].item():.6f}, v2 H={curv_p[2, 1].item():.6f}")

    # ---- Test 4: Real GarmentCodeData mesh ----
    print("\n--- 测试 4: 真实 GarmentCodeData 网格 ---")
    import os, glob
    pattern = '/home/ddd/zkl/GarmentCodeData/GarmentCodeData_v2/garments_5000_0/random_body/rand_*/*_sim.ply'
    ply_files = sorted(glob.glob(pattern))
    if ply_files:
        import trimesh
        mesh = trimesh.load(ply_files[0], process=False)
        verts_arr = mesh.vertices.astype('float32')
        faces_arr = mesh.faces.astype('int64')
        verts_real = torch.from_numpy(verts_arr).float().to(device)
        faces_real = torch.from_numpy(faces_arr).long().to(device)
        print(f"文件: {os.path.basename(ply_files[0])}")
        print(f"顶点: {verts_real.shape[0]}, 面: {faces_real.shape[0]}")

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        curv_real = compute_curvature(verts_real, faces_real)
        t1.record()
        torch.cuda.synchronize()
        elapsed = t0.elapsed_time(t1)

        K_r, H_r = curv_real[:, 0], curv_real[:, 1]
        print(f"耗时: {elapsed:.1f} ms")
        print(f"K 范围: [{K_r.min().item():.4f}, {K_r.max().item():.4f}]")
        print(f"K 均值: {K_r.mean().item():.6f}")
        print(f"H 范围: [{H_r.min().item():.4f}, {H_r.max().item():.4f}]")
        print(f"H 均值: {H_r.mean().item():.6f}")
        n_boundary = (_detect_boundary_vertices(verts_real, faces_real,
                                                 verts_real.shape[0], device)).sum().item()
        print(f"边界顶点: {n_boundary} / {verts_real.shape[0]}")
    else:
        print("未找到 PLY 文件，跳过真实数据测试")

    print("\n所有测试完成 ✓")
