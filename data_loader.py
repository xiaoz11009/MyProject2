"""
Data loader for GarmentCodeData compatible with NeuralTailor / PhysUnfolder.

Two modes:
  - Point cloud mode (full_mesh=False): sample 4096 points from mesh surface
  - Full mesh mode    (full_mesh=True):  return vertices + topology + curvature + GT 2D
                                         (for PhysUnfolderCombined dual-channel model)
"""
import os
import json
import pickle
import numpy as np
import torch
import trimesh
from collections import Counter
from scipy.spatial import cKDTree

# Lazy import: curvature_utils depends on torch, import only when needed
_curvature_utils = None


def _get_curvature_utils():
    global _curvature_utils
    if _curvature_utils is None:
        from curvature_utils import mesh_faces_to_edges, compute_curvature
        _curvature_utils = (mesh_faces_to_edges, compute_curvature)
    return _curvature_utils


# ====================== Label processing ======================

def normalize_label(raw: str) -> str:
    """Normalize labels: stitch vertices → 'seam' for later propagation."""
    if raw.startswith('stitch_'):
        return 'seam'
    return raw


# ====================== 2D GT Extraction ======================

def rotation_matrix_3d(rx, ry, rz):
    """Build 3D rotation matrix (X→Y→Z Euler angles in degrees)."""
    rx, ry, rz = np.radians([rx, ry, rz])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def project_to_panel_2d(points_3d, panel_data):
    """Inverse-transform 3D points to panel local 2D coordinates."""
    translation = np.array(panel_data['translation'], dtype=np.float64)
    rotation = np.array(panel_data['rotation'], dtype=np.float64)
    pts = points_3d.astype(np.float64) - translation
    R = rotation_matrix_3d(*rotation)
    R_inv = R.T
    pts = pts @ R_inv.T
    return pts[:, :2].astype(np.float32)


def _layout_panels_grid(u_gt, valid, labels_s, id_to_part, margin=0.6):
    """Lay out centered panels in a deterministic grid for distinct 2D positions.

    No re-normalization — keeps absolute grid positions so panels are truly
    separated and the model must learn distinct positions per panel.
    """
    bboxes = {}
    for pid in np.unique(labels_s):
        if pid == 0:
            continue
        mask = (labels_s == pid) & valid
        if mask.sum() < 5:
            continue
        u = u_gt[mask]
        bboxes[pid] = {'w': u[:, 0].max() - u[:, 0].min(),
                       'h': u[:, 1].max() - u[:, 1].min()}
    panel_ids = sorted(bboxes.keys(), key=lambda p: id_to_part.get(p, f'p{p}'))
    if len(panel_ids) < 2:
        return u_gt

    n = len(panel_ids)
    cols = max(2, int(np.ceil(np.sqrt(n))))
    rows = (n + cols - 1) // cols
    col_w, row_h = [0.0] * cols, [0.0] * rows
    for i, pid in enumerate(panel_ids):
        r, c = i // cols, i % cols
        col_w[c] = max(col_w[c], bboxes[pid]['w'])
        row_h[r] = max(row_h[r], bboxes[pid]['h'])

    x_cum, y_cum = [0.0], [0.0]
    for w in col_w:
        x_cum.append(x_cum[-1] + w + margin)
    for h in row_h:
        y_cum.append(y_cum[-1] + h + margin)

    u_layout = u_gt.copy()
    for i, pid in enumerate(panel_ids):
        r, c = i // cols, i % cols
        mask = labels_s == pid
        u_p = u_layout[mask]
        cell_cx = x_cum[c] + col_w[c] / 2
        cell_cy = y_cum[r] + row_h[r] / 2
        panel_cx, panel_cy = u_p[:, 0].mean(), u_p[:, 1].mean()
        u_p[:, 0] += cell_cx - panel_cx
        u_p[:, 1] += cell_cy - panel_cy
        u_layout[mask] = u_p

    return u_layout


def extract_gt_2d(sample_name, ply_path, vertices_s, labels_s, centroid, max_dist,
                   part_to_id):
    """Extract GT 2D: per-panel projection + centering + grid layout.

    Panels are laid out in a deterministic grid so the model learns distinct
    2D positions. The loss uses absolute MSE (no per-panel centering).
    """
    spec_path = os.path.join(os.path.dirname(ply_path),
                             f'{sample_name}_specification.json')
    if not os.path.exists(spec_path):
        return None, None

    with open(spec_path, 'r') as f:
        spec = json.load(f)
    panels = spec['pattern']['panels']

    pts_orig = vertices_s * max(max_dist, 1e-8) + centroid

    u_gt = np.zeros((len(vertices_s), 2), dtype=np.float32)
    valid = np.zeros(len(vertices_s), dtype=bool)

    for panel_name, panel_data in panels.items():
        panel_id = part_to_id.get(panel_name, -1)
        if panel_id <= 0:
            continue
        mask = labels_s == panel_id
        if mask.sum() < 5:
            continue
        pts_panel = pts_orig[mask]
        u_panel = project_to_panel_2d(pts_panel, panel_data)
        u_mean = u_panel.mean(axis=0)
        u_panel = u_panel - u_mean
        u_gt[mask] = u_panel
        valid[mask] = True

    if valid.sum() > 0:
        global_scale = np.abs(u_gt[valid]).max()
        if global_scale > 1e-8:
            u_gt = u_gt / global_scale

    # Grid layout for distinct panel positions
    id_to_part_rev = {v: k for k, v in part_to_id.items()}
    u_gt = _layout_panels_grid(u_gt, valid, labels_s, id_to_part_rev)

    u_gt[~valid] = 0.0
    return torch.from_numpy(u_gt).float(), torch.from_numpy(valid)


def propagate_seam_labels(labels, faces, num_vertices, max_iters=10):
    """Propagate seam vertex labels to nearest panel labels via mesh adjacency."""
    labels = labels.copy()
    adj = [[] for _ in range(num_vertices)]
    for a, b, c in faces:
        a, b, c = int(a), int(b), int(c)
        adj[a].extend([b, c])
        adj[b].extend([a, c])
        adj[c].extend([a, b])
    adj = [list(set(nbrs)) for nbrs in adj]

    for _ in range(max_iters):
        seam_mask = labels == 'seam'
        if not seam_mask.any():
            break
        changed = 0
        for idx in np.where(seam_mask)[0]:
            nbr_labels = [labels[n] for n in adj[idx] if labels[n] != 'seam']
            if nbr_labels:
                labels[idx] = Counter(nbr_labels).most_common(1)[0][0]
                changed += 1
        if changed == 0:
            break
    labels[labels == 'seam'] = 'unlabeled'
    return labels


def read_segmentation(seg_path, num_vertices, faces=None):
    """Read segmentation.txt, return normalized label array."""
    labels = np.full(num_vertices, 'unlabeled', dtype=object)
    with open(seg_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_vertices:
                break
            raw = line.strip().split(',')[0]  # take first label for multi-stitch vertices
            if raw:
                labels[i] = normalize_label(raw)
    if faces is not None:
        labels = propagate_seam_labels(labels, faces, num_vertices)
    return labels


# ====================== Point sampling ======================

def fps_sample(points, npoint):
    """Farthest point sampling. Returns indices of sampled points."""
    N = len(points)
    centroids = np.zeros(npoint, dtype=np.int64)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = points[farthest:farthest + 1]
        dist = np.sum((points - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    return centroids


def uniform_mesh_sample(vertices, faces, num_points):
    """Uniformly sample points on mesh surface using barycentric coordinates.
    Returns (points, face_ids, barycentric_weights)."""
    # Compute face areas
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(cross, axis=1) / 2.0
    areas = areas / areas.sum()

    # Sample faces proportional to area
    face_ids = np.random.choice(len(faces), size=num_points, p=areas)
    v0_s = vertices[faces[face_ids, 0]]
    v1_s = vertices[faces[face_ids, 1]]
    v2_s = vertices[faces[face_ids, 2]]

    # Random barycentric coordinates
    r1 = np.random.random(num_points)
    r2 = np.random.random(num_points)
    sqrt_r1 = np.sqrt(r1)
    u = 1 - sqrt_r1
    v = r2 * sqrt_r1
    w = 1 - u - v

    points = u[:, None] * v0_s + v[:, None] * v1_s + w[:, None] * v2_s
    bary = np.stack([u, v, w], axis=-1)  # (num_points, 3)
    return points.astype(np.float32), face_ids, bary


# ====================== Dataset ======================

def build_knn_edges_from_points(points, k=8):
    """Build k-NN edge_index from 3D point positions using cKDTree (CPU)."""
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k + 1)  # k+1 because self is included
    N = len(points)
    src = np.repeat(np.arange(N), k)
    tgt = idx[:, 1:].ravel()  # skip self (column 0)
    edge_index = np.stack([src, tgt], axis=0)
    return torch.from_numpy(edge_index.astype(np.int64))


class GarmentSegDataset:
    """
    Garment panel segmentation dataset.

    Two modes:
      full_mesh=False (default):
        Returns sampled point cloud (num_points, 3) + labels.
        Compatible with the original NeuralTailorSeg model.

      full_mesh=True:
        Returns vertices + edge_index (mesh or k-NN) + curvature + labels + material.
        Uses num_vertices to limit vertex count (FPS sampling) to avoid O(N²) blowup.
        Compatible with PhysUnfolderCombined dual-channel model.
    """

    def __init__(self, data_root, garment_folders=None, body_types=None,
                 num_points=4096, max_samples=None, part_to_id_path=None,
                 normalize=True, full_mesh=False, material_variation=True,
                 num_vertices=4096, knn_edges_k=12,
                 curvature_cache_dir=None):
        if garment_folders is None:
            garment_folders = ['garments_5000_0']
        if body_types is None:
            body_types = ['default_body']

        self.data_root = data_root
        self.num_points = num_points
        self.normalize = normalize
        self.full_mesh = full_mesh
        self.material_variation = material_variation
        self.num_vertices = num_vertices
        self.knn_edges_k = knn_edges_k

        if curvature_cache_dir is None and full_mesh:
            curvature_cache_dir = os.path.join(
                data_root, 'curvature_cache', garment_folders[0], body_types[0])
        self.curvature_cache_dir = curvature_cache_dir

        # Scan samples
        self.samples = self._scan(data_root, garment_folders, body_types, max_samples)

        # Load class mapping
        if part_to_id_path is None:
            part_to_id_path = os.path.join(data_root, 'part_to_id.npy')
        pkl_path = part_to_id_path.replace('.npy', '.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                self.part_to_id = pickle.load(f)
        elif os.path.exists(part_to_id_path):
            self.part_to_id = np.load(part_to_id_path, allow_pickle=True).item()
        else:
            self.part_to_id = {}
        self.num_classes = len(self.part_to_id)
        self.id_to_part = {v: k for k, v in self.part_to_id.items()}

        mode_str = "全网格" if full_mesh else "点云采样"
        print(f"Dataset [{mode_str}]: {len(self.samples)} 样本, {self.num_classes} 面板类别")

    def _scan(self, data_root, garment_folders, body_types, max_samples):
        samples = []
        for gf in garment_folders:
            for bt in body_types:
                body_dir = os.path.join(data_root, gf, bt)
                if not os.path.exists(body_dir):
                    continue
                for item in sorted(os.listdir(body_dir)):
                    item_path = os.path.join(body_dir, item)
                    if not os.path.isdir(item_path) or not item.startswith('rand_'):
                        continue
                    ply_path = os.path.join(item_path, f'{item}_sim.ply')
                    seg_path = os.path.join(item_path, f'{item}_sim_segmentation.txt')
                    if os.path.exists(ply_path) and os.path.exists(seg_path):
                        samples.append((item, ply_path, seg_path))
        if max_samples:
            samples = samples[:max_samples]
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name, ply_path, seg_path = self.samples[idx]

        # 1. Load mesh
        mesh = trimesh.load(ply_path, process=False)
        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int64)
        N = len(vertices)

        # 2. Load segmentation labels
        labels = read_segmentation(seg_path, N, faces)

        # 3. Map labels to integer IDs (unlabeled → 0)
        label_ids = np.zeros(N, dtype=np.int64)
        for i, lbl in enumerate(labels):
            label_ids[i] = self.part_to_id.get(lbl, 0)

        # 4. Normalize: center at origin, scale to unit sphere
        centroid = np.zeros(3, dtype=np.float32)
        max_dist = 1.0
        if self.normalize:
            centroid = vertices.mean(axis=0)
            vertices = vertices - centroid
            max_dist = np.linalg.norm(vertices, axis=1).max()
            vertices = vertices / max(max_dist, 1e-8)

        # === Full mesh mode: dual-channel (GCN on full mesh + EdgeConv on FPS) ===
        if self.full_mesh:
            mesh_faces_to_edges_fn, compute_curvature_fn = _get_curvature_utils()

            # Load curvature from cache
            cache_path = None
            if self.curvature_cache_dir:
                cache_path = os.path.join(self.curvature_cache_dir,
                                          f'{sample_name}_curvature.pt')
            if cache_path and os.path.exists(cache_path):
                curvature_full = torch.load(cache_path, map_location='cpu',
                                            weights_only=True)
            else:
                vt = torch.from_numpy(vertices).float()
                ft = torch.from_numpy(faces).long()
                curvature_full = compute_curvature_fn(vt, ft)

            # Full mesh: real triangle edges for GCN (try cache)
            ei_cache = None
            if self.curvature_cache_dir:
                ei_cache = os.path.join(self.curvature_cache_dir,
                                        f'{sample_name}_edges.pt')
            if ei_cache and os.path.exists(ei_cache):
                edge_index_full = torch.load(ei_cache, map_location='cpu',
                                             weights_only=True)
            else:
                faces_t = torch.from_numpy(faces).long()
                edge_index_full = mesh_faces_to_edges_fn(faces_t)
                if ei_cache:
                    torch.save(edge_index_full, ei_cache)

            vertices_full_t = torch.from_numpy(vertices).float()
            labels_full_t = torch.from_numpy(label_ids).long()

            # FPS-sample vertices (try cache first)
            if N > self.num_vertices:
                fps_cache_path = None
                if self.curvature_cache_dir:
                    fps_cache_path = os.path.join(self.curvature_cache_dir,
                                                  f'{sample_name}_fps{self.num_vertices}.npy')
                if fps_cache_path and os.path.exists(fps_cache_path):
                    sample_idx = np.load(fps_cache_path)
                else:
                    sample_idx = fps_sample(vertices, self.num_vertices)
                    if fps_cache_path:
                        np.save(fps_cache_path, sample_idx)
                vertices_s = vertices[sample_idx]
                curvature_sample = curvature_full[sample_idx].clone()
                labels_s = label_ids[sample_idx]
                sample_indices_t = torch.from_numpy(sample_idx).long()

                # k-NN edges on sampled points for EdgeConv
                edge_index_sample = build_knn_edges_from_points(
                    vertices_s, k=self.knn_edges_k)
                vertices_sample_t = torch.from_numpy(vertices_s).float()
            else:
                # Pad to num_vertices if mesh is too small
                if N < self.num_vertices:
                    pad = self.num_vertices - N
                    sample_indices_t = torch.cat([
                        torch.arange(N, dtype=torch.long),
                        torch.zeros(pad, dtype=torch.long)  # pad with index 0
                    ])
                    vertices_sample_t = torch.cat([
                        vertices_full_t,
                        vertices_full_t[:1].repeat(pad, 1)  # repeat first vertex
                    ], dim=0)
                    curvature_sample = torch.cat([
                        curvature_full,
                        curvature_full[:1].repeat(pad, 1)
                    ], dim=0)
                    edge_index_sample = edge_index_full  # same edges, pad vertices ignored
                    labels_s = np.concatenate([label_ids, np.zeros(pad, dtype=np.int64)])
                else:
                    sample_indices_t = torch.arange(N, dtype=torch.long)
                    vertices_sample_t = vertices_full_t
                    edge_index_sample = edge_index_full
                    curvature_sample = curvature_full
                    labels_s = label_ids

            labels_sample_t = torch.from_numpy(labels_s).long()

            # Garment mask: which panel IDs exist in this sample (41-dim binary)
            garment_mask = np.zeros(self.num_classes, dtype=np.float32)
            present_ids = np.unique(labels_s)
            present_ids = present_ids[present_ids != 0]  # exclude ignore_idx
            garment_mask[present_ids] = 1.0
            garment_mask_t = torch.from_numpy(garment_mask)

            # GT 2D on sampled points (try cache)
            u_gt, u_gt_valid = None, None
            gt_cache = None
            if self.curvature_cache_dir:
                gt_cache = os.path.join(self.curvature_cache_dir,
                                        f'{sample_name}_gt2d_s{self.num_vertices}.pt')
            if gt_cache and os.path.exists(gt_cache):
                cached = torch.load(gt_cache, map_location='cpu', weights_only=True)
                u_gt, u_gt_valid = cached['u_gt'], cached['u_gt_valid']
            else:
                verts_for_gt = vertices_s if N > self.num_vertices else (
                    np.concatenate([vertices, vertices[:1].repeat(self.num_vertices - N, 0)])
                    if N < self.num_vertices else vertices
                )
                u_gt, u_gt_valid = extract_gt_2d(
                    sample_name, ply_path, verts_for_gt,
                    labels_s, centroid, max_dist, self.part_to_id)
                if gt_cache and u_gt is not None:
                    torch.save({'u_gt': u_gt, 'u_gt_valid': u_gt_valid}, gt_cache)

            # Material
            if self.material_variation:
                E = 1.0 + np.random.uniform(-0.2, 0.2)
                B = 0.01 + np.random.uniform(-0.003, 0.003)
                S = 1.0 + np.random.uniform(-0.2, 0.2)
                material = torch.tensor([E, B, S], dtype=torch.float32)
            else:
                material = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

            return {
                # Full mesh (for GCN)
                'vertices_full':    vertices_full_t,    # (N, 3)
                'edge_index_full':  edge_index_full,    # (2, E) — real triangles
                'labels_full':      labels_full_t,      # (N,)
                'curvature_full':   curvature_full,     # (N, 2)
                # Sampled points (for EdgeConv + heads)
                'vertices_sample':  vertices_sample_t,  # (num_vertices, 3)
                'edge_index_sample': edge_index_sample, # (2, E_sample) — k-NN
                'sample_indices':   sample_indices_t,   # (num_vertices,) — maps sample→full
                'curvature_sample': curvature_sample,   # (num_vertices, 2)
                'labels_sample':    labels_sample_t,    # (num_vertices,)
                'garment_mask':     garment_mask_t,     # (num_classes,) binary
                # GT
                'u_gt':             u_gt,
                'u_gt_valid':       u_gt_valid,
                'material':         material,
                'name':             sample_name,
            }

        # === Point cloud mode: sample fixed-size points (original behavior) ===
        else:
            if N > self.num_points:
                pts, face_ids = uniform_mesh_sample(vertices, faces, self.num_points)[:2]
                face_verts = faces[face_ids]
                v0_labels = label_ids[face_verts[:, 0]]
                v1_labels = label_ids[face_verts[:, 1]]
                v2_labels = label_ids[face_verts[:, 2]]
                sampled_labels = np.zeros(self.num_points, dtype=np.int64)
                for i in range(self.num_points):
                    sampled_labels[i] = np.bincount(
                        [v0_labels[i], v1_labels[i], v2_labels[i]]).argmax()
                label_ids = sampled_labels
            elif N < self.num_points:
                pts, face_ids = uniform_mesh_sample(vertices, faces, self.num_points)[:2]
                face_verts = faces[face_ids]
                v0_labels = label_ids[face_verts[:, 0]]
                v1_labels = label_ids[face_verts[:, 1]]
                v2_labels = label_ids[face_verts[:, 2]]
                sampled_labels = np.zeros(self.num_points, dtype=np.int64)
                for i in range(self.num_points):
                    sampled_labels[i] = np.bincount(
                        [v0_labels[i], v1_labels[i], v2_labels[i]]).argmax()
                label_ids = sampled_labels
            else:
                pts = vertices.copy()

            return {
                'positions': torch.tensor(pts, dtype=torch.float32),
                'labels': torch.tensor(label_ids, dtype=torch.long),
                'name': sample_name,
            }


def collate_fn(batch):
    """Batches point-cloud samples. Fixed num_points per sample → stack."""
    positions = torch.stack([item['positions'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    names = [item['name'] for item in batch]
    return positions, labels, names


def collate_mesh_batch(batch):
    """
    Batches variable-size mesh samples into lists.

    Each mesh has different N vertices, E edges, F faces.
    PhysUnfolderCombined handles this natively via batched-graph format.
    """
    return {
        'vertices':     [item['vertices'] for item in batch],
        'faces':        [item['faces'] for item in batch],
        'edge_indices': [item['edge_index'] for item in batch],
        'curvature':    [item['curvature'] for item in batch],
        'labels':       [item['labels'] for item in batch],
        'material':     torch.stack([item['material'] for item in batch]),
        'names':        [item['name'] for item in batch],
    }


def collate_dual_batch(batch):
    """
    Dual-channel collate: full mesh (list, variable) + sampled points (stacked, fixed).

    Returns:
        vertices_full:     list of (N_i, 3)
        edge_indices_full: list of (2, E_i) — real mesh triangles
        vertices_sample:   (B, num_vertices, 3)
        edge_indices_sample: list of (2, E_sample) — k-NN on samples
        sample_indices:    list of (num_vertices,) long — maps sample→full vertex
        curvature_sample:  (B, num_vertices, 2)
        labels_full:       list of (N_i,)
        labels_sample:     (B, num_vertices,)
        u_gt:              (B, num_vertices, 2) or None
        u_gt_valid:        (B, num_vertices,) bool or None
        material:          (B, 3)
        names:             list of str
    """
    B = len(batch)
    n_sample = batch[0]['vertices_sample'].shape[0]
    has_gt = all(item.get('u_gt') is not None for item in batch)

    vertices_sample = torch.stack([item['vertices_sample'] for item in batch])
    curvature_sample = torch.stack([item['curvature_sample'] for item in batch])
    labels_sample = torch.stack([item['labels_sample'] for item in batch])
    u_gt = torch.stack([item['u_gt'] for item in batch]) if has_gt else None
    u_gt_valid = torch.stack([item['u_gt_valid'] for item in batch]) if has_gt else None
    garment_mask = torch.stack([item['garment_mask'] for item in batch])

    return {
        'vertices_full':      [item['vertices_full'] for item in batch],
        'edge_indices_full':  [item['edge_index_full'] for item in batch],
        'vertices_sample':    vertices_sample,
        'edge_indices_sample': [item['edge_index_sample'] for item in batch],
        'sample_indices':     [item['sample_indices'] for item in batch],
        'curvature_sample':   curvature_sample,
        'labels_full':        [item['labels_full'] for item in batch],
        'labels_sample':      labels_sample,
        'garment_mask':       garment_mask,
        'u_gt':               u_gt,
        'u_gt_valid':         u_gt_valid,
        'material':           torch.stack([item['material'] for item in batch]),
        'names':              [item['name'] for item in batch],
    }


def collate_mesh_padded(batch):
    """
    Batches mesh samples by padding to max_N.

    Returns:
        vertices:  (B, max_N, 3)
        mask:      (B, max_N) bool
        edge_indices: list of (2, E_i)
        curvature: (B, max_N, 2)
        labels:    (B, max_N) long
        u_gt:      (B, max_N, 2)  or None if unavailable
        u_gt_valid: (B, max_N) bool
        material:  (B, 3)
        names:     list of str
    """
    B = len(batch)
    max_n = max(item['vertices'].shape[0] for item in batch)
    has_gt = all(item.get('u_gt') is not None for item in batch)

    vertices = torch.zeros(B, max_n, 3)
    mask = torch.zeros(B, max_n, dtype=torch.bool)
    curvature = torch.zeros(B, max_n, 2)
    labels = torch.zeros(B, max_n, dtype=torch.long)
    u_gt = torch.zeros(B, max_n, 2) if has_gt else None
    u_gt_valid = torch.zeros(B, max_n, dtype=torch.bool) if has_gt else None
    edge_indices = []
    materials = []
    names = []

    for i, item in enumerate(batch):
        n = item['vertices'].shape[0]
        vertices[i, :n] = item['vertices']
        mask[i, :n] = True
        curvature[i, :n] = item['curvature']
        labels[i, :n] = item['labels']
        edge_indices.append(item['edge_index'])
        materials.append(item['material'])
        names.append(item['name'])
        if has_gt and item['u_gt'] is not None:
            u_gt[i, :n] = item['u_gt']
            u_gt_valid[i, :n] = item['u_gt_valid']

    return {
        'vertices':     vertices,
        'mask':         mask,
        'edge_indices': edge_indices,
        'curvature':    curvature,
        'labels':       labels,
        'u_gt':         u_gt,
        'u_gt_valid':   u_gt_valid,
        'material':     torch.stack(materials),
        'names':        names,
    }
