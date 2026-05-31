"""
Data loader for GarmentCodeData compatible with NeuralTailor.
Reads 3D mesh, segmentation labels, and produces point cloud samples.

NeuralTailor input: just XYZ coordinates (3 channels).
EdgeConv learns geometric features automatically from point positions.
"""
import os
import pickle
import numpy as np
import torch
import trimesh
from collections import Counter
from scipy.spatial import cKDTree


# ====================== Label processing ======================

def normalize_label(raw: str) -> str:
    """Normalize labels: stitch vertices → 'seam' for later propagation."""
    if raw.startswith('stitch_'):
        return 'seam'
    return raw


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
    """Uniformly sample points on mesh surface using barycentric coordinates."""
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
    return points.astype(np.float32), face_ids


# ====================== Dataset ======================

class GarmentSegDataset:
    """
    Garment panel segmentation dataset compatible with NeuralTailor.

    Each sample returns:
      - positions: (num_points, 3) normalized XYZ coordinates
      - labels: (num_points,) integer panel class labels
    """

    def __init__(self, data_root, garment_folders=None, body_types=None,
                 num_points=4096, max_samples=None, part_to_id_path=None,
                 normalize=True):
        if garment_folders is None:
            garment_folders = ['garments_5000_0']
        if body_types is None:
            body_types = ['default_body']

        self.data_root = data_root
        self.num_points = num_points
        self.normalize = normalize

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
        print(f"Dataset: {len(self.samples)} 样本, {self.num_classes} 个面板类别")

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
        faces = np.array(mesh.faces, dtype=np.int32)
        N = len(vertices)

        # 2. Load segmentation labels
        labels = read_segmentation(seg_path, N, faces)

        # 3. Map labels to integer IDs (unlabeled → 0)
        label_ids = np.zeros(N, dtype=np.int64)
        for i, lbl in enumerate(labels):
            label_ids[i] = self.part_to_id.get(lbl, 0)

        # 4. Sample points from mesh surface (NeuralTailor-style)
        if N > self.num_points:
            # Uniform surface sampling (same as NeuralTailor's igl.random_points_on_mesh)
            pts, face_ids = uniform_mesh_sample(vertices, faces, self.num_points)
            # Map labels to sampled points via nearest neighbor on mesh
            face_verts = faces[face_ids]
            v0_labels = label_ids[face_verts[:, 0]]
            v1_labels = label_ids[face_verts[:, 1]]
            v2_labels = label_ids[face_verts[:, 2]]
            # Use mode of three face vertices (simple and fast)
            sampled_labels = np.zeros(self.num_points, dtype=np.int64)
            for i in range(self.num_points):
                sampled_labels[i] = np.bincount([v0_labels[i], v1_labels[i], v2_labels[i]]).argmax()
            label_ids = sampled_labels
        elif N < self.num_points:
            # Upsample via surface interpolation
            pts, face_ids = uniform_mesh_sample(vertices, faces, self.num_points)
            face_verts = faces[face_ids]
            v0_labels = label_ids[face_verts[:, 0]]
            v1_labels = label_ids[face_verts[:, 1]]
            v2_labels = label_ids[face_verts[:, 2]]
            sampled_labels = np.zeros(self.num_points, dtype=np.int64)
            for i in range(self.num_points):
                sampled_labels[i] = np.bincount([v0_labels[i], v1_labels[i], v2_labels[i]]).argmax()
            label_ids = sampled_labels
        else:
            pts = vertices.copy()

        # 5. Normalize: center at origin, scale to unit sphere
        if self.normalize:
            centroid = pts.mean(axis=0)
            pts = pts - centroid
            max_dist = np.linalg.norm(pts, axis=1).max()
            pts = pts / max(max_dist, 1e-8)

        return {
            'positions': torch.tensor(pts, dtype=torch.float32),
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'name': sample_name,
        }


def collate_fn(batch):
    """Batches multiple samples together.
    Each sample has same num_points, so we can stack directly.
    """
    positions = torch.stack([item['positions'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    names = [item['name'] for item in batch]
    return positions, labels, names
