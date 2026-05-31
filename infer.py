"""
Inference script for NeuralTailor-based garment panel segmentation.
Predicts panel labels for 3D garment point clouds, then extracts 2D panel boundaries.

Outputs:
  - 3D mesh colored by predicted panel labels
  - 2D panel layout visualization
  - Comparison with ground truth pattern
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely import MultiPoint, concave_hull
from shapely.geometry import Polygon as ShapelyPolygon
from scipy.spatial import cKDTree

from model import NeuralTailorSeg
from data_loader import read_segmentation, fps_sample


# ====================== Model loading ======================

def load_model(model_path, num_classes, feat_dim=128, global_dim=256, hidden_dim=256, k=16, device='cpu'):
    model = NeuralTailorSeg(
        num_classes=num_classes, feat_dim=feat_dim,
        global_dim=global_dim, hidden_dim=hidden_dim, k=k
    ).to(device)

    ckpt = torch.load(model_path, map_location=device,weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, val_acc={ckpt.get('val_acc', '?')}")
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


# ====================== Inference helpers ======================

@torch.no_grad()
def predict_labels(model, vertices, num_points=4096, device='cpu'):
    """Predict panel labels for all vertices of a mesh.

    1. Sample num_points from mesh surface (or FPS from vertices)
    2. Run model inference
    3. Propagate labels back to all vertices via nearest-neighbor
    """
    N = len(vertices)

    # Normalize vertices
    centroid = vertices.mean(axis=0)
    v_norm = vertices - centroid
    max_dist = np.linalg.norm(v_norm, axis=1).max()
    v_norm = v_norm / max(max_dist, 1e-8)

    # Sample points for inference
    if N > num_points:
        # FPS on normalized vertices
        idx = fps_sample(v_norm, num_points)
        pts = v_norm[idx]
        sample_idx = idx
    else:
        pts = v_norm
        sample_idx = np.arange(N)

    # Model inference
    pts_tensor = torch.tensor(pts, dtype=torch.float32).unsqueeze(0).to(device)  # (1, S, 3)
    seg_logits = model(pts_tensor)  # (1, S, C)
    pred_sample = seg_logits.squeeze(0).argmax(dim=-1).cpu().numpy()  # (S,)

    # Propagate labels to all vertices via nearest-neighbor
    if N > num_points:
        tree = cKDTree(v_norm[sample_idx])
        _, nn_idx = tree.query(v_norm)
        pred_all = pred_sample[nn_idx]
    else:
        pred_all = pred_sample

    return pred_all.astype(np.int64)


# ====================== 2D Panel Extraction ======================

def rotation_matrix_3d(rx, ry, rz):
    """Build 3D rotation matrix (X→Y→Z Euler angles)."""
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
    translation = np.array(panel_data['translation'], dtype=float)
    rotation = np.array(panel_data['rotation'], dtype=float)
    pts = points_3d - translation
    R = rotation_matrix_3d(*rotation)
    R_inv = R.T
    pts = pts @ R_inv.T
    return pts[:, :2]


def compute_panel_boundary(points_2d, ratio=0.2, min_points=10):
    """Extract smooth panel boundary using concave hull."""
    pts = np.unique(points_2d.round(decimals=4), axis=0)
    if len(pts) < min_points:
        return None
    try:
        mp = MultiPoint(pts)
        hull = concave_hull(mp, ratio=ratio)
        if hull is None or hull.is_empty:
            return None
        if isinstance(hull, ShapelyPolygon):
            boundary = np.array(hull.exterior.coords)
        else:
            boundary = np.array(hull.coords)
        if len(boundary) >= 3:
            return boundary[:, :2]
    except Exception:
        pass
    return None


def layout_panels_grid(panel_boundaries, margin=30.0):
    """Layout panels in a grid to avoid overlap."""
    names = list(panel_boundaries.keys())
    if not names:
        return panel_boundaries

    # Pair left/right panels
    paired_set = set()
    pairs, singles = [], []
    for name in sorted(names):
        if name in paired_set:
            continue
        other = None
        if name.startswith('left_'):
            other = 'right_' + name[5:]
        elif name.startswith('right_'):
            other = 'left_' + name[6:]
        if other and other in panel_boundaries:
            paired_set.add(name)
            paired_set.add(other)
            pairs.append((name, other))
        else:
            singles.append(name)

    items = [(p[0], p[1], True) for p in pairs] + [(s, None, False) for s in sorted(singles)]
    n = len(items)
    if n == 0:
        return panel_boundaries

    def pw(name):
        p = panel_boundaries[name]
        return p[:, 0].max() - p[:, 0].min()

    def ph(name):
        p = panel_boundaries[name]
        return p[:, 1].max() - p[:, 1].min()

    def item_w(item):
        w = pw(item[0])
        if item[2]:
            w += pw(item[1]) + margin
        return w

    def item_h(item):
        h = ph(item[0])
        if item[2]:
            h = max(h, ph(item[1]))
        return h

    items.sort(key=lambda x: -item_w(x))
    cols = max(2, int(np.ceil(np.sqrt(n))))
    rows = int(np.ceil(n / cols))

    col_w = [0.0] * cols
    row_h = [0.0] * rows
    for i, item in enumerate(items):
        r, c = i // cols, i % cols
        col_w[c] = max(col_w[c], item_w(item))
        row_h[r] = max(row_h[r], item_h(item))

    x_cum = [0.0]
    for w in col_w:
        x_cum.append(x_cum[-1] + w + margin)
    y_cum = [0.0]
    for h in row_h:
        y_cum.append(y_cum[-1] + h + margin)

    new_boundaries = {}
    for i, item in enumerate(items):
        r, c = i // cols, i % cols
        dx = x_cum[c]
        dy = y_cum[r]
        if item[2]:
            l_n, r_n = item[0], item[1]
            l_p = panel_boundaries[l_n].copy()
            r_p = panel_boundaries[r_n].copy()
            new_boundaries[l_n] = l_p + [dx - l_p[:, 0].min(), dy - l_p[:, 1].min()]
            new_boundaries[r_n] = r_p + [dx + pw(l_n) + margin - r_p[:, 0].min(), dy - r_p[:, 1].min()]
        else:
            s_n = item[0]
            s_p = panel_boundaries[s_n].copy()
            new_boundaries[s_n] = s_p + [dx - s_p[:, 0].min(), dy - s_p[:, 1].min()]
    return new_boundaries


# ====================== Visualization ======================

PANEL_COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabebe', '#469990', '#e6beff',
    '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
    '#000075', '#a9a9a9', '#dcbeff', '#a9f1e0', '#ff7f50', '#00ffff',
    '#ff00ff', '#008080', '#b8860b', '#006400', '#adff2f', '#ff1493',
    '#7b68ee', '#00fa9a', '#d2691e', '#ff6347', '#8a2be2', '#5f9ea0',
    '#da70d6', '#cd853f', '#bc8f8f', '#4169e1', '#2e8b57', '#6a5acd',
]


def save_3d_segmentation(vertices, faces, pred_labels, id_to_part, save_path):
    """Save 3D mesh colored by predicted panel labels as PLY file."""
    colors = np.zeros((len(vertices), 3), dtype=np.uint8)
    unique_labels = np.unique(pred_labels)
    for lbl in unique_labels:
        mask = pred_labels == lbl
        color_hex = PANEL_COLORS[lbl % len(PANEL_COLORS)]
        r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
        colors[mask] = [r, g, b]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)
    mesh.export(save_path)
    print(f"  3D 分割结果 → {save_path}")


def save_pred_vs_gt(vertices, faces, pred_labels, gt_labels, id_to_part, save_path):
    """Side-by-side comparison of predicted vs ground truth segmentation on 3D mesh."""
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, labels, title in [(ax1, pred_labels, 'Prediction'), (ax2, gt_labels, 'Ground Truth')]:
        colors = np.zeros((len(vertices), 3), dtype=np.uint8)
        unique_labels = np.unique(labels)
        for lbl in unique_labels:
            mask = labels == lbl
            color_hex = PANEL_COLORS[lbl % len(PANEL_COLORS)]
            r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
            colors[mask] = [r, g, b]
        ax.scatter(vertices[::10, 0], vertices[::10, 2], c=colors[::10] / 255.0, s=0.5, alpha=0.8)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  分割对比 → {save_path}")


def save_panel_layout(panel_boundaries, save_path):
    """Save 2D panel layout as an image."""
    if not panel_boundaries:
        print("  没有有效面板边界")
        return

    panel_boundaries = layout_panels_grid(panel_boundaries, margin=30.0)

    all_x = np.concatenate([p[:, 0] for p in panel_boundaries.values()])
    all_y = np.concatenate([p[:, 1] for p in panel_boundaries.values()])
    w, h = all_x.max() - all_x.min(), all_y.max() - all_y.min()
    x_mid, y_mid = (all_x.max() + all_x.min()) / 2, (all_y.max() + all_y.min()) / 2

    pad = 0.08
    canvas_w, canvas_h = w * (1 + pad), h * (1 + pad)

    target_dpi = 150
    target_px = 800
    scale = max(canvas_w, canvas_h) * target_dpi / target_px
    fig_w, fig_h = canvas_w / scale, canvas_h / scale

    _, ax = plt.subplots(figsize=(fig_w, fig_h))
    for i, (name, pts) in enumerate(panel_boundaries.items()):
        color = PANEL_COLORS[i % len(PANEL_COLORS)]
        ax.add_patch(MplPolygon(pts, facecolor=color, edgecolor='#111111',
                                 linewidth=0.5, alpha=0.85))
        center = np.mean(pts, axis=0)
        ax.text(center[0], center[1], name.replace('_', '\n'), ha='center',
                va='center', fontsize=5, color='#222222')

    ax.set_xlim(x_mid - canvas_w / 2, x_mid + canvas_w / 2)
    ax.set_ylim(y_mid - canvas_h / 2, y_mid + canvas_h / 2)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.savefig(save_path, dpi=target_dpi, bbox_inches='tight')
    plt.close()
    print(f"  2D 面板布局 → {save_path}")


def save_comparison_with_gt(gt_pattern_path, panel_boundaries, save_path):
    """GT pattern image vs predicted panel layout side by side."""
    if not os.path.exists(gt_pattern_path):
        return

    gt_img = plt.imread(gt_pattern_path)

    _, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 7))

    ax_l.imshow(gt_img)
    ax_l.set_title('Ground Truth Pattern', fontsize=12, fontweight='bold')
    ax_l.axis('off')

    if panel_boundaries:
        panel_boundaries = layout_panels_grid(panel_boundaries, margin=25.0)
        all_x = np.concatenate([p[:, 0] for p in panel_boundaries.values()])
        all_y = np.concatenate([p[:, 1] for p in panel_boundaries.values()])
        x_mid, y_mid = (all_x.max() + all_x.min()) / 2, (all_y.max() + all_y.min()) / 2
        w, h = all_x.max() - all_x.min(), all_y.max() - all_y.min()
        pad = 0.1
        half_w, half_h = w * (1 + pad) / 2, h * (1 + pad) / 2
        half_w = max(half_w, half_h)  # force square-ish

        for i, (name, pts) in enumerate(panel_boundaries.items()):
            color = PANEL_COLORS[i % len(PANEL_COLORS)]
            ax_r.add_patch(MplPolygon(pts, facecolor=color, edgecolor='#111111',
                                       linewidth=0.5, alpha=0.85))
            center = np.mean(pts, axis=0)
            ax_r.text(center[0], center[1], name.replace('_', '\n'), ha='center',
                      va='center', fontsize=4.5, color='#222222')

        ax_r.set_xlim(x_mid - half_w, x_mid + half_w)
        ax_r.set_ylim(y_mid - half_w, y_mid + half_w)
        ax_r.set_aspect('equal')

    ax_r.set_title('Predicted Panels (NeuralTailor)', fontsize=12, fontweight='bold')
    ax_r.axis('off')

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"  GT对比 → {save_path}")


# ====================== Main inference ======================

def infer_sample(model, sample_dir, output_dir, id_to_part, num_classes,
                 num_points=4096, device='gpu', id_to_part_map=None):
    """Run inference on a single sample."""
    sample_name = os.path.basename(sample_dir.rstrip('/'))
    ply_path = os.path.join(sample_dir, f'{sample_name}_sim.ply')
    seg_path = os.path.join(sample_dir, f'{sample_name}_sim_segmentation.txt')
    spec_path = os.path.join(sample_dir, f'{sample_name}_specification.json')

    if not os.path.exists(ply_path):
        print(f"  跳过 {sample_name}: 缺少 PLY")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Load mesh
    mesh = trimesh.load(ply_path, process=False)
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)

    # Load ground truth labels
    gt_labels = read_segmentation(seg_path, len(vertices), faces)
    gt_label_ids = np.zeros(len(vertices), dtype=np.int64)
    for i, lbl in enumerate(gt_labels):
        gt_label_ids[i] = id_to_part_map.get(lbl, 0)

    # Predict
    pred_labels = predict_labels(model, vertices, num_points, device)

    # 3D segmentation visualization
    save_3d_segmentation(vertices, faces, pred_labels, id_to_part,
                         os.path.join(output_dir, f'{sample_name}_seg3d.ply'))
    save_pred_vs_gt(vertices, faces, pred_labels, gt_label_ids, id_to_part,
                    os.path.join(output_dir, f'{sample_name}_seg_compare.png'))

    # 2D panel extraction
    if os.path.exists(spec_path):
        with open(spec_path, 'r') as f:
            spec = json.load(f)
        panels = spec['pattern']['panels']

        # Undo normalization
        centroid = vertices.mean(axis=0)
        v_centered = vertices - centroid
        max_dist = np.linalg.norm(v_centered, axis=1).max()

        # Extract panel boundaries
        panel_boundaries = {}
        for panel_name in panels:
            panel_id = id_to_part_map.get(panel_name, -1)
            if panel_id < 0:
                continue
            mask = pred_labels == panel_id
            if mask.sum() < 20:
                continue

            # Project to 2D using spec
            panel_verts_3d = vertices[mask]
            # Move back to original scale
            panel_verts_orig = panel_verts_3d * max(max_dist, 1e-8) + centroid

            try:
                panel_2d = project_to_panel_2d(panel_verts_orig, panels[panel_name])
                boundary = compute_panel_boundary(panel_2d, ratio=0.25)
                if boundary is not None:
                    # Map to spec's 2D space (panels in spec have their own coordinate system)
                    panel_boundaries[panel_name] = boundary
            except Exception:
                pass

        # Save panel layout
        save_panel_layout(panel_boundaries,
                          os.path.join(output_dir, f'{sample_name}_panels.png'))

        # GT comparison
        gt_pattern = os.path.join(sample_dir, f'{sample_name}_pattern.png')
        if os.path.exists(gt_pattern):
            save_comparison_with_gt(gt_pattern, panel_boundaries,
                                    os.path.join(output_dir, f'{sample_name}_comparison.png'))

    print(f"  {sample_name}: {len(panel_boundaries) if 'panel_boundaries' in dir() else 0} 面板 → {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='NeuralTailor 面板分割推理')
    parser.add_argument('--model', default='./models/best_model.pth')
    parser.add_argument('--part_to_id', default='../GarmentCodeData/GarmentCodeData_v2/part_to_id.npy')
    parser.add_argument('--sample_dir', default=None)
    parser.add_argument('--data_root', default='../GarmentCodeData/GarmentCodeData_v2')
    parser.add_argument('--garment_folder', default='garments_5000_0')
    parser.add_argument('--body_type', default='default_body')
    parser.add_argument('--output_root', default='./results')
    parser.add_argument('--max_samples', type=int, default=5)
    parser.add_argument('--num_points', type=int, default=4096)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--global_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"设备: {device}")

    # Load class mapping
    pkl_path = args.part_to_id.replace('.npy', '.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            part_to_id = pickle.load(f)
    else:
        part_to_id = np.load(args.part_to_id, allow_pickle=True).item()
    id_to_part = {v: k for k, v in part_to_id.items()}
    num_classes = len(part_to_id)
    print(f"面板类别数: {num_classes}")

    # Load model
    model = load_model(args.model, num_classes=num_classes,
                       feat_dim=args.feat_dim, global_dim=args.global_dim,
                       hidden_dim=args.hidden_dim, k=args.k, device=device)

    # Run inference
    if args.sample_dir:
        sample_name = os.path.basename(args.sample_dir.rstrip('/'))
        output_dir = os.path.join(args.output_root, sample_name)
        infer_sample(model, args.sample_dir, output_dir, id_to_part,
                     num_classes, args.num_points, device, part_to_id)
    else:
        body_dir = os.path.join(args.data_root, args.garment_folder, args.body_type)
        sample_dirs = sorted([
            d for d in os.listdir(body_dir)
            if os.path.isdir(os.path.join(body_dir, d)) and d.startswith('rand_')
        ])
        for sd in sample_dirs[:args.max_samples]:
            sample_path = os.path.join(body_dir, sd)
            output_dir = os.path.join(args.output_root, sd)
            infer_sample(model, sample_path, output_dir, id_to_part,
                         num_classes, args.num_points, device, part_to_id)

    print("推理完成!")


if __name__ == '__main__':
    main()
