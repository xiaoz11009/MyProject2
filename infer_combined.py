"""
Inference script for PhysUnfolderCombined model.

Outputs:
  - 3D mesh colored by predicted panel labels
  - 2D panel layout (from predicted U_final coordinates)
  - Comparison with ground truth pattern image
"""
import os, sys, json, pickle, argparse, numpy as np, torch, trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely import MultiPoint, concave_hull
from shapely.geometry import Polygon as ShapelyPolygon
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev

sys.path.insert(0, os.path.dirname(__file__))
from model_combined import PhysUnfolderCombined
from data_loader import read_segmentation, fps_sample, build_knn_edges_from_points
from curvature_utils import compute_curvature


# ====================== Colors ======================

PANEL_COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabebe', '#469990', '#e6beff',
    '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
    '#000075', '#a9a9a9', '#dcbeff', '#a9f1e0', '#ff7f50', '#00ffff',
    '#ff00ff', '#008080', '#b8860b', '#006400', '#adff2f', '#ff1493',
    '#7b68ee', '#00fa9a', '#d2691e', '#ff6347', '#8a2be2', '#5f9ea0',
    '#da70d6', '#cd853f', '#bc8f8f', '#4169e1', '#2e8b57', '#6a5acd',
]


# ====================== Model inference ======================

@torch.no_grad()
def predict_full(model, vertices, faces, device, num_vertices=4096):
    """
    Run PhysUnfolderCombined on a full mesh.

    Returns:
        pred_labels: (N,) panel labels for all vertices
        U_final_all: (N, 2) 2D coordinates for all vertices
    """
    N = len(vertices)

    # Normalize
    centroid = vertices.mean(axis=0)
    v_norm = vertices - centroid
    max_dist = np.linalg.norm(v_norm, axis=1).max()
    v_norm = v_norm / max(max_dist, 1e-8)

    # Full mesh data
    verts_full_t = torch.from_numpy(v_norm).float()
    faces_t = torch.from_numpy(faces).long()

    # Real triangle edges for GCN
    from curvature_utils import mesh_faces_to_edges
    edge_index_full = mesh_faces_to_edges(faces_t)

    # Curvature on full mesh
    curv_full = compute_curvature(verts_full_t, faces_t)

    # FPS sample
    if N > num_vertices:
        sample_idx = fps_sample(v_norm, num_vertices)
    else:
        sample_idx = np.arange(N)
    verts_sample = v_norm[sample_idx]
    curv_sample = curv_full[torch.from_numpy(sample_idx).long()]

    # k-NN edges on sampled points (for physics loss in training, unused in inference)
    edge_idx_sample = build_knn_edges_from_points(verts_sample, k=12)

    # To GPU
    v_sample = torch.from_numpy(verts_sample).float().unsqueeze(0).to(device)
    e_sample = [edge_idx_sample.to(device)]
    c_sample = curv_sample.unsqueeze(0).to(device)
    mat = torch.tensor([[1.0, 1.0, 1.0]], device=device)
    si = [torch.from_numpy(sample_idx).long()]

    # Forward (GCN on full mesh → index to FPS → heads)
    model.eval()
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        pred = model.forward_dual(
            [verts_full_t], [edge_index_full],
            v_sample, e_sample, si, c_sample, mat)

    # Extract
    seg_sample = pred['seg_logits'][0].float().cpu()
    U_sample = pred['U_final'][0].float().cpu()

    pred_ids_sample = seg_sample.argmax(dim=-1).numpy()
    U_sample_np = U_sample.numpy()

    # Propagate to all vertices via nearest neighbor
    if N > num_vertices:
        tree = cKDTree(verts_sample)
        _, nn_idx = tree.query(v_norm)
        pred_labels = pred_ids_sample[nn_idx]
        U_all = U_sample_np[nn_idx]
    else:
        pred_labels = pred_ids_sample
        U_all = U_sample_np

    return pred_labels.astype(np.int64), U_all.astype(np.float32)


# ====================== 2D Panel Extraction ======================

def _fit_smooth_boundary(points_2d, hull_ratio=0.25, spline_smooth=0.5,
                         n_out=80, min_points=8):
    """
    Extract a smooth panel boundary from scattered 2D points.

    1. Concave hull → rough ordered boundary
    2. B-spline fit → smooth closed curve
    """
    pts = np.unique(points_2d.round(decimals=4), axis=0)
    if len(pts) < min_points:
        return None

    # Step 1: concave hull for rough boundary
    try:
        mp = MultiPoint(pts)
        hull = concave_hull(mp, ratio=hull_ratio)
        if hull is None or hull.is_empty:
            return None
        if isinstance(hull, ShapelyPolygon):
            boundary = np.array(hull.exterior.coords)
        else:
            boundary = np.array(hull.coords)
    except Exception:
        return None

    if len(boundary) < 4:
        return boundary[:, :2] if boundary.shape[1] >= 2 else None

    boundary = boundary[:, :2]

    # Step 2: B-spline smoothing
    try:
        x, y = boundary[:, 0], boundary[:, 1]
        # Close the loop
        if not np.allclose(boundary[0], boundary[-1]):
            x = np.append(x, x[0])
            y = np.append(y, y[0])

        # Fit periodic B-spline
        tck, u = splprep([x, y], s=spline_smooth, per=1)
        u_new = np.linspace(0, 1, n_out)
        x_smooth, y_smooth = splev(u_new, tck)
        return np.column_stack([x_smooth, y_smooth])
    except Exception:
        # Fallback: return unsmoothed concave hull
        return boundary


def compute_panel_boundary(points_2d, ratio=0.25, min_points=10):
    """Compatibility wrapper (keeps old signature)."""
    return _fit_smooth_boundary(points_2d, hull_ratio=ratio, min_points=min_points)


def layout_panels_grid(panel_boundaries, margin=30.0):
    names = list(panel_boundaries.keys())
    if not names:
        return panel_boundaries

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

def save_3d_segmentation(vertices, faces, pred_labels, save_path):
    colors = np.zeros((len(vertices), 3), dtype=np.uint8)
    for lbl in np.unique(pred_labels):
        mask = pred_labels == lbl
        ch = PANEL_COLORS[lbl % len(PANEL_COLORS)]
        colors[mask] = [int(ch[1:3], 16), int(ch[3:5], 16), int(ch[5:7], 16)]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)
    mesh.export(save_path)
    print(f"  3D → {save_path}")


def save_panel_layout(panel_boundaries, save_path, title="Predicted Panels"):
    if not panel_boundaries:
        print("  无面板边界")
        return
    panel_boundaries = layout_panels_grid(panel_boundaries)

    all_x = np.concatenate([p[:, 0] for p in panel_boundaries.values()])
    all_y = np.concatenate([p[:, 1] for p in panel_boundaries.values()])
    w, h = all_x.max() - all_x.min(), all_y.max() - all_y.min()
    x_mid, y_mid = (all_x.max() + all_x.min()) / 2, (all_y.max() + all_y.min()) / 2
    pad = 0.08
    canvas_w, canvas_h = w * (1 + pad), h * (1 + pad)
    scale = max(canvas_w, canvas_h) * 150 / 800
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
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  2D → {save_path}")


def save_comparison(gt_pattern_path, panel_boundaries, save_path):
    if not os.path.exists(gt_pattern_path):
        return
    gt_img = plt.imread(gt_pattern_path)

    _, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 7))
    ax_l.imshow(gt_img)
    ax_l.set_title('Ground Truth Pattern', fontsize=12, fontweight='bold')
    ax_l.axis('off')

    if panel_boundaries:
        panel_boundaries = layout_panels_grid(panel_boundaries)
        all_x = np.concatenate([p[:, 0] for p in panel_boundaries.values()])
        all_y = np.concatenate([p[:, 1] for p in panel_boundaries.values()])
        x_mid, y_mid = (all_x.max() + all_x.min()) / 2, (all_y.max() + all_y.min()) / 2
        w, h = all_x.max() - all_x.min(), all_y.max() - all_y.min()
        pad = 0.1
        half_w = max(w, h) * (1 + pad) / 2
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

    ax_r.set_title('Predicted Panels (PhysUnfolder)', fontsize=12, fontweight='bold')
    ax_r.axis('off')
    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"  对比 → {save_path}")


# ====================== Main ======================

def main():
    parser = argparse.ArgumentParser(description='PhysUnfolder 推理：分割 + 2D 裁片')
    # parser.add_argument('--model', default='./models_combined/final_model.pth')
    parser.add_argument('--model', default='./models_combined/best_model.pth')
    parser.add_argument('--data_root', default='/home/ddd/zkl/GarmentCodeData/GarmentCodeData_v2')
    parser.add_argument('--num_vertices', type=int, default=4096)
    parser.add_argument('--max_samples', type=int, default=3)
    parser.add_argument('--out_dir', default='./seg_results_combined')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    part_to_id_path = os.path.join(args.data_root, 'part_to_id.npy')
    if part_to_id_path.replace('.npy', '.pkl') and os.path.exists(part_to_id_path.replace('.npy', '.pkl')):
        with open(part_to_id_path.replace('.npy', '.pkl'), 'rb') as f:
            part_to_id = pickle.load(f)
    else:
        part_to_id = np.load(part_to_id_path, allow_pickle=True).item()
    id_to_part = {v: k for k, v in part_to_id.items()}
    num_classes = len(part_to_id)

    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    has_film = any('film.' in k for k in state.keys())

    model = PhysUnfolderCombined(num_classes=num_classes, use_film=has_film).to(device)
    model.load_state_dict(state)

    if 'model_state_dict' in ckpt:
        acc = ckpt.get('val_acc', None)
        print(f"模型: epoch {ckpt['epoch']}, val_acc={acc:.4f}" if acc else f"模型: epoch {ckpt['epoch']}")
    else:
        print(f"模型: state_dict (FiLM={has_film})")
    print(f"面板类别: {num_classes}")

    # Scan samples
    body_dir = os.path.join(args.data_root, 'garments_5000_0', 'default_body')
    samples = sorted([
        d for d in os.listdir(body_dir)
        if d.startswith('rand_') and os.path.isdir(os.path.join(body_dir, d))
    ])[:args.max_samples]

    for name in samples:
        ply_path = os.path.join(body_dir, name, f'{name}_sim.ply')
        seg_path = os.path.join(body_dir, name, f'{name}_sim_segmentation.txt')
        gt_pattern = os.path.join(body_dir, name, f'{name}_pattern.png')
        if not os.path.exists(ply_path) or not os.path.exists(seg_path):
            continue

        print(f"\n{'='*50}\n推理: {name}\n{'='*50}")

        # Load mesh
        mesh = trimesh.load(ply_path, process=False)
        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int64)

        # Load GT labels
        gt_raw = read_segmentation(seg_path, len(vertices), faces)
        gt_ids = np.zeros(len(vertices), dtype=np.int64)
        for i, lbl in enumerate(gt_raw):
            gt_ids[i] = part_to_id.get(lbl, 0)

        # Predict (segmentation + 2D coordinates)
        pred_labels, U_final = predict_full(model, vertices, faces, device, args.num_vertices)

        # Accuracy
        mask = gt_ids != 0
        acc = (pred_labels[mask] == gt_ids[mask]).mean()
        print(f"顶点: {len(vertices)}, 精度: {acc:.3f}")

        # 1. 3D segmentation
        save_3d_segmentation(vertices, faces, pred_labels,
                            os.path.join(args.out_dir, f'{name}_seg3d.ply'))

        # 2. 2D panel layout from U_final
        panel_boundaries = {}
        centroids_norm = vertices - vertices.mean(axis=0)
        max_dist = np.linalg.norm(centroids_norm, axis=1).max()

        for pid in np.unique(pred_labels):
            if pid == 0:  # skip unlabeled
                continue
            panel_mask = pred_labels == pid
            if panel_mask.sum() < 20:
                continue
            panel_2d = U_final[panel_mask]  # predicted 2D coords
            boundary = _fit_smooth_boundary(panel_2d, hull_ratio=0.25,
                                            spline_smooth=0.3, n_out=80)
            if boundary is not None:
                pname = id_to_part.get(pid, f'panel_{pid}')
                # Scale for visualization (3D max_dist ≈ garment size)
                panel_boundaries[pname] = boundary * (max_dist + 1e-8)

        print(f"提取到 {len(panel_boundaries)} 个面板边界")

        if panel_boundaries:
            save_panel_layout(panel_boundaries,
                            os.path.join(args.out_dir, f'{name}_panels.png'))
            save_comparison(gt_pattern, panel_boundaries,
                           os.path.join(args.out_dir, f'{name}_comparison.png'))

    print(f"\n完成，结果在 {args.out_dir}/")


if __name__ == '__main__':
    main()
