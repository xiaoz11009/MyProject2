"""Fast inference: seam detection (MLP) → cut panels → unfold (GNN) → visualize.

No GT labels needed at inference time. Edge classification + GNN forward pass
takes ~100ms per garment (vs 40-60s for physics optimization).
"""

import os, sys, argparse, time
import numpy as np
import torch
import trimesh

from curvature_utils import compute_curvature, mesh_faces_to_edges
from seam_detector import SeamDetector, predict_seams
from unfold_net import PhysUnfolder, unfold_panels_fast
from edge_detector import cut_mesh_into_panels
from pipeline import (layout_panels_no_overlap, panel_boundary_convex,
                      _load_gt_2d_panels, _parse_svg_panels, PANEL_COLORS,
                      read_segmentation, normalize_label)
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def extract_edge_features_infer(vertices_t, faces_t, curvature):
    """Extract per-edge features (same as in generate_data.py)."""
    N = vertices_t.shape[0]
    all_edges = mesh_faces_to_edges(faces_t)
    E = all_edges.shape[1]
    src, tgt = all_edges[0], all_edges[1]

    H = curvature[:, 1]
    K = curvature[:, 0]
    disc = (H**2 - K).clamp(min=1e-10)
    shape_idx = (2.0 / np.pi) * torch.atan(H / disc.sqrt())

    v = vertices_t
    edge_len = (v[src] - v[tgt]).norm(dim=-1)
    mean_len = edge_len.mean().clamp(min=1e-8)
    norm_len = edge_len / mean_len

    # Dihedral angles
    faces_arr = faces_t.cpu().numpy()
    edge_to_faces = {}
    for fi in range(faces_t.shape[0]):
        a, b, c = int(faces_arr[fi, 0]), int(faces_arr[fi, 1]), int(faces_arr[fi, 2])
        for e in [(min(a,b), max(a,b)), (min(b,c), max(b,c)), (min(c,a), max(c,a))]:
            edge_to_faces.setdefault(e, []).append(fi)

    v0 = v[faces_t[:, 0]]
    v1 = v[faces_t[:, 1]]
    v2 = v[faces_t[:, 2]]
    cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
    face_normals = cross / cross.norm(dim=-1, keepdim=True).clamp(min=1e-10)

    dihedral = torch.zeros(E)
    for ei in range(E):
        a, b = int(src[ei]), int(tgt[ei])
        key = (min(a, b), max(a, b))
        f_indices = edge_to_faces.get(key, [])
        if len(f_indices) >= 2:
            n1 = face_normals[f_indices[0]]
            n2 = face_normals[f_indices[1]]
            cos_val = (n1 * n2).sum().clamp(-1.0, 1.0)
            dihedral[ei] = torch.acos(cos_val).abs().rad2deg() / 180.0

    dH = (H[src] - H[tgt]).abs()
    dK = (K[src] - K[tgt]).abs()
    dShape = (shape_idx[src] - shape_idx[tgt]).abs()
    dH = dH / max(dH.max().item(), 1e-6)
    dK = dK / max(dK.max().item(), 1e-6)

    feats = torch.stack([
        dihedral, dH, dK, norm_len,
        H[src].abs(), H[tgt].abs(),
        K[src].abs(), K[tgt].abs(),
        dShape,
    ], dim=-1).float()

    return feats, all_edges


def infer_garment(ply_path, seam_model, unfold_model, device='cuda',
                  out_dir='./output', threshold=0.5, visualize=True):
    """Full inference pipeline on a single garment.

    Returns:
        panels, unfolded_2d_results
    """
    t0 = time.time()
    sample_name = os.path.basename(ply_path).replace('_sim.ply', '')

    # 1. Load mesh
    mesh = trimesh.load(ply_path, process=False)
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int64)

    verts_t = torch.from_numpy(vertices).float()
    faces_t = torch.from_numpy(faces).long()

    # 2. Extract edge features
    curv = compute_curvature(verts_t, faces_t)
    edge_feats, all_edges = extract_edge_features_infer(verts_t, faces_t, curv)

    # 3. Seam detection (ML inference)
    seam_mask, seam_probs = predict_seams(seam_model, edge_feats, threshold, device)
    seam_mask_cpu = seam_mask.cpu()
    seam_edges = all_edges[:, seam_mask_cpu]
    panel_edges = all_edges[:, ~seam_mask_cpu]
    print(f"  Seam edges: {seam_edges.shape[1]} / {all_edges.shape[1]} "
          f"({100*seam_mask.float().mean():.1f}%)")

    # 4. Cut into panels
    panels = cut_mesh_into_panels(verts_t, faces_t, panel_edges)
    panels = [p for p in panels if p['vertices'].shape[0] >= 30 and p['faces'].shape[0] >= 1]
    print(f"  Panels: {len(panels)}")

    # 5. Unfold each panel (ML inference — milliseconds)
    results_2d = unfold_panels_fast(unfold_model, panels, device)

    dt = time.time() - t0
    print(f"  Inference time: {dt*1000:.0f}ms")

    # 6. Visualization
    if visualize and len(panels) > 0:
        os.makedirs(out_dir, exist_ok=True)
        n_panels = len(panels)
        layout = layout_panels_no_overlap(
            [{'u_2d': u} for u in results_2d])

        # ===== Comparison image: GT layout vs ML layout =====
        spec_path = os.path.join(os.path.dirname(ply_path),
                                 f'{sample_name}_specification.json')
        gt_panels = _load_gt_2d_panels(spec_path) if os.path.exists(spec_path) else {}

        # Match panel names to GT
        seg_path = os.path.join(os.path.dirname(ply_path),
                                f'{sample_name}_sim_segmentation.txt')
        panel_names = [f'P{idx+1}' for idx in range(n_panels)]
        if os.path.exists(seg_path):
            N_mesh = len(trimesh.load(ply_path, process=False).vertices)
            faces_mesh = trimesh.load(ply_path, process=False).faces
            gt_labels = read_segmentation(seg_path, N_mesh, np.array(faces_mesh, dtype=np.int64))
            unique_lbls = sorted(set(l for l in gt_labels if l != 'unlabeled' and not l.startswith('stitch')))
            part_to_id = {lbl: i+1 for i,lbl in enumerate(unique_lbls)}
            id_to_part = {i+1: lbl for i,lbl in enumerate(unique_lbls)}
            vertex_ids = np.array([part_to_id.get(l,0) for l in gt_labels], dtype=np.int64)
            for pi, p in enumerate(panels):
                gidx = p['global_indices'].numpy()
                counts = Counter(l for l in vertex_ids[gidx] if l > 0)
                if counts:
                    panel_names[pi] = id_to_part.get(counts.most_common(1)[0][0], '?')

        fig_comp, (ax_gt, ax_ml) = plt.subplots(1, 2, figsize=(20, 12))

        # Left: GT — Y already flipped in _load_gt_2d_panels
        if gt_panels:
            for idx, (pname, pts) in enumerate(gt_panels.items()):
                color = PANEL_COLORS[idx % len(PANEL_COLORS)]
                ax_gt.fill(pts[:, 0], pts[:, 1], color=color, alpha=0.5,
                          edgecolor='black', linewidth=0.5)
        ax_gt.set_aspect('equal')
        ax_gt.set_title(f'GT LAYOUT — {sample_name}', fontsize=14, fontweight='bold',
                       color='steelblue')
        ax_gt.axis('off')

        # Right: ML unfolding — grid layout
        from scipy.spatial import ConvexHull as CHull
        n_cols = int(np.ceil(np.sqrt(n_panels)))
        # Compute reasonable cell size even if panels are degenerate
        extents = [np.ptp(results_2d[i], axis=0) for i in range(n_panels) if len(results_2d[i]) > 0]
        cell_w = max(1.0, max(e[0] for e in extents) if extents else 1.0) + 20
        cell_h = max(1.0, max(e[1] for e in extents) if extents else 1.0) + 20
        for idx in range(n_panels):
            u = results_2d[idx] - results_2d[idx].mean(axis=0)
            row_i, col_i = idx // n_cols, idx % n_cols
            u = u + np.array([col_i * cell_w, row_i * cell_h])
            color = PANEL_COLORS[idx % len(PANEL_COLORS)]
            try:
                h = CHull(u)
                ax_ml.fill(u[h.vertices,0], u[h.vertices,1], color=color, alpha=0.4,
                          edgecolor='black', linewidth=1)
            except:
                pass
            ax_ml.scatter(u[:,0], u[:,1], s=8, c=color, alpha=0.5)  # always visible
        ax_ml.set_aspect('equal')
        ax_ml.set_title(f'ML UNFOLDING — {n_panels} panels ({dt*1000:.0f}ms)',
                       fontsize=14, fontweight='bold', color='darkred')
        ax_ml.axis('off')

        fig_comp.tight_layout()
        fig_comp.savefig(os.path.join(out_dir, f'{sample_name}_comparison.png'), dpi=150)
        plt.close(fig_comp)

        # ===== Detailed view: individual GT + ML panels =====
        # Show ALL panels from both sides regardless of name matching
        gt_name_list = sorted(gt_panels.keys()) if gt_panels else []
        n_gt = len(gt_name_list)
        n_ml = n_panels
        n_rows = max(n_gt, n_ml) + 1  # +1 for header

        fig = plt.figure(figsize=(12, n_rows * 3.5))
        gs = fig.add_gridspec(n_rows, 2, hspace=0.3, wspace=0.3)

        # Headers
        for col, title, color in [(0, 'GT PANELS', 'steelblue'), (1, 'ML UNFOLDING', 'darkred')]:
            ax = fig.add_subplot(gs[0, col])
            ax.text(0.5, 0.5, title, ha='center', va='center', fontsize=12, fontweight='bold', color=color)
            ax.axis('off')

        from scipy.spatial import ConvexHull as CH2
        for row_idx in range(max(n_gt, n_ml)):
            # GT (left) — show by index, not by name match
            ax_gt_row = fig.add_subplot(gs[row_idx+1, 0])
            if row_idx < n_gt:
                pname = gt_name_list[row_idx]
                pts = gt_panels[pname]
                ax_gt_row.fill(pts[:,0], pts[:,1], color='steelblue', alpha=0.3, edgecolor='steelblue', linewidth=1)
                ax_gt_row.set_title(f'GT: {pname[:22]}', fontsize=7, color='steelblue')
            ax_gt_row.set_aspect('equal'); ax_gt_row.axis('off')

            # ML (right) — show all panels regardless of name match
            ax_ml_row = fig.add_subplot(gs[row_idx+1, 1])
            if row_idx < n_ml:
                u = results_2d[row_idx] - results_2d[row_idx].mean(axis=0)
                pname = panel_names[row_idx]
                color = PANEL_COLORS[row_idx % len(PANEL_COLORS)]
                try:
                    h = CH2(u)
                    ax_ml_row.fill(u[h.vertices,0], u[h.vertices,1], color=color, alpha=0.4, edgecolor='black', linewidth=1)
                except:
                    pass
                ax_ml_row.scatter(u[:,0], u[:,1], s=8, c=color, alpha=0.6)  # always show points
                ax_ml_row.set_title(f'ML: {pname[:22]} ({len(u)}v)', fontsize=7, color='darkred')
            ax_ml_row.set_aspect('equal'); ax_ml_row.axis('off')

        fig.suptitle(sample_name, fontsize=14)
        fig.savefig(os.path.join(out_dir, f'{sample_name}_panels.png'), dpi=150)
        plt.close(fig)
        print(f"  Comparison → {out_dir}/{sample_name}_comparison.png")
        print(f"  Panels    → {out_dir}/{sample_name}_panels.png")

    return panels, results_2d


def main():
    parser = argparse.ArgumentParser(description='Fast ML inference for panel unfolding')
    parser.add_argument('--ply', help='Single PLY file')
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--data_root', default='/home/ddd/zkl/GarmentCodeData/GarmentCodeData_v2')
    parser.add_argument('--max_samples', type=int, default=5)
    parser.add_argument('--out_dir', default='./output')
    parser.add_argument('--seam_model', default='./models/seam_detector.pth')
    parser.add_argument('--unfold_model', default='./models/unfold_net.pth')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load models
    seam_model = SeamDetector().to(device)
    sd = torch.load(args.seam_model, map_location=device)
    if isinstance(sd, dict) and 'model_state' in sd:
        sd = sd['model_state']
    seam_model.load_state_dict(sd)
    seam_model.eval()
    print(f"Seam detector loaded: {args.seam_model}")

    unfold_model = PhysUnfolder().to(device)
    sd = torch.load(args.unfold_model, map_location=device)
    if isinstance(sd, dict) and 'model_state' in sd:
        sd = sd['model_state']
    unfold_model.load_state_dict(sd)
    unfold_model.eval()
    print(f"Unfold net loaded: {args.unfold_model}")

    if args.batch:
        data_dir = os.path.join(args.data_root, 'garments_5000_0', 'default_body')
        samples = [d for d in sorted(os.listdir(data_dir))
                   if d.startswith('rand_') and os.path.isdir(os.path.join(data_dir, d))]
        samples = samples[:args.max_samples]
        for name in samples:
            ply = os.path.join(data_dir, name, f'{name}_sim.ply')
            if not os.path.exists(ply):
                continue
            print(f"\n{'='*50}")
            print(f"{name}")
            print(f"{'='*50}")
            try:
                infer_garment(ply, seam_model, unfold_model, device,
                             args.out_dir, args.threshold)
            except Exception as e:
                print(f"  Error: {e}")
    else:
        if not args.ply:
            print("Specify --ply or --batch")
            return
        infer_garment(args.ply, seam_model, unfold_model, device,
                     args.out_dir, args.threshold)


if __name__ == '__main__':
    main()
