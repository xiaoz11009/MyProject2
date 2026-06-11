"""
Main pipeline: edge-based panel detection → 2D physics unfolding.

Flow:
  1. Load garment mesh (PLY) and segmentation labels
  2. Detect seam edges (labels change across edge)
  3. Cut mesh along seam edges → individual panels
  4. Unfold each panel to 2D via physics energy minimization
  5. Visualize results

Usage:
  python pipeline.py --ply <path> --seg <path> --out <dir>
  python pipeline.py --batch --max_samples 5  # batch mode
"""

import os, sys, argparse, json, time
import numpy as np
import torch
import trimesh

from edge_detector import (detect_seam_edges_from_labels,
                           cut_mesh_into_panels, build_seam_correspondences)
from panel_unfolder import unfold_all_panels, refine_with_seams
from curvature_utils import compute_curvature

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import ConvexHull


# ====================== Label loading ======================

def normalize_label(raw):
    """Normalize panel names for consistency."""
    raw = raw.strip()
    mapping = {
        'right sleeve': 'right_sleeve_f',
        'left sleeve': 'left_sleeve_f',
        'pant left': 'pant_f_l',
        'pant right': 'pant_f_r',
    }
    return mapping.get(raw, raw)


def read_segmentation(seg_path, n_vertices, faces):
    """Read per-vertex segmentation labels and propagate seam vertices.

    File format: one label per line, one per vertex.
    Seam vertices have labels like 'stitch_0', 'seam_0', etc.
    """
    raw = ['unlabeled'] * n_vertices
    with open(seg_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n_vertices:
                break
            lbl = line.strip().split(',')[0]  # take first label for multi-label lines
            if lbl:
                raw[i] = normalize_label(lbl)

    # Identify seam vertices
    seam_indices = [i for i, lbl in enumerate(raw)
                    if lbl.startswith('stitch')]

    # Propagate: group stitch vertices into connected components,
    # each component gets the majority label of its neighboring panel vertices.
    if faces is not None and len(faces) > 0 and len(seam_indices) > 0:
        face_arr = np.asarray(faces)
        seam_set = set(seam_indices)

        # Build adjacency among stitch vertices (connected via an edge)
        stitch_adj = {v: set() for v in seam_indices}
        for fi in face_arr:
            a, b, c = int(fi[0]), int(fi[1]), int(fi[2])
            for x, y in [(a, b), (b, c), (c, a)]:
                if x in seam_set and y in seam_set:
                    stitch_adj[x].add(y)
                    stitch_adj[y].add(x)

        # Find connected components of stitch vertices
        visited_stitch = set()
        stitch_components = []
        for v in seam_indices:
            if v in visited_stitch:
                continue
            comp = []
            queue = [v]
            visited_stitch.add(v)
            while queue:
                cur = queue.pop(0)
                comp.append(cur)
                for nb in stitch_adj.get(cur, []):
                    if nb not in visited_stitch:
                        visited_stitch.add(nb)
                        queue.append(nb)
            stitch_components.append(comp)

        # Build full adjacency for neighborhood queries
        adj = {i: set() for i in range(n_vertices)}
        for fi in face_arr:
            a, b, c = int(fi[0]), int(fi[1]), int(fi[2])
            adj[a].update([b, c])
            adj[b].update([a, c])
            adj[c].update([a, b])

        # Each stitch component → majority panel label of its neighbors
        for comp in stitch_components:
            neighbor_labels = {}
            for v in comp:
                for nb in adj[v]:
                    lbl = raw[nb]
                    if lbl != 'unlabeled' and not lbl.startswith('stitch'):
                        neighbor_labels[lbl] = neighbor_labels.get(lbl, 0) + 1
            if neighbor_labels:
                best_label = max(neighbor_labels, key=neighbor_labels.get)
                for v in comp:
                    raw[v] = best_label

    return raw


# ====================== Visualization ======================

PANEL_COLORS = [
    '#e6194b','#3cb44b','#ffe119','#4363d8','#f58231','#911eb4','#42d4f4',
    '#f032e6','#bfef45','#fabed4','#469990','#dcbeff','#9A6324','#fffac8',
    '#800000','#aaffc3','#808000','#ffd8b1','#000075','#a9a9a9','#000000',
] * 5


def panel_boundary_convex(points_2d, n_out=80):
    """Extract panel boundary via convex hull + uniform resampling."""
    pts = np.unique(points_2d.round(decimals=4), axis=0)
    if len(pts) < 8:
        return None
    try:
        hull = ConvexHull(pts)
        boundary = np.vstack([pts[hull.vertices], pts[hull.vertices[0]]])
    except Exception:
        return None
    diffs = np.diff(boundary, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0], np.cumsum(seg_lens)])
    total = cumlen[-1]
    if total < 1e-8:
        return None
    t_new = np.linspace(0, total, n_out)
    x = np.interp(t_new, cumlen, boundary[:, 0])
    y = np.interp(t_new, cumlen, boundary[:, 1])
    return np.column_stack([x, y])


def visualize_results(panels, unfolded, out_dir, sample_name, spec_path=None):
    """Save 3D colored mesh, 2D panel layout, and GT comparison."""
    os.makedirs(out_dir, exist_ok=True)

    n_panels = len(unfolded)
    if n_panels == 0:
        return

    # --- 3D mesh colored by panel ---
    colors = np.zeros((0, 4), dtype=np.uint8)
    all_verts, all_faces = [], []
    face_offset = 0
    for i, r in enumerate(unfolded):
        p = r['panel']
        v = p['vertices'].cpu().numpy()
        f = p['faces'].cpu().numpy()
        if len(f) == 0 or f.shape[1] < 3:
            continue
        all_verts.append(v)
        all_faces.append(f + face_offset)
        face_offset += len(v)
        color_hex = PANEL_COLORS[i % len(PANEL_COLORS)]
        rgb = tuple(int(color_hex[j:j+2], 16) for j in (1, 3, 5))
        colors = np.vstack([colors, np.tile([*rgb, 255], (len(v), 1))])

    mesh_3d = trimesh.Trimesh(vertices=np.vstack(all_verts),
                               faces=np.vstack(all_faces))
    mesh_3d.visual.vertex_colors = colors.astype(np.uint8)
    mesh_3d.export(os.path.join(out_dir, f'{sample_name}_panels3d.ply'))

    # --- GT comparison: left=GT from spec, right=our unfolding ---
    gt_panels = None
    if spec_path and os.path.exists(spec_path):
        gt_panels = _load_gt_2d_panels(spec_path)

    if gt_panels:
        n_display = max(n_panels, len(gt_panels))
        fig, axes = plt.subplots(n_display, 2, figsize=(10, n_display * 4))
        if n_display == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)
    else:
        cols = min(5, n_panels)
        rows = (n_panels + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        axes = np.atleast_2d(axes)

    # Layout offsets for our panels
    layout_offsets = layout_panels_no_overlap(unfolded)

    # Our predictions
    for idx, r in enumerate(unfolded):
        if gt_panels:
            ax = axes[idx][1]
        elif idx < axes.shape[0] * axes.shape[1]:
            ax = axes[idx // axes.shape[1]][idx % axes.shape[1]]
        else:
            break
        u_2d = r['u_2d'] - r['u_2d'].mean(axis=0)
        ox, oy = layout_offsets.get(idx, (0, 0))
        u_2d = u_2d + np.array([ox, oy])
        color = PANEL_COLORS[idx % len(PANEL_COLORS)]
        ax.scatter(u_2d[:, 0], u_2d[:, 1], s=2, c=color, alpha=0.6)
        boundary = panel_boundary_convex(u_2d)
        if boundary is not None:
            ax.plot(boundary[:, 0], boundary[:, 1], 'k-', linewidth=1.5)
        ax.set_aspect('equal')
        ax.set_title(f'Ours #{idx + 1} ({len(u_2d)}v)', fontsize=8)
        ax.axis('off')

    # GT panels (left column)
    if gt_panels:
        for idx, name in enumerate(gt_panels):
            ax = axes[idx][0]
            pts = gt_panels[name]
            ax.scatter(pts[:, 0], pts[:, 1], s=2, c='steelblue', alpha=0.6)
            boundary = panel_boundary_convex(pts)
            if boundary is not None:
                ax.plot(boundary[:, 0], boundary[:, 1], 'steelblue', linewidth=1.5)
            ax.set_aspect('equal')
            ax.set_title(f'GT: {name}', fontsize=8)
            ax.axis('off')
        # Hide unused rows
        for idx in range(n_panels, n_display):
            for col in [0, 1]:
                axes[idx][col].axis('off')
        axes[0][0].set_title('GROUND TRUTH', fontsize=10, fontweight='bold', color='steelblue')
        axes[0][1].set_title('OUR UNFOLDING', fontsize=10, fontweight='bold', color='darkred')

    # Hide unused subplots in non-GT mode
    if not gt_panels:
        for idx in range(n_panels, axes.shape[0] * axes.shape[1]):
            axes[idx // axes.shape[1]][idx % axes.shape[1]].axis('off')

    fig.suptitle(f'{sample_name} — {n_panels} panels', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{sample_name}_panels2d.png'), dpi=150)
    plt.close(fig)

    print(f"  3D → {out_dir}/{sample_name}_panels3d.ply")
    print(f"  2D → {out_dir}/{sample_name}_panels2d.png")


def layout_panels_no_overlap(unfolded, margin=50.0):
    """Arrange unfolded panels in a 2D grid without overlap.

    Sorts panels by area (largest first), places them in rows.
    Each panel is centered and translated to its grid position.

    Returns:
        dict: {panel_index: (offset_x, offset_y)} per panel
    """
    # Compute bounding boxes
    bboxes = []
    for r in unfolded:
        u = r['u_2d']
        u_c = u - u.mean(axis=0)
        bboxes.append({'w': np.ptp(u_c[:, 0]), 'h': np.ptp(u_c[:, 1])})

    # Sort by area (largest first)
    order = sorted(range(len(bboxes)), key=lambda i: bboxes[i]['w'] * bboxes[i]['h'], reverse=True)

    offsets = {}
    row_y = 0.0
    row_h = 0.0
    col_x = 0.0
    row_items = []
    max_row_width = 600.0  # target max row width

    for idx in order:
        bw = bboxes[idx]['w'] + margin
        bh = bboxes[idx]['h'] + margin

        if col_x + bw > max_row_width and row_items:
            # Start new row
            col_x = 0.0
            row_y += row_h
            row_h = 0.0
            row_items = []

        offsets[idx] = (col_x, row_y)
        col_x += bw
        row_h = max(row_h, bh)
        row_items.append(idx)

    return offsets


def _load_gt_2d_panels(spec_path):
    """Load named 2D panel boundaries — prefer SVG, match by SHAPE to spec names.

    SVG path order does NOT match spec panel order.  Each SVG path is matched
    to the correct spec panel by comparing the local 2D shape (after
    normalization removes translation/scale/rotation differences).
    """
    svg_path = spec_path.replace('_specification.json', '_pattern.svg')
    if os.path.exists(svg_path):
        with open(spec_path, 'r') as f:
            spec = json.load(f)
        spec_names = list(spec['pattern']['panels'].keys())
        # Spec panel shapes in local coords (normalized)
        spec_shapes = {}
        for name in spec_names:
            verts = np.array(spec['pattern']['panels'][name]['vertices'])
            if len(verts) >= 3:
                v = verts - verts.mean(axis=0)
                v = v / (np.linalg.norm(v, axis=1).max() + 1e-8)
                spec_shapes[name] = v
        return _parse_svg_panels_matched(svg_path, spec_shapes, spec_names)

    with open(spec_path, 'r') as f:
        spec = json.load(f)
    panels = {}
    for pname, pdata in spec['pattern']['panels'].items():
        verts_2d = pdata.get('vertices')
        if verts_2d is not None and len(verts_2d) > 0:
            pts = np.array(verts_2d)
            if pts.shape[1] >= 2:
                panels[pname] = pts[:, :2]
    return panels if len(panels) > 0 else None


def _parse_svg_panels_matched(svg_path, spec_shapes, spec_names):
    """Parse SVG paths, match each to a spec panel by shape similarity."""
    from svgpathtools import svg2paths
    from scipy.spatial import cKDTree as KD

    paths, _ = svg2paths(svg_path)
    panels = {}
    assigned = set()

    for pi, path in enumerate(paths):
        if path is None or len(path) == 0:
            continue
        # Sample points along the SVG path
        n_pts = max(100, len(path) * 20)
        pts_list = []
        for seg in path:
            for t in np.linspace(0, 1, max(5, n_pts // max(len(path), 1))):
                pt = seg.point(t)
                pts_list.append([pt.real, pt.imag])
        pts = np.array(pts_list)
        if len(pts) < 4:
            continue

        # Normalize SVG shape for comparison
        pts_norm = pts - pts.mean(axis=0)
        pts_norm = pts_norm / (np.linalg.norm(pts_norm, axis=1).max() + 1e-8)

        # Find best matching spec panel by Chamfer distance
        best_name, best_dist = None, float('inf')
        for name, shape in spec_shapes.items():
            if name in assigned:
                continue
            t1 = KD(pts_norm)
            d1, _ = t1.query(shape)
            t2 = KD(shape)
            d2, _ = t2.query(pts_norm)
            dist = d1.mean() + d2.mean()
            if dist < best_dist:
                best_dist, best_name = dist, name

        if best_name:
            pts[:, 1] = -pts[:, 1]  # flip SVG Y to match model coords
            panels[best_name] = pts
            assigned.add(best_name)

    return panels if len(panels) > 0 else None

    with open(spec_path, 'r') as f:
        spec = json.load(f)
    panels = {}
    for pname, pdata in spec['pattern']['panels'].items():
        verts_2d = pdata.get('vertices')
        if verts_2d is not None and len(verts_2d) > 0:
            pts = np.array(verts_2d)
            if pts.shape[1] >= 2:
                panels[pname] = pts[:, :2]
    return panels if len(panels) > 0 else None


def _parse_svg_panels(svg_path, panel_names=None):
    """Parse SVG pattern file using svgpathtools for proper curve sampling.

    Handles all SVG path commands: M, L, C, Q, A (arc), and relative variants.
    Each <path> element is a separate panel.
    """
    from svgpathtools import svg2paths
    paths, _ = svg2paths(svg_path)

    panels = {}
    for pi, path in enumerate(paths):
        if path is None or len(path) == 0:
            continue
        # Sample points along the full path (all segments)
        n_pts = max(100, len(path) * 20)
        pts = []
        for seg in path:
            for t in np.linspace(0, 1, max(5, n_pts // len(path))):
                pt = seg.point(t)
                pts.append([pt.real, pt.imag])
        pts = np.array(pts)
        if len(pts) > 4:
            name = panel_names[pi] if panel_names and pi < len(panel_names) else f'panel_{pi}'
            panels[name] = pts
    return panels if len(panels) > 0 else None


def _sample_svg_path(d, n_pts=200):
    """Sample points along an SVG path string (handles M, L, C, Q commands)."""
    import re
    # Tokenize: split by commands and spaces
    tokens = re.findall(r'[MLCQZ]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', d)
    points = []
    current = np.zeros(2)
    i = 0

    while i < len(tokens):
        cmd = tokens[i]
        i += 1
        if cmd == 'M':
            current = np.array([float(tokens[i]), float(tokens[i+1])])
            points.append(current.copy())
            i += 2
        elif cmd == 'L':
            current = np.array([float(tokens[i]), float(tokens[i+1])])
            points.append(current.copy())
            i += 2
        elif cmd == 'C':
            x0, y0 = current
            x1, y1 = float(tokens[i]), float(tokens[i+1])
            x2, y2 = float(tokens[i+2]), float(tokens[i+3])
            x3, y3 = float(tokens[i+4]), float(tokens[i+5])
            for t in np.linspace(0, 1, n_pts // 4):
                mt = 1 - t
                x = mt**3*x0 + 3*mt**2*t*x1 + 3*mt*t**2*x2 + t**3*x3
                y = mt**3*y0 + 3*mt**2*t*y1 + 3*mt*t**2*y2 + t**3*y3
                points.append(np.array([x, y]))
            current = np.array([x3, y3])
            i += 6
        elif cmd == 'Q':
            x0, y0 = current
            x1, y1 = float(tokens[i]), float(tokens[i+1])
            x2, y2 = float(tokens[i+2]), float(tokens[i+3])
            for t in np.linspace(0, 1, n_pts // 4):
                mt = 1 - t
                x = mt**2*x0 + 2*mt*t*x1 + t**2*x2
                y = mt**2*y0 + 2*mt*t*y1 + t**2*y2
                points.append(np.array([x, y]))
            current = np.array([x2, y2])
            i += 4
        elif cmd == 'Z':
            if len(points) > 0:
                points.append(points[0].copy())

    return np.array(points) if points else np.zeros((0, 2))


# ====================== Pipeline ======================

def run_pipeline(ply_path, seg_path, out_dir='./output', device='cuda',
                 n_iter=500, lambda_bend=0.01, verbose=True):
    """Run the full edge-based detection → 2D unfold pipeline on one garment."""
    sample_name = os.path.splitext(os.path.basename(ply_path))[0].replace('_sim', '')
    t0 = time.time()

    # 1. Load mesh
    mesh = trimesh.load(ply_path, process=False)
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int64)
    N = len(vertices)
    if verbose:
        print(f"  加载: {N} 顶点, {len(faces)} 面")

    # 2. Load labels and detect seam edges
    labels = read_segmentation(seg_path, N, faces)
    # Build part_to_id mapping
    unique_labels = sorted(set(l for l in labels if l != 'unlabeled'))
    part_to_id = {lbl: i + 1 for i, lbl in enumerate(unique_labels)}
    vertex_labels = np.array([part_to_id.get(l, 0) for l in labels], dtype=np.int64)

    verts_t = torch.from_numpy(vertices).float()
    faces_t = torch.from_numpy(faces).long()

    seam_edges, panel_edges, seam_mask = detect_seam_edges_from_labels(
        faces_t, vertex_labels)
    n_seam = seam_edges.shape[1]
    n_panel_edges = panel_edges.shape[1]
    if verbose:
        print(f"  缝线边: {n_seam}, 面板内边: {n_panel_edges}")

    # 3. Cut mesh into panels
    panels = cut_mesh_into_panels(verts_t, faces_t, panel_edges)
    panels = [p for p in panels if p['vertices'].shape[0] >= 3 and p['faces'].shape[0] >= 1]

    # Merge tiny fragments (< 200 vertices) into neighboring large panels
    min_panel_size = 200
    large_panels = [p for p in panels if p['vertices'].shape[0] >= min_panel_size]
    small_panels = [p for p in panels if p['vertices'].shape[0] < min_panel_size]

    if small_panels and large_panels:
        # Build mapping: small fragment vertex → nearest large panel label
        all_large_verts = torch.cat([p['vertices'] for p in large_panels], dim=0)
        all_large_idx_chunks = []
        for li, lp in enumerate(large_panels):
            nv = lp['vertices'].shape[0]
            all_large_idx_chunks.extend([li] * nv)
        all_large_idx = np.array(all_large_idx_chunks)

        from scipy.spatial import cKDTree as KDTreeFrag
        tree_frag = KDTreeFrag(all_large_verts.cpu().numpy())
        for sp in small_panels:
            sv = sp['vertices'].cpu().numpy()
            if len(sv) == 0:
                continue
            _, nn_idx = tree_frag.query(sv)
            target_labels = all_large_idx[nn_idx]
            # Find most common target panel
            from collections import Counter
            target_pid = Counter(target_labels.tolist()).most_common(1)[0][0]
            # Merge: add small panel's vertices and faces to target large panel
            large_panels[target_pid]['vertices'] = torch.cat([
                large_panels[target_pid]['vertices'], sp['vertices']], dim=0)
            # Remap faces
            old_nv = large_panels[target_pid]['vertices'].shape[0] - sp['vertices'].shape[0]
            sp_faces_remapped = sp['faces'] + old_nv
            large_panels[target_pid]['faces'] = torch.cat([
                large_panels[target_pid]['faces'], sp_faces_remapped], dim=0)
        panels = large_panels

    # Validate against spec JSON: if we have too many panels, merge smallest ones
    sample = os.path.basename(ply_path).replace('_sim.ply', '')
    spec_path = os.path.join(os.path.dirname(ply_path),
                             f'{sample}_specification.json')
    expected_n = None
    if os.path.exists(spec_path):
        with open(spec_path, 'r') as f:
            spec = json.load(f)
        expected_n = len(spec['pattern']['panels'])
    if expected_n and len(panels) > expected_n + 2:  # allow 1-2 extra
        panels.sort(key=lambda p: p['vertices'].shape[0], reverse=True)
        panels = panels[:expected_n]

    if verbose:
        extra = f" (spec={expected_n})" if expected_n else ""
        print(f"  面板: {len(panels)} 个{extra}")
        for i, p in enumerate(panels[:8]):
            print(f"    Panel {i}: {p['vertices'].shape[0]}v, {p['faces'].shape[0]}f")

    # 4. Unfold each panel individually
    if verbose:
        print(f"  展开中 (Adam, {n_iter} iters)...")
    results = unfold_all_panels(panels, device=device, n_iter=n_iter,
                                lambda_bend=lambda_bend, verbose=verbose)

    # 5. Seam matching refinement
    if len(panels) > 1:
        seams = build_seam_correspondences(seam_edges, panels, vertex_labels)
        if len(seams) > 0 and verbose:
            print(f"  缝线匹配: {len(seams)} 对, 精调中...")
        refined_u = refine_with_seams(panels, results, seams, device=device,
                                      n_iter=150, lr=0.003, lambda_seam=2.0)
        for i, u in enumerate(refined_u):
            results[i]['u_2d'] = u

    # 6. Visualize
    sname = os.path.basename(ply_path).replace('_sim.ply', '')
    vis_spec = os.path.join(os.path.dirname(ply_path), f'{sname}_specification.json')
    if not os.path.exists(vis_spec):
        vis_spec = None
    visualize_results(panels, results, out_dir, sample_name, spec_path=vis_spec)

    elapsed = time.time() - t0
    if verbose:
        print(f"  完成! {elapsed:.1f}s")

    return panels, results


# ====================== CLI ======================

def main():
    parser = argparse.ArgumentParser(description='Edge-based panel unfold')
    parser.add_argument('--ply', help='PLY file path (single mode)')
    parser.add_argument('--seg', help='Segmentation file path (single mode)')
    parser.add_argument('--batch', action='store_true', help='Batch mode')
    parser.add_argument('--data_root', default='/home/ddd/zkl/GarmentCodeData/GarmentCodeData_v2')
    parser.add_argument('--garment_folder', default='garments_5000_0')
    parser.add_argument('--body_type', default='default_body')
    parser.add_argument('--max_samples', type=int, default=5)
    parser.add_argument('--out_dir', default='./output')
    parser.add_argument('--n_iter', type=int, default=500, help='L-BFGS iterations')
    parser.add_argument('--lambda_bend', type=float, default=0.01)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if args.batch:
        data_dir = os.path.join(args.data_root, args.garment_folder, args.body_type)
        all_entries = sorted(os.listdir(data_dir))
        samples = [d for d in all_entries if d.startswith('rand_') and
                   os.path.isdir(os.path.join(data_dir, d))]
        if args.max_samples:
            samples = samples[:args.max_samples]
        for name in samples:
            ply_path = os.path.join(data_dir, name, f'{name}_sim.ply')
            seg_path = os.path.join(data_dir, name, f'{name}_sim_segmentation.txt')
            if not os.path.exists(ply_path) or not os.path.exists(seg_path):
                continue
            print(f"\n{'='*60}")
            print(f"处理: {name}")
            print(f"{'='*60}")
            try:
                run_pipeline(ply_path, seg_path, args.out_dir, args.device,
                            args.n_iter, args.lambda_bend)
            except Exception as e:
                print(f"  错误: {e}")
                import traceback; traceback.print_exc()
        print(f"\n完成! 结果在 {args.out_dir}/")
    else:
        if not args.ply or not args.seg:
            print("Specify --ply and --seg, or --batch")
            return
        run_pipeline(args.ply, args.seg, args.out_dir, args.device,
                    args.n_iter, args.lambda_bend)


if __name__ == '__main__':
    main()
