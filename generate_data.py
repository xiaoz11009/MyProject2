"""
Generate training data: run physics pipeline on many garments,
save (3D panel → 2D coords) pairs and (edge features → seam label) pairs.
"""

import os, sys, json, pickle, argparse, time
import numpy as np
import torch
import trimesh

from pipeline import read_segmentation, run_pipeline
from edge_detector import (detect_seam_edges_from_labels,
                           cut_mesh_into_panels, build_seam_correspondences)
from panel_unfolder import unfold_all_panels, refine_with_seams
from curvature_utils import compute_curvature, mesh_faces_to_edges


def extract_edge_features(vertices_t, faces_t, curvature, seam_mask):
    """Extract per-edge geometric features for seam detection training.

    Features per edge (9 dims):
      0: dihedral angle (degrees, normalized)
      1: |Δ mean curvature| across edge
      2: |Δ gaussian curvature|
      3: edge length (normalized by mean edge length)
      4: mean curvature at src
      5: mean curvature at tgt
      6: gaussian curvature at src
      7: gaussian curvature at tgt
      8: shape index |Δ| across edge
    """
    N = vertices_t.shape[0]
    all_edges = mesh_faces_to_edges(faces_t)
    E = all_edges.shape[1]
    src, tgt = all_edges[0], all_edges[1]

    H = curvature[:, 1]  # mean curvature
    K = curvature[:, 0]  # gaussian curvature

    # Shape index: 2/π * arctan(H / sqrt(H² - K))
    disc = (H**2 - K).clamp(min=1e-10)
    shape_idx = (2.0 / np.pi) * torch.atan(H / disc.sqrt())

    # Edge lengths
    v = vertices_t
    edge_len = (v[src] - v[tgt]).norm(dim=-1)
    mean_len = edge_len.mean().clamp(min=1e-8)
    norm_len = edge_len / mean_len

    # Dihedral angle (fast approximation via face normals)
    # Build edge→face mapping
    faces_arr = faces_t.cpu().numpy()
    edge_to_faces = {}
    for fi in range(faces_t.shape[0]):
        a, b, c = int(faces_arr[fi, 0]), int(faces_arr[fi, 1]), int(faces_arr[fi, 2])
        for e in [(min(a,b), max(a,b)), (min(b,c), max(b,c)), (min(c,a), max(c,a))]:
            edge_to_faces.setdefault(e, []).append(fi)

    # Face normals
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
            dihedral[ei] = torch.acos(cos_val).abs().rad2deg() / 180.0  # [0, 1]

    # Curvature differences
    dH = (H[src] - H[tgt]).abs()
    dK = (K[src] - K[tgt]).abs()
    dShape = (shape_idx[src] - shape_idx[tgt]).abs()

    # Normalize differences
    dH = dH / max(dH.max().item(), 1e-6)
    dK = dK / max(dK.max().item(), 1e-6)

    # Stack features: (E, 9)
    feats = torch.stack([
        dihedral,
        dH, dK, norm_len,
        H[src].abs(), H[tgt].abs(),
        K[src].abs(), K[tgt].abs(),
        dShape,
    ], dim=-1).float()

    labels = torch.from_numpy(seam_mask).float()  # (E,) 0 or 1

    return feats, labels, all_edges


def generate(garment_paths, out_dir, device='cuda', n_iter=300,
             max_panels_per_garment=30):
    """Generate training data from a list of garment paths.

    Saves:
      out_dir/seam_data_{idx}.pt    — edge features + labels
      out_dir/panel_data_{idx}.pt   — 3D vertices, faces, 2D targets
    """
    os.makedirs(out_dir, exist_ok=True)
    seam_idx = 0
    panel_idx = 0

    for gi, (ply_path, seg_path) in enumerate(garment_paths):
        try:
            t0 = time.time()

            # Load mesh
            mesh = trimesh.load(ply_path, process=False)
            vertices = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces, dtype=np.int64)
            N = len(vertices)

            # Load labels
            labels = read_segmentation(seg_path, N, faces)
            unique_lbls = sorted(set(l for l in labels if l != 'unlabeled' and not l.startswith('stitch')))
            part_to_id = {lbl: i + 1 for i, lbl in enumerate(unique_lbls)}
            vertex_labels = np.array([part_to_id.get(l, 0) for l in labels], dtype=np.int64)

            verts_t = torch.from_numpy(vertices).float()
            faces_t = torch.from_numpy(faces).long()

            # === SEAM DATA ===
            seam_edges, panel_edges, seam_mask = detect_seam_edges_from_labels(faces_t, vertex_labels)
            curv_full = compute_curvature(verts_t, faces_t)
            edge_feats, edge_labels, all_edges = extract_edge_features(verts_t, faces_t, curv_full, seam_mask)

            # Save seam data (subsample interior edges for balance)
            seam_pos = edge_labels == 1
            seam_neg = edge_labels == 0
            n_pos = seam_pos.sum().item()
            n_neg_sample = min(n_pos * 3, seam_neg.sum().item())
            neg_indices = torch.where(seam_neg)[0][torch.randperm(seam_neg.sum().item())[:n_neg_sample]]
            pos_indices = torch.where(seam_pos)[0]
            keep = torch.cat([pos_indices, neg_indices])

            torch.save({
                'features': edge_feats[keep],
                'labels': edge_labels[keep].long(),
            }, os.path.join(out_dir, f'seam_data_{gi:05d}.pt'))
            if seam_idx % 50 == 0:
                print(f"  seam_data_{gi:05d}: {len(keep)} edges ({n_pos} pos, {n_neg_sample} neg)")

            # === PANEL DATA ===
            panels = cut_mesh_into_panels(verts_t, faces_t, panel_edges)
            panels = [p for p in panels if p['vertices'].shape[0] >= 30 and p['faces'].shape[0] >= 1]

            if len(panels) == 0:
                continue

            # Physics unfold
            device_t = torch.device(device)
            results = unfold_all_panels(panels, device=device, n_iter=n_iter)

            # Seam refinement
            if len(panels) > 1:
                seams = build_seam_correspondences(seam_edges, panels, vertex_labels)
                if len(seams) > 0:
                    refined_u = refine_with_seams(panels, results, seams, device=device, n_iter=100)
                    for i, u in enumerate(refined_u):
                        results[i]['u_2d'] = u

            # Load spec JSON for GT 2D
            sname = os.path.basename(ply_path).replace('_sim.ply', '')
            spec_path = os.path.join(os.path.dirname(ply_path), f'{sname}_specification.json')
            spec_panels = {}
            if os.path.exists(spec_path):
                with open(spec_path, 'r') as f:
                    spec = json.load(f)
                for pname, pdata in spec['pattern']['panels'].items():
                    spec_panels[pname] = {
                        'translation': pdata['translation'],
                        'rotation': pdata['rotation'],
                    }

            # Match panel to spec name
            unique_lbls = sorted(set(l for l in labels if l != 'unlabeled' and not l.startswith('stitch')))
            part_to_id = {lbl: i + 1 for i, lbl in enumerate(unique_lbls)}
            id_to_part = {i + 1: lbl for i, lbl in enumerate(unique_lbls)}

            # Save panel data
            for pi, r in enumerate(results):
                p = panels[pi]
                u_2d = r['u_2d']
                if u_2d is None or len(u_2d) < 3:
                    continue

                # Find GT 2D by matching panel to spec
                gidx = p['global_indices'].numpy()
                from collections import Counter as Ctr
                counts = Ctr(l for l in vertex_labels[gidx] if l > 0)
                gt_u = None
                if counts:
                    best_id = counts.most_common(1)[0][0]
                    pname = id_to_part.get(best_id)
                    if pname and pname in spec_panels:
                        sp = spec_panels[pname]
                        t = np.array(sp['translation'], dtype=np.float32)
                        theta = sp['rotation'][2]  # Z rotation
                        cos_t, sin_t = np.cos(theta), np.sin(theta)
                        Rz = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
                        # GT 2D = Rz.T @ (3D - translation)[:2]
                        v3d = p['vertices'].cpu().numpy()
                        gt_u = (Rz.T @ (v3d[:, :2] - t[:2]).T).T.astype(np.float32)

                # Downsample large panels
                max_v = 3000
                v = p['vertices']
                f = p['faces']
                if v.shape[0] > max_v:
                    idx = np.random.choice(v.shape[0], max_v, replace=False)
                    idx = sorted(idx)
                    old2new = {o: n for n, o in enumerate(idx)}
                    f_arr = f.cpu().numpy()
                    sub_f = [[old2new[fi[0]], old2new[fi[1]], old2new[fi[2]]]
                             for fi in f_arr
                             if fi[0] in old2new and fi[1] in old2new and fi[2] in old2new]
                    v = v[idx]
                    f = torch.tensor(sub_f, dtype=torch.long)
                    u_2d = u_2d[idx]
                    if gt_u is not None:
                        gt_u = gt_u[idx]

                save_dict = {
                    'vertices_3d': v.cpu().float(),
                    'faces': f,
                    'u_2d': torch.from_numpy(u_2d).float(),
                    'material': torch.tensor([1.0, 0.5, 0.3]),  # [E, B, S] default
                }
                if gt_u is not None:
                    save_dict['u_gt'] = torch.from_numpy(gt_u).float()
                torch.save(save_dict, os.path.join(out_dir, f'panel_data_{panel_idx:06d}.pt'))
                panel_idx += 1

            if panel_idx % 50 == 0 and gi > 0:
                print(f"  Garment {gi}: seam edges={seam_edges.shape[1]}, panels={len(panels)}, "
                      f"total panels saved={panel_idx}, time={time.time()-t0:.1f}s")

        except Exception as e:
            print(f"  Error on garment {gi}: {e}")
            continue

    print(f"\nDone! {panel_idx} panels, {gi+1} garments")
    return panel_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/home/ddd/zkl/GarmentCodeData/GarmentCodeData_v2')
    parser.add_argument('--garment_folder', default='garments_5000_0')
    parser.add_argument('--body_type', default='default_body')
    parser.add_argument('--max_garments', type=int, default=500)
    parser.add_argument('--out_dir', default='./training_data')
    parser.add_argument('--n_iter', type=int, default=300)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    data_dir = os.path.join(args.data_root, args.garment_folder, args.body_type)
    entries = sorted([d for d in os.listdir(data_dir)
                      if d.startswith('rand_') and os.path.isdir(os.path.join(data_dir, d))])

    paths = []
    for name in entries[:args.max_garments]:
        ply = os.path.join(data_dir, name, f'{name}_sim.ply')
        seg = os.path.join(data_dir, name, f'{name}_sim_segmentation.txt')
        if os.path.exists(ply) and os.path.exists(seg):
            paths.append((ply, seg))

    print(f"Found {len(paths)} garments")
    generate(paths, args.out_dir, device=args.device, n_iter=args.n_iter)


if __name__ == '__main__':
    main()
