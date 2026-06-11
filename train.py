"""
Train seam detector and PhysUnfolder on physics-generated data.

PhysUnfolder (DOCX architecture):
  GCN encoder → FiLM material modulation → Geo Head + Curvature-Gated Relax Head
  Loss: L_sup (MSE to PCA-aligned GT) + 0.1*L_mem(curv-gated) + 0.01*L_bend

Usage:
  python train.py --data_dir ./training_data --epochs 50
  python train.py --skip_seam --epochs_unfold 150 --resume_unfold ./models/unfold_checkpoint.pth
"""

import os, sys, argparse, glob, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from seam_detector import SeamDetector
from unfold_net import PhysUnfolder
from curvature_utils import compute_curvature, mesh_faces_to_edges


# ====================== Datasets ======================

class SeamDataset(Dataset):
    """Loads edge features + seam labels from .pt files."""
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, 'seam_data_*.pt')))
        self.features = []
        self.labels = []
        for f in self.files:
            d = torch.load(f, map_location='cpu')
            self.features.append(d['features'])
            self.labels.append(d['labels'])
        self.features = torch.cat(self.features, dim=0)
        self.labels = torch.cat(self.labels, dim=0)
        print(f"SeamDataset: {len(self.features)} edges, "
              f"pos={self.labels.sum().item()}, neg={len(self.labels) - self.labels.sum().item()}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class PanelDataset(Dataset):
    """Loads (3D panel → 2D target + material) pairs."""
    def __init__(self, data_dir, max_vertices=3000,
                 default_mat=(1.0, 0.5, 0.3)):
        self.files = sorted(glob.glob(os.path.join(data_dir, 'panel_data_*.pt')))
        self.max_v = max_vertices
        self.default_mat = torch.tensor(default_mat, dtype=torch.float32)
        print(f"PanelDataset: {len(self.files)} panels")
        # Count panels with GT
        gt_count = 0
        for f in self.files[:100]:  # sample first 100 to estimate
            d = torch.load(f, map_location='cpu')
            if 'u_gt' in d:
                gt_count += 1
        if gt_count > 0:
            print(f"  (sampled {min(100, len(self.files))} files: {gt_count} have u_gt)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], map_location='cpu')
        v = d['vertices_3d']
        f = d['faces']
        u = d.get('u_gt', d['u_2d'])  # prefer GT 2D pattern
        mat = d.get('material', self.default_mat)
        if not isinstance(mat, torch.Tensor):
            mat = torch.tensor(mat, dtype=torch.float32)

        if v.shape[0] > self.max_v:
            idx_r = torch.randperm(v.shape[0])[:self.max_v]
            idx_r = idx_r.sort().values
            v = v[idx_r]
            u = u[idx_r]
            old_set = set(idx_r.tolist())
            old2new = {o: n for n, o in enumerate(idx_r.tolist())}
            f_new = []
            for fi in range(f.shape[0]):
                a, b, c = int(f[fi, 0]), int(f[fi, 1]), int(f[fi, 2])
                if a in old_set and b in old_set and c in old_set:
                    f_new.append([old2new[a], old2new[b], old2new[c]])
            f = torch.tensor(f_new, dtype=torch.long) if f_new else torch.zeros(1, 3, dtype=torch.long)

        return v, f, u, mat


def collate_panels(batch):
    verts  = [b[0] for b in batch]
    faces  = [b[1] for b in batch]
    tgts   = [b[2] for b in batch]
    mats   = [b[3] for b in batch]
    return verts, faces, tgts, mats


# ====================== Training ======================

def train_seam_detector(model, loader, device, epochs=30, lr=1e-3):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, total_acc = 0.0, 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"Seam Epoch {epoch+1:3d}/{epochs}")
        for feats, labels in pbar:
            feats, labels = feats.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_acc += (logits.argmax(-1) == labels).float().mean().item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{total_loss/n_batches:.4f}',
                             'acc': f'{total_acc/n_batches:.4f}'})
        sched.step()
    return model


def train_phys_unfolder(model, loader, device, epochs=50, lr=1e-3, epoch_offset=0,
                        save_dir=None, save_every=10, opt_state=None, sched_state=None):
    """Train PhysUnfolder: physics losses + supervised GT alignment."""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    if opt_state is not None:
        opt.load_state_dict(opt_state)
    if sched_state is not None:
        sched.load_state_dict(sched_state)

    ckpt_path = os.path.join(save_dir, 'unfold_checkpoint.pth') if save_dir else None

    for epoch in range(epochs):
        model.train()
        total_loss, total_strain, n_panels = 0.0, 0.0, 0
        real_epoch = epoch + 1 + epoch_offset
        total_epochs = epochs + epoch_offset
        pbar = tqdm(loader, desc=f"PhysUnfold {real_epoch:3d}/{total_epochs}")

        for verts_list, faces_list, targets_list, mats_list in pbar:
            opt.zero_grad()
            batch_loss_scalar = 0.0
            batch_strain_scalar = 0.0
            n_valid = 0
            batch_loss_tensor = 0.0

            for v, f, t, mat in zip(verts_list, faces_list, targets_list, mats_list):
                if v.shape[0] < 3 or f.shape[0] < 1:
                    continue
                v, f, t, mat = v.to(device), f.to(device), t.to(device), mat.to(device)

                # Normalize
                v_center = v.mean(dim=0, keepdim=True)
                v_norm = v - v_center
                scale = v_norm.norm(dim=-1).max().clamp(min=1e-6)
                v_input = v_norm / (scale + 1e-8)

                # Build edge_index and curvature
                edge_index = mesh_faces_to_edges(f)
                curv = compute_curvature(v, f)

                # PhysUnfolder forward
                pred, delta = model(v_input, edge_index, curv, mat)
                # pred: (N, 2), delta: (N, 2)

                src, tgt = edge_index[0], edge_index[1]

                # ---- Membrane: curvature-gated edge length preservation ----
                L_pred = (pred[src] - pred[tgt]).norm(dim=-1)
                L_3d_norm = (v_norm[src] - v_norm[tgt]).norm(dim=-1) / (scale + 1e-8)
                strain = (L_pred - L_3d_norm) / L_3d_norm.clamp(min=1e-6)
                strain = strain.clamp(-5.0, 5.0)

                # Per-edge curvature gate (panel-normalized, same as model)
                K_abs = curv[:, 0].abs().detach()
                K_edge = (K_abs[src] + K_abs[tgt]) / 2
                K_min, K_max = K_edge.min(), K_edge.max()
                K_edge_norm = (K_edge - K_min) / (K_max - K_min).clamp(min=1e-6)
                gate_edge = torch.sigmoid(model.curv_gamma * K_edge_norm + model.curv_beta)
                L_mem = (((1 - gate_edge) * strain ** 2) + 0.1 * (gate_edge * strain ** 2)).mean()

                # ---- Bending: dihedral angle preservation (vectorized) ----
                N_vtx = v.shape[0]
                F_n = f.shape[0]
                # Face normals in 3D
                v0 = v_norm[f[:,0]]; v1 = v_norm[f[:,1]]; v2 = v_norm[f[:,2]]
                fn_3d = torch.cross(v1-v0, v2-v0, dim=-1)
                fn_3d = fn_3d / fn_3d.norm(dim=-1, keepdim=True).clamp(1e-8)
                # Face normals in 2D (z=0)
                p0 = torch.cat([pred[f[:,0]], torch.zeros(F_n,1,device=device)], dim=-1)
                p1 = torch.cat([pred[f[:,1]], torch.zeros(F_n,1,device=device)], dim=-1)
                p2 = torch.cat([pred[f[:,2]], torch.zeros(F_n,1,device=device)], dim=-1)
                fn_2d = torch.cross(p1-p0, p2-p0, dim=-1)
                fn_2d = fn_2d / fn_2d.norm(dim=-1, keepdim=True).clamp(1e-8)

                # Vectorized edge→face hash table
                all_edges = mesh_faces_to_edges(f)
                es, et = all_edges[0], all_edges[1]
                f_np = f.cpu().numpy()
                a_np = f_np[:,0].astype(np.int64); b_np = f_np[:,1].astype(np.int64); c_np = f_np[:,2].astype(np.int64)
                e0_h = np.minimum(a_np, b_np) * N_vtx + np.maximum(a_np, b_np)
                e1_h = np.minimum(b_np, c_np) * N_vtx + np.maximum(b_np, c_np)
                e2_h = np.minimum(c_np, a_np) * N_vtx + np.maximum(c_np, a_np)
                all_h = np.concatenate([e0_h, e1_h, e2_h])
                all_fi = np.tile(np.arange(F_n, dtype=np.int64), 3)
                sort_idx = np.argsort(all_h)
                sorted_h = all_h[sort_idx]
                sorted_fi = all_fi[sort_idx]

                n_bend = min(all_edges.shape[1], 500)
                rand_idx = torch.randperm(all_edges.shape[1], device=device)[:n_bend]
                es_np = es[rand_idx].cpu().numpy()
                et_np = et[rand_idx].cpu().numpy()
                bend_vals = []
                for ei in range(n_bend):
                    a_e, b_e = int(es_np[ei]), int(et_np[ei])
                    h = min(a_e, b_e) * N_vtx + max(a_e, b_e)
                    left = np.searchsorted(sorted_h, h, side='left')
                    right = np.searchsorted(sorted_h, h, side='right')
                    if right - left >= 2:
                        fi0, fi1 = int(sorted_fi[left]), int(sorted_fi[left+1])
                        cos3d = (fn_3d[fi0] * fn_3d[fi1]).sum().clamp(-1,1)
                        cos2d = (fn_2d[fi0] * fn_2d[fi1]).sum().clamp(-1,1)
                        bend_vals.append((cos2d - cos3d).abs())
                L_bend = torch.stack(bend_vals).mean() if bend_vals else 0.0 * L_mem

                # ---- Supervised: PCA-aligned GT → stable target ----
                # Align GT to 3D PCA frame ONCE (fixed ref, NOT model output)
                v_centered_for_pca = v_norm  # already centered
                _, _, Vh = torch.linalg.svd(v_centered_for_pca, full_matrices=False)
                v_pca = v_centered_for_pca @ Vh[:2].t()  # (N, 2) PCA reference

                # Normalize GT and PCA ref to unit scale for alignment
                gt_norm = t - t.mean(dim=0)
                gt_scale = gt_norm.norm(dim=-1).max().clamp(min=1e-6)
                gt_unit = gt_norm / gt_scale

                v_pca_norm = v_pca - v_pca.mean(dim=0)
                v_pca_scale = v_pca_norm.norm(dim=-1).max().clamp(min=1e-6)
                v_pca_unit = v_pca_norm / v_pca_scale

                # Procrustes: find rotation R that maps GT → PCA frame (fixed per panel)
                H = v_pca_unit.T @ gt_unit  # (2, 2)
                U_r, _, Vt_r = torch.linalg.svd(H)
                R = Vt_r.T @ U_r.T
                gt_aligned = gt_unit @ R.T  # GT in PCA frame (fixed target)

                # Direct MSE: model learns to output in PCA frame
                # Normalize pred to unit scale for fair comparison
                pred_norm = pred - pred.mean(dim=0)
                pred_scale = pred_norm.norm(dim=-1).max().clamp(min=1e-6)
                pred_unit = pred_norm / pred_scale
                L_sup = F.mse_loss(pred_unit, gt_aligned)

                # === Total loss: supervised primary + physics regularization ===
                loss = L_sup + 0.1 * L_mem + 0.01 * L_bend
                batch_loss_tensor += loss
                batch_loss_scalar += loss.item()
                batch_strain_scalar += strain.abs().mean().item()
                n_valid += 1
                n_panels += 1
                if n_panels <= 3:
                    print(f"\n  DEBUG p{n_panels}: L2d={L_pred.mean():.4f} L3d={L_3d_norm.mean():.4f} "
                          f"strain={strain.abs().mean():.4f} sup={L_sup.item():.4f} "
                          f"gate={gate_edge.mean():.3f}")

            if n_valid > 0:
                batch_loss_tensor = batch_loss_tensor / n_valid
            batch_loss_tensor.backward()
            opt.step()
            total_loss += batch_loss_scalar
            total_strain += batch_strain_scalar
            pbar.set_postfix({'loss': f'{total_loss/max(n_panels,1):.4f}',
                             'strain': f'{100*total_strain/max(n_panels,1):.1f}%',
                             'n': f'{n_panels}'})
        sched.step()

        # Save checkpoint
        if ckpt_path and save_every > 0 and (real_epoch % save_every == 0 or real_epoch == total_epochs):
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': opt.state_dict(),
                'scheduler_state': sched.state_dict(),
                'epoch': real_epoch,
            }, ckpt_path)
            pbar.write(f"  Checkpoint saved (epoch {real_epoch}) -> {ckpt_path}")

    return model


# ====================== Main ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./training_data')
    parser.add_argument('--epochs_seam', type=int, default=30)
    parser.add_argument('--epochs_unfold', type=int, default=50)
    parser.add_argument('--batch_size_seam', type=int, default=16384)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', default='./models')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--skip_seam', action='store_true')
    parser.add_argument('--skip_unfold', action='store_true')
    parser.add_argument('--resume_unfold', default=None)
    parser.add_argument('--save_every', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    # === Seam Detector ===
    if not args.skip_seam:
        print("\n" + "=" * 50)
        print("Stage 1: Seam Edge Detector")
        print("=" * 50)
        ds = SeamDataset(args.data_dir)
        loader = DataLoader(ds, batch_size=args.batch_size_seam, shuffle=True)
        model_s = SeamDetector()
        model_s = train_seam_detector(model_s, loader, device, epochs=args.epochs_seam, lr=args.lr)
        torch.save(model_s.state_dict(), os.path.join(args.save_dir, 'seam_detector.pth'))
        print(f"Saved -> {args.save_dir}/seam_detector.pth")

    # === PhysUnfolder ===
    if not args.skip_unfold:
        print("\n" + "=" * 50)
        print("Stage 2: PhysUnfolder (GCN + FiLM + Curvature Gate)")
        print("=" * 50)
        ds = PanelDataset(args.data_dir)
        loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_panels)
        start_epoch = 0
        opt_state, sched_state = None, None

        if args.resume_unfold:
            model_u = PhysUnfolder()
            ckpt = torch.load(args.resume_unfold, map_location='cpu')
            if isinstance(ckpt, dict) and 'model_state' in ckpt:
                model_u.load_state_dict(ckpt['model_state'])
                start_epoch = ckpt.get('epoch', 0)
                opt_state = ckpt.get('optimizer_state', None)
                sched_state = ckpt.get('scheduler_state', None)
                print(f"Resumed from epoch {start_epoch}: {args.resume_unfold}")
            else:
                model_u.load_state_dict(ckpt)
                print(f"Resumed from legacy checkpoint: {args.resume_unfold}")
        else:
            model_u = PhysUnfolder()

        remaining = args.epochs_unfold - start_epoch
        model_u = train_phys_unfolder(model_u, loader, device, epochs=remaining, lr=args.lr,
                                      epoch_offset=start_epoch, save_dir=args.save_dir,
                                      save_every=args.save_every, opt_state=opt_state,
                                      sched_state=sched_state)
        torch.save({'model_state': model_u.state_dict(), 'epoch': args.epochs_unfold},
                   os.path.join(args.save_dir, 'unfold_net.pth'))
        print(f"Saved -> {args.save_dir}/unfold_net.pth")

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
