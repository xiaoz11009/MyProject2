"""
Training script for NeuralTailor-based garment panel segmentation.

Architecture: EdgeConvFeatures backbone + PointSegHead.
Uses cross-entropy loss with per-point panel labels from GarmentCodeData.

Based on NeuralTailor (Korosteleva & Lee, SIGGRAPH 2022).
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_loader import GarmentSegDataset, collate_fn
from model import NeuralTailorSeg


# ====================== Metrics ======================

def compute_accuracy(pred, target, ignore_idx=0):
    """Compute per-point accuracy, ignoring unlabeled points."""
    mask = target != ignore_idx
    if mask.sum() == 0:
        return 1.0
    correct = (pred[mask] == target[mask]).sum().item()
    return correct / mask.sum().item()


def compute_iou(pred, target, num_classes, ignore_idx=0):
    """Compute mean IoU over classes, ignoring unlabeled."""
    ious = []
    for c in range(num_classes):
        if c == ignore_idx:
            continue
        pred_c = pred == c
        target_c = target == c
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0


# ====================== Training ======================

def train_epoch(model, loader, optimizer, device, epoch, ignore_idx=0):
    model.train()
    total_loss = 0
    total_correct = 0
    total_valid = 0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}")
    for positions, labels, names in pbar:
        positions = positions.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        seg_logits = model(positions)  # (B, N, C)
        B, N, C = seg_logits.shape

        # Flatten for cross-entropy
        seg_flat = seg_logits.view(-1, C)
        labels_flat = labels.view(-1)

        loss = F.cross_entropy(seg_flat, labels_flat, ignore_index=ignore_idx)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        pred = seg_logits.argmax(dim=-1)
        mask = labels != ignore_idx
        total_correct += (pred[mask] == labels[mask]).sum().item()
        total_valid += mask.sum().item()
        n_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{total_correct / max(total_valid, 1):.3f}',
        })

    avg_loss = total_loss / max(n_batches, 1)
    avg_acc = total_correct / max(total_valid, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, loader, device, num_classes, ignore_idx=0):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_valid = 0
    n_batches = 0
    all_ious = []

    for positions, labels, names in loader:
        positions = positions.to(device)
        labels = labels.to(device)

        seg_logits = model(positions)
        B, N, C = seg_logits.shape

        seg_flat = seg_logits.view(-1, C)
        labels_flat = labels.view(-1)
        loss = F.cross_entropy(seg_flat, labels_flat, ignore_index=ignore_idx)

        total_loss += loss.item()
        pred = seg_logits.argmax(dim=-1)
        mask = labels != ignore_idx
        total_correct += (pred[mask] == labels[mask]).sum().item()
        total_valid += mask.sum().item()
        n_batches += 1

        # IoU per sample
        for b in range(B):
            iou = compute_iou(pred[b].cpu(), labels[b].cpu(), num_classes, ignore_idx)
            all_ious.append(iou)

    avg_loss = total_loss / max(n_batches, 1)
    avg_acc = total_correct / max(total_valid, 1)
    avg_iou = np.mean(all_ious) if all_ious else 0.0
    return avg_loss, avg_acc, avg_iou


# ====================== Main ======================

def main():
    parser = argparse.ArgumentParser(description='NeuralTailor 面板分割训练')
    parser.add_argument('--data_root', default='/home/ddd/zkl/GarmentCodeData/GarmentCodeData_v2')
    parser.add_argument('--garment_folders', nargs='+', default=['garments_5000_0'])
    parser.add_argument('--body_types', nargs='+', default=['default_body'])
    parser.add_argument('--num_points', type=int, default=4096)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--global_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--save_dir', default='./models')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint path')
    parser.add_argument('--part_to_id', default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if args.quick:
        args.max_samples = 200
        args.num_epochs = 10
        print(">>> 快速测试模式: 200 样本, 10 epochs")

    device = torch.device(args.device)
    print(f"设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Dataset ----
    if args.part_to_id is None:
        part_to_id_path = os.path.join(args.data_root, 'part_to_id.npy')
    else:
        part_to_id_path = args.part_to_id

    full_dataset = GarmentSegDataset(
        data_root=args.data_root,
        garment_folders=args.garment_folders,
        body_types=args.body_types,
        num_points=args.num_points,
        max_samples=args.max_samples,
        part_to_id_path=part_to_id_path,
        normalize=True,
    )
    num_classes = full_dataset.num_classes
    print(f"面板类别数: {num_classes}")

    # Train/val split
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"训练: {n_train}, 验证: {n_val}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=False)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, drop_last=False)

    # ---- Model ----
    model = NeuralTailorSeg(
        num_classes=num_classes,
        feat_dim=args.feat_dim,
        global_dim=args.global_dim,
        hidden_dim=args.hidden_dim,
        k=args.k,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}")

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6)

    # ---- Resume from checkpoint ----
    start_epoch = 0
    best_acc = 0.0
    best_iou = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_iou': []}

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device,weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        best_acc = ckpt.get('val_acc', 0.0)
        best_iou = ckpt.get('val_iou', 0.0)
        history = ckpt.get('history', history)
        # Re-create scheduler with adjusted T_max
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=1e-6)
        for _ in range(start_epoch):
            scheduler.step()
        print(f">>> 从 epoch {start_epoch} 续训, 当前最佳 Acc={best_acc:.4f}, 目标 {args.num_epochs} 轮")

    t_start = time.time()
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_acc, val_iou = validate(model, val_loader, device, num_classes)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_iou'].append(val_iou)

        print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} IoU: {val_iou:.4f}")

        # Save best (by accuracy)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_iou': val_iou,
                'history': history,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"  >>> Best model saved (Acc: {best_acc:.4f})")

        if val_iou > best_iou:
            best_iou = val_iou

        # Periodic checkpoint
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.save_dir, f'checkpoint_e{epoch}.pth'))

    total_min = (time.time() - t_start) / 60
    print(f"\n训练完成! 总耗时: {total_min:.1f} 分钟")
    print(f"最佳 Val Acc: {best_acc:.4f}, 最佳 Val IoU: {best_iou:.4f}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_model.pth'))
    print(f"最终模型 → {args.save_dir}/final_model.pth")

    # ---- Plot training curves ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(history['val_iou'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('mIoU')
    axes[2].set_title('Mean IoU')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_curve.png'), dpi=150)
    plt.close()
    print(f"训练曲线 → {args.save_dir}/training_curve.png")


if __name__ == '__main__':
    main()
