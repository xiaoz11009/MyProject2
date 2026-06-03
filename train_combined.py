"""
Training script for PhysUnfolderCombined (optimized).

Optimizations:
  - Precomputed curvature (GPU cache, no CPU overhead)
  - Optional GCN (--no_gcn for pure EdgeConv, faster)
  - Padded batch processing (batch_size > 1, no per-sample loop)
  - AMP + gradient accumulation
"""
import os, sys, time, argparse, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import GarmentSegDataset, collate_dual_batch
from model_combined import PhysUnfolderCombined
from losses import PhysicsInspiredLoss


def compute_accuracy_padded(seg_logits, labels, mask, ignore_idx=0):
    """Accuracy for padded batch: only count valid+labeled vertices."""
    valid = mask & (labels != ignore_idx)
    if valid.sum() == 0:
        return 1.0
    pred = seg_logits.argmax(dim=-1)
    correct = (pred[valid] == labels[valid]).sum().item()
    return correct / valid.sum().item()


def train_epoch(model, loader, optimizer, scaler, loss_fn, device, epoch, grad_accum=1):
    model.train()
    total_loss, total_seg, total_mem, total_bend, total_reg, total_acc = 0., 0., 0., 0., 0., 0.
    n_steps = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}")
    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        v_sample = batch['vertices_sample'].to(device, non_blocking=True)
        e_sample = [ei.to(device, non_blocking=True) for ei in batch['edge_indices_sample']]
        curv_sample = batch['curvature_sample'].to(device, non_blocking=True)
        mat = batch['material'].to(device, non_blocking=True)
        lbls_sample = batch['labels_sample'].to(device, non_blocking=True)
        # Full mesh data stays on CPU until GCN (per-sample, moved individually)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            pred = model.forward_dual(
                batch['vertices_full'], batch['edge_indices_full'],
                v_sample, e_sample, batch['sample_indices'],
                curv_sample, mat, labels_sample=lbls_sample)

        u_gt = batch.get('u_gt')
        u_gt_valid = batch.get('u_gt_valid')
        if u_gt is not None:
            u_gt = u_gt.to(device, non_blocking=True)
            u_gt_valid = u_gt_valid.to(device, non_blocking=True)

        garment_mask = batch.get('garment_mask')
        if garment_mask is not None:
            garment_mask = garment_mask.to(device, non_blocking=True)

        # Loss on sampled points
        loss, loss_dict = loss_fn.forward_dual(
            pred, lbls_sample, v_sample, e_sample, curv_sample, mat,
            u_gt, u_gt_valid, garment_mask_gt=garment_mask)
        loss = loss / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss_dict['total']
        total_seg += loss_dict['seg']
        total_mem += loss_dict['membrane']
        total_bend += loss_dict['bending']
        total_reg += loss_dict['reg']
        # Accuracy on sampled points
        valid_mask = lbls_sample != loss_fn.ignore_idx
        if valid_mask.sum() > 0:
            acc = (pred['seg_logits'].argmax(-1)[valid_mask] == lbls_sample[valid_mask]).float().mean().item()
        else:
            acc = 1.0
        total_acc += acc
        n_steps += 1

        pbar.set_postfix({
            'loss': f'{total_loss / n_steps:.3f}',
            'seg': f'{total_seg / n_steps:.3f}',
            'mem': f'{total_mem / n_steps:.4f}',
            'bnd': f'{total_bend / n_steps:.4f}',
            'unf': f'{loss_dict.get("unfold", 0):.4f}',
            'acc': f'{total_acc / n_steps:.3f}',
        })

    return (total_loss / n_steps, total_seg / n_steps, total_mem / n_steps,
            total_bend / n_steps, total_reg / n_steps, total_acc / n_steps)


@torch.no_grad()
def validate(model, loader, loss_fn, device, num_classes):
    model.eval()
    total_loss, total_seg, total_mem, total_bend, total_reg, total_acc = 0., 0., 0., 0., 0., 0.
    n_steps = 0

    for batch in loader:
        v_sample = batch['vertices_sample'].to(device, non_blocking=True)
        e_sample = [ei.to(device, non_blocking=True) for ei in batch['edge_indices_sample']]
        curv_sample = batch['curvature_sample'].to(device, non_blocking=True)
        mat = batch['material'].to(device, non_blocking=True)
        lbls_sample = batch['labels_sample'].to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            pred = model.forward_dual(
                batch['vertices_full'], batch['edge_indices_full'],
                v_sample, e_sample, batch['sample_indices'],
                curv_sample, mat)

        u_gt = batch.get('u_gt')
        u_gt_valid = batch.get('u_gt_valid')
        garment_mask = batch.get('garment_mask')
        if u_gt is not None:
            u_gt = u_gt.to(device, non_blocking=True)
            u_gt_valid = u_gt_valid.to(device, non_blocking=True)
        if garment_mask is not None:
            garment_mask = garment_mask.to(device, non_blocking=True)
        _, loss_dict = loss_fn.forward_dual(
            pred, lbls_sample, v_sample, e_sample, curv_sample, mat,
            u_gt, u_gt_valid, garment_mask_gt=garment_mask)

        total_loss += loss_dict['total']
        total_seg += loss_dict['seg']
        total_mem += loss_dict['membrane']
        total_bend += loss_dict['bending']
        total_reg += loss_dict['reg']
        valid_mask = lbls_sample != loss_fn.ignore_idx
        if valid_mask.sum() > 0:
            total_acc += (pred['seg_logits'].argmax(-1)[valid_mask] == lbls_sample[valid_mask]).float().mean().item()
        n_steps += 1

    return (total_loss / n_steps, total_seg / n_steps, total_mem / n_steps,
            total_bend / n_steps, total_reg / n_steps, total_acc / n_steps)


def main():
    parser = argparse.ArgumentParser(description='PhysUnfolder 优化训练')
    parser.add_argument('--data_root', default='/home/ddd/zkl/GarmentCodeData/GarmentCodeData_v2')
    parser.add_argument('--garment_folders', nargs='+', default=['garments_5000_0'])
    parser.add_argument('--body_types', nargs='+', default=['default_body'])
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--val_split', type=float, default=0.05)
    parser.add_argument('--lambda_membrane', type=float, default=1e-3)
    parser.add_argument('--lambda_bending', type=float, default=1e-4)
    parser.add_argument('--lambda_reg', type=float, default=1e-2)
    parser.add_argument('--lambda_unfold', type=float, default=0.3,
                        help='Weight for GT 2D supervision L2 loss')
    parser.add_argument('--lambda_garment', type=float, default=1.0,
                        help='Weight for garment mask BCE loss')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--num_vertices', type=int, default=4096)
    parser.add_argument('--save_dir', default='./models_combined')
    parser.add_argument('--quick', action='store_true', help='Quick test')
    parser.add_argument('--no_film', action='store_true', help='Disable FiLM')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--part_to_id', default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if args.quick:
        args.max_samples = 100
        args.num_epochs = 10
        args.batch_size = 4
        args.grad_accum = 1
        print(">>> 快速测试: 100 样本, 10 epochs")

    device = torch.device(args.device)
    eff_bs = args.batch_size * args.grad_accum
    print(f"设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"batch_size={args.batch_size} × grad_accum={args.grad_accum} = 有效 {eff_bs}")
    print(f"FiLM={'OFF' if args.no_film else 'ON'}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Dataset
    part_to_id_path = args.part_to_id or os.path.join(args.data_root, 'part_to_id.npy')
    full_dataset = GarmentSegDataset(
        data_root=args.data_root, garment_folders=args.garment_folders,
        body_types=args.body_types, max_samples=args.max_samples,
        part_to_id_path=part_to_id_path, full_mesh=True,
        material_variation=False, num_vertices=args.num_vertices,
    )
    num_classes = full_dataset.num_classes
    print(f"面板类别: {num_classes}")

    n_total = len(full_dataset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    print(f"训练: {n_train}, 验证: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_dual_batch, num_workers=2,
                              pin_memory=True, drop_last=True, persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_dual_batch, num_workers=1,
                            pin_memory=True, drop_last=False)

    # Compute class weights: 1/count to help rare panels (cuffs, hoods, etc.)
    print("计算类别权重...")
    class_counts = np.zeros(num_classes, dtype=np.float64)
    for i in tqdm(range(len(train_ds)), desc="统计"):
        item = train_ds[i]
        lbls = item['labels_sample'].numpy()
        counts = np.bincount(lbls[lbls != 0], minlength=num_classes)
        class_counts += counts
    class_counts = np.maximum(class_counts, 1.0)
    class_weights = 1.0 / np.sqrt(class_counts)
    class_weights = class_weights / class_weights[1:].mean()  # normalize so mean=1
    class_weights[0] = 0.0  # ignore_idx
    print(f"  类别权重范围: [{class_weights[1:].min():.3f}, {class_weights[1:].max():.3f}]")
    print(f"  最小权重大面板 id={class_weights[1:].argmin()+1}, 最大权重小面板 id={class_weights[1:].argmax()+1}")

    # Model
    model = PhysUnfolderCombined(
        num_classes=num_classes, hidden_dim=args.hidden_dim,
        use_film=not args.no_film,
    ).to(device)
    print(f"参数: {sum(p.numel() for p in model.parameters()):,}")

    # Loss
    loss_fn = PhysicsInspiredLoss(
        num_classes=num_classes, ignore_idx=0,
        lambda_membrane=args.lambda_membrane,
        lambda_bending=args.lambda_bending,
        lambda_reg=args.lambda_reg,
        lambda_unfold=args.lambda_unfold,
        lambda_garment=args.lambda_garment,
        class_weights=class_weights,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs - args.warmup_epochs, eta_min=1e-6)

    start_epoch = 0
    best_acc = 0.0
    history = {'train_loss': [], 'train_seg': [], 'train_mem': [], 'train_acc': [],
               'val_loss': [], 'val_seg': [], 'val_mem': [], 'val_acc': []}

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt.get('scaler_state_dict', scaler.state_dict()))
        start_epoch = ckpt['epoch']
        best_acc = ckpt.get('val_acc', 0.0)
        history = ckpt.get('history', history)
        print(f">>> 续训自 epoch {start_epoch}, best_acc={best_acc:.4f}")

    t_start = time.time()
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        if epoch <= args.warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * epoch / args.warmup_epochs

        t_loss, t_seg, t_mem, t_bend, t_reg, t_acc = train_epoch(
            model, train_loader, optimizer, scaler, loss_fn, device, epoch,
            grad_accum=args.grad_accum)

        v_loss, v_seg, v_mem, v_bend, v_reg, v_acc = validate(
            model, val_loader, loss_fn, device, num_classes)

        if epoch > args.warmup_epochs:
            scheduler.step()

        history['train_loss'].append(t_loss)
        history['train_seg'].append(t_seg)
        history['train_mem'].append(t_mem)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_seg'].append(v_seg)
        history['val_mem'].append(v_mem)
        history['val_acc'].append(v_acc)

        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:3d} | LR={lr:.2e} | "
              f"Train L={t_loss:.4f} Seg={t_seg:.3f} Acc={t_acc:.4f} | "
              f"Val L={v_loss:.4f} Seg={v_seg:.3f} Acc={v_acc:.4f}")

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_acc': v_acc, 'history': history,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"  >>> Best (Acc={best_acc:.4f})")

        if epoch % 20 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       os.path.join(args.save_dir, f'checkpoint_e{epoch}.pth'))

    print(f"\n完成! {((time.time()-t_start)/60):.1f} min, Best Acc={best_acc:.4f}")
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_model.pth'))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (kt, kv, title) in zip(axes, [
        ('train_loss', 'val_loss', 'Total Loss'),
        ('train_seg', 'val_seg', 'Seg Loss'),
        ('train_acc', 'val_acc', 'Accuracy'),
    ]):
        ax.plot(history[kt], label='Train', alpha=0.7)
        ax.plot(history[kv], label='Val', alpha=0.7)
        ax.set_xlabel('Epoch'); ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_curve.png'), dpi=150)
    plt.close()
    print(f"曲线 → {args.save_dir}/training_curve.png")


if __name__ == '__main__':
    main()
