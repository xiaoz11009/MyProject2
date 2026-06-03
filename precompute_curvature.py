"""
Precompute curvature for all GarmentCodeData samples (GPU-accelerated).
Saves (N, 2) curvature tensors to cache directory.

Usage:
  python precompute_curvature.py
  python precompute_curvature.py --max_samples 100  # subset
"""
import os, sys, argparse, torch, numpy as np, trimesh
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from curvature_utils import compute_curvature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/home/ddd/zkl/GarmentCodeData/GarmentCodeData_v2')
    parser.add_argument('--garment_folder', default='garments_5000_0')
    parser.add_argument('--body_type', default='default_body')
    parser.add_argument('--cache_root', default=None)  # defaults to data_root/curvature_cache
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.cache_root is None:
        args.cache_root = os.path.join(args.data_root, 'curvature_cache',
                                       args.garment_folder, args.body_type)

    body_dir = os.path.join(args.data_root, args.garment_folder, args.body_type)
    samples = sorted([d for d in os.listdir(body_dir)
                      if d.startswith('rand_') and os.path.isdir(os.path.join(body_dir, d))])
    if args.max_samples:
        samples = samples[:args.max_samples]

    os.makedirs(args.cache_root, exist_ok=True)
    total, skipped = 0, 0

    for name in tqdm(samples, desc="预计算曲率"):
        out_path = os.path.join(args.cache_root, f'{name}_curvature.pt')
        if os.path.exists(out_path):
            skipped += 1
            continue

        ply = os.path.join(body_dir, name, f'{name}_sim.ply')
        if not os.path.exists(ply):
            continue

        mesh = trimesh.load(ply, process=False)
        verts = torch.from_numpy(np.array(mesh.vertices, dtype=np.float32)).to(device)
        faces = torch.from_numpy(np.array(mesh.faces, dtype=np.int64)).to(device)

        curv = compute_curvature(verts, faces)  # (N, 2), GPU
        torch.save(curv.cpu(), out_path)
        total += 1

    print(f"完成: {total} 新增, {skipped} 跳过 (已缓存)")
    print(f"缓存目录: {args.cache_root}")


if __name__ == '__main__':
    main()
