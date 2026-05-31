"""
预计算所有样本的干净 GT，多进程并行加速。
训练前运行一次即可，后续数据加载直接读缓存 (~0.2s/样本)。
"""
import os
import sys
import time
import numpy as np
import trimesh
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from data_loader import (
    build_edge_index_from_faces,
    read_segmentation,
    read_spec_panels,
    compute_clean_gt,
)


def process_one(args):
    """计算单个样本的干净 GT 并保存。"""
    name, ply_path, seg_path, spec_path, cache_path = args
    mesh = trimesh.load(ply_path, process=False)
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    edge_index = build_edge_index_from_faces(faces)
    labels = read_segmentation(seg_path, len(vertices), faces)
    panels = read_spec_panels(spec_path)
    gt_2d = compute_clean_gt(vertices, faces, labels, panels, edge_index)
    np.save(cache_path, gt_2d)
    return name


def main():
    config = {
        'data_root': '../GarmentCodeData/GarmentCodeData_v2',
        'garment_folders': ['garments_5000_0'],
        'body_types': ['default_body', 'random_body'],
    }

    samples = []
    for gf in config['garment_folders']:
        for bt in config['body_types']:
            body_dir = os.path.join(config['data_root'], gf, bt)
            if not os.path.exists(body_dir):
                continue
            for item in sorted(os.listdir(body_dir)):
                item_path = os.path.join(body_dir, item)
                if not os.path.isdir(item_path) or not item.startswith('rand_'):
                    continue
                ply_path = os.path.join(item_path, f'{item}_sim.ply')
                seg_path = os.path.join(item_path, f'{item}_sim_segmentation.txt')
                spec_path = os.path.join(item_path, f'{item}_specification.json')
                cache_path = ply_path.replace('.ply', '_clean_gt.npy')
                if os.path.exists(ply_path) and os.path.exists(seg_path) and os.path.exists(spec_path):
                    if not os.path.exists(cache_path):
                        samples.append((item, ply_path, seg_path, spec_path, cache_path))

    print(f"需计算: {len(samples)} 个样本")
    if len(samples) == 0:
        print("全部已缓存。")
        return

    n_workers = min(cpu_count(), 8)
    print(f"使用 {n_workers} 个进程并行计算...")
    t_start = time.time()
    with Pool(n_workers) as pool:
        for name in tqdm(pool.imap_unordered(process_one, samples), total=len(samples)):
            pass

    elapsed = time.time() - t_start
    print(f"完成 {len(samples)} 样本, 耗时 {elapsed:.0f}s ({elapsed/len(samples):.1f}s/样本)")


if __name__ == '__main__':
    main()
