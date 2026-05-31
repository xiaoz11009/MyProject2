"""
DGCNN 分割模型 + 预测器封装，用于为 PhysUnfolder 提供逐顶点面板标签。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ====================== DGCNN 定义 (与 Baseline/train_dgcnn.py 完全一致) ======================
class DGCNN(nn.Module):
    def __init__(self, k=16, num_classes=42):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv2d(12, 64, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64*2, 64, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64*2, 128, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128*2, 256, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)

        self.bn_linear = nn.BatchNorm1d(1024)
        self.linear = nn.Conv1d(64+64+128+256, 1024, 1, bias=False)

        self.head = nn.Sequential(
            nn.Conv1d(1024+64+64+128+256+6, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(256, num_classes, 1)
        )

    def get_graph_feature(self, x):
        B, D, N = x.shape
        xx = x.transpose(2, 1)
        dist = torch.cdist(xx, xx)
        idx = dist.topk(self.k, dim=-1, largest=False)[1]
        idx = idx.reshape(B, N*self.k)
        feat = xx.gather(1, idx.unsqueeze(-1).repeat(1,1,D))
        feat = feat.reshape(B, N, self.k, D)
        xx_expand = xx.unsqueeze(2).repeat(1,1,self.k,1)
        feat = torch.cat([xx_expand, feat - xx_expand], dim=-1)
        return feat.permute(0,3,1,2)

    def forward(self, x):
        xyz = x
        x1 = self.get_graph_feature(x)
        x1 = F.leaky_relu(self.bn1(self.conv1(x1)))
        x1 = x1.max(dim=-1)[0]
        x2 = self.get_graph_feature(x1)
        x2 = F.leaky_relu(self.bn2(self.conv2(x2)))
        x2 = x2.max(dim=-1)[0]
        x3 = self.get_graph_feature(x2)
        x3 = F.leaky_relu(self.bn3(self.conv3(x3)))
        x3 = x3.max(dim=-1)[0]
        x4 = self.get_graph_feature(x3)
        x4 = F.leaky_relu(self.bn4(self.conv4(x4)))
        x4 = x4.max(dim=-1)[0]
        x = torch.cat([x1, x2, x3, x4], dim=1)
        g = F.leaky_relu(self.bn_linear(self.linear(x)))
        g = g.max(dim=-1, keepdim=True)[0].repeat(1, 1, x.shape[-1])
        x = torch.cat([g, x, xyz], dim=1)
        x = self.head(x)
        return x


# ====================== FPS 采样 ======================
def farthest_point_sample(points, npoint):
    N, _ = points.shape
    centroids = np.zeros(npoint, dtype=np.int32)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = points[farthest, :].reshape(1, 3)
        dist = np.sum((points - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    return centroids


# ====================== 标签预测器 ======================
class SegPredictor:
    """加载训练好的 DGCNN，为 3D 网格提供逐顶点面板标签预测"""

    def __init__(self, model_path, part_to_id_path, num_points=4096, device='cpu'):
        # 加载标签映射
        self.part_to_id = np.load(part_to_id_path, allow_pickle=True).item()
        self.id_to_part = {v: k for k, v in self.part_to_id.items()}
        self.num_classes = len(self.part_to_id)
        self.num_points = num_points
        self.device = device

        # 加载模型
        self.model = DGCNN(k=16, num_classes=self.num_classes).to(device)
        ckpt = torch.load(model_path, map_location=device)
        self.model.load_state_dict(ckpt)
        self.model.eval()
        print(f"DGCNN 分割模型已加载: {self.num_classes} 类, 设备={device}")

    @torch.no_grad()
    def predict(self, vertices, faces):
        """
        对完整网格做逐顶点标签预测。
        vertices: (N, 3)
        faces: (F, 3)
        返回: pred_labels (N,) int, pred_probs (N, num_classes)
        """
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        normals = np.array(mesh.vertex_normals, dtype=np.float32)

        N = len(vertices)
        # 归一化
        centroid = vertices.mean(axis=0)
        v = vertices - centroid
        max_dist = np.linalg.norm(v, axis=1).max()
        v = v / max(max_dist, 1e-8)

        # FPS 采样
        n_sample = min(self.num_points, N)
        if n_sample < N:
            idx = farthest_point_sample(v, n_sample)
        else:
            idx = np.arange(N)

        pts = np.concatenate([v[idx], normals[idx]], axis=1)  # (S, 6)
        pts = torch.tensor(pts, dtype=torch.float, device=self.device).unsqueeze(0).transpose(2, 1)

        logits = self.model(pts)  # (1, C, S)
        probs_sample = F.softmax(logits, dim=1).squeeze(0).t()  # (S, C)
        pred_sample = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # (S,)

        # 用最近邻将采样点标签传播回所有顶点
        if n_sample < N:
            v_tensor = torch.tensor(v, dtype=torch.float)
            v_sample = torch.tensor(v[idx], dtype=torch.float)
            dists = torch.cdist(v_tensor, v_sample)  # (N, S)
            nn_idx = torch.argmin(dists, dim=1).numpy()  # (N,)
            pred_all = pred_sample[nn_idx]
            probs_all = probs_sample.cpu().numpy()[nn_idx]  # (N, C)
        else:
            pred_all = pred_sample
            probs_all = probs_sample.cpu().numpy()

        return pred_all.astype(np.int64), probs_all.astype(np.float32)
