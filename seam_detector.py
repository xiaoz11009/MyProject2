"""Seam edge detector: binary MLP classifier (seam vs interior).

Input: 9 geometric features per mesh edge
Output: probability that the edge is a seam

Much simpler than 41-class vertex classification — just a binary edge
classifier using dihedral angle, curvature, edge length, etc.
"""

import torch
import torch.nn as nn


class SeamDetector(nn.Module):
    """MLP that classifies mesh edges as seam (1) or interior (0).

    Input: (E, 9) geometric features
    Output: (E, 2) logits [interior, seam]
    """
    def __init__(self, in_dim=9, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, features):
        return self.net(features)


def predict_seams(model, edge_features, threshold=0.5, device='cuda'):
    """Predict seam edges from geometric features.

    Args:
        model: SeamDetector
        edge_features: (E, 9) tensor
        threshold: probability threshold for seam classification

    Returns:
        seam_mask: (E,) bool
        probs: (E,) float — seam probability
    """
    model.eval()
    with torch.no_grad():
        logits = model(edge_features.to(device))
        probs = torch.softmax(logits, dim=-1)[:, 1]
        return probs > threshold, probs
