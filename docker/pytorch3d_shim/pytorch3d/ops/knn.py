"""Lightweight KNN implementation using pure PyTorch (no CUDA extensions)."""
from collections import namedtuple

import torch

KNN = namedtuple("KNN", ["dists", "idx", "knn"])


def knn_points(p1, p2, K=1, return_nn=False, return_sorted=True):
    """
    K-Nearest Neighbors between two point clouds.

    Args:
        p1: (N, P1, D) first point cloud
        p2: (N, P2, D) second point cloud
        K: number of nearest neighbors
        return_nn: if True, also return the K nearest points from p2
        return_sorted: if True, sort by distance

    Returns:
        KNN namedtuple with dists (N, P1, K), idx (N, P1, K), knn (N, P1, K, D) or None
    """
    dists = torch.cdist(p1.float(), p2.float())
    knn_dists, knn_idx = dists.topk(K, dim=-1, largest=False, sorted=return_sorted)
    knn_dists = knn_dists * knn_dists

    knn_pts = None
    if return_nn:
        knn_pts = knn_gather(p2, knn_idx)

    return KNN(dists=knn_dists, idx=knn_idx, knn=knn_pts)


def knn_gather(x, idx):
    """
    Gather points from x using KNN indices.

    Args:
        x: (N, P, D) point features
        idx: (N, P1, K) indices into x

    Returns:
        (N, P1, K, D) gathered features
    """
    N, P, D = x.shape
    _, P1, K = idx.shape
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, D)
    x_expanded = x.unsqueeze(1).expand(-1, P1, -1, -1)
    return torch.gather(x_expanded, 2, idx_expanded)
