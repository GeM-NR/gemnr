from math import comb
from itertools import combinations

import torch
import numpy as np
from torchvision.transforms import ToTensor

from gemnr.core.utils.roma import RoMaEstimator
from gemnr.core.utils.geometry import relpose_from_absolute
from .metrics_utils import is_connected


def essential_matrix_from_absolute(ext1_w2c, ext2_w2c):
    """Compute the essential matrix from two absolute camera poses."""
    Prel = relpose_from_absolute(ext1_w2c, ext2_w2c)
    Rrel, trel = Prel[:3, :3], Prel[:3, 3]
    return skewsym(trel) @ Rrel


def skewsym(t):
    """Convert a 3D vector to a skew-symmetric matrix."""
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def epipolar_distances(pts1, pts2, F):
    """Compute epipolar distances for matched keypoints given the fundamental matrix.
    Args:
        pts1: Nx2 torch.tensor of keypoints in image 1
        pts2: Nx2 torch.tensor of keypoints in image 2
        F: 3x3 torch.tensor fundamental matrix
    Returns:
        distances: Nx1 torch.tensor of epipolar distances for each matched pair
    """
    pts1_hom = torch.hstack(
        [pts1, torch.ones((pts1.shape[0], 1)).to(pts1)]
    )  # Nx3
    pts2_hom = torch.hstack(
        [pts2, torch.ones((pts2.shape[0], 1)).to(pts1)]
    )  # Nx3
    # Compute the epipolar lines in image 1 for points in image 2
    lines1 = F.T @ pts2_hom.T  # 3xN
    lines1 /= torch.norm(lines1[:2, :], dim=0)  # Normalize the lines
    # Compute the distances from points in image 1 to their corresponding epipolar lines
    distances1 = torch.abs(torch.sum(lines1.T * pts1_hom, dim=1))  # N

    # Compute the epipolar lines in image 2 for points in image 1
    lines2 = F @ pts1_hom.T  # 3xN
    lines2 /= torch.norm(lines2[:2, :], dim=0)  # Normalize the lines
    # Compute the distances from points in image 2 to their corresponding epipolar lines
    distances2 = torch.abs(torch.sum(lines2.T * pts2_hom, dim=1))  # N
    return distances1, distances2


def compute_epipolar_metrics(
    img_pil_list: list,
    poses: list,
    max_pairs=20,
    confidence_thresholds: list = [0.1, 0.25, 0.5],
    accuracy_thresholds: list = [1.0, 2.0, 5.0, 10.0],
    **kwargs,
):
    N = len(img_pil_list)
    if N == 2:
        return compute_epipolar_metrics_pair(
            imgs=img_pil_list,
            poses=poses,
            confidence_thresholds=confidence_thresholds,
            accuracy_thresholds=accuracy_thresholds,
            **kwargs,
        )
    assert len(poses) == N, "Number of images and poses must be the same"
    rs = np.random.RandomState(42)
    N_pairs = comb(N, 2)
    idxs = rs.permutation(N_pairs)[: min(N_pairs, max_pairs)]
    pairs_list = [
        pair for i, pair in enumerate(combinations(range(N), 2)) if i in idxs
    ]
    if N != 10:
        assert is_connected(N, pairs_list)
        # if N == 10, we know it is connected

    distances1 = None
    distances2 = None
    confidences = None
    meta_data_list = []
    for pair in pairs_list:
        img_pair = [img_pil_list[pair[0]], img_pil_list[pair[1]]]
        pose_pair = [poses[pair[0]], poses[pair[1]]]
        metrics_dict_pair, meta_data_pair = compute_epipolar_metrics_pair(
            imgs=img_pair,
            poses=pose_pair,
            confidence_thresholds=confidence_thresholds,
            accuracy_thresholds=accuracy_thresholds,
            **kwargs,
        )
        if distances1 is None:
            distances1 = meta_data_pair["epipolar_distances_im1"]
            distances2 = meta_data_pair["epipolar_distances_im2"]
            confidences = meta_data_pair["confidences"]
        else:
            distances1 = torch.hstack(
                [distances1, meta_data_pair["epipolar_distances_im1"]]
            )
            distances2 = torch.hstack(
                [distances2, meta_data_pair["epipolar_distances_im2"]]
            )
            confidences = torch.hstack(
                [confidences, meta_data_pair["confidences"]]
            )
        meta_data_list.append(meta_data_pair)

    metrics_dict = compute_distance_stats(
        distances1, distances2
    ) | compute_acc_at_and_mAA(
        distances1, distances2, confidences, accuracy_thresholds=accuracy_thresholds, confidence_thresholds=confidence_thresholds
    )
    meta_data = {
        "pair_combinations": pairs_list,
        "meta_data_list": meta_data_list,
        "mAA": metrics_dict["mAA"],
    }
    return metrics_dict, meta_data


def compute_epipolar_metrics_pair(
    imgs: list,
    poses: list,
    intrinsics: np.ndarray,
    roma_estimator: RoMaEstimator | None = None,
    uniform_keypoints_im1: bool = True,
    num_matches: int = 10000,
    accuracy_thresholds: list = [1.0, 2.0, 5.0, 10.0],
    confidence_thresholds: list = [0, 0.05, 0.1, 0.15],
    H_standard: int = 512,
    W_standard: int = 512,
    seed: int = 42,
    device: str = "cuda",
):
    """Compute epipolar metrics for a pair of images and their corresponding poses.
    Args:
        imgs: List of PIL images
        K: Camera intrinsic matrix
        poses: List of camera poses
        roma_estimator: Optional RoMaEstimator instance
        num_matches: Number of matches to use for epipolar distance computation
        device: Device to run the computations on
    """
    assert (
        len(imgs) == 2
    ), f"This function supports 2 images only, but got {len(imgs)}"
    imgs_resized = [im.resize((W_standard, H_standard)) for im in imgs]
    im1 = imgs_resized[0]
    im2 = imgs_resized[1]
    P1, P2 = poses
    if roma_estimator is None:
        W, H = im1.size
        roma_estimator = RoMaEstimator(H=H, W=W, device=device)

    # Compute the essential matrix from the poses
    E = essential_matrix_from_absolute(P1, P2)
    Kinv = np.linalg.inv(intrinsics)
    F = Kinv.T @ E @ Kinv

    im1_tensor = ToTensor()(im1)
    im2_tensor = ToTensor()(im2)

    pts1, pts2, confidences = roma_estimator.estimate_matches(
        im1_tensor,
        im2_tensor,
        uniform_keypoints_im1=uniform_keypoints_im1,
        num_matches=num_matches,
        seed=seed,
    )

    # Compute epipolar constraints and metrics
    distances1, distances2 = epipolar_distances(
        pts1, pts2, torch.tensor(F).to(pts1)
    )

    # Prepare meta data for visualization
    meta_data = {
        "imgs": imgs_resized,
        "pts1": pts1.cpu().numpy(),
        "pts2": pts2.cpu().numpy(),
        "confidences": confidences,
        "essential_matrix": E,
        "fundamental_matrix": F,
        "epipolar_distances_im1": distances1,
        "epipolar_distances_im2": distances2,
        "accuracy_thresholds": accuracy_thresholds,
        "confidence_thresholds": confidence_thresholds,
    }
    metrics_dict = compute_distance_stats(
        distances1, distances2
    ) | compute_acc_at_and_mAA(
        distances1,
        distances2,
        confidences,
        accuracy_thresholds=accuracy_thresholds,
        confidence_thresholds=confidence_thresholds,
    )
    return metrics_dict, meta_data


def compute_distance_stats(distances1, distances2):
    values = {
        "epipolar_distance_median": torch.median(
            torch.cat([distances1, distances2])
        )
        .cpu()
        .item(),
        "epipolar_distance_mean": torch.mean(
            torch.cat([distances1, distances2])
        )
        .cpu()
        .item(),
    }
    return values


def compute_acc_at_and_mAA(
    distances1: torch.Tensor,
    distances2: torch.Tensor,
    confidences: torch.Tensor,
    accuracy_thresholds: list = [1.0, 2.0, 5.0, 10.0],
    confidence_thresholds: list = [0.1, 0.25, 0.5],
):
    values = {"mAA": 0}
    for accuracy_threshold in accuracy_thresholds:
        for confidence_threshold in confidence_thresholds:
            values[f"acacuracy_at_{accuracy_threshold}_conf_{confidence_threshold}"] = ((
                    torch.sum(
                        torch.logical_and(
                            torch.logical_and(
                                distances1 < accuracy_threshold,
                                distances2 < accuracy_threshold,
                            ),
                            confidences > confidence_threshold,
                        )
                    ) / distances1.shape[0] * 100
                ).cpu().item())
            values["mAA"] += values[f"acacuracy_at_{accuracy_threshold}_conf_{confidence_threshold}"]
    values["mAA"] /= len(accuracy_thresholds) * len(confidence_thresholds)
    return values