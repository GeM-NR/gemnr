import torch
import torchvision.transforms.functional as tvF
import numpy as np
from PIL import Image
import poselib

from roma import roma_outdoor
from gemnr.core.utils.geometry import unnormalize_intrinsics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)


def numpy_to_pil(x: np.ndarray):
    """
    Args:
        x: Assumed to be of shape (h,w,c)
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.max() <= 1.01:
        x *= 255
    x = x.astype(np.uint8)
    return Image.fromarray(x)


def tensor_to_pil(x):
    assert x.is_cuda, "Tensor is not on CUDA"
    assert x.dim() == 3, "Tensor must be 3-dimensional"
    assert (
        x.size(1) > 0 and x.size(2) > 0
    ), "Tensor dimensions must be positive"

    x = x.detach().permute(1, 2, 0).cpu().numpy()
    x = np.clip(x, 0.0, 1.0)
    return numpy_to_pil(x)


def estimate_warp_with_roma(
    im1,
    im2,
    H,
    W,
    device,
    roma_model=None,
):
    im1, im2 = im1.to(device), im2.to(device)
    with torch.no_grad():
        if roma_model is None:
            print("RomA Model not provided, creating new model")
            roma_model = roma_outdoor(device=device)
            roma_model.decoder.detach = False
            roma_model.upsample_preds = False

        if len(im1.shape) == 3:
            im1 = im1.unsqueeze(0)
        elif not (len(im1.shape) == 4 and im1.shape[0] == 1):
            raise ValueError(
                f"Expected tensor shape [1,H,W,3], but got {im1.shape}"
            )

        if len(im2.shape) == 3:
            im2 = im2.unsqueeze(0)
        elif not (len(im2.shape) == 4 and im2.shape[0] == 1):
            raise ValueError(
                f"Expected tensor shape [1,H,W,3], but got {im2.shape}"
            )
        one, H_orig, W_orig, three = im1.shape

        im1_roma = tvF.resize(im1, (H, W))
        im2_roma = tvF.resize(im2, (H, W))

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        im1_roma = tvF.normalize(im1_roma, mean=mean, std=std)
        im2_roma = tvF.normalize(im2_roma, mean=mean, std=std)

        warp, certainty_map = roma_model.match(
            im1_roma, im2_roma, device=device, batched=True
        )
        # warp = tvF.resize(warp, (H_orig, W_orig))
        # certainty_map = tvF.resize(certainty_map, (H_orig, W_orig))
        return warp, certainty_map


# from RoMa v2
def to_normalized(x: torch.Tensor, *, H: int, W: int) -> torch.Tensor:
    """0 => -1; n => 1"""
    return torch.stack((2 * x[..., 0] / W, 2 * x[..., 1] / H), dim=-1) - 1


def to_pixel(x: torch.Tensor, *, H: int, W: int) -> torch.Tensor:
    """-1 => -1; 1 => n"""
    return torch.stack(
        ((x[..., 0] + 1) / 2 * W, (x[..., 1] + 1) / 2 * H), dim=-1
    )


def get_normalized_grid(
    B: int,
    H: int,
    W: int,
    overload_device: torch.device | None = None,
) -> torch.Tensor:
    x1_n = torch.meshgrid(
        *[
            torch.linspace(
                -1 + 1 / n, 1 - 1 / n, n, device=overload_device or device
            )
            for n in (B, H, W)
        ],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H, W, 2)
    return x1_n


def get_pixel_grid(
    B: int,
    *,
    H: int,
    W: int,
    overload_device: torch.device | None = None,
) -> torch.Tensor:
    x1_n = torch.meshgrid(
        *[
            torch.arange(n, device=overload_device or device) + 0.5
            for n in (B, H, W)
        ],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H, W, 2)
    return x1_n


def downsample_grid(
    points: torch.Tensor, H: int, W: int, H_out: int, W_out: int
):
    """
    Downsample unordered grid points and return indices.

    Args:
        points: (N, 2) where N = H * W
        H, W: original grid dimensions
        H_out, W_out: target grid dimensions

    Returns:
        sampled_points: (H_out * W_out, 2)
        sampled_indices: indices into original `points`
    """

    N = points.shape[0]
    assert N == H * W

    # recover grid ordering
    sort_idx = torch.argsort(points[:, 1] * 1e6 + points[:, 0])

    pts_sorted = points[sort_idx]

    # reshape
    grid_pts = pts_sorted.reshape(H, W, 2)
    grid_idx = sort_idx.reshape(H, W)

    # evenly spaced rows/cols
    rows = torch.linspace(0, H - 1, H_out).round().long()
    cols = torch.linspace(0, W - 1, W_out).round().long()

    # sample
    sampled_points = grid_pts[rows][:, cols]
    sampled_indices = grid_idx[rows][:, cols]

    return (sampled_points.reshape(-1, 2), sampled_indices.reshape(-1))


def farthest_point_sampling(points, k):
    """
    points: (N, 2) tensor of (x, y)
    k: number of samples

    Returns:
        sampled_points: (k, 2)
        sampled_indices: (k,)
    """

    device = points.device
    N = points.shape[0]

    # indices of sampled points
    sampled_indices = torch.zeros(k, dtype=torch.long, device=device)

    # distance to closest selected point
    distances = torch.full((N,), float("inf"), device=device)

    # randomly choose first point
    farthest = torch.randint(0, N, (1,), device=device).item()

    for i in range(k):
        sampled_indices[i] = farthest

        centroid = points[farthest].unsqueeze(0)  # (1, 2)

        # compute squared distances
        dist = torch.sum((points - centroid) ** 2, dim=1)

        # keep minimum distance to sampled set
        distances = torch.minimum(distances, dist)

        # choose farthest point
        farthest = torch.argmax(distances).item()

    sampled_points = points[sampled_indices]

    return sampled_points, sampled_indices


class RoMaEstimator:
    def __init__(
        self,
        H,
        W,
        device,
    ):
        self.H = H
        self.W = W
        self.device = device
        self.model = roma_outdoor(device=self.device, coarse_res=(H, W))
        self.model.decoder.detach = False
        self.model.upsample_preds = False

    def estimate_warp(
        self,
        im1: torch.Tensor,
        im2: torch.Tensor,
    ):
        im1, im2 = im1.to(self.device), im2.to(self.device)
        with torch.no_grad():
            if len(im1.shape) == 3:
                im1 = im1.unsqueeze(0)
            elif not (len(im1.shape) == 4 and im1.shape[0] == 1):
                raise ValueError(
                    f"Expected tensor shape [1,H,W,3], but got {im1.shape}"
                )

            if len(im2.shape) == 3:
                im2 = im2.unsqueeze(0)
            elif not (len(im2.shape) == 4 and im2.shape[0] == 1):
                raise ValueError(
                    f"Expected tensor shape [1,H,W,3], but got {im2.shape}"
                )
            one, H_orig, W_orig, three = im1.shape

            im1_roma = tvF.resize(im1, (self.H, self.W))
            im2_roma = tvF.resize(im2, (self.H, self.W))

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            im1_roma = tvF.normalize(im1_roma, mean=mean, std=std)
            im2_roma = tvF.normalize(im2_roma, mean=mean, std=std)

            warp, certainty_map = self.model.match(
                im1_roma, im2_roma, device=device, batched=True
            )
            return warp, certainty_map

    def estimate_matches(
        self,
        im1: torch.Tensor,
        im2: torch.Tensor,
        uniform_keypoints_im1: bool = False,
        num_matches: int = 1000,
        certainty_threshold: float = 0,
        seed: int = 42,
    ):
        H1, W1 = im1.shape[1], im1.shape[2]
        H2, W2 = im2.shape[1], im2.shape[2]

        torch.manual_seed(seed)
        warp, certainty_map = self.estimate_warp(im1, im2)
        if uniform_keypoints_im1:
            matches = warp.reshape(-1, 4)
            certainty = certainty_map.reshape(-1)
            H_out = int(round(np.sqrt(num_matches / self.W * self.H)))
            W_out = int(round(H_out * self.W / self.H))

            # Below assumes that the matches have a grid structure
            # which they do for RoMa
            _, valid = downsample_grid(
                matches[:, :2], H=self.H, W=self.W, H_out=H_out, W_out=W_out
            )

            # Otherwise below should be used
            # _, valid = farthest_point_sampling(matches[:, :2], num_matches)

            matches = matches[valid, :]
            certainty = certainty[valid]
        else:
            valid = certainty_map > certainty_threshold
            matches, certainty = self.model.sample(
                warp, certainty_map * valid, num=num_matches
            )
        pts1, pts2 = self.model.to_pixel_coordinates(matches, H1, W1, H2, W2)
        return pts1, pts2, certainty

    def estimate_relpose_from_warp(
        self,
        warp,
        certainty_map,
        K1,
        K2,
        num_matches=1000,
        initial_pose=None,
        max_epipolar_error=4.0,
        certainty_threshold=0,
    ):
        valid = certainty_map > certainty_threshold
        # vidx = valid.reshape(-1)
        # warp_pixels1 = to_pixel(warp[..., :2], H=self.H, W=self.W)
        # warp_pixels2 = to_pixel(warp[..., 2:], H=self.H, W=self.W)
        # pts1 = warp_pixels1[0].reshape(-1, 2)[vidx].to("cpu")
        # pts2 = warp_pixels2[0].reshape(-1, 2)[vidx].to("cpu")
        # random_idxs = torch.randperm(len(pts1))[: min(num_matches, len(pts1))]
        # pts1 = pts1[random_idxs]
        # pts2 = pts2[random_idxs]

        matches, certainty = self.model.sample(
            warp, certainty_map * valid, num=num_matches
        )
        pts1, pts2 = self.model.to_pixel_coordinates(
            matches, self.H, self.W, self.H, self.W
        )

        pts1 = np.asarray(pts1.to("cpu"), dtype=np.float64)
        pts2 = np.asarray(pts2.to("cpu"), dtype=np.float64)
        K1 = unnormalize_intrinsics(
            np.asarray(K1, dtype=np.float64), H=self.H, W=self.W
        )
        K2 = unnormalize_intrinsics(
            np.asarray(K2, dtype=np.float64), H=self.H, W=self.W
        )

        fx1, fy1, cx1, cy1 = K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]
        fx2, fy2, cx2, cy2 = K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2]

        camera1 = {
            "model": "PINHOLE",
            "width": self.W,
            "height": self.H,
            "params": [fx1, fy1, cx1, cy1],
        }
        camera2 = {
            "model": "PINHOLE",
            "width": self.W,
            "height": self.H,
            "params": [fx2, fy2, cx2, cy2],
        }

        initial_pose_poselib = poselib._core.CameraPose()
        initial_pose_poselib.Rt = initial_pose
        pose, info = poselib.estimate_relative_pose(
            pts1,
            pts2,
            camera1,
            camera2,
            {"max_epipolar_error": max_epipolar_error},
            {},
            initial_pose=initial_pose_poselib,
        )
        Rt = np.array(pose.Rt)
        # initial_pose - np.array(pose.Rt)

        t_norm = ((initial_pose[:, 3] ** 2).sum() ** (0.5),)
        Rt[:, 3] *= t_norm / (Rt[:, 3] ** 2).sum() ** (0.5)
        inliers = np.array(info["inliers"], dtype=bool)
        return Rt, inliers, (pts1[inliers], pts2[inliers])
