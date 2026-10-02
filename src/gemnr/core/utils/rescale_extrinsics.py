import numpy as np
from PIL import Image


def _center_crop(image: Image.Image, K: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    """Center square crop of `image`; returns the crop and its intrinsics normalized by crop size."""
    width, height = image.size
    size = min(width, height)
    left, top = (width - size) // 2, (height - size) // 2
    K = np.array(K, dtype=np.float64)
    K[0, 2] -= left
    K[1, 2] -= top
    K[:2] /= size
    return image.crop((left, top, left + size, top + size)), K


def _camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    R, t = extrinsics[:, :3, :3], extrinsics[:, :3, 3]
    return -np.einsum("nji,nj->ni", R, t)  # -R^T t


def rescale_extrinsics_to_metric(
    imgs_pil: list[Image.Image],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    da3,
    max_views: int = 16,
) -> np.ndarray:
    """
    Rescale world-to-camera extrinsics of arbitrary scale (e.g. from SfM) to DA3's metric scale.

    imgs_pil: overlapping views of the scene (>= 2).
    intrinsics: (N, 3, 3) pixel intrinsics of `imgs_pil`.
    extrinsics: (N, 3, 4) or (N, 4, 4) world-to-camera (OpenCV) poses of `imgs_pil`.
    da3: gemnr.core.utils.depth_anything.DA3Estimator
    max_views: maximum number of views, for safety; raises if exceeded.

    Returns extrinsics of the same shape with translations scaled to meters.

    The scale is the median ratio of pairwise camera-centre distances between pose-free DA3
    predictions and `extrinsics`, corrected by true / predicted focal length.
    """
    assert len(imgs_pil) == len(intrinsics) == len(extrinsics) >= 2
    if len(imgs_pil) > max_views:
        raise ValueError(
            f"Got {len(imgs_pil)} views, more than max_views={max_views}; pass a smaller overlapping subset."
        )
    extrinsics = np.asarray(extrinsics, dtype=np.float64)

    crops, K_true = zip(*(_center_crop(im, K) for im, K in zip(imgs_pil, intrinsics)))
    crops = [im.resize((da3.W_model, da3.H_model)) for im in crops]
    preds = da3.run_DepthAnything3(crops, resize=False)

    c_in = _camera_centers(extrinsics)
    c_da3 = _camera_centers(np.asarray(preds.extrinsics, dtype=np.float64))
    ii, jj = np.triu_indices(len(imgs_pil), k=1)
    d_in = np.linalg.norm(c_in[ii] - c_in[jj], axis=-1)
    d_da3 = np.linalg.norm(c_da3[ii] - c_da3[jj], axis=-1)
    valid = d_in > 1e-3 * d_in.max()
    assert valid.any(), "Camera centres coincide; cannot estimate scale."
    scale = np.median(d_da3[valid] / d_in[valid])

    K_true = np.stack(K_true)
    K_pred = preds.intrinsics  # normalized by DA3 resolution
    f_true = (K_true[:, 0, 0] + K_true[:, 1, 1]) / 2
    f_pred = (K_pred[:, 0, 0] + K_pred[:, 1, 1]) / 2
    scale *= np.median(f_true / f_pred)

    extrinsics_metric = extrinsics.copy()
    extrinsics_metric[:, :3, 3] *= scale
    return extrinsics_metric
