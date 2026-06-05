from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from scipy.ndimage import (
    gaussian_filter,
    binary_dilation,
)
from skimage.morphology import remove_small_objects

device = "cuda"


def identify_unedited_regions(img_pil, img_edited_pil, merge_channels=False):
    (width, height) = img_pil.size
    img_difference = np.array(img_edited_pil, dtype=float) - np.array(img_pil)
    img_difference_1D = (img_difference**2).sum(-1) ** (0.5)
    if np.quantile(img_difference_1D, 0.05) > 30:
        img1_edited_mask = np.ones((height, width))
    else:
        gmm = GaussianMixture(n_components=2, random_state=42)
        if merge_channels:
            gmm.fit(img_difference_1D.reshape(-1, 1))
            order = np.argsort((gmm.means_**2).sum(-1))
            gmm.means_ = gmm.means_[order]
            gmm.covariances_ = gmm.covariances_[order]
            gmm.weights_ = gmm.weights_[order]
            gmm.precisions_ = gmm.precisions_[order]
            gmm.precisions_cholesky_ = gmm.precisions_cholesky_[order]
            img1_edited_mask = gmm.predict(
                img_difference_1D.reshape(-1, 1)
            ).reshape(height, width)
        else:
            gmm.fit(img_difference.reshape(-1, 3))
            order = np.argsort((gmm.means_**2).sum(-1))
            gmm.means_ = gmm.means_[order]
            gmm.covariances_ = gmm.covariances_[order]
            gmm.weights_ = gmm.weights_[order]
            gmm.precisions_ = gmm.precisions_[order]
            gmm.precisions_cholesky_ = gmm.precisions_cholesky_[order]
            img1_edited_mask = gmm.predict(
                img_difference.reshape(-1, 3)
            ).reshape(height, width)

    img1_edited_mask = process_mask(
        process_mask(
            img1_edited_mask,
            min_region_size=1,
            dilation_iterations=1,
            dilation_size=3,
            dilation_sigma=1,
        ),
        min_region_size=100,
        dilation_iterations=1,
        dilation_size=5,
        dilation_sigma=1,
    )
    return img1_edited_mask


def erode_mask(mask: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
    """Erode a binary mask with kernel size adaptive to image size.

    Args:
        mask: [H, W] binary mask (0/1 float or bool).
        strength: 0.0 (no erosion) to 1.0 (heavy erosion).

    Returns:
        [H, W] eroded mask (float 0/1).
    """
    max_kernel = max(mask.shape) // 10  # ~10% of larger dimension
    k = max(int(strength * max_kernel) // 2 * 2 + 1, 1)
    if k <= 1:
        return mask.float()
    pad = k // 2
    m = mask.float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    eroded = -F.max_pool2d(-m, kernel_size=k, stride=1, padding=pad)
    return eroded.squeeze(0).squeeze(0)


def tensor_to_pil(t: torch.Tensor, normalized: bool = True) -> Image.Image:
    """[3, H, W] float [0,1] → PIL Image."""
    arr = t.cpu().detach().float().numpy().transpose(1, 2, 0)
    if normalized:
        arr = np.clip(arr, 0.0, 1.0) * 255
    arr = np.round(arr).astype(np.uint8)
    return Image.fromarray(arr)


def gaussian_structure(size=7, sigma=1.5, threshold=0.3):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)

    g = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    g /= g.max()

    return g > threshold


def process_mask(
    mask: np.ndarray,
    min_region_size: int = 50,
    dilation_iterations: int = 1,
    dilation_size: int = 11,
    dilation_sigma: int = 2.5,
    dilation_threshold: int = 0.3,
    gaussian_sigma: float = 0,
    pad_edges: bool = True,
) -> np.ndarray:
    """
    Clean, smooth mask

    Parameters:
        mask: np.ndarray - 2D mask with values 0-1 (or floats)
        gaussian_sigma: float - Sigma for Gaussian convolution
        min_region_size: int - Minimum pixel area to keep
        pad_edges: bool - Pad edges before smoothing to avoid fade artifacts

    Returns:
        np.ndarray -- mask
    """
    final_mask = mask.astype(bool)
    final_mask = remove_small_objects(final_mask, min_size=min_region_size)
    if dilation_iterations > -1:
        final_mask = binary_dilation(
            final_mask,
            structure=gaussian_structure(
                size=dilation_size,
                sigma=dilation_sigma,
                threshold=dilation_threshold,
            ),
            iterations=dilation_iterations,
        )

    if gaussian_sigma > 0:
        if pad_edges:
            pad = int(gaussian_sigma * 3)
            padded_mask = np.pad(
                final_mask, pad, mode="constant", constant_values=0
            )
            final_mask = gaussian_filter(
                padded_mask.astype(float), sigma=gaussian_sigma
            )
            final_mask = final_mask[pad:-pad, pad:-pad]
        else:
            final_mask = gaussian_filter(
                final_mask.astype(float), sigma=gaussian_sigma
            )

    return final_mask
