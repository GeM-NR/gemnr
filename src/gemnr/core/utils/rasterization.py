import torch
import numpy as np
from skimage.morphology import remove_small_objects, binary_dilation, disk

from .geometry import project, backproject, transform_rigid


def backproject_for_rasterization(
    img,
    pix,
    K,
    ext_w2c,
    ext_w2c_to,
    conf,
    depth,
    conf_thresh: float = 0,
    depth_mask=None,
    depth_min: float = 1e-8,
    depth_max: float = 1e3,
):
    rgb = img.reshape((-1, 3))  # (H*W,3)
    pix = pix.reshape((-1, 2))  # (H*W,2)
    conf_flat = conf.reshape(-1)  # H*W
    depth_c1 = depth.reshape(-1)  # H*W

    valid = np.isfinite(depth_c1) & (conf_flat >= conf_thresh)
    valid &= True if depth_mask is None else depth_mask.reshape(-1)
    valid &= (depth_c1 > depth_min) & (depth_c1 < depth_max)
    vidx = np.flatnonzero(valid)  # M
    if len(vidx) == 0:
        print("len(vidx) == 0")
        return [], [], [], []

    depth_c1 = depth_c1[vidx]
    Xw = backproject(pix[vidx], depth_c1, ext_w2c, K)  # (3,M)
    Xc2 = transform_rigid(Xw, ext_w2c_to)  # (3,M)
    depth_c2 = Xc2[2, :]

    valid_c2 = (depth_c2 > depth_min) & (depth_c2 < depth_max)
    valid_c2 &= depth_c1 / depth_c2 < 10
    vidx2 = np.flatnonzero(valid_c2)  # K
    if len(vidx2) == 0:
        print("len(vidx2) == 0")
        return [], [], [], []

    Xc2 = Xc2[:, vidx2]
    depth_c2 = depth_c2[vidx2]
    depth_c1 = depth_c1[vidx2]

    vidx = vidx[vidx2]
    colors = rgb[vidx]
    confidences = conf_flat[vidx]

    radii = depth_c1 / depth_c2

    return Xc2, colors, confidences, radii


def rasterize_points_hard(
    X,
    K,
    features,
    W,
    H,
    radii,
    max_r: int = 10,
    overload_device: torch.device | None = None,
    min_size_percent: float | None = None,  # 5,
    min_size2_percent: float | None = None,  # 0.1,
    erode_radius_percent: float | None = None,  # 1,
):
    # X: (3,M)
    # K: (3,3)
    # features: [(M, F1), ..., (M, FK)]

    device = X.device if isinstance(X, torch.Tensor) else overload_device
    X = torch.as_tensor(X, device=device, dtype=torch.float32)
    K = torch.as_tensor(K, device=device, dtype=torch.float32)
    radii = torch.as_tensor(radii, device=device, dtype=torch.float32)

    assert (X[2, :] > 0).all()

    # Sort points back-to-front so that within-offset duplicate pixel writes
    # resolve correctly (last write wins = closest point).
    order = torch.argsort(X[2, :], descending=True)
    X = X[:, order]
    radii = radii[order]

    depth = torch.full(
        (H, W), float("inf"), dtype=torch.float32, device=device
    )
    features_list = (
        features if isinstance(features, (list, tuple)) else [features]
    )
    image_list = [None] * len(features_list)

    for i, feat in enumerate(features_list):
        feat = torch.as_tensor(feat, device=device)
        feat = feat[:, None] if feat.ndim == 1 else feat
        feat = feat[order]
        F = feat.shape[1]
        features_list[i] = feat
        image_list[i] = torch.zeros((H, W, F), dtype=feat.dtype, device=device)

    max_r = min(int(torch.ceil(radii.max()).item()), max_r)
    pix = project(X, K)

    uu = pix[0, :].floor().long()
    vv = pix[1, :].floor().long()
    frac_u = pix[0, :] - uu.float()
    frac_v = pix[1, :] - vv.float()

    for dy in range(-max_r, max_r + 1):
        for dx in range(-max_r, max_r + 1):
            dist2 = (dx - frac_u) ** 2 + (dy - frac_v) ** 2

            # mask per point
            mask = dist2 <= (radii**2)

            px = uu + dx
            py = vv + dy

            valid = (px >= 0) & (px < W) & (py >= 0) & (py < H) & mask

            if not valid.any():
                continue

            pxv = px[valid]
            pyv = py[valid]
            zv = X[2, valid]

            # flatten indices
            idx = pyv * W + pxv

            depth_flat = depth.view(-1)

            # Atomically update depth buffer with minimum depth per pixel
            # (handles duplicate pixel indices correctly)
            depth_flat.scatter_reduce_(
                0, idx, zv, reduce="amin", include_self=True
            )

            # Write features only for points whose depth matches the buffer
            winners = zv <= depth_flat[idx]

            if winners.any():
                for image, feat in zip(image_list, features_list):
                    fv = feat[valid]
                    image_flat = image.view(-1, feat.shape[1])
                    image_flat[idx[winners]] = fv[winners]

    # Remove small connected components and erode edges
    rendered_mask_tensor = depth < float("inf")
    rendered_mask = rendered_mask_tensor.cpu().numpy()
    if min_size_percent is not None:
        min_size = max(1, (min(H, W) * min_size_percent) // 100)
        cleaned_mask = remove_small_objects(rendered_mask, min_size=min_size)
    else:
        cleaned_mask = rendered_mask

    # Identify where foreground borders large background regions
    if min_size2_percent is not None:
        min_size2 = max(1, (min(H, W) * min_size2_percent) // 100)
        large_bg = remove_small_objects(~cleaned_mask, min_size=min_size2)
    else:
        large_bg = ~cleaned_mask

    # Erode there
    if erode_radius_percent is not None:
        erode_radius = max(1, (min(H, W) * erode_radius_percent) // 100)
        dilated_bg = binary_dilation(large_bg, disk(erode_radius))
    else:
        dilated_bg = large_bg
    cleaned_mask = cleaned_mask & ~dilated_bg

    discard = rendered_mask_tensor & ~torch.as_tensor(
        cleaned_mask, device=device
    )
    depth[discard] = float("inf")
    for image in image_list:
        image[discard] = 0

    mask = (image_list[0] ** 2).sum(-1) > 1e-10
    return [image.permute(2, 0, 1) for image in image_list], depth, mask
