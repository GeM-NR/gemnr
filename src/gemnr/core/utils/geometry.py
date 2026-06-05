import numpy as np
import torch


def relpose_from_absolute(ext1_w2c, ext2_w2c):
    """
    Compute relative pose T_rel = T2 @ T1^{-1} (w2c convention).
    Args:
        T1_3x4: (3, 4) w2c pose [R1|t1]
        T2_3x4: (3, 4) w2c pose [R2|t2]
    Returns:
        (3, 4) relative pose mapping from camera 1 to camera 2
    """
    R1, t1 = ext1_w2c[:3, :3], ext1_w2c[:3, 3]
    R2, t2 = ext2_w2c[:3, :3], ext2_w2c[:3, 3]
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    return np.hstack([R_rel, t_rel[:, None]])


def so3_log(R):
    cos_theta = (torch.trace(R) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    if torch.isclose(theta, torch.tensor(0.0, device=R.device, dtype=R.dtype)):
        return torch.zeros(3, device=R.device, dtype=R.dtype)
    lnR = (
        theta
        / (2 * torch.sin(theta))
        * torch.tensor(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
            device=R.device,
            dtype=R.dtype,
        )
    )
    return lnR


def se3_log(T):
    R = T[:3, :3]
    t = T[:3, 3]
    w = so3_log(R)
    v = t
    return torch.cat([w, v])


def se3_distance_w2c(extrinsics1, extrinsics2):
    """
    SE(3) distance between two (3,4) w2c pose matrices.
    Args:
        extrinsics1, extrinsics2: (3,4) torch tensors
    Returns:
        Scalar distance
    """
    bottom = torch.tensor(
        [[0, 0, 0, 1]], dtype=extrinsics1.dtype, device=extrinsics1.device
    )
    T1 = torch.cat([extrinsics1, bottom], dim=0)
    T2 = torch.cat([extrinsics2, bottom], dim=0)
    T_rel = torch.linalg.inv(T1) @ T2
    xi = se3_log(T_rel)
    return torch.norm(xi)


def transform_rigid(X, xform3x4):
    # X: (3,M)
    X_h = np.vstack([X, np.ones((1, X.shape[1]))])  # (4,M)
    Xt = (xform3x4 @ X_h)[:3]  # (3,M)
    return Xt


def backproject(pix, depths, ext_w2c, K):
    # pix: # (2,M)
    # depths: # M
    pix_h = np.concat([pix, np.ones_like(pix[..., :1])], axis=-1)
    # (3,M)
    K_inv = np.linalg.inv(K)  # (3,3)
    c2w = np.linalg.inv(_as_homogeneous44(ext_w2c))  # (4,4)
    rays = K_inv @ pix_h.T  # (3,M)
    Xc = rays * depths[None, :]  # (3,M)
    Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
    Xw = (c2w @ Xc_h)[:3].astype(np.float32)  # (3,M)
    return Xw


def project(Xw, K, ext_w2c=None):
    # X: (3,M)
    # ext_w2c: (3,4)
    # K: (3,3)
    if ext_w2c is not None:
        Xc = transform_rigid(Xw, ext_w2c)
    else:
        Xc = Xw
    rays = K @ Xc
    pix = rays[:2] / rays[2]  # (2,M)
    return pix


def normalize_intrinsics(intr, W, H):
    intr_norm = intr.copy()
    intr_norm[..., 0, :] /= W
    intr_norm[..., 1, :] /= H
    return intr_norm


def unnormalize_intrinsics(intr_norm, W, H):
    intr = intr_norm.copy()
    intr[..., 0, :] *= W
    intr[..., 1, :] *= H
    return intr


def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    """
    Accept (4,4) or (3,4) extrinsic parameters, return (4,4) homogeneous matrix.
    """
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        H = np.eye(4, dtype=ext.dtype)
        H[:3, :4] = ext
        return H
    raise ValueError(f"extrinsic must be (4,4) or (3,4), got {ext.shape}")
