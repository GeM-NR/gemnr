import math
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

from src.anysplat.misc.image_io import save_video


def generate_spherical_trajectory(pred_extrinsics, pred_intrinsics, num_frames=120):
    """
    Generate a smooth looping trajectory with gentle azimuth and elevation
    variation around the original viewpoint, always looking at the scene center.

    Uses Lissajous-style oscillations so the path forms a smooth loop.
    Camera gaze is biased toward the lookat center to enhance 3D parallax.
    """
    b, v, _, _ = pred_extrinsics.shape
    device = pred_extrinsics.device

    # --- 1. Camera positions (world space) ---
    cam_positions = pred_extrinsics[:, :, :3, 3]  # [b, v, 3]

    # --- 2. Average gaze direction (column 2 of C2W rotation = local +Z) ---
    avg_forward = pred_extrinsics[:, :, :3, 2].mean(dim=1)  # [b, 3]
    avg_forward = avg_forward / (avg_forward.norm(dim=-1, keepdim=True) + 1e-8)

    # --- 3. Look-at point: push forward from camera centroid along gaze ---
    cam_centroid = cam_positions.mean(dim=1)  # [b, 3]
    baseline = (cam_positions - cam_centroid.unsqueeze(1)).norm(dim=-1).mean(dim=1)
    scene_depth = torch.clamp(baseline * 5.0, min=1.0)
    lookat = cam_centroid + avg_forward * scene_depth.unsqueeze(-1)

    # --- 4. Orbit radius from centroid to lookat ---
    center_to_lookat = lookat - cam_centroid
    radius = center_to_lookat.norm(dim=-1)  # [b]

    # --- 5. Reference spherical angles (where the original camera centroid sits) ---
    direction = cam_centroid - lookat  # [b, 3] points from lookat back to camera
    direction_norm = direction / (radius.unsqueeze(-1) + 1e-8)
    ref_elevation = torch.asin(direction_norm[:, 1].clamp(-1, 1))  # [b]
    ref_azimuth = torch.atan2(direction_norm[:, 2], direction_norm[:, 0])  # [b]

    # --- 6. Lissajous oscillation parameters ---
    max_az_deg = 4.0
    max_el_deg = 2
    max_az = math.radians(max_az_deg)
    max_el = math.radians(max_el_deg)

    t = torch.linspace(0, 2 * math.pi, num_frames + 1, device=device)[:-1]

    az_offset = max_az * torch.sin(t)
    el_offset = max_el * torch.sin(2 * t)

    # --- 7. Gaze convergence: how much the look-target shifts toward lookat ---
    # 0.0 = camera looks straight ahead (parallel), 1.0 = looks exactly at lookat
    # Higher values give stronger 3D parallax feeling
    gaze_convergence = 1.35

    # --- 8. World-space up vector (Y-up) ---
    world_up = torch.zeros(b, 3, device=device)
    world_up[:, 1] = 1.0

    traj_extrinsics = []
    for fi in range(num_frames):
        az = ref_azimuth + az_offset[fi]
        el = ref_elevation + el_offset[fi]

        # Spherical -> Cartesian offset from lookat
        x = radius * torch.cos(el) * torch.cos(az)
        y = radius * torch.sin(el)
        z = radius * torch.cos(el) * torch.sin(az)
        cam_pos = lookat + torch.stack([x, y, z], dim=-1)  # [b, 3]

        # --- Gaze target: blend between "straight ahead" and lookat ---
        # "Straight ahead" = a point along the reference forward direction
        # from this camera position (mimics looking parallel / no convergence)
        straight_ahead_target = cam_pos + avg_forward * scene_depth.unsqueeze(-1)
        gaze_target = (1.0 - gaze_convergence) * straight_ahead_target + gaze_convergence * lookat

        # Camera forward direction (toward gaze target, not just lookat)
        forward = gaze_target - cam_pos
        forward = forward / (forward.norm(dim=-1, keepdim=True) + 1e-8)

        # Right = up x forward (standard right-handed look-at formula)
        right = torch.cross(world_up, forward, dim=-1)
        right_norm = right.norm(dim=-1, keepdim=True)
        fallback_up = torch.zeros_like(world_up)
        fallback_up[:, 2] = 1.0
        right_fallback = torch.cross(fallback_up, forward, dim=-1)
        right = torch.where(right_norm > 1e-6, right, right_fallback)
        right = right / (right.norm(dim=-1, keepdim=True) + 1e-8)

        # Corrected up = forward x right
        up = torch.cross(forward, right, dim=-1)
        up = up / (up.norm(dim=-1, keepdim=True) + 1e-8)

        # C2W rotation: columns = [right, up, forward]
        R = torch.stack([right, up, forward], dim=-1)  # [b, 3, 3]

        E = torch.eye(4, device=device).unsqueeze(0).expand(b, -1, -1).clone()
        E[:, :3, :3] = R
        E[:, :3, 3] = cam_pos
        traj_extrinsics.append(E.unsqueeze(1))

    traj_extrinsics = torch.cat(traj_extrinsics, dim=1)

    avg_intrinsic = pred_intrinsics.mean(dim=1, keepdim=True)
    traj_intrinsics = avg_intrinsic.expand(b, num_frames, -1, -1)

    return traj_extrinsics, traj_intrinsics



def render_and_save_trajectory(traj_extrinsics, traj_intrinsics, b, h, w, gaussians, save_path, decoder_func):
    """Render a pre-built trajectory and save RGB + depth videos."""
    num_frames = traj_extrinsics.shape[1]

    output = decoder_func.forward(
        gaussians,
        traj_extrinsics,
        traj_intrinsics.float(),
        torch.ones(b, num_frames, device=traj_extrinsics.device) * 0.1,
        torch.ones(b, num_frames, device=traj_extrinsics.device) * 100,
        (h, w),
    )

    video = output.color[0].clip(min=0, max=1)
    depth = output.depth[0]

    # Sample every N-th frame for quantile to stay within PyTorch's ~16M element limit
    sample_step = max(1, depth.shape[0] // 10)
    depth_sample = depth[::sample_step]
    d_min = depth_sample.quantile(0.01)
    d_max = depth_sample.quantile(0.99)
    depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
    depth_colored = torch.from_numpy(
        plt.cm.turbo(depth_norm.cpu().numpy())[..., :3]
    ).permute(0, 3, 1, 2).to(depth.device).clip(min=0, max=1)

    save_video(video, os.path.join(save_path, "spherical_rgb.mp4"))
    save_video(depth_colored, os.path.join(save_path, "spherical_depth.mp4"))

    # Save individual frames as PNG images
    frames_dir = os.path.join(save_path, "spherical_rgb_frames")
    os.makedirs(frames_dir, exist_ok=True)
    frames_np = (video.cpu().detach().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    for fi, frame in enumerate(frames_np):
        Image.fromarray(frame).save(os.path.join(frames_dir, f"frame_{fi:04d}.png"))

    # Save as GIF
    import imageio
    gif_path = os.path.join(save_path, "spherical_rgb.gif")
    imageio.mimsave(gif_path, frames_np, fps=10)

    traj_alpha = output.alpha[0]  # [T, H, W] accumulated opacity
    return os.path.join(save_path, "spherical_rgb.mp4"), video, traj_alpha


def render_at_input_poses(pred_extrinsics, pred_intrinsics, b, h, w, gaussians, decoder_func):
    """
    Render the scene at the input camera poses.

    Returns:
        color: [b*v, 3, H, W] in [0, 1]
        depth: [b*v, H, W]  z-depth in camera space
    """
    v = pred_extrinsics.shape[1]
    output = decoder_func.forward(
        gaussians,
        pred_extrinsics,
        pred_intrinsics.float(),
        torch.ones(b, v, device=pred_extrinsics.device) * 0.1,
        torch.ones(b, v, device=pred_extrinsics.device) * 100,
        (h, w),
    )
    color = output.color.clamp(0, 1).view(b * v, 3, h, w)   # [b*v, 3, H, W]
    depth = output.depth.view(b * v, h, w)                   # [b*v, H, W]
    return color, depth
