from pathlib import Path
import torch
import os
import time
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.anysplat.misc.image_io import save_interpolated_video, save_image
from src.anysplat.model.ply_export import export_ply
from src.anysplat.model.model.anysplat import AnySplat
from src.anysplat.utils.image import process_image
from render_help_function import (
    generate_spherical_trajectory,
    render_and_save_trajectory,
)


def run_trajectory(
    use_spherical, output_folder, device,
    pred_all_extrinsic, pred_all_intrinsic, b, h, w,
    gaussians, decoder,
):
    if use_spherical:
        traj_extrinsics, traj_intrinsics = generate_spherical_trajectory(
            pred_all_extrinsic, pred_all_intrinsic, num_frames=60
        )
        render_and_save_trajectory(
            traj_extrinsics, traj_intrinsics, b, h, w, gaussians, output_folder, decoder
        )
    else:
        _, _, interpolated_rgb_frames = save_interpolated_video(pred_all_extrinsic, pred_all_intrinsic, b, h, w, gaussians, output_folder, decoder)
        frames_dir = os.path.join(output_folder, "interpolated_rgb_frames")
        os.makedirs(frames_dir, exist_ok=True)
        for idx, frame in enumerate(interpolated_rgb_frames):
            save_image(frame, os.path.join(frames_dir, f"{idx:04d}.png"))


def load_anysplat():
    # Load the model from Hugging Face
    model = AnySplat.from_pretrained("lhjiang/anysplat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def run_anysplat(image_folder, resize_input=False, spherical=True, interpolated=False, both=False, model=None):
    # default to spherical if nothing specified
    if not interpolated and not both:
        spherical = True

    output_folder = os.path.join(image_folder, "AnySplat")
    os.makedirs(output_folder, exist_ok=True)

    if model is None:
        model = load_anysplat()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load Images
    image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if resize_input:
        import tempfile
        _tmp_dir = tempfile.mkdtemp()
        resized_paths = []
        for p in image_paths:
            img = Image.open(p)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img = img.resize((576, 576), Image.LANCZOS)
            out_path = os.path.join(_tmp_dir, os.path.basename(p))
            img.save(out_path)
            resized_paths.append(out_path)
        image_paths = resized_paths
    images = [process_image(img_path) for img_path in image_paths]
    images = torch.stack(images, dim=0).unsqueeze(0).to(device) # [1, K, 3, 448, 448]
    b, v, _, h, w = images.shape

    # Run Inference
    _t0 = time.perf_counter()
    gaussians, pred_context_pose = model.inference((images+1)*0.5)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[AnySplat] 3DGS generation from {v} images: {time.perf_counter() - _t0:.2f}s")

    pred_all_extrinsic = pred_context_pose['extrinsic']
    pred_all_intrinsic = pred_context_pose['intrinsic']

    traj_kwargs = dict(
        output_folder=output_folder,
        device=device,
        pred_all_extrinsic=pred_all_extrinsic,
        pred_all_intrinsic=pred_all_intrinsic,
        b=b, h=h, w=w,
        gaussians=gaussians,
        decoder=model.decoder,
    )

    if both:
        run_trajectory(use_spherical=True, **traj_kwargs)
        run_trajectory(use_spherical=False, **traj_kwargs)
    else:
        run_trajectory(use_spherical=spherical, **traj_kwargs)

    export_ply(gaussians.means[0], gaussians.scales[0], gaussians.rotations[0], gaussians.harmonics[0], gaussians.opacities[0], Path(output_folder) / "edited_gaussian.ply")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, required=True, help="Path to folder containing input images")
    parser.add_argument("--resize_input", action="store_true", default=False, help="Resize input images to 576x576 before feeding to AnySplat")
    traj_group = parser.add_mutually_exclusive_group()
    traj_group.add_argument("--spherical", action="store_true", default=True, help="Use spherical trajectory")
    traj_group.add_argument("--interpolated", action="store_true", default=False, help="Use interpolated input poses")
    traj_group.add_argument("--both", action="store_true", default=False, help="Run both spherical and interpolated trajectories separately")
    args = parser.parse_args()
    run_anysplat(args.image_folder, args.resize_input, args.spherical, args.interpolated, args.both)