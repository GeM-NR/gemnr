import os
import argparse
from pathlib import Path
import warnings

from PIL import Image

from gemnr import GemNR

warnings.filterwarnings("ignore")


def main(args):
    interactive = args.interactive
    inputs = args.input
    edit_text_prompt = args.edit_text
    anchor_idx = args.anchor_idx
    seed = args.seed
    resolution = args.resolution
    token = args.token
    output_folder = args.output

    # Method initialization
    assert (
        token or "HF_TOKEN" in os.environ
    ), "Hugging Face token not provided. Please set the HF_TOKEN environment variable or pass the token as an argument."

    gem_nr = GemNR(resolution=resolution, seed=seed, token=token)

    # Run once or in interactive mode
    while True:
        if interactive:
            inputs = (
                input(
                    f"Input folder or list of image paths separated by commas (image {anchor_idx} used as an anchor) [{inputs}]: "
                )
                or inputs
            )

            edit_text_prompt = (
                input(f"Edit text prompt [{edit_text_prompt}]: ").strip()
                or edit_text_prompt
            )

        # Gathering inputs
        im_pil_list, im_paths = input_to_imgs(inputs)
        im_pil_list = [gem_nr.crop_resize(im_pil) for im_pil in im_pil_list]

        # Editing
        out_im_pil_list = gem_nr.edit(
            im_pil_list,
            edit_text_prompt=edit_text_prompt,
            anchor_idx=anchor_idx,
        )

        # Logging
        edit_id = "_".join(
            edit_text_prompt.lower()
            .replace(",", "")
            .replace(".", "")
            .split(" ")[:10]
        )
        log_dir = Path(f"{output_folder}/{edit_id}")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        for in_im, out_im, im_path in zip(
            im_pil_list, out_im_pil_list, im_paths
        ):
            im_name = Path(im_path).stem
            out_im.save(log_dir / f"edited_{im_name}.jpg")
            in_im.save(log_dir / f"unedited_{im_name}.jpg")
        print(f"Saved to {str(log_dir)}")

        if not interactive:
            return


def input_to_imgs(inputs: str) -> list[Image.Image]:
    img_suffixes = [".jpg", ".jpeg", ".png", ".heic"]
    if " " in inputs or any(
        [inputs.lower().endswith(s) for s in img_suffixes]
    ):
        im_paths = [Path(f.strip()) for f in inputs.split(",")]
    else:
        im_paths = [
            p
            for p in Path(inputs).glob("*")
            if p.suffix.lower() in img_suffixes
        ]
        im_paths = sorted(im_paths)

    assert len(im_paths) > 0, f"No valid image paths found in {inputs}"
    if any(p.suffix.lower() == ".heic" for p in im_paths):
        from pillow_heif import register_heif_opener

        register_heif_opener()

    im_pil_list = [Image.open(im_path).convert("RGB") for im_path in im_paths]
    return im_pil_list, im_paths


if __name__ == "__main__":
    # default_input = "./assets/bike"
    # default_edit_text = "Change the bicycle to a dirt motorbike"
    default_input = "./assets/stone_horse"
    default_edit_text = "Change the horse statues to a lion statue"
    default_anchor_idx = 1
    default_seed = 0
    default_output = "./results"
    default_resolution = 512

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=default_input,
        help="Input folder or list of image paths",
    )
    parser.add_argument(
        "-e",
        "--edit-text",
        type=str,
        default=default_edit_text,
        help="Edit text prompt",
    )
    parser.add_argument(
        "-a",
        "--anchor-idx",
        type=int,
        default=default_anchor_idx,
        help="Anchor image index",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=default_output,
        help="Output folder",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=default_seed, help="Random seed"
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=default_resolution,
        help="Resolution for editing (e.g., 512)",
    )
    parser.add_argument(
        "-t",
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (or set HF_TOKEN environment variable)",
    )
    args = parser.parse_args()
    main(args)
