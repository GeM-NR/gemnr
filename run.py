import os
import argparse
from pathlib import Path
import warnings
import requests
from io import BytesIO

from PIL import Image

warnings.filterwarnings("ignore")


def main(args):
    print("Setting up...")
    if args.interactive:
        from run_interactive import GemNREditingApp

        GemNREditingApp().run(inline=True)
    else:
        inputs = args.input
        anchor_input_ref = args.anchor_input_ref
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

        from gemnr import GemNR

        gem_nr = GemNR(resolution=resolution, seed=seed, token=token)

        # Gathering inputs
        im_pil_list, im_names = input_to_imgs(inputs)
        im_pil_list = [gem_nr.crop_resize(im_pil) for im_pil in im_pil_list]
        anchor_cond_pil = (
            None
            if not anchor_input_ref
            else gem_nr.crop_resize(
                input_to_imgs(anchor_input_ref)[0][0]
            )
        )

        # Editing
        out_im_pil_list = gem_nr.edit(
            im_pil_list,
            edit_text_prompt=edit_text_prompt,
            anchor_idx=anchor_idx,
            anchor_cond_pil=anchor_cond_pil,
        )

        # Logging
        edit_id = "_".join(
            edit_text_prompt.lower()
            .replace(",", "")
            .replace(".", "")
            .split(" ")[:20]
        )
        log_dir = Path(f"{output_folder}/{edit_id}")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        for in_im, out_im, im_name in zip(
            im_pil_list, out_im_pil_list, im_names
        ):
            out_im.save(log_dir / f"edited{seed:02}_{im_name}.jpg")
            in_im.save(log_dir / f"unedited_{im_name}.jpg")
        print(f"Saved to {str(log_dir)}")


def input_to_imgs(inputs: str) -> list[Image.Image]:
    img_suffixes = [".jpg", ".jpeg", ".png", ".heic"]
    if " " in inputs or any(
        [inputs.lower().endswith(s) for s in img_suffixes]
    ):
        if inputs.startswith("[") and inputs.endswith("]"):
            inputs = inputs[1:-1]
        im_paths = inputs.split(",")
    else:
        im_paths = [
            str(p)
            for p in Path(inputs).glob("*")
            if p.suffix.lower() in img_suffixes
        ]
        im_paths = sorted(im_paths)

    assert len(im_paths) > 0, f"No valid image paths found in {inputs}"

    if any(p.lower().endswith(".heic") for p in im_paths):
        from pillow_heif import register_heif_opener

        register_heif_opener()

    im_pil_list = []
    for im_path in im_paths:
        if im_path.lower().startswith(("http://", "https://")):
            response = requests.get(im_path)
            response.raise_for_status()
            img_data = BytesIO(response.content)
        else:
            img_data = im_path
        im_pil_list.append(Image.open(img_data).convert("RGB"))

    im_names = [Path(im_path).stem for im_path in im_paths]
    return im_pil_list, im_names


if __name__ == "__main__":
    # default_input = "./assets/bike"
    # default_edit_text = "Change the bicycle to a dirt motorbike"
    default_input = "./assets/stone_horse"
    default_edit_text = (
        "Change the horse statue to a lion statue without a saddle"
    )
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
        "-c",
        "--anchor-input-ref",
        type=str,
        help="(optional) Path to a reference image for an anchor edit",
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
