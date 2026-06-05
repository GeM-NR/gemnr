from pathlib import Path

import numpy as np
import torch
from matplotlib import cm, colors
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from .viz_utils import add_title_to_img, concat_images_grid


def save_met3r_visuals(meta_data, output_path):
    if "meta_data_list" in meta_data:
        heatmap_imgs = []
        for i, meta_data_pair in enumerate(meta_data["meta_data_list"]):
            img_pair = meta_data_pair["imgs"]
            score_map = meta_data_pair["score_map"]
            score = meta_data_pair["score"]
            heatmap_img = make_heatmap_img(
                *img_pair, score_map, score, font_scale=1.25, add_orig_imgs=False
            )
            heatmap_imgs.append(heatmap_img)

        scores = meta_data["score_list"]
        title = f"Average MEt3R = {np.mean(scores):.4f}"
        out_img = concat_images_grid(heatmap_imgs, n_cols=4)
        out_img = add_title_to_img(out_img, title=title, font_scale=2.5)
    else:
        img_pair = meta_data["imgs"]
        score_map = meta_data["score_map"]
        score = meta_data["score"]
        out_img = make_heatmap_img(*img_pair, score_map, score)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(output_path)
    print(f"Saved to: {output_path}")


def make_heatmap_img(
    img_a: Image.Image,
    img_b: Image.Image,
    heatmap: np.ndarray,
    score: float,
    font_scale: float = 2.2,
    add_orig_imgs: bool = True,
) -> None:

    W, H = img_a.size

    heatmap_np = np.clip(heatmap, 0.0, 1.0)

    cmap = LinearSegmentedColormap.from_list("yellow_red", ["yellow", "red"])
    norm = colors.Normalize(vmin=0, vmax=1, clip=True)
    rgba = cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(heatmap_np)
    heatmap_img = (
        Image.fromarray((rgba * 255).astype(np.uint8))
        .convert("RGB")
        .resize((W, H), Image.BILINEAR)
    )

    if add_orig_imgs:
        final_img = Image.new("RGB", (W * 3, H), "white")
        final_img.paste(img_a, (0, 0))
        final_img.paste(heatmap_img, (W, 0))
        final_img.paste(img_b, (W * 2, 0))
    else:
        final_img = heatmap_img

    title = f"MEt3R = {score:.4f}"
    return add_title_to_img(final_img, title, font_scale=font_scale)
