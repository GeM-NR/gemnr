from math import comb
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from met3r import MEt3R
from .metrics_utils import is_connected


def compute_met3r(
    img_pil_list: list,
    max_pairs: int = 20,
    **kwargs,
):
    N = len(img_pil_list)
    if N==2:
        return compute_met3r_pair(img_pil_list, **kwargs)
    
    rs = np.random.RandomState(42)
    N_pairs = comb(N, 2)
    idxs = rs.permutation(N_pairs)[: min(N_pairs, max_pairs)]
    pairs_list = [
        pair for i, pair in enumerate(combinations(range(N), 2)) if i in idxs
    ]
    if N != 10:
        assert is_connected(N, pairs_list)
        # if N == 10, we know it is connected

    score_list = []
    meta_data_list = []
    for pair in pairs_list:
        img_pair = [img_pil_list[pair[0]], img_pil_list[pair[1]]]
        score_pair, meta_data_pair = compute_met3r_pair(img_pair, **kwargs)
        score_list.append(score_pair)
        meta_data_list.append(meta_data_pair)

    meta_data = {
        "pair_combinations": pairs_list,
        "score_list": score_list,
        "meta_data_list": meta_data_list,
    }
    return np.mean(score_list), meta_data


def compute_met3r_pair(
    img_pair: list,
    met3r_model=None,
):
    if not met3r_model:
        met3r_model = MEt3R(
            img_size=None,  # use input resolution directly — avoids internal padding artifacts
            use_norm=True,
            backbone="mast3r",
            feature_backbone="dino16",
            feature_backbone_weights="mhamilton723/FeatUp",
            upsampler="featup",
            distance="cosine",
            freeze=True,
        ).cuda()

    assert (
        len(img_pair) == 2
    ), f"This function supports 2 images only, but got {len(img_pair)}"
    ref_tensor = pil_to_tensor_met3r(img_pair[0]).cuda()
    gen_tensor = pil_to_tensor_met3r(img_pair[1]).cuda()
    if met3r_model.img_size:
        H, W = met3r_model.img_size, met3r_model.img_size
    else:
        H = min(ref_tensor.shape[1], gen_tensor.shape[1])
        W = min(ref_tensor.shape[2], gen_tensor.shape[2])
    
    if ref_tensor.shape[1] != H and ref_tensor.shape[2] != W:
        ref_tensor = F.interpolate(ref_tensor[None], size=(H, W), mode='bilinear', align_corners=False)[0]
    if gen_tensor.shape[1] != H and gen_tensor.shape[2] != W:
        gen_tensor = F.interpolate(gen_tensor[None], size=(H, W), mode='bilinear', align_corners=False)[0]
    inputs = torch.stack([ref_tensor, gen_tensor], dim=0).unsqueeze(0)
    # (1, 2, 3, H, W)

    with torch.no_grad():
        score, score_map, *_ = met3r_model(
            images=inputs,
            return_overlap_mask=False,
            return_score_map=True,
            return_projections=False,
        )

    score_val = score.mean().item()
    meta_data = {
        "imgs": img_pair,
        "score_map": score_map.squeeze().cpu().float(),
        "score": score_val,
    }
    return score_val, meta_data


def pil_to_tensor_met3r(img) -> torch.Tensor:
    arr = np.array(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)


def tensor_to_pil_met3r(t: torch.Tensor) -> Image.Image:
    """Convert a (3, H, W) tensor in [-1, 1] to a PIL RGB image."""
    arr = (
        ((t.cpu().float().numpy().transpose(1, 2, 0) + 1.0) * 127.5)
        .clip(0, 255)
        .astype(np.uint8)
    )
    return Image.fromarray(arr)