from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm
import torch
import numpy as np

from .viz_utils import add_title_to_img, concat_images_grid


def save_epipolar_visuals(meta_data, pipeline, output_dir):
    if "meta_data_list" in meta_data:
        out_imgs_distances = []
        out_imgs_confidences = []
        for i, meta_data_pair in enumerate(meta_data["meta_data_list"]):
            img_with_distances, img_with_confidences = make_epipolar_img(meta_data_pair)
            out_imgs_distances.append(img_with_distances)
            out_imgs_confidences.append(img_with_confidences)
        title = f"mAA = {meta_data["mAA"]:.2f}"
        out_img_distances = concat_images_grid(out_imgs_distances, n_cols=4)
        out_img_distances = add_title_to_img(out_img_distances, title=title, font_scale=2.5)

        out_img_confidences = concat_images_grid(out_imgs_confidences, n_cols=4)
        out_img_confidences = add_title_to_img(out_img_confidences, title=title, font_scale=2.5)
    else:
        out_img_distances, out_img_confidences = make_epipolar_img(meta_data)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_img_distances.save(output_dir / f"epi_dist_{pipeline}.png")
    out_img_confidences.save(output_dir / f"epi_conf_{pipeline}.png")
    print(f"Saved to {output_dir / f'epi_dist_{pipeline}.png'} and {output_dir / f'epi_conf_{pipeline}.png'}")


def make_epipolar_img(meta_data):
    img1, img2 = meta_data["imgs"]
    pts1 = meta_data["pts1"]
    pts2 = meta_data["pts2"]
    distances1 = meta_data["epipolar_distances_im1"]
    distances2 = meta_data["epipolar_distances_im2"]

    distances1_median = torch.median(distances1).cpu().item()
    distances2_median = torch.median(distances2).cpu().item()

    distances1_mean = torch.mean(distances1).cpu().item()
    distances2_mean = torch.mean(distances2).cpu().item()

    confidences = meta_data["confidences"]
    confidences_mean = torch.mean(confidences).cpu().item()
    confidences_median = torch.median(confidences).cpu().item()

    confidence_thresholds = meta_data["confidence_thresholds"]
    accuracy_thresholds = meta_data["accuracy_thresholds"]
    mAA_im1 = 0
    mAA_im2 = 0
    for accuracy_threshold in accuracy_thresholds:
        for confidence_threshold in confidence_thresholds:
            mAA_im1 += torch.logical_and(
                    distances1 < accuracy_threshold,
                    confidences > confidence_threshold).sum().cpu().item() / pts1.shape[0] * 100
            mAA_im2 += torch.logical_and(
                    distances2 < accuracy_threshold,
                    confidences > confidence_threshold).sum().cpu().item() / pts2.shape[0] * 100
    
    mAA_im1 /= len(accuracy_thresholds) * len(confidence_thresholds)
    mAA_im2 /= len(accuracy_thresholds) * len(confidence_thresholds)

    max_distance = 40  # max(distances1.max(), distances2.max())
    final_img_distances = make_scored_pair(img1, img2, pts1, pts2, scores1=distances1.cpu().numpy(), scores2=distances2.cpu().numpy(), score_max=max_distance, title1=f"Epipolar distance: median: {distances1_median:.4f}, mean: {distances1_mean:.4f}, mAA: {mAA_im1:.2f}% (px thresholds: {accuracy_thresholds})", title2=f"Epipolar distance: median: {distances2_median:.4f}, mean: {distances2_mean:.4f}, mAA: {mAA_im2:.2f}% (px thresholds: {accuracy_thresholds})")

    max_confidence = 1.0
    final_img_confidences = make_scored_pair(img1, img2, pts1, pts2, scores1=confidences.cpu().numpy(), scores2=confidences.cpu().numpy(), score_max=max_confidence, reverse_scores=True, title1=f"RoMa confidence: median: {confidences_median:.4f}, mean: {confidences_mean:.4f}", title2=f"RoMa confidence: median: {confidences_median:.4f}, mean: {confidences_mean:.4f}")
    return final_img_distances, final_img_confidences

def make_scored_pair(img1, img2, pts1, pts2, scores1, scores2, score_max, reverse_scores=False, title1="", title2=""):
    img1_with_pts = draw_pts_with_scores(
        img1,
        pts1,
        scores1,
        top_k=pts1.shape[0],
        score_min=0,
        score_max=score_max,
        reverse_scores=reverse_scores,
        title=title1,
        show_legend=False,
    )
    img2_with_pts = draw_pts_with_scores(
        img2,
        pts2,
        scores2,
        top_k=pts1.shape[0],
        score_max=score_max,
        reverse_scores=reverse_scores,
        title=title2,
    )
    # Concatenate the two images side by side
    final_img = Image.new(
        "RGB",
        (img1_with_pts.width + img2_with_pts.width, img1_with_pts.height),
    )
    final_img.paste(img1_with_pts, (0, 0))
    final_img.paste(img2_with_pts, (img1_with_pts.width, 0))
    return final_img



def draw_pts_with_scores(
    img_pil,
    pts,
    scores,
    top_k=1000,
    min_radius=1,
    max_radius=6,
    cmap_name="rainbow",
    show_legend=True,
    legend_padding=20,
    add_labels=True,
    score_min=None,
    score_max=None,
    reverse_scores=False,
    title=None,
):
    """
    Draw keypoints where:
      - score controls size
      - score controls color (matplotlib cmap)
      - legend shows circle size scale
      - optional title on top
    """

    img_pil = img_pil.copy()

    pts = np.asarray(pts)
    scores = np.asarray(scores).astype(float)

    cmap = cm.get_cmap(cmap_name)

    width, height = img_pil.size

    # -------------------------
    # Score range
    # -------------------------
    if score_min is None:
        score_min = scores.min()
    if score_max is None:
        score_max = scores.max()

    denom = max(score_max - score_min, 1e-12)

    # -------------------------
    # Draw points
    # -------------------------
    draw = ImageDraw.Draw(img_pil)

    k = min(top_k, len(scores))
    top_k_indices = np.argsort(scores)[-k:][::-1]

    for idx in top_k_indices:
        x, y = pts[idx]
        score = scores[idx]

        t = (score - score_min) / denom
        t = np.clip(t, 0.0, 1.0)

        if reverse_scores:
            t = 1.0 - t

        radius = min_radius + t * (max_radius - min_radius)

        rgba = cmap(t)
        color = tuple(int(255 * c) for c in rgba[:3])

        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
        )

    # -------------------------
    # Canvas expansion
    # -------------------------
    extra_width = 140
    top_padding = 60 if title else 20

    out_img = Image.new(
        "RGB",
        (width + extra_width, height + top_padding),
        color=(255, 255, 255),
    )

    out_img.paste(img_pil, (0, top_padding))

    draw_out = ImageDraw.Draw(out_img)
    # -------------------------
    # Title
    # -------------------------
    if title:
        try:
            font = ImageFont.truetype("arial.ttf", size=80)
        except Exception:
            font = None

        draw_out.text(
            (20, 10),
            title,
            fill="black",
            font=font,
        )

    # -------------------------
    # Circle-size legend
    # -------------------------
    if show_legend:
        legend_x = width + 50
        legend_y0 = top_padding + legend_padding + max_radius
        legend_y1 = height - legend_padding

        n_circles = 6

        for i in range(n_circles):
            t = 1.0 - (i / (n_circles - 1))

            radius = min_radius + t * (max_radius - min_radius)
            if reverse_scores:
                score = score_max - t * (score_max - score_min)
            else:
                score = score_min + t * (score_max - score_min)

            y = legend_y0 + i * ((legend_y1 - legend_y0) / (n_circles - 1))

            rgba = cmap(t)
            color = tuple(int(255 * c) for c in rgba[:3])

            draw_out.ellipse(
                (legend_x - radius, y - radius, legend_x + radius, y + radius),
                fill=color,
                outline="black",
            )

            if add_labels:
                draw_out.text(
                    (legend_x + max_radius + 10, y - 6),
                    f"{score:.2f}",
                    fill="black",
                )

    return out_img
