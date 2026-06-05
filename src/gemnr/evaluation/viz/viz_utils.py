import cv2
import numpy as np
from PIL import Image

def add_title_to_img(
    img_pil,
    title,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=2.2,
    thickness=4,
):
    title_height = int(np.ceil(55 * font_scale))
    W, H = img_pil.size
    canvas = Image.new("RGB", (W, H + title_height), "white")
    canvas.paste(img_pil, (0, title_height))
    img_cv = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

    (text_w, text_h), _ = cv2.getTextSize(title, font, font_scale, thickness)
    text_x = (W - text_w) // 2
    text_y = (title_height + text_h) // 2
    cv2.putText(
        img_cv,
        title,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


def concat_images_grid(img_list, n_cols):
    """
    Concatenate a list of PIL images into a grid with n_cols columns.

    Parameters:
        img_list (list): List of PIL.Image objects
        n_cols (int): Number of columns in the grid

    Returns:
        PIL.Image: Concatenated grid image
    """
    if not img_list:
        return None

    N = len(img_list)
    rows = N // n_cols + (1 if N % n_cols else 0)

    # Assuming all images are the same size
    img_width, img_height = img_list[0].size

    # Create a blank canvas for the grid
    grid_img = Image.new(
        "RGB", (n_cols * img_width, rows * img_height), color=(255, 255, 255)
    )

    for idx, img in enumerate(img_list):
        row = idx // n_cols
        col = idx % n_cols
        grid_img.paste(img, (col * img_width, row * img_height))

    return grid_img
