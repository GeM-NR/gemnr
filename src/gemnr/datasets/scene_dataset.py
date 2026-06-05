from pathlib import Path

import numpy as np
from diffusers.utils import load_image

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
}


class SceneDataset:
    def __init__(
        self,
        scene_name,
        input_folder,
        idx_to_img_name_fn,
        device='cpu',
        edit_folder=None,
    ):
        self.input_folder = Path(input_folder)
        self.scene_name = scene_name
        self.device = device
        self.idx_to_img_name = idx_to_img_name_fn

        if edit_folder is not None:
            edit_folder = Path(edit_folder)
        self.edit_folder = edit_folder
        self.mask_folder = self.input_folder / "segmentations" / "mask"
        self.mask_folder_sam = self.input_folder / "segmentations_sam"
        img_names_list = [
            f.name
            for f in self.input_folder.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        self.img_names = np.array(sorted(img_names_list))

    def get_random_set_img_names(self, n_imgs, seed, img_names=None):
        rng = np.random.default_rng(seed)
        pool = np.array(img_names) if img_names is not None else self.img_names
        return list(rng.choice(pool, size=n_imgs, replace=False))

    def load(self, idx, with_mask=False):
        img_name = self.idx_to_img_name(idx, self.scene_name)
        print("image load path: ", str(self.input_folder / img_name))
        img_pil = load_image(str(self.input_folder / img_name))
        if not with_mask:
            return img_pil

        assert self.mask_folder.exists()
        mask_idx = np.where(self.img_names == img_name)[0][0]
        mask_path_candidates = [
            self.mask_folder_sam / img_name,
            self.mask_folder / f"{mask_idx}.png",
            self.mask_folder / f"{idx}.png",
            self.mask_folder / img_name,
        ]
        mask_pil = None
        for mask_path in mask_path_candidates:
            if mask_path.exists():
                mask_pil = load_image(str(mask_path))
                break
        if not mask_pil:
            raise FileNotFoundError(
                f"Mask not found for image {img_name} in {self.mask_folder}"
            )
        mask_pil = mask_pil.convert("L")
        return img_pil, mask_pil

    def load_from_name(self, img_name, with_mask=False):
        print("image load path: ", str(self.input_folder / img_name))
        img_pil = load_image(str(self.input_folder / img_name))
        mask_pil = None
        if not with_mask:
            return img_pil
            
        assert self.mask_folder.exists()
        mask_idx = np.where(self.img_names == img_name)[0][0]
        mask_path_candidates = [
            self.mask_folder_sam / img_name,
            self.mask_folder / f"{mask_idx}.png",
            self.mask_folder / img_name,
        ]
        mask_pil = None
        for mask_path in mask_path_candidates:
            if mask_path.exists():
                mask_pil = load_image(str(mask_path))
                break
        
        if not mask_pil:
            raise FileNotFoundError(
                f"Mask not found for image {img_name} in {self.mask_folder}"
            )
        mask_pil = mask_pil.convert("L")
        return img_pil, mask_pil

    def load_with_edited(self, idx: int):
        img_name = self.idx_to_img_name(idx, self.scene_name)
        img_pil = load_image(str(self.input_folder / img_name))
        if (self.edit_folder / img_name).exists():
            edited_pil = load_image(str(self.edit_folder / img_name))
        else:
            edited_pil = None
        return img_pil, edited_pil
    
    def load_edited(self, idx: int, edited_path=None):
        if not edited_path:
            img_name = self.idx_to_img_name(idx, self.scene_name)
            edited_path = self.edit_folder / img_name
        assert Path(edited_path).exists()
        edited_pil = load_image(str(edited_path))
        return edited_pil

    def load_from_name_with_edited(self, img_name: str):
        assert (self.edit_folder / img_name).exists()
        img_pil = load_image(str(self.input_folder / img_name))
        edited_pil = load_image(str(self.edit_folder / img_name))
        return img_pil, edited_pil

    def setup(self, **kwargs):
        pass
