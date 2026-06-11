from PIL import Image
import torch
from torchvision.transforms import ToTensor

from gemnr.backbones import ImageEditor
from gemnr.backbones.flux import FLUX
from wav3d.core.wav3d import RoMaEstimator, DA3Estimator
from gemnr.core import GemWarper

VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3


class GemNR:
    def __init__(
        self,
        resolution: int,
        anchor_editor: ImageEditor | None = None,
        seed: int = 0,
        token: str | None = None,
        device: torch.device | None = None,
        lazy_init: bool = True,
    ):
        self.H, self.W = resolution, resolution
        self.resolution = resolution
        self.anchor_editor = anchor_editor
        self.seed = seed
        self.token = token
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.lazy_init = lazy_init
        self.da3 = None
        self.roma = None
        self._backbone_editor = None
        if not self.lazy_init:
            self.initialize_models()

    @property
    def backbone_editor(self):
        if not self._backbone_editor:
            self.initialize_models()
        return self._backbone_editor
    
    def initialize_models(self):
        W14 = self.W // 14 * 14
        H14 = self.H // 14 * 14
        self.da3 = DA3Estimator(
            weights="DA3NESTED-GIANT-LARGE-1.1",
            width=W14,
            height=H14,
            device=self.device,
        )
        self.roma = RoMaEstimator(
            H=H14,
            W=W14,
            device=self.device,
        )
        if VRAM_GB < 70:
            print(
                f"Detected GPU with {VRAM_GB:.1f} GB VRAM. Using smaller FLUX model for editing."
            )
            flux_version = "flux2_klein_4B"
        else:
            flux_version = "flux2_klein"
        self._backbone_editor = FLUX(
            version=flux_version,
            seed=self.seed,
            device=self.device,
            token=self.token,
        )

    def set_seed(self, seed):
        self.seed = seed
        if self._backbone_editor:
            self._backbone_editor.set_seed(seed=seed)

    def crop_resize(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        size = min(width, height)
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size
        return image.crop((left, top, right, bottom)).resize((self.W, self.H))

    def edit(
        self,
        ims_pil: Image.Image | list[Image.Image],
        edit_text_prompt: str,
        anchor_idx: int = 0,
        anchor_cond_pil: Image.Image | None = None,
        edited_anchor_pil: Image.Image | None = None,
        save_intermediate_dirpath: str | None = None,
    ) -> list[Image.Image]:
        if isinstance(ims_pil, Image.Image):
            im_pil_list = [ims_pil]
        else:
            im_pil_list = ims_pil

        if any(im.size != (self.W, self.H) for im in im_pil_list):
            print(
                "Resizing and cropping input images to fit the model's expected resolution."
            )
            im_pil_list = [self.crop_resize(im_pil) for im_pil in im_pil_list]
        
        if anchor_cond_pil and anchor_cond_pil.size != (self.W, self.H):
            print(
                "Resizing and cropping anchor conditioning image to fit the model's expected resolution."
            )
            anchor_cond_pil = self.crop_resize(anchor_cond_pil)

        n_imgs = len(im_pil_list)
        assert (
            anchor_idx < n_imgs
        ), f"Anchor index ({anchor_idx}) is not in the valid range [0, {n_imgs-1}]."
        order = list(range(n_imgs))
        if anchor_idx != 0:
            order[0] = anchor_idx
            order[anchor_idx] = 0

        if self.anchor_editor:
            assert (
                edited_anchor_pil is None
            ), "Provide either an editor to edit anchor image or an already edited anchor image, not both."
            anchor_editor = self.anchor_editor
        else:
            anchor_editor = self.backbone_editor
        
        if not edited_anchor_pil:
            anchor_pil = im_pil_list[anchor_idx]
            prompt_end = "" if not anchor_cond_pil else " as shown in the second image." 
            _, edited_anchor_pil = anchor_editor(
                anchor_pil if not anchor_cond_pil else [anchor_pil, anchor_cond_pil],
                prompt=edit_text_prompt + prompt_end,
            )

        if n_imgs == 1:
            return [edited_anchor_pil]

        pipeline = GemNRSetPipeline(
            W=self.W,
            H=self.H,
            editor_model=self.backbone_editor,
            da3_estimator=self.da3,
            roma_estimator=self.roma,
            device=self.device,
        )

        _, edited_pil_list = pipeline.edit(
            [im_pil_list[i] for i in order],
            edited1_pil=edited_anchor_pil,
            prompt=edit_text_prompt,
            save_intermediate_dirpath=save_intermediate_dirpath,
        )

        return [edited_pil_list[i] for i in order]


class GemNRSetPipeline:
    def __init__(
        self,
        editor_model,
        **kwargs,
    ):
        self.editor_model = editor_model
        self.gem_warper = GemWarper(**kwargs)
        self.cfg = self.gem_warper.cfg
        self.prompt_extra = "The suggested appearance is in the second image. Stick to this change, but refine it to keep consistency with respect to the first image."

    def edit(
        self, img_pil_list, prompt, edited1_pil, save_intermediate_dirpath=None
    ):
        """Edit each target view conditioned on the anchor image (first in list) and its edit."""
        img1_pil = img_pil_list[0]
        target_views = img_pil_list[1:]

        full_prompt = f"{prompt}. {self.prompt_extra}"
        prompts = [prompt] + [full_prompt] * (len(target_views) - 1)
        inputs = []

        for i, img2_pil in enumerate(target_views):
            warped2_pil, masked2_pil = self.gem_warper.warp_using_DA3(
                img1_pil,
                edited1_pil,
                img2_pil,
                save_intermediate_dirpath=save_intermediate_dirpath,
            )

            if self.cfg.force_unedited_regions:
                inputs.append([img2_pil, masked2_pil, warped2_pil])
            else:
                inputs.append([img2_pil, warped2_pil])

        # Anchor output: use pre-computed edit directly
        img_tensor_list = [ToTensor()(edited1_pil)]
        edited_pil_list = [edited1_pil]

        for inp, p in zip(inputs, prompts):
            img_tensor, img_pil = self.editor_model(inputs=inp, prompt=p)
            img_tensor_list.append(img_tensor)
            edited_pil_list.append(img_pil)

        return img_tensor_list, edited_pil_list
