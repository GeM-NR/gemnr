from PIL import Image
import torch
from torchvision.transforms import ToTensor

from gemnr.backbones import ImageEditor
from gemnr.backbones.flux import FLUX
from gemnr.core import GemWarper


VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3


class GemNR:
    def __init__(
        self,
        resolution: int,
        anchor_editor: ImageEditor | None = None,
        seed: int = 0,
        device: torch.device | None = None,
    ):
        self.H, self.W = resolution, resolution
        self.anchor_editor = anchor_editor
        self.seed = seed
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.lazy_init_done = False

    @property
    def backbone_editor(self):
        if self.lazy_init_done:
            return self._backbone_editor
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
        )
        self.lazy_init_done = True
        return self._backbone_editor

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
        im_pil_list: list[Image.Image],
        edit_text_prompt: str,
        anchor_idx: int = 0,
        edited_anchor_pil: Image.Image | None = None,
    ) -> list[Image.Image]:

        n_imgs = len(im_pil_list)
        assert n_imgs >= 2, "Number of input images must be at least 2."

        if any(im.size != (self.W, self.H) for im in im_pil_list):
            print(
                "Resizing and cropping input images to fit the model's expected resolution."
            )
            im_pil_list = [self.crop_resize(im_pil) for im_pil in im_pil_list]

        order = list(range(n_imgs))
        if anchor_idx != 0:
            order[0] = anchor_idx
            order[anchor_idx] = 0

        backbone_editor = self.backbone_editor
        if self.anchor_editor:
            assert (
                edited_anchor_pil is None
            ), "Provide either an editor to edit anchor image or an already edited anchor image, not both."
            _, edited_anchor_pil = self.anchor_editor(
                im_pil_list[anchor_idx],
                prompt=edit_text_prompt,
            )
        else:
            _, edited_anchor_pil = backbone_editor(
                im_pil_list[anchor_idx],
                prompt=edit_text_prompt,
            )

        pipeline = GemNRSetPipeline(
            W=self.W,
            H=self.H,
            editor_model=backbone_editor,
            device=self.device,
        )

        _, edited_pil_list = pipeline.edit(
            [im_pil_list[i] for i in order],
            edited1_pil=edited_anchor_pil,
            prompt=edit_text_prompt,
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
