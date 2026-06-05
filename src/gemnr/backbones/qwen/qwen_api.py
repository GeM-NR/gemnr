import os
import torch
from huggingface_hub import login
from torchvision.transforms.functional import to_pil_image
from diffusers import QwenImageEditPipeline, QwenImageEditPlusPipeline

from gemnr.backbones import ImageEditor


class QWEN(ImageEditor):
    def __init__(
        self,
        version: str,
        seed: int,
        token: str | None = None,
        batch_size: int = 1,
        dtype=torch.bfloat16,
        device: str | torch.device = "cuda",
    ):
        assert version in [
            "qwen",
            "qwen_2509",
            "qwen_2511",
        ]
        self.token = token
        self.version = version
        self.seed = seed
        self.dtype = dtype
        self.batch_size = batch_size
        self.device = device

        self.initialized = False  # lazy initialization
        self.H_inp = None
        self.W_inp = None

    def initialize(self, W_inp, H_inp):
        (self.W_inp, self.H_inp) = (W_inp, H_inp)
        login(token=self.token or os.environ["HF_TOKEN"])

        if self.version == "qwen":
            self.pipe = QwenImageEditPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit",
                torch_dtype=self.dtype,
            )
            self.num_inference_steps = 50
            self.guidance_scale = 1.0
            self.true_cfg_scale = 4.0
        elif self.version == "qwen_2509":
            self.pipe = QwenImageEditPlusPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit-2509",
                torch_dtype=self.dtype,
            )
            self.num_inference_steps = 40
            self.guidance_scale = 1.0
            self.true_cfg_scale = 4.0
        elif self.version == "qwen_2511":
            self.pipe = QwenImageEditPlusPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit-2511",
                torch_dtype=self.dtype,
            )
            self.num_inference_steps = 40
            self.guidance_scale = 1.0
            self.true_cfg_scale = 4.0

        self.pipe.to(self.device)
        self.set_seed(self.seed)
        self.initialized = True

    def set_seed(self, seed=None):
        if seed is None:
            seed = self.seed
        torch.manual_seed(seed)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.generator_input_latent = torch.Generator(
            device=self.device
        ).manual_seed(seed)

    def run_pipe(self, inputs, prompt):
        if not isinstance(inputs, list):
            inputs = [inputs]
        img_pil = inputs[0]

        self.set_seed(self.seed)

        (W, H) = img_pil.size

        if not self.initialized:
            self.initialize(W, H)

        with torch.inference_mode():
            output = self.pipe(
                image=[img_pil],
                prompt=prompt,
                negative_prompt=" ",
                guidance_scale=self.guidance_scale,
                true_cfg_scale=self.true_cfg_scale,
                num_inference_steps=self.num_inference_steps,
                num_images_per_prompt=1,
                generator=self.generator,
                output_type="pt",
            )
        img_edited_tensor = output.images[0].float().clamp(0, 1)
        img_edited_pil = to_pil_image(img_edited_tensor).resize((W, H))
        return img_edited_tensor, img_edited_pil
