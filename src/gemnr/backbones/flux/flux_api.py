import os
import torch
from huggingface_hub import login
from torchvision.transforms.functional import to_pil_image
from diffusers.utils.torch_utils import randn_tensor

from gemnr.core.utils.basic_processing import tensor_to_pil
from gemnr.backbones import ImageEditor

from .pipeline_flux_kontext import FluxKontextPipeline
from .pipeline_flux2_klein import Flux2KleinPipeline


class FLUX(ImageEditor):
    def __init__(
        self,
        version: str,
        seed: int,
        token: str | None = None,
        batch_size: int = 1,
        dtype=torch.bfloat16,
        device: str | torch.device = "cuda",
    ):
        assert version in ["flux1_kontext", "flux2_klein", "flux2_klein_4B"]

        self.version = version
        self.token = token
        self.seed = seed
        self.dtype = dtype
        self.H = 1024
        self.W = 1024
        self.batch_size = batch_size
        self.device = device

        self.initialized = False  # lazy initialization
        self.H_inp = None
        self.W_inp = None

    def initialize(self, W_inp, H_inp):
        (self.W_inp, self.H_inp) = (W_inp, H_inp)

        login(token=self.token or os.environ["HF_TOKEN"])
        if self.version == "flux1_kontext":
            self.pipe = FluxKontextPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Kontext-dev",
                torch_dtype=self.dtype,
            )
            self.num_inference_steps = 28
            self.laten_num_channels = 16
            self.latent_H = 128
            self.latent_W = 128
            self.guidance_scale = 2.5
        elif self.version.startswith("flux2_klein"):
            n_params = 9 if self.version == "flux2_klein" else 4
            self.pipe = Flux2KleinPipeline.from_pretrained(
                f"black-forest-labs/FLUX.2-klein-{n_params}B",
                torch_dtype=self.dtype,
            )
            self.num_inference_steps = 4
            self.latent_num_channels = 16 * 8
            vae_scale_factor = 8
            (self.latent_H, self.latent_W) = (
                int(H_inp) // (vae_scale_factor * 2),
                int(W_inp) // (vae_scale_factor * 2),
            )
            self.guidance_scale = 0.0
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

        if not self.initialized:
            (W_inp, H_inp) = inputs[0].size
            self.initialize(W_inp, H_inp)

        self.set_seed(self.seed)

        img_edited_tensor = self.pipe(
            image=inputs,
            prompt=prompt,
            guidance_scale=2.5,
            num_inference_steps=self.num_inference_steps,
            generator=self.generator,  # <-- use seeded generator here
            output_type="pt",
            num_images_per_prompt=len(inputs),
        ).images[0]
        img_edited_tensor = img_edited_tensor.float().clamp(0, 1)
        img_edited_pil = to_pil_image(img_edited_tensor).resize(inputs[0].size)
        return img_edited_tensor, img_edited_pil

    def run_pipe_partial(self, inputs, prompt, latents):
        self.set_seed(self.seed)
        z_T = torch.nn.Parameter(latents, requires_grad=False)

        if self.version == "flux1_kontext":
            latents = self.pipe._pack_latents(
                z_T,
                self.batch_size,
                self.latent_num_channels,
                self.latent_H,
                self.latent_W,
            )
        elif self.version.startswith("flux2_klein"):
            latents = z_T

        latent = self.pipe(
            image=inputs,
            prompt=prompt,
            guidance_scale=2.5,
            num_inference_steps=self.num_inference_steps,
            generator=self.generator,  # <-- use seeded generator here
            latents=latents,
            output_type="latent",
        ).images

        if self.version == "flux1_kontext":
            z_0 = self.pipe._unpack_latents(
                latent,
                self.H,
                self.W,
                self.pipe.vae_scale_factor,
            )
        elif self.version[:11] == "flux2_klein":
            z_0 = latent
        # print("z_0 shape: ", z_0.shape)
        # z_0 = (z_0 - z_T).clone().detach() + z_T

        if self.version == "flux1_kontext":
            latent = (
                z_0 / self.pipe.vae.config.scaling_factor
                + self.pipe.vae.config.shift_factor
            )
        elif self.version[:11] == "flux2_klein":
            latent = self.pipe._unpatchify_latents(z_0)

        img_edited_tensor = self.pipe.image_processor.postprocess(
            self.pipe.vae.decode(latent, return_dict=False)[0],
            output_type="pt",
        )[0]
        img_edited_pil = tensor_to_pil(img_edited_tensor)
        img_edited_pil = img_edited_pil.resize(self.W_input, self.H_input)
        # img2_edited_pil.save(
        #     os.path.join(
        #         output_folder, f"edited_{query_id}_i_opt_{i_opt}.png"
        #     )
        # )
        return img_edited_tensor, img_edited_pil

    def create_latent(self):
        # Create latent
        shape = (
            self.batch_size,
            self.latent_num_channels,
            self.latent_H,
            self.latent_W,
        )
        latents = randn_tensor(
            shape,
            generator=self.generator_input_latent,
            device=self.device,
            dtype=self.dtype,
        )
        # latents_init = latents.clone().detach()
        return latents
