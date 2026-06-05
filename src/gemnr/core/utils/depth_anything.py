import torch
import torchvision.transforms.functional as tvF
import numpy as np
from PIL import Image
import os

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.pose_align import _to44

from depth_anything_3.utils.visualize import visualize_depth
from depth_anything_3.utils.export import export_to_glb

from .roma import get_pixel_grid
from gemnr.core.utils.geometry import (
    backproject,
    normalize_intrinsics,
    unnormalize_intrinsics,
)
from gemnr.core.utils.rasterization import (
    backproject_for_rasterization,
    rasterize_points_hard,
)


class DepthUtils_DA3:
    def __init__(
        self,
        weights: str = "DA3NESTED-GIANT-LARGE",
        width: int = 504,
        height: int = 504,
        output_folder: str | None = None,
        device: str | torch.device = "cpu",
        **inference_kwargs,
    ):
        self.W_model = width
        self.H_model = height
        self.output_folder = output_folder
        self.device = device
        self.model = DepthAnything3.from_pretrained(
            f"depth-anything/{weights}"
        )
        self.model = self.model.to(device=device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.inference_kwargs = inference_kwargs
        self.W_inp = None
        self.H_inp = None

    def run_DepthAnything3(
        self,
        img_pil_list,
        resize=True,
        intrinsics=None,
        extrinsics=None,
    ):
        self.W_inp, self.H_inp = img_pil_list[0].size
        if intrinsics is not None:
            intrinsics = unnormalize_intrinsics(
                intrinsics, W=self.W_inp, H=self.H_inp
            )
        if extrinsics is not None:
            extrinsics = _to44(extrinsics)
        prediction = self.model.inference(
            image=img_pil_list,
            process_res=self.W_model,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            **self.inference_kwargs,
        )
        # prediction.processed_images : [N, H, W, 3] uint8   array
        # prediction.depth            : [N, H, W]    float32 array
        # prediction.conf             : [N, H, W]    float32 array
        # prediction.extrinsics       : [N, 3, 4]    float32 array
        # # opencv w2c or colmap format
        # prediction.intrinsics       : [N, 3, 3]    float32 array
        prediction.intrinsics = normalize_intrinsics(
            prediction.intrinsics, W=self.W_model, H=self.H_model
        )
        if resize:
            depth_tensor = torch.tensor(prediction.depth)
            prediction.depth = tvF.resize(
                depth_tensor, (self.H_inp, self.W_inp)
            ).numpy()

            if prediction.conf is not None:
                conf_tensor = torch.tensor(prediction.conf)
                prediction.conf = tvF.resize(
                    conf_tensor, (self.H_inp, self.W_inp)
                ).numpy()

            prediction.processed_images = np.stack(
                [np.array(img) for img in img_pil_list]
            )
        return prediction

    def export_to_1Dimgs(self, depths, output_name="depth"):
        if isinstance(output_name, list):
            for d, name in zip(depths, output_name):
                self.export_to_1Dimgs(d, name)
            return
        for kk in range(len(depths)):
            depthmap = visualize_depth(depths[kk])
            depthmap_pil = Image.fromarray(depthmap)
            depthmap_pil.save(
                os.path.join(self.output_folder, f"{output_name}_{kk}.png")
            )

    def export_to_imgs(self, images, output_name="img"):
        for kk, img in enumerate(images):
            Image.fromarray(img).save(
                os.path.join(self.output_folder, f"{output_name}_{kk}.png")
            )

    def export_to_glb(
        self, preds, output_name="pcl", select_idx=None, **kwargs
    ):
        if isinstance(output_name, list):
            for idxs, name in zip(select_idx, output_name):
                self.export_to_glb(preds, name, select_idx=idxs, **kwargs)
            return
        output_folder = self.output_folder
        glb_export_kwargs = {
            "export_dir": f"{output_folder}/glb",
            "conf_thresh": 0,  # 1.05,
            "show_cameras": True,
            "export_depth_vis": False,
            "conf_thresh_percentile": 0,
            "num_max_points": 600_000,
            "output_name": output_name,
            **kwargs,
        }
        preds.intrinsics = unnormalize_intrinsics(
            preds.intrinsics, W=self.W_model, H=self.H_model
        )
        if select_idx is not None:
            from depth_anything_3.specs import Prediction

            preds_modified = Prediction(
                depth=preds.depth[select_idx],
                intrinsics=preds.intrinsics[select_idx],
                extrinsics=preds.extrinsics[select_idx],
                processed_images=preds.processed_images[select_idx],
                conf=preds.conf[select_idx],
                is_metric=preds.is_metric,
            )
            export_to_glb(preds_modified, **glb_export_kwargs)
        else:
            export_to_glb(preds, **glb_export_kwargs)

        preds.intrinsics = normalize_intrinsics(
            preds.intrinsics, W=self.W_model, H=self.H_model
        )

    def get_conf_thresh(
        self,
        preds,
        conf_thresh: float,
        conf_thresh_percentile: float | None,
        ensure_thresh_percentile: float | None,
        masks=None,
    ):
        if conf_thresh_percentile is not None:
            if masks is not None:
                conf_pixels = []
                for conf, mask in zip(preds.conf, masks):
                    if mask is not None:
                        conf_pixels.extend(conf[mask])
            else:
                conf_pixels = preds.conf
            conf_pixels = np.array(conf_pixels)
            lower = np.percentile(conf_pixels, conf_thresh_percentile)
            assert ensure_thresh_percentile is not None
            upper = np.percentile(conf_pixels, ensure_thresh_percentile)
            conf_thresh = min(max(conf_thresh, lower), upper)
        return conf_thresh

    def reproject_and_render(
        self,
        preds,
        idx_from: int,
        idx_to: int,
        conf_thr: float = 0,
        conf_thresh_percentile: float | None = None,
        ensure_thresh_percentile: float | None = None,
        masks=None,
        depth_min: float = 1e-8,
        depth_max: float = 1e3,
        **kwargs,
    ):
        conf_thresh = self.get_conf_thresh(
            preds=preds,
            conf_thresh=conf_thr,
            conf_thresh_percentile=conf_thresh_percentile,
            ensure_thresh_percentile=ensure_thresh_percentile,
            masks=masks,
        )

        if isinstance(idx_from, (list, np.ndarray)):
            return [
                self.reproject_and_render(
                    preds=preds,
                    idx_from=i,
                    idx_to=idx_to,
                    merge_from=False,
                    conf_thr=conf_thresh,
                    masks=masks,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    **kwargs,
                )
                for i in idx_from
            ]

        W, H = self.W_inp, self.H_inp

        depth = preds.depth[idx_from]  # (H,W)
        depth_mask = None if masks is None else masks[idx_from]
        K = unnormalize_intrinsics(preds.intrinsics[idx_from], W=W, H=H)
        ext_w2c = preds.extrinsics[idx_from]
        conf = preds.conf[idx_from]
        img = preds.processed_images[idx_from]

        K_to = unnormalize_intrinsics(preds.intrinsics[idx_to], W=W, H=H)
        ext_w2c_to = preds.extrinsics[idx_to]

        pix = get_pixel_grid(B=1, W=W, H=H, overload_device="cpu")[0]

        Xc2, colors, confidences, radii = backproject_for_rasterization(
            img=img,
            pix=pix,
            K=K,
            ext_w2c=ext_w2c,
            ext_w2c_to=ext_w2c_to,
            conf=conf,
            depth=depth,
            conf_thresh=conf_thresh,
            depth_mask=depth_mask,
            depth_min=depth_min,
            depth_max=depth_max,
        )

        if len(Xc2) == 0:
            return (
                torch.zeros((3, W, H)),
                torch.zeros((W, H)),
                torch.zeros((W, H)),
            )

        (img, confindence_map), depth_map, mask = rasterize_points_hard(
            X=Xc2,
            K=K_to,
            features=[colors, confidences],
            W=W,
            H=H,
            radii=radii,
            **kwargs,
        )
        img = img.to("cpu")
        depth_map = depth_map.to("cpu")
        confindence_map = confindence_map[0].to("cpu")
        mask = mask.to("cpu")
        return img, depth_map, confindence_map, mask

    def propagate_cameras(
        self, preds, pose_idxs_from, pose_idxs_to, use_single_intrinsics=True
    ):
        n_imgs = len(pose_idxs_to)
        intrinsics_mean = preds.intrinsics.mean(0)
        intrinsics = np.zeros((n_imgs, 3, 3))
        extrinsics = np.zeros((n_imgs, 3, 4))
        for jj, pose_idx in enumerate(pose_idxs_from):
            ii = np.where(np.array(pose_idxs_to) == pose_idx)[0]
            extrinsics[ii] = preds.extrinsics[jj]
            if use_single_intrinsics:
                intrinsics[ii] = intrinsics_mean
            else:
                intrinsics[ii] = preds.intrinsics[jj]
        return intrinsics, extrinsics
