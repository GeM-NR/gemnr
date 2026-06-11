from dataclasses import dataclass

import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image

from .utils.roma import RoMaEstimator
from .utils.depth_anything import DA3Estimator
from .utils.geometry import relpose_from_absolute
from .utils.basic_processing import (
    identify_unedited_regions,
    tensor_to_pil,
    erode_mask,
)


class GemWarper:
    @dataclass(frozen=True)
    class Cfg:
        force_unedited_regions: bool = False
        erode_mask_strength: float = 0.75

        refine_poses: bool = True
        relpose_max_epipolar_error: float = 1.0
        relpose_certainty_threshold: float = 0.25
        relpose_min_inliers_for_success: float = 0.75

        render_depth_conf_thr: float = 0
        render_depth_conf_thr_percentile: float = 0
        render_depth_ensure_thresh_percentile: float = 90

        raster_min_size_percent: float = 10
        raster_min_size2_percent: float = 0.01
        raster_erode_radius_percent: float = 0.01

    def __init__(
        self,
        H,
        W,
        device="cuda",
        cfg: Cfg | None = None,
        da3_estimator: DA3Estimator | None = None,
        roma_estimator: RoMaEstimator | None = None,
        **kwargs,
    ):
        self.H = H
        self.W = W
        self.device = device

        if cfg is None:
            cfg = GemWarper.Cfg(**kwargs)

        self.cfg = cfg

        W14 = self.W // 14 * 14
        H14 = self.H // 14 * 14

        # Setup DA3
        if da3_estimator:
            self.DA3 = da3_estimator
        else:
            self.DA3 = DA3Estimator(
                weights="DA3NESTED-GIANT-LARGE-1.1",
                width=W14,
                height=H14,
                device=self.device,
            )

        # Setup RoMa
        if roma_estimator:
            self.roma = roma_estimator
        else:
            self.roma = RoMaEstimator(
                H=H14,
                W=W14,
                device=self.device,
            )

    def warp_using_DA3(
        self,
        img1_pil,
        edited1_pil,
        img2_pil,
        mask1_pil=None,
        mask2_pil=None,
        save_intermediate_dirpath=None,
    ):
        depth_conf_thr = self.cfg.render_depth_conf_thr
        depth_conf_thr_percentile = self.cfg.render_depth_conf_thr_percentile
        depth_ensure_thresh_percentile = (
            self.cfg.render_depth_ensure_thresh_percentile
        )
        raster_min_size_percent = self.cfg.raster_min_size_percent
        raster_min_size2_percent = self.cfg.raster_min_size2_percent
        raster_erode_radius_percent = self.cfg.raster_erode_radius_percent

        refine_poses = self.cfg.refine_poses
        relpose_max_epipolar_error = self.cfg.relpose_max_epipolar_error
        relpose_certainty_threshold = self.cfg.relpose_certainty_threshold
        relpose_min_inliers_for_success = (
            self.cfg.relpose_min_inliers_for_success
        )

        if mask1_pil:
            black_bg = Image.new("RGB", img1_pil.size, (0, 0, 0))
            object1_pil = Image.composite(img1_pil, black_bg, mask1_pil)
            edited_object1_pil = Image.composite(
                edited1_pil, black_bg, mask1_pil
            )
            object2_pil = Image.composite(img2_pil, black_bg, mask2_pil)
            mask1_np = np.array(mask1_pil, dtype=np.bool)
            relpose_max_epipolar_error = 1.0
            relpose_certainty_threshold = 0.25
            relpose_min_inliers_for_success = 0.5
        else:
            object1_pil = img1_pil.copy()
            edited_object1_pil = edited1_pil.copy()
            object2_pil = img2_pil.copy()

        # Extract cameras
        preds = self.DA3.run_DepthAnything3([object1_pil, object2_pil])

        # Refine cameras
        if refine_poses:
            object1_tensor = ToTensor()(object1_pil).to(self.device)
            object2_tensor = ToTensor()(object2_pil).to(self.device)
            preds.extrinsics = self.refine_extrinsics_using_roma(
                object1_tensor.unsqueeze(0),
                object2_tensor.unsqueeze(0),
                extrinsics=preds.extrinsics,
                intrinsics=preds.intrinsics,
                max_epipolar_error=relpose_max_epipolar_error,
                certainty_threshold=relpose_certainty_threshold,
                min_inliers_for_success=relpose_min_inliers_for_success,
            )

        # Extract all depths while forcing cameras
        intrinsics, extrinsics = self.DA3.propagate_cameras(
            preds, pose_idxs_from=[0, 1], pose_idxs_to=[0, 0, 1]
        )
        preds = self.DA3.run_DepthAnything3(
            img_pil_list=[object1_pil, edited_object1_pil, object2_pil],
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )

        if save_intermediate_dirpath is not None:
            from pathlib import Path
            from copy import deepcopy

            def preds_subset(preds, idxs):
                preds_new = deepcopy(preds)
                preds_new.processed_images = preds.processed_images[idxs]
                preds_new.depth = preds.depth[idxs]
                preds_new.conf = preds.conf[idxs]
                preds_new.intrinsics = preds.intrinsics[idxs]
                preds_new.extrinsics = preds.extrinsics[idxs]
                return preds_new

            self.DA3.output_folder = Path(save_intermediate_dirpath)
            self.DA3.export_to_imgs(preds.processed_images)
            self.DA3.export_to_1Dimgs(preds.depth)
            preds_unedited = preds_subset(preds, [0, 2])
            self.DA3.export_to_glb(preds_unedited, "pcl_unedited")
            preds_edited = preds_subset(preds, [1])
            self.DA3.export_to_glb(preds_edited, "pcl_edited")
            print(f"Saved intermediate 3D data to {self.DA3.output_folder}")

        # Render edited query image using available depths
        if self.cfg.force_unedited_regions:
            img1_edited_mask = identify_unedited_regions(img1_pil, edited1_pil)
            render_masks = [None, img1_edited_mask, None]
        else:
            if mask1_pil:
                mask1_np = np.array(mask1_pil, dtype=np.bool)
                render_masks = [None, mask1_np, None]
            else:
                render_masks = None

        renderings_DA3 = self.DA3.reproject_and_render(
            preds=preds,
            conf_thr=depth_conf_thr,
            conf_thresh_percentile=depth_conf_thr_percentile,
            ensure_thresh_percentile=depth_ensure_thresh_percentile,
            idx_from=1,
            idx_to=2,
            masks=render_masks,
            min_size_percent=raster_min_size_percent,
            min_size2_percent=raster_min_size2_percent,
            erode_radius_percent=raster_erode_radius_percent,
        )

        warped2 = renderings_DA3[0]
        warped2_pil = tensor_to_pil(warped2, False)

        img2_tensor = ToTensor()(img2_pil)
        if self.cfg.force_unedited_regions:
            warped2_mask = (warped2**2).sum(0) < 1e-10
            warped2_mask = erode_mask(
                warped2_mask, strength=self.cfg.erode_mask_strength
            )
            masked2_tensor = img2_tensor * warped2_mask
            masked2_pil = tensor_to_pil(masked2_tensor)
        elif mask2_pil:
            mask2_np = np.array(mask2_pil, dtype=np.bool)
            masked2_tensor = img2_tensor * torch.tensor(
                np.logical_not(mask2_np)
            )
            masked2_pil = tensor_to_pil(masked2_tensor)
        else:
            masked2_pil = None

        if save_intermediate_dirpath is not None:
            warped2_pil.save(
                Path(save_intermediate_dirpath) / "warped.png"
            )
            if masked2_pil:
                masked2_pil.save(
                    Path(save_intermediate_dirpath) / "masked.png"
                )

        return warped2_pil, masked2_pil

    def refine_extrinsics_using_roma(
        self,
        img1_tensor,
        img2_tensor,
        extrinsics,
        intrinsics,
        max_epipolar_error=None,
        certainty_threshold=None,
        min_inliers_for_success=None,
    ):
        if max_epipolar_error is None:
            max_epipolar_error = self.cfg.relpose_max_epipolar_error
        if certainty_threshold is None:
            certainty_threshold = self.cfg.relpose_certainty_threshold
        if min_inliers_for_success is None:
            min_inliers_for_success = self.cfg.relpose_min_inliers_for_success

        (warp, certainty_map) = self.roma.estimate_warp(
            img1_tensor,
            img2_tensor,
        )
        relpose_initial = relpose_from_absolute(extrinsics[0], extrinsics[1])
        relpose, inliers, _ = self.roma.estimate_relpose_from_warp(
            warp=warp,
            certainty_map=certainty_map,
            K1=intrinsics[0],
            K2=intrinsics[1],
            initial_pose=relpose_initial,
            max_epipolar_error=max_epipolar_error,
            certainty_threshold=certainty_threshold,
        )
        if sum(inliers) / len(inliers) > min_inliers_for_success:
            extrinsics[1] = extrinsics[0] @ np.vstack(
                [relpose, np.array([0, 0, 0, 1])]
            )
            print("Refinement successful.")
        return extrinsics

    def verify(self, **kwargs):
        return True
