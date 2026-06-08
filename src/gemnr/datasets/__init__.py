import yaml
from pathlib import Path
from .scene_dataset import SceneDataset


def get_scenes(subset_type="full_validation_set"):
    file_path = Path(__file__).parent / f"{subset_type}.yaml"
    with open(file_path) as f:
        scenes = yaml.safe_load(f)
    return scenes


def load_scenes(dataset, data_subset):
    if dataset is None:
        file_path = Path(__file__).parent / f"{data_subset}.yaml"
        with open(file_path) as f:
            scenes = yaml.safe_load(f)
        return scenes
    elif dataset == "mipnerf360":
        from gemnr.datasets.mipnerf360 import get_scenes

        return get_scenes(subset_type=data_subset)
    elif dataset == "dreambooth":
        from gemnr.datasets.dreambooth import get_scenes

        return get_scenes(subset_type=data_subset)
    elif dataset == "spinnerf":
        from gemnr.datasets.spinnerf import get_scenes

        return get_scenes(subset_type=data_subset)
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported.")


def get_dataset_fns(dataset_name):
    if dataset_name == "mipnerf360":
        from gemnr.datasets.mipnerf360 import (
            idx_to_img_name,
            get_input_folder,
            get_scenes,
            get_pose_folder,
        )
    elif dataset_name == "dreambooth":
        from gemnr.datasets.dreambooth import (
            idx_to_img_name,
            get_input_folder,
            get_scenes,
        )

        get_pose_folder = None
    elif dataset_name == "spinnerf":
        from gemnr.datasets.spinnerf import (
            idx_to_img_name,
            get_input_folder,
            get_scenes,
            get_pose_folder,
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")
    return idx_to_img_name, get_input_folder, get_scenes, get_pose_folder


def get_dataset_resolution(dataset):
    if dataset == "mipnerf360":
        return 512, 512
    elif dataset == "spinnerf":
        return 576, 576
    elif dataset == "dreambooth":
        return 512, 512
    else:
        raise NotImplementedError(
            f"Resolution for dataset {dataset} is not defined."
        )
