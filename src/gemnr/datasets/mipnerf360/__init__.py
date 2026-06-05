from pathlib import Path
import yaml


def get_input_folder(scene_folder):
    return Path(scene_folder) / "images"


def get_scenes(subset_type="full_validation_set"):
    file_path = Path(__file__).parent / f"{subset_type}.yaml"
    with open(file_path) as f:
        scenes = yaml.safe_load(f)
    return scenes


def get_pose_folder(input_folder):
    scene = Path(input_folder).parent.name
    pycolmap_sparse_rec_folder = Path("./outputs/pycolmap/mipnerf360") / scene
    return pycolmap_sparse_rec_folder


def idx_to_img_name(idx, scene_name):
    if scene_name == "bicycle":
        img_name = f"_DSC{idx}.JPG"
    elif scene_name in ["face", "bear"]:
        img_name = f"frame_{idx:05d}.jpg"
    elif scene_name in ["person", "fangzhou"]:
        img_name = f"frame_{idx:05d}.png"
    elif scene_name == "garden":
        img_name = f"DSC{idx:05d}.JPG"
    elif scene_name == "stone_horse":
        img_name = f"{idx:08d}.jpg"
    elif scene_name == "dinosaur":
        img_name = f"rotated-{idx:08d}.jpg"
    elif scene_name == "soh":
        img_name = f"{idx:03d}.png"
    elif scene_name == "stump":
        img_name = f"_DSC{idx:04d}.JPG"
    else:
        raise ValueError(f"Incorrect scene name: {scene_name}")
    return img_name
