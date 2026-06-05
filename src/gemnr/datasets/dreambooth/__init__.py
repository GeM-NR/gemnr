from pathlib import Path
import yaml


def get_input_folder(scene_folder):
    return scene_folder


def get_scenes(subset_type="validation"):
    file_path = Path(__file__).parent / f"{subset_type}.yaml"
    with open(file_path) as f:
        scenes = yaml.safe_load(f)
    return scenes


def idx_to_img_name(idx, scene_name):
    img_name = f"{idx:02}.jpg"
    return img_name
