from pathlib import Path
import yaml

_DATA_ROOT = Path(
    "/mimer/NOBACKUP/groups/snic2022-6-266/ylochman/3d_consistent_editing/data/SPIn-NeRF/preprocessed_scenes"
)

_prefix_cache = {}


def get_data_root():
    return _DATA_ROOT


def _get_prefix(scene_name):
    if scene_name not in _prefix_cache:
        images_dir = get_data_root() / scene_name
        for f in sorted(images_dir.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png") and "_" in f.stem:
                _prefix_cache[scene_name] = f.stem.split("_")[0]
                break
    return _prefix_cache[scene_name]


def get_input_folder(scene_folder):
    return scene_folder


def get_pose_folder(input_folder):
    scene = Path(input_folder).name
    pycolmap_sparse_rec_folder = Path("./outputs/pycolmap/spinnerf") / scene
    return pycolmap_sparse_rec_folder


def get_scenes(subset_type="validation"):
    file_path = Path(__file__).parent / f"{subset_type}.yaml"
    with open(file_path) as f:
        scenes = yaml.safe_load(f)
    return scenes


def idx_to_img_name(idx, scene_name):
    prefix = _get_prefix(scene_name)
    return f"{prefix}_{idx:06d}.png"
