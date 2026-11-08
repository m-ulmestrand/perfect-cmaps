from pathlib import Path
from appdirs import user_data_dir
import json
import os
import tempfile
from typing import List, Tuple

import sys
sys.path.append(Path(__file__).parent.parent.absolute().as_posix())

import importlib_resources as resources
from perfect_cmaps import lab_control_points
from perfect_cmaps import test_images


app_name = "perfect_cmaps"
data_dir = Path(user_data_dir(app_name))
data_dir.mkdir(parents=True, exist_ok=True)
fit_cache_dir = data_dir / "fit_cache"

SOURCE_PACKAGE = "package"
SOURCE_USER = "user"


def atomic_write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}_", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=4)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def save_data(data: dict, name: str):
    new_name = name
    i = 2
    while True:
        new_json_file = data_dir / f"{new_name}.json"
        if new_json_file.exists():
            new_name = f"{name}_{i}"
            i += 1
        else:
            break
    
    json_file = (data_dir / new_name).with_suffix(".json")
    atomic_write_json(json_file, data)
    
    print("Saved colormap data as", new_name)
    return new_name


def load_data_with_source(filename: str) -> Tuple[dict, str]:
    """
    Load data from the internal package folder first.
    If not found, try the custom app data folder.

    Returns:
        Tuple[dict, str]: The colormap data and its source,
            either ``SOURCE_PACKAGE`` or ``SOURCE_USER``.
    """
    filename = Path(filename).with_suffix(".json")  # Ensure the file has a .json suffix

    # Try loading from the internal package data folder
    try:
        with resources.open_text(lab_control_points, filename.name) as f:
            return json.load(f), SOURCE_PACKAGE
    except (FileNotFoundError, ModuleNotFoundError):
        pass  # File not found internally, fallback to the custom app folder

    # Try loading from the custom app folder
    custom_path = data_dir / filename
    if custom_path.exists():
        with open(custom_path, "r") as f:
            return json.load(f), SOURCE_USER

    # If not found in either location, raise an exception
    raise FileNotFoundError(f"{filename} not found in internal or app data folders.")


def load_fit_side_cache(name: str) -> dict:
    """Load cached lightness fits for a package colormap from the user's app data folder.
    Returns an empty dict if there is no (readable) side cache."""
    side_path = fit_cache_dir / f"{Path(name).stem}.json"
    if not side_path.exists():
        return {}
    try:
        with open(side_path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def update_data(name: str, data: dict, source: str):
    """
    Persist updated colormap data, e.g. after computing a new lightness fit.

    User colormaps are rewritten in place. Package colormaps are never written
    to (the install location may be read-only); their fits are instead stored
    in a side cache in the user's app data folder.
    """
    name = Path(name).stem
    if source == SOURCE_USER:
        atomic_write_json(data_dir / f"{name}.json", data)
    elif source == SOURCE_PACKAGE:
        side = {
            "version": data.get("version"),
            "points_hash": data.get("points_hash"),
            "fits": data.get("fits", {}),
        }
        atomic_write_json(fit_cache_dir / f"{name}.json", side)
    else:
        raise ValueError(f"Unknown colormap source: {source}")


def get_test_img_path() -> Path:
    """
    Return the path to the 'test_images' folder inside the package.
    """
    folder_path = resources.files(test_images)
    return Path(folder_path)


def list_cmaps(print_console: bool = False) -> List[str]:
    """
    Print the list of available colormap files from both the internal package folder
    and the local app data folder.
    """

    output = "----- Available colormaps -----\n"

    # List internal data files
    output += "\nAlgorithmic colormaps:\n  - ectotherm / cold_blooded\n  - ectotherm_l / cold_blooded_l\n"
    output += "\nInternal colormaps:\n"
    cmap_names = ["ectotherm", "ectotherm_l"]

    try:
        internal_files = list(resources.contents(lab_control_points))
        json_internal_files = [file for file in internal_files if file.endswith(".json")]
        if json_internal_files:
            for file in json_internal_files:
                output += f"  - {file}\n"
                cmap_names.append(file.split(".json")[0])
        else:
            output += "  (No JSON files found in the internal data folder.)\n"
    except Exception as e:
        output += f"  (Error reading internal data files: {e})\n"

    # List local data files
    output += "\nLocal colormaps:\n"
    local_files = list(data_dir.glob("*.json"))
    if local_files:
        for file in local_files:
            output += f"  - {file.name}\n"
            cmap_names.append(file.stem)
    else:
        output += "  (No JSON files found in the local app data folder.)\n"

    if print_console:
        print(output)
    
    return cmap_names


if __name__ == "__main__":
    list_cmaps(True)
