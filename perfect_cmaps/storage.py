from pathlib import Path
from appdirs import user_data_dir
import json

import sys
sys.path.append(Path(__file__).parent.parent.absolute().as_posix())

import importlib.resources as pkg_resources
from perfect_cmaps import lab_control_points


app_name = "perfect_cmaps"
data_dir = Path(user_data_dir(app_name))
data_dir.mkdir(parents=True, exist_ok=True)


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
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)
    
    print("Saved colormap data as", new_name)


def load_data(filename: str):
    """
    Load data from the internal package folder first.
    If not found, try the custom app data folder.
    """

    filename = Path(filename).with_suffix(".json")
    # Try loading from the internal package data folder
    try:
        with pkg_resources.open_text(lab_control_points, filename) as f:
            return json.load(f)
    except (FileNotFoundError, ModuleNotFoundError):
        pass  # File not found internally, fallback to the custom app folder

    # Try loading from the custom app folder
    custom_path = data_dir / filename
    if custom_path.exists():
        with open(custom_path, "r") as f:
            return json.load(f)

    # If not found in either location, raise an exception
    raise FileNotFoundError(f"{filename} not found in internal or app data folders.")
