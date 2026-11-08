"""Upgrade colormap JSON files to the current schema and precompute lightness fits.

Usage:
    python -m perfect_cmaps.migrate                       # user app data folder
    python -m perfect_cmaps.migrate --package-dir perfect_cmaps/lab_control_points
    python -m perfect_cmaps.migrate --all-profiles        # fit every supported lightness profile
    python -m perfect_cmaps.migrate --force               # recompute fits that are already stored
"""
import argparse
import json
from pathlib import Path
from typing import List

from perfect_cmaps.color_utils import SUPPORTED_L_PROFILES
from perfect_cmaps.lightness_fit import DEFAULT_INTERPOLATION, ensure_fit, upgrade_schema
from perfect_cmaps.storage import atomic_write_json, data_dir


def migrate_file(
    path: Path,
    all_profiles: bool = False,
    interpolations: List[str] = None,
    force: bool = False,
) -> bool:
    with open(path, "r") as f:
        json_data = json.load(f)

    if "points" not in json_data or "lightness" not in json_data:
        print(f"Skipping {path.name}: not a colormap file")
        return False

    interpolations = interpolations or [DEFAULT_INTERPOLATION]
    profiles = SUPPORTED_L_PROFILES if all_profiles else [json_data["lightness"]]

    changed = upgrade_schema(json_data)
    if force and json_data["fits"]:
        json_data["fits"] = {}
        changed = True
    for interpolation in interpolations:
        for profile in profiles:
            fit, fit_changed = ensure_fit(json_data, profile, interpolation, None)
            if fit_changed:
                print(f"  {path.stem}: {profile}|{interpolation} -> "
                      f"l_dark={fit.l_dark:.2f}, l_light={fit.l_light:.2f}, feasible={fit.feasible}")
            changed |= fit_changed

    if changed:
        atomic_write_json(path, json_data)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package-dir", type=Path, default=None,
                        help="Migrate this folder (e.g. the package's lab_control_points) instead of the user app data folder.")
    parser.add_argument("--all-profiles", action="store_true",
                        help="Precompute fits for all supported lightness profiles, not just the saved one.")
    parser.add_argument("--force", action="store_true",
                        help="Drop and recompute stored fits, e.g. after changes to the optimization.")
    parser.add_argument("--interpolation", nargs="+", default=[DEFAULT_INTERPOLATION],
                        help=f"Interpolation kinds to precompute. Defaults to '{DEFAULT_INTERPOLATION}'.")
    args = parser.parse_args()

    folder = args.package_dir if args.package_dir is not None else data_dir
    files = sorted(folder.glob("*.json"))
    print(f"Migrating {len(files)} colormap file(s) in {folder}")

    n_changed = 0
    for path in files:
        n_changed += migrate_file(path, args.all_profiles, args.interpolation, args.force)

    print(f"Updated {n_changed} file(s).")


if __name__ == "__main__":
    main()
