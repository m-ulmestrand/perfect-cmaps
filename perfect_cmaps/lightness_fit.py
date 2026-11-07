"""Caching of optimized lightness fits for Lab control-point colormaps.

The lightness optimization finds an affine transform ``L = scale * L_profile(t) + offset``
of the intended lightness profile. Both the profile and the interpolated a*/b* curves
are functions of the normalized position t in [0, 1], so the fit is independent of the
number of samples. It is computed once on a dense reference grid and stored in the
colormap's JSON data under ``"fits"``, keyed by lightness profile, interpolation and smoothing.
"""
import hashlib
import json
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from perfect_cmaps.color_utils import (
    interpolate_lab,
    optimize_lightness_genes,
)
from perfect_cmaps.storage import (
    SOURCE_PACKAGE,
    load_data_with_source,
    load_fit_side_cache,
    update_data,
)


SCHEMA_VERSION = 2
N_REF = 1024

DEFAULT_INTERPOLATION = "quadratic"

# In-process memo: (cmap name, points hash, fit key) -> fit dict
_memo: Dict[Tuple[str, str, str], dict] = {}


def fit_key(lightness: str, interpolation: str, smoothing: Optional[float]) -> str:
    smoothing = round(float(smoothing), 4) if smoothing else 0
    return f"{lightness}|{interpolation}|{smoothing:g}"


def points_hash(points) -> str:
    rounded = np.round(np.asarray(points, dtype=np.float64), 6).tolist()
    return hashlib.sha1(json.dumps(rounded).encode()).hexdigest()


def build_lab(
    control_points: np.ndarray,
    n: int,
    lightness: str,
    interpolation: str,
    smoothing: Optional[float],
) -> np.ndarray:
    """Interpolate control points to ``n`` Lab colors with the intended lightness profile."""
    lab = interpolate_lab(control_points, n, lightness, interpolation)
    if smoothing is not None and smoothing > 0.0:
        lab[:, 1:] = gaussian_filter1d(lab[:, 1:], sigma=smoothing * n, axis=0)
    return lab


def compute_fit(
    control_points: np.ndarray,
    lightness: str,
    interpolation: str,
    smoothing: Optional[float],
    n: int = N_REF,
) -> dict:
    lab = build_lab(control_points, n, lightness, interpolation, smoothing)
    scale, offset, fitness = optimize_lightness_genes(lab)
    return {"scale": scale, "offset": offset, "fitness": fitness, "n_ref": n}


def upgrade_schema(json_data: dict) -> bool:
    """Bring legacy (v1) data up to the current schema in place.
    Drops cached fits if the control points changed. Returns True if anything changed."""
    changed = False
    current_hash = points_hash(json_data["points"])
    if json_data.get("points_hash") != current_hash:
        json_data["points_hash"] = current_hash
        json_data["fits"] = {}
        changed = True
    if "fits" not in json_data:
        json_data["fits"] = {}
        changed = True
    if json_data.get("version") != SCHEMA_VERSION:
        json_data["version"] = SCHEMA_VERSION
        changed = True
    return changed


def ensure_fit(
    json_data: dict,
    lightness: str,
    interpolation: str,
    smoothing: Optional[float],
) -> Tuple[dict, bool]:
    """Return the cached fit for the given parameters, computing it if missing.
    Returns (fit, changed), where ``changed`` signals that ``json_data`` was modified."""
    changed = upgrade_schema(json_data)
    key = fit_key(lightness, interpolation, smoothing)
    fit = json_data["fits"].get(key)
    if fit is None:
        control_points = np.asarray(json_data["points"], dtype=np.float64)
        fit = compute_fit(control_points, lightness, interpolation, smoothing)
        json_data["fits"][key] = fit
        changed = True
    return fit, changed


def get_fit(
    cmap_name: str,
    lightness: Optional[str],
    interpolation: str,
    smoothing: Optional[float],
) -> Tuple[dict, dict]:
    """Load a colormap's data and its lightness fit, optimizing and persisting
    the fit on a cache miss.

    Returns:
        Tuple[dict, dict]: (json_data, fit)
    """
    json_data, source = load_data_with_source(cmap_name)
    upgrade_schema(json_data)
    if source == SOURCE_PACKAGE:
        side = load_fit_side_cache(cmap_name)
        if side.get("points_hash") == json_data["points_hash"]:
            # Shipped fits take precedence over locally computed ones
            json_data["fits"] = {**side.get("fits", {}), **json_data["fits"]}

    if lightness is None:
        lightness = json_data["lightness"]

    memo_key = (cmap_name, points_hash(json_data["points"]), fit_key(lightness, interpolation, smoothing))
    if memo_key in _memo:
        return json_data, _memo[memo_key]

    fit, changed = ensure_fit(json_data, lightness, interpolation, smoothing)
    if changed:
        try:
            update_data(cmap_name, json_data, source)
        except OSError as e:
            print(f"Could not store lightness fit for '{cmap_name}': {e}")

    _memo[memo_key] = fit
    return json_data, fit
