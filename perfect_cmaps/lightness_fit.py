"""Caching of optimized lightness fits for Lab control-point colormaps.

The lightness optimization finds the lightness of the darkest and lightest point of a
colormap, ``(l_dark, l_light)``, such that ``L(t) = l_dark + (l_light - l_dark) * p(t)``
stays in the sRGB gamut, where p is the normalized lightness profile. Both the profile and
the interpolated a*/b* curves are functions of the normalized position t in [0, 1], so the
fit is independent of the number of samples. It is computed once on a dense reference grid.

Fits with default (unconstrained) lightness bounds are stored in the colormap's JSON data
under ``"fits"``, keyed by lightness profile, interpolation and smoothing. Fits with
user-given bounds are only memoized in-process. The gamut envelope, which is the expensive
part of a fit and does not depend on the lightness profile or the bounds, is memoized too.
"""
import hashlib
import json
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from perfect_cmaps.color_utils import (
    DEFAULT_BOUNDS_MODE,
    LightnessBounds,
    LightnessFit,
    interpolate_lab,
    lightness_envelope,
    normalize_lightness_bounds,
    normalized_lightness_profile,
    solve_lightness,
)
from perfect_cmaps.storage import (
    SOURCE_PACKAGE,
    load_data_with_source,
    load_fit_side_cache,
    update_data,
)


SCHEMA_VERSION = 3
N_REF = 1024

DEFAULT_INTERPOLATION = "quadratic"

# In-process memos
_memo: Dict[tuple, LightnessFit] = {}
_envelope_memo: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}


def _smoothing_key(smoothing: Optional[float]) -> str:
    smoothing = round(float(smoothing), 4) if smoothing else 0
    return f"{smoothing:g}"


def fit_key(lightness: str, interpolation: str, smoothing: Optional[float]) -> str:
    return f"{lightness}|{interpolation}|{_smoothing_key(smoothing)}"


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


def get_envelope(
    control_points: np.ndarray,
    interpolation: str,
    smoothing: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Gamut envelope (L_min, L_max) of the colormap on the reference grid, memoized."""
    key = (points_hash(control_points), interpolation, _smoothing_key(smoothing))
    if key not in _envelope_memo:
        # The lightness profile doesn't affect a* and b*, so any profile will do here
        lab = build_lab(control_points, N_REF, "linear", interpolation, smoothing)
        _envelope_memo[key] = lightness_envelope(lab)
    return _envelope_memo[key]


def compute_fit(
    control_points: np.ndarray,
    lightness: str,
    interpolation: str,
    smoothing: Optional[float],
    l_dark: LightnessBounds = None,
    l_light: LightnessBounds = None,
    bounds: str = DEFAULT_BOUNDS_MODE,
) -> LightnessFit:
    L_min, L_max = get_envelope(control_points, interpolation, smoothing)
    p_hat = normalized_lightness_profile(N_REF, lightness)
    return solve_lightness(p_hat, L_min, L_max, lightness == "flat", l_dark, l_light, bounds)


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
        # Fits from older schema versions use a different parametrization
        json_data["version"] = SCHEMA_VERSION
        json_data["fits"] = {}
        changed = True
    return changed


def ensure_fit(
    json_data: dict,
    lightness: str,
    interpolation: str,
    smoothing: Optional[float],
) -> Tuple[dict, bool]:
    """Return the stored fit (with default lightness bounds) for the given parameters,
    computing it if missing.
    Returns (fit, changed), where ``changed`` signals that ``json_data`` was modified."""
    changed = upgrade_schema(json_data)
    key = fit_key(lightness, interpolation, smoothing)
    stored = json_data["fits"].get(key)
    if stored is not None:
        return LightnessFit.from_dict(stored), changed

    control_points = np.asarray(json_data["points"], dtype=np.float64)
    fit = compute_fit(control_points, lightness, interpolation, smoothing)
    json_data["fits"][key] = {**fit.to_dict(), "n_ref": N_REF}
    return fit, True


def get_fit(
    cmap_name: str,
    lightness: Optional[str],
    interpolation: str,
    smoothing: Optional[float],
    l_dark: LightnessBounds = None,
    l_light: LightnessBounds = None,
    bounds: str = DEFAULT_BOUNDS_MODE,
) -> Tuple[dict, LightnessFit]:
    """Load a colormap's data and its lightness fit, optimizing on a cache miss.
    Fits with default lightness bounds are persisted to the colormap's data.

    Returns:
        Tuple[dict, LightnessFit]: (json_data, fit)
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

    constrained = l_dark is not None or l_light is not None
    memo_key = (cmap_name, json_data["points_hash"], fit_key(lightness, interpolation, smoothing))
    if constrained:
        memo_key += (
            bounds,
            normalize_lightness_bounds(l_dark, "l_dark", bounds),
            normalize_lightness_bounds(l_light, "l_light", bounds),
        )
    if memo_key in _memo:
        return json_data, _memo[memo_key]

    if constrained:
        control_points = np.asarray(json_data["points"], dtype=np.float64)
        fit = compute_fit(control_points, lightness, interpolation, smoothing, l_dark, l_light, bounds)
        _memo[memo_key] = fit
        return json_data, fit

    fit, changed = ensure_fit(json_data, lightness, interpolation, smoothing)
    if changed:
        try:
            update_data(cmap_name, json_data, source)
        except OSError as e:
            print(f"Could not store lightness fit for '{cmap_name}': {e}")

    _memo[memo_key] = fit
    return json_data, fit
