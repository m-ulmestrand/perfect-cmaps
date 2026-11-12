"""Reoptimize the lightness of existing (e.g. matplotlib) colormaps.

The colormap is sampled and converted to CIELAB. Its a* and b* channels are kept as control
points, and the L* channel is replaced by one of the supported lightness profiles. With a
target lightness range, chroma is lowered where needed to reach it within the sRGB gamut. The result can be saved as a colormap JSON
file, after which it is available through ``get_cmap()`` like any other colormap.
"""
import argparse
from typing import Optional, Tuple, Union

import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap

from perfect_cmaps.color_utils import (
    BOUNDS_MODES,
    DEFAULT_BOUNDS_MODE,
    SUPPORTED_L_PROFILES,
    LightnessBounds,
    apply_lightness_fit,
    in_gamut,
    normalized_lightness_profile,
    warn_infeasible,
)
from perfect_cmaps.colorspace import sRGB_to_Lab
from perfect_cmaps.lightness_fit import N_REF, STORED_BOUNDS_KEY, build_lab, compute_fit
from perfect_cmaps.storage import save_data


# Colormaps whose L* varies less than this are considered flat
FLAT_LIGHTNESS_SPAN = 10.0
MAX_CONTROL_POINTS = 256

# Default number of control points with target_range. Chroma is scaled per control point,
# so fewer points give smoother chroma changes through interpolation.
DEFAULT_TARGET_N_POINTS = 10
# Samples used to measure the original colormap's lightness
LIGHTNESS_SAMPLES = 1024
CHROMA_BISECTION_STEPS = 40
CHROMA_REFINEMENT_STEPS = 60
# Keeps target lightness this far from L* = 0 and 100, where hardly any chroma fits the gamut,
# like the lightness profiles do. The lightness range is still maximized afterwards.
TARGET_L_MARGIN = 0.5
# Colors are scaled to fit the gamut within this distance (in L*) of their target lightness,
# so that the target isn't exactly on the gamut boundary and the lightness fit has some slack
TARGET_L_SLACK = 0.05

TargetRange = Union[None, str, Tuple[float, float]]
TARGET_RANGE_NAMES = ("original", "full")


def _resolve_cmap(cmap: Union[str, Colormap]) -> Colormap:
    if isinstance(cmap, Colormap):
        return cmap
    from matplotlib import colormaps
    try:
        return colormaps[cmap]
    except KeyError:
        raise ValueError(f"'{cmap}' is not a matplotlib colormap") from None


def _is_descending(L: np.ndarray) -> bool:
    t = np.linspace(0.0, 1.0, L.shape[0])
    return np.corrcoef(t, L)[0, 1] < 0


def detect_lightness_profile(L: np.ndarray) -> Tuple[str, bool]:
    """Find the supported lightness profile closest to the lightness curve ``L``.

    Each non-flat profile is fitted as ``L ≈ α + β·p`` with β > 0, where p is the normalized
    profile. The linear profile is also tried in reverse, for colormaps that go from light to dark.

    Returns:
        Tuple[str, bool]: (profile, reversed)
    """
    L = np.asarray(L, dtype=float)
    if np.ptp(L) < FLAT_LIGHTNESS_SPAN:
        return "flat", False

    total = np.sum((L - L.mean()) ** 2)
    candidates = []
    for profile in SUPPORTED_L_PROFILES:
        if profile == "flat":
            continue
        p_hat = normalized_lightness_profile(L.shape[0], profile)
        variants = [(p_hat, False), (p_hat[::-1], True)] if profile == "linear" else [(p_hat, False)]
        for p, reverse in variants:
            A = np.column_stack((np.ones_like(p), p))
            coeffs = np.linalg.lstsq(A, L, rcond=None)[0]
            if coeffs[1] <= 0:
                continue
            residual = np.sum((L - A @ coeffs) ** 2) / total
            candidates.append((residual, profile, reverse))

    if not candidates:
        return "linear", _is_descending(L)
    _, profile, reverse = min(candidates)
    return profile, reverse


def _resolve_target_range(target_range: TargetRange, L: np.ndarray) -> Optional[Tuple[float, float]]:
    if target_range is None:
        return None
    if isinstance(target_range, str):
        if target_range == "original":
            target = (float(L.min()), float(L.max()))
        elif target_range == "full":
            target = (0.0, 100.0)
        else:
            raise ValueError(f"target_range must be one of {TARGET_RANGE_NAMES} or (l_dark, l_light), got '{target_range}'")
    else:
        target = tuple(float(t) for t in target_range)
        if len(target) != 2 or not 0.0 <= target[0] < target[1] <= 100.0:
            raise ValueError(f"target_range must satisfy 0 <= l_dark < l_light <= 100, got {target_range}")
    return max(target[0], TARGET_L_MARGIN), min(target[1], 100.0 - TARGET_L_MARGIN)


def max_chroma_scale(L: np.ndarray, points: np.ndarray) -> np.ndarray:
    """For each color, the largest factor s in [0, 1] such that (L, s·a*, s·b*) is in the sRGB gamut.
    Grey (s = 0) is in the gamut for any L* in (0, 100)."""
    a, b = points[:, 0], points[:, 1]
    inside = in_gamut(L, a, b)
    low = np.where(inside, 1.0, 0.0)
    high = np.ones_like(L)
    for _ in range(CHROMA_BISECTION_STEPS):
        mid = (low + high) / 2
        mid_inside = in_gamut(L, mid * a, mid * b)
        low = np.where(inside, 1.0, np.where(mid_inside, mid, low))
        high = np.where(inside, 1.0, np.where(mid_inside, high, mid))
    return low


def _max_chroma_scale_with_slack(L: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Like ``max_chroma_scale``, but the color must fit the gamut for every lightness within
    ``TARGET_L_SLACK`` of ``L``. For fixed a* and b*, the in-gamut lightness is an interval,
    so checking both ends and the middle is enough."""
    scales = [
        max_chroma_scale(np.clip(L + offset, TARGET_L_MARGIN, 100.0 - TARGET_L_MARGIN), points)
        for offset in (-TARGET_L_SLACK, 0.0, TARGET_L_SLACK)
    ]
    return np.minimum.reduce(scales)


def _target_lightness(n_values: int, lightness: str, target: Tuple[float, float]) -> np.ndarray:
    return target[0] + (target[1] - target[0]) * normalized_lightness_profile(n_values, lightness)


def _neighbour_minimum(dense_values: np.ndarray, n_points: int, width: int = 1) -> np.ndarray:
    """For each control point, the minimum of ``dense_values`` (on an evenly spaced grid over
    the same interval) within ``width`` control-point intervals of it."""
    position = np.floor(np.linspace(0.0, n_points - 1, dense_values.shape[0])).astype(int)
    result = np.ones(n_points)
    for offset in range(1 - width, width + 1):
        np.minimum.at(result, np.clip(position + offset, 0, n_points - 1), dense_values)
    return result


def fit_chroma(
    points: np.ndarray,
    lightness: str,
    target: Tuple[float, float],
    interpolation: str = "quadratic",
    smoothing: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale a*/b* control points towards grey (keeping their hue) where the colormap can't
    reach the target lightness profile within the gamut. Points are in dark-to-light order.
    The chroma varies smoothly between control points through interpolation, so fewer
    control points give more gradual chroma changes.

    The required scaling is first estimated at the control points, then refined on the
    interpolated colormap (the reference grid used by the lightness fit), since interpolation
    can overshoot between control points.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (scaled points, scale factors)
    """
    n_points = points.shape[0]
    L_dense = _target_lightness(N_REF, lightness, target)

    scale = _max_chroma_scale_with_slack(_target_lightness(n_points, lightness, target), points)
    for step in range(CHROMA_REFINEMENT_STEPS):
        # The lightness profile doesn't affect a* and b*, so any profile will do here
        dense = build_lab(points * scale[:, None], N_REF, "linear", interpolation, smoothing)
        dense_scale = _max_chroma_scale_with_slack(L_dense, dense[:, 1:])
        if np.all(dense_scale >= 1.0):
            break
        # Interpolation between control points isn't local (e.g. quadratic splines), so if lowering
        # the nearest control points isn't enough, a growing neighbourhood is lowered. This always
        # terminates, since grey fits the gamut.
        scale = scale * _neighbour_minimum(dense_scale, n_points, width=1 + step // 2)
    else:
        print("Chroma scaling did not fully converge; the lightness range may be slightly below the target.")
    return points * scale[:, None], scale


def optimize_cmap(
    cmap: Union[str, Colormap],
    n: int = 256,
    num_control: Optional[int] = None,
    lightness: Optional[str] = None,
    target_range: TargetRange = None,
    l_dark: LightnessBounds = None,
    l_light: LightnessBounds = None,
    bounds: str = DEFAULT_BOUNDS_MODE,
    interpolation: str = "quadratic",
    smoothing: Optional[float] = None,
    save_name: Optional[str] = None,
    plot: bool = False,
) -> LinearSegmentedColormap:
    """Reoptimize the lightness of an existing colormap, e.g. one from matplotlib.

    L* is replaced by a lightness profile. By default, the a* and b* channels are kept and
    the lightness range stays within the colormap's original range. With ``target_range``,
    colors are moved towards grey where needed to reach a wider lightness range.

    Args:
        cmap (str | Colormap): A matplotlib colormap name, or a Colormap object.
        n (int): Number of colors in the returned colormap. Defaults to 256.
        num_control (int, optional): Number of control points sampled from the colormap.
            Defaults to the number of colors in the colormap (at most 256) without
            ``target_range``, and 32 with it: chroma is scaled per control point, and fewer
            control points spread the change more gradually through interpolation.
        lightness (str, optional): Lightness profile, one of ``SUPPORTED_L_PROFILES``.
            Defaults to None, which detects the profile closest to the colormap's own lightness.
        target_range (str | Tuple[float, float], optional): Lightness (L*, 0-100) to reach at
            the darkest and lightest points, as (l_dark, l_light), "original" for the
            colormap's own range, or "full" for (0, 100), aimed at as (0.5, 99.5). The chroma of control points that
            can't reach their target lightness within the gamut is scaled down (keeping the
            hue) just enough to fit. The lightness range is then maximized as usual, so it is
            at least the target range. Not supported for the flat profile. Defaults to None:
            chroma is kept as is, and the lightness range is maximized only within the
            original range of the colormap.
        l_dark, l_light (optional): Lightness bounds of the darkest and lightest points,
            see ``get_cmap``. If given, they replace the original-range bounds used without
            ``target_range``.
        bounds (str): Whether ``l_dark`` and ``l_light`` are "relative" (default) or
            "absolute", see ``get_cmap``.
        interpolation (str): Interpolation method for the a* and b* channels between
            control points. Defaults to 'quadratic'.
        smoothing (float, optional): Gaussian smoothing in Lab space, see ``get_cmap``.
        save_name (str, optional): If given, save the control points as a colormap JSON file
            under this name (a suffix is added if the name is taken), loadable with ``get_cmap``.
            The lightness bounds are saved too, and ``get_cmap`` uses them unless it is given
            ``l_dark`` or ``l_light``.
        plot (bool): Plot the original and optimized colormaps with their lightness. Defaults to False.

    Returns:
        LinearSegmentedColormap: The optimized colormap. If saved, its name is the saved name.
    """
    if bounds not in BOUNDS_MODES:
        raise ValueError(f"bounds must be one of {BOUNDS_MODES}, got '{bounds}'")
    if lightness is not None and lightness not in SUPPORTED_L_PROFILES:
        raise ValueError(f"Lightness profile '{lightness}' not supported, choose from {SUPPORTED_L_PROFILES}")

    original = _resolve_cmap(cmap)
    if num_control is None:
        num_control = min(original.N, MAX_CONTROL_POINTS if target_range is None else DEFAULT_TARGET_N_POINTS)
    if num_control < 3:
        raise ValueError(f"n_points must be at least 3, got {num_control}")

    lab = sRGB_to_Lab(original(np.linspace(0.0, 1.0, num_control))[:, :3])
    # The original lightness is measured densely, independent of the number of control points
    L = sRGB_to_Lab(original(np.linspace(0.0, 1.0, LIGHTNESS_SAMPLES))[:, :3])[:, 0]

    if lightness is None:
        lightness, reverse = detect_lightness_profile(L)
        print(f"Detected lightness profile for '{original.name}': {lightness}" + (" (reversed)" if reverse else ""))
    else:
        reverse = lightness == "linear" and _is_descending(L)

    points = lab[:, 1:]
    if reverse:
        points = points[::-1]
    points = np.ascontiguousarray(points)

    target = _resolve_target_range(target_range, L)
    if target is not None:
        if lightness == "flat":
            raise ValueError("target_range is not supported for the flat lightness profile")
        points, scale = fit_chroma(points, lightness, target, interpolation, smoothing)
        print(
            f"Chroma scaled to reach L* {target[0]:.1f}-{target[1]:.1f}: "
            f"average chroma scale {scale.mean():.2f}, minimum {scale.min():.2f}"
        )

    elif l_dark is None and l_light is None:
        # Keep the lightness range within the original one
        l_dark, l_light, bounds = (float(L.min()), 100.0), (0.0, float(L.max())), "absolute"

    if save_name is not None:
        from perfect_cmaps.colormaps import get_cmap  # Imported here to avoid a circular import
        data = {"points": points.tolist(), "lightness": lightness}
        if reverse:
            data["reversed"] = True
        if l_dark is not None or l_light is not None:
            data[STORED_BOUNDS_KEY] = {"l_dark": l_dark, "l_light": l_light, "bounds": bounds}
        saved_name = save_data(data, save_name)
        result = get_cmap(
            saved_name, n, lightness=lightness, interpolation=interpolation, smoothing=smoothing,
            l_dark=l_dark, l_light=l_light, bounds=bounds,
        )
    else:
        fit = compute_fit(points, lightness, interpolation, smoothing, l_dark, l_light, bounds)
        if not fit.feasible:
            warn_infeasible()
        lab_values = build_lab(points, n, lightness, interpolation, smoothing)
        p_hat = normalized_lightness_profile(n, lightness)
        rgb_values = np.clip(apply_lightness_fit(lab_values, p_hat, fit), 0.0, 1.0)
        if reverse:
            rgb_values = rgb_values[::-1]
        result = LinearSegmentedColormap.from_list(f"{original.name}_optimized", rgb_values, N=n)

    if plot:
        plot_optimization(original, result)
    return result


def plot_optimization(original: Colormap, optimized: Colormap, num_points: int = 512):
    """Plot two colormaps and their lightness, e.g. before and after optimization."""
    from matplotlib import pyplot as plt

    x = np.linspace(0.0, 1.0, num_points)
    gradient = np.vstack((x, x))

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=(1, 1, 4))
    for row, (label, cmap) in enumerate((("Original", original), ("Optimized", optimized))):
        ax = fig.add_subplot(gs[row, :])
        ax.imshow(gradient, aspect="auto", cmap=cmap)
        ax.set_axis_off()
        ax.set_title(f"{label}: {cmap.name}", fontsize=12, loc="left")

    ax_L = fig.add_subplot(gs[2, 0])
    ax_C = fig.add_subplot(gs[2, 1], sharex=ax_L)
    for label, cmap, style, color in (("Original", original, "--", "grey"), ("Optimized", optimized, "-", "black")):
        lab = sRGB_to_Lab(np.clip(cmap(x)[:, :3], 0.0, 1.0))
        ax_L.plot(x, lab[:, 0], style, color=color, linewidth=2, label=label)
        ax_C.plot(x, np.hypot(lab[:, 1], lab[:, 2]), style, color=color, linewidth=2, label=label)
    ax_L.set_xlim(0.0, 1.0)
    ax_L.set_ylim(0.0, 100.0)
    ax_L.set_ylabel("Lightness L*")
    ax_C.set_ylim(bottom=0.0)
    ax_C.set_ylabel("Chroma C*")
    for ax in (ax_L, ax_C):
        ax.set_xlabel("Position")
        ax.legend()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Reoptimize the lightness of a matplotlib colormap.")
    parser.add_argument("--colormap", "-c", type=str, default="viridis", help="Name of a matplotlib colormap")
    parser.add_argument(
        "--lightness", "-l", type=str, default="auto",
        help=f"Lightness profile: auto, or one of {', '.join(SUPPORTED_L_PROFILES)}",
    )
    parser.add_argument("--num_points", "-n", type=int, default=256, help="Number of colors in the result")
    parser.add_argument("--num_control", type=int, default=None, help="Number of control points to sample")
    parser.add_argument(
        "--target_range", "-t", type=str, nargs="+", default=None,
        help="Lightness range to reach by lowering chroma where needed: original, full, or two L* values. "
             "Without it, the colors are kept and the lightness stays within the original range",
    )
    parser.add_argument("--interpolation", "-i", type=str, default="quadratic", help="Interpolation of a* and b*")
    parser.add_argument("--smoothing", "-s", type=float, default=None, help="Gaussian smoothing in Lab space")
    parser.add_argument("--save", type=str, default=None, help="Save the optimized colormap under this name")
    parser.add_argument("--no_plot", action="store_true", help="Don't plot the result")
    return parser.parse_args()


def main():
    args = parse_args()
    target_range = args.target_range
    if target_range is not None:
        target_range = target_range[0] if len(target_range) == 1 else tuple(float(t) for t in target_range)
    optimize_cmap(
        args.colormap,
        lightness=None if args.lightness == "auto" else args.lightness,
        n=args.num_points,
        num_control=args.num_control,
        target_range=target_range,
        interpolation=args.interpolation,
        smoothing=args.smoothing,
        save_name=args.save,
        plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
