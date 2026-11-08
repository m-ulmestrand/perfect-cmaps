import matplotlib.colors as mcolors

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from typing import Tuple, List, Optional
import argparse
from pathlib import Path
from tqdm import tqdm

from perfect_cmaps.color_utils import *
from perfect_cmaps.storage import get_test_img_path
from perfect_cmaps.lightness_fit import get_fit, build_lab


# Tolerance for out-of-gamut values when applying a cached lightness fit
# (computed on a reference grid) at a different resolution.
GAMUT_TOLERANCE = 1e-3


def _apply_fit(
    lab_values: np.ndarray,
    fit: LightnessFit,
    lightness: str,
    l_dark: LightnessBounds,
    l_light: LightnessBounds,
    bounds: str,
) -> np.ndarray:
    """Apply a cached lightness fit, falling back to a fresh optimization
    if the result is noticeably out of gamut at this resolution."""
    p_hat = normalized_lightness_profile(lab_values.shape[0], lightness)
    if not fit.feasible:
        warn_infeasible()
        return np.clip(apply_lightness_fit(lab_values, p_hat, fit), 0.0, 1.0)

    rgb_values = apply_lightness_fit(lab_values, p_hat, fit)
    if np.any(rgb_values < -GAMUT_TOLERANCE) or np.any(rgb_values > 1.0 + GAMUT_TOLERANCE):
        rgb_values = rgb_renormalized_lightness(lab_values, lightness, l_dark, l_light, bounds)
    return np.clip(rgb_values, 0.0, 1.0)


def _validate_truncate(truncate: Optional[Tuple[float, float]]) -> Tuple[float, float]:
    if truncate is None:
        return 0.0, 1.0
    low, high = (float(t) for t in truncate)
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"truncate must satisfy 0 <= low < high <= 1, got ({low}, {high})")
    return low, high


def get_cmap(
        cmap_name: str, 
        n: int = 100, 
        ijk: Tuple = None, 
        lightness: str = None,
        interpolation: str = "quadratic",
        smoothing: float = None,
        l_dark: LightnessBounds = None,
        l_light: LightnessBounds = None,
        bounds: str = DEFAULT_BOUNDS_MODE,
        truncate: Optional[Tuple[float, float]] = None,
    ) -> mcolors.LinearSegmentedColormap:
    
    """Get a custom colormap as a matplotlib LinearSegmentedColormap. 
    Two algorithmically generated colormaps are currently available:
        - 'cold_blooded' a.k.a 'ectotherm'
        - 'copper_salt'
    The 'cold_blooded' map also has variants with completely linear lightness: 
        - 'cold_blooded_l', or 'ectotherm_l'.
        
    Additionally, several control points in Lab format are also available in the "lab_control_points" folder.
    These are custom generated through the script 'create_custom_cmap', and consist of a collection 
    points from 'a' and 'b' channels in Lab, along with a lightness profile.
    These points are used in an optimization task, which optimizes the envelope for maximal expressiveness.

    You can change the lightness profile in this script by choosing a lightness profile,
    but note that the optimization task might not be feasible for certain lightness profiles.

    Args:
        cmap_name (str): The name of the colormap, either from the control points folder,
            or the two algorithmically generated colormaps.
        n (int): Number of points in the colormap sequence.
        ijk (Tuple, optional): Order of the channels for the two algorithmic colormaps.
            Typically you would choose a permutation of (0, 1, 2), such as (2, 1, 0). 
            Defaults to None, which amounts to the default permutation of the colormaps.
        lightness (str, optional): Lightness profile. Defaults to None, which means the
            lightness profile the chosen colormap was saved with, or the native lightness
            of the algorithmic colormaps.
        interpolation (str): Interpolation method for the a* and b* channels between
            control points. Defaults to 'quadratic'.
        smoothing (float): Gaussian smoothing of colormap in Lab space, 
            applied after interpolation. Defaults to None.
        l_dark (float | Tuple[float, float], optional): Bounds on the lightness of the
            darkest point of the colormap, as (low, high) or a single fixed value.
            For diverging profiles, this is the lightness of the edges (or the center,
            for the inverted variants). Defaults to None, meaning no constraint.
        l_light (float | Tuple[float, float], optional): Bounds on the lightness of the
            lightest point of the colormap, like ``l_dark``. For the flat profile, the
            constant lightness must satisfy both ``l_dark`` and ``l_light``.
            Within the bounds, the lightness range is maximized (for the flat profile,
            the middle of the feasible lightness interval is chosen). Raises a
            ValueError if the bounds can't be met within the sRGB gamut.
        bounds (str): How ``l_dark`` and ``l_light`` are interpreted. Defaults to "relative":
            fractions (0-1) of the lightness each of them can reach within the gamut,
            e.g. ``l_dark=0.1`` is 10% of the way from the darkest to the lightest
            reachable value of the darkest point. Note that high ``l_dark`` or low ``l_light``
            values leave little room for the other, since the range shrinks to zero at
            the ends. The reachable ranges depend on the lightness profile, interpolation
            and smoothing. With "absolute", the bounds are L* values (0-100).
        truncate (Tuple[float, float], optional): Keep only the part of the colormap
            between these positions (0-1), e.g. (0.2, 0.9). The colors are exactly those of
            the full colormap at the same positions, so lightness bounds refer to the full
            colormap. The truncated colormap still has ``n`` colors. Defaults to None.

    Returns:
        mcolors.LinearSegmentedColormap: Matplotlib-formatted colormap
    """
    if bounds not in BOUNDS_MODES:
        raise ValueError(f"bounds must be one of {BOUNDS_MODES}, got '{bounds}'")
    truncate_low, truncate_high = _validate_truncate(truncate)

    # Build the full colormap with enough samples that the kept part has at least n of them
    n_full = n if truncate is None else max(n, int(np.ceil((n - 1) / (truncate_high - truncate_low))) + 1)
    space = np.linspace(0, 1, n_full)

    if not cmap_name in CMAP_DICT.keys():
        if lightness is not None:
            assert lightness in SUPPORTED_L_PROFILES, f"Lightness profile {lightness} not supported"

        json_data, fit = get_fit(cmap_name, lightness, interpolation, smoothing, l_dark, l_light, bounds)
        if lightness is None:
            lightness = json_data["lightness"]

        control_points = np.array(json_data["points"])
        lab_values = build_lab(control_points, n_full, lightness, interpolation, smoothing)
        rgb_values = _apply_fit(lab_values, fit, lightness, l_dark, l_light, bounds)

    else:
        color_function = CMAP_DICT[cmap_name]
        rgb_values = color_function(space, ijk)
        reoptimize_map = False
        
        if color_function in (cold_blooded_l, cold_blooded) and lightness == "linear" \
                and l_dark is None and l_light is None:
            reoptimize_map = False
        
        elif lightness is not None or l_dark is not None or l_light is not None:
            reoptimize_map = True

        if reoptimize_map:
            if lightness is None:
                lightness = "linear"
            assert lightness in SUPPORTED_L_PROFILES, f"Lightness profile {lightness} not supported"
            lab_values = XYZ_to_Lab(sRGB_to_XYZ(rgb_values[:, :3]))
            rgb_values = np.clip(rgb_renormalized_lightness(lab_values, lightness, l_dark, l_light, bounds), 0.0, 1.0)

    if truncate is not None:
        positions = np.linspace(truncate_low, truncate_high, n)
        rgb_values = np.column_stack([np.interp(positions, space, rgb_values[:, c]) for c in range(3)])
        space = np.linspace(0, 1, n)

    cdict = dict()

    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [[space[i], rgb_values[i][num], rgb_values[i][num]] for i in range(n)]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap(cmap_name, segmentdata=cdict, N=n)
    return cmp


def compare_cmaps(
        names: List[str], 
        num_points: int = 100, 
        figsize: Tuple[float, float] = (8, 8),
        interpolation: str = "quadratic",
        show_plot: bool = True,
        smoothing: Optional[float] = None,
        save_path: Optional[str] = None,
        l_dark: LightnessBounds = None,
        l_light: LightnessBounds = None,
        bounds: str = DEFAULT_BOUNDS_MODE,
        truncate: Optional[Tuple[float, float]] = None,
    ):
    """Show a comparison of colormaps, given their names. Can be used with list_cmaps(), for example.

    Args:
        names (List[str]): Names of colormaps.
        num_points (int, optional): Number of points to display in gradients. Defaults to 100.
        figsize (Tuple[float, float], optional): Size of the comparison plot. Defaults to (8, 8).
        interpolation (str, optional): Interpolation method for the a* and b* channels
            between control points. Defaults to 'quadratic'.
        show_plot (bool, optional): Whether to show the comparison in a window. Defaults to True.
        smoothing (float, optional): Gaussian smoothing of colormap in Lab space, 
            applied after interpolation. Defaults to None.
        save_path (Optional[str], optional): Optional save path for the plot. Defaults to None.
        l_dark, l_light (optional): Lightness bounds of the darkest and lightest points,
            see ``get_cmap``. Defaults to None.
        bounds (str): Whether ``l_dark`` and ``l_light`` are "relative" (default) or
            "absolute", see ``get_cmap``.
        truncate (Tuple[float, float], optional): Keep only this part of each colormap,
            see ``get_cmap``. Defaults to None.
    """
    from matplotlib import pyplot as plt

    assert show_plot or save_path is not None, "Must either display plot with argument 'show_plot' " \
        "or save figure with argument 'save_path'."
    
    gradient = np.linspace(0, 1, num_points)
    gradient = np.vstack((gradient, gradient))

    fig, axes = plt.subplots(nrows=len(names), figsize=figsize)
    if len(names) == 1:
        axes = [axes]  # Make iterable if only one axis

    for ax, name in tqdm(zip(axes, names), "Creating colormaps", total=len(names)):
        cmap = get_cmap(
            name, num_points, interpolation=interpolation, smoothing=smoothing,
            l_dark=l_dark, l_light=l_light, bounds=bounds, truncate=truncate,
        )
        ax: plt.Axes
        ax.imshow(gradient, aspect="auto", cmap=cmap)
        ax.set_axis_off()
        ax.set_title(name, loc='left', fontsize=10, pad=4)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()


def plot_images_with_colormap(image_paths: List[Path], colormap='viridis'):
    import cv2
    from matplotlib import pyplot as plt

    # Create a figure with two rows, one for greyscale and one for colormap images
    num_images = len(image_paths)
    
    fig, axs = plt.subplots(2, num_images, figsize=(25, 10))
    axs[0, 0].set_ylabel("Grayscale", fontsize=20)
    axs[1, 0].set_ylabel("Custom colormap", fontsize=20)
    
    for idx, img_path in enumerate(image_paths):
        # Load the image in greyscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img[:int(img.shape[0] * 0.94)]
        
        # Display the greyscale image
        axs[0, idx].imshow(img, cmap='gray', vmin=0, vmax=255)
        
        # Display the image with the specified colormap
        axs[1, idx].imshow(img, cmap=colormap)
        axs[0, idx].set_xticks([])
        axs[0, idx].set_yticks([])
        axs[1, idx].set_xticks([])
        axs[1, idx].set_yticks([])
    
    plt.tight_layout()
    plt.show()


def parse_args():
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument(
        "--colormap", 
        "-c", 
        type=str, 
        default="ectotherm_l", 
        help="The name of the colormap to be loaded"
    )
    arg_parser.add_argument(
        "--num_points", 
        "-n", 
        type=int, 
        default=1000, 
        help="Number of points in the colormap spectrum"
    )
    arg_parser.add_argument(
        "--lightness", 
        "-l", 
        type=str, 
        default=None, 
        help="The lightness profile of the colormap. Allowed values are: " + 
             "linear, diverging, diverging_inverted, flat."
    )
    arg_parser.add_argument(
        "--interpolation",
        "-i",
        type=str,
        default="quadratic",
        help="Interpolation method for Lab control points"
    )
    arg_parser.add_argument(
        "--ijk",
        type=int,
        nargs='+',
        default=(0, 1, 2),
        help="Channel permutation for algorithmically generated colormaps"
    )
    arg_parser.add_argument(
        "--smoothing",
        "-s",
        type=float,
        default=None,
        help="Gaussian smoothing of colormap in Lab space, applied after interpolation"
    )
    arg_parser.add_argument(
        "--mpl_cmap",
        "-mpl",
        action="store_true",
        default=False,
        help="If specified, looks for a matplotlib colormap instead of one made with this library"
    )
    arg_parser.add_argument(
        "--help", 
        "-h", 
        action="help", 
        default=argparse.SUPPRESS,
        help="Python script for colormap handling with perceptually uniform colormaps. " +
             "Run as a standalone script or import and use the function 'get_colormap' in your own code."
    )
    arg_parser.add_argument(
        "--l_dark",
        type=float,
        default=None,
        help="The lightness value for the dark end of the colormap"
    )
    arg_parser.add_argument(
        "--l_light",
        type=float,
        default=None,
        help="The lightness value for the light end of the colormap"
    )
    return arg_parser.parse_args()


"""
If used as a main script, you can try colormaps out on test images, 
and inspect the RGB channel profile of the colormap along with lightness and luminance.
"""
def main():
    args = parse_args()
    
    if args.mpl_cmap:
        from matplotlib import colormaps
        cmap = colormaps[args.colormap]
    else:
        cmap = get_cmap(
            args.colormap, 
            args.num_points, 
            ijk=args.ijk,
            lightness=args.lightness, 
            interpolation=args.interpolation,
            smoothing=args.smoothing,
            l_dark=args.l_dark,
            l_light=args.l_light
        )

    plot_colormap(cmap, args.num_points)

    test_images = [
        "charcoal4_49721153188_o-scaled.jpg",
        "Lily-Pollen.png",
        "melamine-foam_49722002717_o-scaled.jpg",
        "PhenomPharos-Gallery_Pyrite.jpg",
        "PhenomXLGallery_5.jpg",
    ]
    
    test_image_path = get_test_img_path()
    test_images = [test_image_path / "sem" / img_name for img_name in test_images]
    plot_images_with_colormap(test_images, cmap)


if __name__ == "__main__":
    main()