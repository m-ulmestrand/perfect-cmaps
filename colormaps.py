import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from matplotlib import colormaps

import numpy as np
from typing import Tuple, List
from color_utils import *
import argparse
import cv2
from scipy.ndimage import gaussian_filter


def get_colormap(
        cmap_name: str, 
        n: int = 100, 
        ijk: Tuple | None = None, 
        lightness: str | None = None,
        interpolation: str = "quadratic",
        smoothing: float | None = None
    ) -> mcolors.LinearSegmentedColormap:
    
    """Function for getting custom colormaps. 
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
        ijk (Tuple | None, optional): Order of the channels for the two algorithmic colormaps.
            Typically you would choose a permutation of (0, 1, 2), such as (2, 1, 0). 
            Defaults to None, which amounts to the default permutation of the colormaps.
        lightness (str | None, optional): Lightness profile - only applies for the Lab
            control points. Defaults to None, which means the lightness profile 
            the chosen colormap was saved with.
        interpolation (str): Interpolation method for L values. Defaults to 'quadratic.'
        smoothing (float | None): Gaussian smoothing of colormap in Lab space, 
            applied after interpolation. Defaults to None.

    Returns:
        mcolors.LinearSegmentedColormap: Matplotlib-formatted colormap
    """
    space = np.linspace(0, 1, n)

    if not cmap_name in CMAP_DICT.keys():
        json_data = load_json(cmap_name)
        control_points = np.array(json_data["points"])

        if lightness is None:
            lightness = json_data["lightness"]
        else:
            assert lightness in SUPPORTED_L_PROFILES, f"Lightness profile {lightness} not supported"

        interpolated_values = interpolate_lab(control_points, n, lightness, interpolation)
        if smoothing is not None and smoothing > 0.0:
            sigma = smoothing * n
            interpolated_values[:, 1:] = gaussian_filter(interpolated_values[:, 1:], sigma=sigma, axes=0)

        rgb_values = rgb_renormalized_lightness(interpolated_values, 500, 500)
    
    else:
        color_function = CMAP_DICT[cmap_name]
        rgb_values = color_function(space, ijk)
        reoptimize_map = False
        
        if color_function in (cold_blooded_l, cold_blooded) and lightness == "linear":
            reoptimize_map = False
        
        elif lightness is not None:
            reoptimize_map = True

        if reoptimize_map:
            lab_values = XYZ_to_Lab(sRGB_to_XYZ(rgb_values[:, :3]))
            lab_values[:, 0] = get_lightness_profile(n, lightness)
            rgb_values, _, _ = rgb_renormalized_lightness(lab_values, 200, 500)
    
    cdict = dict()

    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [[space[i], rgb_values[i][num], rgb_values[i][num]] for i in range(n)]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap(cmap_name, segmentdata=cdict, N=n)
    return cmp


def plot_images_with_colormap(image_paths: List[Path], colormap='viridis'):
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
        default="cold_blood", 
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
    return arg_parser.parse_args()


CMAP_DICT = {
    "cold_blooded": cold_blooded,
    "cold_blooded_l": cold_blooded_l,
    "ectotherm": cold_blooded,
    "ectotherm_l": cold_blooded_l,
    "copper_salt": copper_salt
}


"""
If used as a main script, you can try colormaps out on test images, 
and inspect the RGB channel profile of the colormap along with lightness and luminance.
"""
if __name__ == "__main__":
    args = parse_args()
    
    if args.mpl_cmap:
        cmap = colormaps[args.colormap]
    else:
        cmap = get_colormap(
            args.colormap, 
            args.num_points, 
            ijk=args.ijk,
            lightness=args.lightness, 
            interpolation=args.interpolation,
            smoothing=args.smoothing
        )

    plot_colormap(cmap, args.num_points)

    test_images = [
        "charcoal4_49721153188_o-scaled.jpg",
        "Lily-Pollen.png",
        "melamine-foam_49722002717_o-scaled.jpg",
        "PhenomPharos-Gallery_Pyrite.jpg",
        "PhenomXLGallery_5.jpg",
    ]
    
    test_image_path = Path(__file__).parent / "test_images"
    test_images = [test_image_path / "sem" / img_name for img_name in test_images]
    plot_images_with_colormap(test_images, cmap)
