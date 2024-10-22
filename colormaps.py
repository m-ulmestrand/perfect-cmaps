import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from matplotlib import colormaps

import numpy as np
from typing import Tuple
from color_utils import *
import argparse


def get_colormap(
        cmap_name: str, 
        n: int = 100, 
        ijk: Tuple | None = None, 
        lightness: str | None = None
    ) -> mcolors.LinearSegmentedColormap:
    
    """Function for getting custom colormaps. 
    Two algorithmically generated colormaps are currently available:
    - 'cold_blood' a.k.a 'ectotherm'
    - 'copper_salt'

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

    Returns:
        mcolors.LinearSegmentedColormap: Matplotlib-formatted colormap
    """
    space = np.linspace(0, 1, n)

    if not cmap_name in CMAP_DICT.keys():
        json_data = load_json(cmap_name, n)
        control_points = np.array(json_data["points"])

        if lightness is None:
            lightness = json_data["lightness"]
        else:
            assert lightness in SUPPORTED_L_PROFILES, f"Lightness profile {lightness} not supported"

        interpolated_values = interpolate_lab(control_points, n, lightness)
        rgb_values, _, _ = rgb_renormalized_lightness(interpolated_values, lightness)
    else:
        rgb_values = CMAP_DICT[cmap_name](space, ijk)
    
    cdict = dict()

    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [[space[i], rgb_values[i][num], rgb_values[i][num]] for i in range(n)]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap(cmap_name, segmentdata=cdict, N=n)
    return cmp


CMAP_DICT = {
    "cold_blood": cold_blood,
    "ectotherm": cold_blood,
    "copper_salt": copper_salt
}


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
        "--help", 
        "-h", 
        action="help", 
        default=argparse.SUPPRESS,
        help="Python script for colormap handling with perceptually uniform colormaps. " +
             "Run as a standalone script or import and use the function 'get_colormap' in your own scripts."
    )
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n = args.num_points
    cmap = get_colormap(args.colormap, n, lightness=args.lightness)
    # cmap = colormaps["cividis"]

    gradient = np.linspace(0, 1, n)
    gradient = np.vstack((gradient, gradient))

    # Get RGB values from colormap
    gradient_rgb = cmap(gradient)

    # Convert the RGB values to grayscale
    lightness = rgb_to_grayscale_lightness(gradient_rgb[0][:, :3])
    luminance = rgb_to_grayscale(gradient_rgb[0][:, :3])
    gradient_gray = np.vstack((lightness, lightness))

    # Plotting
    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # Define the axes
    ax0 = fig.add_subplot(gs[0, 0])  # Top-left subplot
    ax1 = fig.add_subplot(gs[1, 0])  # Bottom-left subplot
    ax2 = fig.add_subplot(gs[:, 1])  # Right subplot spanning both rows

    # Plot RGB gradient on ax0
    ax0.imshow(gradient_rgb, aspect="auto", vmin=0.0, vmax=1.0)
    ax0.set_title("RGB gradient", fontsize=20)
    ax0.axis("off")

    # Plot Grayscale gradient on ax1
    ax1.imshow(gradient_gray, cmap="gray", aspect="auto", vmin=0.0, vmax=1.0)
    ax1.set_title("Lightness gradient", fontsize=20)
    ax1.axis("off")

    # Plot the color channels and luminance on ax2
    for c, color_string in zip(range(3), ["red", "green", "blue"]):
        ax2.plot(gradient_rgb[0, :, c], color=color_string, label=color_string, linewidth=4)

    ax2.plot(lightness, color="grey", linestyle="--", label="lightness", linewidth=4)
    ax2.plot(luminance, color="black", label="luminance")
    ax2.set_title("Color channel intensities", fontsize=20)
    ax2.legend(loc=0, fontsize=20)
    plt.show()

    test_image = Path(__file__).parent / "test_images" / "DEM3.gif"
    dem_data = plt.imread(test_image)
    
    if len(dem_data.shape) == 3:
        dem_data = dem_data[:, :, 0]
        
    plt.imshow(dem_data, cmap=cmap)
    plt.title("Example digital elevation data", fontsize=20)
    plt.colorbar()
    plt.show()