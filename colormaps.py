import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import numpy as np
from typing import Tuple
from color_utils import *
from argparse import ArgumentParser


def get_continuous_cmap(cmap_name: str, n: int, ijk: Tuple | None = None):
    space = np.linspace(0, 1, n)

    if cmap_name.endswith(".json"):
        vals = load_json_points(cmap_name, n)
        vals = interpolate_lab_to_rgb(vals, n)
    else:
        vals = CMAP_DICT[cmap_name](space, ijk)
    
    cdict = dict()

    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [[space[i], vals[i][num], vals[i][num]] for i in range(n)]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap(cmap_name, segmentdata=cdict, N=n)
    return cmp


def get_custom_cmap(name: str = "cold_blood", n: int = 1000, ijk: tuple = None):
    cmp = get_continuous_cmap(name, n, ijk)
    return cmp


CMAP_DICT = {
    "cold_blood": cold_blood,
    "copper_salt": copper_salt
}


def parse_args():
    arg_parser = ArgumentParser()
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
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n = args.num_points
    cmap = get_custom_cmap(args.colormap, n)
    # cmap = colormaps["magma"]

    gradient = np.linspace(0, 1, n)
    gradient = np.vstack((gradient, gradient))

    # Get RGB values from colormap
    gradient_rgb = cmap(gradient)

    # Convert the RGB values to grayscale
    lightness = rgb_to_grayscale_lightness(gradient_rgb[0][:, :3])
    luminance = rgb_to_grayscale(gradient_rgb[0][:, :3])
    gradient_gray = np.vstack((lightness, lightness))

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

    # Plot RGB gradient
    axes[0].imshow(gradient_rgb, aspect='auto', vmin=0.0, vmax=1.0)
    axes[0].set_title('RGB gradient')
    axes[0].axis('off')

    # Plot Grayscale gradient
    axes[1].imshow(gradient_gray, cmap='gray', aspect='auto', vmin=0.0, vmax=1.0)
    axes[1].set_title('Lightness gradient')
    axes[1].axis('off')

    for c, color_string in zip(range(3), ["red", "green", "blue"]):
        axes[2].plot(gradient_rgb[0, :, c], color=color_string, label=color_string)
    
    axes[2].plot(lightness, color="grey", linestyle="--", label="lightness")
    axes[2].plot(luminance, color="black", label="luminance")
    plt.legend()

    plt.show()

    for c, color_string in zip(range(3), ["red", "green", "blue"]):
        plt.plot(gradient_rgb[0, :, c], color=color_string, label=color_string)
    
    plt.plot(lightness, color="grey", linestyle="--", label="lightness")
    plt.plot(luminance, color="black", label="luminance")
    plt.legend()
    plt.show()

    dem_data = plt.imread("/home/mattias/imgs/DEM3.gif")
    
    if len(dem_data.shape) == 3:
        dem_data = dem_data[:, :, 0]
        
    plt.imshow(dem_data, cmap=cmap)
    plt.colorbar()
    plt.show()