import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from colour import Lab_to_XYZ, XYZ_to_sRGB
from numba import njit
import argparse
from scipy.ndimage import gaussian_filter1d

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from perfect_cmaps.color_utils import (
    get_lightness_profile, 
    plot_colormap, 
    rgb_renormalized_lightness, 
    interpolate_lab,
    Lab_to_sRGB,
    SUPPORTED_L_PROFILES
)
from perfect_cmaps.storage import save_data


@njit
def compute_occupancy(RGB_values: np.ndarray, num_bins: int) -> np.ndarray:
    occupancy_matrix = np.zeros((num_bins, num_bins), dtype=np.uint8)
    
    for i in range(num_bins):
        for j in range(num_bins):
            index = i * num_bins + j
            R, G, B = RGB_values[index]
            if 0 <= R <= 1 and 0 <= G <= 1 and 0 <= B <= 1:  # Check if all RGB values are valid
                occupancy_matrix[i, j] = 1
    
    return occupancy_matrix


def get_ab_occupancy(L_value: float, ab_values: np.ndarray, num_bins: int) -> np.ndarray: 
    Lab_values = np.append(np.full((ab_values.shape[0], 1), L_value), ab_values, axis=1)
    RGB_values = Lab_to_sRGB(Lab_values)
    return compute_occupancy(RGB_values, num_bins).astype(bool)


def create_background_image(num_bins: int, num_steps: int = 8):
    image = np.full((num_bins, num_bins, 3), 0.8)
    line_positions = np.arange(0, num_bins, num_bins // num_steps)[1:]
    image[line_positions, :, :] = 0
    image[:, line_positions, :] = 0
    return image


def project_onto_gamut(x: float, y: float, ab_slice, num_bins: int):
    # Convert click (x, y) in a*, b* coordinates to transposed occupancy matrix indices
    b_bin = int((x + 100) * (num_bins / 200))  # Now x corresponds to b* due to transposition
    a_bin = int((y + 100) * (num_bins / 200))  # Now y corresponds to a* due to transposition

    # Clip indices to ensure they're within bounds
    a_bin = np.clip(a_bin, 0, num_bins - 1)
    b_bin = np.clip(b_bin, 0, num_bins - 1)

    # If the click is already within the gamut, return the original coordinates
    if ab_slice[a_bin, b_bin]:  # Access ab_slice with (a*, b*) order after transposition
        return x, y

    # Find valid points in the current a*, b* slice in the correct (b*, a*) order
    valid_points = np.argwhere(ab_slice)
    valid_a = valid_points[:, 0] * (200 / num_bins) - 100
    valid_b = valid_points[:, 1] * (200 / num_bins) - 100

    # Calculate Euclidean distances in a*, b* space to the clicked point
    distances = np.sqrt((valid_b - x) ** 2 + (valid_a - y) ** 2)
    nearest_index = np.argmin(distances)

    # Return the closest valid a*, b* coordinates
    return valid_b[nearest_index], valid_a[nearest_index]


def parse_args():
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument(
        "--num_control_points", 
        "-n", 
        type=int, 
        default=20, 
        help="Number of control points to choose"
    )
    arg_parser.add_argument(
        "--lightness", 
        "-l", 
        type=str, 
        default="linear", 
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


def create_custom_colormap(
        num_control_points: int = 20, 
        lightness: str = "linear", 
        num_bins: int = 500
    ):
    """Lets the user define control points for a La*b* colormap 
    by iterating through a*b* slices for fixed L values.
    The colormap is interpolated in La*b* space between these control points.

    Args:
        num_control_points (int, optional): Number of control points. Defaults to 20.
        lightness (str, optional): Lightness profile for the colormap. Defaults to "linear".
        num_bins (int, optional): Number of bins for visualization of slices. Defaults to 500.
    """

    assert lightness in SUPPORTED_L_PROFILES, \
        f"Lightness profile not in supported profiles. Valid choices are {SUPPORTED_L_PROFILES}"
    
    L_values = get_lightness_profile(num_control_points, lightness)
    num_bins = 500

    # Initialize for storing clicked points and tracking current L* level
    clicked_points = []
    current_L_index = 0

    # Setup plot
    fig, ax = plt.subplots(figsize=(6, 6))
    image = create_background_image(num_bins)
    implot = ax.imshow(image, extent=[-100, 100, -100, 100], origin="lower")
    ax.set_title("Click to define a control point for each L* level")
    ax.set_xlabel("a* values")
    ax.set_ylabel("b* values")

    a_values = np.linspace(-100, 100, num_bins)
    b_values = np.linspace(-100, 100, num_bins)
    A, B = np.meshgrid(a_values, b_values, indexing='xy')
    ab_values = np.column_stack([A.ravel(), B.ravel()])

    def plot_next_L_slice():
        if current_L_index >= len(L_values):
            plt.close(fig)  # Close the figure once all slices are processed
            return
        
        L = L_values[current_L_index]
        ab_slice = get_ab_occupancy(L, ab_values, num_bins)
        
        # Generate Lab values for this slice
        Lab_slice = np.zeros((num_bins, num_bins, 3))
        Lab_slice[..., 0] = L
        Lab_slice[..., 1] = b_values[None, :]
        Lab_slice[..., 2] = a_values[:, None]
        
        # Convert Lab to RGB for visualization
        XYZ_slice = Lab_to_XYZ(Lab_slice)
        RGB_slice = XYZ_to_sRGB(XYZ_slice)
        RGB_slice = np.clip(RGB_slice, 0, 1)
        
        # Update background with RGB values where occupancy is True
        image = create_background_image(num_bins)
        image[ab_slice] = RGB_slice[ab_slice]
        
        # Display the current slice
        implot.set_data(image)
        ax.set_title(f"a*b* slice at L* = {L:.1f}")
        plt.draw()

    """I really don't like the design of matplotlib, 
       but we're going to have to live with this ugly nested function."""
    def onclick(event):
        nonlocal current_L_index
        if event.inaxes == ax:

            x, y = event.xdata, event.ydata
            L = L_values[current_L_index]
            
            # Project click onto gamut if outside
            ab_slice = get_ab_occupancy(L, ab_values, num_bins)
            x, y = project_onto_gamut(x, y, ab_slice, num_bins)
            
            clicked_points.append((x, y))
            ax.plot(x, y, 'ro', markersize=5)
            plt.draw()
            
            # Move to the next L* slice
            current_L_index += 1
            plot_next_L_slice()

            
    # Initial plot for the first L slice
    plot_next_L_slice()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # Save the clicked points after closing the figure
    clicked_points_array = np.array(clicked_points)
    points_dict = {
        "points": clicked_points_array.tolist(),
        "lightness": lightness
    }

    num_points = 1000
    lab_points = interpolate_lab(clicked_points_array, num_values=num_points, profile=lightness)
    lab_points[:, 1:] = gaussian_filter1d(lab_points[:, 1:], sigma=num_points * 0.03, axis=0)
    rgb_points = rgb_renormalized_lightness(lab_points)

    cdict = dict()
    space = np.linspace(0, 1, num_points)

    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [[space[i], rgb_points[i][num], rgb_points[i][num]] for i in range(num_points)]
        cdict[col] = col_list

    colormap = LinearSegmentedColormap("custom_colormap", segmentdata=cdict, N=num_points)
    plot_colormap(colormap, num_points)

    # Prompt user for colormap name
    cmap_name = input("\nEnter desired colormap name. Quit with 'q'.\nControl points are saved in your local app registry.\nColormap name: ")
    if cmap_name.strip() == 'q':
        return
    elif len(cmap_name.strip()) == 0:
        cmap_name = "custom_cmap"

    save_data(points_dict, cmap_name)


if __name__ == "__main__":
    args = parse_args()
    create_custom_colormap(args.num_control_points, args.lightness)