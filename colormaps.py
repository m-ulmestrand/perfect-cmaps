import matplotlib.colors as mcolors
from skimage import color

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import beta
from scipy.interpolate import interp1d
from matplotlib import colormaps
from pathlib import Path
from typing import Tuple
import json


def load_json_points(cmap_name: str, n: int):
    file_path = Path(__file__).parent / "lab_control_points" / Path(cmap_name).name
    with open(file_path) as json_file:
        lab_control_points = json.load(json_file)["points"]
    
    lab_control_points = np.array(lab_control_points)
    return convert_from_lab(lab_control_points, n)


def get_continuous_cmap(cmap_name: str, n: int, ijk: Tuple | None = None):
    space = np.linspace(0, 1, n)

    if cmap_name.endswith(".json"):
        vals = load_json_points(cmap_name, n)
    else:
        vals = CMAP_DICT[cmap_name](space, ijk)
    
    cdict = dict()

    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [[space[i], vals[i][num], vals[i][num]] for i in range(n)]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap(cmap_name, segmentdata=cdict, N=n)
    return cmp


def diverging_envelope(x: np.ndarray, c=4, x1=0.25):
    result = np.zeros(x.shape)
    m = x1 * (c - 2) / (1 - 2 * x1 + 1e-5)

    below_x1 = x <= x1
    result[below_x1] = c*x[below_x1]

    before_middle = np.logical_and(x > x1, x <= 0.5)
    result[before_middle] = m + 2*(1-m) * x[before_middle]

    after_middle = np.logical_and(x > 0.5, x <= 1-x1)
    result[after_middle] = 2 - m + 2*(m-1) * x[after_middle]

    in_middle = np.logical_and(x > x1, x <= 1 - x1)
    result[in_middle] = result.max()

    after_x2 = x > 1-x1
    result[after_x2] = c * (1 - x[after_x2])
    return result


def dying_cos(x, offset: float = 0.0):
    result = np.zeros_like(x)
    values = np.mod(np.round(x - offset), 2) == 0
    result[values] = (1 + np.cos(2*np.pi*(x[values]-offset)))/2
    return result


def unwrap_channels(ijk: Tuple | None = None):
    if ijk is None:
        ijk = (0, 1, 2)
    
    return ijk


def cold_blood(x, ijk: Tuple | None = None):
    i, j, k = unwrap_channels(ijk)

    result = np.zeros([*x.shape, 4])
    period = 2*np.pi
    result[..., j] = x**2 - x/period * np.sin(period*x)
    result[..., k] = ((1 - np.cos(period/2*x))/2)**2
    result[..., i] = 3*x**2 - result[..., j] - result[..., k]
    result[..., :3] = np.sqrt(result[..., :3])

    result[..., :3] /= np.max(np.max(result))
    result[..., 3] = 1
    
    return result


def copper_salt(x, ijk: Tuple | None = None):
    i, j, k = unwrap_channels(ijk)

    result = np.zeros([*x.shape, 4])
    dist = beta(2, 4)
    pdf = dist.pdf
    envelope = diverging_envelope(x, c=4, x1=0.5)
    result[..., i] = pdf(x)
    result[..., k] = pdf(1-x)
    result[..., i] /= np.max(result[..., i])
    result[..., k] /= np.max(result[..., k])
    result[..., j] = np.sqrt(3 * envelope - result[..., i] ** 2 - result[..., k] ** 2)
    weight = 1 / np.sqrt(RGB_WEIGHT)
    
    result[..., :3] = np.einsum(
        'i...j,k...j->i...j', 
        result[..., :3],
        weight.reshape(1, *weight.shape)
    )
    result[..., :3] /= np.max(result)
    result[..., 3] = 1
    return result


def convert_from_lab(control_points: np.ndarray, num_values: int):
    # Interpolate between the control points
    lab_colors = np.zeros((num_values, 3))
    for i in range(3):
        interpolator = interp1d(np.linspace(0, 1, control_points.shape[0]), control_points[:, i], kind='cubic')
        lab_colors[:, i] = interpolator(np.linspace(0, 1, num_values))

    # Convert the LAB colors to RGB
    return color.lab2rgb(lab_colors)


def rgb_to_grayscale(rgb: np.ndarray):
    """Convert RGB values to grayscale using luminosity values of RGB channels."""
    return np.sqrt(np.dot(rgb[...,:3] ** 2, RGB_WEIGHT))


def truncate_colormap(cmap: mcolors.LinearSegmentedColormap, minval: float = 0.0, maxval: float = 1.0, n: int = 1000):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def get_custom_cmap(name: str = "cold_blood", n: int = 1000, ijk: tuple = None):
    cmp = get_continuous_cmap(name, n, ijk)
    return cmp


RGB_WEIGHT = np.array([0.2989, 0.5870, 0.1140])
CMAP_DICT = {
    "cold_blood": cold_blood,
    "copper_salt": copper_salt
}


if __name__ == "__main__":
    n = 1000
    cmap = get_custom_cmap("copper_salt", n)
    # cmap = colormaps["magma"]

    gradient = np.linspace(0, 1, n)
    gradient = np.vstack((gradient, gradient))

    # Get RGB values from colormap
    gradient_rgb = cmap(gradient)

    # Convert the RGB values to grayscale
    gradient_gray = rgb_to_grayscale(gradient_rgb)

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

    # Plot RGB gradient
    axes[0].imshow(gradient_rgb, aspect='auto', vmin=0.0, vmax=1.0)
    axes[0].set_title('RGB Gradient')
    axes[0].axis('off')

    # Plot Grayscale gradient
    axes[1].imshow(gradient_gray, cmap='gray', aspect='auto', vmin=0.0, vmax=1.0)
    axes[1].set_title('Grayscale Gradient')
    axes[1].axis('off')

    for c, color_string in zip(range(3), ["red", "green", "blue"]):
        axes[2].plot(gradient_rgb[0, :, c], color=color_string)
    
    luminance = rgb_to_grayscale(gradient_rgb[0])
    axes[2].plot(luminance, color="black")

    plt.show()

    for c, color_string in zip(range(3), ["red", "green", "blue"]):
        plt.plot(gradient_rgb[0, :, c], color=color_string)
    
    luminance = rgb_to_grayscale(gradient_rgb[0])
    plt.plot(luminance, color="black")
    plt.show()

    dem_data = plt.imread("/home/mattias/imgs/DEM3.gif")
    
    if len(dem_data.shape) == 3:
        dem_data = dem_data[:, :, 0]
        
    plt.imshow(dem_data, cmap=cmap)
    plt.colorbar()
    plt.show()