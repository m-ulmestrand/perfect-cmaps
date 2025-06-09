import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import beta
from scipy.interpolate import interp1d
from typing import Tuple, Union
from colour import sRGB_to_XYZ, XYZ_to_Lab, Lab_to_XYZ, XYZ_to_sRGB
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import math

from perfect_cmaps.optimization import genetic_algorithm
from perfect_cmaps.storage import get_test_img_path


RGB_WEIGHT = np.array([0.2989, 0.5870, 0.1140])
SUPPORTED_L_PROFILES = [
    "linear", 
    "diverging",
    "diverging_inverted",
    "diverging_sharper",
    "diverging_inverted_sharper",
    "flat"
]


def diverging_envelope(x: np.ndarray, c: float = 4, x1: float = 0.25) -> np.ndarray:
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


def dying_cos(x: np.ndarray, offset: float = 0.0):
    result = np.zeros_like(x)
    values = np.mod(np.round(x - offset), 2) == 0
    result[values] = (1 + np.cos(2*np.pi*(x[values]-offset)))/2
    return result


def unwrap_channels(ijk: Union[Tuple[int, int, int], None] = None) -> tuple:
    if ijk is None:
        ijk = (0, 1, 2)
    
    return ijk


def linearize_rgba(rgba_colors: np.ndarray) -> np.ndarray:
    lab_colors = XYZ_to_Lab(sRGB_to_XYZ(rgba_colors[:, :3]))
    lab_colors[:, 0] = np.linspace(0, 100, lab_colors.shape[0])
    rgb_colors_renormalized = Lab_to_sRGB(lab_colors)
    rgb_colors_renormalized = np.concatenate(
        (rgb_colors_renormalized, rgba_colors[:, 3:4]), axis=1
    )
    return rgb_colors_renormalized


def cold_blooded(x: np.ndarray, ijk: Union[Tuple[int, int, int], None] = None) -> np.ndarray:
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


def cold_blooded_l(x: np.ndarray, ijk: Union[Tuple[int, int, int], None] = None) -> np.ndarray:
    normal_colors = cold_blooded(x, ijk)
    return linearize_rgba(normal_colors)


def copper_salt(x: np.ndarray, ijk: Union[Tuple[int, int, int], None] = None) -> np.ndarray:
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


def get_lightness_profile(n_values: int, profile: str = "linear") -> np.ndarray:
    assert profile in SUPPORTED_L_PROFILES, \
        f"Lightness profile not in supported profiles. Valid choices are {SUPPORTED_L_PROFILES}"
    
    max_L = 99.5
    min_L = 0.5

    if profile == "linear":
        L_values = np.linspace(min_L, max_L, n_values)
    elif profile.startswith("diverging"):
        sqrt_range = math.sqrt(max_L - min_L)
        linspace = np.linspace(-sqrt_range, sqrt_range, n_values)
        quadratic_profile = linspace ** 2 + min_L

        if profile == "diverging":
            L_values = 100 - quadratic_profile

        elif profile == "diverging_sharper":
            L_values = 100 - (quadratic_profile + np.abs(linspace) * sqrt_range + min_L) / 2

        elif profile == "diverging_inverted":
            L_values = quadratic_profile
        
        elif profile == "diverging_inverted_sharper":
            L_values = (quadratic_profile + np.abs(linspace) * sqrt_range + min_L) / 2

    elif profile == "flat":
        # Use a flat lightness profile, e.g., L = 50
        L_values = np.full(n_values, 50)
    else:
        # Default to linear
        L_values = np.linspace(min_L, max_L, n_values)
    
    return L_values


def interpolate_lab(
    control_points: np.ndarray, 
    num_values: int = 1000, 
    profile: str = "linear", 
    interpolation: str = "quadratic"
) -> np.ndarray:
    # Interpolate between the control points
    lab_colors = np.zeros((num_values, 3))
    space = np.linspace(0, 1, control_points.shape[0])
    for i in range(2):
        interpolator = interp1d(space, control_points[:, i], kind=interpolation)
        lab_colors[:, i + 1] = interpolator(np.linspace(0, 1, num_values))

    lab_colors[:, 0] = get_lightness_profile(num_values, profile)
    return lab_colors


def rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB values to grayscale using luminosity values of RGB channels."""
    return np.sqrt(np.dot(rgb[...,:3] ** 2, RGB_WEIGHT))


def rgb_to_grayscale_lightness(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB values to grayscale values representing perceived lightness (L*).

    Parameters
    ----------
    rgb : ndarray
        An array of RGB values in sRGB color space, with values in the range [0, 1].

    Returns
    -------
    L_normalized : ndarray
        Grayscale values representing the L* component, normalized to [0, 1].
    """
    # Ensure RGB values are within [0, 1]
    rgb = np.clip(rgb, 0, 1)

    # Convert sRGB to XYZ
    XYZ = sRGB_to_XYZ(rgb)

    # Convert XYZ to CIELAB
    Lab = XYZ_to_Lab(XYZ)

    # Extract the L* component
    L = Lab[..., 0]  # L* ranges from 0 to 100

    # Normalize L* to [0, 1]
    L_normalized = L / 100.0

    return L_normalized


def truncate_colormap(
        cmap: mcolors.LinearSegmentedColormap, 
        minval: float = 0.0, 
        maxval: float = 1.0, 
        n: int = 1000
    ) -> mcolors.LinearSegmentedColormap:

    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def Lab_to_sRGB(lab_colors: np.ndarray) -> np.ndarray:
    """
    Convert CIE-LAB colors to sRGB color space.
    
    Parameters
    ----------
    lab_colors : ndarray
        An array of Lab colors with shape (..., 3).
    
    Returns
    -------
    rgb_colors : ndarray
        An array of sRGB colors with values in [0, 1].
    """
    # Convert Lab to XYZ
    XYZ = Lab_to_XYZ(lab_colors)
    
    # Convert XYZ to sRGB
    RGB = XYZ_to_sRGB(XYZ)
    
    return RGB


def find_valid_L_range(a_star: float, b_star: float) -> Tuple[float, float]:
    # Vectorized L* samples
    L_samples = np.linspace(0, 100, 100)
    # Create an array of Lab colors
    lab = np.column_stack((L_samples, np.full_like(L_samples, a_star), np.full_like(L_samples, b_star)))
    
    # Convert all Lab colors to RGB at once
    rgb = Lab_to_sRGB(lab)
    
    # Determine which colors are in gamut
    in_gamut = np.all((rgb >= 0) & (rgb <= 1), axis=1)
    
    if np.any(in_gamut):
        L_in_gamut = L_samples[in_gamut]
        L_min_valid = L_in_gamut.min()
        L_max_valid = L_in_gamut.max()
    else:
        L_min_valid = None
        L_max_valid = None
    
    return L_min_valid, L_max_valid


def find_valid_L_range_coarse_optim(a_star: float, b_star: float) -> Tuple[float, float]:
    # Coarse sampling
    L_samples_coarse = np.linspace(0, 100, 50)
    lab_coarse = np.column_stack((L_samples_coarse, np.full_like(L_samples_coarse, a_star), np.full_like(L_samples_coarse, b_star)))
    rgb_coarse = Lab_to_sRGB(lab_coarse)
    in_gamut_coarse = np.all((rgb_coarse >= 0) & (rgb_coarse <= 1), axis=1)
    
    if np.any(in_gamut_coarse):
        # Approximate L_min and L_max
        L_in_gamut = L_samples_coarse[in_gamut_coarse]
        L_min_approx = L_in_gamut.min()
        L_max_approx = L_in_gamut.max()
        
        # Fine sampling near L_min_approx
        L_samples_min = np.linspace(max(L_min_approx - 1, 0), L_min_approx, 20)
        lab_min = np.column_stack((L_samples_min, np.full_like(L_samples_min, a_star), np.full_like(L_samples_min, b_star)))
        rgb_min = Lab_to_sRGB(lab_min)
        in_gamut_min = np.all((rgb_min >= 0) & (rgb_min <= 1), axis=1)
        L_min_valid = L_samples_min[in_gamut_min].min() if np.any(in_gamut_min) else L_min_approx
        
        # Fine sampling near L_max_approx
        L_samples_max = np.linspace(L_max_approx, min(L_max_approx + 1, 100), 20)
        lab_max = np.column_stack((L_samples_max, np.full_like(L_samples_max, a_star), np.full_like(L_samples_max, b_star)))
        rgb_max = Lab_to_sRGB(lab_max)
        in_gamut_max = np.all((rgb_max >= 0) & (rgb_max <= 1), axis=1)
        L_max_valid = L_samples_max[in_gamut_max].max() if np.any(in_gamut_max) else L_max_approx
    else:
        L_min_valid = None
        L_max_valid = None
    
    return L_min_valid, L_max_valid


def rgb_renormalized_lightness(
    lab_colors: np.ndarray, 
    population_size: int = 100, 
    num_generations: int = 200,
    verbose: bool = False,
) -> np.ndarray:
    
    # Compute L_min and L_max for each color
    n_colors = lab_colors.shape[0]
    L_min = np.zeros(n_colors)
    L_max = L_min.copy()

    for i in range(n_colors):
        a_star = lab_colors[i, 1]
        b_star = lab_colors[i, 2]
        # Compute L_min[i] and L_max[i] for this (a*, b*) pair
        L_min[i], L_max[i] = find_valid_L_range(a_star, b_star)

    L_intended = lab_colors[:, 0]

    # Optimize m and c
    gene_limits = np.array([[0.0, 100.0], [-100.0, 100.0]])
    genes, best_fitness = genetic_algorithm(population_size, num_generations, gene_limits, L_intended, L_min, L_max)

    if best_fitness == 0.0:
        print("Optimization failed for chosen lightness profile.")
        print("Colormap is not perfectly perceptually uniform.")
        return Lab_to_sRGB(lab_colors), 1.0, 0.0
    
    if verbose:
        print(f"Optimized colormap lightness range: {100 * best_fitness:.2f} %")

    L_adjusted = L_intended * genes[0] + genes[1]

    # Update L* values in lab_colors
    lab_colors[:, 0] = L_adjusted

    # Convert adjusted Lab colors to RGB
    return Lab_to_sRGB(lab_colors)


def plot_colormap(colormap: LinearSegmentedColormap, num_points: int = 1000):
    gradient = np.linspace(0, 1, num_points)
    gradient = np.vstack((gradient, gradient))

    # Get RGB values from colormap
    gradient_rgb = colormap(gradient)

    # Convert the RGB values to grayscale
    lightness = rgb_to_grayscale_lightness(gradient_rgb[0][:, :3])
    luminance = rgb_to_grayscale(gradient_rgb[0][:, :3])
    gradient_gray = np.vstack((lightness, lightness))

    # Plotting
    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)

    # Define the axes
    ax0 = fig.add_subplot(gs[0, 0])  # Top-left subplot
    ax1 = fig.add_subplot(gs[1, 0])  # Mid-left subplot
    ax2 = fig.add_subplot(gs[2, 0])  # Bottom-left subplot
    ax3 = fig.add_subplot(gs[:, 1])  # Right subplot spanning all rows

    # Plot RGB gradient on ax0
    ax0.imshow(gradient_rgb, aspect="auto", vmin=0.0, vmax=1.0)
    ax0.set_title("RGB gradient", fontsize=20)
    ax0.axis("off")

    # Plot Grayscale gradient on ax1
    ax1.imshow(gradient_gray, cmap="gray", aspect="auto", vmin=0.0, vmax=1.0)
    ax1.set_title("Lightness gradient", fontsize=20)
    ax1.axis("off")

    test_image_path = get_test_img_path()
    colormap_test_img_path = test_image_path / "colourmaptest.tif"
    colormap_test_image = plt.imread(colormap_test_img_path)
    ax2.imshow(colormap_test_image, cmap=colormap)
    ax2.set_title("Colormap test image", fontsize=20)
    ax2.axis("off")

    # Plot the color channels and luminance on ax2
    for c, color_string in zip(range(3), ["red", "green", "blue"]):
        ax3.plot(gradient_rgb[0, :, c], color=color_string, label=color_string, linewidth=4)

    ax3.plot(luminance, color="black", linestyle="-.", label="luminance")
    ax3.plot(lightness, color="grey", label="lightness", linewidth=4)
    ax3.set_title("Color channel intensities", fontsize=20)
    ax3.legend(loc=0, fontsize=20)
    plt.show()


CMAP_DICT = {
    "cold_blooded": cold_blooded,
    "cold_blooded_l": cold_blooded_l,
    "ectotherm": cold_blooded,
    "ectotherm_l": cold_blooded_l,
    "copper_salt": copper_salt
}