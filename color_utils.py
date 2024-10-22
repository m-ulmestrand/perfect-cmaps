import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import beta
from scipy.interpolate import interp1d
from pathlib import Path
from typing import Tuple
import json
from colour import sRGB_to_XYZ, XYZ_to_Lab, Lab_to_XYZ, XYZ_to_sRGB
from scipy.optimize import bisect
from scipy.optimize import linprog


RGB_WEIGHT = np.array([0.2989, 0.5870, 0.1140])
SUPPORTED_L_PROFILES = [
    "linear", 
    "diverging",
    "diverging_inverted",
    "flat"
]


def load_json(cmap_name: str, n: int):
    file_path = Path(__file__).parent / "lab_control_points" / Path(cmap_name).name

    # Adding .with_suffix() so the user can provide without the suffix
    file_path = file_path.with_suffix(".json")
    try:
        with open(file_path) as json_file:
            return json.load(json_file)
    
    except FileNotFoundError:
        print(f"Colormap {cmap_name} not found in control point files.")
        exit(1)


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


def get_lightness_profile(n_values: int, profile: str = "linear"):
    if profile == "linear":
        L_values = np.linspace(0, 100, n_values)
    elif profile == "diverging":
        L_values = 100 - np.linspace(-10, 10, n_values) ** 2
    elif profile == "diverging_inverted":
        L_values = np.linspace(-10, 10, n_values) ** 2
    elif profile == "flat":
        # Use a flat lightness profile, e.g., L = 50
        L_values = np.full(n_values, 50)
    else:
        # Default to linear
        L_values = np.linspace(0, 100, n_values)
    
    return L_values


def interpolate_lab(control_points: np.ndarray, num_values: int = 1000, profile: str = "linear"):
    # Interpolate between the control points
    lab_colors = np.zeros((num_values, 3))
    space = np.linspace(0, 1, control_points.shape[0])
    for i in range(2):
        interpolator = interp1d(space, control_points[:, i], kind='cubic')
        lab_colors[:, i + 1] = interpolator(np.linspace(0, 1, num_values))

    lab_colors[:, 0] = get_lightness_profile(num_values, profile)
    return lab_colors


def rgb_to_grayscale(rgb: np.ndarray):
    """Convert RGB values to grayscale using luminosity values of RGB channels."""
    return np.sqrt(np.dot(rgb[...,:3] ** 2, RGB_WEIGHT))


def rgb_to_grayscale_lightness(rgb: np.ndarray):
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


def truncate_colormap(cmap: mcolors.LinearSegmentedColormap, minval: float = 0.0, maxval: float = 1.0, n: int = 1000):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def Lab_to_sRGB(lab_colors: np.ndarray):
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


def find_valid_L_range_slow(a_star: float, b_star: float, L_initial: int = 50):
    """
    Find the valid L* range for a given a* and b* such that the Lab color
    maps to valid RGB colors without clipping.

    Parameters
    ----------
    a_star : float
        The a* component of Lab color.
    b_star : float
        The b* component of Lab color.

    Returns
    -------
    L_min : float
        The minimum L* value.
    L_max : float
        The maximum L* value.
    """
    # Define functions to check if Lab(L*, a*, b*) maps to valid RGB
    def is_valid_rgb(L):
        lab = np.array([L, a_star, b_star])
        rgb = Lab_to_sRGB(lab)
        return np.all((rgb >= 0) & (rgb <= 1))
    
    # Find L_min
    L_min = 0
    L_max_possible = 100
    if is_valid_rgb(L_min):
        L_min_valid = L_min
    else:
        # Use bisection to find L_min_valid
        L_min_valid = bisect(lambda L: is_valid_rgb(L) - 0.5, L_min, L_initial)
    
    # Find L_max
    if is_valid_rgb(L_max_possible):
        L_max_valid = L_max_possible
    else:
        # Use bisection to find L_max_valid
        L_max_valid = bisect(lambda L: is_valid_rgb(L) - 0.5, L_initial, L_max_possible)
    
    return L_min_valid, L_max_valid


def find_valid_L_range(a_star: float, b_star: float):
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


def find_valid_L_range_coarse_optim(a_star: float, b_star: float):
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


def optimize_parameters(L_intended: np.ndarray, L_min: float, L_max: float, lightness_profile: str | None = None):
    n = len(L_intended)
    optimize_c = lightness_profile != "flat"

    if optimize_c:
        # Objective function: minimize -m (maximize m)
        c_obj = [-1, 0]  # Coefficients for variables [m, c]

        # Inequality constraints: A_ub * x <= b_ub
        A_ub = np.zeros((2 * n, 2))  # Two variables: m and c
        b_ub = np.zeros(2 * n)

        for i in range(n):
            # Constraint: -L_intended_i * m - c <= -L_min_i
            A_ub[2 * i] = [-L_intended[i], -1]
            b_ub[2 * i] = -L_min[i]

            # Constraint: L_intended_i * m + c <= L_max_i
            A_ub[2 * i + 1] = [L_intended[i], 1]
            b_ub[2 * i + 1] = L_max[i]

        # Variable bounds
        bounds = [
            (0, 1),      # m >= 0
            (-100, 100)  # c between -100 and 100
        ]
    else:
        # Objective function: minimize -m (maximize m)
        c_obj = [-1]  # Coefficient for variable m

        # Inequality constraints: A_ub * x <= b_ub
        A_ub = np.zeros((2 * n, 1))  # One variable: m
        b_ub = np.zeros(2 * n)

        for i in range(n):
            # Constraint: -L_intended_i * m <= -L_min_i
            A_ub[2 * i] = [-L_intended[i]]
            b_ub[2 * i] = -L_min[i]

            # Constraint: L_intended_i * m <= L_max_i
            A_ub[2 * i + 1] = [L_intended[i]]
            b_ub[2 * i + 1] = L_max[i]

        # Variable bounds
        bounds = [
            (0, 2)  # m >= 0
        ]

        c_opt = 0  # c is fixed to zero

    # Solve the linear programming problem
    res = linprog(
        c=c_obj,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method='highs'
    )

    if res.success:
        m_opt = res.x[0]
        if optimize_c:
            c_opt = res.x[1]
        print(f"Optimal m: {m_opt}, Optimal c: {c_opt}")
        L_adjusted = m_opt * L_intended + c_opt
        return L_adjusted, m_opt, c_opt
    else:
        raise ValueError("Optimization infeasible for the chosen lightness profile")


def rgb_renormalized_lightness(lab_colors: np.ndarray, lightness_profile: str | None = None):
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
    L_adjusted, m_opt, c_opt = optimize_parameters(L_intended, L_min, L_max, lightness_profile)

    # Update L* values in lab_colors
    lab_colors[:, 0] = L_adjusted

    # Convert adjusted Lab colors to RGB
    return Lab_to_sRGB(lab_colors), m_opt, c_opt