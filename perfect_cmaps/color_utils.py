import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import beta
from scipy.interpolate import interp1d
from scipy.optimize import linprog
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from colour import sRGB_to_XYZ, XYZ_to_Lab, Lab_to_XYZ, XYZ_to_sRGB
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import math

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


# Margin (in L* units) kept from the gamut boundaries found by lightness_envelope
LIGHTNESS_MARGIN = 1e-6

# Lightness range spanned by every non-flat profile from get_lightness_profile
PROFILE_MIN_L = 0.5
PROFILE_MAX_L = 99.5

DEFAULT_L_BOUNDS = (0.0, 100.0)

LightnessBounds = Union[float, Tuple[float, float], None]

# "relative": bounds are fractions (0-1) of the reachable range of l_dark / l_light.
# "absolute": bounds are L* values (0-100).
BOUNDS_MODES = ("relative", "absolute")
DEFAULT_BOUNDS_MODE = "relative"


def normalize_lightness_bounds(
    bounds: LightnessBounds,
    name: str = "bounds",
    mode: str = DEFAULT_BOUNDS_MODE,
) -> Tuple[float, float]:
    """Turn a user-facing lightness bound into a (low, high) tuple in the given mode.
    ``None`` means unconstrained and a single number fixes the value."""
    if mode not in BOUNDS_MODES:
        raise ValueError(f"Bounds mode must be one of {BOUNDS_MODES}, got '{mode}'")
    upper = 1.0 if mode == "relative" else 100.0
    if bounds is None:
        return 0.0, upper
    if isinstance(bounds, (int, float, np.integer, np.floating)):
        low = high = float(bounds)
    else:
        low, high = (float(b) for b in bounds)
    if not 0.0 <= low <= high <= upper:
        raise ValueError(f"{name} must satisfy 0 <= low <= high <= {upper:g} for {mode} bounds, got ({low}, {high})")
    return low, high


def normalized_lightness_profile(n_values: int, profile: str = "linear") -> np.ndarray:
    """The lightness profile mapped to [0, 1], where 0 is the darkest and 1 the
    lightest point. The flat profile has no dark or light points and maps to zeros."""
    if profile == "flat":
        return np.zeros(n_values)
    L_values = get_lightness_profile(n_values, profile)
    return (L_values - PROFILE_MIN_L) / (PROFILE_MAX_L - PROFILE_MIN_L)


@dataclass(frozen=True)
class LightnessFit:
    """Lightness of the darkest and lightest point of a colormap.
    The lightness at normalized profile value p is ``l_dark + (l_light - l_dark) * p``.
    For the flat profile, ``l_dark == l_light`` is the constant lightness."""
    l_dark: float
    l_light: float
    feasible: bool

    @property
    def lightness_range(self) -> float:
        return self.l_light - self.l_dark

    def to_dict(self) -> dict:
        return {"l_dark": self.l_dark, "l_light": self.l_light, "feasible": self.feasible}

    @classmethod
    def from_dict(cls, data: dict) -> "LightnessFit":
        return cls(float(data["l_dark"]), float(data["l_light"]), bool(data["feasible"]))

    @classmethod
    def fallback(cls, flat: bool) -> "LightnessFit":
        """The unoptimized profile, used when no in-gamut fit exists."""
        if flat:
            return cls(50.0, 50.0, False)
        return cls(PROFILE_MIN_L, PROFILE_MAX_L, False)


# Number of evenly spaced L* samples used to bracket the gamut edges in lightness_envelope.
# A color whose in-gamut L* interval is narrower than the spacing may be missed.
ENVELOPE_COARSE_SAMPLES = 100
ENVELOPE_BISECTION_STEPS = 40


def _in_gamut(L: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    rgb = Lab_to_sRGB(np.column_stack((L, a, b)))
    return np.all((rgb >= 0.0) & (rgb <= 1.0), axis=1)


def lightness_envelope(lab_colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For each color, the range of L* for which its (a*, b*) is inside the sRGB gamut.
    Colors without any in-gamut lightness get NaN. Only depends on a* and b*.

    The edges are first bracketed by sampling L* evenly, then refined by bisection,
    for all colors at once."""
    a = lab_colors[:, 1]
    b = lab_colors[:, 2]
    n_colors = lab_colors.shape[0]
    n_samples = ENVELOPE_COARSE_SAMPLES

    L_samples = np.linspace(0.0, 100.0, n_samples)
    inside = _in_gamut(
        np.repeat(L_samples, n_colors), np.tile(a, n_samples), np.tile(b, n_samples)
    ).reshape(n_samples, n_colors)

    has_gamut = inside.any(axis=0)
    i_low = inside.argmax(axis=0)                                # first in-gamut sample
    i_high = n_samples - 1 - inside[::-1].argmax(axis=0)         # last in-gamut sample

    def bisect(L_inside: np.ndarray, L_outside: np.ndarray) -> np.ndarray:
        """Move L_inside towards the gamut edge between L_inside and L_outside."""
        for _ in range(ENVELOPE_BISECTION_STEPS):
            L_mid = (L_inside + L_outside) / 2
            mid_inside = _in_gamut(L_mid, a, b)
            L_inside = np.where(mid_inside, L_mid, L_inside)
            L_outside = np.where(mid_inside, L_outside, L_mid)
        return L_inside

    # Edges at L* = 0 or 100 need no refinement
    L_min = np.where(i_low > 0, bisect(L_samples[i_low], L_samples[np.maximum(i_low - 1, 0)]), L_samples[i_low])
    L_max = np.where(
        i_high < n_samples - 1,
        bisect(L_samples[i_high], L_samples[np.minimum(i_high + 1, n_samples - 1)]),
        L_samples[i_high],
    )
    return np.where(has_gamut, L_min, np.nan), np.where(has_gamut, L_max, np.nan)


class LightnessBoundsError(ValueError):
    """Raised when the requested lightness bounds can't be met within the sRGB gamut."""


def _gamut_rows(p_hat: np.ndarray, L_min: np.ndarray, L_max: np.ndarray):
    """Constraints L_min + margin <= d * (1 - p) + l * p <= L_max - margin, as A @ (d, l) <= b."""
    A = np.vstack([
        np.column_stack((-(1.0 - p_hat), -p_hat)),
        np.column_stack((1.0 - p_hat, p_hat)),
    ])
    b = np.concatenate((-(L_min + LIGHTNESS_MARGIN), L_max - LIGHTNESS_MARGIN))
    return A, b


def _solve_range(p_hat, L_min, L_max, dark_bounds, light_bounds) -> Optional[Tuple[float, float]]:
    """Maximize l - d such that every color stays in the gamut. Returns None if infeasible.

    The optimum is unique in practice: at maximal range, the gamut constraints are
    tight at both ends, so the range can't slide up or down."""
    A_gamut, b_gamut = _gamut_rows(p_hat, L_min, L_max)
    A = np.vstack([A_gamut, [1.0, -1.0]])  # d <= l
    b = np.concatenate([b_gamut, [0.0]])

    result = linprog(c=[1.0, -1.0], A_ub=A, b_ub=b, bounds=[dark_bounds, light_bounds], method="highs")
    if not result.success:
        return None
    d, l = result.x
    return float(d), float(l)


def _flat_interval(L_min, L_max, dark_bounds, light_bounds) -> Tuple[float, float]:
    low = max(np.max(L_min) + LIGHTNESS_MARGIN, dark_bounds[0], light_bounds[0])
    high = min(np.min(L_max) - LIGHTNESS_MARGIN, dark_bounds[1], light_bounds[1])
    return low, high


def _reachable_ranges(p_hat, L_min, L_max) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """The reachable intervals of l_dark and l_light without user bounds."""
    A_gamut, b_gamut = _gamut_rows(p_hat, L_min, L_max)
    A = np.vstack([A_gamut, [1.0, -1.0]])
    b = np.concatenate([b_gamut, [0.0]])
    bounds = [DEFAULT_L_BOUNDS, DEFAULT_L_BOUNDS]
    extremes = []
    for c in ([1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]):
        result = linprog(c=c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        extremes.append(float(np.dot(np.abs(c), result.x)))
    return (extremes[0], extremes[1]), (extremes[2], extremes[3])


def _relative_to_absolute(bounds: Tuple[float, float], reach: Tuple[float, float]) -> Tuple[float, float]:
    span = reach[1] - reach[0]
    return reach[0] + bounds[0] * span, reach[0] + bounds[1] * span


def _format_bounds(bounds: Tuple[float, float]) -> str:
    if bounds[0] == bounds[1]:
        return f"{bounds[0]:.4g}"
    return f"({bounds[0]:.4g}, {bounds[1]:.4g})"


def solve_lightness(
    p_hat: np.ndarray,
    L_min: np.ndarray,
    L_max: np.ndarray,
    flat: bool = False,
    l_dark: LightnessBounds = None,
    l_light: LightnessBounds = None,
    bounds: str = DEFAULT_BOUNDS_MODE,
) -> LightnessFit:
    """Find the lightness of the darkest and lightest point of a colormap.

    For non-flat profiles, the lightness range ``l_light - l_dark`` is maximized while
    keeping every color in the sRGB gamut. For the flat profile, the middle of the feasible
    lightness interval is chosen.

    Args:
        p_hat: Normalized lightness profile, see ``normalized_lightness_profile``.
        L_min, L_max: Gamut envelope, see ``lightness_envelope``.
        flat: Whether the profile is flat.
        l_dark, l_light: Optional bounds on the lightness of the darkest and lightest
            point, as a (low, high) tuple or a single fixed value.
        bounds: How ``l_dark`` and ``l_light`` are interpreted. With "relative", they are
            fractions (0-1) of the reachable range of each value on its own: 0 is the
            lowest and 1 the highest lightness that value can take within the gamut.
            With "absolute", they are L* values (0-100).

    Raises:
        LightnessBoundsError: If the colormap fits in the gamut, but not within the given bounds.
    """
    requested_dark = normalize_lightness_bounds(l_dark, "l_dark", bounds)
    requested_light = normalize_lightness_bounds(l_light, "l_light", bounds)
    constrained = l_dark is not None or l_light is not None

    if np.any(np.isnan(L_min)):
        # Some (a*, b*) pair has no in-gamut lightness at all
        return LightnessFit.fallback(flat)

    # Reachable ranges of l_dark and l_light without user bounds
    if flat:
        flat_reach = _flat_interval(L_min, L_max, DEFAULT_L_BOUNDS, DEFAULT_L_BOUNDS)
        if flat_reach[0] > flat_reach[1]:
            return LightnessFit.fallback(flat)
        dark_reach = light_reach = flat_reach
    else:
        if not constrained:
            solution = _solve_range(p_hat, L_min, L_max, DEFAULT_L_BOUNDS, DEFAULT_L_BOUNDS)
            return LightnessFit(*solution, True) if solution is not None else LightnessFit.fallback(flat)
        if _solve_range(p_hat, L_min, L_max, DEFAULT_L_BOUNDS, DEFAULT_L_BOUNDS) is None:
            return LightnessFit.fallback(flat)
        dark_reach, light_reach = _reachable_ranges(p_hat, L_min, L_max)

    if bounds == "relative":
        dark_bounds = _relative_to_absolute(requested_dark, dark_reach)
        light_bounds = _relative_to_absolute(requested_light, light_reach)
    else:
        dark_bounds, light_bounds = requested_dark, requested_light

    if flat:
        low, high = _flat_interval(L_min, L_max, dark_bounds, light_bounds)
        if low <= high:
            level = (low + high) / 2
            return LightnessFit(level, level, True)
    else:
        solution = _solve_range(p_hat, L_min, L_max, dark_bounds, light_bounds)
        if solution is not None:
            return LightnessFit(solution[0], solution[1], True)

    message = (
        f"The colormap can't stay in gamut with {bounds} bounds "
        f"l_dark={_format_bounds(requested_dark)}, l_light={_format_bounds(requested_light)}"
    )
    if bounds == "relative":
        message += f" (absolute: l_dark={_format_bounds(dark_bounds)}, l_light={_format_bounds(light_bounds)})"
    if flat:
        message += f". Reachable flat lightness: {_format_bounds(flat_reach)}."
    else:
        message += (
            f". Reachable absolute values (not necessarily jointly): "
            f"l_dark in {_format_bounds(dark_reach)}, l_light in {_format_bounds(light_reach)}."
        )
    raise LightnessBoundsError(message)


def apply_lightness_fit(lab_colors: np.ndarray, p_hat: np.ndarray, fit: LightnessFit) -> np.ndarray:
    """Set the lightness of Lab colors according to a fit and convert to sRGB.
    Does not modify ``lab_colors``."""
    lab_adjusted = lab_colors.copy()
    lab_adjusted[:, 0] = fit.l_dark + fit.lightness_range * p_hat
    return Lab_to_sRGB(lab_adjusted)


def warn_infeasible():
    print("Optimization failed for chosen lightness profile.")
    print("Colormap is not perfectly perceptually uniform.")


def rgb_renormalized_lightness(
    lab_colors: np.ndarray,
    profile: str,
    l_dark: LightnessBounds = None,
    l_light: LightnessBounds = None,
    bounds: str = DEFAULT_BOUNDS_MODE,
    verbose: bool = False,
) -> np.ndarray:
    """Optimize the lightness of ``lab_colors`` (its a* and b* channels) for the given
    lightness profile and convert to sRGB. The L* channel of ``lab_colors`` is ignored."""
    p_hat = normalized_lightness_profile(lab_colors.shape[0], profile)
    L_min, L_max = lightness_envelope(lab_colors)
    fit = solve_lightness(p_hat, L_min, L_max, profile == "flat", l_dark, l_light, bounds)

    if not fit.feasible:
        warn_infeasible()
    elif verbose:
        print(f"Optimized lightness: {fit.l_dark:.2f} to {fit.l_light:.2f}")

    return apply_lightness_fit(lab_colors, p_hat, fit)


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