import matplotlib.colors as mcolors
from skimage import color

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import beta
from scipy.interpolate import interp1d
from matplotlib import colormaps


rgb_weight = np.array([0.2989, 0.5870, 0.1140])


def get_continuous_cmap(n, name: str, ijk: tuple = None):
    space = np.linspace(0, 1, n)
    if ijk is not None:
        vals = cmap_dict[name](space, *ijk)
    else:
        vals = cmap_dict[name](space)
    cdict = dict()

    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[space[i], vals[i][num], vals[i][num]] for i in range(n)]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap(name, segmentdata=cdict, N=n)
    return cmp


def diverging_envelope(x: np.ndarray, c=4, x1=0.25):
    result = np.zeros(x.shape)
    m = x1 * (c - 2) / (1 - 2 * x1)

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


def cold_blood(x, i: int = 1, j: int = 2, k: int = 0):
    result = np.zeros([*x.shape, 4])
    period = 2*np.pi
    result[..., i] = x**2 - x/period * np.sin(period*x)
    result[..., j] = ((1 - np.cos(period/2*x))/2)**2
    result[..., k] = 3*x**2 - result[..., i] - result[..., j]
    result[..., :3] = np.sqrt(result[..., :3])

    result[..., :3] /= np.max(np.max(result))
    result[..., 3] = 1
    # result = np.hstack((result[:, i:i+1], result[:, j:j+1], result[:, k:k+1]))
    
    return result


def better_rgb(x, i: int = 0, j: int = 1, k: int = 2):
    offset = 2
    intensity = 1
    decay = 4
    I = (1 - np.exp(-decay * x)) * intensity
    I = intensity * x
    result = np.zeros([*x.shape, 4])
    result[..., i] = dying_cos(x, 1-offset)
    result[..., j] = dying_cos(x, 1/2-offset)
    result[..., k] = dying_cos(x, 0-offset)
    result[..., :3] = np.sqrt(result[..., :3])

    weight = 1 / np.sqrt(rgb_weight)
    result[..., :3] = np.einsum(
        'i...j,k...j,i...k->i...j', 
        result[..., :3],
        weight.reshape(1, weight.size), 
        I.reshape(*x.shape, 1)
    )
    result[..., :3] /= np.max(np.max(result))
    result[..., 3] = 1
    return result


def copper_salt(x, i: int = 0, j: int = 1, k: int = 2):
    result = np.zeros([*x.shape, 4])
    dist = beta(2, 4)
    pdf = dist.pdf
    envelope = diverging_envelope(x, c=4, x1=0.45)
    result[..., i] = pdf(x)
    result[..., k] = pdf(1-x)
    result[..., i] /= np.max(result[..., i])
    result[..., k] /= np.max(result[..., k])
    result[..., j] = np.sqrt(3 * envelope - result[..., i] ** 2 - result[..., k] ** 2)
    weight = 1 / np.sqrt(rgb_weight)
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


def rgb_spiral(x: np.ndarray, *args):
    lab_control_points = np.array([
        [   0.,           45.75328685,  -44.59742262],
        [   8.33333333,   61.44635547,  -26.7540478 ],
        [  16.66666667,   65.06783284,   -7.72111466],
        [  25.,           61.04396909,   12.89789624],
        [  33.33333333,   50.98430973,   25.98303778],
        [  41.66666667,   33.68169561,   36.29254323],
        [  50.,           15.17192237,   40.65425708],
        [  58.33333333,   -1.32591899,   41.44729596],
        [  66.66666667,  -16.61660123,   37.08558211],
        [  75.,          -24.66432873,   24.39696002],
        [  83.33333333,  -25.0667151,     9.32922128],
        [  91.66666667,  -18.22614673,    0.20927415],
        [ 100.,            0.,            0.        ]
    ])

    return convert_from_lab(lab_control_points, x.size)


def brg_spiral(x: np.ndarray, *args):
    lab_control_points = np.array([
        [   0.,           27.64589999,  -50.54521423],
        [  10.,           50.58192335,  -40.23570878],
        [  20.,           60.64158272,  -17.63410067],
        [  30.,           57.42249172,    4.57098799],
        [  40.,           43.33896861,   22.41436282],
        [  50.,           23.21964987,   34.30994603],
        [  60.,           -2.53307812,   39.06817931],
        [  70.,          -22.25001048,   31.53430995],
        [  80.,          -26.27387423,   10.91529904],
        [  90.,          -15.00705573,    0.60579359],
        [ 100.,            0.,            0.        ]
    ])

    return convert_from_lab(lab_control_points, x.size)


def rbg_spiral(x: np.ndarray, *args):
    lab_control_points = np.array([
        [   0.,           71.10362846,   59.68719022],
        [  11.11111111,   63.45828734,   26.37955722],
        [  22.22222222,   52.59385522,  -17.23758123],
        [  33.33333333,   21.20771799,  -37.85659214],
        [  44.44444444,  -17.82376036,  -21.99581452],
        [  55.55555556,  -43.17410197,   17.65612953],
        [  66.66666667,  -45.58842021,   63.25586518],
        [  77.77777778,  -29.8953516,    76.73752616],
        [  88.88888889,  -10.58080561,   55.32547637],
        [ 100.,            0.,            0.,       ]
    ])

    return convert_from_lab(lab_control_points, x.size)


def rbg_spiral2(x: np.ndarray, *args):
    lab_control_points = np.array([
        [   0.,           55.41055985,   47.79160701],
        [  10.,           56.61771897,   18.05264897],
        [  20.,           53.39862797,   -8.91067298],
        [  30.,           42.93658223,  -27.94360612],
        [  40.,           22.81726349,  -32.70183941],
        [  50.,           -3.33785087,  -29.13316445],
        [  60.,          -20.64046498,  -13.27238683],
        [  70.,          -30.70012435,   10.91529904],
        [  80.,          -21.04285135,   25.98303778],
        [  90.,           -7.76410099,   18.05264897],
        [ 100.,            0.,            0.        ]
    ])

    return convert_from_lab(lab_control_points, x.size)


def rgb_to_grayscale(rgb: np.ndarray):
    """Convert RGB values to grayscale using luminosity values of RGB channels."""
    return np.sqrt(np.dot(rgb[...,:3] ** 2, rgb_weight))

cmap_dict = {
    "cold_blood": cold_blood,
    "better_rgb": better_rgb,
    "copper_salt": copper_salt,
    "rgb_spiral": rgb_spiral,
    "brg_spiral": brg_spiral,
    "rbg_spiral": rbg_spiral,
    "rbg_spiral2": rbg_spiral2
}


def get_custom_cmap(name: str = "cold_blood", n: int = 1000, ijk: tuple = None):
    cmp = get_continuous_cmap(n, name, ijk)
    return cmp


if __name__ == "__main__":
    cmap = get_custom_cmap("rbg_spiral2", 1000, None)
    # cmap = colormaps["viridis"]

    gradient = np.linspace(0, 1, 1000)
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
