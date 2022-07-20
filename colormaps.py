import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import beta


rgb_weight = np.array([0.2989, 0.5870, 0.1140])


def get_continuous_cmap(n, name: str, ijk: tuple, x1: float):
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
    result[..., :3] = np.einsum('i...j,k...j,i...k->i...j', result[..., :3],
                              weight.reshape(1, weight.size), I.reshape(*x.shape, 1))
    result[..., :3] /= np.max(np.max(result))
    result[..., 3] = 1
    return result


def copper_salt(x, i: int = 0, j: int = 1, k: int = 2):
    result = np.zeros([*x.shape, 4])
    dist = beta(2, 4)
    pdf = dist.pdf
    envelope = diverging_envelope(x, c=4, x1=0.4)
    result[..., i] = pdf(x)
    result[..., k] = pdf(1-x)
    result[..., i] /= np.max(result[..., i])
    result[..., k] /= np.max(result[..., k])
    result[..., j] = np.sqrt(3 * envelope - result[..., i] ** 2 - result[..., k] ** 2)
    weight = 1 / np.sqrt(rgb_weight)
    result[..., :3] = np.einsum('i...j,k...j->i...j', result[..., :3],
                                weight.reshape(1, *weight.shape))

    result[..., :3] /= np.max(result)
    result[..., 3] = 1
    return result



cmap_dict = {"cold_blood": cold_blood,
             "better_rgb": better_rgb,
             "copper_salt": copper_salt}


def get_custom_cmap(name: str = "diverging_linear", n: int = 1000, ijk: tuple = None):
    cmp = get_continuous_cmap(n, name, ijk)
    return cmp