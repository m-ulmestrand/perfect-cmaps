import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from matplotlib.colors import ListedColormap
# from numba import njit, prange
from scipy.interpolate import interp1d
from matplotlib import colormaps


def main(lab_control_points: np.array, n_colors: int = 1000):
    if lab_control_points.size == 0:
        # Define the control points in LAB space

        lab_control_points = np.array([
            [   0.        ,   58.62965085,  -56.49300584],
            [  11.11111111,   32.07215011,  -54.51040864],
            [  22.22222222,    6.7218085 ,  -42.61482542],
            [  33.33333333,  -12.99512386,  -25.56448948],
            [  44.44444444,  -24.26194235,  -10.1002313 ],
            [  55.55555556,  -32.30966985,   11.31181848],
            [  66.66666667,  -37.94307909,   33.12038771],
            [  77.77777778,  -39.55262459,   54.92895693],
            [  88.88888889,  -29.8953516 ,   70.39321511],
            [ 100.        ,    0.        ,    0.        ]])
        
    # Interpolate between the control points
    lab_colors = np.zeros((n_colors, 3))
    for i in range(3):
        interpolator = interp1d(np.linspace(0, 1, lab_control_points.shape[0]), lab_control_points[:, i], kind='cubic')
        lab_colors[:, i] = interpolator(np.linspace(0, 1, n_colors))

    # Convert the LAB colors to RGB
    rgb_colors = color.lab2rgb(lab_colors)

    # Ensure the RGB values are in the valid range
    rgb_colors = np.clip(rgb_colors, 0, 1)

    # Create and register the colormap
    cmap = ListedColormap(rgb_colors, name='perceptually_linear_cmap')

    # Convert RGB to greyscale
    greyscale_colors = np.sqrt(np.dot(rgb_colors[...,:3] ** 2, [0.299, 0.587, 0.114]))
    greyscale_colors = np.stack((greyscale_colors,) * 3, axis=-1)
    greyscale_cmap = ListedColormap(greyscale_colors, name='greyscale_cmap')

    # Test the colormap
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4))

    ax1.imshow(gradient, aspect='auto', cmap=cmap)
    ax1.set_title('Perceptually Linear Colormap')
    ax1.axis('off')

    ax2.imshow(gradient, aspect='auto', cmap=greyscale_cmap)
    ax2.set_title('Greyscale Colormap')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    for c, color_string in zip(range(3), ["red", "green", "blue"]):
        plt.plot(rgb_colors[:, c], color=color_string)

    rgb_weights = np.array([0.2989, 0.5870, 0.1140])
    luminance = np.sqrt(np.dot(rgb_colors ** 2, rgb_weights))
    plt.plot(luminance, color="black")
    plt.show()

    dem_data = plt.imread("/home/mattias/imgs/DEM3.gif")
    
    if len(dem_data.shape) == 3:
        dem_data = dem_data[:, :, 0]
        
    plt.imshow(dem_data, cmap=cmap)
    plt.show()


if __name__ == "__main__":
    main(np.zeros((0, 3)))
