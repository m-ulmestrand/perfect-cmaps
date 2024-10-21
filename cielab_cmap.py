import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from color_utils import convert_from_lab, load_json, interpolate_lab_to_rgb


def main(lab_control_points: np.array, n_colors: int = 1000):
    if lab_control_points.size == 0:
        lab_control_points = load_json("rainbow2", n_colors)

    # Interpolate between the control points
    rgb_colors = interpolate_lab_to_rgb(lab_control_points, n_colors)

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
