import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import RadioButtons
from pathlib import Path
import json
from color_utils import (
    rgb_to_grayscale_lightness, 
    interpolate_lab, 
    rgb_renormalized_lightness, 
)


root_dir = Path(__file__).parent
extent = [-100, 100, -100, 100]

# Plot the results
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot the Max L* Intensity image
ax1 = axs[0, 0]
max_L_array = np.load(root_dir / "data" / "cielab_gamut_max_L.npy")
im1 = ax1.imshow(max_L_array, extent=extent, origin='lower', cmap='gray', aspect='equal')
ax1.set_title('Max L* Intensity for (a*, b*) pairs')
ax1.set_xlabel('a* values')
ax1.set_ylabel('b* values')

# Plot the Brightest RGB Colors image
ax2 = axs[0, 1]
max_L_array_RGB = np.load(root_dir / "data" / "cielab_gamut_max_L_RGB.npy")
im2 = ax2.imshow(max_L_array_RGB, extent=extent, origin='lower', aspect='equal')
ax2.set_title('Brightest RGB Colors for (a*, b*) pairs')
ax2.set_xlabel('a* values')
ax2.set_ylabel('b* values')

plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.05)
rax = plt.axes([0.02, 0.4, 0.2, 0.15])

# Define the labels for the RadioButtons
lightness_profiles = ('Linear', 'Diverging', "Diverging_sharper", "Diverging_inverted", "Diverging_inverted_sharper", 'Flat')

# Create the RadioButtons
radio = RadioButtons(rax, lightness_profiles)

# Prepare axes for the gradient plots (initially empty)
ax_rgb = axs[1, 1]
ax_gray = axs[1, 0]

plt.tight_layout()

# Initialize a list to store clicked points and a variable for the line handle
clicked_points = []
line_handle = None  # This will store the line object
ax2.plot(0, 0, 'rx', markersize=5)
c, m = 0.0, 0.0


def update_gradient_plot(rgb_colors, ax_rgb, ax_gray):
    cmap = ListedColormap(rgb_colors)
    greyscale_colors = rgb_to_grayscale_lightness(rgb_colors)
    greyscale_colors = np.stack([greyscale_colors]*3, axis=-1)
    greyscale_cmap = ListedColormap(greyscale_colors)

    gradient = np.linspace(0, 1, rgb_colors.shape[0])
    gradient = np.vstack((gradient, gradient))

    ax_rgb.clear()
    ax_rgb.imshow(gradient, aspect='auto', cmap=cmap)
    ax_rgb.set_title('Perceptually uniform Colormap')
    ax_rgb.axis('off')

    ax_gray.clear()
    ax_gray.imshow(gradient, aspect='auto', cmap=greyscale_cmap)
    ax_gray.set_title('Greyscale colormap')
    ax_gray.axis('off')

    # Redraw the figure
    plt.draw()


def update_colormap(event):
    global line_handle, c, m
    # Prepare the control points array
    control_points = np.array(clicked_points)
    n_colors = 256 
    # Interpolate between the control points in LAB space
    lightness_profile = radio.value_selected.lower()
    lab_colors = interpolate_lab(control_points, n_colors, profile=lightness_profile)
    # Remove previous line if it exists
    if line_handle is not None:
        line_handle.remove()
    # Plot new line and store the line object
    line_handle, = ax2.plot(lab_colors[:, 1], lab_colors[:, 2], color="black")
    rgb_colors, m, c = rgb_renormalized_lightness(lab_colors)
    rgb_colors = np.clip(rgb_colors, 0, 1)
    update_gradient_plot(rgb_colors, ax_rgb, ax_gray)


def onclick(event):
    # Check if the click is within the second subplot
    if event.inaxes == ax2:
        global clicked_points, line_handle
        x, y = event.xdata, event.ydata
        clicked_points.append((x, y))
        ax2.plot(x, y, 'ro', markersize=5)
        plt.draw()
        # Update the gradient plot if enough points
        if len(clicked_points) >= 3:
            update_colormap(event)

radio.on_clicked(update_colormap)
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

clicked_points = np.array(clicked_points)
points_dict = {
    "points": clicked_points.tolist(),
    "lightness": radio.value_selected.lower()
}
cmap_name = input("\nEnter desired colormap name. Quit with 'q'.\nControl points are saved in folder ./lab_control_points.\nColormap name: ")
cmap_name = cmap_name.strip()

if (len(cmap_name) == 0):
    cmap_name = "custom_cmap"
    
elif cmap_name == 'q':
    exit()

file_path = Path(__file__).parent / "lab_control_points" / cmap_name

parent_dir = file_path.parent
new_file_name = file_path.name

i = 2
while True:
    new_json_file = (parent_dir / new_file_name).with_suffix(".json")
    if new_json_file.exists():
        new_file_name = file_path.name + f"_{i}"
        i += 1
    else:
        break

with open(new_json_file, "w+") as json_file:
    json.dump(points_dict, json_file, indent=2)
