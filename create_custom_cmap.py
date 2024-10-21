import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colour import sRGB_to_XYZ, XYZ_to_Lab, Lab_to_XYZ, XYZ_to_sRGB
from matplotlib.colors import ListedColormap
from matplotlib.widgets import RadioButtons
from pathlib import Path
import json
from scipy.interpolate import interp1d
from color_utils import rgb_to_grayscale_lightness, interpolate_lab, rgb_renormalized_lightness, rgb_renormalized_lightness


# Generate a grid of sRGB values
steps = 128  # Adjust for desired granularity (higher value = more data points)
r = np.linspace(0, 1, steps)
g = np.linspace(0, 1, steps)
b = np.linspace(0, 1, steps)

# Create a meshgrid of RGB combinations
R, G, B = np.meshgrid(r, g, b)
RGB = np.stack((R.flatten(), G.flatten(), B.flatten()), axis=-1)

# Convert sRGB to XYZ
XYZ = sRGB_to_XYZ(RGB)

# Convert XYZ to CIELAB
Lab = XYZ_to_Lab(XYZ)

# Store RGB and Lab values in a DataFrame
data = pd.DataFrame({
    'R': RGB[:, 0],
    'G': RGB[:, 1],
    'B': RGB[:, 2],
    'L': Lab[:, 0],
    'a': Lab[:, 1],
    'b': Lab[:, 2]
})

# Remove duplicate Lab values
data.drop_duplicates(subset=['L', 'a', 'b'], inplace=True)

# Bin the a* and b* values
n_bins = 200  # Adjust as needed for resolution
a_bins = np.linspace(-100, 100, n_bins + 1)
b_bins = np.linspace(-100, 100, n_bins + 1)

# Assign each data point to a bin
data['a_bin'] = np.digitize(data['a'], a_bins) - 1  # bins are 0-indexed
data['b_bin'] = np.digitize(data['b'], b_bins) - 1

# Filter out data points that are outside the bins
data = data[(data['a_bin'] >= 0) & (data['a_bin'] < n_bins) &
            (data['b_bin'] >= 0) & (data['b_bin'] < n_bins)]

# Group the data by (a_bin, b_bin) and find the maximum L* and corresponding RGB values
grouped = data.groupby(['a_bin', 'b_bin'], as_index=False)

# For each group, find the row with the maximum L*
max_L_df = grouped.apply(lambda x: x.loc[x['L'].idxmax()])

# Initialize arrays for maximum L* values and RGB colors
max_L_array = np.full((n_bins, n_bins), np.nan)
rgb_array = np.zeros((n_bins, n_bins, 3))

# Populate the arrays with the maximum L* values and RGB colors
for _, row in max_L_df.iterrows():
    a_idx = int(row['a_bin'])
    b_idx = int(row['b_bin'])
    max_L_array[a_idx, b_idx] = row['L']
    rgb_array[a_idx, b_idx, :] = [row['R'], row['G'], row['B']]

# Prepare for plotting
# Calculate the centers of the bins for accurate axis labels
a_centers = (a_bins[:-1] + a_bins[1:]) / 2
b_centers = (b_bins[:-1] + b_bins[1:]) / 2
extent = [a_bins[0], a_bins[-1], b_bins[0], b_bins[-1]]

# Plot the results
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot the Max L* Intensity image
ax1 = axs[0, 0]
im1 = ax1.imshow(max_L_array.T, extent=extent, origin='lower', cmap='gray', aspect='auto')
ax1.set_title('Max L* Intensity for (a*, b*) pairs')
ax1.set_xlabel('a* values')
ax1.set_ylabel('b* values')

# Plot the Brightest RGB Colors image
ax2 = axs[0, 1]
im2 = ax2.imshow(rgb_array.transpose(1, 0, 2), extent=extent, origin='lower', aspect='auto')
ax2.set_title('Brightest RGB Colors for (a*, b*) pairs')
ax2.set_xlabel('a* values')
ax2.set_ylabel('b* values')

# Adjust the subplot parameters to make room for the radio buttons
plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.05)

# Create an axes for the RadioButtons
rax = plt.axes([0.02, 0.4, 0.2, 0.15])  # [left, bottom, width, height]

# Define the labels for the RadioButtons
lightness_profiles = ('Linear', 'Diverging', "Diverging_quadratic", 'Flat')

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

def lab_to_rgb(lab_colors):
    # Convert LAB to XYZ
    XYZ = Lab_to_XYZ(lab_colors)
    # Convert XYZ to sRGB
    RGB = XYZ_to_sRGB(XYZ)
    return RGB

def update_gradient_plot(rgb_colors, ax_rgb, ax_gray):
    cmap = ListedColormap(rgb_colors)
    greyscale_colors = rgb_to_grayscale_lightness(rgb_colors)
    greyscale_colors = np.stack([greyscale_colors]*3, axis=-1)
    greyscale_cmap = ListedColormap(greyscale_colors)

    gradient = np.linspace(0, 1, rgb_colors.shape[0])
    gradient = np.vstack((gradient, gradient))

    ax_rgb.clear()
    ax_rgb.imshow(gradient, aspect='auto', cmap=cmap)
    ax_rgb.set_title('Perceptually Linear Colormap')
    ax_rgb.axis('off')

    ax_gray.clear()
    ax_gray.imshow(gradient, aspect='auto', cmap=greyscale_cmap)
    ax_gray.set_title('Greyscale Colormap')
    ax_gray.axis('off')

    # Redraw the figure
    plt.draw()


def update_colormap(event):
    global line_handle, c, m
    # Prepare the control points array
    control_points = np.array(clicked_points)
    n_colors = 256 
    # Interpolate between the control points in LAB space
    lab_colors = interpolate_lab(control_points, n_colors, profile=radio.value_selected.lower())
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
        print(f"Clicked at a* = {x:.2f}, b* = {y:.2f}")
        clicked_points.append((x, y))
        # Always append (0, 0) at the end
        # clicked_points.append((0, 0))
        # Plot a marker at the clicked position
        ax2.plot(x, y, 'ro', markersize=5)
        plt.draw()  # Update the plot
        # Update the gradient plot if enough points
        if len(clicked_points) >= 4:
            update_colormap(event)


# Connect the RadioButtons to the update_colormap function
radio.on_clicked(update_colormap)

# Connect the event handler to the figure
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

# After the plot is closed, convert the list to a NumPy array
clicked_points = np.array(clicked_points)

points_dict = {
    "points": clicked_points.tolist(),
    "profile": radio.value_selected.lower()
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
