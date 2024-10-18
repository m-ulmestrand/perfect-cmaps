import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colour import sRGB_to_XYZ, XYZ_to_Lab
import cielab_cmap
from pathlib import Path
import json


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
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the Max L* Intensity image
ax1 = axs[0]
im1 = ax1.imshow(max_L_array.T, extent=extent, origin='lower', cmap='gray', aspect='auto')
ax1.set_title('Max L* Intensity for (a*, b*) pairs')
ax1.set_xlabel('a* values')
ax1.set_ylabel('b* values')

# Plot the Brightest RGB Colors image
ax2 = axs[1]
im2 = ax2.imshow(rgb_array.transpose(1, 0, 2), extent=extent, origin='lower', aspect='auto')
ax2.set_title('Brightest RGB Colors for (a*, b*) pairs')
ax2.set_xlabel('a* values')
ax2.set_ylabel('b* values')

plt.tight_layout()
plt.savefig("cielab_gamut.png", format="png")

# Initialize a list to store clicked points
clicked_points = []

def onclick(event):
    # Check if the click is within the second subplot
    if event.inaxes == ax2:
        x, y = event.xdata, event.ydata
        print(f"Clicked at a* = {x:.2f}, b* = {y:.2f}")
        clicked_points.append((x, y))
        # Plot a marker at the clicked position
        ax2.plot(x, y, 'ro', markersize=5)
        plt.draw()  # Update the plot

# Connect the event handler to the figure
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

# After the plot is closed, convert the list to a NumPy array
clicked_points = np.array(clicked_points)

if len(clicked_points) != 0:
    clicked_points = np.append(clicked_points, np.array([[0, 0]]), axis=0)
    clicked_points = np.append(100 * np.linspace(0, 1, clicked_points.shape[0])[:, None], clicked_points, axis=1)
else:
    clicked_points = np.zeros((0, 3))

points_dict = {"points": clicked_points.tolist()}
cmap_name = input("Enter desired colormap name.\nControl points are saved in folder ./lab_control_points.\nColormap name:")
cmap_name = cmap_name.strip()

if (len(cmap_name) == 0):
    cmap_name = "custom_cmap"

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
    
cielab_cmap.main(clicked_points)