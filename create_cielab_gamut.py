import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colour import sRGB_to_XYZ, XYZ_to_Lab
from pathlib import Path

# Generate a grid of sRGB values
steps = 250  # Adjust for desired granularity (higher value = more data points)
r = np.linspace(0, 1, steps)
g = np.linspace(0, 1, steps)
b = np.linspace(0, 1, steps)

R, G, B = np.meshgrid(r, g, b)
RGB = np.stack((R.flatten(), G.flatten(), B.flatten()), axis=-1)

XYZ = sRGB_to_XYZ(RGB)
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

data.drop_duplicates(subset=['L', 'a', 'b'], inplace=True)

n_bins = 400  # Adjust as needed for resolution
a_bins = np.linspace(-100, 100, n_bins + 1)
b_bins = np.linspace(-100, 100, n_bins + 1)

# Assign each data point to a bin
data['a_bin'] = np.digitize(data['a'], a_bins) - 1  # bins are 0-indexed
data['b_bin'] = np.digitize(data['b'], b_bins) - 1

# Filter out data points that are outside the bins
data = data[(data['a_bin'] >= 0) & (data['a_bin'] < n_bins) &
            (data['b_bin'] >= 0) & (data['b_bin'] < n_bins)]

grouped = data.groupby(['a_bin', 'b_bin'], as_index=False)
max_L_df = grouped.apply(lambda x: x.loc[x['L'].idxmax()])

max_L_array = np.full((n_bins, n_bins), np.nan)
rgb_array = np.zeros((n_bins, n_bins, 3))

for _, row in max_L_df.iterrows():
    a_idx = int(row['a_bin'])
    b_idx = int(row['b_bin'])
    max_L_array[a_idx, b_idx] = row['L']
    rgb_array[a_idx, b_idx, :] = [row['R'], row['G'], row['B']]

a_centers = (a_bins[:-1] + a_bins[1:]) / 2
b_centers = (b_bins[:-1] + b_bins[1:]) / 2
extent = [a_bins[0], a_bins[-1], b_bins[0], b_bins[-1]]

data_path = Path(__file__).parent / "data"
np.save(data_path / "cielab_gamut_max_L_RGB", rgb_array.transpose(1, 0, 2))
np.save(data_path / "cielab_gamut_max_L", max_L_array.T)
plt.imshow(rgb_array.transpose(1, 0, 2), extent=extent, origin='lower', aspect='equal')
plt.show()