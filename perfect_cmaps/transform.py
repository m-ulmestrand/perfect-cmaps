import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from typing import Tuple
from numba import njit
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import cv2
from scipy.interpolate import Rbf


def get_boundary(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Compute boundary as the difference between the shape and its eroded version
    int_mask = mask.astype(np.uint8)
    boundary_matrix = int_mask - binary_erosion(int_mask)
    return np.column_stack(np.where(boundary_matrix > 0)), boundary_matrix


def fill_missing_rgb(
        occupancy: np.ndarray,
        new_RGB_slice: np.ndarray
):
    x_valid, y_valid = np.where(occupancy)
    values = new_RGB_slice[x_valid, y_valid]

    # Get all grid points
    num_bins = occupancy.shape[0]
    arange = np.arange(num_bins)
    x_all, y_all = np.meshgrid(arange, arange, indexing='ij')

    # Interpolate for each channel separately
    rgb_interpolated = np.zeros_like(new_RGB_slice, dtype=np.float32)
    for i in range(3):  # Iterate over R, G, B channels
        rgb_interpolated[..., i] = griddata(
            (x_valid, y_valid), values[:, i], (x_all, y_all), method='linear', fill_value=0
        )
    
    return rgb_interpolated


def idw_fill_missing_rgb(occupancy: np.ndarray, new_RGB_slice: np.ndarray, k=4, power=2):
    num_bins = occupancy.shape[0]
    x_valid, y_valid = np.where(occupancy)
    values = new_RGB_slice[x_valid, y_valid]

    # KDTree for efficient nearest neighbor search
    tree = cKDTree(np.column_stack([x_valid, y_valid]))
    x_all, y_all = np.meshgrid(np.arange(num_bins), np.arange(num_bins), indexing='ij')
    grid_points = np.column_stack([x_all.ravel(), y_all.ravel()])

    # Query the k nearest known points for each missing point
    distances, indices = tree.query(grid_points, k=k)

    # Compute weights
    weights = 1 / (distances ** power + 1e-8)  # Avoid div by zero
    weights /= np.sum(weights, axis=1, keepdims=True)  # Normalize

    # Perform interpolation per channel
    rgb_interpolated = np.zeros((num_bins, num_bins, 3), dtype=np.float32)
    for i in range(3):
        interpolated_values = np.sum(weights * values[indices, i], axis=1)
        rgb_interpolated[..., i] = interpolated_values.reshape(num_bins, num_bins)

    return rgb_interpolated


@njit
def transform_inner(
        inner_points: np.ndarray,
        new_ab_slice: np.ndarray,
        RGB_slice: np.ndarray,
        new_RGB_slice: np.ndarray,
        boundary_points: np.ndarray,
        a_values: np.ndarray,
        b_values: np.ndarray,
        scales: np.ndarray
):
    step_size = 200 / b_values.size
    for i in range(inner_points.shape[0]):
        for j in range(inner_points.shape[1]):
            if not inner_points[i, j]:
                continue

            distances = np.sqrt((boundary_points[:, 0] - i) ** 2 + (boundary_points[:, 1] - j) ** 2)
            min_dist_index = np.argmin(distances)
            scale = scales[min_dist_index]

            new_a = a_values[j] * scale
            new_b = b_values[i] * scale
            a_bin = int((new_a + 100) // step_size)
            b_bin = int((new_b + 100) // step_size)
            if not (0 <= a_bin < a_values.size):
                continue
            if not (0 <= b_bin < b_values.size):
                continue
            new_ab_slice[b_bin, a_bin] = 1
            new_RGB_slice[b_bin, a_bin, :] = RGB_slice[i, j, :]


def transform(
        ab_slice: np.ndarray, 
        RGB_slice: np.ndarray,
        boundary_points: np.ndarray, 
        a_values: np.ndarray, 
        b_values: np.ndarray
    ):
    step_size = 200 / b_values.size
    new_boundary_points = boundary_points.copy()
    scales = np.zeros(boundary_points.shape[0])
    new_ab_slice = np.zeros((b_values.size, a_values.size), dtype=bool)
    inner_points = ab_slice.copy()
    new_RGB_slice = np.zeros_like(RGB_slice)
    
    for i, point in enumerate(boundary_points):
        b = b_values[point[0]]
        a = a_values[point[1]]
        scale = 99.5 / max(abs(a), abs(b))
        scales[i] = scale
        new_a = a * scale
        new_b = b * scale

        a_bin = int((new_a + 100) // step_size)
        b_bin = int((new_b + 100) // step_size)
        new_boundary_points[i] = b_bin, a_bin
        new_ab_slice[b_bin, a_bin] = 1
        inner_points[point[0], point[1]] = 0
        new_RGB_slice[b_bin, a_bin, :] = RGB_slice[point[0], point[1], :]

    transform_inner(
        inner_points, new_ab_slice, RGB_slice, new_RGB_slice, boundary_points, a_values, b_values, scales
    )

    new_RGB_slice = idw_fill_missing_rgb(new_ab_slice, new_RGB_slice)
    
    return new_RGB_slice


def stretch_to_square(occupancy: np.ndarray, rgb_matrix: np.ndarray, output_size=200):
    """
    Stretches the filled shape in the RGB matrix to a square.
    
    Args:
    - occupancy (np.ndarray): A binary matrix where 1 indicates a valid pixel.
    - rgb_matrix (np.ndarray): The original RGB matrix with missing values.
    - output_size (int): The size of the output square (default 256x256).
    
    Returns:
    - np.ndarray: The warped RGB image.
    """

    # Get coordinates of valid pixels
    y_valid, x_valid = np.where(occupancy)

    # Find bounding box of non-empty pixels
    x_min, x_max = np.min(x_valid), np.max(x_valid)
    y_min, y_max = np.min(y_valid), np.max(y_valid)

    # Define original quadrilateral (source points)
    src_pts = np.array([
        [x_min, y_min],  # Top-left
        [x_max, y_min],  # Top-right
        [x_max, y_max],  # Bottom-right
        [x_min, y_max]   # Bottom-left
    ], dtype=np.float32)

    # Define target square (destination points)
    dst_pts = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype=np.float32)

    # Compute perspective transformation matrix
    transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply warp to RGB matrix
    warped_rgb = cv2.warpPerspective(rgb_matrix, transform_matrix, (output_size, output_size), flags=cv2.INTER_NEAREST)

    return warped_rgb

def find_shape_corners(occupancy: np.ndarray):
    """
    Finds four corners of the filled region using convex hull.
    """
    y_valid, x_valid = np.where(occupancy)
    points = np.column_stack((x_valid, y_valid))  # (x, y) format

    # Get convex hull
    hull = cv2.convexHull(points)

    # Approximate to get 4 corner points
    epsilon = 0.02 * cv2.arcLength(hull, True)  # Tune this for better results
    approx = cv2.approxPolyDP(hull, epsilon, True)

    if len(approx) != 4:
        # If we don't get exactly 4 points, fallback to bounding box
        x_min, x_max = np.min(x_valid), np.max(x_valid)
        y_min, y_max = np.min(y_valid), np.max(y_valid)
        approx = np.array([
            [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]
        ], dtype=np.float32)

    return approx.reshape(4, 2).astype(np.float32)

def thin_plate_spline_warp(occupancy: np.ndarray, rgb_matrix: np.ndarray, output_size=256):
    """
    Warps an irregular shape to a square using Thin-Plate Spline (TPS) interpolation.

    Args:
    - occupancy (np.ndarray): Binary mask of valid pixels.
    - rgb_matrix (np.ndarray): The original RGB matrix with missing values.
    - output_size (int): The desired square output size.

    Returns:
    - np.ndarray: Warped RGB image.
    """

    # Find key corner points
    src_pts = find_shape_corners(occupancy)

    # Define uniform target square grid
    dst_pts = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype=np.float32)

    # Normalize source points for better TPS performance
    image_height, image_width = rgb_matrix.shape[:2]
    src_pts[:, 0] /= image_width  # Normalize x
    src_pts[:, 1] /= image_height  # Normalize y

    # Normalize target points
    dst_pts[:, 0] /= output_size  # Normalize x
    dst_pts[:, 1] /= output_size  # Normalize y

    # Create target coordinate grid
    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, output_size), 
                                 np.linspace(0, 1, output_size))

    x_grid_flat = x_grid.ravel()
    y_grid_flat = y_grid.ravel()

    # Apply TPS separately for X and Y coordinates
    rbf_x = Rbf(src_pts[:, 0], src_pts[:, 1], dst_pts[:, 0], function='thin_plate', smooth=0)
    rbf_y = Rbf(src_pts[:, 0], src_pts[:, 1], dst_pts[:, 1], function='thin_plate', smooth=0)

    mapped_x = rbf_x(x_grid_flat, y_grid_flat).reshape(output_size, output_size) * image_width
    mapped_y = rbf_y(x_grid_flat, y_grid_flat).reshape(output_size, output_size) * image_height

    # Clamp coordinates to avoid out-of-bounds errors
    mapped_x = np.clip(mapped_x, 0, image_width - 1).astype(np.float32)
    mapped_y = np.clip(mapped_y, 0, image_height - 1).astype(np.float32)

    # Warp RGB image using remapping
    warped_rgb = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    for i in range(3):  # Iterate over R, G, B channels
        warped_rgb[..., i] = cv2.remap(
            rgb_matrix[..., i], mapped_x, mapped_y,
            interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101
        )

    return warped_rgb, src_pts