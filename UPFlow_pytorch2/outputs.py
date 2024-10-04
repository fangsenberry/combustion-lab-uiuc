import sys
import os

# Add the utils directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))

from utils.fv_util import FlowInitialization as Fizi
from utils.fv_util import FlowAnalysis as Flay
from utils.fv_util import FlowConfig
import torch
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from statsmodels.graphics.tsaplots import plot_acf
from scipy.fftpack import fft, ifft, fftfreq
from scipy.optimize import curve_fit
import random
from scipy.spatial import distance
from collections import deque
import time

def convert_mask_to_binary(binary_mask):
    # Convert 255 to 1, leave 0 as is
    binary_mask = (binary_mask == 255).astype(np.uint8)
    return binary_mask

def segment_regions(binary_mask):
    # Convert the mask to binary (0s and 1s)
    binary_mask = convert_mask_to_binary(binary_mask)
    
    # Perform connected component analysis
    num_labels, labels = cv2.connectedComponents(binary_mask)
    print(f"Number of regions detected: {num_labels - 1}")  # Subtract 1 for background (label 0)
    return labels, num_labels

def compute_region_velocity(u, v, labels, num_labels):
    region_velocities = []

    for label in range(1, num_labels):  # Skip the background label (0)
        mask = labels == label
        mean_u = np.mean(u[mask])
        mean_v = np.mean(v[mask])
        region_velocities.append((mean_u, mean_v))

    return np.array(region_velocities)

def compute_region_centroids(labels, num_labels):
    centroids = []

    for label in range(1, num_labels):  # Skip the background label
        mask = labels == label
        y, x = np.where(mask)

        if len(x) > 0 and len(y) > 0:  # Ensure the region is non-empty
            centroid_x = np.mean(x)
            centroid_y = np.mean(y)
            centroids.append((centroid_x, centroid_y))  # Append as tuple
        else:
            print(f"Label {label} has no valid pixels.")
            centroids.append((np.nan, np.nan))  # Handle empty regions

    return np.array(centroids)  # Return as NumPy array for consistency

def compute_region_sizes(labels, num_labels):
    region_sizes = []
    
    for label in range(1, num_labels):  # Skip background label
        mask = labels == label
        region_size = np.sum(mask)  # Count the number of pixels in the region
        region_sizes.append(region_size)
    
    return np.array(region_sizes)

def detect_breakup_by_size(region_sizes, prev_region_sizes, matches, threshold_change=0.5, num_labels=None):
    # Initialize a list for breakup points (default to False for all regions)
    breakup_points = [False] * (num_labels - 1)
    
    # Compare matched regions' sizes and detect breakup
    for i, j in matches:  # i is the index in current frame, j is the index in previous frame
        if prev_region_sizes[j] > 0:  # Ensure there was a valid previous size
            size_change = abs(region_sizes[i] - prev_region_sizes[j]) / prev_region_sizes[j]
            # Mark region as breakup if size change exceeds threshold
            if size_change > threshold_change:
                breakup_points[i] = True
    
    return breakup_points

def visualize_breakup(u, v, labels, breakup_points, centroids):
    plt.imshow(np.sqrt(u**2 + v**2), cmap='jet')
    plt.scatter(centroids[:, 0], centroids[:, 1], color='white', label='Centroids')

    # Highlight breakup points
    for i, is_breakup in enumerate(breakup_points):
        if is_breakup:
            plt.scatter(centroids[i + 1, 0], centroids[i + 1, 1], color='black', label='Breakup Point')

    plt.legend()
    plt.show()

def visualize_centroids(u, v, centroids):
    plt.figure(figsize=(10, 6))
    
    # Plot velocity magnitude as a background
    plt.imshow(np.sqrt(u**2 + v**2), cmap='jet')
    
    # Plot all centroids
    for i, (cx, cy) in enumerate(centroids):
        if not np.isnan(cx) and not np.isnan(cy):
            plt.scatter(cx, cy, color='yellow', s=100, label=f'Centroid {i}' if i == 0 else "")

    plt.title('Centroids of Fuel Regions')
    plt.legend(loc='upper right')
    plt.show()

def visualize_colored_regions(binary_mask, u, v, labels, breakup_points, num_labels, original_image=None):
    # Generate a color map with as many colors as there are regions
    colormap = plt.get_cmap('tab20', num_labels)

    # Create a new image to hold the colored regions
    colored_image = np.zeros((*binary_mask.shape, 3), dtype=np.uint8)
    
    # Ensure that breakup_points array has the correct size
    assert len(breakup_points) == num_labels - 1, "Size of breakup_points must match the number of regions."

    # Assign each region a color
    for label in range(1, num_labels):  # Skip label 0 (background)
        mask = labels == label
        
        # Handle case where label exceeds breakup_points array
        if label - 1 < len(breakup_points) and breakup_points[label - 1]:  # Highlight regions with breakup
            colored_image[mask] = [255, 0, 0]  # Red color for breakup
        else:
            color = colormap(label)[:3]  # Get RGB values from colormap
            color = (np.array(color) * 255).astype(np.uint8)  # Convert to 0-255 range
            colored_image[mask] = color
    
    plt.figure(figsize=(10, 6))
    
    # If you have an original image, overlay it
    if original_image is not None:
        plt.imshow(original_image, cmap='gray', alpha=0.7)  # Assuming the original image is grayscale
    
    # Overlay the colored regions
    plt.imshow(colored_image, alpha=0.5)  # Overlay the colored regions with transparency
    
    plt.title('Colored Regions with Breakup Highlighted')
    plt.show()

def match_regions_by_centroids(region_centroids, prev_centroids, threshold=10.0):
    """
    Match regions between current and previous frames by comparing centroids.
    
    Args:
        region_centroids: Centroids of the current frame's regions.
        prev_centroids: Centroids of the previous frame's regions.
        threshold: Maximum distance to consider two regions as a match.
    
    Returns:
        matches: A list of tuples where each tuple (i, j) means that region i in the
                 current frame corresponds to region j in the previous frame.
    """
    matches = []
    
    # Compute the distance matrix between current and previous centroids
    distances = cdist(region_centroids, prev_centroids)
    
    # Find matches based on proximity
    for i, row in enumerate(distances):
        min_dist = np.min(row)
        if min_dist < threshold:
            j = np.argmin(row)  # Index of the closest centroid in the previous frame
            matches.append((i, j))
    
    return matches
#############################################################################################
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

def plot_dbscan_labels(labels, img):
    """
    Visualize the DBSCAN labels overlaid on the original image.
    
    Args:
        labels: DBSCAN cluster labels for each pixel.
        img: Original grayscale image.
    """
    # Create a color map for the labels
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    
    # Generate a color map for the labels
    plt.figure(figsize=(8, 8))
    
    # Plot the original image in the background
    plt.imshow(img, cmap='gray', alpha=0.5)
    
    # Overlay the DBSCAN labels using a colormap
    plt.imshow(labels, cmap='jet', alpha=0.5)
    plt.colorbar(label="Cluster Label")
    plt.title(f'DBSCAN Cluster Labels (Number of Clusters: {num_labels})')
    plt.show()

def apply_dbscan_with_velocity(img, u_vectors, v_vectors, eps=5, min_samples=5):
    """Apply DBSCAN to raw image pixels and velocity vectors."""
    # Get coordinates of all pixels
    height, width = img.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Flatten all arrays to create feature vectors for each pixel
    pixel_coords = np.stack((x.ravel(), y.ravel()), axis=1)
    pixel_intensity = img.ravel().reshape(-1, 1)  # Grayscale intensity of each pixel
    velocity_u = u_vectors.ravel().reshape(-1, 1)  # Horizontal velocity (u)
    velocity_v = v_vectors.ravel().reshape(-1, 1)  # Vertical velocity (v)

    # Feature vector: [x, y, intensity, u_velocity, v_velocity]
    features = np.hstack((pixel_coords, pixel_intensity, velocity_u, velocity_v))

    # Apply DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(features)

    # Reshape labels to match the original image
    labels = db.labels_.reshape(img.shape)
    plot_dbscan_labels(labels, img)

    return labels

def track_regions(labels_prev, labels_curr, u_prev, v_prev, u_curr, v_curr, threshold=5.0):
    """Track regions between frames and detect breakups."""
    # Get centroids of regions in the previous and current frames
    centroids_prev = compute_region_centroids(labels_prev)
    centroids_curr = compute_region_centroids(labels_curr)

    # Match regions based on centroid proximity
    distances = cdist(centroids_prev, centroids_curr)
    matches = []
    for i, row in enumerate(distances):
        min_dist = np.min(row)
        if min_dist < threshold:
            j = np.argmin(row)  # Closest region in current frame
            matches.append((i, j))

    # Check for breakups (one region in prev splitting into multiple in curr)
    breakups = []
    for i, j in matches:
        # Compare velocities of the matched regions
        velocity_prev = np.mean(np.stack([u_prev[labels_prev == i], v_prev[labels_prev == i]]), axis=1)
        velocity_curr = np.mean(np.stack([u_curr[labels_curr == j], v_curr[labels_curr == j]]), axis=1)
        velocity_diff = np.linalg.norm(velocity_prev - velocity_curr)

        # If velocity difference is large, consider it a breakup
        if velocity_diff > threshold:
            breakups.append((i, j))

    return matches, breakups

def compute_region_centroids(labels):
    """Compute centroids of labeled regions."""
    centroids = []
    for label in np.unique(labels):
        if label == -1:  # Skip noise
            continue
        mask = labels == label
        y, x = np.where(mask)
        centroids.append((np.mean(x), np.mean(y)))
    return np.array(centroids)

def process_data_with_dbscan(loaded_data, threshold=5.0, eps=5, min_samples=5):
    # Unpack data
    u_vectors = loaded_data['u_vectors']  # Shape: (num_frames, height, width)
    v_vectors = loaded_data['v_vectors']  # Shape: (num_frames, height, width)
    img_list = loaded_data['img_list']  # Shape: (num_frames, height, width)

    num_frames = u_vectors.shape[0]

    # Loop through all frames and track regions between consecutive frames
    for frame_idx in range(1, num_frames):
        print(f"Processing frame {frame_idx}/{num_frames - 1}...")

        # Get data for current and previous frames
        img_prev = img_list[frame_idx - 1]
        img_curr = img_list[frame_idx]
        u_prev = u_vectors[frame_idx - 1]
        v_prev = v_vectors[frame_idx - 1]
        u_curr = u_vectors[frame_idx]
        v_curr = v_vectors[frame_idx]

        # Apply DBSCAN to identify regions in both frames
        labels_prev = apply_dbscan_with_velocity(img_prev, u_prev, v_prev, eps=eps, min_samples=min_samples)
        labels_curr = apply_dbscan_with_velocity(img_curr, u_curr, v_curr, eps=eps, min_samples=min_samples)

        # Track regions and detect breakups
        matches, breakups = track_regions(labels_prev, labels_curr, u_prev, v_prev, u_curr, v_curr, threshold=threshold)
        
        # print(f"Frame {frame_idx}: Matches between regions: {matches}")
        print(f"Frame {frame_idx}: Breakups detected: {breakups}")

def process_data_with_dbscan_and_visualization(loaded_data, threshold=10.0, eps=5, min_samples=3):
    """
    Process data to detect regions, track them across frames using DBSCAN, and visualize breakups.
    """
    # Unpack data from loaded_data
    u_vectors = loaded_data['u_vectors']  # Shape: (num_frames, height, width)
    v_vectors = loaded_data['v_vectors']  # Shape: (num_frames, height, width)
    img_list = loaded_data['img_list']  # Shape: (num_frames, height, width)

    num_frames = u_vectors.shape[0]

    # Loop through all frames and track regions between consecutive frames
    for frame_idx in range(1, num_frames):
        print(f"Processing frame {frame_idx}/{num_frames - 1}...")

        # Get data for the current and previous frames
        img_prev = img_list[frame_idx - 1]
        img_curr = img_list[frame_idx]
        u_prev = u_vectors[frame_idx - 1]
        v_prev = v_vectors[frame_idx - 1]
        u_curr = u_vectors[frame_idx]
        v_curr = v_vectors[frame_idx]

        # Apply DBSCAN to identify regions in both frames
        labels_prev = apply_dbscan_with_velocity(img_prev, u_prev, v_prev, eps=eps, min_samples=min_samples)
        labels_curr = apply_dbscan_with_velocity(img_curr, u_curr, v_curr, eps=eps, min_samples=min_samples)

        # Track regions and detect breakups
        matches, breakups = track_regions(labels_prev, labels_curr, u_prev, v_prev, u_curr, v_curr, threshold=threshold)
        
        # Log matches and breakups for debug purposes
        # print(f"Frame {frame_idx}: Matches between regions: {matches}")
        print(f"Frame {frame_idx}: Breakups detected: {breakups}")

        # Visualize breakups if any were detected
        if breakups:
            visualize_breakup(img_prev, img_curr, labels_prev, labels_curr, breakups)

def visualize_breakup(img_prev, img_curr, labels_prev, labels_curr, breakup_pairs):
    """
    Visualize breakup regions between two frames.
    
    Args:
        img_prev: The image from the previous frame.
        img_curr: The image from the current frame.
        labels_prev: Region labels from the previous frame.
        labels_curr: Region labels from the current frame.
        breakup_pairs: List of tuples indicating breakup regions (prev, curr).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the previous frame with regions that experienced breakups
    axes[0].imshow(img_prev, cmap='gray')
    axes[0].set_title('Previous Frame - Breakup Regions')
    
    # Plot the current frame with regions that correspond to breakups
    axes[1].imshow(img_curr, cmap='gray')
    axes[1].set_title('Current Frame - Resulting Regions')
    
    # Loop through breakup pairs and unpack the tuple
    for prev, curr in breakup_pairs:
        # Use the prev and curr values to plot contours for the corresponding regions
        axes[0].contour(labels_prev == prev, colors='red', linewidths=2, label=f'Region {prev}')
        axes[1].contour(labels_curr == curr, colors='blue', linewidths=2, label=f'Region {curr}')
    
    plt.show()

#############################################################################################
def trace_continuous_path(binary_img):
    height, width = binary_img.shape
    path = []

    # Start at the leftmost point in the top row
    for y in range(height):
        if binary_img[y, 0] == 1:
            current_point = (0, y)
            path.append(current_point)
            break
    else:
        print("No valid starting point found on the left edge.")
        return path

    # Trace the path downward
    while current_point[0] < width - 1:  # Stop before the right edge
        x, y = current_point
        neighbors = [(x + 1, y), (x + 1, y - 1), (x + 1, y + 1)]  # Right, down-right, up-right
        
        # Choose the first valid neighbor that is part of the binary object
        valid_neighbor = None
        for nx, ny in neighbors:
            if 0 <= ny < height and binary_img[ny, nx] == 1:
                valid_neighbor = (nx, ny)
                break

        if valid_neighbor:
            path.append(valid_neighbor)
            current_point = valid_neighbor
        else:
            break  # No more valid neighbors, so stop tracing

    return path

def resample_contour(contour, num_points=500):
    # Convert the contour into an array of distances along the contour
    contour = contour[:, 0, :]
    distances = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0)  # Add the starting point

    # Interpolate over a uniform set of distances to resample the contour
    interp_distances = np.linspace(0, distances[-1], num_points)
    interp_func_x = interp1d(distances, contour[:, 0], kind='linear')
    interp_func_y = interp1d(distances, contour[:, 1], kind='linear')

    # Get the resampled x and y coordinates
    resampled_x = interp_func_x(interp_distances)
    resampled_y = interp_func_y(interp_distances)

    return np.vstack((resampled_x, resampled_y)).T  # Return as Nx2 array

def running_average(values, window_size):
    return np.convolve(values, np.ones(window_size), 'valid') / window_size

############################################################ CDF functions ############################################################
def calculate_magnitude(u, v):
    """Calculate the magnitude of flow vectors."""
    return np.sqrt(u**2 + v**2)

def plot_cdf_velocity_all_frames(u_filtered, v_filtered, num_slices, axis='y-slices', resolution=9.3):
    """
    Plot separate CDFs for u (x-velocity) and v (y-velocity) for different positions (rows or columns) across all frames.
    
    Parameters:
    u_filtered: 3D array of filtered u (x-velocity) vectors (frames, height, width)
    v_filtered: 3D array of filtered v (y-velocity) vectors (frames, height, width)
    num_slices: Number of rows/columns to plot
    axis: 'x-slices' (rows) or 'y-slices' (columns)
    resolution: Pixel resolution in micrometers (default is 9.3 um per pixel)
    """
    # Convert pixel resolution to mm
    resolution_mm = resolution * 1e-3  # Convert μm to mm

    # Get the image dimensions
    num_frames, height, width = u_filtered.shape[:3]
    print(f"Data dimensions: {num_frames} frames, {height} x {width}")

    # Convert velocity to the desired units
    u_filtered = u_filtered * resolution * 10**-6 * 90517  # Convert u-velocity to appropriate units
    v_filtered = v_filtered * resolution * 10**-6 * 90517  # Convert v-velocity to appropriate units

    # Exclude the last 10 rows for column-based slicing if 'y-slices'
    
    u_filtered = u_filtered[:, :, :-10]  # Remove the last 10 columns for all frames
    v_filtered = v_filtered[:, :, :-10]  # Remove the last 10 columns for all frames

    u_filtered = np.flip(u_filtered, axis=2)  # Flip along width (axis=2)
    v_filtered = np.flip(v_filtered, axis=2)  # Flip along width (axis=2)

    # Adapt based on the axis (rows or columns)
    if axis == 'x-slices':  # Column-wise slicing
        width -= 10  # Adjust the width after removing columns
        slices_to_plot = np.linspace(0, width-1, num_slices, dtype=int)
        label_text = 'x-slice'
        u_data = lambda slice_idx: u_filtered[:, :, slice_idx].flatten()  # Extract column data across all frames
        v_data = lambda slice_idx: v_filtered[:, :, slice_idx].flatten()  # Extract column data across all frames
        positions = slices_to_plot * resolution_mm  # Convert column indices to positional values in mm
    elif axis == 'y-slices':  # Row-wise slicing
        slices_to_plot = np.linspace(0, height-1, num_slices, dtype=int)
        label_text = 'y-slice'
        u_data = lambda slice_idx: u_filtered[:, slice_idx, :].flatten()  # Extract row data across all frames
        v_data = lambda slice_idx: v_filtered[:, slice_idx, :].flatten()  # Extract row data across all frames
        centerline = (height - 1) / 2  # Calculate the centerline of the image
        positions = -(slices_to_plot - centerline) * resolution_mm  # Convert row indices to centered positions in mm
    else:
        raise ValueError("Axis must be either 'x-slices' or 'y-slices'.")

    positions = np.round(positions, decimals=2)
    positions[np.abs(positions) < 1e-6] = 0
    # Plot CDF for u (x-velocity)
    plt.figure(figsize=(10, 6))
    for slice_idx, pos in zip(slices_to_plot, positions):
        # Extract u (x-velocity) for the specific row/column across all frames
        u_slice = u_data(slice_idx)
        
        # Remove NaN values caused by masking non-white regions
        u_slice = u_slice[~np.isnan(u_slice)]

        # Calculate CDF
        sorted_u = np.sort(u_slice)
        cdf_u = np.arange(1, len(sorted_u)+1) / len(sorted_u)

        # Plot the CDF for this row/column
        plt.plot(sorted_u, cdf_u, label=f'Position {pos:.2f} mm', linewidth=2)

    # Add labels and legend for u-velocity
    plt.tick_params(axis='both', which='both', direction='in', length=6, width=2, colors='black')
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    # plt.title(f'CDF of X-Velocity at Different {label_text} Positions Across All Frames')
    plt.xlabel('X-Velocity [m/s]', fontsize=18)
    plt.ylabel('CDF', fontsize=18)
    plt.legend(loc='lower right', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'CDF_{label_text}_x_vel.png')
    # plt.show()

    # Plot CDF for v (y-velocity)
    plt.figure(figsize=(10, 6))
    for slice_idx, pos in zip(slices_to_plot, positions):
        # Extract v (y-velocity) for the specific row/column across all frames
        v_slice = v_data(slice_idx)
        
        # Remove NaN values caused by masking non-white regions
        v_slice = v_slice[~np.isnan(v_slice)]

        # Calculate CDF
        sorted_v = np.sort(v_slice)
        cdf_v = np.arange(1, len(sorted_v)+1) / len(sorted_v)

        # Plot the CDF for this row/column
        plt.plot(sorted_v, cdf_v, label=f'Position {pos:.2f} mm', linewidth=2)

    # Add labels and legend for v-velocity
    plt.tick_params(axis='both', which='both', direction='in', length=6, width=2, colors='black')
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    # plt.title(f'CDF of Y-Velocity at Different {label_text} Positions Across All Frames')
    
    plt.xlabel('Y-Velocity [m/s]', fontsize=18)
    plt.ylabel('CDF', fontsize=18)
    plt.legend(loc='lower right', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'CDF_{label_text}_y_vel.png')
    # plt.show()
#############################################################################################
def process_breakup_length(binary_image_list=None, breakup_x_threshold=50):
    print('created lists')
    

    previous_avg_x = None  # To store the previous frame's average x-value
    pre_breakup_x = None   # To store the x-value before a breakup
    breakup_occurred = False  # Flag to track if a breakup has occurred
    best_segment_list = []  # List to store best segments

    # Two lists to store raw and smoothed x-values
    raw_x_y_values = []  # To store raw x-values before clamping
    smoothed_x_y_values = []  # To store x-values after clamping
    frame_idx_breakup = np.array([])  # To store the frame index of the breakup

    print('starting loop')

    # Process each binarized image (i.e., each frame)
    for frame_idx, binarized_image in enumerate(binary_image_list):
        height, width = binarized_image.shape

        # Find contours
        contours, _ = cv2.findContours(binarized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        furthest_left_x = float('inf')
        leftmost_contour = None

        # Loop through each contour and check if it spans from top to bottom
        for contour in contours:
            y_coords = contour[:, :, 1]  # Extract y-coordinates of the contour
            if np.any(y_coords == 0) and np.any(y_coords == height - 1):  # Check if the contour spans top to bottom
                # Calculate the centroid of the contour
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])  # Calculate the x-coordinate of the centroid
                    if cx < furthest_left_x:  # Select the contour with the furthest left centroid
                        furthest_left_x = cx
                        leftmost_contour = contour

        # Ensure we have a valid contour and extract the leftmost points
        if leftmost_contour is not None:
            contour_points = leftmost_contour[:, 0, :]
            x_values = contour_points[:, 0]
            y_values = contour_points[:, 1]

            # Initialize a list to store segments
            segments = []
            current_segment_x = []
            current_segment_y = []

            # Loop through each consecutive pair of points to check the slope and group segments
            for i in range(len(x_values) - 1):
                x1, y1 = x_values[i], y_values[i]
                x2, y2 = x_values[i + 1], y_values[i + 1]
                current_segment_x.append(x1)
                current_segment_y.append(y1)

                # If segment spans top to bottom, save it
                if y1 == y2 and y1 == height - 1:
                    if 0 in current_segment_y and height - 1 in current_segment_y:
                        segments.append((current_segment_x, current_segment_y))
                    current_segment_x = []  # Reset for next segment
                    current_segment_y = []

            # Add the last segment if it's non-empty and spans from top to bottom
            if current_segment_x and 0 in current_segment_y and height - 1 in current_segment_y:
                segments.append((current_segment_x, current_segment_y))

            # Find the best segment with the lowest average x-value
            min_avg_x = float('inf')
            best_segment = None
            for segment in segments:
                segment_x_values, segment_y_values = segment
                avg_x = np.mean(segment_x_values)
                if avg_x < min_avg_x:
                    min_avg_x = avg_x
                    best_segment = segment

            # If there's a valid best segment
            if best_segment:
                best_x_values = np.array(best_segment[0])
                best_y_values = np.array(best_segment[1])

                # Round y-values to whole numbers to correspond to the image height
                rounded_y_values = np.round(best_y_values).astype(int)

                # Initialize the list to store the average x-values for each unique y-coordinate
                avg_x_per_y = []
                for y in range(height):  # Loop through each unique y-value
                    x_at_y = best_x_values[rounded_y_values == y]
                    avg_x_per_y.append(np.mean(x_at_y) if len(x_at_y) > 0 else np.nan)

                # Filter out NaN values to calculate the overall average x-value
                valid_avg_x_per_y = [x for x in avg_x_per_y if not np.isnan(x)]
                overall_avg_x = np.mean(valid_avg_x_per_y)

                # Save the raw x-value before any smoothing or clamping
                raw_x_y_values.append(overall_avg_x)

                # Suppress artificial rise after a breakup
                if breakup_occurred and pre_breakup_x is not None and (overall_avg_x - pre_breakup_x > breakup_x_threshold):
                    print(f"Artificial rise detected. Clamping x-value to {pre_breakup_x}.")
                    overall_avg_x = pre_breakup_x  # Suppress the jump by clamping to pre-breakup value
                    breakup_occurred = False  # Reset breakup flag

                # Detect a breakup: If x-value drops significantly compared to the previous frame
                if previous_avg_x is not None and (previous_avg_x - overall_avg_x > breakup_x_threshold):
                    breakup_occurred = True  # Set breakup flag
                    frame_idx_breakup = np.append(frame_idx_breakup, frame_idx - 1)  # Store the frame index of the breakup

                # Store the current average x-value for the next frame (smoothed or clamped if needed)
                previous_avg_x = overall_avg_x  # Update previous x-value

                # Save the smoothed/corrected x-value
                smoothed_x_y_values.append(overall_avg_x)

                # Store the best segment for further analysis or plotting
                best_segment_list.append(best_segment)
        else:
            frame_idx_breakup = np.append(frame_idx_breakup, frame_idx - 1)  # Store the frame index of the breakup
            smoothed_x_y_values.append(width)
            raw_x_y_values.append(width)

    print('done with loop')

    # Post-process breakup frames
    smoothed_x_y_values = np.array(smoothed_x_y_values)
    frame_idx_breakup = np.array(frame_idx_breakup, dtype=int)
    

    # Filter out any indices in frame_idx_breakup that are out of bounds
    frame_idx_breakup = frame_idx_breakup[frame_idx_breakup < len(smoothed_x_y_values)]
    
    # Modify values at breakup points by averaging previous and next points
    for idx in frame_idx_breakup:
        if 1 <= idx < len(smoothed_x_y_values) - 1:
            avg_value = (smoothed_x_y_values[idx - 1] + smoothed_x_y_values[idx + 1]) / 2
            smoothed_x_y_values[idx] = avg_value

    return raw_x_y_values, smoothed_x_y_values, frame_idx_breakup, best_segment_list

def calculate_hole_porosity_with_visualization(binary_image, breakup_length):
    """
    Calculates the porosity (ratio of hole area to total area) below the breakup length
    and visualizes the detected holes on the binary image.
    
    Parameters:
    - binary_image: Binary image (fuel = 1, background = 0).
    - breakup_length: The y-coordinate below which the analysis is performed.
    
    Returns:
    - porosity: The ratio of hole area to total area in the region.
    """
    height, width = binary_image.shape
    
    # Ensure breakup_length is an integer
    breakup_length = int(breakup_length)
    
    # Extract the region below the breakup length
    region_of_interest = binary_image[:, breakup_length:]
    
    # Invert the binary image to find holes (fuel = 0, background = 1)
    inverted_region = 255 - region_of_interest
    
    
    # Find contours of the holes in the inverted region
    contours, _ = cv2.findContours(inverted_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate the area of holes
    hole_area = sum(cv2.contourArea(contour) for contour in contours)
    
    # Calculate total area of the region of interest
    total_area = region_of_interest.size
    
    # Calculate porosity (ratio of hole area to total area)
    porosity = hole_area / total_area if total_area > 0 else 0
    num_holes = len(contours)
    average_hole_size = hole_area / num_holes if num_holes > 0 else 0
    largest_hole_size = max([cv2.contourArea(c) for c in contours], default=0)
    smallest_hole_size = min([cv2.contourArea(c) for c in contours], default=0)
    
    hole_stats = {
        'num_holes': num_holes,
        'average_hole_size': average_hole_size,
        'largest_hole_size': largest_hole_size,
        'smallest_hole_size': smallest_hole_size
    }
    
    # Visualization of holes and contours
    # Create a color version of the original binary image for visualization
    # color_image = cv2.cvtColor(binary_image * 255, cv2.COLOR_GRAY2BGR)
    
    # # Offset the contours' x-values by adding the breakup_length to draw them on the original image
    # for contour in contours:
    #     contour[:, :, 0] += breakup_length  # Offset by the breakup length

    # # Fill each contour with a random color
    # for contour in contours:
    #     random_color = [random.randint(0, 255) for _ in range(3)]  # Generate a random color
    #     cv2.drawContours(color_image, [contour], -1, random_color, thickness=cv2.FILLED)

    # #Draw the contours outlines on the filled image (optional, but useful for clear visualization)
    # cv2.drawContours(color_image, contours, -1, (0, 0, 0), 1)
    
    # # Plot the original binary image and the visualization of holes
    # plt.figure(figsize=(10, 5))

    # # # Display the original binary image
    # # plt.subplot(1, 2, 1)
    # # plt.imshow(binary_image, cmap='gray')
    # # plt.title("Original Binary Image")
    
    # # # Display the image with holes highlighted and filled with different colors
    # # plt.subplot(1, 2, 2)
    # plt.imshow(color_image)
    # plt.axis('off')
    # # plt.title(f"Hole Porosity: {porosity:.4f}")
    # plt.tight_layout()
    
    # plt.show()
    
    return porosity, hole_stats

def calculate_fuel_density_entire_image(binary_image):
    """
    Calculates the fuel density (proportion of fuel pixels) for the entire binary image.
    
    Parameters:
    - binary_image: Binary image (fuel = 1, background = 0).
    
    Returns:
    - fuel_density: The proportion of fuel pixels in the entire image.
    """
    # Calculate fuel density (ratio of fuel pixels to total pixels)
    fuel_pixels = np.sum(binary_image == 255)
    total_pixels = binary_image.size
    
    fuel_density = fuel_pixels / total_pixels if total_pixels > 0 else 0
    
    return fuel_density


def calculate_fuel_density_regions(binary_image, x_regions, y_regions):
    """
    Calculates the fuel density (proportion of fuel pixels) for different regions of the binary image.
    
    Parameters:
    - binary_image: Binary image (fuel = 255, background = 0).
    - x_regions: Number of regions to divide the image along the x-axis (width).
    - y_regions: Number of regions to divide the image along the y-axis (height).
    
    Returns:
    - fuel_density_x_regions: List of fuel densities for each region along the x-axis.
    - fuel_density_y_regions: List of fuel densities for each region along the y-axis.
    """
    height, width = binary_image.shape
    fuel_density_x_regions = []
    fuel_density_y_regions = []

    # Calculate the size of each region
    region_height = height // y_regions
    region_width = width // x_regions

    # Loop over each region along the y-axis (horizontal slices)
    for y in range(y_regions):
        y_start = y * region_height
        y_end = (y + 1) * region_height if y != y_regions - 1 else height  # Handle remaining pixels for the last region

        # Extract the region
        region_y = binary_image[y_start:y_end, :]
        
        # Calculate the fuel density (fuel pixels / total pixels)
        fuel_pixels_y = np.sum(region_y == 255)  # Count fuel pixels (assuming fuel = 255)
        total_pixels_y = region_y.size  # Total pixels in the region
        fuel_density_y = fuel_pixels_y / total_pixels_y if total_pixels_y > 0 else 0
        fuel_density_y_regions.append(fuel_density_y)

    # Loop over each region along the x-axis (vertical slices)
    for x in range(x_regions):
        x_start = x * region_width
        x_end = (x + 1) * region_width if x != x_regions - 1 else width  # Handle remaining pixels for the last region

        # Extract the region
        region_x = binary_image[:, x_start:x_end]
        
        # Calculate the fuel density (fuel pixels / total pixels)
        fuel_pixels_x = np.sum(region_x == 255)  # Count fuel pixels (assuming fuel = 255)
        total_pixels_x = region_x.size  # Total pixels in the region
        fuel_density_x = fuel_pixels_x / total_pixels_x if total_pixels_x > 0 else 0
        fuel_density_x_regions.append(fuel_density_x)

    return fuel_density_x_regions, fuel_density_y_regions

def plot_cdf_fuel_density_all_regions(binary_image_list, x_regions, y_regions, resolution=9.3):
    """
    Plot separate CDFs for fuel density across different x and y regions across all frames,
    labeling each region by its central position in mm. The y-center is adjusted so that the middle
    slice has y=0, and the x-axis is flipped (x=0 is now x=600).
    
    Parameters:
    - binary_image_list: List of binary images across frames.
    - x_regions: Number of regions along the x-axis (width).
    - y_regions: Number of regions along the y-axis (height).
    - resolution: Pixel resolution in micrometers (default is 9.3 µm per pixel).
    """
    resolution_mm = resolution * 1e-3  # Convert from µm to mm
    height, width = binary_image_list[0].shape  # Assuming all images have the same dimensions
    
    # Calculate the center position of each x-region and y-region in mm
    region_height = height // y_regions
    region_width = width // x_regions

    # Flip x coordinates (x=600 becomes x=0)
    x_region_centers = [(width - (x * region_width + region_width // 2)) * resolution_mm for x in range(x_regions)]
    
    
    # Set y-center so the middle slice is 0 mm
    centerline = (height - 1) / 2  # Centerline of the image
    y_region_centers = [(y * region_height + region_height // 2 - centerline) * resolution_mm for y in range(y_regions)]
    
    # Precompute fuel density for all regions across all frames
    all_x_regions_density = [[] for _ in range(x_regions)]
    all_y_regions_density = [[] for _ in range(y_regions)]
    
    for binary_image in binary_image_list:
        fuel_density_x_regions, fuel_density_y_regions = calculate_fuel_density_regions(binary_image, x_regions, y_regions)
        
        for i in range(x_regions):
            all_x_regions_density[i].append(fuel_density_x_regions[i])
        
        for i in range(y_regions):
            all_y_regions_density[i].append(fuel_density_y_regions[i])

    # Plot the CDF for each x-region
    plt.figure(figsize=(10, 6))
    for i in range(x_regions):
        x_region_density_flattened = np.array(all_x_regions_density[i])
        sorted_x_density = np.sort(x_region_density_flattened)
        cdf_x_density = np.arange(1, len(sorted_x_density) + 1) / len(sorted_x_density)

        # Plot the CDF for this x-region, label by its center in mm
        plt.plot(sorted_x_density, cdf_x_density, label=f'X Center: {x_region_centers[i]:.2f} mm', linewidth=2)
    
    plt.xlabel('Fuel Density', fontsize=18)
    plt.ylabel('CDF', fontsize=18)
    plt.legend(loc='lower right', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.title('CDF of Fuel Density Across X Regions (Flipped X-axis)')
    plt.show()

    # Plot the CDF for each y-region
    plt.figure(figsize=(10, 6))
    for i in range(y_regions):
        y_region_density_flattened = np.array(all_y_regions_density[i])
        sorted_y_density = np.sort(y_region_density_flattened)
        cdf_y_density = np.arange(1, len(sorted_y_density) + 1) / len(sorted_y_density)

        # Plot the CDF for this y-region, label by its center in mm
        plt.plot(sorted_y_density, cdf_y_density, label=f'Y Center: {y_region_centers[i]:.2f} mm', linewidth=2)
    
    plt.xlabel('Fuel Density', fontsize=18)
    plt.ylabel('CDF', fontsize=18)
    plt.legend(loc='lower right', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.title('CDF of Fuel Density Across Y Regions (Y Center = 0 at Middle Slice)')
    plt.show()

def plot_average_image(image_list=None):
    """
    Computes and plots the average of the original images from the binary image list.
    
    Parameters:
    - image_list: List of binary images (fuel = 255, background = 0).
    
    Returns:
    - average_image: The averaged image.
    """
    # Check if the list is empty
    if image_list is None or len(image_list) == 0:
        raise ValueError("The image list is empty.")
    
    average_image = np.mean(image_list, axis=0)
    
    # Plot the average image
    plt.figure(figsize=(10, 6))
    plt.imshow(average_image, cmap='gray')
    plt.colorbar(label='Average Pixel Intensity')
    plt.title('Average of Original Images')
    plt.axis('off')
    plt.axvline(x=483, color='r', linestyle='--')
    plt.show()
    
    return average_image

def extract_and_plot_contours(binary_image, breakup_x=None):
    """
    Extracts and visualizes both external and internal contours from a binary image up to a given breakup x-coordinate.
    
    Parameters:
    - binary_image: Binary image (fuel = 255, background = 0).
    - breakup_x: The x-coordinate up to which contours will be extracted (for the breakup region).
    
    Returns:
    - contours: List of contours extracted from the binary image.
    """
    # Crop the region up to the breakup point
    if breakup_x is not None:
        region_of_interest = binary_image[:, breakup_x:]  # Corrected slicing
    else:
        region_of_interest = binary_image
    
    # Find contours in the region of interest (including internal contours)

    contours, hierarchy = cv2.findContours(region_of_interest, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create a color image for visualization
    color_image = cv2.cvtColor(region_of_interest, cv2.COLOR_GRAY2BGR)
    
    # Randomly color each contour
    for i, contour in enumerate(contours):
        random_color = [random.randint(0, 255) for _ in range(3)]
        cv2.drawContours(color_image, [contour], -1, random_color, 2)
    
    # Plot the contours
    plt.figure(figsize=(8, 12))
    plt.imshow(color_image)
    plt.title(f"Extracted Contours (Breakup Region up to x={breakup_x})")
    plt.axis('off')
    plt.show()
    
    return contours, hierarchy

def bfs_shortest_path(binary_image, start, end):
    """
    Perform BFS to find the shortest path between two points (start, end) without crossing the background (0 pixels).
    
    Parameters:
    - binary_image: Binary image (fuel = 255, background = 0).
    - start: Starting point (x, y) on the edge of the liquid sheet.
    - end: Target point (x, y) on the opposite edge of the liquid sheet.

    Returns:
    - path: List of points along the shortest path if found, or empty list if no path exists.
    """
    height, width = binary_image.shape
    queue = deque([(start, [start])])  # Queue of (current_point, path) tuples
    visited = set([start])  # Track visited points
    
    while queue:
        current_point, path = queue.popleft()
        
        if current_point == end:
            return path  # Path found
        
        x, y = current_point
        neighbors = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]  # 4-connected neighbors

        for nx, ny in neighbors:
            if 0 <= nx < width and 0 <= ny < height and binary_image[ny, nx] == 255 and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))  # Add neighbor to queue with updated path

    return []  # No path found

# No longer passing `max_distance_threshold` to `bfs_shortest_path`
def find_closest_opposite_contour_points(binary_image, contours, num_samples=5, min_distance_threshold=5, max_distance_threshold=25):
    """
    Finds the closest contour points across the same or different contours. Allows self-connections with distance constraints.
    
    Parameters:
    - binary_image: Binary image (fuel = 255, background = 0).
    - contours: List of contours extracted from the binary image.
    - num_samples: Number of points to sample on each contour.
    - min_distance_threshold: Minimum Euclidean distance to consider a valid path (to avoid internal loops).
    - max_distance_threshold: Maximum Euclidean distance to consider a valid path (to avoid distant paths).
    
    Returns:
    - shortest_distances: List of shortest distances between sampled contour points across different or valid opposite regions.
    - paths: List of paths showing the shortest valid paths between contours.
    """
    height, width = binary_image.shape
    shortest_distances = []
    paths = []

    # Iterate over each contour
    for contour_idx1, contour1 in enumerate(contours):
        sampled_points1 = contour1[::num_samples, 0, :]  # Sample points along the contour
        
        # Check distances between points within the same contour
        for point1 in sampled_points1:
            min_distance = float('inf')
            best_path = []

            # Compare the same contour to itself
            for point2 in sampled_points1:
                if np.array_equal(point1, point2):
                    continue  # Skip the same point

                # Compute Euclidean distance between the two points
                euclidean_distance = np.linalg.norm(point1 - point2)

                # Skip points that are too close or too far
                if euclidean_distance < min_distance_threshold or euclidean_distance > max_distance_threshold:
                    continue  # Ignore too-close or too-far points

                # Find shortest path using BFS, ensuring the path doesn't cross the background
                path = bfs_shortest_path(binary_image, tuple(point1), tuple(point2))
                if path:
                    path_distance = len(path)  # Number of steps in the path (as BFS is grid-based)
                    if path_distance < min_distance:
                        min_distance = path_distance
                        best_path = path

            if best_path:
                shortest_distances.append(min_distance)
                paths.append(best_path)  # Store the path for visualization

        # Check distances between this contour and all other contours
        for contour_idx2, contour2 in enumerate(contours):
            if contour_idx1 == contour_idx2:
                continue  # Skip self for now, as we already handled self-connection above

            sampled_points2 = contour2[::num_samples, 0, :]  # Sample points along the other contour

            # Check all pairs of points between the two contours
            for point1 in sampled_points1:
                min_distance = float('inf')
                best_path = []

                # Try to find the shortest path to any point on the second contour
                for point2 in sampled_points2:
                    # Compute Euclidean distance between the two points
                    euclidean_distance = np.linalg.norm(point1 - point2)

                    # Skip points that are too close or too far
                    if euclidean_distance < min_distance_threshold or euclidean_distance > max_distance_threshold:
                        continue  # Ignore too-close or too-far points

                    # Find shortest path using BFS, ensuring the path doesn't cross the background
                    path = bfs_shortest_path(binary_image, tuple(point1), tuple(point2))
                    if path:
                        path_distance = len(path)  # Number of steps in the path (as BFS is grid-based)
                        if path_distance < min_distance:
                            min_distance = path_distance
                            best_path = path

                if best_path:
                    shortest_distances.append(min_distance)
                    paths.append(best_path)  # Store the path for visualization

    return shortest_distances, paths

def visualize_shortest_paths_with_contours(binary_image, contours, paths, breakup_x=None):
    """
    Visualizes the extracted contours and overlays the shortest paths on the binary image.
    
    Parameters:
    - binary_image: Binary image (fuel = 255, background = 0).
    - contours: List of contours extracted from the binary image.
    - paths: List of paths to visualize (each path is a list of points).
    - breakup_x: The x-coordinate representing the breakup point for the visualization.
    """
    # Create a color version of the binary image
    color_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    
    # Draw contours in different random colors
    for contour in contours:
        random_color = [random.randint(0, 255) for _ in range(3)]
        cv2.drawContours(color_image, [contour], -1, random_color, 2)

    # Draw the paths in red
    for path in paths:
        for i in range(len(path) - 1):
            pt1 = path[i]
            pt2 = path[i + 1]
            cv2.line(color_image, pt1, pt2, (0, 0, 255), 1)  # Red line

    # Mark the breakup point
    if breakup_x is not None:
        cv2.line(color_image, (breakup_x, 0), (breakup_x, binary_image.shape[0]), (0, 255, 0), 2)  # Green vertical line for breakup point

    # Show the image with contours and paths
    plt.figure(figsize=(10, 12))
    plt.imshow(color_image)
    plt.title(f"Contours and Shortest Paths (up to x={breakup_x})")
    plt.axis('off')
    plt.show()

def visualize_individual_paths(binary_image, contours, paths, breakup_x=None):
    """
    Visualizes the extracted contours and overlays each path individually on the binary image for inspection.
    
    Parameters:
    - binary_image: Binary image (fuel = 255, background = 0).
    - contours: List of contours extracted from the binary image.
    - paths: List of paths to visualize (each path is a list of points).
    - breakup_x: The x-coordinate representing the breakup point for the visualization.
    """
    # Create a color version of the binary image
    color_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    
    # Draw contours in different random colors
    for contour in contours:
        random_color = [random.randint(0, 255) for _ in range(3)]
        cv2.drawContours(color_image, [contour], -1, random_color, 2)

    # Plot each path individually
    for idx, path in enumerate(paths):
        # Make a copy of the color image for each path visualization
        path_image = color_image.copy()

        # Draw the path in red
        for i in range(len(path) - 1):
            pt1 = path[i]
            pt2 = path[i + 1]
            cv2.line(path_image, pt1, pt2, (0, 0, 255), 2)  # Red line for the path

        # Mark the breakup point
        if breakup_x is not None:
            cv2.line(path_image, (breakup_x, 0), (breakup_x, binary_image.shape[0]), (0, 255, 0), 2)  # Green vertical line for breakup point

        # Show the image with the current path
        plt.figure(figsize=(10, 12))
        plt.imshow(path_image)
        plt.title(f"Path {idx + 1} (up to x={breakup_x})")
        plt.axis('off')
        plt.show()

#############################################################################################
from scipy.signal import savgol_filter
def main():
    # Define your experiment parameters
    config = FlowConfig(
        trial_path=r"D:\test_cases\UPF_A01_C_DP_35_trial_35",
        img_path=r"D:\final_corrected_512-complex-27-6-24.pth_inference",
        dir_ext=r'flow_npy\result_',
        step=1,
        start_x=0,
        end_x=None,
        start_y=0,
        end_y=None,
        reverse_flow=False,
        binary_image_analysis=True,
        warp_analysis=True,
        custom_range='end',
        hdf5_path='flow_data.h5'
    )
    
    # Initialize FlowAnalysis with the configuration
    flow_analysis = Fizi(config)
    
    try:
        # Process the data and save to HDF5 (uncomment if needed)
        # flow_analysis.process_and_save_data()
        
        # Optionally, load the data back from HDF5 (uncomment if needed)
        loaded_data = Fizi.load_from_hdf5(config.hdf5_path)
        print(f"Loaded data with keys: {loaded_data.keys()}")
        # process_data_with_dbscan_and_visualization(loaded_data)

        # previous_centroids = None
        # prev_region_sizes = None  # Initialize the first time as None

        # # Process each frame
        # for frame_idx in range(u_vectors.shape[0]):
        #     u = u_vectors[frame_idx]
        #     v = v_vectors[frame_idx]
        #     binary_mask = binary_images[frame_idx]
        #     orig_image=orig_images[frame_idx]
            
        #     # Step 1: Compute velocity magnitude
        #     velocity_magnitude = np.sqrt(u**2 + v**2)
            
        #     # Step 2: Segment regions using connected components on the binary mask (thresholded velocity magnitude)
        #     labels, num_labels = segment_regions(binary_mask)
            
        #     # Step 3: Compute region velocities and centroids for the current frame
        #     # region_velocities = compute_region_velocity(u, v, labels, num_labels)
        #     region_centroids = compute_region_centroids(labels, num_labels)
        #     region_sizes=compute_region_sizes(labels, num_labels)
            
        #     # Step 4: Detect breakup points by comparing current centroids with the previous frame's centroids
        #     # if previous_centroids is not None:
        #     #     breakup_points = detect_breakup_by_size(region_sizes, num_labels, threshold_change=0.5)
                
        #         # Visualize the breakup points
        #         # visualize_breakup(u, v, labels, breakup_points, region_centroids)
        #         # visualize_centroids(u, v, region_centroids)
        #         # visualize_colored_regions(binary_mask, u, v, labels, breakup_points, num_labels, original_image=orig_image)
            
        #     # If there are previous regions, match them and detect breakup
        #     if previous_centroids is not None and prev_region_sizes is not None:
        #         matches = match_regions_by_centroids(region_centroids, previous_centroids)
        #         print(f"Frame {frame_idx}: Matches found between regions: {matches}")
                
        #         matched_current_sizes = [region_sizes[i] for i, _ in matches]
        #         matched_prev_sizes = [prev_region_sizes[j] for _, j in matches]
                
        #         # Detect breakup points based on size change
        #         breakup_points = detect_breakup_by_size(region_sizes, prev_region_sizes, matches, threshold_change=0.1, num_labels=num_labels)
        #         print(f"Frame {frame_idx}: Breakup points: {breakup_points}")
    
        #         visualize_colored_regions(binary_mask, u, v, labels, breakup_points, num_labels, original_image=orig_image)

            
        #     # Update the previous centroids for tracking
        #     previous_centroids = region_centroids
        #     prev_region_sizes = region_sizes

        # Prepare directories and lists for warped images
        
        # Generate flow lists and save warped images
        # flow_vis_list, img_list, warped_img_list, gradient_list, binary_image_list, x, y = flow_analysis.create_flow_lists(
        #     config.trial_path, 
        #     config.img_path, 
        #     config.dir_ext, 
        #     step=config.step, 
        #     start_x=config.start_x, 
        #     warp=config.warp_analysis, 
        #     binary_image=config.binary_image_analysis,
        #     custom_range=config.custom_range, flow_vis_type='basic'
        # )
        
        # flow_analysis.plot_and_save_losses()
        # flow_analysis.generate_global_heatmaps(gradient_list)
        # flow_analysis.average_heatmaps_with_confidence_intervals(gradient_list)
        # flow_analysis.save_warped_images(warped_img_list)
        # print('done with initial part')
        # flow_extraction=Flay(config, flow_vis_list, binary_image_list, image_list=img_list, x=x, y=y)
        # flow_extraction.save_flow_vectors(apply_mask=True, flip=True)
        # flow_extraction.plot_vorticity(save_dir=config.trial_path, save_data='vorticity')
        # flow_extraction.plot_and_save_flowmaps(edge_mask=True, flip=True)
        # # Plot shear stress
        # flow_extraction.plot_shear_stress(save_dir=config.trial_path, save_data='shear_stress')

        # # Plot strain rate
        # flow_extraction.plot_strain_rate(save_dir=config.trial_path, save_data='strain_rate')

        ################################## CDF code ##################################
        # print('created lists')
        # binary_image_list = loaded_data['binary_image_list'] 
        # print(binary_image_list[0].max())
        # img_list = loaded_data['img_list']
        # u_vectors = loaded_data['u_vectors']
        # v_vectors = loaded_data['v_vectors']
        # print(u_vectors.shape)
        # mask = binary_image_list == 255  # Create a mask where white regions are True
        # u_filtered = np.where(mask, u_vectors, np.nan)  # Set non-white regions to NaN or 0 if preferred
        # v_filtered = np.where(mask, v_vectors, np.nan)
        # num_columns_to_plot = 5
        # num_rows_to_plot = 5
        # plot_cdf_velocity_all_frames(u_filtered, v_filtered, num_columns_to_plot, axis='y-slices')
        # plot_cdf_velocity_all_frames(u_filtered, v_filtered, num_rows_to_plot, axis='x-slices')
        ################################## start of breakup length ##################################
        binary_image_list = loaded_data['binary_image_list']
        #save binary_image_list[0] as a png
        cv2.imwrite('binary_image_ex.png',binary_image_list[0])
        image_list=loaded_data['img_list']
        raw_x_y_values, smoothed_x_y_values, frame_idx_breakup, best_segment_list = process_breakup_length(binary_image_list=binary_image_list, breakup_x_threshold=50)
        print('done with breakup length')
        # Visualize hole propagation for each binary image
        porosity_values = []
        fuel_density= []
        print(f'Length of binary_image_list: {len(binary_image_list)}')
        print(f'Length of smoothed_x_y_values: {len(smoothed_x_y_values)}')
        # Visualize hole propagation and calculate porosity for each binary image
        for frame_idx, binary_image in enumerate(binary_image_list):
            if frame_idx < len(smoothed_x_y_values):  # Bounds check to prevent out-of-bounds error
                porosity,_ = calculate_hole_porosity_with_visualization(binary_image, smoothed_x_y_values[frame_idx])
                porosity_values.append(porosity)  # Store the porosity for each frame
                fuel_d=calculate_fuel_density_entire_image(binary_image)
                fuel_density.append(fuel_d)
                breakup_x = int(smoothed_x_y_values[0])  # Convert to integer if it's a float
                # average_thickness, thickness_samples = measure_shortest_radial_thickness(binary_image_list[0], breakup_x, num_samples=100)
                # print(f"Average thickness: {average_thickness:.2f} pixels")
                # print(f"Thickness samples: {thickness_samples}")
            else:
                print(f"Skipping frame {frame_idx} as it's out of bounds for smoothed_x_y_values.")

        # binary_image = binary_image_list[0]  # Use the first image
        # breakup_x=int(smoothed_x_y_values[0])

        # contours,_ = extract_and_plot_contours(binary_image, breakup_x=breakup_x)

        # # Find closest contour points without passing through the background
        # # Step 2: Measure shortest paths across distinct regions of contours
        # shortest_distances, paths = find_closest_opposite_contour_points(binary_image[:, breakup_x:], contours, min_distance_threshold=1)
        # # Check if we have any distances calculated
        # if shortest_distances:
        #     # Calculate the average shortest distance
        #     average_shortest_distance = sum(shortest_distances) / len(shortest_distances)
        #     print(f"Average Shortest Distance: {average_shortest_distance:.2f} pixels")
        # else:
        #     print("No shortest distances were calculated.")
        # # Step 3: Visualize the contours and overlay the shortest paths
        # visualize_shortest_paths_with_contours(binary_image[:, breakup_x:], contours, paths, breakup_x=breakup_x)
        # visualize_individual_paths(binary_image[:, breakup_x:], contours, paths, breakup_x=breakup_x)

        # # find the average of the porosity values
        # print('porosity:',np.mean(porosity_values))
        # # # Plot porosity over time (frame index)
        # plt.figure(figsize=(10, 6))
        # plt.plot(range(len(porosity_values)), porosity_values, marker='o', linestyle='-', color='b')
        # plt.xlabel('Frame Index (Time)')
        # plt.ylabel('Hole Porosity')
        # plt.title('Hole Porosity Over Time')
        # plt.grid(True)
        # plt.show()

        # plt.figure(figsize=(10, 6))
        # plt.plot(range(len(fuel_density)), fuel_density, marker='o', linestyle='-', color='g')
        # plt.xlabel('Frame Index (Time)')
        # plt.ylabel('Fuel Density')
        # plt.title('Fuel Density Over Time')
        # plt.grid(True)
        # plt.show()

        # num_slices = 5  # Number of rows/columns to calculate fuel density for
        # axis = 'y-slices'  # You can change to 'x-slices' if needed
        # axis2='x-slices'

        # # Plot the CDF for fuel density across all frames
        # x_regions = 5  # Specify how many regions you want along the x-axis (width)
        # y_regions = 5  # Specify how many regions you want along the y-axis (height)

        # # Plot the CDFs for the x and y regions
        # plot_cdf_fuel_density_all_regions(binary_image_list, x_regions, y_regions)
        
        # from numpy.polynomial.polynomial import Polynomial
        # x=plot_average_image(image_list=binary_image_list)
        # average_x=np.mean(x, axis=0)
        # # print(len(average_x))

        # degree = 5  # You can adjust this degree to capture the main trend and peak
        # p = Polynomial.fit(range(len(average_x)), average_x, degree)

        # # Evaluate the fitted polynomial
        # average_x_fitted = p(range(len(average_x)))

        # # Compute first and second derivatives of the fitted curve
        # dx_fitted = np.gradient(average_x_fitted)
        # d2x_fitted = np.gradient(dx_fitted)

        # zero_crossings = np.where(np.diff(np.sign(d2x_fitted)))[0]
        # print('zero_crossings:',zero_crossings) 

        # # Plot the original data and fitted curve
        # plt.figure(figsize=(8, 6))
        # plt.plot(average_x, label="Original Data", color='lightgray')
        # plt.plot(average_x_fitted, label="Fitted Polynomial", color='blue')
        # plt.scatter(zero_crossings, average_x_fitted[zero_crossings], color='red', label='Zero Crossing of 2nd Derivative')
        # plt.title('Fitted Curve vs Original Data')
        # plt.legend()
        # plt.show()

        # # Plot the first derivative of the fitted curve
        # plt.figure(figsize=(8, 6))
        # plt.plot(dx_fitted, label="Fitted First Derivative", color='blue')
        # plt.scatter(zero_crossings, dx_fitted[zero_crossings], color='red', label='Zero Crossing of 2nd Derivative')
        # plt.title('First Derivative of Fitted Curve')
        # plt.legend()
        # plt.show()

        # # Plot the second derivative of the fitted curve and highlight zero crossings
        # plt.figure(figsize=(8, 6))
        # plt.plot(d2x_fitted, label="Fitted Second Derivative", color='orange')
        # plt.scatter(zero_crossings, d2x_fitted[zero_crossings], color='red', label='Zero Crossings of 2nd Derivative')
        # plt.title('Second Derivative of Fitted Curve')
        # plt.legend()
        # plt.show()




        
        
        # Smooth the x-values using a running average
        window_size = 100  # Window size for the running average
        # smoothed_x_y_values = running_average(smoothed_x_y_values, window_size=window_size)
        x_values = np.arange(window_size - 1, len(smoothed_x_y_values) + (window_size - 1))  # Correct the x-values

        # Adjust frame_idx_breakup to exclude values that are now out of bounds after smoothing
        # frame_idx_breakup = frame_idx_breakup[frame_idx_breakup < len(smoothed_x_y_values) + (window_size - 1)]
        def fourier_series(x, *a):
            result = np.zeros_like(x)
            N = len(a) // 2
            for n in range(N):
                result += a[2*n] * np.sin((n + 1) * 2 * np.pi * x / len(x)) + a[2*n + 1] * np.cos((n + 1) * 2 * np.pi * x / len(x))
            return result

        smoothed_x_y_values = np.array(smoothed_x_y_values)
        # smoothed_x_y_values= width-smoothed_x_y_values
        smoothed_x_y_values = 9.3*smoothed_x_y_values*10**-3

        # Step 1: Subtract the mean from the data
        data_mean = np.mean(smoothed_x_y_values)
        mean_adjusted_data = smoothed_x_y_values - data_mean

        # Generate x-values for the mean-adjusted data
        x_values = np.arange(len(mean_adjusted_data))

        # Initial guess for the Fourier coefficients (up to 3 harmonics)
        initial_guess = [1] * 12  # For a Fourier series with 3 terms (sin and cos)

        # Step 2: Perform the curve fitting to fit the Fourier series to your mean-adjusted data
        params, params_covariance = curve_fit(fourier_series, x_values, mean_adjusted_data, p0=initial_guess)

        # Step 3: Reconstruct the continuous wave using the fitted Fourier series
        x_dense = np.linspace(x_values.min(), x_values.max(), 1000)  # Higher density of x-values for smooth wave
        fitted_wave = fourier_series(x_dense, *params)

        # Step 4: Add the mean back to the fitted wave
        fitted_wave_with_mean = fitted_wave + data_mean

        ######################## this section above was all used previously ############################
        # Plot the original smoothed data and the fitted Fourier series wave
        # plt.figure(figsize=(10, 6))
        # plt.plot(x_values, smoothed_x_y_values, linestyle='-', color='blue')  # Original smoothed data
        # plt.axhline(y=data_mean, color='red', linestyle='--', label=f'Mean breakup length: {data_mean:.2f} mm')  # Mean value
        # # plt.plot(x_dense, fitted_wave_with_mean, label='Fitted Fourier Series', linestyle='-', color='green')  # Fitted Fourier series

        # # Add titles and labels
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.xlabel('Time [s]', fontsize=18)
        # plt.ylabel('Breakup Length [mm]', fontsize=18)
        # plt.legend(fontsize=16)
        # plt.tight_layout()
        # # plt.grid(True)
        # plt.show()
        

        # print('fourier_params:', params)


        # # Create x-values representing the frame indices
        # x_values = np.arange(len(smoothed_x_y_values))  # Assuming frame indices are sequential

        # # Create a cubic interpolation function
        # interp_func = interp1d(x_values, smoothed_x_y_values, kind='cubic')

        # # Generate new x-values for a denser, smoother curve
        # x_dense = np.linspace(x_values.min(), x_values.max(), num=10000)  # 1000 points for a smooth curve

        # # Generate new smoothed y-values using the interpolation function
        # smoothed_x_y_dense = interp_func(x_dense)

        # # Plot the original smoothed x-values and the denser interpolated curve
        # plt.figure(figsize=(10, 6))

        # # # Plot the original raw x-values (before modification)
        # # plt.plot(x_values, raw_x_y_values, color='red', label='Raw X Value Over Time')

        # # Plot the interpolated, denser smoothed curve
        # plt.plot(x_values, smoothed_x_y_values, color='green', linestyle='-', label='Interpolated Smoothed X Value Over Time')

        # # Scatter plot of the breakup frame points (ensure indices are valid)
        # plt.scatter(frame_idx_breakup, smoothed_x_y_values[frame_idx_breakup], color='black', label='Breakup Frame')

        # plt.xlabel('Frame Index')
        # plt.ylabel('Average X Value')
        # plt.title('Raw vs Interpolated Smoothed X Values Over Time')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # dx = np.gradient(smoothed_x_y_values)  # First derivative (rate of change)
        # d2x = np.gradient(dx)  # Second derivative (acceleration)

        # # Create x-values representing the frame indices (for smoothed dense data)
        # x_dense = np.linspace(0, len(smoothed_x_y_values), num=len(smoothed_x_y_values))

        # # Plot the original smoothed x-values
        # plt.figure(figsize=(10, 6))
        # plt.plot(x_dense, smoothed_x_y_values, color='green', label='Smoothed X Value')

        # # Plot the first derivative (rate of change)
        # plt.plot(x_dense, dx, color='blue', label='First Derivative (Rate of Change)')

        # # Plot the second derivative (acceleration)
        # plt.plot(x_dense, d2x, color='orange', label='Second Derivative (Acceleration)')

        # plt.xlabel('Frame Index')
        # plt.ylabel('X Value / Derivative')
        # plt.title('X Values and Their Derivatives Over Time')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        breakup_length_data = np.array(smoothed_x_y_values)

        # # Compute and plot autocorrelation
        plt.figure(figsize=(10, 6))
        plot_acf(breakup_length_data, lags=100, title="Autocorrelation of Breakup Length")
        plt.xlabel('Lag (Frame Index)')
        plt.ylabel('Autocorrelation')
        plt.show()
        fft_result = np.fft.fft(breakup_length_data)
        fft_magnitude = np.abs(fft_result)  # Get the magnitude of the Fourier coefficients
        fft_freq = np.fft.fftfreq(len(breakup_length_data))  # Compute the frequency bins

        data_demeaned = breakup_length_data - np.mean(breakup_length_data)

        # Perform Fourier Transform
        fft_result = np.fft.fft(data_demeaned)
        fft_magnitude = np.abs(fft_result)  # Get the magnitude of the Fourier coefficients
        fft_freq = np.fft.fftfreq(len(breakup_length_data))  # Compute the frequency bins

        # Plot the magnitude of the Fourier transform
        plt.figure(figsize=(10, 6))
        plt.plot(fft_freq, fft_magnitude)
        plt.title('Fourier Transform of Demeaned Breakup Length Data')
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.show()

        # import pywt
        # wavelet = 'mexh'

        # # Define the scales for the wavelet transform
        # scales = np.arange(1, 128)

        # # Perform the continuous wavelet transform (CWT)
        # coefficients, frequencies = pywt.cwt(smoothed_x_y_values, scales, wavelet)

        # # Plot the CWT coefficients as a heatmap to analyze time-frequency features
        # plt.figure(figsize=(10, 6))
        # plt.imshow(coefficients, extent=[0, len(smoothed_x_y_values), 1, 128], cmap='PRGn', aspect='auto',
        #         vmax=abs(coefficients).max(), vmin=-abs(coefficients).max())
        # plt.title("Continuous Wavelet Transform (CWT) - Mexican Hat Wavelet")
        # plt.ylabel("Scale (Related to Frequency)")
        # plt.xlabel("Time (Frame Index)")
        # plt.colorbar(label="Magnitude")
        # plt.show()

        # frame_window = 100
        # num_frames = len(smoothed_x_y_values)

        # # Remove duplicate frame indices in frame_idx_breakup
        # frame_idx_breakup = np.array(frame_idx_breakup, dtype=int)
        # frame_idx_breakup = np.unique(frame_idx_breakup)

        # # Loop through and plot frames in 500-frame chunks
        # for start_idx in range(0, num_frames, frame_window):
        #     end_idx = min(start_idx + frame_window, num_frames)  # Ensure we don't exceed the number of frames

        #     # Extract data for the current window
        #     x_values_window = x_values[start_idx:end_idx]
        #     raw_x_y_values_window = raw_x_y_values[start_idx:end_idx]
        #     smoothed_x_y_values_window = smoothed_x_y_values[start_idx:end_idx]
            
        #     # Get frame indices for the breakup points that are within the current window
        #     frame_idx_breakup_window = frame_idx_breakup[(frame_idx_breakup >= start_idx) & (frame_idx_breakup < end_idx)]
            
        #     # Adjust frame_idx_breakup_window to relative index within this window
        #     frame_idx_breakup_window = frame_idx_breakup_window - start_idx
            
        #     # Modify values at breakup points by averaging previous and next points within this window
        #     for idx in frame_idx_breakup_window:
        #         # Ensure we are not out of bounds (idx must be between 1 and len(smoothed_x_y_values_window) - 2)
        #         if 1 <= idx < len(smoothed_x_y_values_window) - 1:
        #             avg_value = (smoothed_x_y_values_window[idx - 1] + smoothed_x_y_values_window[idx + 1]) / 2
        #             smoothed_x_y_values_window[idx] = avg_value

        #     # Generate new x-values for a denser, smoother curve within this window
        #     x_dense_window = np.linspace(x_values_window.min(), x_values_window.max(), num=1000)

        #     # Create a cubic interpolation function for this window
        #     interp_func = interp1d(x_values_window, smoothed_x_y_values_window, kind='cubic')

        #     # Generate new smoothed y-values using the interpolation function
        #     smoothed_x_y_dense_window = interp_func(x_dense_window)

        #     # Plot the original smoothed x-values and the denser interpolated curve for this window
        #     plt.figure(figsize=(10, 6))

        #     # Plot the original raw x-values (before modification) for this window
        #     plt.plot(x_values_window, raw_x_y_values_window, color='red', label='Raw X Value Over Time')

        #     # Plot the interpolated, denser smoothed curve for this window
        #     plt.plot(x_dense_window, smoothed_x_y_dense_window, color='green', linestyle='-', label='Interpolated Smoothed X Value Over Time')

        #     # Scatter plot of the breakup frame points (only within this window)
        #     plt.scatter(frame_idx_breakup_window, smoothed_x_y_values_window[frame_idx_breakup_window], color='black', label='Breakup Frame')

        #     plt.xlabel('Frame Index')
        #     plt.ylabel('Average X Value')
        #     plt.title(f'Raw vs Interpolated Smoothed X Values: Frames {start_idx} to {end_idx}')
        #     plt.legend()
        #     plt.grid(True)
        #     plt.show()



        # # plot the avg x values over all the best segments over time
        # # avg_x_values = [np.mean(segment[0]) for segment in best_segment_list]
        # x_values = np.arange(len(avg_x_y_values))  # Assuming frame indices are sequential

        # # Create a cubic interpolation function
        # interp_func = interp1d(x_values, avg_x_y_values, kind='cubic')

        # # Generate new x-values for a denser, smoother curve
        # x_dense = np.linspace(x_values.min(), x_values.max(), num=1000)  # 1000 points for a smooth curve

        # # Generate new smoothed y-values using the interpolation function
        # smoothed_avg_x_values = interp_func(x_dense)

        # # Plot the interpolated, smoother curve
        # plt.figure(figsize=(10, 6))
        # plt.plot(x_dense, smoothed_avg_x_values, color='blue', label='Smoothed Average X Value Over Time')
        # plt.xlabel('Frame Index')
        # plt.ylabel('Average X Value')
        # plt.title('Smoothed Average X Value Over Time (Cubic Interpolation)')
        # plt.legend()
        # plt.grid(True)
        # plt.show()


        




        # fps = 10  # Frames per second for the video
        # output_file = 'breakup_length_video_original.mp4'

        # # Initialize video writer with codec
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify codec for MP4
        # height, width = img_list[0].shape  # Assuming all original images are the same size
        # out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        # # Iterate through each original image and plot breakup length
        # def draw_dotted_line(img, x, color, thickness=2, dot_length=10, gap_length=5):
        #     for y in range(0, height, dot_length + gap_length):
        #         cv2.line(img, (x, y), (x, min(y + dot_length, height)), color, thickness)

        # # Iterate through each original image and plot breakup length
        # for frame_idx, (original_image, breakup_length) in enumerate(zip(img_list, smoothed_x_y_values)):
        #     # Flip the original image horizontally
        #     flipped_image = cv2.flip(original_image, 1)

        #     # Convert to RGB for overlay
        #     flipped_image_rgb = cv2.cvtColor(flipped_image, cv2.COLOR_GRAY2RGB)

        #     # Convert breakup length from mm to pixels (based on the width conversion factor)
        #     breakup_pixel = int(width - (breakup_length / 9.3 * 1000))  # Convert mm to pixel

        #     # Overlay the breakup length on the flipped image as a red dotted vertical line
        #     draw_dotted_line(flipped_image_rgb, breakup_pixel, color=(0, 0, 255), thickness=2)

        #     # Write the frame to the video
        #     out.write(flipped_image_rgb)

        # # Release the video writer and close the file
        # out.release()

        # print(f"Video saved as {output_file}")



    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

# base_path = r"D:\test_cases"
# # Define a list to hold all the data
# all_data = []
# missing_trials = [9, 11, 13, 14, 15, 18, 19, 20, 22]
# # Iterate through the trials
# for trial_number in range(1, 25):  # Assuming you have trials numbered 1 to 24
#     if trial_number in missing_trials:
#         print(f"Skipping trial {trial_number} because it is missing.")
#         continue

#     trial_folder = f'UPF_A01_C_DP_35_trial_{trial_number}'
#     trial_path = os.path.join(base_path, trial_folder)

#     if not os.path.exists(trial_path):
#         print(f"Trial {trial_number} directory not found: {trial_path}")
#         continue

#     pth_files = [f for f in os.listdir(trial_path) if f.endswith('.pth')]

#     # Process pth_files as needed
#     print(f"Found {len(pth_files)} .pth files in trial {trial_number}.")
    
#     if pth_files:
#         model_path = os.path.join(trial_path, pth_files[0])  # Assuming there's only one .pth file per folder
        
#         state_dict = torch.load(model_path)
#         training_params = state_dict['training_params']
#         param_dict = state_dict['model_params']

#         # Combine both dictionaries
#         combined_params = {**training_params, **param_dict}
        
#         # Add trial number to each parameter for identification
#         combined_params['Trial'] = f"trial_{trial_number}"
        
#         # Append the dictionary to the list
#         all_data.append(combined_params)

# # Convert the list of dictionaries to a DataFrame
# df = pd.DataFrame(all_data)

# # Save the DataFrame to a CSV file
# csv_path = os.path.join(base_path, "all_parameters1.csv")
# df.to_csv(csv_path, index=False)

# print(f'All parameters saved to {csv_path}')

def consolidate_error_metrics(base_path, output_filename, missing_trials=None,):
    """
    Consolidates error metrics from multiple trial directories into a single CSV file.

    Parameters:
    - base_path (str): The base directory where trial folders are stored.
    - missing_trials (list): List of trial numbers to skip because they are missing.
    - output_filename (str): The name of the output CSV file to save the consolidated metrics.

    Returns:
    - None
    """
    if missing_trials is None:
        missing_trials=[]
    # Define a list to hold all the data
    all_data = []

    # Iterate through the trials
    for trial_number in range(1, 29):  # Assuming you have trials numbered 1 to 24
        if trial_number in missing_trials:
            print(f"Skipping trial {trial_number} because it is missing.")
            continue

        trial_folder = f'UPF_A01_C_DP_35_trial_{trial_number}'
        trial_path = os.path.join(base_path, trial_folder)

        if not os.path.exists(trial_path):
            print(f"Trial {trial_number} directory not found: {trial_path}")
            continue

        # Look for the error_metrics.txt file
        error_metrics_path = os.path.join(trial_path, 'error_metrics.txt')
        
        if os.path.exists(error_metrics_path):
            print(f"Processing error metrics for trial {trial_number}.")
            
            # Read the error metrics file
            with open(error_metrics_path, 'r') as f:
                metrics = f.read().strip().split('\n')
            
            # Parse the metrics into a dictionary
            metrics_dict = {}
            for metric in metrics:
                key, value = metric.split(':')
                metrics_dict[key.strip()] = float(value.strip())
            
            # Add trial number to the dictionary for identification
            metrics_dict['Trial'] = f"trial_{trial_number}"
            
            # Append the dictionary to the list
            all_data.append(metrics_dict)
        else:
            print(f"No error_metrics.txt found for trial {trial_number}.")
            
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(all_data)

    # Save the DataFrame to a CSV file
    csv_path = os.path.join(base_path, output_filename)
    df.to_csv(csv_path, index=False)

    print(f'All error metrics saved to {csv_path}')

# Example usage
# base_path = r"D:\test_cases"
# missing_trials = None
# output_filename = "consolidated_error_metrics.csv"

# consolidate_error_metrics(base_path, output_filename, missing_trials=missing_trials)


#############################################################################################
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

def apply_dbscan_with_velocity(img, u_vectors, v_vectors, eps=5, min_samples=5):
    """Apply DBSCAN to raw image pixels and velocity vectors."""
    # Get coordinates of all pixels
    height, width = img.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Flatten all arrays to create feature vectors for each pixel
    pixel_coords = np.stack((x.ravel(), y.ravel()), axis=1)
    pixel_intensity = img.ravel().reshape(-1, 1)  # Grayscale intensity of each pixel
    velocity_u = u_vectors.ravel().reshape(-1, 1)  # Horizontal velocity (u)
    velocity_v = v_vectors.ravel().reshape(-1, 1)  # Vertical velocity (v)

    # Feature vector: [x, y, intensity, u_velocity, v_velocity]
    features = np.hstack((pixel_coords, pixel_intensity, velocity_u, velocity_v))

    # Apply DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(features)

    # Reshape labels to match the original image
    labels = db.labels_.reshape(img.shape)

    return labels

def track_regions(labels_prev, labels_curr, u_prev, v_prev, u_curr, v_curr, threshold=10.0):
    """Track regions between frames and detect breakups."""
    # Get centroids of regions in the previous and current frames
    centroids_prev = compute_region_centroids(labels_prev)
    centroids_curr = compute_region_centroids(labels_curr)

    # Match regions based on centroid proximity
    distances = cdist(centroids_prev, centroids_curr)
    matches = []
    for i, row in enumerate(distances):
        min_dist = np.min(row)
        if min_dist < threshold:
            j = np.argmin(row)  # Closest region in current frame
            matches.append((i, j))

    # Check for breakups (one region in prev splitting into multiple in curr)
    breakups = []
    for i, j in matches:
        # Compare velocities of the matched regions
        velocity_prev = np.mean(np.stack([u_prev[labels_prev == i], v_prev[labels_prev == i]]), axis=1)
        velocity_curr = np.mean(np.stack([u_curr[labels_curr == j], v_curr[labels_curr == j]]), axis=1)
        velocity_diff = np.linalg.norm(velocity_prev - velocity_curr)

        # If velocity difference is large, consider it a breakup
        if velocity_diff > threshold:
            breakups.append((i, j))

    return matches, breakups

def compute_region_centroids(labels):
    """Compute centroids of labeled regions."""
    centroids = []
    for label in np.unique(labels):
        if label == -1:  # Skip noise
            continue
        mask = labels == label
        y, x = np.where(mask)
        centroids.append((np.mean(x), np.mean(y)))
    return np.array(centroids)