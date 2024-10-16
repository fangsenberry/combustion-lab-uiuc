import sys
import os
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
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit

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
    - binary_image: Binary image (fuel = 255, background = 0).
    
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

def extract_and_plot_contours(binary_image, breakup_x_pixel=None):
    """
    Extracts and visualizes both external and internal contours from a binary image up to a given breakup x-coordinate.
    
    Parameters:
    - binary_image: Binary image (fuel = 255, background = 0).
    - breakup_x_pixel: The x-coordinate up to which contours will be extracted (for the breakup region in pixels).
    
    Returns:
    - contours: List of contours extracted from the binary image.
    """
    # Crop the region up to the breakup point
    if breakup_x_pixel is not None:
        region_of_interest = binary_image[:, breakup_x_pixel:]  # Corrected slicing
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
    plt.title(f"Extracted Contours (Breakup Region up to x={breakup_x_pixel})")
    plt.axis('off')
    plt.show()
    
    return contours, hierarchy

def generate_breakup_length_video(img_list, breakup_x_pixel, output_file='breakup_length_video.mp4', fps=10, width_conversion_factor=9.3):
    """
    Generate a video with a red dotted line indicating breakup length on each frame.

    Parameters:
    - img_list: List of original grayscale images.
    - breakup_x_value: List of breakup x-values for each frame in pixels
    - output_file: Name of the output video file.
    - fps: Frames per second for the video.
    - width_conversion_factor: Conversion factor to convert mm to pixels based on image width.

    Returns:
    - None
    """
    # Initialize video writer with codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify codec for MP4
    height, width = img_list[0].shape  # Assuming all original images are the same size
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Function to draw a dotted line
    def draw_dotted_line(img, x, color, thickness=2, dot_length=10, gap_length=5):
        for y in range(0, height, dot_length + gap_length):
            cv2.line(img, (x, y), (x, min(y + dot_length, height)), color, thickness)

    # Iterate through each original image and plot breakup length
    for frame_idx, (original_image, breakup_length) in enumerate(zip(img_list, breakup_x_pixel)):
        # Flip the original image horizontally
        flipped_image = cv2.flip(original_image, 1)

        # Convert to RGB for overlay
        flipped_image_rgb = cv2.cvtColor(flipped_image, cv2.COLOR_GRAY2RGB)

        # Convert breakup length from mm to pixels (based on the width conversion factor)
        

        # Overlay the breakup length on the flipped image as a red dotted vertical line
        draw_dotted_line(flipped_image_rgb, breakup_x_pixel, color=(0, 0, 255), thickness=2)

        # Write the frame to the video
        out.write(flipped_image_rgb)

    # Release the video writer and close the file
    out.release()

    print(f"Video saved as {output_file}")


def analyze_curve_with_zero_crossings(data, degree=5):
    """
    Fits a polynomial to the input data, calculates first and second derivatives, 
    identifies zero crossings of the second derivative, and plots the results.
    
    Parameters:
    - data: The input data (1D array or list).
    - degree: Degree of the polynomial to fit to the data (default is 5).
    
    Returns:
    - zero_crossings: Indices where the second derivative crosses zero.
    """
    # Step 1: Fit a polynomial to the data
    p = Polynomial.fit(range(len(data)), data, degree)
    
    # Step 2: Evaluate the fitted polynomial
    data_fitted = p(range(len(data)))

    # Step 3: Compute first and second derivatives of the fitted curve
    dx_fitted = np.gradient(data_fitted)
    d2x_fitted = np.gradient(dx_fitted)

    # Step 4: Find where the second derivative is approximately zero (zero crossings)
    zero_crossings = np.where(np.diff(np.sign(d2x_fitted)))[0]
    
    # Step 5: Print the zero crossings (for reference)
    print('Zero Crossings of 2nd Derivative:', zero_crossings)

    # Step 6: Plot the original data and fitted curve
    plt.figure(figsize=(8, 6))
    plt.plot(data, label="Original Data", color='lightgray')
    plt.plot(data_fitted, label="Fitted Polynomial", color='blue')
    plt.scatter(zero_crossings, data_fitted[zero_crossings], color='red', label='Zero Crossing of 2nd Derivative')
    plt.title('Fitted Curve vs Original Data')
    plt.legend()
    plt.show()

    # Step 7: Plot the first derivative of the fitted curve
    plt.figure(figsize=(8, 6))
    plt.plot(dx_fitted, label="Fitted First Derivative", color='blue')
    plt.scatter(zero_crossings, dx_fitted[zero_crossings], color='red', label='Zero Crossing of 2nd Derivative')
    plt.title('First Derivative of Fitted Curve')
    plt.legend()
    plt.show()

    # Step 8: Plot the second derivative of the fitted curve and highlight zero crossings
    plt.figure(figsize=(8, 6))
    plt.plot(d2x_fitted, label="Fitted Second Derivative", color='orange')
    plt.scatter(zero_crossings, d2x_fitted[zero_crossings], color='red', label='Zero Crossings of 2nd Derivative')
    plt.title('Second Derivative of Fitted Curve')
    plt.legend()
    plt.show()
    
    return zero_crossings

def rosin_rammler_cdf(d, d0, q):
    """
    CDF of Rosin-Rammler distribution.
    
    Parameters:
    - d: Independent variable (e.g., size or data values).
    - d0: Characteristic size.
    - q: Spread parameter.
    
    Returns:
    - CDF value for the given size.
    """
    return 1 - np.exp(-(d / d0) ** q)

def rosin_rammler(d, d_m, n):
    """
    Rosin-Rammler equation for fitting CDF-like data.
    
    Parameters:
    - d: Independent variable (e.g., size, distance).
    - d_m: Characteristic size (scale parameter).
    - n: Shape parameter (spread).
    
    Returns:
    - R(d): Cumulative percentage or fraction for the given independent variable.
    """
    return 100 * np.exp(- (d / d_m) ** n)

def fit_rosin_rammler_to_data(data, cumulative_percent):
    """
    Fits the Rosin-Rammler distribution to the given data.

    Parameters:
    - data: The independent variable (e.g., x-axis values, like size or distance).
    - cumulative_percent: The dependent variable (y-values, like cumulative percentage).

    Returns:
    - Fitted parameters (d_m, n)
    - Fitted curve (for plotting)
    """
    # Initial guess for d_m and n (you can adjust based on data behavior)
    initial_guess = [np.mean(data), 1.5]  # Guess values can be tuned

    # Fit the Rosin-Rammler equation to the data
    params, covariance = curve_fit(rosin_rammler, data, cumulative_percent, p0=initial_guess)

    # Extract the fitted parameters
    d_m_fitted, n_fitted = params

    # Generate the fitted curve using the parameters
    fitted_curve = rosin_rammler(data, d_m_fitted, n_fitted)

    return d_m_fitted, n_fitted, fitted_curve

def generalized_sigmoid(x, ymin, ymax, k, x0):
    # Clip the argument of the exponential to prevent overflow
    z = np.clip(k * (x - x0), -500, 500)  # Clip the value to avoid overflow in exp
    return ymin + (ymax - ymin) / (1 + np.exp(-z))

# Function to fit the generalized sigmoid curve to data
def fit_generalized_sigmoid(x_data, y_data):
    """
    Fits a generalized sigmoid curve to the input data, allowing for custom starting and ending values.

    Parameters:
    - x_data: Independent variable (e.g., time, size).
    - y_data: Dependent variable (e.g., cumulative percentages or CDF-like data).

    Returns:
    - ymin_fitted: Fitted minimum y value.
    - ymax_fitted: Fitted maximum y value.
    - k_fitted: Steepness of the curve.
    - x0_fitted: Midpoint of the sigmoid curve.
    - y_fitted: Fitted y values using the generalized sigmoid model.
    - Plots the original data and the fitted curve.
    """
    
    # Initial guess for ymin, ymax, k, and x0 (adjust based on your data)
    initial_guess = [min(y_data), max(y_data), 1, np.median(x_data)]

    # Fit the generalized sigmoid function to the data
    popt, pcov = curve_fit(generalized_sigmoid, x_data, y_data, p0=initial_guess, maxfev=5000)

    # Extract the fitted parameters
    ymin_fitted, ymax_fitted, k_fitted, x0_fitted = popt

    # Generate the fitted curve
    y_fitted = generalized_sigmoid(x_data, ymin_fitted, ymax_fitted, k_fitted, x0_fitted)

    # Print the fitted parameters
    print(f"Fitted ymin: {ymin_fitted}")
    print(f"Fitted ymax: {ymax_fitted}")
    print(f"Fitted k: {k_fitted}")
    print(f"Fitted x0: {x0_fitted}")

    # Plot the original data and the fitted generalized sigmoid curve
    plt.figure(figsize=(8, 6))
    plt.plot(x_data, y_data, 'bo', label='Data')  # Original data points
    plt.plot(x_data, y_fitted, 'r-', label='Fitted Generalized Sigmoid Curve')  # Fitted curve
    plt.xlabel('Independent Variable (x)')
    plt.ylabel('Dependent Variable (y)')
    plt.legend()
    plt.show()

    # Return the fitted parameters and the fitted curve
    return ymin_fitted, ymax_fitted, k_fitted, x0_fitted, y_fitted
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin

# Custom model class to allow RANSAC to fit a sigmoid curve
class SigmoidModel(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.params = None  # This will store the parameters after fitting

    def fit(self, X, y):
        # Ensure X is flattened for fitting
        X = X.flatten()

        try:
            # Use curve_fit to fit the sigmoid model
            popt, _ = curve_fit(
                generalized_sigmoid, 
                X, y, 
                p0=[min(y), max(y), 1, np.median(X)]  # Initial guess for the parameters
            )

            # Store the fitted parameters
            self.params = popt
        except Exception as e:
            print(f"An error occurred in curve_fit: {e}")
            self.params = None
        
        return self

    def predict(self, X):
        # Ensure X is flattened for prediction
        X = X.flatten()

        # Check if params are None before predicting
        if self.params is None:
            raise ValueError("Model parameters are not fitted. Fit the model before predicting.")
        
        # Use the fitted parameters to predict new values
        ymin_fitted, ymax_fitted, k_fitted, x0_fitted = self.params
        return generalized_sigmoid(X, ymin_fitted, ymax_fitted, k_fitted, x0_fitted)
