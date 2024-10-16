from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
#import necessary libraries
import cv2
from scipy.spatial.distance import cdist

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


############ random unorganized code snippets ################
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