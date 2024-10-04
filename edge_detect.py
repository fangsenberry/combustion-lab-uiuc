import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.measure import label, regionprops

def manual_crop(image, start_x=0, end_x=None, start_y=0, end_y=None):
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Set default values for end_x and end_y if not provided
    if end_x is None:
        end_x = width
    if end_y is None:
        end_y = height
    
    # Crop the image using the provided coordinates
    cropped_image = image[start_y:end_y, start_x:end_x]
    
    return cropped_image

def process_and_crop_image_canny(image_path, output_dir, canny_threshold1=50, canny_threshold2=100, margin=0):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    # Manually crop the image (example coordinates)
    manually_cropped_image = manual_crop(image, start_y=5, end_y=270)
    
    # Display the image after cropping
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(manually_cropped_image, cv2.COLOR_BGR2RGB))
    plt.title('Manually Cropped')
    plt.axis('off')
    plt.show()
    
    # Perform Canny edge detection
    edges = cv2.Canny(manually_cropped_image, canny_threshold1, canny_threshold2)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Array to hold cropped images
    cropped_images = []
    
    # Loop over contours and crop images
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Expand the bounding box by the specified margin
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(manually_cropped_image.shape[1] - x, w + 2 * margin)
        h = min(manually_cropped_image.shape[0] - y, h + 2 * margin)
        
        cropped_image = manually_cropped_image[y:y+h, x:x+w]
        cropped_images.append(cropped_image)
        
        # Save each cropped image to the specified directory
        cropped_image_path = os.path.join(output_dir, f'cropped_{i}.png')
        cv2.imwrite(cropped_image_path, cropped_image)
    
    # Optionally display the edges and the original image with contours for verification
    plt.figure(figsize=(10, 10))
    plt.imshow(edges, cmap='gray')
    plt.title('Edges')
    plt.axis('off')
    plt.show()
    
    image_with_contours = manually_cropped_image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
    plt.title('Image with Contours')
    plt.axis('off')
    plt.show()
    
    return cropped_images

def process_and_crop_image_canny_with_morph(image_path, output_dir, canny_threshold1=50, canny_threshold2=100, margin=2, kernel_size=(2, 2), kernel_shape='rect', iterations=1):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    # Manually crop the image (example coordinates)
    manually_cropped_image = manual_crop(image, start_y=5, end_y=270)
    
    # Convert to grayscale
    gray = cv2.cvtColor(manually_cropped_image, cv2.COLOR_BGR2GRAY)
    
    # Display the image after cropping
    plt.figure(figsize=(10, 10))
    plt.imshow(gray, cmap='gray')
    plt.title('Manually Cropped')
    plt.axis('off')
    plt.show()
    
    # Perform Canny edge detection
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)
    
    # Define the structuring element shape
    if kernel_shape == 'rect':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    elif kernel_shape == 'ellipse':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    elif kernel_shape == 'cross':
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    else:
        raise ValueError(f"Unsupported kernel shape: {kernel_shape}")
    
    # Use morphological operations to close gaps in the ligaments
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # Find contours
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Array to hold cropped images
    cropped_images = []
    
    # Loop over contours and crop images
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Expand the bounding box by the specified margin
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(manually_cropped_image.shape[1] - x, w + 2 * margin)
        h = min(manually_cropped_image.shape[0] - y, h + 2 * margin)
        
        cropped_image = manually_cropped_image[y:y+h, x:x+w]
        cropped_images.append(cropped_image)
        
        # Save each cropped image to the specified directory
        cropped_image_path = os.path.join(output_dir, f'cropped_{i}.png')
        cv2.imwrite(cropped_image_path, cropped_image)
    
    # Optionally display the edges and the original image with contours for verification
    plt.figure(figsize=(10, 10))
    plt.imshow(edges, cmap='gray')
    plt.title('Edges')
    plt.axis('off')
    plt.show()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(edges_closed, cmap='gray')
    plt.title('Edges after Morphological Closing')
    plt.axis('off')
    plt.show()
    
    image_with_contours = manually_cropped_image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
    plt.figure(figsize=(20, 20))
    plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
    plt.title('Image with Contours')
    plt.axis('off')
    plt.show()
    
    return cropped_images


def upscale_image_bicubic(image, scale_factor=2):
    height, width = image.shape[:2]
    new_dimensions = (int(width * scale_factor), int(height * scale_factor))
    upscaled_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_CUBIC)
    return upscaled_image

def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def equalize_histogram(image):
    img_y_cr_cb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)
    y = cv2.equalizeHist(y)
    img_y_cr_cb = cv2.merge((y, cr, cb))
    return cv2.cvtColor(img_y_cr_cb, cv2.COLOR_YCrCb2BGR)

def apply_canny_edge_detection(image, threshold1=50, threshold2=100):
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    return cv2.Canny(blurred_image, threshold1, threshold2)

def apply_morphological_operations(edges):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed

def process_and_crop_image_canny_new(image_path, output_dir, canny_threshold1=50, canny_threshold2=100, margin=5):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    # Upscale image
    image = upscale_image_bicubic(image, scale_factor=1)
    
    # Denoise image
    # image = denoise_image(image)
    
    # Equalize histogram
    # image = equalize_histogram(image)
    
    # Manually crop the image (example coordinates)
    manually_cropped_image = manual_crop(image, start_y=5, end_y=270)
    
    # Display the image after cropping
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(manually_cropped_image, cv2.COLOR_BGR2RGB))
    plt.title('Manually Cropped')
    plt.axis('off')
    plt.show()
    
    # Perform Canny edge detection
    edges = apply_canny_edge_detection(manually_cropped_image, canny_threshold1, canny_threshold2)
    
    # Apply morphological operations
    edges = apply_morphological_operations(edges)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Array to hold cropped images
    cropped_images = []
    
    # Loop over contours and crop images
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Expand the bounding box by the specified margin
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(manually_cropped_image.shape[1] - x, w + 2 * margin)
        h = min(manually_cropped_image.shape[0] - y, h + 2 * margin)
        
        cropped_image = manually_cropped_image[y:y+h, x:x+w]
        cropped_images.append(cropped_image)
        
        # Save each cropped image to the specified directory
        cropped_image_path = os.path.join(output_dir, f'cropped_{i}.png')
        cv2.imwrite(cropped_image_path, cropped_image)
    
    # Optionally display the edges and the original image with contours for verification
    plt.figure(figsize=(10, 10))
    plt.imshow(edges, cmap='gray')
    plt.title('Edges')
    plt.axis('off')
    plt.show()
    
    image_with_contours = manually_cropped_image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
    plt.title('Image with Contours')
    plt.axis('off')
    plt.show()
    
    return cropped_images
#######################################################################################################
def pad_with_mean(roi, target_length):
    mean_value = np.mean(roi)
    padded_roi = np.pad(roi, (0, target_length - len(roi)), 'constant', constant_values=(mean_value,))
    return padded_roi

def apply_canny_edge_detection(image, threshold1=50, threshold2=100):
    return cv2.Canny(image, threshold1, threshold2)

def process_contours(contours, hierarchy, min_area, max_area, image, binary_image, valid_contours, roi_data):
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] != -1:  # Skip inner contours
            continue
        process_single_contour(contour, min_area, max_area, image, binary_image, valid_contours, roi_data, fill_small=True)
        
        # Process second-level contours
        first_level_inner_idx = hierarchy[0][i][2]
        if first_level_inner_idx != -1:
            process_inner_contours(contours, hierarchy, first_level_inner_idx, min_area, max_area, image, binary_image, valid_contours, roi_data)

def process_single_contour(contour, min_area, max_area, image, binary_image, valid_contours, roi_data, fill_small=False):
    area = cv2.contourArea(contour)
    if fill_small and area < min_area:
        cv2.drawContours(binary_image, [contour], -1, 0, thickness=cv2.FILLED)
        return

    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    roi = cv2.bitwise_and(image, image, mask=mask)
    roi_values = roi[mask == 255]

    if max_area is None or area < max_area:
        roi_data.append(roi_values.tolist())
        valid_contours.append(contour)

def process_inner_contours(contours, hierarchy, idx, min_area, max_area, image, binary_image, valid_contours, roi_data):
    while idx != -1:
        second_level_inner_idx = hierarchy[0][idx][2]  # Get the second level inner contour
        while second_level_inner_idx != -1:
            inner_contour = contours[second_level_inner_idx]
            process_single_contour(inner_contour, min_area, max_area, image, binary_image, valid_contours, roi_data)
            second_level_inner_idx = hierarchy[0][second_level_inner_idx][0]  # Move to the next inner contour
        idx = hierarchy[0][idx][0]  # Move to the next first-level inner contour

def perform_kmeans_clustering(roi_data, k):
    max_length = max(len(roi) for roi in roi_data)
    roi_data_padded = [pad_with_mean(roi, max_length) for roi in roi_data]
    roi_data_padded = np.array(roi_data_padded)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    clusters = kmeans.fit_predict(roi_data_padded)
    
    # Compute the mean for each cluster
    cluster_means = [np.mean([pixel for roi in [roi_data_padded[i] for i in range(len(clusters)) if clusters[i] == cluster_id] for pixel in roi]) for cluster_id in range(k)]
    
    if k == 2:
        min_cluster_id = np.argmin(cluster_means)
        return clusters, [min_cluster_id]
    elif k == 3:
        sorted_cluster_indices = np.argsort(cluster_means)[:2]
        return clusters, sorted_cluster_indices
    else:
        raise ValueError("This function is only designed to handle k=2 or k=3.")

def highlight_clusters(image, binary_image, contours, clusters, min_cluster_ids, colors, k):
    # Create a copy of the image for each cluster
    highlighted_images = [image.copy() for _ in range(k)]
    
    for i, contour in enumerate(contours):
        cluster_id = clusters[i]
        color = tuple(int(c * 255) for c in colors(cluster_id)[:3])
        
        # Check if the cluster is one of the min clusters
        if cluster_id in min_cluster_ids:
            cv2.drawContours(binary_image, [contour], -1, 0, thickness=cv2.FILLED)
        
        # Draw the contour on the corresponding highlighted image
        cv2.drawContours(highlighted_images[cluster_id], [contour], -1, color, 2)
    
    return highlighted_images, binary_image


def lig_segment(image_path, canny_threshold1=20, canny_threshold2=100, min_area=10, max_area=1000, k=2, plot_kmeans=None):
    os.environ['OMP_NUM_THREADS'] = '1'
    if isinstance(image_path, str):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")
    else:
        image = image_path

    manually_cropped_image = manual_crop(image, start_x=0, end_x=None, start_y=0, end_y=None)
    roi_data = []
    valid_contours = []

    edges = apply_canny_edge_detection(manually_cropped_image, canny_threshold1, canny_threshold2)
    small_rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated_edges = cv2.dilate(edges, small_rect_kernel, iterations=1)

    filled_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, small_rect_kernel, iterations=1)
    
    binary_image = cv2.bitwise_not(filled_edges)
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    process_contours(contours, hierarchy, min_area, max_area, manually_cropped_image, binary_image, valid_contours, roi_data)

    clusters, min_cluster_ids = perform_kmeans_clustering(roi_data, k)

    colors = plt.colormaps['tab10']
    original_image_bgr = cv2.cvtColor(manually_cropped_image, cv2.COLOR_GRAY2BGR)
    highlighted_images, binary_image = highlight_clusters(original_image_bgr, binary_image, valid_contours, clusters, min_cluster_ids, colors, k)

    if plot_kmeans:
        fig, axes = plt.subplots(1, k, figsize=(15, 5))
        for cluster_id in range(k):
            axes[cluster_id].imshow(cv2.cvtColor(highlighted_images[cluster_id], cv2.COLOR_BGR2RGB))
            axes[cluster_id].set_title(f'Cluster {cluster_id}')
            axes[cluster_id].axis('off')
        plt.show()
    

    # # Perform morphological opening to remove small objects
    # binary_image = cv2.erode(binary_image, small_rect_kernel, iterations=1)
    
    binary_image = cv2.bitwise_not(binary_image)

    return manually_cropped_image, binary_image, highlighted_images, filled_edges

#########################################################################################################
def identify_rois(binary_image):
    """
    Identify ROIs in the binary image using connected component analysis.
    
    Args:
        binary_image (np.ndarray): Input binary image.

    Returns:
        list: List of region properties.
    """
    labeled_image = label(binary_image)
    regions = regionprops(labeled_image)
    return regions

def extract_features(regions):
    """
    Extract features from ROIs for clustering.
    
    Args:
        regions (list): List of region properties.

    Returns:
        np.ndarray: Array of extracted features.
    """
    features = []
    for region in regions:
        # Extract properties like area, eccentricity, and solidity
        area = region.area
        eccentricity = region.eccentricity
        solidity = region.solidity
        features.append([area, eccentricity, solidity])
    return np.array(features)

def cluster_rois(features, n_clusters=2):
    """
    Cluster ROIs using k-means clustering.
    
    Args:
        features (np.ndarray): Array of extracted features.
        n_clusters (int): Number of clusters for k-means.

    Returns:
        np.ndarray: Cluster labels for each ROI.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(features)
    return kmeans.labels_

def generate_new_binary_image(binary_image, regions, labels):
    """
    Generate a new binary image based on clustering results.
    
    Args:
        binary_image (np.ndarray): Input binary image.
        regions (list): List of region properties.
        labels (np.ndarray): Cluster labels for each ROI.

    Returns:
        np.ndarray: New binary image with separated ROIs and a color-coded visualization.
    """
    color_coded_image = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Colors for two clusters: red and green
    # colors = [(255, 0, 0), (0, 255, 0)]  # Colors for two clusters: red and green
    for region, label in zip(regions, labels):
        for coord in region.coords:
            # new_binary_image[coord[0], coord[1], coord[2]] = 1  # Set binary mask
            color_coded_image[coord[0], coord[1]] = colors[label]  # Set color-coded visualization

    return color_coded_image

def process_binary_image(binary_image, n_clusters=3):
    """
    Process the binary image to identify, cluster, and re-binarize ROIs.
    
    Args:
        binary_image (np.ndarray): Input binary image.

    Returns:
        np.ndarray: Processed binary image with clustered ROIs.
        np.ndarray: Color-coded visualization of the clusters.
    """
    regions = identify_rois(binary_image)
    features = extract_features(regions)
    labels = cluster_rois(features, n_clusters=n_clusters)
    color_coded_image = generate_new_binary_image(binary_image, regions, labels)
    return color_coded_image
def identify_rois(binary_image):
    """
    Identify ROIs in the binary image using connected component analysis.
    
    Args:
        binary_image (np.ndarray): Input binary image.

    Returns:
        list: List of region properties.
    """
    labeled_image = label(binary_image)
    regions = regionprops(labeled_image)
    return regions

def extract_features(regions):
    """
    Extract features from ROIs for clustering.
    
    Args:
        regions (list): List of region properties.

    Returns:
        np.ndarray: Array of extracted features.
    """
    features = []
    for region in regions:
        # Extract properties like area, eccentricity, and solidity
        area = region.area
        eccentricity = region.eccentricity
        solidity = region.solidity
        features.append([area, eccentricity, solidity])
    return np.array(features)

def cluster_rois(features, n_clusters=2):
    """
    Cluster ROIs using k-means clustering.
    
    Args:
        features (np.ndarray): Array of extracted features.
        n_clusters (int): Number of clusters for k-means.

    Returns:
        np.ndarray: Cluster labels for each ROI.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(features)
    return kmeans.labels_

def generate_new_binary_image(binary_image, regions, labels):
    """
    Generate a new binary image based on clustering results.
    
    Args:
        binary_image (np.ndarray): Input binary image.
        regions (list): List of region properties.
        labels (np.ndarray): Cluster labels for each ROI.

    Returns:
        np.ndarray: New binary image with separated ROIs and a color-coded visualization.
    """
    color_coded_image = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Colors for two clusters: red and green
    # colors = [(255, 0, 0), (0, 255, 0)]  # Colors for two clusters: red and green
    for region, label in zip(regions, labels):
        for coord in region.coords:
            # new_binary_image[coord[0], coord[1], coord[2]] = 1  # Set binary mask
            color_coded_image[coord[0], coord[1]] = colors[label]  # Set color-coded visualization

    return color_coded_image

def process_binary_image(binary_image, n_clusters=3):
    """
    Process the binary image to identify, cluster, and re-binarize ROIs.
    
    Args:
        binary_image (np.ndarray): Input binary image.

    Returns:
        np.ndarray: Processed binary image with clustered ROIs.
        np.ndarray: Color-coded visualization of the clusters.
    """
    regions = identify_rois(binary_image)
    features = extract_features(regions)
    labels = cluster_rois(features, n_clusters=n_clusters)
    color_coded_image = generate_new_binary_image(binary_image, regions, labels)
    return color_coded_image
