import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

def pad_with_mean(roi, target_length):
    mean_value = np.mean(roi)
    padded_roi = np.pad(roi, (0, target_length - len(roi)), 'constant', constant_values=(mean_value,))
    return padded_roi

# Ensure all ROIs are the same length by padding them with their mean value

def lig_segment(image_path, canny_threshold1=20, canny_threshold2=100, min_area=10, max_area=1000, k=2, plot_kmeans=None):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    # Manually crop the image (example coordinates)
    manually_cropped_image = manual_crop(image, start_x=300, end_x=600, start_y=5, end_y=270)
    roi_data = []
    valid_contours = []
    edges = apply_canny_edge_detection(manually_cropped_image, canny_threshold1, canny_threshold2)

    small_rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    small_ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    small_cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    # Create a mask by dilating the edges with a smaller kernel and fewer iterations
    dilated_edges = cv2.dilate(edges, small_rect_kernel, iterations=1)

    # Apply morphological closing with a smaller kernel to minimally fill the edges
    filled_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, small_rect_kernel, iterations=1)

    binary_image = cv2.bitwise_not(filled_edges)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, contour in enumerate(contours):

        area = cv2.contourArea(contour)
        if area < min_area:
            cv2.drawContours(binary_image, [contour], -1, 0, thickness=cv2.FILLED)
            continue

        # Create a mask for the current contour
        mask = np.zeros(manually_cropped_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Extract the ROI using the mask
        roi = cv2.bitwise_and(manually_cropped_image, manually_cropped_image, mask=mask)
        
        # Get the pixel values of the ROI
        roi_values = roi[mask == 255]
        image_with_contour = cv2.cvtColor(manually_cropped_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image_with_contour, [contour], -1, (0, 255, 0), 2)

        if max_area is None or area < max_area:
            roi_data.append(roi_values.tolist())
            valid_contours.append(contour)
 
     # Flatten the ROI data
    max_length = max(len(roi) for roi in roi_data)
    roi_data_padded = [pad_with_mean(roi, max_length) for roi in roi_data]

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=0)
    clusters = kmeans.fit_predict(roi_data_padded)
    cluster_means = []
    for cluster_id in range(k):
        cluster_pixels = [roi_data_padded[i] for i in range(len(clusters)) if clusters[i] == cluster_id]
        cluster_mean = np.mean([pixel for roi in cluster_pixels for pixel in roi])
        cluster_means.append(cluster_mean)

    # Identify the cluster with the lowest mean pixel value
    min_cluster_id = np.argmin(cluster_means)

    # Define a color palette for clusters
    colors = plt.cm.get_cmap('tab10', k)  # Use a colormap with k distinct colors

    original_image_bgr = cv2.cvtColor(manually_cropped_image, cv2.COLOR_GRAY2BGR)
    highlighted_images = [original_image_bgr.copy() for _ in range(k)]

    # Highlight the ROIs based on their cluster assignment
    for i, contour in enumerate(valid_contours):
        cluster_id = clusters[i]
        color = tuple(int(c * 255) for c in colors(cluster_id)[:3])
        if cluster_id == min_cluster_id:
            cv2.drawContours(binary_image, [contour], -1, 0, thickness=cv2.FILLED)  # Fill with black
        cv2.drawContours(highlighted_images[cluster_id], [contour], -1, color, 2)


    # Display the original image with highlighted ROIs for each cluster
    if plot_kmeans:
        plt.figure(figsize=(18, 6))
        for cluster_id in range(k):
            plt.subplot(1, k, cluster_id + 1)
            plt.title(f'ROIs of Cluster {cluster_id} Highlighted')
            plt.imshow(highlighted_images[cluster_id])
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    binary_image = cv2.bitwise_not(binary_image)
    contours, _= cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(original_image_bgr, contours, -1, (0, 255, 0), 2)
    
    return manually_cropped_image, binary_image, original_image_bgr
