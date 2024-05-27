import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

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

import os
import cv2
import matplotlib.pyplot as plt

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

def manual_crop(image, start_y, end_y):
    return image[start_y:end_y, :]

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
