import os
import cv2
import numpy as np
from skimage import io, exposure, img_as_uint
from tqdm import tqdm

def extract_frames(tiff_path, num_frames_to_skip=200):
    """
    Extract frames from a .tiff file, skipping the first n frames.

    Parameters:
        tiff_path (str): Path to the .tiff file.
        num_frames_to_skip (int): Number of frames to skip.

    Returns:
        List of extracted frames.
    """
    # Read the tiff file
    tiff = io.imread(tiff_path)
    
    # Store the extracted frames in a list
    extracted_frames = []

    # Iterate over the frames, skipping the first n frames
    for i in tqdm(range(num_frames_to_skip, tiff.shape[0]), colour='green', desc='Extracting frames', leave=False):
        frame = tiff[i]
        
        # Normalize the frame to the range [0, 1]
        frame = exposure.rescale_intensity(frame, out_range=(0, 1))
        
        # Convert to 16-bit unsigned integer format (as PNG supports up to 16-bit)
        frame = img_as_uint(frame)
        
        # Store the frame in the list (no saving to disk)
        extracted_frames.append(frame)
    
    return extracted_frames

def manual_crop(image, start_x=0, start_y=0, end_x=None, end_y=None):
    """
    Crop an image according to the specified dimensions.

    Parameters:
        image (numpy array): Image array.
        start_x (int): Starting x coordinate for cropping.
        start_y (int): Starting y coordinate for cropping.
        end_x (int): Ending x coordinate for cropping.
        end_y (int): Ending y coordinate for cropping.

    Returns:
        Cropped image.
    """
    height, width = image.shape

    # Set default values for end_x and end_y if not provided
    if end_x is None:
        end_x = width
    if end_y is None:
        end_y = height

    # Crop the image
    cropped_image = image[start_y:end_y, start_x:end_x]

    return cropped_image

def remove_vignetting(frames):
    """
    Remove vignetting from a set of images.

    Parameters:
        frames (list): List of image frames.

    Returns:
        List of vignetting-corrected frames.
    """
    num_images = len(frames)
    
    # Initialize the sum image
    sum_image = np.zeros_like(frames[0], dtype=np.float32)

    # Loop through the images and sum them
    for img in frames:
        sum_image += img.astype(np.float32)

    # Calculate the average image
    avg_image = sum_image / num_images

    # Apply Gaussian blur to estimate the vignetting pattern
    kernel_size = (51, 51)  # Adjust kernel size based on image size and vignetting extent
    vignetting_pattern = cv2.GaussianBlur(avg_image, kernel_size, 0)

    # Normalize the vignetting pattern to create the correction mask
    correction_mask = vignetting_pattern / np.max(vignetting_pattern)

    # Correct vignetting for each frame
    corrected_frames = []
    for img in frames:
        corrected_img = img / correction_mask
        corrected_img = np.clip(corrected_img, 0, 255).astype(np.uint8)
        corrected_frames.append(corrected_img)

    return corrected_frames

def calculate_average_frame(frames):
    """
    Calculate the average frame from a list of frames.

    Parameters:
        frames (list): List of image frames.

    Returns:
        The average frame.
    """
    num_images = len(frames)
    sum_image = np.zeros_like(frames[0], dtype=np.float32)

    # Loop through the images and sum them
    for img in frames:
        sum_image += img.astype(np.float32)

    # Calculate the average image
    avg_image = sum_image / num_images

    return avg_image

def remove_striations(frames, avg_image):
    """
    Remove striations from a set of images using the average image.

    Parameters:
        frames (list): List of image frames.
        avg_image (numpy array): Average image used for striation removal.

    Returns:
        List of striation-corrected frames.
    """
    corrected_images = []
    
    for img in frames:
        corrected_img = img - avg_image
        
        # Normalize the corrected image to the range [0, 255]
        corrected_img_normalized = cv2.normalize(corrected_img, None, 0, 255, cv2.NORM_MINMAX)
        corrected_img_normalized = corrected_img_normalized.astype(np.uint8)
        
        corrected_images.append(corrected_img_normalized)

    return corrected_images

def process_tiff_file(tiff_path, output_dir, start_x, start_y, end_x, end_y, num_frames_to_skip):
    """
    Process a .tiff file by extracting, cropping, removing vignetting and striations.

    Parameters:
        tiff_path (str): Path to the .tiff file.
        output_dir (str): Directory to save the final corrected frames.
        start_x (int): Starting x coordinate for cropping.
        start_y (int): Starting y coordinate for cropping.
        end_x (int): Ending x coordinate for cropping.
        end_y (int): Ending y coordinate for cropping.
        num_frames_to_skip (int): Number of frames to skip.

    Returns:
        Saves the final corrected images in the output directory.
    """
    # Step 1: Extract frames
    frames = extract_frames(tiff_path, num_frames_to_skip)

    # Step 2: Crop each frame
    cropped_frames = [manual_crop(frame, start_x, start_y, end_x, end_y) for frame in frames]

    # Step 3: Remove vignetting from the frames
    vignette_corrected_frames = remove_vignetting(cropped_frames)

    # Step 4: Calculate the average frame for striation removal
    avg_image = calculate_average_frame(vignette_corrected_frames)

    # Step 5: Remove striations
    final_frames = remove_striations(vignette_corrected_frames, avg_image)

    # Step 6: Save only the final corrected frames
    final_output_dir = os.path.join(output_dir, 'final_corrected')
    os.makedirs(final_output_dir, exist_ok=True)
    
    for i, final_frame in enumerate(final_frames):
        final_frame_path = os.path.join(final_output_dir, f"corrected_frame_{i}.png")
        cv2.imwrite(final_frame_path, final_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def main():
    tiff_file_path = r"Y:\copy0\2019ArgonneCampaign\ConvertedImages\StandardImaging\C1\50C\1ATM\DP5\O16_C_DS_45.0.tif"
    start_x, start_y, end_x, end_y = 0, 0, 600, None  # Set your cropping dimensions here
    num_frames_to_skip = 200  # Set the number of frames to skip here

    # Create a new directory for preprocessed images
    preprocessed_dir = tiff_file_path + '_preprocessed'
    os.makedirs(preprocessed_dir, exist_ok=True)

    # Process the .tiff file and save only the final corrected frames
    process_tiff_file(tiff_file_path, preprocessed_dir, start_x, start_y, end_x, end_y, num_frames_to_skip)

if __name__ == '__main__':
    main()