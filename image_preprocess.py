import os
from skimage import io, exposure, img_as_uint
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import cv2
from glob import glob
import matplotlib.pyplot as plt

def extract_frames_to_png(tiff_path, output_dir, num_frames_to_skip=200):
    """
    Extract frames from a .tiff file to .png format, skipping the first n frames.

    Parameters:
        tiff_path (str): Path to the .tiff file.
        output_dir (str): Directory to save the extracted .png frames.
        num_frames_to_skip (int): Number of frames to skip.
    """
    # Read the tiff file
    tiff = io.imread(tiff_path)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over the frames, skipping the first n frames
    for i in tqdm(range(num_frames_to_skip, tiff.shape[0]), colour='green', desc='Extracting frames', leave=False):
        frame = tiff[i]
        
        # Normalize the frame to the range [0, 1]
        frame = exposure.rescale_intensity(frame, out_range=(0, 1))
        
        # Convert to 16-bit unsigned integer format (as PNG supports up to 16-bit)
        frame = img_as_uint(frame)

        # Save the frame as a .png file
        frame_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(tiff_path))[0]}_frame_{i - num_frames_to_skip}.png")
        Image.fromarray(frame).save(frame_path, format='PNG')

def manual_crop(image_path, start_x=0, start_y=0, end_x=None, end_y=None):
    """
    Crop an image according to the specified dimensions.
    
    Parameters:
        image_path (str): Path to the image file.
        start_x (int): Starting x coordinate for cropping.
        start_y (int): Starting y coordinate for cropping.
        end_x (int): Ending x coordinate for cropping.
        end_y (int): Ending y coordinate for cropping.
    """
    # Read the image
    image = Image.open(image_path)
    width, height = image.size

    # Set default values for end_x and end_y if not provided
    if end_x is None:
        end_x = width
    if end_y is None:
        end_y = height

    # Crop the image
    cropped_image = image.crop((start_x, start_y, end_x, end_y))

    # Save the cropped image back
    cropped_image.save(image_path, format='PNG')

def remove_vignetting(image_files, output_dir):
    num_images = len(image_files)
    print(f"found {num_images} images")
    
    # Initialize the sum image
    sum_image = None

    # Loop through the images and sum them
    for file in tqdm(image_files):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if sum_image is None:
            sum_image = np.zeros_like(img)
        sum_image += img

    # Calculate the average image
    avg_image = sum_image / num_images

    # Apply Gaussian blur to estimate the vignetting pattern
    kernel_size = (51, 51)  # Adjust kernel size based on image size and vignetting extent
    vignetting_pattern = cv2.GaussianBlur(avg_image, kernel_size, 0)

    # Normalize the vignetting pattern to create the correction mask
    correction_mask = vignetting_pattern / np.max(vignetting_pattern)

    # Process each image to correct vignetting
    for file in tqdm(image_files):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        corrected_img = img / correction_mask
        corrected_img = np.clip(corrected_img, 0, 255)  # Clip values to maintain valid intensity range
        corrected_img = corrected_img.astype(np.uint8)
        # Save the image with the same compression settings as the original
        corrected_path = os.path.join(output_dir, f'v_corrected_{os.path.basename(file)}')
        cv2.imwrite(corrected_path, corrected_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    return glob(f'{output_dir}/*.png')

def calculate_average_frame(image_files):
    # Initialize the sum image
    sum_image = None
    num_images = len(image_files)
    
    # Loop through the images and sum them
    for file in tqdm(image_files, desc="Calculating average frame"):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if sum_image is None:
            sum_image = np.zeros_like(img)
        sum_image += img

    # Calculate the average image
    avg_image = sum_image / num_images
    
    # Normalize the average image to the range [0, 255]
    avg_image_normalized = cv2.normalize(avg_image, None, 0, 255, cv2.NORM_MINMAX)
    avg_image_normalized = avg_image_normalized.astype(np.uint8)
    
    # Display the average image
    # plt.figure(figsize=(10, 10))
    # plt.imshow(avg_image_normalized, cmap='gray')
    # plt.title('Average Frame')
    # plt.show()
    
    return avg_image

def normalize_intensity(image, target_mean=128):
    current_mean = np.mean(image)
    adjustment_factor = target_mean / current_mean
    normalized_image = image * adjustment_factor
    normalized_image = np.clip(normalized_image, 0, 255)
    return normalized_image.astype(np.uint8)

def remove_striations(image_files, avg_image, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for file in tqdm(image_files, desc="Removing striations"):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        corrected_img = img - avg_image
        
        # Normalize the corrected image to the range [0, 255]
        corrected_img_normalized = cv2.normalize(corrected_img, None, 0, 255, cv2.NORM_MINMAX)
        corrected_img_normalized = corrected_img_normalized.astype(np.uint8)
        
        # Post-process to normalize the average pixel intensity
        normalized_img = normalize_intensity(corrected_img_normalized)
        
        output_file = os.path.join(output_dir, f's_corrected_{os.path.basename(file)}')
        cv2.imwrite(output_file, normalized_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def process_tiff_file(tiff_path, output_dir, start_x, start_y, end_x, end_y, num_frames_to_skip):
    extracted_frames_dir = os.path.join(output_dir, 'extracted_frames')
    os.makedirs(extracted_frames_dir, exist_ok=True)

    # Extract frames to .png format, skipping the first num_frames_to_skip frames
    extract_frames_to_png(tiff_path, extracted_frames_dir, num_frames_to_skip)

    # Crop each extracted .png file
    for frame_filename in tqdm(os.listdir(extracted_frames_dir), leave=False, colour='blue', desc='Cropping frames'):
        if frame_filename.endswith('.png'):
            frame_path = os.path.join(extracted_frames_dir, frame_filename)
            manual_crop(frame_path, start_x, start_y, end_x, end_y)

    # Remove vignetting
    vignette_corrected_dir = os.path.join(output_dir, 'vignette_corrected')
    os.makedirs(vignette_corrected_dir, exist_ok=True)
    cropped_files = glob(f'{extracted_frames_dir}/*.png')
    vignette_corrected_files = remove_vignetting(cropped_files, vignette_corrected_dir)

    # Calculate the average frame for striation removal
    avg_image = calculate_average_frame(vignette_corrected_files)

    # Remove striations
    final_output_dir = os.path.join(output_dir, 'final_corrected')
    os.makedirs(final_output_dir, exist_ok=True)
    remove_striations(vignette_corrected_files, avg_image, final_output_dir)

def main():
    tiff_file_path = r"D:\case_F24_centerline\f24_centerline_100C_C001H001S0007_S0001.tif"
    start_x, start_y, end_x, end_y = 0, 0, 600, None  # Set your cropping dimensions here
    num_frames_to_skip = 200  # Set the number of frames to skip here

    # Create a new directory for preprocessed images
    preprocessed_dir = tiff_file_path + '_preprocessed'
    os.makedirs(preprocessed_dir, exist_ok=True)

    # Process the .tiff file (no os.listdir() needed as we are working with a single file)
    process_tiff_file(tiff_file_path, preprocessed_dir, start_x, start_y, end_x, end_y, num_frames_to_skip)

if __name__ == '__main__':
    main()