import os
from skimage import io, exposure, img_as_uint
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
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
    # print(f"Image size: {width} x {height}")
    # Set default values for end_x and end_y if not provided
    if end_x is None:
        end_x = width
    if end_y is None:
        end_y = height

    # Crop the image
    cropped_image = image.crop((start_x, start_y, end_x, end_y))

    # Save the cropped image back
    cropped_image.save(image_path, format='PNG')
    
    # Display the original image
    # plt.figure(figsize=(20, 20))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image, cmap='gray')
    # plt.title("Before Cropping")
    # plt.axis('off')

    # # Crop the image
    # cropped_image = image.crop((start_x, start_y, end_x, end_y))

    # # Display the cropped image
    # plt.subplot(1, 2, 2)
    # plt.imshow(cropped_image, cmap='gray')
    # plt.title("After Cropping")
    # plt.axis('off')
    # plt.show()

def main():
    dir_path = 'noisy_images'  # Change this to your directory path
    start_x, start_y, end_x, end_y = 0, 0, 600, None  # Set your cropping dimensions here
    num_frames_to_skip = 200  # Set the number of frames to skip here

    # Create a new directory for preprocessed images
    preprocessed_dir = dir_path + '_preprocessed'
    os.makedirs(preprocessed_dir, exist_ok=True)

    # Process each .tiff file in the directory
    for i, filename in enumerate(os.listdir(dir_path)):
        print(f"File Progress: {i + 1}/{len(os.listdir(dir_path))}")
        if filename.endswith('.tif'):
            tiff_path = os.path.join(dir_path, filename)
            # Create a subdirectory for the extracted frames
            extracted_frames_dir = os.path.join(preprocessed_dir, os.path.splitext(filename)[0])
            os.makedirs(extracted_frames_dir, exist_ok=True)
            
            # Extract frames to .png format, skipping the first num_frames_to_skip frames
            extract_frames_to_png(tiff_path, extracted_frames_dir, num_frames_to_skip)

            # Crop each extracted .png file
            for frame_filename in tqdm(os.listdir(extracted_frames_dir), leave=False, colour='blue', desc='Cropping frames'):
                if frame_filename.endswith('.png'):
                    frame_path = os.path.join(extracted_frames_dir, frame_filename)
                    manual_crop(frame_path, start_x, start_y, end_x, end_y)

if __name__ == '__main__':
    main()
