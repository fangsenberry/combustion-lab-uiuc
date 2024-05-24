import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm.auto import tqdm
import time

import matplotlib.pyplot as plt
from PIL import Image

def show_image(image_path, trunc: bool = True, top_k: int = 3):
    # Load the .tif file
    image = Image.open(image_path)

    # Check if the image has multiple frames
    try:
        i = 0
        while True:
            if trunc and i >= top_k: break
            image.seek(i)
            # Display the image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.axis('off')  # Hide the axis
            plt.title(f'Fuel Droplets in Combustion Chamber - Frame {i+1}')
            plt.show()
            i += 1
    except EOFError:
        pass  # End of file reached

def get_total_frames(image_path):
    image = Image.open(image_path)
    total_frames = 0
    try:
        while True:
            image.seek(total_frames)
            total_frames += 1
    except EOFError:
        pass  # End of file reached
    return total_frames

def process_images(input_path, output_dir, output_filename, start_x=None, end_x=None, start_y=None, end_y=None, trunc=False, top_k=3, save_cropped=False, show_image: bool = True):
    start_time = time.time()
    # Load the .tif file
    image = Image.open(input_path)
    print(f"Image Dim: {image.width} x {image.height}")
    
    total_frames = get_total_frames(input_path)
    print(f"Total Frames: {total_frames}, time taken: {time.time() - start_time:.2f}s")

    # Set default cropping coordinates if not provided
    if start_x is None: start_x = 0
    if start_y is None: start_y = 0
    if end_x is None: end_x = image.width
    if end_y is None: end_y = image.height

    cropped_frames = []

    # Check if the image has multiple frames
    for i in tqdm(range(total_frames)):
        if trunc and i >= top_k: break
        image.seek(i)
        # Crop the image
        cropped_image = image.crop((start_x, start_y, end_x, end_y))
        cropped_frames.append(cropped_image)

        if show_image:
            # Display the cropped image
            plt.figure(figsize=(15, 15))
            plt.imshow(cropped_image)
            plt.axis('off')  # Hide the axis
            plt.title(f'Fuel Droplets in Combustion Chamber - Frame {i+1}')
            plt.show()
            plt.close()  # Close the plot to clear the output

    print(f"Number of frames processed: {len(cropped_frames)}, time taken: {time.time() - start_time:.2f}s")

    # Save the cropped images if required
    if save_cropped:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, output_filename)
        cropped_frames[0].save(output_path, save_all=True, append_images=cropped_frames[1:])
        print(f"Cropped images saved to {output_path}")

    print(f"Total time taken: {time.time() - start_time:.2f}s")


