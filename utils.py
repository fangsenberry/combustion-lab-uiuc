import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm.auto import tqdm
import time
import cv2
import numpy as np

'''
Just for the Jupyter stuff
'''
from IPython.display import display, clear_output

'''
Only shows the first three by default, but can be changed by setting trunc=False and top_k to the desired number of frames.
Also the clearing of the plot is done for usage in Jupyter.
'''
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
            display(plt.gcf())  # Display the current figure
            time.sleep(0.75)  # Pause for a second
            clear_output(wait=True)  # Clear the output for the next frame
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

# import numpy as np
# from PIL import Image

def convert_tif_to_npy(tif_path, npy_path):
    image = Image.open(tif_path)
    frames = []
    try:
        for i in tqdm(range(get_total_frames(tif_path))):
            frame = np.array(image) / 255.0  # Normalize to [0, 1]
            frames.append(frame)
            image.seek(image.tell() + 1)
    except EOFError:
        pass  # End of file reached

    np.save(npy_path, np.array(frames))