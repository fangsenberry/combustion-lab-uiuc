import cv2
import os
import re
from tqdm.auto import tqdm

def extract_frame_number(filename):
    match = re.search(r'frame_(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1  # Or some other default value or error handling

def images_to_video(image_dir, output_video, fps=30):
    # Get list of images in the directory
    images = [img for img in os.listdir(image_dir) if img.endswith(".png")]
    
    # Sort images by frame number
    images.sort(key=extract_frame_number)

    # Read the first image to get the size
    frame = cv2.imread(os.path.join(image_dir, images[0]))
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID'
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in tqdm(images, colour='red', desc="writing frames", leave=False):
        img_path = os.path.join(image_dir, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_directory = "data/noisy_images_preprocessed/A01_C_DP_35.0/extracted_frames"
    output_video_path = "output_video.mp4"
    images_to_video(image_directory, output_video_path, fps=30)