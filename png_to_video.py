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

def numerical_sort_key(filename):
    return int(''.join(filter(str.isdigit, filename)))

def images_to_video(image_dir, output_video, fps=30):
    # Get list of images in the directory and sort them using the numerical sort key
    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')], key=numerical_sort_key)

    # Ensure there are images to process
    if not images:
        print("No images found in the directory.")
        return

    # Print the list of images to be processed
    # print("Files to be processed:")
    # for img in images:
    #     print(img)

    # Read the first image to get the size
    frame = cv2.imread(os.path.join(image_dir, images[0]))
    if frame is None:
        print(f"Error: Could not read {images[0]}.")
        return

    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID'
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in tqdm(images, colour='red', desc="Writing frames", leave=False):
        img_path = os.path.join(image_dir, image)
        print(f"Processing file: {img_path}")  # Print the file being processed
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Could not read {img_path}. Skipping.")
            continue
        
        # Check if the frame size matches the expected size
        if frame.shape[0] != height or frame.shape[1] != width:
            print(f"Warning: {img_path} has different dimensions. Resizing.")
            frame = cv2.resize(frame, (width, height))

        video.write(frame)

    video.release()
    cv2.destroyAllWindows()

    print(f"Video {output_video} has been successfully created.")

if __name__ == "__main__":
    image_directory = r"D:\test_cases\UPF_A01_C_DP_35_trial_33\flow_analysis_plots\shear_stress\shear_y"
    output_video_path = os.path.join(image_directory, "output_video.mp4")
    images_to_video(image_directory, output_video_path, fps=0.5)