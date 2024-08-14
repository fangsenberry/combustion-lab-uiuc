import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter, median_filter
import matplotlib.gridspec as gridspec
from skimage.transform import warp
from PIL import Image
from matplotlib.colors import ListedColormap
import flow_viz

current_dir = Path(__file__).resolve()

# Go back two directories
two_dirs_up = current_dir.parents[2]

# Add the path to sys.path
sys.path.append(str(two_dirs_up))

import edge_detect as ed

########################################################################################
def pastel_colormap():
    """Create a pastel colormap."""
    colors = [
        (204, 229, 255),  # Light blue
        (255, 204, 204),  # Light red
        (204, 255, 204),  # Light green
        (255, 255, 204),  # Light yellow
        (255, 204, 255),  # Light magenta
        (204, 255, 255),  # Light cyan
        (255, 229, 204),  # Light orange
        (229, 204, 255),  # Light violet
        (229, 255, 229)   # Light pastel green
    ]
    return ListedColormap(colors, name='pastel')

def manual_crop(image, start_x=0, end_x=None, start_y=0, end_y=None):
    if image.ndim == 3:
        _, height, width = image.shape
    else:
        height, width = image.shape
    
    end_x = width if end_x is None else end_x
    end_y = height if end_y is None else end_y

    start_x = max(0, start_x)
    end_x = min(width, end_x)
    start_y = max(0, start_y)
    end_y = min(height, end_y)

    if image.ndim == 3:
        return image[:, start_y:end_y, start_x:end_x]
    else:
        return image[start_y:end_y, start_x:end_x]

def visualize_flow(flow, type='basic'):
    """Visualize optical flow."""
    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[0], flow[1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    color_map = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if type == 'pastel':
        pastel_map=pastel_colormap()
        color_map=pastel_map(hsv[...,2]).astype(np.uint8)
    elif type == 'custom':
        color_map = flow_viz.flow_to_image(flow)
    return color_map


def warp_image_skimage(image, flow):
    h, w = image.shape
    if flow.shape != (2, h, w):
        raise ValueError(f"Expected flow shape (2, {h}, {w}), but got {flow.shape}")

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x, map_y = x - flow[0], y - flow[1]
    coords = np.stack([map_y, map_x], axis=0)

    return warp(image, coords, mode='wrap', order=3)

def normalize_image(image, target_mean, target_std):
    image = image.astype(np.float32)
    image_mean, image_std = np.mean(image), np.std(image)
    normalized_image = (image - image_mean) / image_std * target_std + target_mean
    return np.clip(normalized_image, 0, 255).astype(np.uint8)

def compute_gradient(image1, image2):
    # Normalize images to have similar mean and standard deviation
    # target_mean = (np.mean(image1) + np.mean(image2)) / 2
    # target_std = (np.std(image1) + np.std(image2)) / 2
    # image1 = normalize_image(image1, target_mean, target_std)
    # image2 = normalize_image(image2, target_mean, target_std)
    
    # Compute the signed gradient
    gradient = image2.astype(np.float32) - image1.astype(np.float32)
    return gradient

def gradient_to_heatmap(gradient, global_min, global_max):
    # Normalize the gradient to the range [0, 1] using global min and max
    normalized_gradient = (gradient - global_min) / (global_max - global_min)
    
    # Apply the 'viridis' colormap
    heatmap = cm.coolwarm(normalized_gradient)  # Use 'viridis' colormap and discard the alpha channel
    heatmap = (heatmap * 510).astype(np.uint8)
    return heatmap

def save_heatmap_with_colorbar(heatmap, global_min, global_max, filename):
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap='coolwarm', vmin=global_min, vmax=global_max)
    plt.colorbar(label='Gradient Intensity')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def flow_params(flow, step=10, start_x=0, end_x=None, start_y=0, end_y=None, reverse_flow=False, flow_vis_type='basic'):
    """
    extract flow parameters for visualization
    """
    
    H, W = flow.shape[1:]
    y, x = np.mgrid[0:H:step, 0:W:step]
    u = flow[0, ::step, ::step]
    v = flow[1, ::step, ::step]

    if reverse_flow:
        u, v = -u, -v

    flow_vis = visualize_flow(flow, type=flow_vis_type)

    return flow_vis, x, y, u, v

def numerical_sort_key(file_name):
    return int(file_name.split('_')[-1].split('.')[0])

def flow_checks(flow):
    if flow.ndim == 4:
        # Assuming the shape is (N, 2, H, W) and we take the first element (N should be 1 for batch size 1)
        flow = flow[0]
    
    if flow.shape[0] != 2:
        #move final channel to first channel
        flow = np.moveaxis(flow, -1, 0)
    
    return flow

def load_and_visualize_flows(directory, im_dir, base, step=10, start_y=0, end_y=None, start_x=0, end_x=None, reverse_flow=False, binary_image=False, warp=False, custom_range=25, flow_vis_type='basic'):
    flow_vis_list = []
    img_list = []
    warped_img_list = []
    gradient_list = []
    binary_image_list = []

    target_height = None
    target_width = None

    image_files = sorted([f for f in os.listdir(im_dir) if f.endswith('.png')], key=numerical_sort_key)
    if custom_range == 'end':
        custom_range = len(image_files)-1
    
    for idx in range(custom_range):
        # Load flow files
        filepath = os.path.join(directory, f"{base}{idx}.npy")
        flow = flow_checks(np.load(filepath))
        
        # Determine target size from the first flow
        if idx == 0:
            target_height, target_width = flow.shape[1], flow.shape[2]
        
        flow_vis, x, y, u, v = flow_params(flow, step, start_x=start_x, end_x=end_x, start_y=start_y, end_y=end_y, reverse_flow=reverse_flow, flow_vis_type=flow_vis_type)
        flow_vis_list.append((flow_vis, x, y, u, v))

        # Load and process images
        img_path = os.path.join(im_dir, image_files[idx])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        #set image to have a mean of 127
        img = (img - np.mean(img) + 127).clip(0, 255)
        
        # Convert to 8-bit and perform segmentation
        if binary_image:
            _, bin_im, _, _ = ed.lig_segment(img_path, canny_threshold1=20, canny_threshold2=100, min_area=10, max_area=1000, k=3, plot_kmeans=None)
            cropped_bin_im = cv2.resize(bin_im, (target_width, target_height))
            binary_image_list.append(cropped_bin_im)
        
        # Resize/crop the image and binary image to the target dimensions
        cropped_img = cv2.resize(img, (target_width, target_height))

        
        img_list.append(cropped_img)

        if idx > 0 and warp:
            prev_flow_path = os.path.join(directory, f"{base}{idx-1}.npy")
            prev_flow = flow_checks(np.load(prev_flow_path))
            warped_img = warp_image_skimage(img_list[idx-1], prev_flow)
            warped_img_list.append(warped_img)
            
            # Compute the gradient between the warped image and the current image
            gradient = compute_gradient(cropped_img, warped_img)
            gradient_list.append(gradient)
        else:
            warped_img_list.append(cropped_img)
            gradient_list.append(np.zeros_like(cropped_img))  # No gradient for the first frame

    return flow_vis_list, img_list, warped_img_list, gradient_list, binary_image_list

def plot_flow_vectors(flow_vis_list, img_list, binary_image_list, start_x=0, end_x=None, start_y=0, end_y=None, step=10):
    # for idx in range(len(flow_vis_list)):
    for idx in range(30):
        flow_vis, x, y, u, v = flow_vis_list[idx]
        original_img = img_list[idx]
        binary_img = binary_image_list[idx]
        
        # Crop the images
        cropped_img = manual_crop(original_img, start_x=start_x, end_x=end_x, start_y=start_y, end_y=end_y)
        cropped_binary_img = manual_crop(binary_img, start_x=start_x, end_x=end_x, start_y=start_y, end_y=end_y)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(cropped_img, cmap='gray')
        
        # Determine cropping bounds
        if end_x is None:
            end_x = original_img.shape[1]
        if end_y is None:
            end_y = original_img.shape[0]
        
        # Plot flow vectors only where binary_img is non-zero (indicating fuel regions)
        for i in range(0, cropped_img.shape[0], step):
            for j in range(0, cropped_img.shape[1], step):
                u_idx = (i + start_y) // step
                v_idx = (j + start_x) // step
                if u_idx < u.shape[0] and v_idx < u.shape[1] and cropped_binary_img[i, j] > 0:
                    plt.arrow(j, i, u[u_idx, v_idx], v[u_idx, v_idx], color='red', head_width=1, head_length=1)
        
        # plt.title(f'Flow Vectors for Frame {idx+1}')
        plt.axis('off')
        plt.imshow(cropped_img, cmap='gray')
        plt.tight_layout()
        plt.show()

def plot_flow_vectors_as_video(flow_vis_list, img_list, binary_image_list, start_x=0, end_x=None, start_y=0, end_y=None, step=10, video_filename='flow_vectors.mp4'):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Determine the cropping bounds for consistent plot size
    if end_x is None:
        end_x = img_list[0].shape[1]
    if end_y is None:
        end_y = img_list[0].shape[0]

    def update(idx):
        ax.clear()
        flow_vis, x, y, u, v = flow_vis_list[idx]
        original_img = img_list[idx]
        binary_img = binary_image_list[idx]

        # Crop the images
        cropped_img = manual_crop(original_img, start_x=start_x, end_x=end_x, start_y=start_y, end_y=end_y)
        cropped_binary_img = manual_crop(binary_img, start_x=start_x, end_x=end_x, start_y=start_y, end_y=end_y)
        
        ax.imshow(cropped_img, cmap='gray')

        # Plot flow vectors only where binary_img is non-zero (indicating fuel regions)
        for i in range(0, cropped_img.shape[0], step):
            for j in range(0, cropped_img.shape[1], step):
                u_idx = (i + start_y) // step
                v_idx = (j + start_x) // step
                if u_idx < u.shape[0] and v_idx < u.shape[1] and cropped_binary_img[i, j] > 0:
                    ax.arrow(j, i, u[u_idx, v_idx], v[u_idx, v_idx], color='red', head_width=1, head_length=1)
        
        ax.axis('off')
        ax.set_xlim(0, cropped_img.shape[1])
        ax.set_ylim(cropped_img.shape[0], 0)

    anim = FuncAnimation(fig, update, frames=len(flow_vis_list), repeat=False)
    Writer = writers['ffmpeg']
    writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(video_filename, writer=writer)

def plot_flow_and_colorwheel(flow_vis, binary_image=None):
    # Apply the binary mask to the flow visualization
    if binary_image is not None:
        masked_flow_vis = cv2.bitwise_and(flow_vis, flow_vis, mask=binary_image)
    else:
        masked_flow_vis = flow_vis
    
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    # Plot the optical flow RGB image over the binary image
    ax0 = plt.subplot(gs[0])
    ax0.imshow(masked_flow_vis)
    ax0.set_title('Optical Flow', fontsize=18)
    ax0.axis('off')

    # Create and plot the color wheel
    ax1 = plt.subplot(gs[1])
    hsv_colorwheel = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            dy = i - 128
            dx = j - 128
            angle = np.arctan2(dy, dx)
            magnitude = np.sqrt(dx**2 + dy**2)
            hue = (angle * 180 / np.pi / 2 + 180) % 180
            hsv_colorwheel[i, j, 0] = hue
            hsv_colorwheel[i, j, 1] = 255
            hsv_colorwheel[i, j, 2] = np.clip(magnitude * (255 / 128), 0, 255)

    colorwheel = cv2.cvtColor(hsv_colorwheel, cv2.COLOR_HSV2BGR)
    ax1.imshow(colorwheel)
    ax1.set_title('Colorwheel: Hue (Direction), Intensity (Magnitude)', fontsize=18)
    ax1.axis('off')

    # plt.show()
    plt.savefig('flow_and_colorwheel.png', bbox_inches='tight', pad_inches=0)

def plot_and_save_flow(flow_vis_list, img_list, output_dir='output_directory', plot_type='flow_vis', fps=2, mask=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, ax = plt.subplots(1, 1, figsize=(18, 6))

    def update_plot(i):
        ax.cla()
        
        flow_vis, x, y, u, v = flow_vis_list[i]
        img = img_list[i]

        # Ensure x, y, u, and v have the same shape as img
        if x.shape != img.shape:
            x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))


        # Apply the mask to flow vectors
        u_resized = cv2.resize(u, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        v_resized = cv2.resize(v, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        if mask==True:
            mask = (img > 0).astype(np.uint8) * 255  # Convert mask to 8-bit image
            u = u_resized[mask > 0]
            v = v_resized[mask > 0]
            x = x[mask > 0]
            y = y[mask > 0]
            flow_vis = cv2.bitwise_and(flow_vis.astype(np.uint8), flow_vis.astype(np.uint8), mask=mask)

        if plot_type in ['flow_vis', 'both']:
            # Plot color flow visualization
            ax.imshow(flow_vis)
            ax.axis('off')
            plt.savefig(os.path.join(output_dir, f'flow_vis_{i:04d}.png'), bbox_inches='tight', pad_inches=0)
        
        if plot_type in ['quiver', 'both']:
            if plot_type == 'both':
                ax.cla()  # Clear the current plot for the next quiver plot
            # Plot quiver plot
            ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=5, color='r')
            ax.invert_yaxis()
            ax.axis('off')
            plt.savefig(os.path.join(output_dir, f'quiver_plot_{i:04d}.png'), bbox_inches='tight', pad_inches=0)
        
    for i in range(len(flow_vis_list)):
        update_plot(i)
        plt.cla()  # Clear the current plot for the next frame

    plt.close(fig)

def save_warped_images(warped_img_list, save_dir):
    """
    Save warped images to the specified directory.

    Parameters:
    warped_img_list (list): List of warped images.
    save_dir (str): Directory where images will be saved.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i, warped_img in enumerate(warped_img_list):
        if warped_img.dtype != np.uint8:
            warped_img = convert_to_uint8(warped_img)
        img_path = os.path.join(save_dir, f'warped_image_{i:04d}.png')
        Image.fromarray(warped_img).save(img_path)

def generate_and_save_global_heatmaps(gradient_list, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Combine all gradients
    all_gradients = np.concatenate([gradient.flatten() for gradient in gradient_list])
    absolute_gradients = np.abs(all_gradients)
    
    # Compute global min and max
    global_min = np.min(all_gradients)
    global_max = np.max(all_gradients)
    print(f"Global min: {global_min}, Global max: {global_max}")
    
    # Compute average of absolute gradients
    average_absolute_gradient = np.mean(absolute_gradients)
    print(f"Average of the absolute gradients: {average_absolute_gradient}")
    
    # Generate and save heatmap images
    for i, gradient in enumerate(gradient_list):
        heatmap = gradient_to_heatmap(gradient, global_min, global_max)
        filename = os.path.join(save_dir, f"gradient_{i:04d}.png")
        save_heatmap_with_colorbar(heatmap, global_min, global_max, filename)

def save_binary_original_overlap(original_image, binary_image, path=None):
    overlay_image=np.zeros_like(original_image)
    mask= binary_image>0
    overlay_image[mask]=original_image[mask]
    plt.imshow(overlay_image, cmap='gray')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def convert_to_uint8(image):
    """
    Convert an image to uint8 format.
    """
    image = (image - image.min()) / (image.max() - image.min()) * 255
    return image.astype(np.uint8)

def save_synethetic_warp_pairs(img_list, warped_img_list, save_dir):
    """
    Save images in the specified order.

    Parameters:
    img_list (list): List of original images.
    warped_img_list (list): List of warped images.
    save_dir (str): Directory where images will be saved.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i in range(len(img_list)):
        # Save the first image from img_list
        img1 = convert_to_uint8(img_list[i])
        img1_path = os.path.join(save_dir, f'image_{2*i+1:04d}.png')
        Image.fromarray(img1).save(img1_path)

        # Save the second image from warped_img_list if it exists
        if i+1 < len(warped_img_list):
            img2 = convert_to_uint8(warped_img_list[i+1])
            img2_path = os.path.join(save_dir, f'image_{2*i+2:04d}.png')
            Image.fromarray(img2).save(img2_path)

##################################################

class FlowAnalysis:
    def __init__(self, flow_vis_list):
        self.u_vectors, self.v_vectors = self.extract_flow_vectors(flow_vis_list)
        self.mean_u = None
        self.mean_v = None
        self.u_fluctuations = None
        self.v_fluctuations = None

    def extract_flow_vectors(self, flow_vis_list):
        u_vectors = []
        v_vectors = []

        for flow_vis in flow_vis_list:
            _, _, _, u, v = flow_vis  # Assuming flow_vis_list contains tuples with flow vectors
            u_vectors.append(u)
            v_vectors.append(v)
        
        # Crop the first 10 columns of the flow vectors
        u_vectors = np.array(u_vectors)[:, :, 10:]
        v_vectors = np.array(v_vectors)[:, :, 10:]
        
        return u_vectors, v_vectors

    def compute_rms(self, vectors):
        rms_values = np.sqrt(np.mean(vectors**2, axis=0))
        return rms_values

    def save_flow_vectors(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        for idx, (u, v) in enumerate(zip(self.u_vectors, self.v_vectors)):
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            
            # Plot u vectors
            im1 = ax[0].imshow(u, cmap='jet', aspect='auto')
            ax[0].set_title(f'u vectors - Frame {idx}')
            cbar1 = fig.colorbar(im1, ax=ax[0], orientation='vertical')
            cbar1.set_label('Intensity')

            # Plot v vectors
            im2 = ax[1].imshow(v, cmap='jet', aspect='auto')
            ax[1].set_title(f'v vectors - Frame {idx}')
            cbar2 = fig.colorbar(im2, ax=ax[1], orientation='vertical')
            cbar2.set_label('Intensity')

            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'u_v_vectors_frame_{idx}.png'))
            plt.close()

    def compute_mean_velocities(self):
        self.mean_u = np.mean(self.u_vectors, axis=0)
        self.mean_v = np.mean(self.v_vectors, axis=0)
        return self.mean_u, self.mean_v

    def compute_fluctuating_components(self):
        if self.mean_u is None or self.mean_v is None:
            self.compute_mean_velocities()
        self.u_fluctuations = self.u_vectors - self.mean_u
        self.v_fluctuations = self.v_vectors - self.mean_v
        return self.u_fluctuations, self.v_fluctuations

    def plot_fluctuating_components(self, time_step=0):
        if self.u_fluctuations is None or self.v_fluctuations is None:
            self.compute_fluctuating_components()

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(self.u_fluctuations[time_step], cmap='jet', aspect='auto')
        plt.colorbar()
        plt.title("u' values at time step {}".format(time_step))
        plt.xlabel('X Position')
        plt.ylabel('Y Position')

        plt.subplot(1, 2, 2)
        plt.imshow(self.v_fluctuations[time_step], cmap='jet', aspect='auto')
        plt.colorbar()
        plt.title("v' values at time step {}".format(time_step))
        plt.xlabel('X Position')
        plt.ylabel('Y Position')

        plt.tight_layout()
        plt.show()

    def plot_rms_values(self, vectors, title):
        rms_values = self.compute_rms(vectors)

        plt.figure(figsize=(12, 6))

        plt.imshow(rms_values, cmap='jet', aspect='auto')
        plt.colorbar()
        plt.title(f'{title}_rms values', fontsize=16)
        plt.xlabel('X Position', fontsize=16)
        plt.ylabel('Y Position', fontsize=16)
        plt.tick_params(labelsize=14)

        plt.tight_layout()
        plt.show()

    def plot_3d_fluctuating_components(self, smooth=False, sigma=2):
        if self.u_fluctuations is None or self.v_fluctuations is None:
            self.compute_fluctuating_components()

        u_fluctuations_mean = np.mean(self.u_fluctuations, axis=1)
        v_fluctuations_mean = np.mean(self.v_fluctuations, axis=1)
        if smooth:
            # Apply Gaussian filter for smoothing
            u_fluctuations_mean = gaussian_filter(u_fluctuations_mean, sigma=sigma)
            v_fluctuations_mean = gaussian_filter(v_fluctuations_mean, sigma=sigma)

        time_steps = np.arange(self.u_fluctuations.shape[0])
        x_positions = np.arange(self.u_fluctuations.shape[2])

        X, T = np.meshgrid(x_positions, time_steps)

        fig = plt.figure(figsize=(18, 6))

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(X, T, u_fluctuations_mean, cmap='viridis')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Time Step')
        ax1.set_zlabel("u'")
        ax1.set_title("3D Plot of u' over Time")

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(X, T, v_fluctuations_mean, cmap='viridis')
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Time Step')
        ax2.set_zlabel("v'")
        ax2.set_title("3D Plot of v' over Time")

        plt.tight_layout()
        plt.show()

####################################################################################
def compare_binary_images_with_color_blending(binary_image_list, warped_img_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    iou_scores = []

    for idx, binary_img in enumerate(binary_image_list):
        if idx < len(warped_img_list):
            warped_img = warped_img_list[idx]
            _, binary_warped_img, _, _ = ed.lig_segment(warped_img, canny_threshold1=20, canny_threshold2=100, min_area=10, max_area=1000, k=3, plot_kmeans=None)

            # Compute Intersection over Union (IoU)
            intersection = np.logical_and(binary_img, binary_warped_img)
            union = np.logical_or(binary_img, binary_warped_img)
            iou_score = np.sum(intersection) / np.sum(union)
            iou_scores.append(iou_score)

            # Create color blended image
            binary_img_color = cv2.merge([binary_img, np.zeros_like(binary_img), np.zeros_like(binary_img)])  # Red
            binary_warped_img_color = cv2.merge([np.zeros_like(binary_warped_img), np.zeros_like(binary_warped_img), binary_warped_img])  # Blue

            blended_image = cv2.addWeighted(binary_img_color, 0.5, binary_warped_img_color, 0.5, 0)

            # Visualize the comparison
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.title('Original Binary Image')
            plt.imshow(binary_img, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title('Warped Binary Image')
            plt.imshow(binary_warped_img, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title('Blended Image')
            plt.imshow(blended_image)
            plt.axis('off')

            plt.suptitle(f'Frame {idx+1} - IoU: {iou_score:.4f}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'comparison_{idx+1}.png'))
            plt.close()

    return iou_scores

# # Directory to save the comparison images
# comparison_output_dir = "D:\\binary_image_comparisons_color"

# # Compare binary images with color blending and get IoU scores
# iou_scores = compare_binary_images_with_color_blending(binary_image_list, warped_img_list, comparison_output_dir)

# # Print average IoU score
# average_iou = np.mean(iou_scores)
# print(f'Average IoU Score: {average_iou:.4f}')

##################################################
# def extract_and_filter_contours(binary_image, min_contour_area=100):
#     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_contour_area]
#     return filtered_contours

# def measure_wave_parameters(contours):
#     wavelengths = []
#     amplitudes = []

#     for contour in contours:
#         epsilon = 0.01 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)

#         for i in range(len(approx) - 1):
#             p1 = approx[i][0]
#             p2 = approx[i+1][0]
#             wavelength = np.linalg.norm(p1 - p2)
#             wavelengths.append(wavelength)
#             amplitudes.append(abs(p1[1] - p2[1]))

#     mean_wavelength = np.mean(wavelengths) if wavelengths else 0
#     mean_amplitude = np.mean(amplitudes) if amplitudes else 0
#     return mean_wavelength, mean_amplitude

# def analyze_temporal_changes(binary_image_list, min_contour_area=100):
#     wavelengths = []
#     amplitudes = []

#     for binary_image in binary_image_list:
#         contours = extract_and_filter_contours(binary_image, min_contour_area=min_contour_area)
#         mean_wavelength, mean_amplitude = measure_wave_parameters(contours)
#         wavelengths.append(mean_wavelength)
#         amplitudes.append(mean_amplitude)

#     return wavelengths, amplitudes

# def plot_wave_parameters(wavelengths, amplitudes):
#     plt.figure(figsize=(12, 6))

#     plt.subplot(1, 2, 1)
#     plt.plot(wavelengths, marker='o')
#     plt.xlabel('Time Step')
#     plt.ylabel('Wavelength')
#     plt.title('Temporal Evolution of Wavelength')

#     plt.subplot(1, 2, 2)
#     plt.plot(amplitudes, marker='o')
#     plt.xlabel('Time Step')
#     plt.ylabel('Amplitude')
#     plt.title('Temporal Evolution of Amplitude')

#     plt.tight_layout()
#     plt.show()

# wavelengths, amplitudes = analyze_temporal_changes(binary_image_list)

# # Plot wave parameters over time
# plot_wave_parameters(wavelengths, amplitudes)

#################################################
# plt.imshow(binary_image_list[0], cmap='gray')
# plt.show()
# skeleton= morphology.skeletonize(binary_image_list[0])
# plt.imshow(skeleton, cmap='gray')
# plt.show()

# processed_binary_image, color_coded_image = process_binary_image(binary_image_list[0])
# # plt.imshow(processed_binary_image, cmap='gray')
# # plt.show()

# plt.imshow(color_coded_image)
# plt.axis('off')
# plt.show()

# cropped_im=edge_detect.process_and_crop_image_canny(img_list[0], "output_dir", canny_threshold1=50, canny_threshold2=100, margin=0)
# # cropped2=edge_detect.process_and_crop_image_canny_with_morph(img_list[0], "output_dir", canny_threshold1=50, canny_threshold2=100, margin=2, kernel_size=(2, 2), kernel_shape='rect', iterations=1)
# image=img_list[0]
# edges = cv2.Canny(image, threshold1=50, threshold2=100)
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# filled_image = np.zeros_like(edges)

# # Fill the contours
# cv2.drawContours(filled_image, contours, -1, (255), thickness=cv2.FILLED)

# # Convert the filled image to binary
# binary_image = filled_image > 0
# plt.title('Binary Image from Contours')
# plt.imshow(binary_image, cmap='gray')
# plt.show()
# edges = cv2.Canny(image, threshold1=20, threshold2=100)
# kernel = np.ones((3, 3), np.uint8)

# # Dilate the edges to enhance the ligament structures
# # dilated_edges = cv2.dilate(edges, kernel, iterations=1)
# # # Create a mask for flood filling
# inverted_edges = cv2.bitwise_not(edges)
# inverted_edges=edges

# # Create a mask for flood filling
# h, w = inverted_edges.shape
# mask = np.zeros((h+2, w+2), np.uint8)

# # Flood fill from the edges of the image
# cv2.floodFill(inverted_edges, mask, (0, 0), 255)

# # Invert back the filled image
# filled_image = cv2.bitwise_not(inverted_edges)

# # Convert the filled image to binary
# binary_image = filled_image > 0



# # Display the result
# plt.figure(figsize=(10, 10))

# # kernel = np.ones((2, 2), np.uint8)

# # # Apply morphological closing to fill the gaps inside the edges
# # closed_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
# # padded_edges = cv2.copyMakeBorder(edges, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)

# # # Apply morphological closing to fill gaps inside the edges
# # closed_image = cv2.morphologyEx(padded_edges, cv2.MORPH_CLOSE, kernel, iterations=3)

# # # Crop back to the original size
# # closed_image_cropped = closed_image[10:-10, 10:-10]
# # # Find contours in the closed image
# # contours, _ = cv2.findContours(closed_image_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # # Create an empty mask to draw the filled contours
# # filled_image = np.zeros_like(closed_image_cropped)

# # # Fill the contours
# # cv2.drawContours(filled_image, contours, -1, (255), thickness=cv2.FILLED)

# # # Convert the filled image to binary
# # binary_image = filled_image > 0

# # plt.subplot(1, 3, 1)
# # plt.title('Original Image')
# # plt.imshow(image, cmap='gray')

# # plt.subplot(1, 3, 2)
# # plt.title('Edges')
# # plt.imshow(edges, cmap='gray')

# # plt.subplot(1, 3, 3)
# plt.title('Binary Image from Edges')
# plt.imshow(binary_image, cmap='gray')
# plt.show()
#######################################################################