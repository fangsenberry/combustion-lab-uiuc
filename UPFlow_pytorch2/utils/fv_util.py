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
import pickle
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable

current_dir = Path(__file__).resolve()

# Go back two directories
two_dirs_up = current_dir.parents[2]

# Add the path to sys.path
sys.path.append(str(two_dirs_up))

import edge_detect as ed

########################################################################################
class FlowConfig:
    def __init__(self, **kwargs):
        self.trial_path = kwargs.get('trial_path', r"D:\test_cases\UPF_A01_C_DP_35_trial_12")
        self.img_path = kwargs.get('img_path', r"D:\final_corrected_512-complex-27-6-24.pth_inference")
        self.dir_ext = kwargs.get('dir_ext', r'flow_npy\result_')
        self.step = kwargs.get('step', 1)
        self.start_x = kwargs.get('start_x', 0)
        self.end_x = kwargs.get('end_x', None)
        self.start_y = kwargs.get('start_y', 0)
        self.end_y = kwargs.get('end_y', None)
        self.reverse_flow = kwargs.get('reverse_flow', False)
        self.binary_image_analysis = kwargs.get('binary_image_analysis', False)
        self.warp_analysis = kwargs.get('warp_analysis', False)
        self.custom_range = kwargs.get('custom_range', 25)
        self.hdf5_path = kwargs.get('hdf5_path', 'flow_data.h5')

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f:
            kwargs = pickle.load(f)
        return FlowConfig(**kwargs)

class FlowInitialization:
    def __init__(self, config):
        self.config = config
        self.data = {}

    @staticmethod
    def save_to_hdf5(file_path, **kwargs):
        with h5py.File(file_path, 'w') as f:
            for key, value in kwargs.items():
                if value is not None:
                    f.create_dataset(key, data=value)

    @staticmethod
    def load_from_hdf5(file_path):
        data = {}
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                data[key] = np.array(f[key])
        return data
    
    def plot_and_save_losses(self, loss_data_file=None, log_scale=True):
        """
        Plot and save loss curves for each specified loss in the data file.

        Parameters:
            loss_data_file (str): The name of the file containing the loss data. Defaults to 'loss_data.pkl' in the trial path.
            log_scale (bool): If True, the y-axis will be plotted on a logarithmic scale. Default is True.
        """
        # Default loss data file if not provided
        if loss_data_file is None:
            loss_data_file = os.path.join(self.config.trial_path, 'loss_data.pkl')
        
        if not os.path.exists(loss_data_file):
            print(f"Loss data could not be found for this trial path: {self.config.trial_path}")
            return  # Exit the function if the file does not exist
        
        # Create folder to save the plots
        folder_path = os.path.join(self.config.trial_path, 'loss_info')
        os.makedirs(folder_path, exist_ok=True)

        # Load loss data
        with open(loss_data_file, 'rb') as f:
            loaded_loss_data = pickle.load(f)

        # Extract the keys dynamically
        loss_keys = list(loaded_loss_data.keys())

        # Initialize a list to accumulate valid loss data for total loss calculation
        valid_loss_data = []

        # Iterate over each loss type
        for key in loss_keys:
            losses = loaded_loss_data[key]

            # Skip empty losses
            if not losses:
                continue

            # Plot and save the loss curve
            plt.figure(figsize=(10, 6))
            plt.plot(losses, label=key)
            plt.xlabel('Batch')
            plt.ylabel(key)
            if log_scale:
                plt.yscale('log')
            plt.title(f'{key} Over Time')
            plt.legend()
            # plt.grid(True)
            plt.savefig(os.path.join(folder_path, f'{key}.png'))
            # plt.show()
            plt.close()

            # Append to valid loss data for total loss calculation
            valid_loss_data.append(losses)

        # Calculate and plot total loss if there are valid losses
        if valid_loss_data:
            total_losses = [
                sum(loss_tuple) for loss_tuple in zip(*valid_loss_data)
            ]

            if any(total_losses):  # Check if total_losses contains any non-zero values
                plt.figure(figsize=(10, 6))
                plt.plot(total_losses, label='total_loss')
                plt.xlabel('Batch')
                plt.ylabel('Total Loss')
                if log_scale:
                    plt.yscale('log')
                plt.title('Total Loss Over Time')
                plt.legend()
                # plt.grid(True)
                plt.savefig(os.path.join(folder_path, 'total_loss.png'))
                # plt.show()
                plt.close()

    def process_and_save_data(self):
        flow_vis_list, self.data['img_list'], self.data['warped_img_list'], self.data['gradient_list'], self.data['binary_image_list'], self.data['x'], self.data['y'] = self.create_flow_lists(
            self.config.trial_path, 
            self.config.img_path, 
            self.config.dir_ext, 
            step=self.config.step, 
            start_x=self.config.start_x, 
            end_x=self.config.end_x, 
            start_y=self.config.start_y, 
            end_y=self.config.end_y, 
            reverse_flow=self.config.reverse_flow, 
            binary_image=self.config.binary_image_analysis, 
            warp=self.config.warp_analysis, 
            custom_range=self.config.custom_range
        )

        self.data['flow_vis_images'] = [flow_vis for flow_vis, _, _ in flow_vis_list]
        self.data['u_vectors'] = [u for _, u, _ in flow_vis_list]
        self.data['v_vectors'] = [v for _, _, v in flow_vis_list]

        self.save_to_hdf5(self.config.hdf5_path, **self.data)

    def load_data(self):
        self.data = self.load_from_hdf5(self.config.hdf5_path)

    @staticmethod
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

    @staticmethod
    def warp_image_skimage(image, flow):
        h, w = image.shape
        if flow.shape != (2, h, w):
            raise ValueError(f"Expected flow shape (2, {h}, {w}), but got {flow.shape}")

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x, map_y = x + flow[0], y + flow[1]
        coords = np.stack([map_y, map_x], axis=0)

        return warp(image, coords, mode='wrap', order=3)

    @staticmethod
    def normalize_image(image, target_mean, target_std):
        image = image.astype(np.float32)
        image_mean, image_std = np.mean(image), np.std(image)
        normalized_image = (image - image_mean) / image_std * target_std + target_mean
        return np.clip(normalized_image, 0, 255).astype(np.uint8)

    @staticmethod
    def convert_to_uint8(image):
        image = (image - image.min()) / (image.max() - image.min()) * 255
        return image.astype(np.uint8)

    @staticmethod
    def compute_gradient(image1, image2):
        gradient = image2.astype(np.float32) - image1.astype(np.float32)
        return gradient

    @staticmethod
    def gradient_to_heatmap(gradient, global_min, global_max):
        normalized_gradient = (gradient - global_min) / (global_max - global_min)
        heatmap = cm.coolwarm(normalized_gradient)  # Use 'coolwarm' colormap
        heatmap = (heatmap * 255).astype(np.uint8)
        return heatmap

    @staticmethod
    def save_heatmap_with_colorbar(heatmap, global_min, global_max, filename):
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))

        # Display the heatmap
        img = ax.imshow(heatmap, cmap='coolwarm', vmin=global_min, vmax=global_max)
        ax.axis('off')  # Turn off the axis for the heatmap

        # Create an axis on the right side of the heatmap with the same height
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("right", size="5%", pad=0.05)

        # Add the colorbar to the new axis
        cbar = plt.colorbar(img, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=8)  # Adjust the size of the color bar ticks
        cbar.set_label('Gradient Intensity', size=8)
        
        # Save the figure
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    @staticmethod
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
    
    @staticmethod
    def visualize_flow(flow, type='basic'):
        """Visualize optical flow."""
        hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
        mag, ang = cv2.cartToPolar(flow[0], flow[1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        color_map = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if type == 'pastel':
            pastel_map = FlowInitialization.pastel_colormap()
            color_map = pastel_map(hsv[..., 2]).astype(np.uint8)
        elif type == 'custom':
            color_map = flow_viz.flow_to_image(flow)
        return color_map

    @staticmethod
    def flow_params(flow, step=10, start_x=0, end_x=None, start_y=0, end_y=None, reverse_flow=False, flow_vis_type='basic'):
        """Extract flow parameters for visualization."""
        u = flow[0, ::step, ::step]
        v = flow[1, ::step, ::step]

        if reverse_flow:
            u, v = -u, -v

        flow_vis = FlowInitialization.visualize_flow(flow, type=flow_vis_type)

        return flow_vis, u, v

    @staticmethod
    def plot_flow_vectors(flow_vis_list, img_list, binary_image_list, x, y, start_x=0, end_x=None, start_y=0, end_y=None, step=10):
        for idx in range(len(flow_vis_list)):
            flow_vis, u, v = flow_vis_list[idx]
            original_img = img_list[idx]
            binary_img = binary_image_list[idx]

            # Crop the images
            cropped_img = FlowInitialization.manual_crop(original_img, start_x=start_x, end_x=end_x, start_y=start_y, end_y=end_y)
            cropped_binary_img = FlowInitialization.manual_crop(binary_img, start_x=start_x, end_x=end_x, start_y=start_y, end_y=end_y)

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

            plt.axis('off')
            plt.imshow(cropped_img, cmap='gray')
            plt.tight_layout()
            plt.show()
            plt.close()

    @staticmethod
    def plot_flow_vectors_as_video(flow_vis_list, img_list, binary_image_list, x, y, start_x=0, end_x=None, start_y=0, end_y=None, step=10, video_filename='flow_vectors.mp4'):
        fig, ax = plt.subplots(figsize=(10, 10))

        # Determine the cropping bounds for consistent plot size
        if end_x is None:
            end_x = img_list[0].shape[1]
        if end_y is None:
            end_y = img_list[0].shape[0]

        def update(idx):
            ax.clear()
            flow_vis, u, v = flow_vis_list[idx]
            original_img = img_list[idx]
            binary_img = binary_image_list[idx]

            # Crop the images
            cropped_img = FlowInitialization.manual_crop(original_img, start_x=start_x, end_x=end_x, start_y=start_y, end_y=end_y)
            cropped_binary_img = FlowInitialization.manual_crop(binary_img, start_x=start_x, end_x=end_x, start_y=start_y, end_y=end_y)

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

    @staticmethod
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

        plt.savefig('flow_and_colorwheel.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    @staticmethod
    def numerical_sort_key(file_name):
        return int(file_name.split('_')[-1].split('.')[0])

    @staticmethod
    def flow_checks(flow):
        if flow.ndim == 4:
            # Assuming the shape is (N, 2, H, W) and we take the first element (N should be 1 for batch size 1)
            flow = flow[0]

        if flow.shape[0] != 2:
            # Move final channel to first channel
            flow = np.moveaxis(flow, -1, 0)

        return flow

    def create_flow_lists(self, directory, im_dir, base, step=10, start_y=0, end_y=None, start_x=0, end_x=None, reverse_flow=False, binary_image=False, warp=False, custom_range=25, flow_vis_type='basic'):
        flow_vis_list = []
        img_list = []
        warped_img_list = []
        gradient_list = []
        binary_image_list = []

        target_height = None
        target_width = None

        image_files = sorted([f for f in os.listdir(im_dir) if f.endswith('.png')], key=FlowInitialization.numerical_sort_key)
        if custom_range == 'end':
            custom_range = len(image_files) - 1

        for idx in range(custom_range):
            # Load flow files
            filepath = os.path.join(directory, f"{base}{idx}.npy")
            flow = FlowInitialization.flow_checks(np.load(filepath))

            # Determine target size from the first flow
            if idx == 0:
                target_height, target_width = flow.shape[1], flow.shape[2]
                y, x = np.mgrid[0:target_height:step, 0:target_width:step]

            flow_vis, u, v = FlowInitialization.flow_params(flow, step, start_x=start_x, end_x=end_x, start_y=start_y, end_y=end_y, reverse_flow=reverse_flow, flow_vis_type=flow_vis_type)
            flow_vis_list.append((flow_vis, u, v))

            # Load and process images
            img_path = os.path.join(im_dir, image_files[idx])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            # Set image to have a mean of 127
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
                prev_flow_path = os.path.join(directory, f"{base}{idx - 1}.npy")
                prev_flow = FlowInitialization.flow_checks(np.load(prev_flow_path))
                warped_img = FlowInitialization.warp_image_skimage(img_list[idx - 1], prev_flow)
                warped_img_list.append(warped_img)

                # Compute the gradient between the warped image and the current image
                gradient = FlowInitialization.compute_gradient(cropped_img, warped_img)
                gradient_list.append(gradient)
            else:
                warped_img_list.append(cropped_img)
                gradient_list.append(np.zeros_like(cropped_img))  # No gradient for the first frame

        return flow_vis_list, img_list, warped_img_list, gradient_list, binary_image_list, x, y

    def save_warped_images(self, warped_img_list):
        folder_path = os.path.join(self.config.trial_path, 'warped_images')
        os.makedirs(folder_path, exist_ok=True)

        for i, warped_img in enumerate(warped_img_list):
            if warped_img.dtype != np.uint8:
                warped_img = FlowInitialization.convert_to_uint8(warped_img)
            img_path = os.path.join(folder_path, f'warped_image_{i}.png')
            Image.fromarray(warped_img).save(img_path)

    def generate_global_heatmaps(self, gradient_list):
        folder_path = os.path.join(self.config.trial_path, 'gradient_heatmaps')
        os.makedirs(folder_path, exist_ok=True)

        # Combine all gradients
        all_gradients = np.concatenate([gradient.flatten() for gradient in gradient_list])
        absolute_gradients = np.abs(all_gradients)

        # Compute global min and max
        global_min = np.min(all_gradients)
        global_max = np.max(all_gradients)
        # print(f"Global min: {global_min}, Global max: {global_max}")

        # Compute average of absolute gradients
        average_absolute_gradient = np.mean(absolute_gradients)
        # print(f"Average of the absolute gradients: {average_absolute_gradient}")

        # Generate and save heatmap images
        for i, gradient in enumerate(gradient_list):
            heatmap = FlowInitialization.gradient_to_heatmap(gradient, global_min, global_max)
            filename = os.path.join(folder_path, f"gradient_{i}.png")
            FlowInitialization.save_heatmap_with_colorbar(heatmap, global_min, global_max, filename)
    
    def average_heatmaps_with_confidence_intervals(self, gradient_list):
        folder_path = os.path.join(self.config.trial_path, 'average_heatmaps_2D')
        os.makedirs(folder_path, exist_ok=True)

        # Calculate the mean and standard deviation per pixel across all frames
        absolute_gradients = np.abs(np.array(gradient_list))
        mean_error_per_pixel = np.mean(absolute_gradients, axis=0)
        std_dev_per_pixel = np.std(absolute_gradients, axis=0)

        # Calculate MSE and RMSE
        mse_error_per_pixel = np.mean(absolute_gradients ** 2, axis=0)
        overall_mse = np.mean(mse_error_per_pixel)
        overall_rmse = np.sqrt(overall_mse)

        # Save the 2D heatmap of the mean error
        global_min = np.min(mean_error_per_pixel)
        global_max = np.max(mean_error_per_pixel)
        heatmap = self.gradient_to_heatmap(mean_error_per_pixel, global_min, global_max)
        heatmap_filename = os.path.join(folder_path, "mean_error_heatmap.png")
        self.save_heatmap_with_colorbar(heatmap, global_min, global_max, heatmap_filename)

        # Overlay standard deviation contours on the heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        img = ax.imshow(mean_error_per_pixel, cmap='coolwarm', vmin=global_min, vmax=global_max)
        ax.axis('off')
        
        # Overlay standard deviation as contours
        contour_levels = np.linspace(np.min(std_dev_per_pixel), np.max(std_dev_per_pixel), 10)
        cs = ax.contour(std_dev_per_pixel, levels=contour_levels, colors='black', linewidths=0.5)
        ax.clabel(cs, inline=1, fontsize=8, fmt='%1.2f')

        # Add colorbar for the mean error heatmap
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(img, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('Gradient Intensity', size=8)
        
        # Save the figure with contours
        overlay_filename = os.path.join(folder_path, "mean_error_with_std_contours.png")
        plt.savefig(overlay_filename, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # Calculate and save the average error, standard deviation, MSE, and RMSE for the entire dataset
        overall_mean_error = np.mean(mean_error_per_pixel)
        overall_std_dev = np.mean(std_dev_per_pixel)
        with open(os.path.join(self.config.trial_path, "error_metrics.txt"), "w") as f:
            f.write(f"Average Error for the dataset: {overall_mean_error}\n")
            f.write(f"Overall Standard Deviation for the dataset: {overall_std_dev}\n")
            f.write(f"Overall MSE for the dataset: {overall_mse}\n")
            f.write(f"Overall RMSE for the dataset: {overall_rmse}\n")

    def plot_and_save_flow(self, flow_vis_list, img_list, x, y, output_dir='output_directory', plot_type='flow_vis', fps=2, mask=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig, ax = plt.subplots(1, 1, figsize=(18, 6))

        def update_plot(i):
            ax.cla()
            
            flow_vis, u, v = flow_vis_list[i]
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

    @staticmethod
    def save_binary_original_overlap(original_image, binary_image, path=None):
        overlay_image = np.zeros_like(original_image)
        mask = binary_image > 0
        overlay_image[mask] = original_image[mask]
        plt.imshow(overlay_image, cmap='gray')
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()

    @staticmethod
    def save_synthetic_warp_pairs(img_list, warped_img_list, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in range(len(img_list)):
            # Save the first image from img_list
            img1 = FlowInitialization.convert_to_uint8(img_list[i])
            img1_path = os.path.join(save_dir, f'image_{2 * i + 1:04d}.png')
            Image.fromarray(img1).save(img1_path)

            # Save the second image from warped_img_list if it exists
            if i + 1 < len(warped_img_list):
                img2 = FlowInitialization.convert_to_uint8(warped_img_list[i + 1])
                img2_path = os.path.join(save_dir, f'image_{2 * i + 2:04d}.png')
                Image.fromarray(img2).save(img2_path)
    ##################################################

class FlowAnalysis:
    def __init__(self, config, flow_vis_list, binary_mask_list):
        self.u_vectors, self.v_vectors = self.extract_flow_vectors(flow_vis_list)
        self.binary_masks = np.array(binary_mask_list)  # Store the list of binary masks
        self.mean_u = None
        self.mean_v = None
        self.u_fluctuations = None
        self.v_fluctuations = None
        self.config = config

    def extract_flow_vectors(self, flow_vis_list):
        u_vectors = []
        v_vectors = []

        for flow_vis in flow_vis_list:
            _, u, v = flow_vis  # Assuming flow_vis_list contains tuples with flow vectors
            u_vectors.append(u)
            v_vectors.append(v)
        
        # Crop the first 10 columns of the flow vectors
        u_vectors = np.array(u_vectors)
        v_vectors = np.array(v_vectors)
        
        return u_vectors, v_vectors

    def compute_flow_quantities(self):
        # Get the velocity gradients using numpy.gradient
        du_dx = np.gradient(self.u_vectors, axis=2)
        du_dy = np.gradient(self.u_vectors, axis=1)
        dv_dx = np.gradient(self.v_vectors, axis=2)
        dv_dy = np.gradient(self.v_vectors, axis=1)

        # 1. Compute Vorticity (dv/dx - du/dy)
        vorticity = dv_dx - du_dy

        # 2. Compute Shear Stress (shear components)
        shear_stress_x = du_dy  # Shear stress in x direction (du/dy)
        shear_stress_y = dv_dx  # Shear stress in y direction (dv/dx)

        # 3. Compute Strain Rate Tensor (normal and shear strain rate)
        strain_rate_xx = du_dx  # Normal strain rate in x direction
        strain_rate_yy = dv_dy  # Normal strain rate in y direction
        strain_rate_xy = 0.5 * (du_dy + dv_dx)  # Symmetric shear strain rate

        return vorticity, shear_stress_x, shear_stress_y, strain_rate_xx, strain_rate_yy, strain_rate_xy

    def apply_mask(self, quantity, mask_value=np.nan):
        num_frames = quantity.shape[0]  # Assuming quantity has shape (num_frames, height, width)
        masked_quantity = np.zeros_like(quantity)

        for frame_idx in range(num_frames):
            # Mask the regions outside the binary mask with the mask_value
            masked_quantity[frame_idx] = np.where(self.binary_masks[frame_idx], quantity[frame_idx], mask_value)
        
        return masked_quantity

    # Function to compute global min and max across all frames (ignoring NaN values)
    def get_global_min_max(self, quantity):
        global_min = np.nanmin(quantity)
        global_max = np.nanmax(quantity)
        return global_min, global_max

    def plot_flow_quantity(self, quantity, title, mask_value=np.nan, save_dir="plots", save_data='example', file_format="png"):
        # Apply mask to the quantity (e.g., vorticity, shear stress, strain rate)
        save_dir=os.path.join(save_dir, f'flow_analysis_plots/{save_data}')

        masked_quantity = self.apply_mask(quantity, mask_value)

        # Get the global min and max for the colorbar range
        global_min, global_max = self.get_global_min_max(masked_quantity)

        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Plot each frame with the same colorbar range and masked regions
        num_frames = masked_quantity.shape[0]

        for frame_idx in range(num_frames):
            plt.figure(figsize=(12, 6))

            # Plot the masked quantity for this frame, using global color limits
            plt.imshow(masked_quantity[frame_idx], cmap='jet', aspect='auto', vmin=global_min, vmax=global_max)

            plt.colorbar()
            plt.title(f"{title} - Frame {frame_idx}", fontsize=16)
            plt.xlabel('X Position', fontsize=16)
            plt.ylabel('Y Position', fontsize=16)
            plt.tight_layout()

            # Save the plot to a file
            file_name = os.path.join(save_dir, f"{title.replace(' ', '_').lower()}_frame_{frame_idx}.{file_format}")
            plt.savefig(file_name)
            plt.close()  # Close the plot to free up memory

        print(f"Plots saved to {save_dir}")
    
    # Function to plot vorticity
    def plot_vorticity(self, save_dir, save_data):
        vorticity, _, _, _, _, _ = self.compute_flow_quantities()
        self.plot_flow_quantity(vorticity, "Vorticity Field", save_dir=save_dir, save_data=save_data)

    # Function to plot shear stress
    def plot_shear_stress(self, save_dir, save_data):
        _, shear_stress_x, shear_stress_y, _, _, _ = self.compute_flow_quantities()
        self.plot_flow_quantity(shear_stress_x, "Shear Stress in X Direction", save_dir=save_dir, save_data=save_data)
        self.plot_flow_quantity(shear_stress_y, "Shear Stress in Y Direction", save_dir=save_dir, save_data=save_data)

    # Function to plot strain rate
    def plot_strain_rate(self, save_dir, save_data):
        _, _, _, strain_rate_xx, strain_rate_yy, strain_rate_xy = self.compute_flow_quantities()
        self.plot_flow_quantity(strain_rate_xx, "Normal Strain Rate (xx)", save_dir=save_dir, save_data=save_data)
        self.plot_flow_quantity(strain_rate_yy, "Normal Strain Rate (yy)", save_dir=save_dir, save_data=save_data)
        self.plot_flow_quantity(strain_rate_xy, "Shear Strain Rate (xy)", save_dir=save_dir, save_data=save_data)

    def compute_rms(self, vectors):
        rms_values = np.sqrt(np.mean(vectors**2, axis=0))
        return rms_values

    def save_flow_vectors(self):
        # Create separate directories for u and v plots
        u_dir = os.path.join(self.config.trial_path, 'UV_plots/u_plots')
        v_dir = os.path.join(self.config.trial_path, 'UV_plots/v_plots')
        os.makedirs(u_dir, exist_ok=True)
        os.makedirs(v_dir, exist_ok=True)

        # Determine global min and max for u and v vectors
        u_min, u_max = float('inf'), float('-inf')
        v_min, v_max = float('inf'), float('-inf')

        for u, v in zip(self.u_vectors, self.v_vectors):
            u_min = min(u_min, u.min())
            u_max = max(u_max, u.max())
            v_min = min(v_min, v.min())
            v_max = max(v_max, v.max())

        # Plot and save u and v vectors with consistent axis limits
        for idx, (u, v) in enumerate(zip(self.u_vectors, self.v_vectors)):
            # Plot u vectors
            fig_u, ax_u = plt.subplots(figsize=(6, 6))
            im_u = ax_u.imshow(u, cmap='jet', aspect='auto', vmin=u_min, vmax=u_max)
            ax_u.set_title(f'u vectors - Frame {idx}')
            cbar_u = fig_u.colorbar(im_u, ax=ax_u, orientation='vertical')
            cbar_u.set_label('Intensity')
            
            # Save u vector plot in u_plots directory
            plt.tight_layout()
            plt.savefig(os.path.join(u_dir, f'u_vectors_frame_{idx}.png'))
            plt.close(fig_u)

            # Plot v vectors
            fig_v, ax_v = plt.subplots(figsize=(6, 6))
            im_v = ax_v.imshow(v, cmap='jet', aspect='auto', vmin=v_min, vmax=v_max)
            ax_v.set_title(f'v vectors - Frame {idx}')
            cbar_v = fig_v.colorbar(im_v, ax=ax_v, orientation='vertical')
            cbar_v.set_label('Intensity')
            
            # Save v vector plot in v_plots directory
            plt.tight_layout()
            plt.savefig(os.path.join(v_dir, f'v_vectors_frame_{idx}.png'))
            plt.close(fig_v)

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