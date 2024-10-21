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
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import torch
import piq
import lpips
import torchmetrics

########################################################################################
class FlowConfig:
    def __init__(self, **kwargs):
        self.trial_path = kwargs.get('trial_path', r"D:\test_cases\UPF_A01_C_DP_35_trial_12")
        self.img_path = kwargs.get('img_path', r"D:\final_corrected_512-complex-27-6-24.pth_inference")
        self.dir_ext = kwargs.get('dir_ext', r'flow_npy\result_')
        self.step = kwargs.get('step', 1)
        self.custom_range = kwargs.get('custom_range', 25)
        self.array_save_path = kwargs.get('array_save_path', 'flow_data.npz')
        self.file_format = kwargs.get('file_format', 'npz')
        self.image_save_range = kwargs.get('image_save_range', 35)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f:
            kwargs = pickle.load(f)
        return FlowConfig(**kwargs)
########################################################################################
class FlowInitialization:
    def __init__(self, config):
        self.config = config
        self.data = {}
        self.single_image = None
        self.binary_image = None
        self.warped_image = None
        self.file_format = self.config.file_format
        self.step= self.config.step
        self.image_save_range = self.config.image_save_range

    def save_data(self, file_path, append=False, **arrays):
        """
        Save data to either NPZ or HDF5 format.
        
        Args:
            file_path: Path to save the file.
            file_format: 'npz' or 'hdf5', depending on the desired format.
            arrays: Data to be saved as key-value pairs.
        """
        total_keys = len(arrays)  # Count total number of datasets to save
        with tqdm(total=total_keys, desc=f"Saving data to {self.file_format.upper()} file", unit="dataset") as pbar:
            if self.file_format == 'npz':
                np.savez_compressed(file_path, **arrays)
                pbar.update(total_keys)  # Since all data is saved at once, mark all as done
            elif self.file_format == 'hdf5':
                # Save data to HDF5 file
                mode = 'a' if append else 'w'
                with h5py.File(file_path, mode) as f:
                    for key, value in arrays.items():
                        if value is not None:
                            if append and key in f:
                                # If appending, delete existing dataset before writing
                                del f[key]
                            f.create_dataset(key, data=value)
                            pbar.update(1)  # Update progress bar after saving each dataset
            else:
                raise ValueError(f"Unsupported file format: {self.file_format}")
        print(f"Data saved to {file_path}")

    ### Unified Load Function for NPZ and HDF5 ###
    def load_data(self, file_path):
        """
        Load data from either NPZ or HDF5 format.

        Args:
            file_path: Path to the file to be loaded.
            file_format: 'npz' or 'hdf5', depending on the desired format.

        Returns:
            A dictionary with the loaded data.
        """
        data = {}
        if self.file_format == 'npz':
            loaded_data = np.load(file_path)
            data = {key: loaded_data[key] for key in loaded_data}
        elif self.file_format == 'hdf5':
            with h5py.File(file_path, 'r') as f:
                for key in f.keys():
                    data[key] = np.array(f[key])
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")
        print(f"Data loaded from {file_path}")
        return data

    ### Centralized Data Preparation Method ###
    def process_and_save_data(self):
        """
        Process data and save it in either NPZ or HDF5 format.

        Args:
            file_format: The format to save the data in ('npz' or 'hdf5').
        """
        flow_data = self.create_flow_arrays()
        print(f"Flow data arrays created with shapes: {[(key, value.shape) for key, value in flow_data.items()]}")

        # Prepare arrays for saving
        arrays_to_save = {
            'flow_vis_map': flow_data['flow_vis_array'],
            'u_vectors': flow_data['u_array'],
            'v_vectors': flow_data['v_array'],
            'original_image_array': flow_data['original_image_array'],
            'warped_image_array': flow_data['warped_image_array'],
            'binary_image_array': flow_data['binary_image_array'],
            'x_positions': flow_data['x'],
            'y_positions': flow_data['y']
        }

        # Determine file path and save
        file_path = os.path.join(self.config.trial_path, self.config.array_save_path)
        self.save_data(file_path, **arrays_to_save)
        self.save_warped_images(flow_data['warped_image_array'])

    
    @staticmethod
    def numerical_sort_key(file_name):
        return int(file_name.split('_')[-1].split('.')[0])
        
    def plot_and_save_losses(self, log_scale=True):
        """
        Plot and save loss curves for each specified loss in the data file.

        Parameters:
            loss_data_file (str): The name of the file containing the loss data. Defaults to 'loss_data.pkl' in the trial path.
            log_scale (bool): If True, the y-axis will be plotted on a logarithmic scale. Default is True.
        """
        # Default to 'loss_data.pkl' if no file is provided
        loss_data_file = os.path.join(self.config.trial_path, 'loss_data.pkl')

        # Check if the loss data file exists
        if not os.path.exists(loss_data_file):
            print(f"Loss data could not be found for this trial path: {loss_data_file}")
            return  # Exit the function if the file does not exist

        # Create a folder to save the plots
        folder_path = os.path.join(self.config.trial_path, 'loss_plots')
        os.makedirs(folder_path, exist_ok=True)

        # Load loss data from the file
        try:
            with open(loss_data_file, 'rb') as f:
                loss_data = pickle.load(f)
        except Exception as e:
            print(f"Error loading loss data from {loss_data_file}: {e}")
            return

        # Ensure loss_data is a dictionary
        if not isinstance(loss_data, dict):
            print(f"Invalid loss data format in {loss_data_file}")
            return

        # Initialize a list to accumulate valid loss data for total loss calculation
        valid_loss_data = []

        # Iterate over each loss key in the data and plot the corresponding loss curve
        for loss_type, losses in loss_data.items():
            # Skip if there are no losses
            if not losses:
                continue

            # Use the helper method to plot each individual loss
            self._plot_single_loss(loss_type, losses, log_scale, folder_path)

            # Collect the loss data for total loss calculation
            valid_loss_data.append(losses)

        # If valid losses were found, calculate and plot the total loss
        if valid_loss_data:
            # Zip the losses together and sum them to compute total losses per batch
            total_losses = [sum(losses_per_batch) for losses_per_batch in zip(*valid_loss_data)]

            # Plot and save the total loss curve using the helper method
            if any(total_losses):  # Ensure there are non-zero losses
                self._plot_single_loss('Total Loss', total_losses, log_scale, folder_path)

        print(f"Loss plots saved in {folder_path}")

    @staticmethod
    def _plot_single_loss(loss_type, losses, log_scale, save_dir):
        """
        Helper function to plot and save individual loss curves.

        Parameters:
            loss_type (str): The type/name of the loss (used for labels and title).
            losses (list): The loss values over time (typically across batches).
            log_scale (bool): Whether to use logarithmic scaling for the y-axis.
            save_dir (str): Directory to save the plot.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label=loss_type)
        plt.xlabel('Batch')
        plt.ylabel(loss_type)
        if log_scale:
            plt.yscale('log')
        plt.title(f'{loss_type} Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'{loss_type}.png'))
        plt.close()

    @staticmethod
    def pastel_colormap():
        """Create a pastel colormap."""
        colors = [
        (170 / 255.0, 210 / 255.0, 255 / 255.0),  # Slightly deeper light blue
        (255 / 255.0, 170 / 255.0, 170 / 255.0),  # Slightly deeper light red
        (170 / 255.0, 255 / 255.0, 170 / 255.0),  # Slightly deeper light green
        (255 / 255.0, 255 / 255.0, 170 / 255.0),  # Slightly deeper light yellow
        (255 / 255.0, 170 / 255.0, 255 / 255.0),  # Slightly deeper light magenta
        (170 / 255.0, 255 / 255.0, 255 / 255.0),  # Slightly deeper light cyan
        (255 / 255.0, 210 / 255.0, 170 / 255.0),  # Slightly deeper light orange
        (210 / 255.0, 170 / 255.0, 255 / 255.0),  # Slightly deeper light violet
        (210 / 255.0, 255 / 255.0, 210 / 255.0)   # Slightly deeper light pastel green
    ]
        return ListedColormap(colors, name='pastel')
    
    @staticmethod
    def visualize_flow(flow, type='basic'):
        """Visualize optical flow."""
        hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
        mag, ang = cv2.cartToPolar(-flow[0], flow[1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        if type == 'pastel':
            # Normalize hsv[..., 2] to the range [0, 1]
            normalized_value = hsv[..., 2] / 255.0

            # Get the pastel colormap
            pastel_map = FlowInitialization.pastel_colormap()

            # Apply the colormap (this returns RGBA)
            rgba_image = pastel_map(normalized_value)

            # Convert RGBA to RGB by ignoring the alpha channel
            color_map = (rgba_image[..., :3] * 255).astype(np.uint8)
        elif type == 'custom':
            #move first channel to last channel

            
            flow=np.moveaxis(flow, 0, -1)
            # print(flow.shape)
            color_map = flow_viz.flow_to_image(flow)
        elif type == 'basic':

            # Flip the hues around the midpoint (90 degrees)
            # hsv[..., 0] = (hsv[..., 0] + 180) % 180  # Shift hues by 180 degrees and wrap using modulo 180

            color_map = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            raise ValueError(f"Invalid flow visualization type: {type}")
        return color_map

    def flow_checks(self, flow):
        if flow.ndim == 4:
            # Assuming the shape is (N, 2, H, W) and we take the first element (N should be 1 for batch size 1)
            flow = flow[0]

        if flow.shape[0] != 2:
            # Move final channel to first channel
            flow = np.moveaxis(flow, -1, 0)

        return flow

    def flow_params(self, flow):
        u = flow[0, ::self.step, ::self.step]
        v = flow[1, ::self.step, ::self.step]
        flow_vis = FlowInitialization.visualize_flow(flow)
        return flow_vis, u, v

    def _get_image_files(self):
        return sorted([f for f in os.listdir(self.config.img_path) if f.endswith('.png')],
                      key=FlowInitialization.numerical_sort_key)

    def _load_flow(self, idx):
        filepath = os.path.join(self.config.trial_path, f"{self.config.dir_ext}{idx}.npy")
        flow = self.flow_checks(np.load(filepath))  # Use self to call the instance method
        flow_vis, u, v = self.flow_params(flow)     # Use self to call the instance method
        return flow, flow_vis, u, v

    def _process_image(self, image_files, idx):
        img_path = os.path.join(self.config.img_path, image_files[idx])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        
        # Ensure that self.single_image is assigned here
        single_image = np.clip((img - np.mean(img)) + 128, 0, 255).astype(np.uint8)
        binary_image= self._apply_binary_segmentation(single_image)
        return single_image, binary_image
    
    def _apply_binary_segmentation(self, image):
        adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                                cv2.THRESH_BINARY, 35, 7)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded_image = cv2.erode(adaptive_thresh, kernel1, iterations=1)
        dilated_image = cv2.dilate(eroded_image, kernel2, iterations=1)
        return cv2.bitwise_not(dilated_image)

    def warp_image_skimage(self, image, flow):
        """
        Warp the given image using the provided optical flow.

        Args:
            image (numpy array): The image to warp.
            flow (numpy array): The optical flow used for warping.

        Returns:
            numpy array: The warped image.
        """
        h, w = image.shape  # Now using the passed 'image' argument
        if flow.shape != (2, h, w):
            raise ValueError(f"Expected flow shape (2, {h}, {w}), but got {flow.shape}")
        
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x, map_y = x + flow[0], y + flow[1]
        coords = np.stack([map_y, map_x], axis=0)

        warped_image = warp(image, coords, mode='reflect', order=3)
    
        # Now multiply by 255 to get it back into the [0, 255] range
        warped_image = warped_image * 255.0  # Rescale to [0, 255]

        return warped_image.astype(np.uint8)

    def _warp_lists(self, idx, image_list):
        if idx > 0:
            prev_flow = self.flow_checks(np.load(os.path.join(self.config.trial_path, f"{self.config.dir_ext}{idx - 1}.npy")))
            self.warped_image = self.warp_image_skimage(image_list[idx-1], prev_flow)
        else:
            self.warped_image = self.single_image

    def save_warped_images(self, warped_img_array):
        """
        Save a specified number of warped images from the warped image array.

        Parameters:
            warped_img_array (numpy array): Array containing warped images.
            num_images_to_save (int, optional): The number of images to save. If None, all images are saved. Default is None.
        """
        folder_path = os.path.join(self.config.trial_path, 'warped_images')
        os.makedirs(folder_path, exist_ok=True)

        # Determine how many images to save
        num_images = warped_img_array.shape[0]
        if self.image_save_range is not None:
            num_images = min(num_images, self.image_save_range)

        for i in range(num_images):
            warped_img = warped_img_array[i]
            
            if warped_img.dtype != np.uint8:
                warped_img = self.convert_to_uint8(warped_img)
            
            img_path = os.path.join(folder_path, f'warped_image_{i}.png')
            Image.fromarray(warped_img).save(img_path)

        print(f"{num_images} warped images saved in: {folder_path}")
    
    def create_flow_arrays(self):
        flow_vis_list, u_vectors, v_vectors, img_list, warped_img_list, binary_image_list = [], [], [], [], [], []

        image_files = self._get_image_files()
        if self.config.custom_range == 'end':
                custom_range = len(image_files)  # This converts 'end' to the actual number of files
        else:
            custom_range = int(self.config.custom_range)  # Ensure custom_range is converted to an integer

        # Process each image within the custom range
        print(custom_range)
        with tqdm(total=custom_range, desc="Processing Images and Flow Files", unit="file") as pbar:
            for idx in range(custom_range):
                flow_file_path = os.path.join(self.config.trial_path, f"{self.config.dir_ext}{idx}.npy")
                if not os.path.exists(flow_file_path):
                    break
                else:
                    flow, flow_vis, u, v = self._load_flow(idx)

                if idx == 0:
                    target_height, target_width = flow.shape[1], flow.shape[2]
                    self.y, self.x = np.mgrid[0:target_height:self.step, 0:target_width:self.step]

                self.single_image, self.binary_image=self._process_image(image_files, idx)
                flow_vis_list.append(flow_vis)
                u_vectors.append(u)
                v_vectors.append(v)
                img_list.append(self.single_image)
                binary_image_list.append(self.binary_image)
                self._warp_lists(idx, img_list)

                warped_img_list.append(self.warped_image)
                pbar.update(1)

        # Convert lists to numpy arrays
        warped_img_array = np.array(warped_img_list)-np.mean(np.array(warped_img_list))+128
        warped_img_array = np.clip(warped_img_array, 0, 255).astype(np.uint8)
        
        arrays = {
            'flow_vis_array': np.array(flow_vis_list),
            'u_array': np.array(u_vectors),
            'v_array': np.array(v_vectors),
            'original_image_array': np.array(img_list),
            'warped_image_array': warped_img_array,
            'binary_image_array': np.array(binary_image_list),
            'x': self.x,
            'y': self.y
        }

        return arrays
    
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
    
    def run_save_case(self):
        self.process_and_save_data()
        self.plot_and_save_losses()

    def load_and_display_data(self):
        file_path = os.path.join(self.config.trial_path, self.config.array_save_path)
        self.data = self.load_data(file_path)
        print(f"Loaded data keys: {self.data.keys()}")
    
    def run_gradient_calculations(self):
        self.load_and_display_data()
        gradient_analysis = GradientAnalysis(self)
        # gradient_analysis.generate_global_heatmaps()
        # gradient_analysis.average_heatmaps_with_confidence_intervals()
        
        # Save SSIM maps
        # gradient_analysis.generate_dssim_heatmaps()
        gradient_analysis.compute_metric(gradient_analysis.compute_lpips, 'LPIPS')
        gradient_analysis.generate_heatmaps('LPIPS', gradient_analysis.lpips_maps, 'magma', 'LPIPS Value')
        # gradient_analysis.generate_msssim_heatmaps()
        


    
class GradientAnalysis:
    def __init__(self, flow_case):
        self.warped_image_array = flow_case.data['warped_image_array']
        self.original_image_array = flow_case.data['original_image_array']
        self.global_min, self.global_max = None, None
        self.image_save_range = flow_case.image_save_range
        self.gradient_array = None
        self.trial_path = flow_case.config.trial_path
        self.ssim_maps = []
        self.dssim_maps = []
        self.lpips_maps = []
        self.msssim_maps = []
        self.lpips_values = []
        self.msssim_values = []
        if len(self.original_image_array) > 0:
            self.image_height, self.image_width = self.original_image_array[0].shape[:2]

    def compute_gradient(self):
        """
        Compute the gradient between the original and warped images across all frames.

        Args:
            img_array (numpy array): Original image array (shape: num_frames x H x W).
            warped_img_array (numpy array): Warped image array (shape: num_frames x H x W).

        Returns:
            numpy array: Gradient array (shape: num_frames x H x W).
        """
        return self.warped_image_array.astype(np.float32) - self.original_image_array.astype(np.float32)

    def gradient_to_heatmap(self, gradient):
        normalized_gradient = (gradient - self.global_min) / (self.global_max - self.global_min)
        heatmap = cm.coolwarm(normalized_gradient)  # Use 'coolwarm' colormap
        heatmap = (heatmap * 255).astype(np.uint8)
        return heatmap

    
    def save_heatmap_with_colorbar(self, heatmap, filename):
        fig, ax = plt.subplots(figsize=(10, 8))
        img = ax.imshow(heatmap, cmap='coolwarm', vmin=self.global_min, vmax=self.global_max)
        ax.axis('off')

        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(img, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('Gradient Intensity', size=8)

        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def generate_global_heatmaps(self):
        """
        Generate heatmaps for gradients computed between original and warped images
        """
        folder_path = os.path.join(self.trial_path, 'gradient_heatmaps')
        os.makedirs(folder_path, exist_ok=True)

        # Compute gradient
        self.gradient_array = self.compute_gradient()
        # Flatten and find global min and max gradients
        all_gradients = self.gradient_array.flatten()
        self.global_min = np.min(all_gradients)
        self.global_max = np.max(all_gradients)

        # Generate heatmaps for each frame
        for i in range(self.image_save_range):  # Assuming gradient_array is (num_frames, H, W)
            heatmap = self.gradient_to_heatmap(self.gradient_array[i])
            filename = os.path.join(folder_path, f"gradient_{i}.png")
            self.save_heatmap_with_colorbar(heatmap, filename)

    def average_heatmaps_with_confidence_intervals(self):
        """
        Generate average heatmap with confidence intervals for gradient between original and warped images.
        
        Args:
            img_array (numpy array): Original image array.
            warped_img_array (numpy array): Warped image array.
        """
        folder_path = os.path.join(self.trial_path, 'average_heatmaps_2D')
        os.makedirs(folder_path, exist_ok=True)

        # Calculate statistics
        absolute_gradients = np.abs(self.gradient_array)
        mean_error_per_pixel = np.mean(absolute_gradients, axis=0)
        std_dev_per_pixel = np.std(absolute_gradients, axis=0)
        mse_error_per_pixel = np.mean(absolute_gradients ** 2, axis=0)
        overall_mse = np.mean(mse_error_per_pixel)
        overall_rmse = np.sqrt(overall_mse)

        # Generate and save heatmap of mean errors
        self.global_min = np.min(mean_error_per_pixel)
        self.global_max = np.max(mean_error_per_pixel)
        heatmap = self.gradient_to_heatmap(mean_error_per_pixel)
        heatmap_filename = os.path.join(folder_path, "mean_error_heatmap.png")
        self.save_heatmap_with_colorbar(heatmap, heatmap_filename)

        # Plot mean error with standard deviation contours
        fig, ax = plt.subplots(figsize=(10, 8))
        img = ax.imshow(mean_error_per_pixel, cmap='coolwarm', vmin=self.global_min, vmax=self.global_max)
        ax.axis('off')

        contour_levels = np.linspace(np.min(std_dev_per_pixel), np.max(std_dev_per_pixel), 10)
        cs = ax.contour(std_dev_per_pixel, levels=contour_levels, colors='black', linewidths=0.5)
        ax.clabel(cs, inline=1, fontsize=8, fmt='%1.2f')

        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(img, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('Gradient Intensity', size=8)

        overlay_filename = os.path.join(folder_path, "mean_error_with_std_contours.png")
        plt.savefig(overlay_filename, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # Write overall statistics to file
        overall_mean_error = np.mean(mean_error_per_pixel)
        overall_std_dev = np.mean(std_dev_per_pixel)
        with open(os.path.join(self.trial_path, "error_metrics.txt"), "w") as f:
            f.write(f"Average Error for the dataset: {overall_mean_error}\n")
            f.write(f"Overall Standard Deviation for the dataset: {overall_std_dev}\n")
            f.write(f"Overall MSE for the dataset: {overall_mse}\n")
            f.write(f"Overall RMSE for the dataset: {overall_rmse}\n")
        

        ########### SSIM ###########
    def normalize_image(self, image):
        """Normalize image to range [0, 1]."""
        return image.astype(np.float32) / image.max()

    def compute_average_metric_value(self, metric_values):
        """
        Compute the average value of a metric (e.g., SSIM, DSSIM, LPIPS).
        
        Args:
            metric_values (list): List of metric values/maps for each frame.
            
        Returns:
            float: The average value of the metric.
        """
        # Handle both scalar values and spatial maps
        if isinstance(metric_values[0], np.ndarray):  # Spatial maps (e.g., LPIPS in spatial mode)
            avg_value = np.mean([np.mean(metric_map) for metric_map in metric_values])
        else:  # Scalar values (e.g., SSIM, DSSIM)
            avg_value = np.mean(metric_values)
        
        return avg_value

    def compute_average_metric_map(self, metric_values):
        """
        Compute the average spatial map for a metric.
        
        Args:
            metric_values (list): List of metric maps for each frame.
            
        Returns:
            numpy array: The average spatial map.
        """
        # Sum all maps and divide by the number of frames to get the average map
        avg_map = np.mean(metric_values, axis=0)
        return avg_map
    
    def compute_metric(self, metric_func, metric_name):
        """
        Generalized function to compute a metric (SSIM, DSSIM, LPIPS, MS-SSIM) for all frames.

        Args:
            metric_func (function): The metric computation function (e.g., SSIM, LPIPS).
            metric_name (str): The name of the metric (used for logging).
        """
        print(f"Computing {metric_name}...")

        # Compute the metric for each frame
        for i in tqdm(range(len(self.original_image_array)), desc=f"Computing {metric_name}", unit="frame"):
            metric_func(i)

    def compute_ssim_map(self, i):
        ssim_value, ssim_map = ssim(
            self.original_image_array[i], 
            self.warped_image_array[i], 
            full=True, 
            data_range=self.warped_image_array[i].max() - self.warped_image_array[i].min()
        )
        self.ssim_maps.append(ssim_map)

    def compute_dssim_map(self, i):
        ssim_value, ssim_map = ssim(
            self.original_image_array[i], 
            self.warped_image_array[i], 
            full=True, 
            data_range=self.warped_image_array[i].max() - self.warped_image_array[i].min()
        )
        dssim_map = (1 - ssim_map) / 2
        self.dssim_maps.append(dssim_map)

    def compute_lpips(self, i):
        """
        Compute LPIPS for a single frame between the original and warped images in spatial mode.

        Args:
            i (int): Index of the current frame.
        """
        # Ensure LPIPS model is initialized only once and in spatial mode
        if not hasattr(self, 'lpips_model'):
            self.lpips_model = lpips.LPIPS(net='alex', spatial=True)  # Enable spatial mode here

        # Handle grayscale (2D) or color (3D) images
        if self.original_image_array[i].ndim == 2:
            # Add batch and channel dimensions for grayscale images
            original_tensor = torch.from_numpy(self.original_image_array[i]).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
            warped_tensor = torch.from_numpy(self.warped_image_array[i]).unsqueeze(0).unsqueeze(0)
        else:
            # Permute dimensions for color images to match LPIPS input (NCHW format)
            original_tensor = torch.from_numpy(self.original_image_array[i]).permute(2, 0, 1).unsqueeze(0)  # Shape: [1, 3, H, W]
            warped_tensor = torch.from_numpy(self.warped_image_array[i]).permute(2, 0, 1).unsqueeze(0)

        # Compute LPIPS in spatial mode
        lpips_map = self.lpips_model(original_tensor, warped_tensor)

        # Convert LPIPS map to a NumPy array after detaching from the computation graph
        self.lpips_maps.append(lpips_map.detach().squeeze().cpu().numpy())  # Ensure detach before converting to NumPy

    def compute_msssim_value(self, i):
        msssim_metric = torchmetrics.functional.multiscale_structural_similarity_index_measure
        original_tensor = torch.from_numpy(self.original_image_array[i]).unsqueeze(0).unsqueeze(0) if self.original_image_array[i].ndim == 2 else torch.from_numpy(self.original_image_array[i]).permute(2, 0, 1).unsqueeze(0)
        warped_tensor = torch.from_numpy(self.warped_image_array[i]).unsqueeze(0).unsqueeze(0) if self.warped_image_array[i].ndim == 2 else torch.from_numpy(self.warped_image_array[i]).permute(2, 0, 1).unsqueeze(0)
        msssim_value = msssim_metric(original_tensor, warped_tensor, data_range=original_tensor.max() - original_tensor.min())
        self.msssim_values.append(msssim_value.item())

    def metric_to_heatmap(self, metric_value, colormap, vmin=None, vmax=None):
        """
        Generalized function to convert a metric value/map to a heatmap, with optional min/max values.
        
        Args:
            metric_value (float or numpy array): The metric value or map.
            colormap (str): The colormap to use.
            vmin (float, optional): Minimum value for color scale. Defaults to None.
            vmax (float, optional): Maximum value for color scale. Defaults to None.

        Returns:
            numpy array: Heatmap of the metric values.
        """
        # Convert metric_value to heatmap form
        heatmap = np.full((self.image_height, self.image_width), metric_value) if isinstance(metric_value, float) else metric_value
        
        # If vmin and vmax are None, compute them from the data
        if vmin is None:
            vmin = np.min(heatmap)
        if vmax is None:
            vmax = np.max(heatmap)

        # Avoid division by zero if vmin and vmax are the same
        if vmin == vmax:
            # If the entire heatmap is a constant value, normalize to 0.5 so the heatmap shows a mid-range color
            norm_heatmap = np.full_like(heatmap, 0.5)
        else:
            # Normalize the heatmap data to range [0, 1] using vmin and vmax
            norm_heatmap = (heatmap - vmin) / (vmax - vmin)

        # Apply colormap to normalized heatmap
        cmap = cm.get_cmap(colormap)
        colored_heatmap = cmap(norm_heatmap)

        # Convert to uint8 format (0-255 scale)
        heatmap_uint8 = (colored_heatmap * 255).astype(np.uint8)

        return heatmap_uint8

    def save_heatmap_with_colorbar(self, heatmap, filename, label, vmin=None, vmax=None):
        """
        Save heatmap with colorbar.

        Args:
            heatmap (numpy array): Heatmap to save.
            filename (str): Path to save the heatmap.
            label (str): Label for the colorbar.
            vmin (float, optional): Minimum value for color scale. Defaults to None.
            vmax (float, optional): Maximum value for color scale. Defaults to None.
        """
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the heatmap with specified min and max values
        img = ax.imshow(heatmap, vmin=vmin, vmax=vmax, cmap='magma')  # You can replace 'magma' with the chosen colormap
        ax.axis('off')  # Remove axis

        # Add colorbar
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(img, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(label, size=8)

        # Save the heatmap to the given filename
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def generate_heatmaps(self, metric_name, metric_values, colormap, label):
        """
        Generalized function to generate heatmaps for a given metric, including average map and value.

        Args:
            metric_name (str): Name of the metric (e.g., 'SSIM', 'DSSIM', 'LPIPS').
            metric_values (list): List of metric values/maps for each frame.
            colormap (str): Colormap to use for the heatmap.
            label (str): Label for the colorbar.
        """
        folder_path = os.path.join(self.trial_path, f'{metric_name.lower()}_heatmaps')
        os.makedirs(folder_path, exist_ok=True)

        # Compute the min and max values across all frames (for dynamic color scaling)
        all_values = np.concatenate([metric_map.flatten() if isinstance(metric_map, np.ndarray) else [metric_map] for metric_map in metric_values])
        vmin, vmax = np.min(all_values), np.max(all_values)

        # Compute and save the average value and map
        avg_value = self.compute_average_metric_value(metric_values)
        avg_map = self.compute_average_metric_map(metric_values)

        print(f"Generating {metric_name} heatmaps for {self.image_save_range} frames...")

        # Save the average map
        avg_heatmap = self.metric_to_heatmap(avg_map, colormap, vmin=vmin, vmax=vmax)
        avg_filename = os.path.join(folder_path, f"{metric_name.lower()}_average_map.png")
        self.save_heatmap_with_colorbar(avg_heatmap, avg_filename, f"Average {label}", vmin=vmin, vmax=vmax)

        # Save the average value in a text file
        avg_value_filename = os.path.join(folder_path, f"{metric_name.lower()}_average_value.txt")
        with open(avg_value_filename, 'w') as f:
            f.write(f"Average {metric_name} value: {avg_value}\n")

        # Generate heatmaps for the specified number of frames
        save_range = min(self.image_save_range, len(metric_values))
        for i in tqdm(range(save_range), desc=f"Generating {metric_name} Heatmaps", unit="frame"):
            heatmap = self.metric_to_heatmap(metric_values[i], colormap, vmin=vmin, vmax=vmax)
            filename = os.path.join(folder_path, f"{metric_name.lower()}_map_{i}.png")
            self.save_heatmap_with_colorbar(heatmap, filename, label, vmin=vmin, vmax=vmax)
########################################################################################
class FlowAnalysis:
    def __init__(self, config, flow_vis_list, binary_mask_list, image_list, x, y):
        self.u_vectors, self.v_vectors, self.flow_vis_images = self.extract_flow_vectors(flow_vis_list)
        self.binary_masks = np.array(binary_mask_list)  # Store the list of binary masks
        self.image_list = np.array(image_list)  # Store the list of images
        self.x, self.y = x, y
        self.mean_u = None
        self.mean_v = None
        self.u_fluctuations = None
        self.v_fluctuations = None
        self.config = config

    def extract_flow_vectors(self, flow_vis_list):
        u_vectors = []
        v_vectors = []
        flow_vis_images = []

        for flow_vis in flow_vis_list:
            flow_vis, u, v = flow_vis  # Assuming flow_vis_list contains tuples with flow vectors
            flow_vis_images.append(flow_vis)
            u_vectors.append(u)
            v_vectors.append(v)
        
        # Crop the first 10 columns of the flow vectors
        u_vectors = np.array(u_vectors)
        v_vectors = np.array(v_vectors)
        
        return u_vectors, v_vectors, flow_vis_images

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

    def save_flow_vectors(self, apply_mask=False, flip=False):
        # Determine directory names based on whether mask is applied
        if apply_mask:
            u_dir = os.path.join(self.config.trial_path, 'UV_plots_masked/u_plots')
            v_dir = os.path.join(self.config.trial_path, 'UV_plots_masked/v_plots')
        else:
            u_dir = os.path.join(self.config.trial_path, 'UV_plots/u_plots')
            v_dir = os.path.join(self.config.trial_path, 'UV_plots/v_plots')

        # Create directories
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
        
        print(f"Global min and max for u vectors: {u_min}, {u_max}")
        print(f"Global min and max for v vectors: {v_min}, {v_max}")

        # Plot and save u and v vectors with consistent axis limits
        for idx, (u, v) in enumerate(zip(self.u_vectors, self.v_vectors)):
            if apply_mask:
                # Apply binary mask if the option is enabled
                binary_mask = self.binary_masks[idx]  # Assuming binary_images is a list of binary masks

                # Mask the u and v vectors using the binary mask
                u_masked = np.where(binary_mask, u, np.nan)
                v_masked = np.where(binary_mask, v, np.nan)
            else:
                # No mask applied, use original vectors
                u_masked = u
                v_masked = v
            
            if flip:
                # Flip the vectors vertically
                u_masked = np.fliplr(u_masked)  # Flip horizontally
                v_masked = np.fliplr(v_masked)

            # Plot u vectors
            fig_u, ax_u = plt.subplots(figsize=(6, 6))
            im_u = ax_u.imshow(u_masked, cmap='jet', aspect='auto', vmin=u_min, vmax=u_max)
            # ax_u.set_title(f'u vectors - Frame {idx}')
            # cbar_u = fig_u.colorbar(im_u, ax=ax_u, orientation='vertical')
            # cbar_u.set_label('Intensity')

            # Save u vector plot in appropriate directory
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(u_dir, f'u_vectors_frame_{idx}.png'))
            plt.close(fig_u)

            # Plot v vectors
            fig_v, ax_v = plt.subplots(figsize=(6, 6))
            im_v = ax_v.imshow(v_masked, cmap='jet', aspect='auto', vmin=v_min, vmax=v_max)
            # ax_v.set_title(f'v vectors - Frame {idx}')
            # cbar_v = fig_v.colorbar(im_v, ax=ax_v, orientation='vertical')
            # cbar_v.set_label('Intensity')

            # Save v vector plot in appropriate directory
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(v_dir, f'v_vectors_frame_{idx}.png'))
            plt.close(fig_v)
    
    @staticmethod
    def detect_edges_sobel(images):
        edges = []
        for image in images:
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Normalize the gradient magnitude to the range [0, 1]
            magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            
            edges.append(magnitude)
        return edges
        
    def plot_and_save_flowmaps(self, plot_type='flow_maps', edge_mask=False, flip=False):
        output_dir = self.config.trial_path

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig, ax = plt.subplots(1, 1, figsize=(18, 6))

        # Detect edges using Sobel filter
        edges_list = self.detect_edges_sobel(self.image_list)

        # Get the minimum number of frames to prevent out-of-bounds errors
        num_frames = min(len(self.flow_vis_images), len(self.image_list), len(edges_list))

        def update_plot(i):
            ax.cla()

            # Access each flow visualization directly from the list
            flow_vis = self.flow_vis_images[i]

            # Use the edges as a mask if edge_mask is True
            if edge_mask:
                edges = edges_list[i]

                # Ensure the edge mask matches the shape of the flow_vis (height and width)
                if edges.shape != flow_vis.shape[:2]:
                    edges_resized = cv2.resize(edges, (flow_vis.shape[1], flow_vis.shape[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    edges_resized = edges

                # Apply the threshold to the Sobel edges
                mask = edges_resized > 0.075 * 255

                # Convert mask to 3 channels to match the flow_vis shape (280x600x3)
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

                # Apply the mask via element-wise multiplication
                flow_vis_masked = flow_vis * mask
            else:
                flow_vis_masked = flow_vis
            
            if flip:
                flow_vis_masked = np.fliplr(flow_vis_masked)
                

            # Plot the masked flow visualization
            if plot_type in ['flow_maps', 'both']:
                ax.imshow(flow_vis_masked)
                ax.axis('off')
                save_path = os.path.join(output_dir, f'flow_maps/flow_maps_{i}.png')
                #if path doesnt exist, create it
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

        # Loop through each frame and update the plot
        for i in range(num_frames):
            update_plot(i)
            plt.cla()  # Clear the current plot for the next frame

        plt.close(fig)

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