import sys
import os

# Add the utils directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))

from utils.fv_util import FlowInitialization as Fizi
from utils.fv_util import FlowAnalysis as Flay
from utils.fv_util import FlowConfig
import torch
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from statsmodels.graphics.tsaplots import plot_acf
from scipy.fftpack import fft, ifft, fftfreq
from scipy.optimize import curve_fit
import random
from scipy.spatial import distance
from collections import deque
import time
import breakup_util as bu

from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin

def apply_pod(u_list, v_list):
    snapshots = []

    # Loop over all snapshots and flatten the u and v fields
    for u, v in zip(u_list, v_list):
        u_flattened = u.flatten()  # (280 * 600 = 168000)
        v_flattened = v.flatten()  # (280 * 600 = 168000)
        velocity_snapshot = np.concatenate([u_flattened, v_flattened])  # Concatenate (336000,)
        snapshots.append(velocity_snapshot)

    # Stack all snapshots into a data matrix (336000 spatial points per snapshot, 3791 snapshots)
    X = np.array(snapshots).T  # Transpose to make each column a snapshot (336000, 3791)

    # Subtract the mean flow
    X_mean = np.nanmean(X, axis=1, keepdims=True)
    X_prime = X - X_mean  # Subtract the mean flow

    # Snapshot method: Calculate covariance matrix in smaller space
    C = np.dot(X_prime.T, X_prime) / X_prime.shape[1]  # Shape (3791, 3791)

    # Perform eigenvalue decomposition on the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # Sort by descending eigenvalue magnitude
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute the POD modes (project back to the original space)
    POD_modes = np.dot(X_prime, eigenvectors)

    # Optionally reconstruct the flow using k dominant modes
    k = 5  # Number of modes to keep
    reconstructed_field = np.dot(POD_modes[:, :k], eigenvectors[:, :k].T) + X_mean
    print(f"POD_modes[:, :k] shape: {POD_modes[:, :k].shape}")  # Should be (336000, k)
    print(f"eigenvectors[:, :k] shape: {eigenvectors[:, :k].shape}")  # Should be (3791, k)

    # Perform the reconstruction
    reconstructed_field = np.dot(POD_modes[:, :k], eigenvectors[:, :k].T) + X_mean
    print(f"reconstructed_field shape: {reconstructed_field.shape}")  # Should be (336000, 3791)

    return POD_modes, reconstructed_field, eigenvalues

def visualize_pod_modes(POD_modes, u_shape, v_shape, mode_indices=[0, 1, 2]):
    """ Visualizes the selected POD modes by reshaping them into 2D arrays """
    num_modes = len(mode_indices)

    fig, axs = plt.subplots(2, num_modes, figsize=(15, 6))

    for i, mode_idx in enumerate(mode_indices):
        # Reshape u and v components of the POD mode
        u_mode = POD_modes[:np.prod(u_shape), mode_idx].reshape(u_shape)
        v_mode = POD_modes[np.prod(u_shape):, mode_idx].reshape(v_shape)

        # Plot u-mode
        axs[0, i].imshow(u_mode, cmap='jet', origin='lower')
        axs[0, i].set_title(f'POD Mode {mode_idx + 1} (u-component)')
        axs[0, i].axis('off')

        # Plot v-mode
        axs[1, i].imshow(v_mode, cmap='jet', origin='lower')
        axs[1, i].set_title(f'POD Mode {mode_idx + 1} (v-component)')
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_reconstructed_field(reconstructed_field, u_shape, v_shape, time_step=0):
    """ Visualizes the reconstructed velocity field for a single time step """
    
    # Extract the data for the specified time step (single column from reconstructed_field)
    single_time_step = reconstructed_field[:, time_step]
    
    # Reshape u and v components of the reconstructed field
    u_reconstructed = single_time_step[:np.prod(u_shape)].reshape(u_shape)
    v_reconstructed = single_time_step[np.prod(u_shape):].reshape(v_shape)

    # Plot the reconstructed u and v fields
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot u-component
    axs[0].imshow(u_reconstructed, cmap='jet', origin='lower')
    axs[0].set_title(f'Reconstructed u-component (time step {time_step})')
    axs[0].axis('off')

    # Plot v-component
    axs[1].imshow(v_reconstructed, cmap='jet', origin='lower')
    axs[1].set_title(f'Reconstructed v-component (time step {time_step})')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

#############################################################################################
def main():
    # Define your experiment parameters
    config = FlowConfig(
        trial_path=r"D:\test_cases\UPF_A01_C_DP_35_trial_35",
        img_path=r"D:\final_corrected_512-complex-27-6-24.pth_inference",
        dir_ext=r'flow_npy\result_',
        step=1,
        start_x=0,
        end_x=None,
        start_y=0,
        end_y=None,
        reverse_flow=False,
        binary_image_analysis=True,
        warp_analysis=True,
        custom_range='end',
        hdf5_path='flow_data.h5'
    )
    
    # Initialize FlowAnalysis with the configuration
    flow_analysis = Fizi(config)
    
    try:
        # Process the data and save to HDF5 (uncomment if needed)
        # flow_analysis.process_and_save_data()
        
        # Optionally, load the data back from HDF5 (uncomment if needed)
        loaded_data = Fizi.load_from_hdf5(config.hdf5_path)
        print(f"Loaded data with keys: {loaded_data.keys()}")
        
        # Generate flow lists and save warped images
        # flow_vis_list, img_list, warped_img_list, gradient_list, binary_image_list, x, y = flow_analysis.create_flow_lists(
        #     config.trial_path, 
        #     config.img_path, 
        #     config.dir_ext, 
        #     step=config.step, 
        #     start_x=config.start_x, 
        #     warp=config.warp_analysis, 
        #     binary_image=config.binary_image_analysis,
        #     custom_range=config.custom_range, flow_vis_type='basic'
        # )
        
        # flow_analysis.plot_and_save_losses()
        # flow_analysis.generate_global_heatmaps(gradient_list)
        # flow_analysis.average_heatmaps_with_confidence_intervals(gradient_list)
        # flow_analysis.save_warped_images(warped_img_list)
        # print('done with initial part')
        # flow_extraction=Flay(config, flow_vis_list, binary_image_list, image_list=img_list, x=x, y=y)
        # flow_extraction.save_flow_vectors(apply_mask=True, flip=True)
        # flow_extraction.plot_vorticity(save_dir=config.trial_path, save_data='vorticity')
        # flow_extraction.plot_and_save_flowmaps(edge_mask=True, flip=True)
        # # Plot shear stress
        # flow_extraction.plot_shear_stress(save_dir=config.trial_path, save_data='shear_stress')

        # # Plot strain rate
        # flow_extraction.plot_strain_rate(save_dir=config.trial_path, save_data='strain_rate')

        ################################## CDF code ##################################
        # print('created lists')
        # binary_image_list = loaded_data['binary_image_list'] 
        # print(binary_image_list[0].max())
        # img_list = loaded_data['img_list']
        # u_vectors = loaded_data['u_vectors']
        # v_vectors = loaded_data['v_vectors']
        # print(u_vectors.shape)
        # mask = binary_image_list == 255  # Create a mask where white regions are True
        # u_filtered = np.where(mask, u_vectors, np.nan)  # Set non-white regions to NaN or 0 if preferred
        # v_filtered = np.where(mask, v_vectors, np.nan)
        # num_columns_to_plot = 5
        # num_rows_to_plot = 5
        # plot_cdf_velocity_all_frames(u_filtered, v_filtered, num_columns_to_plot, axis='y-slices')
        # plot_cdf_velocity_all_frames(u_filtered, v_filtered, num_rows_to_plot, axis='x-slices')
        ################################## start of breakup length ##################################
        binary_image_list = loaded_data['binary_image_list']
        u=loaded_data['u_vectors']
        v=loaded_data['v_vectors']
        # Step 1: Average u and v along the first axis
        u_avg = np.mean(u, axis=(0, 2))  # Average along the first two axes
        v_avg = np.mean(v, axis=(0, 2))  # Average along the first two axes

        print(f"u_avg shape: {u_avg.shape}, v_avg shape: {v_avg.shape}")

        plt.plot(u_avg, label='Average U Velocity')
        plt.plot(v_avg, label='Average V Velocity')
        plt.xlabel('Frame Index')
        plt.ylabel('Velocity')
        
        plt.legend()
        plt.show()

        mask = (binary_image_list == 255)

        # Step 1: Apply the mask and average u and v in the binarized regions
        u_masked = np.where(mask, u, np.nan)  # Set non-binarized regions to NaN
        v_masked = np.where(mask, v, np.nan)  # Set non-binarized regions to NaN
        # POD_modes, reconstructed_field, eigenvalues = apply_pod(u, v)

        # # Ensure that u_shape and v_shape correspond to the original dimensions of your fields
        # u_shape = (280, 600)
        # v_shape = (280, 600)

        # # Visualize the first three POD modes
        # visualize_pod_modes(POD_modes, u_shape, v_shape, mode_indices=[0, 1, 2])

        # # Visualize the reconstructed velocity field (using the first k modes)
        # visualize_reconstructed_field(reconstructed_field, u_shape, v_shape, time_step=0)  # First time step

        # Step 2: Compute the mean only in the binarized regions (ignoring NaNs)
        u_avg_binarized = np.nanmean(u_masked, axis=(0, 2))  # Average along the first two axes
        v_avg_binarized = np.nanmean(v_masked, axis=(0, 2))  # Average along the first two axes

        # Step 3: Plot the results
        plt.plot(u_avg_binarized, label='Average U Velocity (Binarized Regions)')
        plt.plot(v_avg_binarized, label='Average V Velocity (Binarized Regions)')
        plt.xlabel('Frame Index')
        plt.ylabel('Velocity')
        plt.legend()
        plt.show()

        #save binary_image_list[0] as a png
        # cv2.imwrite('binary_image_ex.png',binary_image_list[0])
        image_list=loaded_data['img_list']
        raw_x_y_values, smoothed_x_y_values, frame_idx_breakup, best_segment_list = bu.process_breakup_length(binary_image_list=binary_image_list, breakup_x_threshold=50)
        print('done with breakup length')
        #print raw_x_y_values and smoothed_x_y_values mean
        print('raw_x_y_values:',np.mean(raw_x_y_values))
        print('smoothed_x_y_values:',np.mean(smoothed_x_y_values))
        # Visualize hole propagation for each binary image
        porosity_values = []
        fuel_density= []
        print(f'Length of binary_image_list: {len(binary_image_list)}')
        print(f'Length of smoothed_x_y_values: {len(smoothed_x_y_values)}')
        # Visualize hole propagation and calculate porosity for each binary image
        for frame_idx, binary_image in enumerate(binary_image_list):
            if frame_idx < len(smoothed_x_y_values):  # Bounds check to prevent out-of-bounds error
                porosity,_ = bu.calculate_hole_porosity_with_visualization(binary_image, smoothed_x_y_values[frame_idx])
                porosity_values.append(porosity)  # Store the porosity for each frame
                fuel_d=bu.calculate_fuel_density_entire_image(binary_image)
                fuel_density.append(fuel_d)
                breakup_x = int(smoothed_x_y_values[0])  # Convert to integer if it's a float
            else:
                print(f"Skipping frame {frame_idx} as it's out of bounds for smoothed_x_y_values.")


        

        # # find the average of the porosity values
        # print('porosity:',np.mean(porosity_values))
        # # # Plot porosity over time (frame index)
        # plt.figure(figsize=(10, 6))
        # plt.plot(range(len(porosity_values)), porosity_values, marker='o', linestyle='-', color='b')
        # plt.xlabel('Frame Index (Time)')
        # plt.ylabel('Hole Porosity')
        # plt.title('Hole Porosity Over Time')
        # plt.grid(True)
        # plt.show()

        # plt.figure(figsize=(10, 6))
        # plt.plot(range(len(fuel_density)), fuel_density, marker='o', linestyle='-', color='g')
        # plt.xlabel('Frame Index (Time)')
        # plt.ylabel('Fuel Density')
        # plt.title('Fuel Density Over Time')
        # plt.grid(True)
        # plt.show()

        # num_slices = 5  # Number of rows/columns to calculate fuel density for
        # axis = 'y-slices'  # You can change to 'x-slices' if needed
        # axis2='x-slices'

        # # Plot the CDF for fuel density across all frames
        # x_regions = 5  # Specify how many regions you want along the x-axis (width)
        # y_regions = 5  # Specify how many regions you want along the y-axis (height)

        # # Plot the CDFs for the x and y regions
        # plot_cdf_fuel_density_all_regions(binary_image_list, x_regions, y_regions)


        ############################ average Binary Image ########################################
        # y = bu.plot_average_image(image_list=binary_image_list)
        # y = np.mean(y, axis=0)

        # # Analyze the curve with zero crossings (assuming this is a pre-processing step)
        # a = bu.analyze_curve_with_zero_crossings(y, degree=5)

       
        # x = np.arange(len(y))
        # filter= (x>20) & (x<580)
        # x=x[filter]
        # y=y[filter]
        # x=x.reshape(-1,1)
        # # Define the custom sigmoid model
        # sigmoid_model = bu.SigmoidModel()

        # # Create a RANSAC regressor with `min_samples` set and fit the data
        # ransac = RANSACRegressor(estimator=sigmoid_model, residual_threshold=4.0, max_trials=1000, min_samples=0.9)
        # ransac.fit(x, y)
        # # Predict the inliers using the RANSAC-fitted model
        # y_fitted_ransac = ransac.predict(x)

        # inlier_mask = ransac.inlier_mask_

        # # Force the sigmoid model to fit only the inliers
        # x_inliers = x[inlier_mask].flatten()  # Get the inlier x values
        # y_inliers = y[inlier_mask]  # Get the inlier y values

        # # Refitting the sigmoid model using the inliers
        # sigmoid_model.fit(x_inliers, y_inliers)

        # # Directly access sigmoid_model.params after refitting to the inliers
        # if sigmoid_model.params is not None:
        #     # Retrieve the fitted sigmoid parameters, including x0
        #     ymin_fitted, ymax_fitted, k_fitted, x0_fitted = sigmoid_model.params

        #     # Print the final fit parameters
        #     print(f"Final fitted parameters (from inliers):")
        #     print(f"ymin: {ymin_fitted}, ymax: {ymax_fitted}, k: {k_fitted}, x0 (inflection point): {x0_fitted}")
        # else:
        #     print("Fitting failed in the underlying model. No parameters were returned.")

        # # Plotting the original data points
        # plt.figure(figsize=(10, 8))

        # # Plot the fitted generalized sigmoid curve using RANSAC
        # plt.plot(x, y_fitted_ransac, 'r-', label='Fitted Generalized Sigmoid (RANSAC)')  # Plot fitted sigmoid

        # # Plot original data
        # plt.plot(x, y, 'bo', label='Original Data')  # Original data

        # # Plot the vertical dashed line at x0
        # plt.axvline(x=x0_fitted, color='k', linestyle='--', label=f'$x_0 = {int(x0_fitted)}$')  # Dashed line for x0

        # # Add labels and legend
        # plt.xlabel('Image Width', fontsize=18)
        # plt.ylabel('Image Intensity', fontsize=18)
        # plt.legend(fontsize=18)
        # plt.tick_params(axis='both', which='major', labelsize=16)

        # # Show plot
        # plt.show()



        
        
        # Smooth the x-values using a running average
        window_size = 100  # Window size for the running average
        # smoothed_x_y_values = running_average(smoothed_x_y_values, window_size=window_size)
        x_values = np.arange(window_size - 1, len(smoothed_x_y_values) + (window_size - 1))  # Correct the x-values

        # Adjust frame_idx_breakup to exclude values that are now out of bounds after smoothing
        # frame_idx_breakup = frame_idx_breakup[frame_idx_breakup < len(smoothed_x_y_values) + (window_size - 1)]
        def fourier_series(x, *a):
            result = np.zeros_like(x)
            N = len(a) // 2
            for n in range(N):
                result += a[2*n] * np.sin((n + 1) * 2 * np.pi * x / len(x)) + a[2*n + 1] * np.cos((n + 1) * 2 * np.pi * x / len(x))
            return result

        smoothed_x_y_values = np.array(smoothed_x_y_values)
        # smoothed_x_y_values= width-smoothed_x_y_values
        smoothed_x_y_values = 9.3*smoothed_x_y_values*10**-3

        # Step 1: Subtract the mean from the data
        data_mean = np.mean(smoothed_x_y_values)
        mean_adjusted_data = smoothed_x_y_values - data_mean

        # Generate x-values for the mean-adjusted data
        x_values = np.arange(len(mean_adjusted_data))

        # Initial guess for the Fourier coefficients (up to 3 harmonics)
        initial_guess = [1] * 12  # For a Fourier series with 3 terms (sin and cos)

        # Step 2: Perform the curve fitting to fit the Fourier series to your mean-adjusted data
        params, params_covariance = curve_fit(fourier_series, x_values, mean_adjusted_data, p0=initial_guess)

        # Step 3: Reconstruct the continuous wave using the fitted Fourier series
        x_dense = np.linspace(x_values.min(), x_values.max(), 1000)  # Higher density of x-values for smooth wave
        fitted_wave = fourier_series(x_dense, *params)

        # Step 4: Add the mean back to the fitted wave
        fitted_wave_with_mean = fitted_wave + data_mean

        ######################## this section above was all used previously ############################
        # Plot the original smoothed data and the fitted Fourier series wave
        # plt.figure(figsize=(10, 6))
        # plt.plot(x_values, smoothed_x_y_values, linestyle='-', color='blue')  # Original smoothed data
        # plt.axhline(y=data_mean, color='red', linestyle='--', label=f'Mean breakup length: {data_mean:.2f} mm')  # Mean value
        # # plt.plot(x_dense, fitted_wave_with_mean, label='Fitted Fourier Series', linestyle='-', color='green')  # Fitted Fourier series

        # # Add titles and labels
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.xlabel('Time [s]', fontsize=18)
        # plt.ylabel('Breakup Length [mm]', fontsize=18)
        # plt.legend(fontsize=16)
        # plt.tight_layout()
        # # plt.grid(True)
        # plt.show()
        

        # print('fourier_params:', params)


        # # Create x-values representing the frame indices
        # x_values = np.arange(len(smoothed_x_y_values))  # Assuming frame indices are sequential

        # # Create a cubic interpolation function
        # interp_func = interp1d(x_values, smoothed_x_y_values, kind='cubic')

        # # Generate new x-values for a denser, smoother curve
        # x_dense = np.linspace(x_values.min(), x_values.max(), num=10000)  # 1000 points for a smooth curve

        # # Generate new smoothed y-values using the interpolation function
        # smoothed_x_y_dense = interp_func(x_dense)

        # # Plot the original smoothed x-values and the denser interpolated curve
        # plt.figure(figsize=(10, 6))

        # # # Plot the original raw x-values (before modification)
        # # plt.plot(x_values, raw_x_y_values, color='red', label='Raw X Value Over Time')

        # # Plot the interpolated, denser smoothed curve
        # plt.plot(x_values, smoothed_x_y_values, color='green', linestyle='-', label='Interpolated Smoothed X Value Over Time')

        # # Scatter plot of the breakup frame points (ensure indices are valid)
        # plt.scatter(frame_idx_breakup, smoothed_x_y_values[frame_idx_breakup], color='black', label='Breakup Frame')

        # plt.xlabel('Frame Index')
        # plt.ylabel('Average X Value')
        # plt.title('Raw vs Interpolated Smoothed X Values Over Time')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # dx = np.gradient(smoothed_x_y_values)  # First derivative (rate of change)
        # d2x = np.gradient(dx)  # Second derivative (acceleration)

        # # Create x-values representing the frame indices (for smoothed dense data)
        # x_dense = np.linspace(0, len(smoothed_x_y_values), num=len(smoothed_x_y_values))

        # # Plot the original smoothed x-values
        # plt.figure(figsize=(10, 6))
        # plt.plot(x_dense, smoothed_x_y_values, color='green', label='Smoothed X Value')

        # # Plot the first derivative (rate of change)
        # plt.plot(x_dense, dx, color='blue', label='First Derivative (Rate of Change)')

        # # Plot the second derivative (acceleration)
        # plt.plot(x_dense, d2x, color='orange', label='Second Derivative (Acceleration)')

        # plt.xlabel('Frame Index')
        # plt.ylabel('X Value / Derivative')
        # plt.title('X Values and Their Derivatives Over Time')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        breakup_length_data = np.array(smoothed_x_y_values)

        # # Compute and plot autocorrelation
        # plt.figure(figsize=(10, 6))
        # plot_acf(breakup_length_data, lags=100, title="Autocorrelation of Breakup Length")
        # plt.xlabel('Lag (Frame Index)')
        # plt.ylabel('Autocorrelation')
        # plt.show()
        fft_result = np.fft.fft(breakup_length_data)
        fft_magnitude = np.abs(fft_result)  # Get the magnitude of the Fourier coefficients
        fft_freq = np.fft.fftfreq(len(breakup_length_data))  # Compute the frequency bins

        data_demeaned = breakup_length_data - np.mean(breakup_length_data)

        # Perform Fourier Transform
        fft_result = np.fft.fft(data_demeaned)
        fft_magnitude = np.abs(fft_result)  # Get the magnitude of the Fourier coefficients
        fft_freq = np.fft.fftfreq(len(breakup_length_data))  # Compute the frequency bins

        # Plot the magnitude of the Fourier transform
        # plt.figure(figsize=(10, 6))
        # plt.plot(fft_freq, fft_magnitude)
        # plt.title('Fourier Transform of Demeaned Breakup Length Data')
        # plt.xlabel('Frequency')
        # plt.ylabel('Magnitude')
        # plt.grid(True)
        # plt.show()

        # import pywt
        # wavelet = 'mexh'

        # # Define the scales for the wavelet transform
        # scales = np.arange(1, 128)

        # # Perform the continuous wavelet transform (CWT)
        # coefficients, frequencies = pywt.cwt(smoothed_x_y_values, scales, wavelet)

        # # Plot the CWT coefficients as a heatmap to analyze time-frequency features
        # plt.figure(figsize=(10, 6))
        # plt.imshow(coefficients, extent=[0, len(smoothed_x_y_values), 1, 128], cmap='PRGn', aspect='auto',
        #         vmax=abs(coefficients).max(), vmin=-abs(coefficients).max())
        # plt.title("Continuous Wavelet Transform (CWT) - Mexican Hat Wavelet")
        # plt.ylabel("Scale (Related to Frequency)")
        # plt.xlabel("Time (Frame Index)")
        # plt.colorbar(label="Magnitude")
        # plt.show()

        # frame_window = 100
        # num_frames = len(smoothed_x_y_values)

        # # Remove duplicate frame indices in frame_idx_breakup
        # frame_idx_breakup = np.array(frame_idx_breakup, dtype=int)
        # frame_idx_breakup = np.unique(frame_idx_breakup)

        # # Loop through and plot frames in 500-frame chunks
        # for start_idx in range(0, num_frames, frame_window):
        #     end_idx = min(start_idx + frame_window, num_frames)  # Ensure we don't exceed the number of frames

        #     # Extract data for the current window
        #     x_values_window = x_values[start_idx:end_idx]
        #     raw_x_y_values_window = raw_x_y_values[start_idx:end_idx]
        #     smoothed_x_y_values_window = smoothed_x_y_values[start_idx:end_idx]
            
        #     # Get frame indices for the breakup points that are within the current window
        #     frame_idx_breakup_window = frame_idx_breakup[(frame_idx_breakup >= start_idx) & (frame_idx_breakup < end_idx)]
            
        #     # Adjust frame_idx_breakup_window to relative index within this window
        #     frame_idx_breakup_window = frame_idx_breakup_window - start_idx
            
        #     # Modify values at breakup points by averaging previous and next points within this window
        #     for idx in frame_idx_breakup_window:
        #         # Ensure we are not out of bounds (idx must be between 1 and len(smoothed_x_y_values_window) - 2)
        #         if 1 <= idx < len(smoothed_x_y_values_window) - 1:
        #             avg_value = (smoothed_x_y_values_window[idx - 1] + smoothed_x_y_values_window[idx + 1]) / 2
        #             smoothed_x_y_values_window[idx] = avg_value

        #     # Generate new x-values for a denser, smoother curve within this window
        #     x_dense_window = np.linspace(x_values_window.min(), x_values_window.max(), num=1000)

        #     # Create a cubic interpolation function for this window
        #     interp_func = interp1d(x_values_window, smoothed_x_y_values_window, kind='cubic')

        #     # Generate new smoothed y-values using the interpolation function
        #     smoothed_x_y_dense_window = interp_func(x_dense_window)

        #     # Plot the original smoothed x-values and the denser interpolated curve for this window
        #     plt.figure(figsize=(10, 6))

        #     # Plot the original raw x-values (before modification) for this window
        #     plt.plot(x_values_window, raw_x_y_values_window, color='red', label='Raw X Value Over Time')

        #     # Plot the interpolated, denser smoothed curve for this window
        #     plt.plot(x_dense_window, smoothed_x_y_dense_window, color='green', linestyle='-', label='Interpolated Smoothed X Value Over Time')

        #     # Scatter plot of the breakup frame points (only within this window)
        #     plt.scatter(frame_idx_breakup_window, smoothed_x_y_values_window[frame_idx_breakup_window], color='black', label='Breakup Frame')

        #     plt.xlabel('Frame Index')
        #     plt.ylabel('Average X Value')
        #     plt.title(f'Raw vs Interpolated Smoothed X Values: Frames {start_idx} to {end_idx}')
        #     plt.legend()
        #     plt.grid(True)
        #     plt.show()



        # # plot the avg x values over all the best segments over time
        # # avg_x_values = [np.mean(segment[0]) for segment in best_segment_list]
        # x_values = np.arange(len(avg_x_y_values))  # Assuming frame indices are sequential

        # # Create a cubic interpolation function
        # interp_func = interp1d(x_values, avg_x_y_values, kind='cubic')

        # # Generate new x-values for a denser, smoother curve
        # x_dense = np.linspace(x_values.min(), x_values.max(), num=1000)  # 1000 points for a smooth curve

        # # Generate new smoothed y-values using the interpolation function
        # smoothed_avg_x_values = interp_func(x_dense)

        # # Plot the interpolated, smoother curve
        # plt.figure(figsize=(10, 6))
        # plt.plot(x_dense, smoothed_avg_x_values, color='blue', label='Smoothed Average X Value Over Time')
        # plt.xlabel('Frame Index')
        # plt.ylabel('Average X Value')
        # plt.title('Smoothed Average X Value Over Time (Cubic Interpolation)')
        # plt.legend()
        # plt.grid(True)
        # plt.show()


    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()


############################### Consolidate Error Metrics ########################################
# base_path = r"D:\test_cases"
# # Define a list to hold all the data
# all_data = []
# missing_trials = [9, 11, 13, 14, 15, 18, 19, 20, 22]


def consolidate_error_metrics(base_path, output_filename, missing_trials=None,):
    """
    Consolidates error metrics from multiple trial directories into a single CSV file.

    Parameters:
    - base_path (str): The base directory where trial folders are stored.
    - missing_trials (list): List of trial numbers to skip because they are missing.
    - output_filename (str): The name of the output CSV file to save the consolidated metrics.

    Returns:
    - None
    """
    
    if missing_trials is None:
        missing_trials=[]
    # Define a list to hold all the data
    all_data = []

    # Iterate through the trials
    for trial_number in range(1, 29):  # Assuming you have trials numbered 1 to 24
        if trial_number in missing_trials:
            print(f"Skipping trial {trial_number} because it is missing.")
            continue

        trial_folder = f'UPF_A01_C_DP_35_trial_{trial_number}'
        trial_path = os.path.join(base_path, trial_folder)

        if not os.path.exists(trial_path):
            print(f"Trial {trial_number} directory not found: {trial_path}")
            continue

        # Look for the error_metrics.txt file
        error_metrics_path = os.path.join(trial_path, 'error_metrics.txt')
        
        if os.path.exists(error_metrics_path):
            print(f"Processing error metrics for trial {trial_number}.")
            
            # Read the error metrics file
            with open(error_metrics_path, 'r') as f:
                metrics = f.read().strip().split('\n')
            
            # Parse the metrics into a dictionary
            metrics_dict = {}
            for metric in metrics:
                key, value = metric.split(':')
                metrics_dict[key.strip()] = float(value.strip())
            
            # Add trial number to the dictionary for identification
            metrics_dict['Trial'] = f"trial_{trial_number}"
            
            # Append the dictionary to the list
            all_data.append(metrics_dict)
        else:
            print(f"No error_metrics.txt found for trial {trial_number}.")
            
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(all_data)

    # Save the DataFrame to a CSV file
    csv_path = os.path.join(base_path, output_filename)
    df.to_csv(csv_path, index=False)

    print(f'All error metrics saved to {csv_path}')