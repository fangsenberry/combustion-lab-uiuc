import sys
import os

# Add the utils directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))

from utils.fv_util import FlowInitialization as Fizi
from utils.fv_util import FlowAnalysis as Flay
from utils.fv_util import FlowConfig
import torch
import pandas as pd

# Directory containing the .npy files
# trial_path = r"D:\test_cases\UPF_A01_C_DP_35_trial_12"
# img_path = r"D:\final_corrected_512-complex-27-6-24.pth_inference"
# dir_ext=r'flow_npy\result_'
# step=1
# start_x=0
# end_x=None
# start_y=0
# end_y=None
# reverse_flow=False
# binary_image_analysis=False # Set to True to create binary images
# warp_analysis=False # Set to True to warp the images
# custom_range=25 # set to 'end' to use the entire range or specify a custom range for selection of number of frames

# # # Visualize the flow vectors with color and quiver overlay
# flow_vis_list, img_list, warped_img_list, gradient_list, binary_image_list = Fizi.create_flow_lists(trial_path, img_path, dir_ext, step=step, start_x=start_x, 
#                                                                                                     end_x=end_x, start_y=start_y, end_y=end_y, reverse_flow=reverse_flow, 
#                                                                                                     binary_image=binary_image_analysis, warp=warp_analysis, custom_range=custom_range)

# Flay.extract_flow_vectors(flow_vis_list) #extracts u, v components of velocity

def main():
    # Define your experiment parameters
    config = FlowConfig(
        trial_path=r"D:\test_cases\UPF_A01_C_DP_35_trial_24",
        img_path=r"D:\final_corrected_512-complex-27-6-24.pth_inference",
        dir_ext=r'flow_npy\result_',
        step=1,
        start_x=0,
        end_x=None,
        start_y=0,
        end_y=None,
        reverse_flow=False,
        binary_image_analysis=False,
        warp_analysis=True,
        custom_range=25,
        hdf5_path='flow_data.h5'
    )
    
    # Initialize FlowAnalysis with the configuration
    flow_analysis = Fizi(config)
    
    try:
        # Process the data and save to HDF5 (uncomment if needed)
        # flow_analysis.process_and_save_data()
        
        # Optionally, load the data back from HDF5 (uncomment if needed)
        # loaded_data = Fizi.load_from_hdf5(config.hdf5_path)
        # print(loaded_data.keys())

        # Prepare directories and lists for warped images
        
        # Generate flow lists and save warped images
        flow_vis_list, img_list, warped_img_list, gradient_list, binary_image_list, x, y = flow_analysis.create_flow_lists(
            config.trial_path, 
            config.img_path, 
            config.dir_ext, 
            step=config.step, 
            start_x=config.start_x, 
            warp=config.warp_analysis, 
            binary_image=config.binary_image_analysis,
            custom_range=config.custom_range
        )
        flow_analysis.plot_and_save_losses()
        flow_analysis.generate_global_heatmaps(gradient_list)
        flow_analysis.average_heatmaps_with_confidence_intervals(gradient_list)
        flow_analysis.save_warped_images(warped_img_list)
        flow_extraction=Flay(config, flow_vis_list)
        flow_extraction.save_flow_vectors()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

# base_path = r"D:\test_cases"
# # Define a list to hold all the data
# all_data = []
# missing_trials = [9, 11, 13, 14, 15, 18, 19, 20, 22]
# # Iterate through the trials
# for trial_number in range(1, 25):  # Assuming you have trials numbered 1 to 24
#     if trial_number in missing_trials:
#         print(f"Skipping trial {trial_number} because it is missing.")
#         continue

#     trial_folder = f'UPF_A01_C_DP_35_trial_{trial_number}'
#     trial_path = os.path.join(base_path, trial_folder)

#     if not os.path.exists(trial_path):
#         print(f"Trial {trial_number} directory not found: {trial_path}")
#         continue

#     pth_files = [f for f in os.listdir(trial_path) if f.endswith('.pth')]

#     # Process pth_files as needed
#     print(f"Found {len(pth_files)} .pth files in trial {trial_number}.")
    
#     if pth_files:
#         model_path = os.path.join(trial_path, pth_files[0])  # Assuming there's only one .pth file per folder
        
#         state_dict = torch.load(model_path)
#         training_params = state_dict['training_params']
#         param_dict = state_dict['model_params']

#         # Combine both dictionaries
#         combined_params = {**training_params, **param_dict}
        
#         # Add trial number to each parameter for identification
#         combined_params['Trial'] = f"trial_{trial_number}"
        
#         # Append the dictionary to the list
#         all_data.append(combined_params)

# # Convert the list of dictionaries to a DataFrame
# df = pd.DataFrame(all_data)

# # Save the DataFrame to a CSV file
# csv_path = os.path.join(base_path, "all_parameters1.csv")
# df.to_csv(csv_path, index=False)

# print(f'All parameters saved to {csv_path}')