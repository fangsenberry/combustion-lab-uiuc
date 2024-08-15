import sys
import os

# Add the utils directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))

from utils.fv_util import FlowInitialization as Fizi
from utils.fv_util import FlowAnalysis as Flay
from utils.fv_util import FlowConfig


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
        trial_path=r"D:\test_cases\UPF_A01_C_DP_35_trial_12",
        img_path=r"D:\final_corrected_512-complex-27-6-24.pth_inference",
        dir_ext=r'flow_npy\result_',
        step=1,
        start_x=0,
        end_x=None,
        start_y=0,
        end_y=None,
        reverse_flow=False,
        binary_image_analysis=False,
        warp_analysis=False,
        custom_range=25,
        hdf5_path='flow_data.h5'
    )
    
    # Initialize FlowAnalysis with the configuration
    flow_analysis = Fizi(config)
    
    # Process the data and save to HDF5
    flow_analysis.process_and_save_data()
    
    # Optionally, load the data back from HDF5
    loaded_data = Fizi.load_from_hdf5(config.hdf5_path)
    flow_vis_list = loaded_data['flow_vis_list']
    img_list = loaded_data['img_list']
    warped_img_list = loaded_data['warped_img_list']
    gradient_list = loaded_data['gradient_list']
    binary_image_list = loaded_data['binary_image_list']
    
    # Example usage of the loaded data
    print("Loaded data successfully:")
    print(f"Number of flow visualizations: {len(flow_vis_list)}")
    
    # Additional operations can be performed using flow_analysis methods
    # For example, plotting fluctuating components
    Flay.plot_fluctuating_components(time_step=0)

if __name__ == "__main__":
    main()