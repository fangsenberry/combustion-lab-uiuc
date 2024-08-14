from utils import fv_util as FVu
# Directory containing the .npy files
result_path = r"D:\test_cases\UPF_A01_C_DP_35_trial_12\flow_npy"

img_path = r"D:\final_corrected_512-complex-27-6-24.pth_inference"
base='result_'

# Visualize the flow vectors with color and quiver overlay
flow_vis_list, img_list, warped_img_list, gradient_list, binary_image_list = FVu.load_and_visualize_flows(result_path, img_path, base, step=2, start_x=0, end_x=None, start_y=0, end_y=None, reverse_flow=False, binary_image=True, warp=True, custom_range=20)
# plot_flow_and_colorwheel(flow_vis_list[0][0])
# # Call the function to plot flow vectors