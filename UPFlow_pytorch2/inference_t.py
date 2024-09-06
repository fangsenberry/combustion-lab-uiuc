import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model.upflow import UPFlow_net
from utils.tools import tools
from implement2 import CustomFlowDataset, Trainer, Config
from utils.tools import tools
import sys
import logging
from datetime import datetime

# Add the utils directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))

from utils.fv_util import FlowInitialization as Fizi
from utils.fv_util import FlowAnalysis as Flay
from utils.fv_util import FlowConfig

def suppress_output():
    sys.stdout = open(os.devnull, 'w')

def restore_output():
    sys.stdout = sys.__stdout__
# Get the current date
current_date = datetime.now().strftime('%Y-%m-%d')

# Define the log file name with the current date
log_file = f'D:\\test_cases\\processing_log_{current_date}.txt'

# Set up logging to write to the log file
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

def inference(model_path, dataset_path, result_path):
    try:
        # Load the model state dictionary
        suppress_output()
        state_dict = torch.load(model_path)
        training_params = state_dict['training_params']
        param_dict = state_dict['model_params']

        # Initialize Config with the loaded training parameters
        config = Config(**training_params)
        
        # Initialize Trainer with the loaded configuration and parameter dictionary
        trainer = Trainer(config, param_dict=param_dict)
        
        # Load the model for inference
        model = trainer.load_model(model_path=model_path, mode='inference')

        # Move the model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Load dataset
        dataset = CustomFlowDataset(dataset_path, transform=transforms.Compose([transforms.ToTensor()]), target_mean=0.5, crop_size=None, num_crops_per_image=1)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        restore_output()

        # Inference
        for i, (img1, img2) in enumerate(dataloader):
            img1, img2 = img1.to(device), img2.to(device)
            with torch.no_grad():
                start = torch.zeros((img1.size(0), 2, 1, 1), device=device)
                input_dict = {
                    'im1': img1,
                    'im2': img2,
                    'im1_raw': img1,
                    'im2_raw': img2,
                    'start': start,
                    'if_loss': False
                }
                output_dict = model(input_dict)
                flow_fw = output_dict['flow_b_out'].cpu().numpy()
                np.save(f'{result_path}/result_{i}.npy', flow_fw)

    except RuntimeError as e:
        restore_output()
        logging.error(f"Skipping trial {model_path} due to error: {e}")
        raise  # Re-raise the exception to stop further processing

def process_test_cases(base_path, specific_trial_numbers=None):
    # If specific trial numbers are provided, only process those
    if specific_trial_numbers is not None:
        trial_dirs = []
        for trial_dir in os.listdir(base_path):
            try:
                trial_number = int(trial_dir.split('_')[-1])
                if trial_number in specific_trial_numbers:
                    trial_dirs.append(os.path.join(base_path, trial_dir))
            except ValueError:
                # Skip any directories or files that don't end in a valid number
                logging.info(f"Skipping directory or file {trial_dir} as it does not have a valid trial number.")
                continue
    else:
        # Otherwise, process all directories in the base path
        trial_dirs = [os.path.join(base_path, trial_dir) for trial_dir in os.listdir(base_path)]

    for trial_path in trial_dirs:
        trial_dir = os.path.basename(trial_path)

        try:
            trial_number = int(trial_dir.split('_')[-1])  # Assuming the trial number is at the end of the directory name
        except ValueError:
            logging.info(f"Skipping directory {trial_dir} as it does not have a valid trial number.")
            continue
        
        if os.path.isdir(trial_path):
            logging.info(f"Processing {trial_path}...")

            # Set paths for model, dataset, and result
            model_path = os.path.join(trial_path, f"{trial_dir}.pth")
            if not os.path.exists(model_path):
                logging.info(f'Model path does not exist for {trial_path}')
                continue

            dataset_path = r"D:\final_corrected_512-complex-27-6-24.pth_inference"
            result_path = os.path.join(trial_path, 'flow_npy')
            os.makedirs(result_path, exist_ok=True)
            
            try:
                # Run inference
                inference(model_path, dataset_path, result_path)
                logging.info(f'Finished inference for {trial_path}')
                
                # Define your experiment parameters for analysis with custom_range=35
                config_35 = FlowConfig(
                    trial_path=trial_path,
                    img_path=dataset_path,
                    dir_ext=r'flow_npy/result_',
                    step=1,
                    start_x=0,
                    end_x=None,
                    start_y=0,
                    end_y=None,
                    reverse_flow=False,
                    binary_image_analysis=False,
                    warp_analysis=True,
                    custom_range=35,  # Set custom_range to 35 for all other analyses
                    hdf5_path='flow_data.h5'
                )
                
                # Initialize FlowAnalysis with the configuration for custom_range=35
                flow_analysis_35 = Fizi(config_35)
                
                # Prepare directories and lists for warped images
                flow_vis_list, img_list, warped_img_list, gradient_list, binary_image_list, x, y = flow_analysis_35.create_flow_lists(
                    config_35.trial_path, 
                    config_35.img_path, 
                    config_35.dir_ext, 
                    step=config_35.step, 
                    start_x=config_35.start_x, 
                    warp=config_35.warp_analysis, 
                    binary_image=config_35.binary_image_analysis,
                    custom_range=config_35.custom_range
                )
                
                # Perform analyses with custom_range=35
                flow_analysis_35.plot_and_save_losses()
                flow_analysis_35.generate_global_heatmaps(gradient_list)
                flow_analysis_35.save_warped_images(warped_img_list)
                flow_extraction=Flay(config_35, flow_vis_list)
                flow_extraction.save_flow_vectors()
                
                # Now perform the average_heatmaps_with_confidence_intervals with custom_range='end'
                config_end = FlowConfig(
                    trial_path=trial_path,
                    img_path=dataset_path,
                    dir_ext='flow_npy/result_',
                    step=1,
                    start_x=0,
                    end_x=None,
                    start_y=0,
                    end_y=None,
                    reverse_flow=False,
                    binary_image_analysis=False,
                    warp_analysis=True,
                    custom_range='end',  # Set custom_range to 'end' for this specific analysis
                    hdf5_path='flow_data.h5'
                )
                
                flow_analysis_end = Fizi(config_end)
                flow_vis_list2, img_list2, warped_img_list2, gradient_list2, binary_image_list, x, y = flow_analysis_end.create_flow_lists(
                    config_35.trial_path, 
                    config_35.img_path, 
                    config_35.dir_ext, 
                    step=config_35.step, 
                    start_x=config_35.start_x, 
                    warp=config_35.warp_analysis, 
                    binary_image=config_35.binary_image_analysis,
                    custom_range=config_35.custom_range
                )
                flow_analysis_end.average_heatmaps_with_confidence_intervals(gradient_list2)
                plt.close('all')
                logging.info(f'Completed flow analysis for {trial_path}')
                
            except Exception as e:
                print(f"An error occurred during processing of {trial_path}")
                logging.error(f"An error occurred during processing of {trial_path}: {e}")

if __name__ == "__main__":
    base_path = r"D:\test_cases"
    specific_trial_numbers = [35]  # Specify the trial numbers you want to run
    process_test_cases(base_path, specific_trial_numbers)