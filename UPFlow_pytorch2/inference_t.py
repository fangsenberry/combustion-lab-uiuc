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


def inference(model_path, dataset_path, result_path):
    # Load the model state dictionary
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
            flow_fw = output_dict['flow_f_out'].cpu().numpy()
            np.save(f'{result_path}/result_{i}.npy', flow_fw)
            
            # # Optionally visualize the flow
            # flow_vis = visualize_flow(flow_fw[0])
            # cv2.imwrite(f'{result_path}/result_{i}.png', flow_vis)
            print(f'Saved flow result for pair {i}')


def visualize_flow(flow):
    """Visualize optical flow."""
    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[0], flow[1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


if __name__ == "__main__":
    model_path = r"/home/caseyjo2/combustion-lab-uiuc/test_cases/UPF_A01_C_DP_35_trial_14/UPF_A01_C_DP_35_trial_14.pth"
    dataset_path = r"/home/caseyjo2/combustion-lab-uiuc/data/noisy_images_preprocessed/A01_C_DP_35.0/512_edges/"
    result_path = r"/home/caseyjo2/combustion-lab-uiuc/test_cases/UPF_A01_C_DP_35_trial_14/flow_npy"
    os.makedirs(result_path, exist_ok=True)
    inference(model_path, dataset_path, result_path)