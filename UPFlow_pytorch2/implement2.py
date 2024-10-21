import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from skimage import io, img_as_float32
from model.upflow import UPFlow_net
from utils.tools import tools
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
import argparse
from tqdm import tqdm
from colorama import Fore, Style

class ComposePairs:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2):
        for t in self.transforms:
            img1, img2 = t(img1, img2)
        return img1, img2

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2):
        if random.random() < self.p:
            img1 = np.fliplr(img1).copy()  # Ensure positive strides
            img2 = np.fliplr(img2).copy()  # Ensure positive strides
        return img1, img2

class CustomFlowDataset(Dataset):
    def __init__(self, root_dir, transform=None, augmentations=None, target_mean=0.5, crop_size=(140, 400), num_crops_per_image=3, divide_into_regions=False):
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.Compose([transforms.ToTensor()])
        self.augmentations = augmentations
        self.target_mean = target_mean
        self.crop_size = crop_size
        self.num_crops_per_image = num_crops_per_image
        self.divide_into_regions = divide_into_regions
        self.image_pairs = self._load_image_pairs()

    def _load_image_pairs(self):
        def numerical_sort_key(file_name):
            return int(file_name.split('_')[-1].split('.')[0])  # Extracts the number after 'frame_'

        image_files = sorted([f for f in os.listdir(self.root_dir) if f.endswith('.png')], key=numerical_sort_key)
        image_pairs = []
        
        for i in range(len(image_files) - 1):
            img1 = os.path.join(self.root_dir, image_files[i])
            img2 = os.path.join(self.root_dir, image_files[i + 1])
            image_pairs.append((img1, img2))

        return image_pairs

    def __len__(self):
        return len(self.image_pairs) * self.num_crops_per_image

    def __getitem__(self, idx):
        actual_idx = idx // self.num_crops_per_image
        region_idx = idx % 4  # Assuming 4 regions

        img1_path, img2_path = self.image_pairs[actual_idx]
        img1 = io.imread(img1_path)
        img2 = io.imread(img2_path)

        # Ensure the images are grayscale
        if len(img1.shape) == 3:
            img1 = img1[:, :, 0]
        if len(img2.shape) == 3:
            img2 = img2[:, :, 0]

        # Random cropping
        img1, img2 = self.random_crop(img1, img2)
        
        # Normalize to [0, 1] range
        img1 = img_as_float32(img1)
        img2 = img_as_float32(img2)

        # Adjust mean intensity if target_mean is provided
        if self.target_mean is not None:
            img1 = img1 - img1.mean() + self.target_mean
            img2 = img2 - img2.mean() + self.target_mean

        # Ensure arrays have positive strides
        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)

        if self.divide_into_regions:
            # Divide each image into four regions
            regions_img1 = self.divide_into_regions_method(img1)
            regions_img2 = self.divide_into_regions_method(img2)

            # Select the specific region to return based on region_idx
            region_img1 = regions_img1[region_idx]
            region_img2 = regions_img2[region_idx]

            # Upsample the selected region
            upsampled_img1 = self.upsample_region(region_img1)
            upsampled_img2 = self.upsample_region(region_img2)
            
            if self.augmentations:
                upsampled_img1, upsampled_img2 = self.augmentations(upsampled_img1, upsampled_img2)
            if self.transform:
                upsampled_img1 = self.transform(upsampled_img1)
                upsampled_img2 = self.transform(upsampled_img2)

            return upsampled_img1, upsampled_img2  # Return only one pair of the selected region

        else:
            # Process the entire image without upsampling
            if self.augmentations:
                img1, img2 = self.augmentations(img1, img2)

            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return img1, img2  # Single image pair

    def random_crop(self, img1, img2):
        if self.crop_size is None:
            return img1, img2
        else:
            h, w = img1.shape
            new_h, new_w = self.crop_size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            img1 = img1[top: top + new_h, left: left + new_w]
            img2 = img2[top: top + new_h, left: left + new_w]

            return img1, img2

    def divide_into_regions_method(self, img):
        h, w = img.shape
        upper_left = img[0:h//2, 0:w//2]
        upper_right = img[0:h//2, w//2:w]
        lower_left = img[h//2:h, 0:w//2]
        lower_right = img[h//2:h, w//2:w]
        return [upper_left, upper_right, lower_left, lower_right]

    def upsample_region(self, region):
        h, w = region.shape
        return cv2.resize(region, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

class Loss_manager:
    def __init__(self):
        self.error_meter = tools.Avg_meter_ls()

    def fetch_loss(self, loss, loss_dict, name, batch_N, short_name=None):
        if name not in loss_dict.keys():
            pass
        elif loss_dict[name] is None:
            pass
        else:
            this_loss = loss_dict[name].mean()
            self.error_meter.update(name=name, val=this_loss.item(), num=batch_N, short_name=short_name)
            loss = loss + this_loss
        return loss

    def prepare_epoch(self):
        self.error_meter.reset()

    def log_info(self):
        p_str = self.error_meter.print_all_losses()
        return p_str

    def compute_loss(self, loss_dict, batch_N):
        loss = 0
        loss = self.fetch_loss(loss=loss, loss_dict=loss_dict, name='photo_loss', short_name='ph', batch_N=batch_N)
        loss = self.fetch_loss(loss=loss, loss_dict=loss_dict, name='smooth_loss', short_name='sm', batch_N=batch_N)
        loss = self.fetch_loss(loss=loss, loss_dict=loss_dict, name='census_loss', short_name='cen', batch_N=batch_N)
        loss = self.fetch_loss(loss=loss, loss_dict=loss_dict, name='msd_loss', short_name='msd', batch_N=batch_N)
        loss = self.fetch_loss(loss=loss, loss_dict=loss_dict, name='eq_loss', short_name='eq', batch_N=batch_N)
        loss = self.fetch_loss(loss=loss, loss_dict=loss_dict, name='oi_loss', short_name='oi', batch_N=batch_N)
        return loss

class Config:
    def __init__(self, **kwargs):
        self.exp_dir = './demo_exp'
        self.if_cuda = torch.cuda.is_available()
        self.batchsize = 12
        self.NUM_WORKERS = 16
        self.n_epoch = 20
        self.batch_per_epoch = 500
        self.batch_per_print = 25
        self.lr = 5e-4
        self.weight_decay = 1e-4
        self.scheduler_gamma = 1
        self.model_save_path = 'upflow_net_state_dict_final.pth'
        self.image_path = 'ex_images.png'
        self.pretrain_path = None

        self.update(kwargs)

    def update(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def save(self):
        config_path = os.path.join(self.exp_dir, 'config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(exp_dir):
        config_path = os.path.join(exp_dir, 'config.pkl')
        with open(config_path, 'rb') as f:
            kwargs = pickle.load(f)
        return Config(**kwargs)

class Trainer:
    def __init__(self, config, param_dict, mode='train'):
        self.config = config
        self.param_dict = param_dict
        self.pretrain_path = config.pretrain_path
        self.image_path = config.image_path
        self.mode = mode
        self.model_path = os.path.join(self.config.exp_dir, self.config.model_save_path)

        tools.check_dir(self.config.exp_dir)

        self.net = self.load_model()

        self.train_set = self.load_training_dataset()
        self.dataloader = DataLoader(
            self.train_set,
            batch_size=self.config.batchsize if self.mode == 'train' else 1,  # Adjust for inference
            shuffle=self.mode == 'train',  # Only shuffle during training
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            drop_last=self.mode == 'train'  # Only drop last batch during training
        )

    def run(self):
        """Run training or inference depending on the mode."""
        if self.mode == 'train':
            self.train()
        elif self.mode == 'inference':
            self.infer()

    def train(self):
        """Training loop."""
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr, amsgrad=True, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=5)
        loss_manager = Loss_manager()
        timer = tools.time_clock()

        print("Start training" + '=' * 10)
        best_val_loss = float('inf')

        losses = { 'smooth_loss': [], 'photo_loss': [], 'census_loss': [], 'msd_loss': [], 'eq_loss': [], 'oi_loss': [] }

        for epoch in range(self.config.n_epoch):
            running_loss = 0.0
            loss_manager.prepare_epoch()

            timer.start()
            for i_batch, (img1, img2) in enumerate(self.dataloader):
                img1, img2 = img1.to(self.device), img2.to(self.device)

                optimizer.zero_grad()

                input_dict = self.prepare_input(img1, img2, self.device, loss_flag=True)
                out_data = self.net(input_dict)

                loss = self.calculate_loss(out_data, loss_manager, batch_size=img1.shape[0])
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i_batch % self.config.batch_per_print == 0:
                    print(f"Epoch [{epoch+1}/{self.config.n_epoch}], Batch [{i_batch+1}/{len(self.dataloader)}], Loss: {loss.item()}")

            # Adjust learning rate scheduler
            avg_loss = running_loss / len(self.dataloader)
            scheduler.step(avg_loss)
            timer.end()
            print(f' === Epoch {epoch+1} use time {timer.get_during():.2f} seconds ===')


            # Save the best model
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                self.save_model(os.path.join(self.config.exp_dir, self.config.model_save_path))
                print(f"Best model saved at epoch {epoch+1} with loss {best_val_loss}")

            if epoch % 1 == 0:  # Adjust the frequency as needed
                inter_flow = out_data['interpolation_flow'][0][0].detach().cpu().numpy()
                inter_map = out_data['interpolation_map'][0][0].detach().cpu().numpy()
                self.visualize_interpolation(inter_flow, inter_map, epoch)

            with open(os.path.join(self.config.exp_dir, 'loss_data.pkl'), 'wb') as f:
                pickle.dump(losses, f)
            
            for param_group in optimizer.param_groups:
                print(f"Current learning rate: {param_group['lr']}")
    
    def infer(self):
        """Inference loop."""
        for i, (img1, img2) in enumerate(tqdm(self.dataloader, desc=f"{Fore.GREEN}Inference Progress")):
            img1, img2 = img1.to(self.device), img2.to(self.device)
            with torch.no_grad():
                # Use prepare_input to build the input dictionary
                input_dict = self.prepare_input(img1, img2, self.device, loss_flag=False)
                
                # Run the model
                output_dict = self.net(input_dict)
                
                # Save the flow output
                flow_fw = output_dict['flow_b_out'].cpu().numpy()
                flow_dir = os.path.join(self.config.exp_dir, 'flow_npy')
                os.makedirs(flow_dir, exist_ok=True)
                np.save(f'{flow_dir}/result_{i}.npy', flow_fw)
    
    def prepare_input(self, img1, img2, device, loss_flag=True):
        """Prepare input dictionary for the model."""
        start = torch.zeros((img1.size(0), 2, 1, 1), device=device)
        return {
            'im1': img1,
            'im2': img2,
            'im1_raw': img1,
            'im2_raw': img2,
            'start': start,
            'if_loss': loss_flag
        }

    def calculate_loss(self, out_data, loss_manager, batch_size):
        """Calculate the loss based on the output data."""
        loss_dict = {key: out_data[key] for key in ['smooth_loss', 'photo_loss', 'census_loss', 'msd_loss', 'eq_loss', 'oi_loss'] if key in out_data}
        return loss_manager.compute_loss(loss_dict=loss_dict, batch_N=batch_size)
    
    def visualize_interpolation(self, inter_flow, inter_map, epoch):
        inter_flow = np.array(inter_flow)
        inter_map = np.array(inter_map)

        # Print shapes for debugging
        print(f"inter_flow shape: {inter_flow.shape}")
        print(f"inter_map shape: {inter_map.shape}")

        if inter_flow.ndim == 3:
            inter_flow = inter_flow[0]  # Assuming you want the first channel
        elif inter_flow.ndim == 1:
            raise ValueError("inter_flow should be at least 2D for visualization")

        if inter_map.ndim == 3:
            inter_map = inter_map[0]  # Assuming you want the first channel
        elif inter_map.ndim == 1:
            raise ValueError("inter_map should be at least 2D for visualization")

        # Create the visualization directory if it doesn't exist
        vis_dir = os.path.join(self.config.exp_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # Plotting the interpolation flow and map
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(inter_flow, cmap='jet')
        plt.title(f'Interpolation Flow - Epoch {epoch}')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(inter_map, cmap='jet')
        plt.title(f'Interpolation Map - Epoch {epoch}')
        plt.colorbar()

        plt.savefig(os.path.join(vis_dir, f'interpolation_visualization_epoch_{epoch}.png'))
        plt.close()

    def save_model(self, path):
        model_state = {
            'state_dict': self.net.state_dict(),
            'training_params': self.config.__dict__,
            'model_params': self.param_dict
        }
        torch.save(model_state, path)

    def load_model(self):
        """Load the model based on mode (train or inference)."""
        net_conf = UPFlow_net.config()
        net_conf.update(self.param_dict)
        net = UPFlow_net(net_conf)

        # Load model weights based on mode
        if self.mode == 'inference' and self.model_path:
            state_dict = torch.load(self.model_path)
            net.load_state_dict(state_dict['state_dict'], strict=False)
            net.eval()  # Set the model to evaluation mode for inference
        elif self.mode == 'train':
            if self.pretrain_path is not None:
                state_dict = torch.load(self.pretrain_path)
                net.load_state_dict(state_dict['state_dict'], strict=False)
            net.train()  # Set the model to training mode by default

        # Move the model to GPU if available, otherwise keep it on CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config.if_cuda else 'cpu')
        net.to(self.device)

        return net

    def load_training_dataset(self):
        if self.mode == 'train':
            augmentations = ComposePairs([RandomHorizontalFlip(p=0.2)])
            transform = transforms.Compose([transforms.ToTensor()])
            dataset = CustomFlowDataset(self.image_path, transform=transform, augmentations=None, target_mean=0.5, crop_size=None, num_crops_per_image=1, divide_into_regions=False)
        elif self.mode == 'inference':
            transform = transforms.Compose([transforms.ToTensor()])
            dataset = CustomFlowDataset(self.image_path, transform=transform, augmentations=None, target_mean=0.5, crop_size=None, num_crops_per_image=1, divide_into_regions=False)
        else:
            raise ValueError("Mode should be either 'train' or 'inference'.")
        return dataset

def load_config(json_file):
    with open(json_file, 'r') as f:
        config = json.load(f)
    experiment_name = config['experiment_name']
    config['training_param']['exp_dir'] = config['training_param']['exp_dir'].format(experiment_name=experiment_name)
    config['training_param']['model_save_path'] = config['training_param']['model_save_path'].format(experiment_name=experiment_name)
    return config

if __name__ == "__main__":
    # Set up argument parser to accept a config file as an argument
    parser = argparse.ArgumentParser(description="Train a model with parameters from a config file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the JSON config file e.g. config.json")
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True, help="Mode to run the model: 'train' or 'inference'")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Load the configuration from the provided JSON file
    config = load_config(args.config)

    # Extracting parameters from config
    experiment_name = config['experiment_name']
    training_param = config['training_param']
    param_dict = config['param_dict']

    # Initialize Config and Trainer
    conf = Config(**training_param)
    trainer = Trainer(conf, param_dict=param_dict, mode=args.mode)

    # Start training
    trainer.run()


"""
# This is an example of a typical JSON configuration that can be used in your project.

{
    "experiment_name": "UPF_A03_C_DP_30_trial_1",
    "training_param": {
        "exp_dir": "D:/Spray conditions/A03_512-complex-09-08-24.pth_inference/UPF_A03_C_DP_30_trial_1",  # Directory for the experiment
        "batchsize": 2,  # Batch size for training
        "NUM_WORKERS": 16,  # Number of workers for data loading
        "n_epoch": 60,  # Number of epochs for training
        "batch_per_epoch": 500,  # Number of batches per epoch
        "batch_per_print": 25,  # Number of batches after which the results are printed
        "lr": 0.0002,  # Learning rate for the optimizer
        "weight_decay": 0.00001,  # Weight decay for regularization
        "scheduler_gamma": 1,  # Learning rate scheduler gamma
        "model_save_path": "UPF_A03_C_DP_30_trial_1.pth"  # Path to save the trained model
    },
    "param_dict": {
        "occ_type": "for_back_check",  # Type of occlusion handling
        "alpha_1": 0.1,  # Parameter alpha_1 for the model
        "alpha_2": 0.5,  # Parameter alpha_2 for the model
        "occ_check_obj_out_all": "obj",  # Occlusion check mode
        "stop_occ_gradient": false,  # Whether to stop the gradient during occlusion check
        "smooth_level": "final",  # Smoothing level for post-processing
        "smooth_type": "edge",  # Type of smoothing used (edge-based smoothing)
        "smooth_order_1_weight": 0,  # Weight for first-order smoothing
        "smooth_order_2_weight": 0.000001,  # Weight for second-order smoothing
        "photo_loss_type": "SSIM",  # Type of photometric loss (e.g., SSIM)
        "photo_loss_delta": 0.4,  # Delta value for photometric loss
        "photo_loss_use_occ": false,  # Whether to use occlusion in photo loss calculation
        "photo_loss_census_weight": 0.5,  # Weight for census-based photometric loss
        "if_norm_before_cost_volume": true,  # Normalize before calculating cost volume
        "norm_moments_across_channels": false,  # Normalize moments across channels
        "norm_moments_across_images": false,  # Normalize moments across images
        "multi_scale_distillation_weight": 0.01,  # Weight for multi-scale distillation loss
        "multi_scale_distillation_style": "upup",  # Style of multi-scale distillation
        "multi_scale_distillation_occ": false,  # Whether to apply occlusion in distillation
        "if_froze_pwc": false,  # Whether to freeze the PWC-Net layers
        "input_or_sp_input": 1,  # Input type for the model
        "if_use_boundary_warp": true,  # Whether to use boundary warping in the model
        "if_sgu_upsample": true,  # Whether to use SGU upsample method
        "if_use_cor_pytorch": false,  # Use PyTorch implementation of correlation
        "photo_weighting": false,  # Whether to apply photo weighting in the loss function
        "if_attention_mechanism": false  # Whether to apply attention mechanism in the model
    },
    "pretrain_path": null,  # Path to a pre-trained model (if any), otherwise null
    "root_dir": "D:/Spray conditions/A03_512-complex-09-08-24.pth_inference/denoised_images"  # Root directory for the denoised images
}
"""