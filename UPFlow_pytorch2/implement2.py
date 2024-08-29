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
    def __init__(self, config, param_dict, pretrain_path=None, root_dir=None):
        self.config = config
        self.param_dict = param_dict
        self.pretrain_path = pretrain_path
        self.root_dir = root_dir

        tools.check_dir(self.config.exp_dir)

        self.net = self.load_model(pretrain_path=self.pretrain_path)
        self.train_set = self.load_training_dataset(self.root_dir)

    def training(self):
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.config.batchsize,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr, amsgrad=True, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        loss_manager = Loss_manager()
        timer = tools.time_clock()
        print("start training" + '=' * 10)
        best_val_loss = float('inf')
        timer.start()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        self.net.to(device)

        losses = { 'smooth_loss': [], 'photo_loss': [], 'census_loss': [], 'msd_loss': [], 'eq_loss': [], 'oi_loss': [] }
        # Select a specific pair of images for visualization
        fixed_img1, fixed_img2 = next(iter(train_loader))
        fixed_img1, fixed_img2 = fixed_img1[0].to(device).unsqueeze(0), fixed_img2[0].to(device).unsqueeze(0)

        for epoch in range(self.config.n_epoch):
            running_loss = 0.0
            loss_manager.prepare_epoch()
            for i_batch, (img1, img2) in enumerate(train_loader):
                img1, img2 = img1.to(device), img2.to(device)
                batchsize = img1.shape[0]

                self.net.train()
                optimizer.zero_grad()
                start = torch.zeros((img1.size(0), 2, 1, 1), device=device)
                input_dict = {
                    'im1': img1,
                    'im2': img2,
                    'im1_raw': img1,
                    'im2_raw': img2,
                    'start': start,
                    'if_loss': True
                }
                out_data = self.net(input_dict)

                loss_dict = {key: out_data[key] for key in losses.keys() if key in out_data}
                if not loss_dict:
                    print("Error: loss_dict is empty.")
                    continue

                loss = loss_manager.compute_loss(loss_dict=loss_dict, batch_N=batchsize)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                for key in losses.keys():
                    if key in out_data:
                        losses[key].append(out_data[key].item())

                if i_batch % self.config.batch_per_print == 0:
                    print(f"Epoch [{epoch+1}/{self.config.n_epoch}], Batch [{i_batch+1}/{len(train_loader)}], Loss: {loss.item()}")
                    loss_values = ", ".join([f"{key.replace('_', ' ').capitalize()}: {out_data[key].item()}" for key in loss_dict])
                    print(f"Losses: {loss_values}")

            avg_loss = running_loss / len(train_loader)
            scheduler.step(avg_loss)
            timer.end()
            print(f' === Epoch {epoch+1} use time {timer.get_during():.2f} seconds ===')
            timer.start()

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

    def load_model(self, model_path=None, mode='train', pretrain_path=None):
        net_conf = UPFlow_net.config()
        net_conf.update(self.param_dict)
        net = UPFlow_net(net_conf)

        if mode == 'inference' and model_path:
            state_dict = torch.load(model_path)
            net.load_state_dict(state_dict['state_dict'], strict=False)
            net.eval()
        else:
            if pretrain_path is not None:
                state_dict = torch.load(pretrain_path)
                net.load_state_dict(state_dict['state_dict'], strict=False)
            if self.config.if_cuda:
                net = net.cuda()
            net.train()
        return net

    def load_training_dataset(self, root_dir):
        augmentations = ComposePairs([RandomHorizontalFlip(p=0.2)])
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = CustomFlowDataset(root_dir, transform=transform, augmentations=None, target_mean=0.5, crop_size=None, num_crops_per_image=4, divide_into_regions=True)
        return dataset

if __name__ == "__main__":
    # Training parameters
    experiment_name = "UPF_A01_C_DP_35_trial_27"

    # Training parameters
    training_param = {
        'exp_dir': os.path.join(r"D:\test_cases", experiment_name),  # Use the base name for exp_dir
        'batchsize': 12,
        'NUM_WORKERS': 16,
        'n_epoch': 25,
        'batch_per_epoch': 500,
        'batch_per_print': 25,
        'lr': 2e-4,
        'weight_decay': 1e-4,
        'scheduler_gamma': 1,
        'model_save_path': f'{experiment_name}.pth'  # Use the base name for model_save_path
    }
    pretrain_path =r"D:\test_cases\UPF_A01_C_DP_35_trial_26\UPF_A01_C_DP_35_trial_26.pth"  # Add the path to the pretrained model if needed
    root_dir = r"D:\contrast_adjusted_512-complex"
    
    param_dict = {
        'occ_type': 'for_back_check',
        'alpha_1': 0.1,
        'alpha_2': 0.5,
        'occ_check_obj_out_all': 'obj',
        'stop_occ_gradient': False,
        'smooth_level': 'final',
        'smooth_type': 'edge',
        'smooth_order_1_weight': 0,
        'smooth_order_2_weight': 1e-12,
        'photo_loss_type': 'abs_robust',
        'photo_loss_delta': 0.4,
        'photo_loss_use_occ': True,
        'photo_loss_census_weight': 0.5,
        'if_norm_before_cost_volume': True,
        'norm_moments_across_channels': False,
        'norm_moments_across_images': False,
        'multi_scale_distillation_weight': 0.01,
        'multi_scale_distillation_style': 'upup',
        'multi_scale_distillation_occ': True,
        'if_froze_pwc': False,
        'input_or_sp_input': 1,
        'if_use_boundary_warp': True,
        'if_sgu_upsample': True,  # if use sgu upsampling
        'if_use_cor_pytorch': False,
        'photo_weighting': False,
    }
    
    conf = Config(**training_param)
    trainer = Trainer(conf, param_dict=param_dict, pretrain_path=pretrain_path, root_dir=root_dir)
    trainer.training()