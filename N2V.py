import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import os
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float32
import csv
import argparse
from tabulate import tabulate
import torch.nn as nn
import torch.optim as optim
from glob import glob
from torch.cuda.amp import autocast, GradScaler

import image_preprocess


# Define the dataset class
class NoisyImageDataset(Dataset):
    def __init__(self, image_dir, mask_prob=0.1, transform=None):
        self.image_files = glob(os.path.join(image_dir, '*.png'))
        self.mask_prob = mask_prob
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = io.imread(self.image_files[idx])
        image = img_as_float32(image)  # Normalize to [0, 1]

        if self.transform:
            image = self.transform(image)

        noisy_image = image.copy()
        mask = np.random.rand(*image.shape) < self.mask_prob
        noisy_image[mask] = 0  # Masking some pixels

        return noisy_image, image, mask  # Input is the masked image, target is the original image, mask is the masked positions

# class DenoisingCNN(nn.Module):
#     def __init__(self, num_features=256):
#         print(f"init model with {num_features} features")
#         super(DenoisingCNN, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, num_features, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(num_features, num_features, kernel_size=2, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(num_features, 1, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.encoder(x)  # Pass the input through the encoder
#         x = self.decoder(x)  # Pass the encoded output through the decoder
#         return x             # Return the decoded output

class DenoisingCNN(nn.Module):
    def __init__(self, num_features=256):
        print(f"init model with arg: {num_features}")
        super(DenoisingCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, num_features, kernel_size=3, padding=1),  # Output: (num_features, H, W)
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),  # Output: (num_features, H, W)
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: (num_features, H/2, W/2)
            nn.Conv2d(num_features, num_features*2, kernel_size=3, padding=1),  # Output: (num_features*2, H/2, W/2)
            nn.BatchNorm2d(num_features*2),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: (num_features*2, H/4, W/4)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_features*2, num_features, kernel_size=2, stride=2),  # Output: (num_features, H/2, W/2)
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
            nn.ConvTranspose2d(num_features, num_features, kernel_size=2, stride=2),  # Output: (num_features, H, W)
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
            nn.Conv2d(num_features, 1, kernel_size=3, padding=1),  # Output: (1, H, W)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def setup_ddp(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:29500',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def train_denoiser(rank, world_size, image_dir, models_dir, model_save_path_base, model_save_path, num_epochs=50, batch_size=16, learning_rate=0.001, num_features=256, mixed_precision=False):
    try:
        setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{rank}")
        
        # Load dataset
        dataset = NoisyImageDataset(image_dir)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, sampler=sampler)
        
        print(f"[Rank {rank}] There are {len(sampler)} samples assigned to this rank out of {len(dataset)} total samples.")
        
        # Initialize model
        model = DenoisingCNN(num_features=num_features).to(device)
        model = DDP(model, device_ids=[rank])

        # Loss and optimizer
        criterion = nn.MSELoss().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
        
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        train_losses = []

        # Training loop
        start_train_time = time.time()
        best_train_loss = float('inf')
        model.train()

        # Initialize the GradScaler for mixed precision
        if mixed_precision:
            scaler = GradScaler()

        for epoch in range(num_epochs):
            running_loss = 0.0
            dataloader.sampler.set_epoch(epoch)  # Ensure proper shuffling for each epoch
            for inputs, targets, masks in tqdm(dataloader, colour='blue', leave=False, desc=f"Epoch {epoch+1}/{num_epochs}"):
                inputs = inputs.unsqueeze(1).float().to(device)  # Add channel dimension and convert to float
                targets = targets.unsqueeze(1).float().to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass with mixed precision (autocast)
                if mixed_precision:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs * masks.unsqueeze(1).float().to(device), targets * masks.unsqueeze(1).float().to(device))

                    # Backward pass with scaled gradients
                    scaler.scale(loss).backward()

                    # Step the optimizer using the scaler
                    scaler.step(optimizer)

                    # Update the scale factor for next iteration
                    scaler.update()
                    
                else:

                
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs * masks.unsqueeze(1).float().to(device), targets * masks.unsqueeze(1).float().to(device))
                    
                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()
                    
                running_loss += loss.item() * inputs.size(0)
            
            lr_scheduler.step(running_loss)
            epoch_loss = running_loss / len(dataloader.dataset)
            train_losses.append(epoch_loss)
            
            # Save the model only on the primary process (rank 0)
            if rank == 0 and epoch_loss < best_train_loss:
                best_train_loss = epoch_loss
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved to {model_save_path} with loss {best_train_loss:.10f}")
            
            track_train_perf_dir = os.path.join(models_dir, f'track_train_perf_{model_save_path_base}')
            os.makedirs(track_train_perf_dir, exist_ok=True)
            
            test_images = sorted(glob(os.path.join(image_dir, '*.png')))
            if rank==0 and epoch % 5 == 0:
              model.eval()
              with torch.no_grad():
                  single_test_im_path = test_images[5]
                  s_t_im = io.imread(single_test_im_path)
                  s_t_im = img_as_float32(s_t_im)
                  s_t_input_tensor = torch.tensor(s_t_im).unsqueeze(0).unsqueeze(0).float().to(device)  # Add batch and channel dimensions
                  s_t_output_tensor = model(s_t_input_tensor)
                  s_t_output_image = s_t_output_tensor.squeeze().cpu().numpy()
                  o_path = f"{track_train_perf_dir}/{model_save_path_base} {epoch}.png"
                  plt.imsave(o_path, s_t_output_image, cmap='gray')
            
            if rank==0:
                elapsed_time = time.time() - start_train_time
                elapsed_minutes = int(elapsed_time // 60)
                elapsed_seconds = int(elapsed_time % 60)
                
                # Estimated remaining time in minutes and seconds
                total_estimated_time = (elapsed_time * num_epochs) / (epoch + 1)
                remaining_time = total_estimated_time - elapsed_time
                remaining_minutes = int(remaining_time // 60)
                remaining_seconds = int(remaining_time % 60)
                
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.10f}, Current Learning Rate: {optimizer.param_groups[0]['lr']}, Total Train Time Elapsed: {elapsed_minutes}m {elapsed_seconds}s, Estimated Time Remaining for Train: {remaining_minutes}m {remaining_seconds}s.")

        losses_file_path = os.path.join(models_dir, f'training_losses_{model_save_path_base}.csv')
        fig_file_path= os.path.join(models_dir, f'training_loss_plot_{model_save_path_base}.png')
        with open(losses_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'Training Loss'])
            for epoch, loss in enumerate(train_losses, start=1):
                writer.writerow([epoch, loss])
        
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(fig_file_path)  # Save the plot as a PNG file
    finally:
        cleanup_ddp()

def denoise_image(model, image_path, device, num_passes=1):
    model.eval()
    with torch.no_grad():
        image = io.imread(image_path)
        image = img_as_float32(image)  # Normalize to [0, 1]
        input_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().to(device)  # Add batch and channel dimensions
        
        curr_input_tensor = input_tensor
        for i in range(num_passes):
            output_tensor = model(curr_input_tensor)
            curr_input_tensor = output_tensor
        
        output_image = output_tensor.squeeze().cpu().numpy()
        return output_image

def main():
    parser = argparse.ArgumentParser(description='Description of your program.')
    
    parser.add_argument('--num_features', type=int, default=256, help='Number of Channels for Conv')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=int, default=2e-4, help='learning rate')
    parser.add_argument('--image_base', type=str, default='data/noisy_images_preprocessed/A05_C_DP_26.0', help='Path to the directory of a certain image set')
    parser.add_argument('--date', type=str, default='10-14-24', help='Date of running the model')
    parser.add_argument('--cuda_devices', type=str, default='0,1,2', help='Comma-separated list of CUDA devices to use, e.g., "0,1"')
    parser.add_argument('--image_path', type=str, help='Path to the image to denoise after training')

    args = parser.parse_args()

    # Set CUDA_VISIBLE_DEVICES based on the provided argument
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices

    num_features = args.num_features
    batch_size = args.batch_size
    image_base = args.image_base
    date = args.date
    learning_rate=args.lr

    world_size = len(args.cuda_devices.split(','))  # Number of GPUs based on the input

    # Paths and parameters
    image_dir = os.path.join(image_base, 'final_corrected') # Directory containing final corrected images
    model_save_path_base = f'{num_features}-complex-{date}.pth'  # Path to save the trained model
    models_dir = os.path.join(image_base, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, model_save_path_base)

    params = [
        ["Image Base Directory", image_base],
        ["Model Save Path", model_save_path],
        ["Number of Epochs", 50],
        ["Batch Size", batch_size],
        ["Learning Rate", learning_rate],
        ["Number of Features", num_features],
        ["CUDA Devices", args.cuda_devices],
    ]
    
    print(tabulate(params, headers=["Parameter", "Value"], tablefmt="pretty"))

    # Spawn multiple processes based on the number of GPUs specified
    mp.spawn(
        train_denoiser,
        args=(world_size, image_dir, models_dir, model_save_path_base, model_save_path, 50, batch_size, learning_rate, num_features),
        nprocs=world_size,
        join=True
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trained_model = DenoisingCNN(num_features=num_features).to(device)
    state_dict = torch.load(model_save_path, map_location=device, weights_only=True)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix if present
        else:
            new_state_dict[k] = v

    trained_model.load_state_dict(new_state_dict)

    # Create inference directory
    inference_dir = f"{model_save_path}_inference"
    os.makedirs(inference_dir, exist_ok=True)

    # Run inference on all images in the directory
    for image_file in tqdm(glob(os.path.join(image_dir, '*.png')), desc="Running inference", colour="green"):
        denoised_image = denoise_image(trained_model, image_file, device)
        output_path = os.path.join(inference_dir, os.path.basename(image_file))
        plt.imsave(output_path, denoised_image, cmap='gray')

if __name__ == '__main__':
    main()
