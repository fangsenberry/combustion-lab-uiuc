import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
import math
import pandas as pd


# Backbone conv net.
# Zhang, Kai, et al. "Beyond a gaussian denoiser: Residual learning of deep cnn for
# image denoising." IEEE transactions on image processing 26.7 (2017): 3142-3155.
# https://github.com/SaoYan/DnCNN-PyTorch/blob/master/models.py,
class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=17, dropout_rate = 0.25):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_rate))  # Adding dropout after each ReLU #TODO: this was added, can comment out since CNNs not guaranteed to work well with this, especially with this denoising task.
        layers.append(nn.Conv2d(in_channels=features, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        # out = self.dncnn(x)
        # return out
        return self.dncnn(x)


# Sequence image set; dataset_path takes numpy files shaped (N, H, W) in [0,1], in sequential order.
class Sequential_Dataset(Dataset):
    def __init__(self, dataset_path, noise_std=1.0, use_stacking=True):
        df = torch.tensor(np.load(dataset_path)).float() / 255
        self.df1 = df[:-2, :, :]
        self.df2 = df[1:-1, :, :]
        self.df3 = df[2:, :, :]
        self.noise_std = noise_std
        self.use_stacking = use_stacking

    def __len__(self):
        return len(self.df2)

    def __getitem__(self, index):
        x1 = self.df1[index, :, :].unsqueeze(0)
        x2 = self.df2[index, :, :].unsqueeze(0)
        x3 = self.df3[index, :, :].unsqueeze(0)
        # Adding Gaussian noise sampled from N(0, 1)
        noise = torch.randn_like(x2) * self.noise_std
        x2_noisy = x2 + noise
        
        if self.use_stacking:
            input_data = torch.cat((x1, x2_noisy, x3), dim=0)
        else:
            input_data = x2_noisy

        return {'input': input_data, 'target': x2}

def Seq2S(train_set_path, test_set_path, save_path,
          noise_std=1.0, use_stacking=True, batch_size=1, max_epoch=25,
          adam_lr=1e-4, adam_beta1=0.5, lr_sched_ss=1, lr_sched_gamma=0.9,
          early_stop_patience=3):
    assert(torch.cuda.is_available()), 'No GPU or CUDA not enabled.'

    start_time = time.time()
    device = 'cuda:0'
    os.makedirs(save_path, exist_ok=True)

    # Model & optimizer
    model = DnCNN(channels=3 if use_stacking else 1)
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=adam_lr, betas=(adam_beta1, 0.99))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Datasets:
    train_set = Sequential_Dataset(train_set_path, noise_std=noise_std, use_stacking=use_stacking)
    test_set = Sequential_Dataset(test_set_path, noise_std=noise_std, use_stacking=use_stacking)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Training loop:
    epoch_range = trange(max_epoch)
    print('Training:')

    train_loss_list, test_loss_list, iteration_list = [], [], []
    iteration, patience, min_test_loss = 0, 0, float('inf')

    for epoch in epoch_range:
        model.train()
        train_loss_avg = 0.0
        for batch in train_loader:
            optimizer.zero_grad()

            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()

            iteration += 1
            train_loss_avg += loss.item() * inputs.size(0) / len(train_set)

        train_loss_list.append(train_loss_avg)

        # Validation
        model.eval()
        test_loss_avg = 0.0
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)

                outputs = model(inputs)
                loss = F.mse_loss(outputs, targets)
                test_loss_avg += loss.item() * inputs.size(0) / len(test_set)

        test_loss_list.append(test_loss_avg)
        scheduler.step(test_loss_avg)

        epoch_range.set_description(
            f'Epoch {epoch + 1}/{max_epoch}, Train Loss: {train_loss_avg:.4f}, Test Loss: {test_loss_avg:.4f}')

        iteration_list.append(iteration)

    torch.save(model.state_dict(), os.path.join(save_path, "model_weights.pth"))

    # Inference
    print("Inference:")
    inference_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    inference_save_path = os.path.join(save_path, "denoised_images")
    os.makedirs(inference_save_path, exist_ok=True)

    model.eval()
    transform_to_pil = transforms.ToPILImage()

    for index, batch in enumerate(tqdm(inference_loader)):
        inputs = batch['input'].to(device)
        denoised_output = model(inputs).detach().cpu()
        denoised_image = transform_to_pil(denoised_output.squeeze(0))
        denoised_image.save(os.path.join(inference_save_path, f"{index}.png"))

    total_time = (time.time() - start_time) / 60
    print(f"Finished in {total_time:.2f} minutes")

    log_df = pd.DataFrame({'iteration': iteration_list, 'train_loss': train_loss_list, 'test_loss': test_loss_list})
    log_df.to_csv(os.path.join(save_path, f"log_{int(total_time * 60)}s.csv"), index=False)

if __name__ == '__main__':

    fuel_train_set_path = f'F24_dataset_test.npy'
    fuel_test_set_path = f'F24_dataset_test.npy'

    Seq2S(train_set_path=fuel_train_set_path,
          test_set_path=fuel_test_set_path,
          save_path=f'F24_Seq2S_30%',
          p=0.3)