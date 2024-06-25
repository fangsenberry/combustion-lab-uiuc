import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, util
import os

class NoisyDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = io.imread(image_path, as_gray=True)
        noisy_image = util.random_noise(image, mode='gaussian', var=0.01)  # Add Gaussian noise
        sample = {'image': image, 'noisy_image': noisy_image}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, util
import os

class NoisyDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = io.imread(image_path, as_gray=True)
        noisy_image = util.random_noise(image, mode='gaussian', var=0.01)  # Add Gaussian noise
        sample = {'image': image, 'noisy_image': noisy_image}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2

def train_model(model, dataloader, num_epochs=25):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch in dataloader:
            inputs = batch['noisy_image'].float().unsqueeze(1)  # Add channel dimension
            targets = batch['image'].float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    return model

# Parameters
image_dir = 'path/to/your/images'
batch_size = 4
num_epochs = 25

# Load data
dataset = NoisyDataset(image_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize and train model
model = UNet().cuda()  # Move model to GPU
model = train_model(model, dataloader, num_epochs)

torch.save(model.state_dict(), 'noise2void_model.pth')

# Load the model for inference
model.load_state_dict(torch.load('noise2void_model.pth'))
model.eval()

# Example of inference
with torch.no_grad():
    sample = dataset[0]
    noisy_image = torch.tensor(sample['noisy_image']).float().unsqueeze(0).unsqueeze(0).cuda()
    denoised_image = model(noisy_image).cpu().squeeze().numpy()
    io.imsave('denoised_image.tif', denoised_image)