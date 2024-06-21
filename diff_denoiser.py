import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io
from skimage.util import random_noise
from tqdm import tqdm

image_dir = 'data/noisy_images_preprocessed/A01_C_DP_35.0'
output_dir = 'data/denoised_images'

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == "cuda": torch.cuda.empty_cache()
print(f"Using device: {device}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

def print_model_info(model):
    # Calculate number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    # Calculate model size in bytes
    model_size = sum(p.element_size() * p.numel() for p in model.parameters())
    
    # Convert model size to megabytes (MB)
    model_size_MB = model_size / (1024 * 1024)
    
    print(f"Number of parameters: {num_params}")
    print(f"Model size: {model_size_MB:.2f} MB")

class NoisyImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = glob.glob(os.path.join(image_dir, '**', '*.png'), recursive=True)
        print(f"Found {len(self.image_paths)} images in {image_dir}")
        if len(self.image_paths) == 0:
            print(f"No images found. Check the directory path: {self.image_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = io.imread(image_path)

        if len(image.shape) == 3:  # Convert to grayscale if the image is RGB
            image = np.mean(image, axis=2)

        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        noisy_image = random_noise(image, mode='gaussian', var=0.01).astype(np.float32)

        if self.transform:
            image = self.transform(image)
            noisy_image = self.transform(noisy_image)

        return noisy_image, image

transform = transforms.Compose([transforms.ToTensor()])

dataset = NoisyImageDataset(image_dir=image_dir, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

class SimpleDnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=3, features=512, dropout_rate=0.0):
        super(SimpleDnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
            # layers.append(nn.Dropout(p=dropout_rate))  # Adding dropout after each ReLU
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

        # Residual connection
        # self.residual = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        # out = self.dncnn(x)
        # return out + self.residual(x)  # Adding the residual connection
        return self.dncnn(x)

model = SimpleDnCNN().to(device)
model = nn.DataParallel(model)
print_model_info(model)

def train(model, train_loader, criterion, optimizer, scheduler, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (noisy_images, images) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        print(f"Batch {batch_idx + 1}/{len(train_loader)}")  # Debugging line
        noisy_images = noisy_images.to(device, dtype=torch.float32)
        images = images.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(noisy_images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f"Batch {batch_idx + 1} loss: {loss.item()}")  # Debugging line

    print(f"Epoch {epoch} Training Loss: {running_loss / len(train_loader)}")
    return running_loss / len(train_loader)

def test(model, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_idx, (noisy_images, images) in enumerate(tqdm(test_loader, desc=f"Testing Epoch {epoch}")):
            print(f"Batch {batch_idx + 1}/{len(test_loader)}")  # Debugging line
            noisy_images = noisy_images.to(device, dtype=torch.float32)
            images = images.to(device, dtype=torch.float32)

            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            test_loss += loss.item()
            print(f"Batch {batch_idx + 1} loss: {loss.item()}")  # Debugging line

    print(f"Epoch {epoch} Testing Loss: {test_loss / len(test_loader)}")
    return test_loss / len(test_loader)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
num_epochs = 25

train_losses = []
test_losses = []
best_test_loss = float('inf')
best_model_path = 'dncnn_best.pth'

for epoch in range(1, num_epochs + 1):
    print(f"Learning rate at epoch {epoch}: {optimizer.param_groups[0]['lr']}")
    train_loss = train(model, train_loader, criterion, optimizer, scheduler, epoch)
    test_loss = test(model, test_loader, criterion, epoch)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    scheduler.step(test_loss)

    # Save the best model
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), best_model_path)

# Plotting train and test losses
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()

# Load the best model
model.load_state_dict(torch.load(best_model_path))

def denoise_image(model, image_path):
    model.eval()
    image = io.imread(image_path).astype(np.float32) / 255.0
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    noisy_image = random_noise(image, mode='gaussian', var=0.01).astype(np.float32)

    transform = transforms.Compose([transforms.ToTensor()])
    noisy_image_tensor = transform(noisy_image).unsqueeze(0).to(device, dtype=torch.float32)
    
    with torch.no_grad():
        denoised_image_tensor = model(noisy_image_tensor).squeeze(0).cpu()
    
    denoised_image = denoised_image_tensor.permute(1, 2, 0).numpy()
    return noisy_image, denoised_image

os.makedirs(output_dir, exist_ok=True)

for image_path in glob.glob(os.path.join(image_dir, '**', '*.png'), recursive=True):
    print(f"Processing {image_path}")  # Debug print to check paths
    noisy_image, denoised_image = denoise_image(model, image_path)
    image_name = os.path.basename(image_path)
    
    plt.imsave(os.path.join(output_dir, f"noisy_{image_name}"), noisy_image, cmap='gray')
    plt.imsave(os.path.join(output_dir, f"denoised_{image_name}"), denoised_image, cmap='gray')

    print(f"Processed {image_name}")