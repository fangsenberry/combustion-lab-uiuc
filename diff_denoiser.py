import os
from skimage import io, exposure, img_as_float32
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from glob import glob
# from tqdm.auto import tqdm
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Define the dataset class
class NoisyImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_files = glob(os.path.join(image_dir, '*.png'))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = io.imread(self.image_files[idx])
        image = img_as_float32(image)  # Normalize to [0, 1]
        if self.transform:
            image = self.transform(image)
        return image, image  # Input and target are the same for self-supervised learning

# Define the denoising CNN model
class DenoisingCNN(nn.Module):
    def __init__(self):
        super(DenoisingCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)  # Pass the input through the encoder
        x = self.decoder(x)  # Pass the encoded output through the decoder
        return x             # Return the decoded output

def train_denoiser(image_dir, model_save_path, num_epochs=50, batch_size=16, learning_rate=0.001):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset = NoisyImageDataset(image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    print(f"There are {len(dataset)} training samples.")
    
    # Initialize model
    model = DenoisingCNN().to(device)

    # Loss and optimizer
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    train_losses = []

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in tqdm(dataloader, colour='blue', leave=False, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.unsqueeze(1).float().to(device)  # Add channel dimension and convert to float
            targets = targets.unsqueeze(1).float().to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        lr_scheduler.step(running_loss)
        epoch_loss = running_loss / len(dataloader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.10f}, Current Learning Rate: {optimizer.param_groups[0]['lr']}")

    # Save the model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Plot the train losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_plot.png')  # Save the plot as a PNG file
    plt.show()  # Display the plot

def denoise_image(model, image_path, device):
    model.eval()
    with torch.no_grad():
        image = io.imread(image_path)
        image = img_as_float32(image)  # Normalize to [0, 1]
        input_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().to(device)  # Add batch and channel dimensions
        output_tensor = model(input_tensor)
        output_image = output_tensor.squeeze().cpu().numpy()
        return output_image

def main():
    # Paths and parameters
    image_dir = 'data/noisy_images_preprocessed/A01_C_DP_35.0/final_corrected'  # Directory containing final corrected images
    model_save_path = 'denoising_cnn.pth'  # Path to save the trained model
    num_epochs = 200
    batch_size = 16
    learning_rate = 1e-3

    # Train the denoiser
    train_denoiser(image_dir, model_save_path, num_epochs, batch_size, learning_rate)

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingCNN().to(device)
    model.load_state_dict(torch.load(model_save_path))

    # Create inference directory
    inference_dir = f"{image_dir}_inference"
    os.makedirs(inference_dir, exist_ok=True)

    # Run inference on all images in the directory
    for image_file in tqdm(glob(os.path.join(image_dir, '*.png')), desc="Running inference", colour="green"):
        denoised_image = denoise_image(model, image_file, device)
        output_path = os.path.join(inference_dir, os.path.basename(image_file))
        plt.imsave(output_path, denoised_image, cmap='gray')

if __name__ == '__main__':
    main()
