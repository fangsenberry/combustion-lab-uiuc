import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SyntheticEdgeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(image, 100, 200)
        
        if self.transform:
            image = self.transform(image)
            edges = self.transform(edges)
        
        return image, edges

# Example usage:
# dataset = SyntheticEdgeDataset('data/train_images', transform=transforms.ToTensor())
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

import torch
import segmentation_models_pytorch as smp

# Define the model
model = smp.Unet('resnet34', encoder_weights='imagenet', in_channels=1, classes=1, activation=None)

# Define loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop (simplified)
def train_model(dataloader, model, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for images, edges in dataloader:
            images = images.unsqueeze(1).float()  # Add channel dimension and convert to float
            edges = edges.unsqueeze(1).float()    # Add channel dimension and convert to float
            
            outputs = model(images)
            loss = criterion(outputs, edges)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Example usage:
# train_model(dataloader, model, criterion, optimizer)

def process_and_crop_image_self_supervised(model, image_path, output_dir, margin=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    input_image = transforms.ToTensor()(image).unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimensions
    
    # Predict edges using the model
    with torch.no_grad():
        model.eval()
        edge_pred = model(input_image).squeeze().cpu().numpy()
        edge_pred = (edge_pred > 0.5).astype(np.uint8) * 255  # Binarize the prediction

    # Find contours
    contours, _ = cv2.findContours(edge_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Array to hold cropped images
    cropped_images = []

    # Loop over contours and crop images
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Expand the bounding box by the specified margin
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        
        cropped_image = image[y:y+h, x:x+w]
        cropped_images.append(cropped_image)
        
        # Save each cropped image to the specified directory
        cropped_image_path = os.path.join(output_dir, f'cropped_{i}.png')
        cv2.imwrite(cropped_image_path, cropped_image)
    
    # Display the edges and the original image with contours for verification
    plt.figure(figsize=(10, 10))
    plt.imshow(edge_pred, cmap='gray')
    plt.title('Predicted Edges')
    plt.axis('off')
    plt.show()
    
    image_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_with_contours)
    plt.title('Image with Contours')
    plt.axis('off')
    plt.show()
    
    return cropped_images

# Save this function in a file named `image_processing.py`, and import it in your Jupyter notebook.