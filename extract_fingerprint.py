import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torch.nn import BCEWithLogitsLoss
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from datasets.real_fake_dataset import RealFakeDataset
from loss.c_loss import ContrastiveLoss
from model import FingerprintOptimizer, FreezeGradients
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
# print('|Imports Done|')


# Define transformations for ResNet18
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Declare datasets and dataloaders
real_dir = "data/real"
fake_dir = "data/fake-biggans"

# Split the dataset into training and validation sets
dataset = RealFakeDataset(real_dir=real_dir, fake_dir=fake_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

batch_size = 32
clamper = 5
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# print('|Loader Initialized|')

# Instantiate model, optimizer, and loss function
device = torch.device("mps")
noise_input = torch.empty(batch_size, 3, 224, 224, device=device, requires_grad=True)
# prev_noise_input = torch.zeros_like(noise_input, device=device)
model = FingerprintOptimizer().to(device)
model = FreezeGradients(model)
optimizer = optim.Adam([noise_input], lr=0.001)
# criterion = ContrastiveLoss()
criterion = BCEWithLogitsLoss()
num_epochs = 5

# print('|Parameters and Model Initialized|')


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    train_loader = tqdm(train_loader, desc='Train')
    val_loader = tqdm(val_loader, desc='Validation')

    # Training pass
    batches = 0
    for images, labels in train_loader:
        if batches > clamper:
            break
        batches += 1
        images, labels = images.to(device), labels.to(device).float()

        optimizer.zero_grad()

        outputs = model(noise_input, images)
        outputs = torch.tensor(outputs, requires_grad=True)
        # loss = criterion(outputs[0], outputs[1], labels)
        loss = criterion(outputs.view(32), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    # print(noise_input - prev_noise_input)
    # prev_noise_input = noise_input
    # Validation pass
    model.eval()
    val_loss = 0.0
    batches = 0
    with torch.no_grad():
        for images, labels in val_loader:
            if batches > clamper:
                break
            batches += 1
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(noise_input, images)
            # loss = criterion(outputs[0], outputs[1], labels)
            loss = criterion(outputs.view(32), labels)
            val_loss += loss.item()

    # Print epoch results
    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {running_loss / len(train_loader):.4f}, '
          f'Validation Loss: {val_loss / len(val_loader):.4f}')


output_dir = "output"
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
noise_input_path = os.path.join(output_dir, "optimized_noise.pt")
torch.save(noise_input.cpu(), noise_input_path)  # Save on CPU to avoid issues with device


# Plotting one fingerprint output
plt.figure(figsize=(8, 8))
fingerprint_image = noise_input[0].cpu().detach().numpy().transpose(1, 2, 0)  # Convert to HxWxC
plt.imshow(fingerprint_image)
plt.title('Resulting Fingerprint from Optimization Process')
plt.axis('off')
plt.show()
