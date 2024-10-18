
import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset

class RealFakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir)
                            if f.endswith('.jpg') or f.endswith('.png') and not f.startswith('.DS_Store')]
        self.fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                            if f.endswith('.jpg') or f.endswith('.png') and not f.startswith('.DS_Store')]
        self.transform = transform

        # Create labels: 0 for real and 1 for fake
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        self.images = self.real_images + self.fake_images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        label_tensor = torch.tensor(label)
        return image, label_tensor