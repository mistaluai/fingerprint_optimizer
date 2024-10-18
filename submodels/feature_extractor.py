import torch
import torch.nn as nn
from torchvision import models

resnet18 = models.resnet18(pretrained=True)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-1])

    def forward(self, x):
        features = self.resnet18(x)
        return features.view(features.size(0), -1)