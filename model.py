import torch
import torch.nn as nn

from submodels.feature_extractor import FeatureExtractor
from submodels.fully_connected_predictor import FullyConnectedPredictor
from submodels.noise_extractor import NoiseExtractor


class FingerprintOptimizer(nn.Module):
    def __init__(self, noise_depth=25, input_channels=3):
        super(FingerprintOptimizer, self).__init__()
        self.noise_extractor = NoiseExtractor(depth=noise_depth, input_channels=input_channels)
        self.feature_extractor = FeatureExtractor()

        self.fc = FullyConnectedPredictor(hidden_layers=3)
    def forward(self, x1, x2):
        x1 = self.feature_extractor(x1)
        x2 = self.feature_extractor(self.noise_extractor(x2))
        x = torch.cat((x1, x2), dim=1)
        return self.fc(x)


class FreezeGradients(nn.Module):
    def __init__(self, module):
        super(FreezeGradients, self).__init__()
        self.module = module

    def forward(self, x1, x2):
        with torch.no_grad():
            return self.module(x1, x2)
