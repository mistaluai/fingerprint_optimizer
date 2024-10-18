import torch.nn as nn

class NoiseExtractor(nn.Module):
    def __init__(self, depth=25, num_features=64, input_channels=3):
        super(NoiseExtractor, self).__init__()
        layers = []
        # First layer
        layers.append(nn.Conv2d(input_channels, num_features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        # Middle layers
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        # Final layer
        layers.append(nn.Conv2d(num_features, input_channels, kernel_size=3, padding=1, bias=False))
        self.denoiser = nn.Sequential(*layers)

    def forward(self, x):
        denoised = self.denoiser(x)
        noise = x - denoised
        return noise