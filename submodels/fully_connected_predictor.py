import torch.nn as nn
from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle


class FullyConnectedPredictor(nn.Module):
    def __init__(self, feature_size=1024, hidden_layers=1):
        super(FullyConnectedPredictor, self).__init__()

        self.layers = nn.Sequential()
        i = 1

        self.layers.append(nn.Linear(feature_size, feature_size // (2 ** i)))
        self.layers.append(nn.ReLU())

        for _ in range(hidden_layers):
            i += 1
            self.layers.append(
                nn.Linear(
                    feature_size // (2 ** (i-1)),
                    feature_size // (2 ** i)
                )
            )
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(feature_size // (2 ** i), 1))

    def forward(self, x):
        return self.layers(x)