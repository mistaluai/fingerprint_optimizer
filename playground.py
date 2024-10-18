from PIL import Image

from loss.c_loss import ContrastiveLoss
from submodels.feature_extractor import FeatureExtractor
from submodels.fully_connected_predictor import FullyConnectedPredictor
from submodels.noise_extractor import NoiseExtractor
import torchvision.transforms as transforms
import torch
import torch.optim as optim


# f = FeatureExtractor()
# n = NoiseExtractor()
# fc = FullyConnectedPredictor(hidden_layers=3)
#
# transform = transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.ToTensor()
#     ])
#
#
# image = Image.open('/Users/mistaluai/PycharmProjects/fingerprint_noise_optimizer/data/expermintal/formal_image.png').convert('RGB')
# image = transform(image).unsqueeze(0)
#
#
# image = n(image)
# print(image.shape)
# image = f(image)
# print(image.shape)
# image = fc(image)
# print(image.shape)

loss_fn = ContrastiveLoss(margin=1.0)
output1 = torch.randn(8, 512, requires_grad=True)  # Example feature vectors (batch size of 8, 512 features)
output2 = torch.randn(8, 512, requires_grad=True)  # Example feature vectors (batch size of 8, 512 features)
labels = torch.randint(0, 2, (8,))  # Random labels (1 for similar, 0 for dissimilar)

optimizer = optim.Adam([output1, output2], lr=0.001)


for i in range(1000):
    optimizer.zero_grad()
    loss = loss_fn(output1, output2, labels)
    loss.backward()
    optimizer.step()
    print(loss.item())
