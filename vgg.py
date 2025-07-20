import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data import DataLoader
import os

# Build the model class
class cd_vgg(nn.Module):
    def __init__(self, input_channels=6, num_classes=2):
        super().__init__()

        # Load pretrained VGG16
        self.vgg = vgg16(weights=VGG16_Weights.DEFAULT)

        # Replace the first conv layer to accept 6 input channels instead of 3
        old_conv = self.vgg.features[0]
        new_conv = nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=1, padding=1
        )
        # Copy the weights from the old conv for the first 3 channels
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = old_conv.weight
            new_conv.bias = old_conv.bias

        self.vgg.features[0] = new_conv

        # Replace final layer
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg(x)