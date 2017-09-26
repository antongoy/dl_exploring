import argparse

import torch
import torch.nn as nn

from inceptions import InceptionV1


class MiniGoogLeNet(nn.Module):
    def __init__(self):
        self.relu = nn.ReLU(inplace=True)
        self.first_conv = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.inception1 = InceptionV1(32, 16, 16, 16, 8, 16, 16)
        self.inception2 = InceptionV1(64, 32, 16, 32, 8, 32, 32)
        self.inception3 = InceptionV1(128, 64, 32, 64, 16, 64, 64)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception4 = InceptionV1(256, 256, 128, 256, 64, 256, 256)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.relu(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.max_pool(x)
        x = self.inception4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x