import torch
import torch.nn as nn


class Branch1x1(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Branch1x1, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(inplanes, outplanes, kernel_size=1)

    def forward(self, x):
        return self.activation(self.conv1x1(x))


class Branch3x3(nn.Module):
    def __init__(self, inplanes, squeeze_planes, outplanes):
        super(Branch3x3, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.conv3x3 = nn.Conv2d(squeeze_planes, outplanes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.activation(self.conv1x1(x))
        return self.activation(self.conv3x3(x))


class Branch5x5(nn.Module):
    def __init__(self, inplanes, squeeze_planes, outplanes):
        super(Branch5x5, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.conv5x5 = nn.Conv2d(squeeze_planes, outplanes, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.activation(self.conv1x1(x))
        return self.activation(self.conv5x5(x))


class BranchMaxPool(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(BranchMaxPool, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(inplanes, outplanes, kernel_size=1)

    def forward(self, x):
        x = self.max_pool(x)
        return self.activation(self.conv1x1(x))


class InceptionV1(nn.Module):
    def __init__(self, inplanes, expand1x1_planes, squeeze3x3_planes, expand3x3_planes,
                 squeeze5x5_planes, expand5x5_planes, expand_max_pool_planes):
        super(InceptionV1, self).__init__()
        self.branch1x1 = Branch1x1(inplanes, expand1x1_planes)
        self.branch3x3 = Branch3x3(inplanes, squeeze3x3_planes, expand3x3_planes)
        self.branch5x5 = Branch5x5(inplanes, squeeze5x5_planes, expand5x5_planes)
        self.branch_max_pool = BranchMaxPool(inplanes, expand_max_pool_planes)

    def forward(self, x):
        return torch.cat([
            self.branch1x1(x),
            self.branch3x3(x),
            self.branch5x5(x),
            self.branch_max_pool(x)
        ], 1)