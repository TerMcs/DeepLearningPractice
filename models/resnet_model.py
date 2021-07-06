import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        if in_channels != out_channels or stride != 1:
            self.branch1 = nn.Sequential(nn.BatchNorm2d(in_channels),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False))
        else:
            self.branch1 = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU())

        self.brach2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        def forward(self, x):
            x1 = self.brach1(x)
            x2 = self.brach2(x)
            return x1 + x2

class ResNet(nn.Module):
    def __init__(self, in_channels=3, n_features=64, drop_rate=0.2, n_classes=10):
        super().__init__()
        c = [n_features, 2 * n_features, 4 * n_features, 4 * n_features]
        self.prep = nn.Conv2d(in_channels, n_features, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = nn.Sequential(ResidualBlock(c[0], c[0], 1),
                                    ResidualBlock(c[0], c[0], 1))

        self.layer2 = nn.Sequential(ResidualBlock(c[0], c[1], 2),
                                    ResidualBlock(c[1], c[1], 1))

        self.layer3 = nn.Sequential(ResidualBlock(c[1], c[2], 2),
                                    ResidualBlock(c[2], c[2], 1))

        self.layer4 = nn.Sequential(ResidualBlock(c[2], c[3], 2),
                                    ResidualBlock(c[3], c[3], 1))

        self.drop = nn.Dropout(drop_rate)

        self.avgpool = nn.AdaptiveAvgPool2d(4)

        self.linear = nn.Linear(c[3], n_classes, bias=True)


    def forward(self, x):
        out = self.prep(x)
        out = self.layer1(out)
        out = self.drop(x)
        out = self.layer2(out)
        out = self.drop(x)
        out = self.layer3(out)
        out = self.drop(x)
        out = self.layer4(out)
        out = self.avgpool(out).view(out.shape[0], -1)
        out = self.drop(out)
        output = self.linear(out)
        return output
