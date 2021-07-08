import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, ):
        super().__init__()

class ResNet(nn.Module):
    def __init__(self, ):
        super().__init__()
        c = [  ]
        self.prep = nn.Conv2d(  )

        self.layer1 = nn.Sequential(ResidualBlock(c[0], c[0], 1),
                                    ResidualBlock(c[0], c[0], 1))

        self.layer2 = nn.Sequential(ResidualBlock(c[0], c[1], 2),
                                    ResidualBlock(c[1], c[1], 1))

        self.layer3 = nn.Sequential(ResidualBlock(c[1], c[2], 2),
                                    ResidualBlock(c[2], c[2], 1))

        self.layer4 = nn.Sequential(ResidualBlock(c[2], c[3], 2),
                                    ResidualBlock(c[3], c[3], 1))

        self.drop = nn.Dropout(drop_rate)

        self.avgpool = nn.AdaptiveAvgPool2d(  )

        self.linear = nn.Linear(c[3], )


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
