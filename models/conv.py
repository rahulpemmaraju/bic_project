import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
import snntorch.surrogate as surrogate

from typing import Union, List, Dict

class TestCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TestCNN, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, 5, padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 8, 3, padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 16, 3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.out = nn.Linear(32 * 16 * 16, out_channels)
        
    def forward(self, x):

        x = self.conv_1(x)
        x = F.max_pool2d(x, 2)

        x = self.conv_2(x)
        x = F.max_pool2d(x, 2)

        x = self.conv_3(x)
        x = F.max_pool2d(x, 2)

        x = x.reshape(x.shape[0], -1)
        x = self.out(x)

        return x

if __name__ == '__main__':
    net = TestCNN(1, 1)
    x = torch.randn((4, 1, 128, 128))

    y = net(x)
    print(y.shape)
