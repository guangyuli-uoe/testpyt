import torch
import torch.nn as nn
import torch.nn.functional as F

class DemoCNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DemoCNN, self).__init__()
        '''
            Feature batch shape: torch.Size([8, 3, 224, 224])
            b, c, h, w
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=16, kernel_size=5, stride=1, padding=2),
            # [8, 16, 224, 224]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # [8, 16, 112, 112]
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            # [8, 32, 112, 112]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # [8, 32, 56, 56]
        )

        self.fc1 = nn.Sequential(
            nn.Linear(32*56*56, out_features=out_channel)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x