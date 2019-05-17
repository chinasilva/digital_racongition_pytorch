import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义网络
class MyMnistNet(nn.Module):

    def __init__(self,):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(24, 48, 5, 1, 2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(48, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(3 * 3 * 64, 3136),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(3136, 10),
        )

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, 3 * 3 * 64)
        output = self.classifier(x)
        return output