import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, multi_label=False):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        if multi_label:
            self.fc1 = nn.Linear(28*28*64, 128)
        else:
            self.fc1 = nn.Linear(64*14*14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = self.dropout1(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        return out


class DeeperCNN(nn.Module):
    def __init__(self, multi_label = False):
        super(DeeperCNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1,  out_channels=16, kernel_size=3, stride=1, padding=1)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.cnn5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.cnn6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        if multi_label:
            self.fc1 = nn.Linear(32*7*7*4, 10)
        else:
            self.fc1 = nn.Linear(32*7*7, 10)

        self.pool_conv = nn.Conv2d(in_channels=16,  out_channels=16, kernel_size=3, stride=2, padding=1)
        self.pool_conv_2 = nn.Conv2d(in_channels=32,  out_channels=32, kernel_size=3, stride=2, padding=1)


    def forward(self, x):
        out = F.relu(self.cnn1(x))
        out = F.relu(self.cnn2(out))
        out = F.relu(self.cnn3(out))
        out = self.pool_conv(out)
        out = F.relu(self.cnn4(out))
        out = F.relu(self.cnn5(out))
        out = F.relu(self.cnn6(out))
        out = self.pool_conv_2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out