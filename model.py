import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelMLP(nn.Module):

    def __init__(self):
        super(ModelMLP, self).__init__()
        self.fc0 = nn.Linear(784, 256)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, input_):
        h1 = F.relu(self.fc0(input_))
        h1 = F.dropout(h1, p=0.5, training=self.training)  # drop rate 0.25, keep rate 0.75
        h2 = F.relu(self.fc1(h1))
        h2 = F.dropout(h2, p=0.5, training=self.training)
        h3 = self.fc2(h2)
        return h3


class ModelCNN(nn.Module):

    def __init__(self):
        super(ModelCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (5, 5))
        self.conv2 = nn.Conv2d(16, 32, (5, 5))
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, input_):
        h1 = F.relu(F.max_pool2d(self.conv1(input_), 2))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(F.max_pool2d(self.conv2(h1), 2))
        h2 = F.dropout(h2, p=0.5, training=self.training)
        h2 = h2.view(-1, 512)
        h3 = F.relu(self.fc1(h2))
        h3 = F.dropout(h3, p=0.5, training=self.training)
        h4 = self.fc2(h3)
        return h4
