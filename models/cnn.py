import torch
import torch.nn as nn
import torch.nn.functional as F


class QsoCNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d()
        self.fc1 = nn.Linear()

    def forward(self, input):
        conv1_res = F.relu(self.conv1(input))
        subsample_res = F.max_pool2d(conv1_res, (2, 2))
