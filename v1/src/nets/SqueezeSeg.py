""" SqueezeSeg Model """

import torch.nn as nn
import torch.nn.functional as F

class Conv( nn.Module ):
    def __init__(self,inputs, outputs, stride=1, kernel_size=3, padding=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(inputs, outputs, kernel_size=kernel_size, stride=stride, padding=padding)
    def forward(self, x):
        return F.relu(self.conv(x))

class Fire( nn.Module ):

