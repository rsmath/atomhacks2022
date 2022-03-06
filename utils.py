"""
@author: Ramansh Sharma
PyTorch util functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def repu(x, order=2):
    zero = torch.tensor([0.])
    try:
        zero.cuda()
    except:
        pass
    return torch.pow(torch.maximum(x, zero), order)


class RePU(nn.Module):
    def __init__(self, order=2):
        super().__init__()

        self.order = float(order)
        if self.order < 2:
            self.order = 2

    def forward(self, input):
        return repu(input, order=self.order)
