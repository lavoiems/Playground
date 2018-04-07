import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch


class Gan(nn.Module):
    def __init__(self, generator):
        super(Gan, self).__init__()
        self.generator = generator

    def forward(self, x):
        generated = self.generator(x)
        return F.tanh(generated)
