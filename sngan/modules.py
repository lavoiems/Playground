import torch.nn as nn


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


def n_maps(x, dim_h, default):
    return max(2 ** (x - 1), 1) * dim_h if x != 0 else default


def n_maps2(x, dim_h, default):
    return max(2 ** x, 1) * dim_h if x != 0 else default


