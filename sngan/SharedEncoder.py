from torch import nn


class SharedEncoder(nn.Module):
    def __init__(self, shape, shared_encoder, dim_h):
        super(SharedEncoder, self).__init__()
        self.shared_encoder = shared_encoder
        self.initial = nn.Sequential()
        self.initial.add_module('s_conv', nn.Conv2d(shape[0], dim_h, 1, 1, 0, bias=False))
        self.initial.add_module('_bn', nn.BatchNorm2d(dim_h))

    def forward(self, input):
        initial = self.initial(input)
        return self.shared_encoder(initial)
