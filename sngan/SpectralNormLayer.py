from torch import nn
import torch.nn.functional as F
import torch


def l2normalize(v, esp=1e-8):
    return v / (v.norm() + esp)


def sn_weight(weight, u, height, n_power_iterations):
    for _ in range(n_power_iterations):
        v = l2normalize(torch.mv(weight.view(height, -1).data.t(), u))
        u = l2normalize(torch.mv(weight.view(height, -1).data, v))

    sigma = u.dot(weight.data.view(height, -1).mv(v))
    return torch.div(weight, sigma), u


class SNConv2d(nn.Conv2d):
    def __init__(self, *args, n_power_iterations=1, **kwargs):
        super(SNConv2d, self).__init__(*args, **kwargs)
        self.n_power_iterations = n_power_iterations
        self.height = self.weight.data.shape[0]
        self.register_buffer('u', l2normalize(self.weight.data.new(self.height).normal_(0, 1)))

    def forward(self, input):
        w_sn, self.u = sn_weight(self.weight, self.u, self.height, self.n_power_iterations)
        return F.conv2d(input, w_sn, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class SNLinear(nn.Linear):
    def __init__(self, *args, n_power_iterations=1, **kwargs):
        super(SNLinear, self).__init__(*args, **kwargs)
        self.n_power_iterations = n_power_iterations
        self.height = self.weight.data.shape[0]
        self.register_buffer('u', l2normalize(self.weight.data.new(self.height).normal_(0, 1)))

    def forward(self, input):
        w_sn, self.u = sn_weight(self.weight, self.u, self.height, self.n_power_iterations)
        return F.linear(input, w_sn, self.bias)
