import torch.nn as nn
import torch.nn.functional as F
from operator import mul
from torch.autograd import Variable
import torch


def l2normalize(v, esp=1e-8):
    return v / (v.norm() + esp)


class SpectralNorm(nn.Module):
    def __init__(self, module, n_power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.n_power_iterations = n_power_iterations
        w = self.module.weight
        height = w.data.shape[0]
        width = reduce(mul, w.data.shape[1:], 1)

        u = Variable(w.data.new(height).normal_(0, 1)).cuda()
        v = Variable(w.data.new(width).normal_(0, 1)).cuda()
        u = l2normalize(u)
        v = l2normalize(v)

        self.u = u
        self.v = v
        self.old_w = module.weight.data

    def update(self):
        height = self.module.weight.data.shape[0]
        for _ in range(self.n_power_iterations):
            self.v.data = l2normalize(torch.mv(self.module.weight.view(height, -1).data.t(), self.u.data))
            self.u.data = l2normalize(torch.mv(self.module.weight.view(height, -1).data, self.v.data))

        sigma = self.u.dot(self.module.weight.view(height, -1).mv(self.v))
        self.old_w = self.module.weight.data
        self.module.weight.data = self.module.weight.data / sigma.expand_as(self.module.weight.data).data

    def restore(self):
        self.module.weight.data = self.old_w

    def forward(self, *args):
        self.update()
        return self.module(*args)


class Gan(nn.Module):
    def __init__(self, generator):
        super(Gan, self).__init__()
        self.generator = generator

    def forward(self, x):
        generated = self.generator(x)
        return F.tanh(generated)


def restore(m):
    classname = m.__class__.__name__
    if classname.find('SpectralNorm') != -1:
        m.restore()


class Discriminator(nn.Module):
    def __init__(self, encoder):
        super(Discriminator, self).__init__()
        self.encoder = encoder

    def forward(self, x):
        encoded = self.encoder(x)
        return F.sigmoid(encoded)

    def restore(self):
        self.apply(restore)

