import math
import torch.nn as nn
from torch.autograd import Variable

from modules import View, n_maps


class Classifier(nn.Module):
    def __init__(self, dim_in, dim_h, batch_norm, dropout, desired_nonlinearity):
        super(Classifier, self).__init__()
        assert hasattr(nn, desired_nonlinearity), '%s is not a valid nonlinearity' % desired_nonlinearity
        nonlinearity = getattr(nn, desired_nonlinearity)()
        model = nn.Sequential()
        dim = dim_in
        for i, h in enumerate(dim_h):
            model.add_module('Dense_%s' %i, nn.Linear(dim, h))
            dim = h
            print(dim)
            if batch_norm:
                model.add_module('BatchNorm_%s' %i, nn.BatchNorm1d(dim))
            if dropout:
                model.add_module('Dropout_%s' %i, nn.Dropout1d(p=dropout))
            model.add_module('Nonlinearity_%s' %i, nonlinearity)
        print(dim)
        model.add_module('Output', nn.Linear(dim, 1))
        model.add_module('Sigmoid', nn.Sigmoid())
        self.classifier = model

    def forward(self, x):
        return self.classifier(x)

