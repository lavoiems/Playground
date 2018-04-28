import math
import torch.nn as nn
from torch.autograd import Variable

from loader import load_svhn
from Encoder import Encoder
from modules import View, n_maps2


def conv_decoder(shape, dim_h, z_size, batch_norm, nonlinearity, min_dim):
    model = nn.Sequential()
    n_convolution = int(math.log(shape[1] / min_dim, 2))
    c, x, y = 2 ** n_convolution * dim_h, int(shape[1] / 2 ** n_convolution), int(shape[2] / (2 ** n_convolution))
    model.add_module('lin', nn.Linear(z_size, c * x * y))
    model.add_module('expand', View(-1, c, x, y))
    f_size, stride, pad = 4, 2, 1
    for c in reversed(range(n_convolution)):
        n_maps_in = n_maps2(c + 1, dim_h, dim_h)
        n_maps_out = n_maps2(c, dim_h, dim_h)
        conv = nn.ConvTranspose2d(n_maps_in, n_maps_out, f_size, stride, pad, bias=False)
        name = 'conv_%s_%s' % (n_maps_in, n_maps_out)
        model.add_module(name, conv)
        if batch_norm:
            model.add_module(name + '_bn', nn.BatchNorm2d(n_maps_out))
        model.add_module('%s_%s' % (name, nonlinearity), nonlinearity)
    conv = nn.Conv2d(dim_h, 3, 3, stride=1, padding=1, bias=False)
    model.add_module('conv', conv)
    return model


class Decoder(nn.Module):
    def __init__(self, shape, dim_h, z_size, batch_norm, nonlinearity, min_dim):
        super(Decoder, self).__init__()
        self.decoder = conv_decoder(shape, dim_h, z_size, batch_norm, nonlinearity, min_dim)
        self.decoder.add_module('nonlinearity', nn.Tanh())

    def forward(self, x):
        return self.decoder(x)


if __name__ == '__main__':
    train_loader, test_loader = load_svhn('/Tmp/lavoiems', 32)
    shape = train_loader.dataset.data.shape
    encoder = Encoder(shape, 64, 100, True, None, 'ReLU', 2).cuda()
    decoder = Decoder(shape, 64, 100, True, None, 'ReLU', 2).cuda()
    for train_data in train_loader:
        inputs, labels = Variable(train_data[0].cuda()), Variable(train_data[1].cuda())
        i = encoder(inputs)
        print(decoder(i).data.cpu().numpy().min())
        exit(0)
