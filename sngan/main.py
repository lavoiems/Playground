from loader import load_svhn, load_mnist, load_cifar
from viz import setup, save_images
import time
from Decoder import Decoder
from Encoder import Encoder
from Gan import Gan, Discriminator, SpectralNorm
from SharedEncoder import SharedEncoder
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import sys
import torch
import numpy as np
import viz

batch_size = 64
h_size = 64
z_size = 64


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_data(loader):
    data = iter(loader).next()
    data[0] = 2 * data[0] - 1.
    return Variable(data[0].cuda()), Variable(data[1].cuda())


def get_gen_loss(gan, discriminator, batch_size, z_size):
    criterion = nn.Softplus().cuda()
    z = Variable(torch.Tensor(batch_size, z_size).normal_(0,1)).cuda()
    prime = gan(z)
    fake_out = discriminator(prime)
    fake_loss = criterion(-fake_out).mean()
    fake_loss.backward()

    return fake_loss


def get_disc_loss(train_data, gan, discriminator, batch_size, z_size):
    criterion = nn.Softplus().cuda()
    real = Variable(train_data[0].cuda())
    real = 2 * real - 1.
    real_out = discriminator(real)
    real_loss = criterion(-real_out).mean()
    real_loss.backward()

    z = Variable(torch.FloatTensor(batch_size, z_size).normal_(0,1).cuda())
    prime = gan(z).detach()
    fake_out = discriminator(prime)
    fake_loss = (criterion(-fake_out) + fake_out).mean()
    fake_loss.backward()

    discriminator.restore()
    return real_loss, fake_loss


def vizualize(inputs, net, id):
    z = Variable(torch.FloatTensor(batch_size, z_size).normal_(0,1)).cuda()
    prime = net(z)
    save_images(inputs.data.cpu().numpy(), 8, 8, out_file='/Tmp/lavoiems/test.png', title='Real image', image_id=id)
    save_images(prime.data.cpu().numpy(), 8, 8, out_file='/Tmp/lavoiems/test.png', image_id=id+1, title='Generated image')


def compute_tsne(datas, encoders, labels, dim_l):
    latents = [encoder(data) for data, encoder in zip(datas, encoders)]
    from sklearn.manifold import TSNE
    tsne = TSNE(2, perplexity=40, n_iter=300, init='pca')
    points = sum([latent.data.tolist() for latent in latents], [])
    points = tsne.fit_transform(points)
    idx = 0
    colormap = np.random.rand(dim_l, 3)
    labels_name = map(str, range(dim_l))
    for i, latent in enumerate(latents):
        point = points[idx:idx+len(latent), :]
        idx += len(latent)
        viz.save_tsne(np.array(point).transpose(), labels[i].data.tolist(), colormap, '/Tmp/lavoiems/test.png', id=i,
                      labels_name=labels_name, title='tsne_latent')


if __name__ == '__main__':
    setup(sys.argv[1], sys.argv[2], env='test', use_tanh=True)
    train_loader, test_loader, shape = load_svhn('/Tmp/lavoiems', batch_size, test_batch_size=64)

    gan = Decoder(shape, h_size, z_size, True, None, 'ReLU', 4).cuda()
    discriminator = Discriminator(Encoder(shape, h_size, 1, True, None, 'LeakyReLU', 4, SpectralNorm)).cuda()
    gan.apply(weights_init)
    discriminator.apply(weights_init)

    generator_optimizer = optim.Adam(gan.parameters(), lr=0.0001, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    for i in range(200):
        print('Epoch: %s' % i)
        gan.train()
        discriminator.train()
        start = time.time()
        for data in train_loader:
            discriminator_optimizer.zero_grad()
            loss = get_disc_loss(data, gan, discriminator, data[0].shape[0], z_size)
            discriminator_optimizer.step()
            generator_optimizer.zero_grad()
            loss = get_gen_loss(gan, discriminator, data[0].shape[0], z_size)
            generator_optimizer.step()

        end = time.time()
        print('Epoch time: %s' % (end - start))
        gan.eval()
        discriminator.eval()

        data, labels = get_data(test_loader)
        vizualize(data, gan, 0)
        # compute_tsne([mnist_data, svhn_data], [mnist_encoder, svhn_encoder], [mnist_labels, svhn_labels], 10)

