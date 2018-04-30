from loader import load_svhn, load_mnist, load_cifar
import time
from Decoder import Decoder
from Encoder import SNEncoder, Encoder
from Gan import Gan
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import sys
import torch
import numpy as np
import argparse
import getpass
import os
import torchvision
import objectives
import json


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


def vizualize(inputs, net, id, z_size, viz, save_path, batch_size):
    z = Variable(torch.FloatTensor(batch_size, z_size).normal_(0,1)).cuda()
    prime = net(z)
    viz.save_images(inputs.data.cpu().numpy(), 8, 8, out_file=os.path.join(save_path, 'real.png'), title='Real image', image_id=id)
    viz.save_images(prime.data.cpu().numpy(), 8, 8, out_file=os.path.join(save_path, 'generated.png'), image_id=id+1, title='Generated image')


def compute_tsne(datas, encoders, labels, dim_l, viz):
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


class MockVisdom():
    def save_train_error(self, *args, **kwargs):
        pass

    def save_images(self, *args, **kwargs):
        imgs = args[0]
        rows = args[1]
        torchvision.utils.save_image((torch.FloatTensor(imgs) + 1) / 2, kwargs['out_file'], nrow=rows)

    def save_tsne(self, *args, **kwargs):
        pass


def prepare_save_path(args):
    if not os.path.isdir(args.save_path):
        print('%s did not exist. Creating it' % args.save_path)
        os.makedirs(args.save_path)
    os.makedirs(args.model_path, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description='SNGAN')
    parser.add_argument('--exp-name', default='test')
    parser.add_argument('--dataset-loc', default='/Tmp/%s' % getpass.getuser())
    parser.add_argument('--server')
    parser.add_argument('--port')
    parser.add_argument('--use-visdom', action='store_true')
    parser.add_argument('--use-sn', action='store_true')
    parser.add_argument('--objective', default='jsgan', choices=['jsgan', 'wgan'])
    parser.add_argument('--use-penalty', default=False, action='store_true')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--test-batch-size', default=64, type=int)
    parser.add_argument('--gen-h-size', default=64, type=int)
    parser.add_argument('--disc-h-size', default=64, type=int)
    parser.add_argument('--z-size', default=128, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--beta1', default=0.5, help='Adam parameter', type=float)
    parser.add_argument('--beta2', default=0.9, help='Adam parameter', type=float)
    parser.add_argument('--n-dis', default=5, help='Number of discriminator interation for each generator interation', type=int)
    parser.add_argument('--root-path', default='/data/milatmp1/%s/sngan' % getpass.getuser())
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    args.save_path = os.path.join(args.root_path, args.exp_name)
    args.model_path = os.path.join(args.save_path, 'model')
    prepare_save_path(args)
    get_gen_loss, get_disc_loss = objectives.losses[args.objective]
    json.dump(vars(args), open(os.path.join(args.save_path, 'args.json'), 'w'))
    if args.use_visdom:
        import viz
        viz.setup(args.server, args.port, env=args.exp_name, use_tanh=True)
    else:
        viz = MockVisdom()
    encoder = SNEncoder if args.use_sn else Encoder
    train_loader, test_loader, shape = load_cifar(args.dataset_loc, args.batch_size, args.test_batch_size)

    gan = Decoder(shape, args.gen_h_size, args.z_size, True, nn.ReLU(True), 4).cuda()
    discriminator = SNEncoder(shape, args.gen_h_size, 1, True, nn.LeakyReLU(0.1, True), 4).cuda()
    gan.apply(weights_init)
    discriminator.apply(weights_init)

    generator_optimizer = optim.Adam(gan.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    t = 0
    for i in range(201):
        print('Epoch: %s' % i)
        gan.train()
        discriminator.train()
        start = time.time()
        for data in train_loader:
            t += 1
            discriminator_optimizer.zero_grad()
            loss = get_disc_loss(data, gan, discriminator, data[0].shape[0], args.z_size, args.use_penalty)
            discriminator_optimizer.step()
            if t == args.n_dis:
                t = 0
                generator_optimizer.zero_grad()
                loss = get_gen_loss(gan, discriminator, data[0].shape[0], args.z_size)
                generator_optimizer.step()

        end = time.time()
        print('Epoch time: %s' % (end - start))
        gan.eval()
        discriminator.eval()

        data, labels = get_data(test_loader)
        vizualize(data, gan, 0, args.z_size, viz, args.save_path, args.batch_size)
        if i % 10 == 0:
            torch.save(gan.state_dict(), os.path.join(args.model_path, 'model[%s].ph' % i))

