from loader import load_svhn
from viz import setup, save_images
import time
from Decoder import Decoder
from Encoder import Encoder
from Autoencoder import Autoencoder
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import sys


if __name__ == '__main__':
    train_loader, test_loader = load_svhn('/Tmp/lavoiems', 32)
    setup(sys.argv[1], sys.argv[2], env='test', use_tanh=True)
    shape = train_loader.dataset.data.shape
    encoder = Encoder(shape, 32, True, None, 'ReLU', 4).cuda()
    decoder = Decoder(shape, 32, True, None, 'ReLU', 4).cuda()
    autoencoder = Autoencoder(encoder, decoder).cuda()
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-4, betas=(0.5, 0.999))
    criterion = nn.MSELoss()

    for i in range(200):
        print('Epoch: %s' % i)
        autoencoder.train()
        start = time.time()
        for train_data in train_loader:
            inputs, labels = Variable(train_data[0].cuda()), Variable(train_data[1].cuda())
            inputs = 2 * inputs - 1.
            prime = autoencoder(inputs)
            loss = criterion(prime, inputs)
            loss.backward()
            optimizer.step()

        end = time.time()
        print('Epoch time: %s' % (end - start))
        autoencoder.eval()

        data = iter(test_loader).next()
        data[0] = 2 * data[0] - 1.
        inputs, labels = Variable(data[0].cuda()), Variable(data[1].cuda())
        prime = autoencoder(inputs)
        loss = criterion(prime, inputs)
        save_images(data[0].cpu().numpy(), 6, 5, out_file='/Tmp/lavoiems/test.png', title='Real image')
        save_images(prime.data.cpu().numpy(), 6, 5, out_file='/Tmp/lavoiems/test.png', image_id=1, title='Generated image')

