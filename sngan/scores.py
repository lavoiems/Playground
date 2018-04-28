import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import os
from scipy.ndimage import imread
from Decoder import Decoder
from loader import load_cifar

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy


def get_prediction(inception_model, data):
    x = Variable(data[0]).cuda()
    up = nn.Upsample(size=(299, 299), mode='bilinear')
    x = up(x)
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    x = inception_model(x)
    return F.softmax(x, dim=1).data.cpu().numpy()


def inception_score(dataset, batch_size=32, splits=1):
    N = len(dataset)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    inception_model = inception_v3(pretrained=True, transform_input=False).cuda()
    inception_model.eval()

    preds = np.concatenate([get_prediction(inception_model, batch) for batch in dataloader])

    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def generate_samples(batch_size, net, z_size):
    z = Variable(torch.FloatTensor(batch_size, z_size).normal_(0,1)).cuda()
    z_data = net(z).data.cpu()
    return ((sample, 0) for sample in z_data)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluation scores')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--h-size', default=64)
    parser.add_argument('--z-size', default=128)
    args = parser.parse_args()

    model = Decoder((3, 64, 64), args.h_size, args.z_size, True, nn.ReLU(True), 4).cuda()
    model.load_state_dict(torch.load(args.model_path))
    print('Generating samples')
    batches = [generate_samples(1000, model, args.z_size) for _ in range(50)]
    samples = [sample for batch in batches for sample in batch]
    print('Generating Inception score')
    print('Inception score:', inception_score(samples, batch_size=64, splits=10))
