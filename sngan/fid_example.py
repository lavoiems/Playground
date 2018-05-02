#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import os
import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import fid
from scipy.misc import imread
import tensorflow as tf

import torch
from torch.autograd import Variable
import argparse
from Decoder import Decoder
import json
from torch import nn


def generate_samples(batch_size, net, z_size):
    z = Variable(torch.FloatTensor(batch_size, z_size).normal_(0,1)).cuda()
    z_data = ((net(z).data.cpu().numpy() + 1) / 2.) * 255.
    return z_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation scores")
    parser.add_argument('--path', required=True)
    parser.add_argument('--epoch', type=int,
                        help='Which model epoch you want to use. Must be present', default=200)
    args = parser.parse_args()
    stats_path = 'fid_stats.npz' # training set statistics
    model_args = json.load(open(os.path.join(args.path, 'args.json')))
    inception_path = fid.check_or_download_inception(None) # download inception network

    model = Decoder((3, 32, 32), model_args['gen_h_size'], model_args['z_size'], True, nn.ReLU(True), 4).cuda()
    model.load_state_dict(torch.load(os.path.join(model_args['model_path'], 'model[%s].ph' % args.epoch)))
    model.eval()
    print('Generating samples')
    batches = [generate_samples(1000, model, model_args['z_size']) for _ in range(50)]
    images = np.array([sample for batch in batches for sample in batch]).transpose(0, 2, 3, 1)

    # load precalculated training set statistics
    f = np.load(stats_path)
    mu_real, sigma_real = f['mu'][:], f['sigma'][:]
    f.close()

    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mu_gen, sigma_gen = fid.calculate_activation_statistics(images, sess, batch_size=100)

    fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    print("FID: %s" % fid_value)
