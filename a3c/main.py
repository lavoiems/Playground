import argparse
import os
import numpy as np

import torch
import torch.multiprocessing as mp

import shared_optimizer
from envs import create_atari_env
from model import ActorCritic
from test import test
from train import train
import time
#import visdom

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=int(time.time()),
                    help='random seed')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='PongDeterministic-v4',
                    help='environment to train on (default: PongDeterministic-v4)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--exp-name', default='main',
                    help='Name of the environment')
parser.add_argument('--save-dir', default='.',
                    help='Root directory where to save results')
parser.add_argument('--backward', default='None',
                    help='Backward procedure')
parser.add_argument('--use-sn', default=False,
                    help='Use spectral normalisation')
parser.add_argument('--server', help='Visdom server')
parser.add_argument('--port', help='Visdom port')


if __name__ == '__main__':
    args = parser.parse_args()
    #vis = visdom.Visdom(server=args.server, port=args.port, env=args.exp_name)
    vis=None

    torch.manual_seed(args.seed)
    env = create_atari_env(args.env_name)
    shared_model = ActorCritic(
        env.observation_space.shape[0], env.action_space, args.use_sn)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = shared_optimizer.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter, vis))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
