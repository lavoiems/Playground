from __future__ import print_function

import argparse
import os
import json

import torch
import torch.multiprocessing as mp
import numpy as np

import optim
from envs import create_atari_env
from model import ActorCritic
from evaluation import evaluation
from train import train


def parse_arg():
    parser = argparse.ArgumentParser(description='A3C')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy', type=float, default=0.001)
    parser.add_argument('--value-loss', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--num-processes', type=int, default=4)
    parser.add_argument('--num-steps', type=int, default=20)
    parser.add_argument('--max-episode-length', type=int, default=1000000)
    parser.add_argument('--num-episodes', type=int, default=200)
    parser.add_argument('--env-name', default='PongDeterministic-v4')
    parser.add_argument('--no-shared', default=False)
    parser.add_argument('--use-sn-critic', default=False, type=bool)
    parser.add_argument('--use-sn-actor', default=False, type=bool)
    parser.add_argument('--use-sn-shared', default=False, type=bool)
    parser.add_argument('--depth-actor', default=0, type=int)
    parser.add_argument('--depth-critic', default=0, type=int)
    parser.add_argument('--use-visdom', default=False, type=bool)
    parser.add_argument('--server', help='Visdom server')
    parser.add_argument('--port', help='Visdom port')
    parser.add_argument('--exp-name', default='main')
    parser.add_argument('--root-path', default='.')
    return parser.parse_args()


def prepare_save_path(args):
    if not os.path.isdir(args.save_path):
        print('%s did not exist. Creating it' % args.save_path)
        os.makedirs(args.save_path)
    os.makedirs(args.model_path, exist_ok=True)


class MockVisdom():
    def text(self, *args, **kwargs):
        pass

    def line(self, *args, **kwargs):
        pass

    def histogram(self, *args, **kwargs):
        pass

if __name__ == '__main__':
    args = parse_arg()
    args.save_path = os.path.join(args.root_path, args.exp_name)
    args.model_path = os.path.join(args.save_path, 'model')
    if args.use_visdom:
        import visdom
        vis = visdom.Visdom(server=args.server, port=args.port, env=args.exp_name)
    else:
        vis = MockVisdom()
    vis.text(repr(args), win='args')
    prepare_save_path(args)

    json.dump(vars(args), open(os.path.join(args.save_path, 'args.json'), 'w'))
    torch.manual_seed(args.seed)
    env = create_atari_env(args.env_name)
    shared_model = ActorCritic(
        env.observation_space.shape[0], env.action_space, args.use_sn_critic, args.use_sn_actor,
        args.use_sn_shared, args.depth_actor, args.depth_critic)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []

#    mp.set_start_method('spawn')
    counter = mp.Value('i', 0)
    n_episodes = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=evaluation, args=(args.num_processes, args, shared_model, counter, n_episodes, vis, optimizer))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, counter, n_episodes, lock, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
