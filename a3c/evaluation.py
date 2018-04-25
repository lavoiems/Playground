import time
import os
from collections import deque

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from envs import create_atari_env
from model import ActorCritic
import numpy as np


def evaluation(rank, args, shared_model, counter, vis):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space, args.use_sn_critic, args.use_sn_actor,
                        args.use_sn_shared, args.depth)

    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    rewards_sum = np.array([])
    episode_lengths = np.array([])
    actions = []
    episode_length = 0
    n_episode = 0
    all_entropies = np.array([])
    entropies = []
    while True:
        episode_length += 1
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = Variable(torch.zeros(1, 256), volatile=True)
            hx = Variable(torch.zeros(1, 256), volatile=True)
        else:
            cx = Variable(cx.data, volatile=True)
            hx = Variable(hx.data, volatile=True)

        value, logit, (hx, cx) = model((Variable(
            state.unsqueeze(0), volatile=True), (hx, cx)))
        prob = F.softmax(logit, 1)
        log_prob = F.log_softmax(logit, 1)
        entropy = -(log_prob * prob).sum(1, keepdim=True).data.numpy()
        entropies.append(entropy)
        action = prob.max(1, keepdim=True)[1].data.numpy()

        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        actions.append(action[0, 0])

        if done:
            rewards_sum = np.append(rewards_sum, [reward_sum])
            all_entropies = np.append(all_entropies, [np.mean(entropies)])
            print("Steps: %s, Reward: %s, Length %s, entropy %s, Mean reward: %s" %
                  (counter.value, reward_sum, episode_length, all_entropies[-1], rewards_sum.mean()))
            episode_lengths = np.append(episode_lengths, [episode_length])
            vis.line(np.array(rewards_sum), X=np.array(range(len(rewards_sum))), win='rewards', opts={'title': 'Rewards'})
            np.save(os.path.join(args.save_path, 'rewards.npy'), rewards_sum)
            vis.line(np.array(episode_lengths), X=np.array(range(len(episode_lengths))), win='episodes', opts={'title': 'Episodes'})
            np.save(os.path.join(args.save_path, 'episodes.npy'), episode_lengths)
            vis.line(np.array(all_entropies), X=np.array(range(len(all_entropies))), win='entropies', opts={'title': 'Entropy'})
            np.save(os.path.join(args.save_path, 'entropy.npy'), all_entropies)
            vis.histogram(np.array(actions), win='actions', opts={'numbins': env.action_space.n})
            if n_episode % 10 == 0:
                torch.save(shared_model.state_dict(), os.path.join(args.model_path, 'model[%s]' % n_episode))


            reward_sum = 0
            episode_length = 0
            actions = [0]
            state = env.reset()
            time.sleep(60)

        state = torch.from_numpy(state)
