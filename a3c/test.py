import time
import numpy as np
from collections import deque

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from envs import create_atari_env
from model import ActorCritic


def test(rank, args, shared_model, counter, vis):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space, args.use_sn)

    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    rewards_sum = np.array([])
    reward_sum = 0
    done = True

    start_time = time.time()

    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())

        value, logit = model((Variable(
            state.unsqueeze(0), volatile=True)))
        prob = F.softmax(logit, 1)
        action = prob.max(1, keepdim=True)[1].data.numpy()

        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        if done:
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))
            rewards_sum = np.append(rewards_sum, [reward_sum])
            vis.line(rewards_sum, win='rewards')
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(60)

        state = torch.from_numpy(state)
