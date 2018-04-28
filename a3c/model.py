import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from SpectralNormLayer import SNConv2d, SNLinear


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, action_space, use_sn_critic, use_sn_actor, use_sn_shared, depth_actor, depth_critic):
        super(ActorCritic, self).__init__()
        if use_sn_critic:
            LinearCritic = SNLinear
        else:
            LinearCritic = nn.Linear
        if use_sn_actor:
            LinearActor = SNLinear
        else:
            LinearActor = nn.Linear
        if use_sn_shared:
            Conv2d = SNConv2d
        else:
            Conv2d = nn.Conv2d

        self.conv1 = Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)

        num_outputs = action_space.n
        self.critic_linear = nn.Sequential()
        self.actor_linear = nn.Sequential()
        for i in range(depth_actor):
            self.critic_linear.add_module('critic_linear_%s' % i, LinearCritic(256, 256))
            self.critic_linear.add_module('critic_relu_%s' % i, nn.ReLU())
        for i in range(depth_critic):
            self.actor_linear.add_module('actor_linear_%s' % i, LinearActor(256, 256))
            self.actor_linear.add_module('actor_relu_%s' % i, nn.ReLU())

        self.critic_linear.add_module('critic', LinearCritic(256, 1))
        self.actor_linear.add_module('actor', LinearActor(256, num_outputs))

        self.apply(weights_init)
        #self.actor_linear.weight.data = normalized_columns_initializer(
        #    self.actor_linear.weight.data, 0.01)
        #self.actor_linear.bias.data.fill_(0)
        #self.critic_linear.weight.data = normalized_columns_initializer(
        #    self.critic_linear.weight.data, 1.0)
        #self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, 32 * 3 * 3)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
