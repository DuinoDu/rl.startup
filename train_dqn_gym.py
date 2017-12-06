#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import argparse
import numpy as np
from collections import namedtuple
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T


############
# parameters
############

def str2bool(i):
    return i in ['1', 'y', 't']

parser = argparse.ArgumentParser(description='Train DQN in gym.')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--gamma', default=0.999, type=float, help='discounted rate')
parser.add_argument('--eps_start', default=0.9, type=float, help='eps start')
parser.add_argument('--eps_end', default=0.05, type=float, help='eps end')
parser.add_argument('--eps_decay', default=200, type=int, help='eps decay')
parser.add_argument('--memory_size', default=10000, type=int, help='replay memory size')
parser.add_argument('--num_episodes', default=10, type=int, help='nums of episodes')
parser.add_argument('--cuda', default=1, type=str2bool, help='if using gpu')
parser.add_argument('--gpuID', default=0, type=int, help='use which gpu')
args = parser.parse_args()

use_cuda = torch.cuda.is_available() and args.cuda
if use_cuda:
    torch.cuda.set_device(int(args.gpuID))
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


############
# enviroment
############
env = gym.make('CartPole-v0').unwrapped
screen_width = 600

def get_cart_location():
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)

resize = T.Compose([T.ToPILImage(),
                    T.Scale(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen():
    screen = env.render(mode='rgb_array').transpose((2,0,1))
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = get_cart_location()

    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2, 
                            cart_location + view_width // 2)

    screen = screen[:, :, slice_range] # to get a square image centered on the cart
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0).type()


#################
# data (dsX, dsY)
#################

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    """Replay memory described in Deepmind NIPS2013 paper"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

memory = ReplayMemory(args.memory_size)

#############
# model [DQN]
#############

class DQN(nn.Module):
    """
    Deep Q-Network
    """ 

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

model = DQN()
if use_cuda:
    model.cuda()

# cost
# None

# optimizer
optimizer = optim.RMSprop(model.parameters())


# summary


# train
steps_done = 0
episode_duration = []
last_sync = 0
for i_episode in range(args.num_episodes):
    env.reset()
    # fetch init state
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen()

    for t in count():
        action = select_action()



# evaluate
