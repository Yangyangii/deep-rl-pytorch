# coding: utf-8

import time, random, math, datetime
import numpy as np

import gym

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from collections import deque
from itertools import count
from PIL import Image

from skimage.transform import resize
from skimage.color import rgb2gray

from hyperparams import Hyperparams as hp
from tensorboardX import SummaryWriter

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(hp.NO_SEQ, 16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.head = nn.Linear(math.ceil(hp.HEIGHT/8)*math.ceil(hp.WIDTH/8)*64, NO_ACTIONS)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

def prepro(X):
    x = np.uint8(resize(rgb2gray(X), (hp.HEIGHT, hp.WIDTH), mode='reflect') * 255)
    return np.expand_dims(x, 0)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def get_seq_state(seq_screen):
    assert len(seq_screen) == hp.NO_SEQ
    seq_state = np.concatenate([prepro(screen) for screen in seq_screen])
    state = torch.from_numpy(seq_state).float().to(device).unsqueeze(0)
    return state.to(device)

def select_action(state, action_space):
    global steps_done
    global eps
    action_space[action_space.argmax()] = 0
    sample = random.random()
    eps = hp.EPS_END + (hp.EPS_START - hp.EPS_END) * math.exp(-1. * steps_done / hp.EPS_DECAY)
    steps_done += 1
    # the larger eps_threshold, the more explore
    if sample > eps:
        with torch.no_grad():
            action = policy_net(state).max(1)[1].view(1, 1)
    else:
        action = torch.tensor([random.sample(range(env.action_space.n), 1)], device=device, dtype=torch.long)
    return action_space, action

def optimize_model():
    if len(memory) < hp.BATCH_SIZE:
        return
    transitions = memory.sample(hp.BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of action_space taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(hp.BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states)[0, policy_net(non_final_next_states).argmax(dim=-1)].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * hp.GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def summary(writer, epsilon, score, mean_score):
    global steps_done
    writer.add_scalar("epsilon", epsilon, steps_done)
    writer.add_scalar("score", score, steps_done)
    writer.add_scalar("score_mean", mean_score, steps_done)
    return True
    

if __name__ == '__main__':
    env = gym.make(hp.ENV)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    
    NO_ACTIONS = env.action_space.n

    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    writer = SummaryWriter(comment="-" + hp.ENV + hp.MODEL + "-01")

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    memory = ReplayMemory(hp.NO_REPLAY)

    steps_done = 0
    epsilon = 0.
    
    action_space = torch.zeros(env.action_space.n, dtype=torch.long).to(device)
    skip_frame = 4
    prev_mean = -21.0
    score_mean = 0.
    latest_reward = deque([-21.0], maxlen=10)
    
    f = open('game.log', 'w')
    
    for ep in range(hp.no_episodes):
        # Initialize the environment and state
        seq_screen = deque(maxlen=hp.NO_SEQ)

        screen = env.reset()
        seq_screen.append(screen)

        for _ in range(hp.NO_SEQ-1):
            screen, reward, done, info = env.step(0)
            max_life = info['ale.lives']
            seq_screen.append(screen)

        state = get_seq_state(seq_screen)
        cur_t = 0
        ep_reward = 0

        for t in count():
            action_space, action = select_action(state, action_space)
            screen, reward, done, info = env.step(action)
            seq_screen.append(screen)

            if not done:
                next_state = get_seq_state(seq_screen)
            else:
                next_state = None

            ep_reward += reward
            reward = torch.tensor([reward], dtype=torch.float, device=device)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            if steps_done > hp.NO_REPLAY and steps_done % skip_frame == 0:
                optimize_model()

            if done:
                prev_mean = np.mean(latest_reward)
                latest_reward.append(ep_reward)
                score_mean = np.mean(latest_reward)
                print('EP: {}, Steps: {}, Score: {}, Mean: {:.1f}, Eps: {:.2f}, Cnt: {}, Time: {}'.format(
                    ep, steps_done, ep_reward, score_mean, eps, t, str(datetime.datetime.today())[2:-10]))
                f.write('EP: {}, Steps: {}, Score: {}, Mean: {:.1f}, Eps: {:.2f}, Cnt: {}, Time: `{}\n'.format(
                    ep, steps_done, ep_reward, score_mean, eps, t, str(datetime.datetime.today())[2:-10]))
                
                summary(writer, score=ep_reward, mean_score=score_mean, epsilon=eps)
                break
                
            # Update the target network
            if steps_done % hp.TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
            if steps_done % 50000 == 0 or np.mean(latest_reward) > prev_mean+1:
                save_checkpoint({
                    'epoch': ep+1,
                    'arch': 'dqn_pong',
                    'state_dict': target_net.state_dict(),
                    'mean_reward': score_mean,
                    'optimizer' : optimizer.state_dict(),
                })
                
        if score_mean > hp.stop_score:
            save_checkpoint({
                'epoch': ep+1,
                'arch': 'dqn_pong',
                'state_dict': target_net.state_dict(),
                'mean_reward': score_mean,
                'optimizer' : optimizer.state_dict(),
            })
            break
    
    print('Training Complete')
    f.close()
