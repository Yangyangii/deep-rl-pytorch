
# coding: utf-8

# In[1]:

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


class ValueNet(nn.Module):

    def __init__(self):
        super(ValueNet, self).__init__()
        self.conv1 = nn.Conv2d(hp.NO_SEQ, 16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.linear = nn.Linear(math.ceil(hp.HEIGHT/8)*math.ceil(hp.WIDTH/8)*64, 512)
        self.head = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.linear(x))
        return self.head(x)


# In[4]:

class PolicyNet(nn.Module):

    def __init__(self):
        super(PolicyNet, self).__init__()
        self.conv1 = nn.Conv2d(hp.NO_SEQ, 16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.linear = nn.Linear(math.ceil(hp.HEIGHT/8)*math.ceil(hp.WIDTH/8)*64, 512)
        self.head = nn.Linear(512, NO_ACTIONS)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.linear(x))
        return F.softmax(self.head(x), dim=-1)


# In[5]:

def prepro(X):
    x = np.uint8(resize(rgb2gray(X), (hp.HEIGHT, hp.WIDTH), mode='reflect') * 255)
    return np.expand_dims(x, 0)


# In[6]:

def save_checkpoint(state, filename='-checkpoint.pth.tar'):
    torch.save(state, hp.MODEL + filename)


# In[7]:

def get_seq_state(seq_screen):
    assert len(seq_screen) == hp.NO_SEQ
    seq_state = np.concatenate([prepro(screen) for screen in seq_screen])
    state = torch.from_numpy(seq_state).float().to(device).unsqueeze(0)
    return state.to(device)


# In[8]:

def select_action(state):
    global steps_done
    steps_done += 1
    policy = policy_net(state).view(-1).detach().cpu().numpy()
    policy = np.round(policy, 4) # Float precision
    return np.random.choice(NO_ACTIONS, 1, p=policy)[0]


# In[9]:

def train_model(state, action, reward, next_state=None, done=False):
#     state = torch.tensor(state, device=device, dtype=torch.float)
    reward = torch.tensor([reward], device=device, dtype=torch.float)
    
    state_values = value_net(state)[0, 0]
    fixed_state_values = torch.tensor([state_values], device=device, dtype=torch.float).detach()
    
    if done:
        advantage = reward - fixed_state_values # for actor
        target = reward # for critic
    else:
#         next_state = torch.tensor(next_state, device=device, dtype=torch.float)
        next_state_values = value_net(next_state)[0, 0]
        discounted_reward = reward + hp.GAMMA * next_state_values
        discounted_reward = torch.tensor(discounted_reward, device=device, dtype=torch.float).detach()
        # calculate target
        advantage = discounted_reward - fixed_state_values
        target = discounted_reward
    
    policy_loss = torch.log(policy_net(state)[0, action]) * advantage
    policy_loss = -torch.sum(policy_loss)
    p_opt.zero_grad()
    policy_loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    p_opt.step()
    
#     critic_loss = F.smooth_l1_loss(state_values, discounted_reward)
    critic_loss = F.mse_loss(state_values, target)
    v_opt.zero_grad()
    critic_loss.backward()
    for param in value_net.parameters():
        param.grad.data.clamp_(-1, 1)
    v_opt.step()
    
    return True


# In[10]:

def summary(writer, score, mean_score, epsilon=None):
    global steps_done
    if epsilon is not None:
        writer.add_scalar("epsilon", epsilon, steps_done)
    writer.add_scalar("score", score, steps_done)
    writer.add_scalar("score_mean", mean_score, steps_done)
    return True


# In[14]:

if __name__ == '__main__':
    env = gym.make(hp.ENV)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    NO_ACTIONS = env.action_space.n if hp.MODE == 'normal' else hp.NO_KEYS

#     PolicyGradient = PG().to(device) # Sharing conv
    policy_net = PolicyNet().to(device)
    value_net = ValueNet().to(device)
    p_opt = optim.Adam(policy_net.parameters(), lr=1e-4)
    v_opt = optim.Adam(value_net.parameters(), lr=5e-4)
    
#     skip_frame = 4
    steps_done = 0
    prev_mean = -21.0
    score_mean = 0.
    latest_reward = deque([-21.0], maxlen=10)
    
    writer = SummaryWriter(comment="-" + hp.ENV + hp.MODEL + "-01")
    f = open(hp.MODEL + '-' + str(datetime.datetime.today())[2:10] + '-game.log', 'w')
    
    for ep in range(hp.no_episodes):
        # Initialize the environment and state
        screen = env.reset()
        seq_screen = deque(maxlen=hp.NO_SEQ)
        seq_screen.append(screen)

        for _ in range(hp.NO_SEQ-1):
            screen, reward, done, info = env.step(0)
            max_life = info['ale.lives']
            seq_screen.append(screen)

        state = get_seq_state(seq_screen)
        action = select_action(state)
        ep_reward = 0.
        
        # Play game
        for t in count():
            screen, reward, done, info = env.step(action) if hp.MODE == 'normal' else env.step(action+2)
            seq_screen.append(screen)
            
            if not done:
                next_state = get_seq_state(seq_screen)
                next_action = select_action(next_state)
                train_model(state, action, reward, next_state=next_state)
                # Move to the next state
                state = next_state
                action = next_action
            else:
                train_model(state, action, reward, done=done)
            
            ep_reward += reward

            if done:
                prev_mean = np.mean(latest_reward)
                latest_reward.append(ep_reward)
                score_mean = np.mean(latest_reward)
                print('EP: {}, Steps: {}, Score: {}, Mean: {:.1f}, Cnt: {}, Time: {}'.format(
                    ep, steps_done, ep_reward, score_mean, t, str(datetime.datetime.today())[2:-10]))
                f.write('EP: {}, Steps: {}, Score: {}, Mean: {:.1f}, Cnt: {}, Time: `{}\n'.format(
                    ep, steps_done, ep_reward, score_mean, t, str(datetime.datetime.today())[2:-10]))
                summary(writer, score=ep_reward, mean_score=score_mean)
                break
                
            if steps_done % 20000 == 0:
                save_checkpoint({
                    'epoch': ep+1,
                    'arch': hp.MODEL,
                    'state_dict': policy_net.state_dict(),
                    'mean_reward': score_mean,
                    'optimizer' : p_opt.state_dict(),
                })
                
        if score_mean > hp.stop_score:
            save_checkpoint({
                'epoch': ep+1,
                'arch': hp.MODEL,
                'state_dict': policy_net.state_dict(),
                'mean_reward': score_mean,
                'optimizer' : p_opt.state_dict(),
            })
            break
    
    print('Training Complete')
    f.close()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



