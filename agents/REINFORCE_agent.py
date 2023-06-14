import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple
import numpy as np
from agents.model import Net
from utils.buffer import ReplayBuffer


class REINFORCE(Net):

    def __init__(self, input_size, output_size, hidden_size, alpha, gamma,beta,num_frames,max_norm):
        super(REINFORCE,self).__init__(input_size, hidden_size, output_size, alpha,num_frames)
         # Initialize replay buffer
        self.gamma = gamma
        self.beta = beta
        self.max_norm = max_norm
        self.transition = namedtuple('Transition', ('id', 'trajectory','state', 'action','reward', 'next_state', 'log_prob', 'done')) 
        self.buffer = ReplayBuffer(capacity=0,transition=self.transition)

    def scheduler_step(self):
        self.scheduler.step()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=1)
    
    def take_action(self, state):
        state = torch.from_numpy(state).unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action), m.entropy()
    
    def update(self, rewards, log_probs, entropies):

        # Compute the discounted return
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = self.gamma*G+reward
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32) 

        # Compute the policy loss
        policy_loss = []
        for log_prob, disc_return in zip(log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        # Compute entropy term
        entropy = -torch.mean(torch.tensor(entropies)).detach()
        entropy *= self.beta  # entropy weight
        policy_loss += entropy.squeeze()

        # Take gradient step
        self.optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_norm)
        self.optimizer.step()
        
        return policy_loss.clone().detach().numpy()