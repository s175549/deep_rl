import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import random
from collections import namedtuple
from utils.buffer import ReplayBuffer
from agents.model import Net

class SAC:
    def __init__(self, input_size, output_size, hidden_size, gamma, alpha, tau, lr, max_norm, num_frames, buffer_size, env):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.lr = lr
        self.max_norm = max_norm
        self.buffer_size = buffer_size

        # Q-Networks
        self.q_net1 = Net(input_size, hidden_size, output_size, lr,num_frames)
        self.q_net2 = Net(input_size, hidden_size, output_size, lr,num_frames)
        self.target_q_net1 = Net(input_size, hidden_size, output_size, lr,num_frames)
        self.target_q_net2 = Net(input_size, hidden_size, output_size, lr,num_frames)

        # Target networks
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        # Policy network
        self.policy_net = Net(input_size, hidden_size, output_size, lr,num_frames)

        # Optimizers
        self.q_optimizer = optim.Adam(list(self.q_net1.parameters()) + list(self.q_net2.parameters()), lr=self.lr)
        self.scheduler = lr_scheduler.StepLR(self.q_optimizer, step_size=int(num_frames/2), gamma=0.9)
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.pi_scheduler = lr_scheduler.StepLR(self.policy_optimizer, step_size=int(num_frames/2), gamma=0.9)

        # Initialize replay buffer
        self.buffer_size = buffer_size
        self.transition = namedtuple('Transition', ('state', 'action','reward', 'next_state', 'log_prob', 'done'))
        self.buffer = ReplayBuffer(capacity=self.buffer_size, transition=self.transition)
        self.fill_buffer(env)

    def scheduler_step(self):
        self.scheduler.step()
        self.pi_scheduler.step()

    def fill_buffer(self,env,n_init=5000):

        state = env.reset()

        for _ in range(n_init):

            # Take a random action
            action = env.action_space.sample()
            next_state, reward, terminated, _ = env.step(action)

            # Store the transition in the replay buffer
            self.buffer.store(state, action, reward, next_state, terminated)

            if terminated:
                state = env.reset()
            else:
                state = next_state

        env.close()

    def take_action(self, state):
        state = torch.from_numpy(state).unsqueeze(0)
        with torch.no_grad():
            action_logits = self.policy_net(state)
            action_probs = nn.Softmax(dim=1)(action_logits)
            action_dist = torch.distributions.Categorical(probs=action_probs)
            action = action_dist.sample()
        return action.item(), None, None
    
    def update(self, batch_size, num_epochs, entropies=None):

        # Sample batch from replay buffer
        transitions = self.buffer.sample(batch_size,rand=True)
        batch = self.transition(*zip(*transitions))

        # Extract from batch 
        states = torch.tensor(np.array(batch.state))
        actions = torch.LongTensor(batch.action).unsqueeze(1)
        rewards = torch.FloatTensor(batch.reward)
        next_states = torch.FloatTensor(np.array(batch.next_state))
        done = torch.FloatTensor(batch.done)

        # Compute Q-values
        q1 = self.q_net1(states).gather(1, actions)
        q2 = self.q_net2(states).gather(1, actions)

        # Compute target Q-values
        with torch.no_grad():
            next_action_probs = nn.Softmax(dim=1)(self.policy_net(next_states))
            next_action_dist = torch.distributions.Categorical(probs=next_action_probs)
            next_actions = next_action_dist.sample().unsqueeze(1)
            next_log_probs = torch.log(next_action_probs.gather(1,next_actions))
            next_q1 = self.target_q_net1(next_states).gather(1, next_actions)
            next_q2 = self.target_q_net2(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + self.gamma * (1 - done.unsqueeze(1)) * (torch.min(next_q1, next_q2)-self.alpha*next_log_probs)
        
        # Compute the Q-loss
        q1_loss = nn.MSELoss()(q1, target_q.detach())
        q2_loss = nn.MSELoss()(q2, target_q.detach())
        q_loss = q1_loss + q2_loss

        # Update the Q-networks
        self.q_optimizer.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.q_net1.parameters(), max_norm=self.max_norm)
        nn.utils.clip_grad_norm_(self.q_net2.parameters(), max_norm=self.max_norm)
        self.q_optimizer.step()

        # Compute policy loss
        action_probs = nn.Softmax(dim=1)(self.policy_net(states))
        log_probs = torch.log(action_probs.gather(1,actions))
        q1_pi = self.q_net1(states).gather(1, actions)
        q2_pi = self.q_net2(states).gather(1, actions)
        policy_loss = (torch.min(q1_pi,q2_pi) - self.alpha * log_probs).mean()

        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=self.max_norm)
        self.policy_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return q_loss.clone().detach().numpy()+policy_loss.clone().detach().numpy()