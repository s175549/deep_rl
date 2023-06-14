import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from agents.model import Net
from utils.buffer import ReplayBuffer

class DQN(Net):
    def __init__(self, input_size, hidden_size, output_size, alpha, gamma,  
                 eps, eps_decay, eps_end, buffer_size, env, num_frames):
        super(DQN,self).__init__(input_size, hidden_size, output_size, alpha, num_frames)
        self.output_size = output_size
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_end = eps_end

        # Initialize and pre-fill replay buffer
        self.buffer_size = buffer_size
        self.n_init = 5000
        self.transition = namedtuple('Transition', ('id', 'trajectory', 'state', 'action','reward', 'next_state', 'log_prob', 'done'))
        self.buffer = ReplayBuffer(capacity=self.buffer_size, transition=self.transition)
        self.fill_buffer(env=env)

    def scheduler_step(self):
        self.scheduler.step()

    def fill_buffer(self,env):

        state = env.reset()

        for _ in range(self.n_init):

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

    def update_eps(self):
        self.eps = max(self.eps_end,self.eps-self.eps_decay)

    def take_action(self, state):
        state_torch = torch.tensor(state)
        if torch.rand(1) < self.eps:
            action = torch.randint(0, self.output_size, (1,))
        else:
            with torch.no_grad():
                q_values = self.forward(state_torch)
                action = torch.argmax(q_values).view(1)
        return action.numpy()[0], None, None
    
    def update(self, batch_size, num_epochs,entropies=None):
        
        losses = []
        for _ in range(num_epochs):
            # Sample batch from replay buffer
            transitions = self.buffer.sample(batch_size)
            batch = self.transition(*zip(*transitions))

            # Extract from batch 
            action_batch = torch.LongTensor(batch.action)
            reward_batch = torch.FloatTensor(batch.reward)
            next_state_batch = torch.FloatTensor(np.array(batch.next_state))
            done_batch = torch.FloatTensor(batch.done)

            # Compute Q_policy(s_{t+1},argmax_a Q_policy(s_{t+1}, a)) 
            q_values = self(torch.tensor(np.array(batch.state))).gather(1, action_batch.unsqueeze(1)).squeeze(1)
            next_q_values = self(next_state_batch).detach().max(1)[0]

            # Compute the expected DQN values
            expected_q_values = (next_q_values * self.gamma * (1 - done_batch)) + reward_batch

            # Compute MSE loss
            loss = self.criterion(q_values,expected_q_values)

            # Update model using gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.clone().detach().numpy())

        return np.array(losses)