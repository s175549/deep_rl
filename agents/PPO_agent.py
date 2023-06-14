import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from collections import namedtuple
from utils.buffer import ReplayBuffer
from agents.AC_agent import AC

class PPO(AC):
    def __init__(self, input_size, num_actions, hidden_size, 
                 alpha, gamma, beta, lambda_, var_eps, buffer_size, horizon, num_frames, max_norm):
        super(PPO, self).__init__(input_size, num_actions, hidden_size, alpha, 
                 alpha, gamma, beta, lambda_, buffer_size, horizon, num_frames, max_norm)
        self.var_eps = var_eps
        self.horizon = horizon

        # Initialize replay buffer
        self.buffer_size = buffer_size
        self.transition = namedtuple('Transition', ('id', 'trajectory', 'state', 'action','reward', 'next_state', 'log_prob', 'done'))
        self.buffer = ReplayBuffer(capacity=self.buffer_size, transition=self.transition)

    def compute_advantage(self,batch):

        # Extract from trajectory
        states = torch.tensor(np.array(batch.state))
        rewards = torch.FloatTensor(batch.reward)
        next_states = torch.FloatTensor(np.array(batch.next_state))
        done = torch.FloatTensor(batch.done)

        # Get the predicted values of the current states and next states
        _, values = self.forward(states)
        _, next_values = self.forward(next_states)

        # Compute the TD target
        td_target = rewards + self.gamma * next_values.squeeze().detach() * (1 - done.float())

        # Compute the TD error and advantages
        td_error = td_target - values.squeeze()
        advantages = torch.zeros_like(rewards)
        acc = 0
        for t in reversed(range(len(rewards))):
            acc = td_error[t] + self.gamma * self.lambda_ * acc * (1 - done[t].float())
            advantages[t] = acc

        return advantages.detach()
    
    def update(self,batch_size=0,num_epochs=4,entropies=[],batch=None):
        
        # Sample trajectory from replay buffer
        transitions = self.buffer.sample(batch_size,rand=False)
        batch = self.transition(*zip(*transitions))

        # Create minibatches
        minibatches = [i for i in range(0, batch_size+1, 32)]
        if minibatches[-1] != batch_size:
            minibatches.append(batch_size)
        
        # Extract step-size alpha
        #alpha = self.scheduler._last_lr[0]/self.scheduler.base_lrs[0]
        alpha = 1.0
        
        # Compute generalized advantage based on trajectory of len T
        advantages = self.compute_advantage(batch)

        # Loop over epochs
        for _ in range(num_epochs):

            # Loop over mini_batches
            for i, m in enumerate(minibatches[1:]):
                
                # Extract from batch 
                states = torch.tensor(np.array(batch.state))[minibatches[i]:m]
                actions = torch.LongTensor(batch.action)[minibatches[i]:m]
                rewards = torch.FloatTensor(batch.reward)[minibatches[i]:m]
                next_states = torch.FloatTensor(np.array(batch.next_state))[minibatches[i]:m]
                log_prob_old = torch.tensor(batch.log_prob)[minibatches[i]:m]
                done = torch.FloatTensor(batch.done)[minibatches[i]:m]

                # Get the predicted values of the current states and next states
                probs, values = self.forward(states)
                _, next_values = self.forward(next_states)

                # Compute the TD target
                td_target = rewards + self.gamma * next_values.squeeze().detach() * (1 - done.float())
                
                advantages_batch = advantages[minibatches[i]:m]
                log_probs = torch.log(probs.gather(1,actions.unsqueeze(1)))

                # Clip the policy distribution
                ratio = torch.exp(log_probs.squeeze() - log_prob_old)
                policy_loss_1 = ratio * advantages_batch
                policy_loss_2 = torch.clamp(ratio, 1 - self.var_eps*alpha, 1 + self.var_eps*alpha) * advantages_batch
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Update actor network
                self.actor_optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_norm)
                self.actor_optimizer.step()

                # Calculate the value loss and entropy bonus
                critic_loss = nn.MSELoss()(values.squeeze(),td_target.detach())
                entropy = torch.mean(torch.tensor(entropies))
                critic_loss += self.beta*entropy

                # Update the network weights
                self.critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.max_norm)
                self.critic_optimizer.step()

            self.scheduler_step()

        # Clear buffer
        self.buffer.clear()

        return policy_loss.clone().detach().numpy()+critic_loss.clone().detach().numpy()