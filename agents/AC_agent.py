import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from collections import namedtuple
from utils.buffer import ReplayBuffer

class AC(nn.Module):
    def __init__(self, input_size, num_actions, hidden_size, alpha_actor, 
                 alpha_critic, gamma, beta, lambda_, buffer_size, horizon, num_frames, max_norm):
        super(AC, self).__init__()
        torch.manual_seed(64)

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], num_actions),
            nn.ReLU(),
            nn.Softmax(dim=-1)
        )
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha_actor+alpha_actor*0.1)
        self.scheduler = lr_scheduler.StepLR(self.actor_optimizer, step_size=int(num_frames/(200)), gamma=0.9)

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size[0]),
            nn.ReLU(),            
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], 1)
        )
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=alpha_critic)
        self.critic_scheduler = lr_scheduler.StepLR(self.critic_optimizer, step_size=int(num_frames/(200)), gamma=0.9)

        self.gamma = gamma
        self.beta = beta
        self.lambda_ = lambda_
        self.max_norm = max_norm

        # Initialize replay buffer
        self.buffer_size = buffer_size
        self.transition = namedtuple('Transition', ('id', 'trajectory', 'state', 'action','reward', 'next_state', 'log_prob', 'done'))
        self.buffer = ReplayBuffer(capacity=self.buffer_size, transition=self.transition)

    def forward(self, x):
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value

    def scheduler_step(self):
        self.scheduler.step()
        self.critic_scheduler.step()

    def take_action(self, state):
        state = torch.from_numpy(state).unsqueeze(0)
        probs = self.actor(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action).detach(), m.entropy()

    def update(self,batch_size=0,num_epochs=4,entropies=[],batch=None):
        
        if batch_size>0:
            # Sample batch from replay buffer
            transitions = self.buffer.sample(batch_size,rand=False)
            batch = self.transition(*zip(*transitions))

            # Extract from batch 
            states = torch.tensor(np.array(batch.state))
            actions = torch.LongTensor(batch.action)
            rewards = torch.FloatTensor(batch.reward)
            next_states = torch.FloatTensor(np.array(batch.next_state))
            done = torch.FloatTensor(batch.done)
        else:
            states, actions, rewards, next_states, done = map(lambda x: torch.tensor(x), batch)

        # Get the predicted values of the current states and next states
        probs, values = self.forward(states)
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
        
        # Compute actor loss
        probs = probs.gather(1, actions.unsqueeze(1))
        log_probs = torch.log(probs)
        entropy = torch.mean(torch.tensor(entropies))
        actor_loss = -(log_probs.squeeze() * advantages.detach()).mean() + self.beta * entropy

        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_norm)
        self.actor_optimizer.step()


        # Compute critic loss
        critic_loss = nn.MSELoss()(values.squeeze(),td_target.detach())
        entropy = torch.mean(torch.tensor(entropies))
        critic_loss += self.beta*entropy

        # Update critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.max_norm)
        self.critic_optimizer.step()

        self.scheduler_step()

        # Clear buffer
        self.buffer.clear()

        return critic_loss.clone().detach().numpy()+actor_loss.clone().detach().numpy()