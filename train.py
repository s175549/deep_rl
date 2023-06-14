import torch
import numpy as np
from tqdm import tqdm

def train(env,max_frames_per_episode,horizon,max_frames,agent,num_epochs=1,gamma=1,batch_size=0,
          eps_decay_rate=0,agent_name='DQN',agent_path='',buffer=None):

    # Get initial state
    init_state = env.reset()
    returns = []
    all_rewards = []
    loss = []
    lengths = []
    best_return = 0
    num_frames = 0
    i = 0

    def print_status():
        print("Episode: ", i)
        print("Frame: {} of {}".format(num_frames,max_frames))
        if agent_name == "DQN":
            print("Eps: ", round(agent.eps,3))
        print("Moving average return: ", round(np.mean(returns[-30:]),3))
        print("Moving lr: ", round(agent.scheduler._last_lr[0],5))

    state = init_state
    while num_frames < max_frames:
        terminated = False

        rewards = []
        log_probs = []
        entropies = []
        loss_episode = []
        reward_episode = 0
        u = 0
        # Generate an episode
        while not terminated:

            # Take an action
            action, log_prob, entropy = agent.take_action(state)
            log_probs.append(log_prob)
            entropies.append(entropy)
            
            # Observe next_state, reward and if terminated
            next_state, reward, terminated, _ = env.step(action)
            rewards.append(reward)
            all_rewards.append(reward)
            reward_episode += reward

            # Store in replay buffer
            buffer.store(u, i, state, action, reward, next_state, log_prob, terminated)

            # Update agent 
            if agent_name != 'REINFORCE':

                # Only update PPO or AC if the buffer is 'large enough' (T or terminated)
                if (buffer.size() == horizon) or (terminated==True) & (agent_name in ['PPO', 'AC']):
                    M = horizon if not terminated else len(rewards)
                    l = agent.update(batch_size=M,num_epochs=num_epochs,entropies=entropies)
                    loss_episode.append(l)
                    entropies = []
                    rewards = []

                # We can update DQN each timestep by sampling from buffer since it is off-policy 
                elif agent_name == "DQN":
                    M = batch_size if not terminated else len(rewards)
                    l = agent.update(batch_size=M,num_epochs=num_epochs,entropies=entropies)
                    loss_episode.append(l)
            
            # Transition to next state
            state = next_state
            u += 1

            # Check if we have reach maximum number of frames
            if u == max_frames_per_episode:
                terminated = True

            if ((u+num_frames) % (int(max_frames/50))) == 0:
                print_status()

        num_frames += u
        lengths.append(u)

        # Update REINFORCE agent
        if agent_name == "REINFORCE":
            l = agent.update(rewards,log_probs,entropies)
            loss_episode.append(l)

        # Decay epsilon (DQN)
        if eps_decay_rate:
            if (i+1) % eps_decay_rate == 0:
                agent.update_eps()

        returns.append(reward_episode)
        loss.append(np.mean(loss_episode))

        # Check if we should save the agent weights based on last 10 episodes 
        if np.mean(returns[-10:]) > best_return:
            best_return = np.mean(returns[-10:])
            torch.save(agent.state_dict(), agent_path)

        # Reset enviroment to generate new episode
        state = env.reset()
        i += 1

    return returns, loss, all_rewards, num_frames, lengths