import os, json, gym, torch
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.get_agent import get_agent
from utils.play_agent import play_agent
from utils.save_experiment import save_experiment
from utils.plot import plot
from train import train

def run_main(env,method,hyperparameters,train_agent=True):

    if method == "DQN":
        hyperparameters['eps_decay_rate'] = hyperparameters['num_episodes'] * hyperparameters['eps_decay']
    print("Hyperparameters: ", hyperparameters)

    experiment_name = "{}_{}".format(env_name,method)
    experiment_path = save_experiment(experiment_name,method,hyperparameters)

    agent_path = os.path.join(experiment_path, 'dqn.pth')

    if train_agent:

        # Now we initialize our agent
        agent = get_agent(method,input_size=n_observations,output_size=n_actions,
                        hyperparameters=hyperparameters,env=env)
        
        # Call train function to train agent and evaluate performance
        results, loss, rewards, num_frames, lengths = train(env=env,
                        max_frames_per_episode=hyperparameters['max_frames_per_episode'],
                        horizon=hyperparameters['horizon'],
                        max_frames=hyperparameters['num_frames'],
                        agent=agent,
                        num_epochs=hyperparameters['num_epochs'],
                        gamma=hyperparameters['gamma'], 
                        batch_size=hyperparameters['batch_size'],
                        eps_decay_rate=hyperparameters['eps_decay_rate'],
                        agent_name=method,
                        agent_path=agent_path,
                        buffer=agent.buffer)

        # Plot results
        plot(method, experiment_path, results, loss, rewards, num_frames, lengths)
        
    hyperparameters_path = os.path.join(experiment_path, 'hyperparameters.json')

    # Open the JSON file with hyperparameters
    with open(hyperparameters_path, 'r') as json_file:
        hyperparameters = json.load(json_file)
    
    # Set epsilon to zero for DQN
    hyperparameters['eps'] = 0.0

    # Create agent from loaded hyperparameters
    agent = get_agent(method,input_size=n_observations,output_size=n_actions,
            hyperparameters=hyperparameters,env=env)
    
    # Load pretrained agent weights
    agent.load_state_dict(torch.load(agent_path))

    # Watch agent play 10 episodes
    play_agent(env,agent,method)

if __name__=="__main__":

    torch.manual_seed(1234)
    # First, we create our environment
    #env_name = "CartPole-v1"
    env_name = "LunarLander-v2" 
    continuous = False
    env = gym.make(env_name)
    state = env.reset()
    print("Env name: ", env_name)

    # Define learning algorithm
    #method = 'DQN'
    #method = 'REINFORCE'
    #method = 'AC'
    method = 'PPO'
    #method = 'SAC'

    # Get the number of state observations and actions
    n_observations = len(state)
    n_actions = env.action_space.n

    # Here we define the hyperparameters
    hyperparameters = {
        'num_episodes': 200,
        'num_frames': 30000,
        'num_epochs': 4,
        'batch_size': 16,
        'horizon': 64,
        'num_hidden': [128,64],
        'gamma': 0.99,
        'alpha': 3e-4,
        'lambda': 0.95,
        'beta': 0.01,
        'max_norm': 1.0,
        'var_eps': 0.1,
        'tau': 0.001,
        'n_steps': 16,
        'buffer_size': 20000,
        'eps': 1.0,
        'eps_end': 0.001,
        'eps_decay': 0.01,
        'eps_decay_rate': None,
        'continuous': False,
        'max_frames_per_episode': 1000
    }

    run_main(env=env,method=method,hyperparameters=hyperparameters)