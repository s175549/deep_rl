import os, json, gym, torch
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.get_agent import get_agent
from utils.play_agent import play_agent
from utils.save_experiment import save_experiment
from utils.plot import plot
from train import train

def run_eval(env,method):

    # Get the number of state observations and actions
    n_observations = len(state)
    n_actions = env.action_space.n

    experiment_name = "{}_{}".format(env_name,method)
    experiment_path = save_experiment(experiment_name,method)

    agent_path = os.path.join(experiment_path, 'dqn.pth')

    hyperparameters_path = os.path.join(experiment_path, 'hyperparameters.json')

    # Open the JSON file with hyperparameters
    with open(hyperparameters_path, 'r') as json_file:
        hyperparameters = json.load(json_file)
    
    # Set epsilon to zero
    hyperparameters['eps'] = 0.0
    #hyperparameters['num_hidden'] = [32, 32]

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
    env_name = "LunarLander-v2" #
    env = gym.make(env_name)
    state = env.reset()
    print("Env name: ", env_name)

    # Define learning algorithm
    #method = 'DQN'
    #method = 'REINFORCE'
    #method = 'AC'
    method = 'PPO'

    run_eval(env=env,method=method)