# Deep Reinforcement Learning

In this repo all code and results related to the special course 'Deep Reinforcement Learning' is published and described in the following sections. 

## Overview of code

main.py: This is the main file to run all experiments. There is a function 'run_main(env,method,hyperparameters)' where a certain method (DeepRL algorithm) is trained on a specific enviroment with a number of hyperparameters. 
* env: A string 'LunarLander-v2' or 'CartPole-v1'
* method: A string 'DQN', 'REINFORCE', 'AC' or 'PPO'
* hyperparameters: A dictonary with all parameters. An example could be:     
hyperparameters = {
        'num_episodes': 200,
        'num_frames': 500000,
        'num_epochs': 4,
        'batch_size': 32,
        'num_hidden': [128,64],
        'gamma': 0.99,
        'alpha': 1e-3,
        'lambda': 0.9,
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
* eval_agent: boolean indicating whether to train the agent (False) or watch agent play (True)

The agent get initialized with 'get_agent(...)' from utils and then the agent can either be trained or evaluated. If 'eval_agent' = True, the function calls 'train' from 'train.py' which returns the results from the training. The results are plotted and saved in the 'experiments' folder alongside the network parameters.

train.py: 

