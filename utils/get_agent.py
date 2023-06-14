from agents.DQN_agent import DQN
from agents.REINFORCE_agent import REINFORCE
from agents.AC_agent import AC
from agents.PPO_agent import PPO
from agents.SAC_agent import SAC

def get_agent(agent_name,input_size,output_size,hyperparameters,env):
    agent = None
    if agent_name == 'DQN':
        agent = DQN(input_size=input_size,hidden_size=hyperparameters['num_hidden'],
            output_size=output_size,alpha=hyperparameters['alpha'],  
            eps=hyperparameters['eps'],eps_decay=hyperparameters['eps_decay'],
            eps_end=hyperparameters['eps_end'], gamma=hyperparameters['gamma'],
            buffer_size = hyperparameters['buffer_size'], env=env,
            num_frames=hyperparameters['num_frames'])
    elif agent_name == 'REINFORCE':
        agent = REINFORCE(input_size=input_size,hidden_size=hyperparameters['num_hidden'],
            output_size=output_size,alpha=hyperparameters['alpha'],
            gamma=hyperparameters['gamma'], beta=hyperparameters['beta'],
            num_frames=hyperparameters['num_frames'], max_norm= hyperparameters['max_norm'])
    elif agent_name == 'AC':
        agent = AC(input_size=input_size,num_actions=output_size,
                   hidden_size=hyperparameters['num_hidden'],
                   alpha_actor=hyperparameters['alpha'],alpha_critic=hyperparameters['alpha'],
                   gamma=hyperparameters['gamma'],beta=hyperparameters['beta'],
                   lambda_=hyperparameters['lambda'],buffer_size=hyperparameters['buffer_size'],
                   horizon=hyperparameters['horizon'],num_frames=hyperparameters['num_frames'], 
                   max_norm= hyperparameters['max_norm'])
    elif agent_name == 'PPO':
        agent = PPO(input_size=input_size,num_actions=output_size,
                    hidden_size=hyperparameters['num_hidden'],
                    alpha=hyperparameters['alpha'],gamma=hyperparameters['gamma'],
                    beta=hyperparameters['beta'],lambda_=hyperparameters['lambda'],
                    var_eps=hyperparameters['var_eps'],buffer_size=hyperparameters['buffer_size'],
                    horizon=hyperparameters['horizon'],num_frames=hyperparameters['num_frames'], 
                    max_norm= hyperparameters['max_norm'])
    elif agent_name == 'SAC':
        agent = SAC(input_size=input_size,output_size=output_size,hidden_size=hyperparameters['num_hidden'],
                    gamma=hyperparameters['gamma'],alpha=0.2,tau=hyperparameters['tau'],
                    lr=hyperparameters['alpha'],max_norm=hyperparameters['max_norm'],
                    num_frames=hyperparameters['num_frames'],
                    buffer_size=hyperparameters['buffer_size'],env=env)
    return agent