import os
import json

def save_experiment(experiment_name, agent_name,hyperparameters=None):
    # Create the experiments directory if it doesn't exist
    if not os.path.exists('experiments'):
        os.makedirs('experiments')

    # Create a subfolder for the experiment
    experiment_path = os.path.join('experiments', experiment_name)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    if hyperparameters is not None:
        # Save the hyperparameters to a JSON file in the experiments directory
        with open('experiments/{}/hyperparameters.json'.format(experiment_name), 'w') as f:
            json.dump(hyperparameters, f)

    print("Method: ", agent_name)
    return experiment_path