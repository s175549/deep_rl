import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot(method, experiment_path, results, loss, rewards, num_frames, lengths):
    
    best_return_id = np.where(np.array(results)==max(np.array(results)))[0][0]
    # Plot results 
    plt.figure(1)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    if min(results)<-1000:
        plt.ylim((-1000,max(results)+30))
    plt.title('{}'.format(method))
    plt.scatter(x=best_return_id,y=max(results),c='r',marker='o',s=50)
    plt.plot(np.array(results))
    plt.legend(['Max return = {}'.format(round(max(results),1))])
    #plt.show()

    # Save plots
    plot_path = os.path.join(experiment_path, 'returns.png')
    plt.savefig(plot_path)

    # Plot smoothned rewards (reward per 10000 frame)
    rewards_smooth = []
    frames = np.arange(1,num_frames,10000)
    rewards_smooth.append(np.mean(rewards[frames[0]:frames[1]]))
    for i, j in enumerate(frames[1:]):
        rewards_smooth.append(np.mean(rewards[frames[i]:j]))

    best_reward_id = np.where(rewards_smooth==max(rewards_smooth))

    # Save plots
    plt.figure(2)
    plt.xlabel('Frame')
    plt.ylabel('Reward')
    plt.title('{}'.format(method))
    plt.scatter(x=frames[best_reward_id],y=np.array(rewards_smooth)[best_reward_id],c='r',marker='o',s=50)
    sns.lineplot(x=frames,y=np.array(rewards_smooth))
    plt.legend(['Max avg. reward = {}'.format(round(np.array(rewards_smooth)[best_reward_id][0],2))])

    # Save experiment
    plot_path = os.path.join(experiment_path, 'rewards.png')
    plt.savefig(plot_path)

    # Save plots
    plt.figure(3)
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.title('{}, episode lengths'.format(method))
    plt.plot(np.array(lengths))

    # Save experiment
    plot_path = os.path.join(experiment_path, 'lengths.png')
    plt.savefig(plot_path)

    # Plot results 
    plt.figure(4)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('{}, loss'.format(method))
    plt.plot(np.array(loss))
    #plt.show()

    # Save experiment
    plot_path = os.path.join(experiment_path, 'loss.png')
    plt.savefig(plot_path)