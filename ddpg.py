# !pip -q install ./python

# Imports.
import time
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# High-resolution plots for retina displays.
%config InlineBackend.figure_format = 'retina'

# Utility imports.
from itertools import count
from collections import deque

# Hide any deprecate warnings.
import warnings
warnings.filterwarnings("ignore")

# Agents interact with, and learns from environments.
from agent import Agent

%load_ext autoreload
%autoreload 2


# Unity environment.
from unityagents import UnityEnvironment

# Select this option to load version 1 (with a single agent) of the environment.
# env = UnityEnvironment(file_name='/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')

# Select this option to load version 2 (with 20 agents) of the environment.
env = UnityEnvironment(file_name='/data/Reacher_Linux_NoVis/Reacher.x86_64')


# Get the default brain.
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# Set the working device on the NVIDIA Tesla K80 accelerator GPU (depending on availability).
# Otherwise we use the CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', str(device).upper())
print()

# Log additional info (when using the NVIDIA Tesla K80 accelerator).
# See <https://www.nvidia.com/en-gb/data-center/tesla-k80/>.
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


# Reset the environment.
env_info = env.reset(train_mode=True)[brain_name]

# Number of agents.
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# Size of each action.
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# Examine the state space.
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


env_info = env.reset(train_mode=True)[brain_name]      # Reset the environment.
states = env_info.vector_observations                  # Get the current state (for each agent).
scores = np.zeros(num_agents)                          # Initialize the score (for each agent).
while True:
    actions = np.random.randn(num_agents, action_size) # Select an action (for each agent).
    actions = np.clip(actions, -1, 1)                  # All actions between -1 and 1.
    env_info = env.step(actions)[brain_name]           # Send all actions to tne environment.
    next_states = env_info.vector_observations         # Get next state (for each agent).
    rewards = env_info.rewards                         # Get reward (for each agent).
    dones = env_info.local_done                        # See if episode finished.
    scores += env_info.rewards                         # Update the score (for each agent).
    states = next_states                               # Roll over states to next time step.
    if np.any(dones):                                  # Exit loop if episode finished.
        break

print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


# Monitor training time (start time).
start_time = time.time()

# Now, we train the agent.
scores, mean = ddpg()

# Monitor training time (end time).
end_time = (time.time()-start_time)/6e1

# Log the runtime.
print("\nTotal runtime {:.2f} minutes.".format(end_time))


# Plot the scores using matplotlib.
plt.style.use('ggplot')

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
plt.title('Rewards - Using DDPG')
# plt.plot(np.arange(1, len(scores)+1), scores)
plt.plot(np.arange(len(scores)), scores)
plt.plot(np.arange(len(mean)), mean)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.legend(('Score', 'Mean'), fontsize='xx-large')

# Reveal the plot.
plt.show()


env.close()


def ddpg(n_episodes=int(3e2), max_t=int(5e3)):
    """ Implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm.
        See <https://spinningup.openai.com/en/latest/algorithms/ddpg.html>.
    
    Params
    ======
        n_episodes (int): Maximum number of training episodes
        max_t (int): Maximum number of time steps per episode
    """

    # List containing scores from each episode.
    scores = []
    
    # List the mean of the window scores.
    scores_mean = []

    # Initialize other parameters.
    scores_per_episode = []
    scores_window = deque(maxlen=int(1e2))
    agents = []                                              # Agent list.
    
    # Initialize, and add new agents.
    for i in range(num_agents):
        agents.append(Agent(state_size, action_size, seed=0))

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]    # Reset the environment.
        states = env_info.vector_observations                # Get the current state (for each agent).
        
#         tick = time.time()
        
        # Reset each agent (and noise).
        for agent in agents:
            agent.reset()

        scores = np.zeros(num_agents)                        # Initialize the score (for each agent).

        for t in range(max_t):
            actions = np.array([agents[i].act(states[i]) for i in range(num_agents)])
            env_info = env.step(actions)[brain_name]         # Send all actions to tne environment.
            next_states = env_info.vector_observations       # Get next state (for each agent).
            rewards = env_info.rewards                       # Get reward (for each agent).
            dones = env_info.local_done                      # See if episode finished.
            
            # Update the score (for each agent).
            for i in range(num_agents):
                agents[i].step(t, states[i], actions[i], rewards[i], next_states[i], dones[i]) 
            
            states = next_states                             # Roll over states to the next time step.
            scores += rewards
            
#             if t % 20:
#                 print('\r\tTIMESTEP {}\tμ: {:.2f}\t\tMAX: {:.2f}\tMIN: {:.2f}'
#                     '\tLEN: {}\tTIME: {:.2f}'.format(t,
#                                                      scores_mean[-1],
#                                                      np.max(scores),
#                                                      np.min(scores),
#                                                      len(agent.memory),
#                                                     time.time()-tick),
#                       end="\n")
    
            if np.any(dones):                                # Exit loop if the episode finished.
                break
        
        score = np.mean(scores)
        scores_window.append(score)                          # Save the most recent score.
        scores_mean.append(np.mean(scores_window))

        scores_per_episode.append(score)

        print('\rEPISODE {}\tSCORE: {:.2f}\tAVG (μ): {:.2f}\tMAX: {:.2f}\tMIN: {:.2f}'.format(
            i_episode, score, scores_mean[-1], np.max(scores_window), np.min(scores_window)), end="\n")
        
#         if i_episode % 100 == 0:
#             print('\rEPISODE {}\tAVG SCORE (μ): {:.2f}'.format(i_episode, np.mean(scores_window)))
            
        if np.mean(scores_window) >= 30.:
            print('\nEnvironment solved in {:d} episodes.\tAverage score (μ): {:.2f}'.format(
                i_episode-100, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'models/checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'models/checkpoint_critic.pth')
            print("\nModels saved successfully.")            
            
    # Return the scores.
    return scores_per_episode, scores_mean