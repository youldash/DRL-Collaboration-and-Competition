# Imports.
import copy
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim # For optimizer support.

# Other imports.
from collections import namedtuple, deque

# DDPG algorithm imports.
from ddpg.hyperparams import *
from ddpg.model import Actor, Critic
from ddpg.buffer import ReplayBuffer
from ddpg.noise import OUNoise


class Agent():
    """ Class implementation of a so-called "intelligent" agent.
        This agent interacts with and learns from the environment.
        This agent employs the DDPG algorithm to solve this problem.
        See <https://spinningup.openai.com/en/latest/algorithms/ddpg.html>.
    """        
    
    def __init__(self, state_size, action_size, num_agents, add_noise=True):
        """ Initialize an Agent instance.
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            num_agents (int): Number of agents in the environment
            add_noise (bool): Toggle for using the stochastic process
        """
        
        # Set the parameters.
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(RANDOM_SEED)
        self.num_agents = num_agents

        # Setting the Actor network (with the Target Network).
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)

        # Optimize the Actor using Adam.
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
                
        # Setting the Critic network (with the Target Network).
        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        
        # Optimize the Critic using Adam.
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
                
        # Set up noise processing.
        if add_noise:
            self.noise = OUNoise(action_size)
        
        # Use the Replay memory buffer (once per class).
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        
        # Set up a working directory to save the model.
        self.model_dir = os.getcwd() + "/ddpg/models"
        os.makedirs(self.model_dir, exist_ok=True)
        
    
    def step(self, states, actions, rewards, next_states, dones):
        """ Update the network on each step.
            In other words, save the experience in replay memory,
            and then use random sampling from the buffer to learn.
        """
        
        # Save experience/reward tuples for each agent to a shared Replay memory buffer prior sampling.
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        
        # Save the experience in replay memory, then use random sampling from the buffer to learn.
        self.sample_and_learn()


    def sample_and_learn(self):
        """ For a specified number of agents,
            use random sampling from the buffer to learn.
        """
        
        # If enough samples are available in memory, get a random subset and learn.
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)


    def print_arch(self):
        """ Print the network architecture for this agent's actors and critics.
        """
        
        print("\rActor (Local):")
        print(self.actor_local)
        
        print("\rActor (Target):")
        print(self.actor_target)
        
        print("\rCritic (Local):")
        print(self.critic_local)
        
        print("\rCritic (Target):")
        print(self.critic_target)
        

    def act(self, states, add_noise=True):
        """ Return the actions for a given state as per current policy.
        
        Params
        ======
            states (array_like): Current states
            add_noise (bool): Toggle for using the stochastic process
        """
            
        states = torch.from_numpy(states).float().to(device)
        
        # List of actions.
        actions = np.zeros((self.num_agents, self.action_size))
        
        self.actor_local.eval()
        with torch.no_grad():
            for i, state in enumerate(states):
                # Populate a list of actions (one state at a time).
                actions[i, :] = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        # If the stochastic process is enabled.
        if add_noise:
            actions += self.noise.sample()
                
        # Return the action.
        return np.clip(actions, -1, 1)
        

    def reset(self):
        """ Reset the state.
        """
        
        # Reset the internal state (noise) to mean (mu).
        self.noise.reset()
        
        
    def learn(self, experiences, gamma):
        """ Update value parameters using given batch of experience tuples.
            i.e.,
            Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
            where
                actor_target(state) -> action, and
                critic_target(state, action) -> Q-value.
        
        Params
        ======
            experiences (Tuple[torch.Tensor]): Tuple of (s, a, r, s', done, w) tuples 
            gamma (float): Discount factor
        """

        # Set the parameters.
        states, actions, rewards, next_states, dones = experiences
        
        """ Update the Critic.
        """
        # Get the predicted next-state actions and Q-values from the target models.
        # Calculate the pair action/reward for each of the next_states.
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Compute Q-targets for the current states, (y_i).
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute the Critic loss.
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss.
        self.critic_optimizer.zero_grad()        
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
                
        """ Update the Actor.
        """
        # Compute the Actor loss.
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize the loss.
        self.actor_optimizer.zero_grad()        
        # torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        actor_loss.backward()
        self.actor_optimizer.step()

        """ Update the target networks.
        """
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
        

    def soft_update(self, local_model, target_model, tau):
        """ Soft update model parameters.
            i.e.,
            θ_target = τ * θ_local + (1 - τ) * θ_target.

        Params
        ======
            local_model (PyTorch model): Weights will be copied from
            target_model (PyTorch model): Weights will be copied to
            tau (float): Interpolation parameter 
        """

        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1. - tau) * target_param.data)
            
    
    def save(self):
        """ Save model weights to file, using the current working directory (i.e. model_dir).
        """

        # For each model (specifically, both Actor and Critic instances),
        # save their params and optimizer data to file (depending on the number of agents in the environment).
        torch.save(
            self.actor_local.state_dict(),
            os.path.join(self.model_dir, 'actor_params.pth'))           # Actor params.
        torch.save(
            self.actor_optimizer.state_dict(),
            os.path.join(self.model_dir, 'actor_optim_params.pth'))     # Actor optimizer params.
        torch.save(
            self.critic_local.state_dict(),
            os.path.join(self.model_dir, 'critic_params.pth'))          # Critic params.
        torch.save(
            self.critic_optimizer.state_dict(),
            os.path.join(self.model_dir, 'critic_optim_params.pth'))    # Critic optimizer params.
    
    
    def load(self):
        """ Load the weights from the saved model (if it exists).
        """
        
        # For each model (specifically, both Actor and Critic instances),
        # load their params and optimizer data from file (depending on the number of agents in the environment).
        self.actor_local.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'actor_params.pth')))           # Actor params.
        self.actor_optimizer.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'actor_optim_params.pth')))     # Actor optimizer params.
        self.critic_local.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'critic_params.pth')))          # Critic params.
        self.critic_optimizer.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'critic_optim_params.pth')))    # Critic optimizer params.
                    