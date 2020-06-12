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

# MADDPG algorithm imports.
from maddpg.hyperparams import *
from maddpg.model import Actor, Critic
from maddpg.buffer import ReplayBuffer
from maddpg.noise import OUNoise


class Agent():
    """ Class implementation of a so-called "intelligent" agent.
        This agent interacts with and learns from the environment.
        This agent employs the DDPG algorithm to solve this problem.
        See <https://spinningup.openai.com/en/latest/algorithms/ddpg.html>.
    """        
    
    def __init__(self, state_size, action_size, agent_id, add_noise=True):
        """ Initialize an Agent instance.
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            agent_id (int): Identifier number for this agent
            add_noise (bool): Toggle for using the stochastic process
        """
        
        # Set the parameters.
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(RANDOM_SEED)
        self.agent_id = agent_id

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
        
        # Arget-local model pairs should be initialized to the same weights.
        self.hard_update(self.actor_local, self.actor_target)
        self.hard_update(self.critic_local, self.critic_target)
        
        # Set up noise processing.
        if add_noise:
            self.noise = OUNoise(action_size)
        
        # Exploration noise amplification.
        self.noise_ampl = NOISE_AMPL
        
        # Noise amplification decay.
        self.noise_ampl_decay = NOISE_AMPL_DECAY

        # Log the network.
        self.print_arch()

    
    def print_arch(self):
        """ Print the network architecture for this agent's actors and critics.
        """
        
        print("\rAgent [#{}]".format(self.agent_id), end="\n\n")
        
        print("\rActor (Local):")
        print(self.actor_local)
        
        print("\rActor (Target):")
        print(self.actor_target)
        
        print("\rCritic (Local):")
        print(self.critic_local)
        
        print("\rCritic (Target):")
        print(self.critic_target)
        
        if self.agent_id != NUM_AGENTS:
            print("\r_______________________________________________________________", end="\n\n")


    def act(self, state, add_noise=False):
        """ Return the actions for a given state as per current policy.
        
        Params
        ======
            state (array_like): Current state
            add_noise (bool): Toggle for using the stochastic process
        """
            
        state = torch.from_numpy(state).float().to(device)
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        # If the stochastic process is enabled.
        if add_noise:
            action += self.noise.sample()
            self.decay_noise_ampl()
                
        # Return the action.
        return np.clip(action, -1, 1)
                

    def decay_noise_ampl(self):
        """ Decay exploration noise amplification.
        """
        
        self.noise_ampl *= self.noise_ampl_decay


    def reset(self):
        """ Reset the state.
        """
        
        # Reset the internal state (noise) to mean (mu).
        self.noise.reset()
        
        
    def learn(self, experiences, next_actions, actions_pred):
        """ Update value parameters using given batch of experience tuples.
            i.e.,
            Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
            where
                actor_target(state) -> action, and
                critic_target(state, action) -> Q-value.
        
        Params
        ======
            experiences (Tuple[torch.Tensor]): Tuple of (s, a, r, s', done, w) tuples 
            next_actions (list): Next actions computed from each agent
            actions_pred (list): Predicted actions for the current states from each agent
        """

        # Set the parameters.
        states, actions, rewards, next_states, dones = experiences
        
        # Each agent is assigned a tensor_id.
        tensor_id = torch.tensor([self.agent_id-1]).to(device)
        
        """ Update the Critic.
        """
        self.critic_optimizer.zero_grad()      
        Q_targets_next = self.critic_target(next_states, next_actions)        
        Q_targets = rewards.index_select(1, tensor_id) + (GAMMA * Q_targets_next * (1 - dones.index_select(1, tensor_id)))
        Q_expected = self.critic_local(states, actions)
        
        # Minimize the loss.
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        critic_loss.backward()
        self.critic_optimizer.step()
        
        """ Update the Actor.
        """
        self.actor_optimizer.zero_grad()
        
        # Minimize the loss.
        actor_loss = -self.critic_local(states, actions_pred).mean()
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

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1. - tau) * target_param.data)
            
    
    def hard_update(self, local_model, target_model):
        """ Hard update model parameters.
            i.e.,
            θ_target = θ_local.
        
        Params
        ======
            local_model (PyTorch model): Weights will be copied from
            target_model (PyTorch model): Weights will be copied to
        """
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
        