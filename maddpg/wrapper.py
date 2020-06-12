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
from maddpg.agent import Agent
from maddpg.model import Actor, Critic
from maddpg.buffer import ReplayBuffer
from maddpg.noise import OUNoise


class Wrapper():
    """ Wrapper class for managing so-called "intelligent" agents in the Tennis environment.
        These agents interact with and learn from the environment.
        These agents employ the MADDPG algorithm to solve this problem.
    """
    
    def __init__(self, num_agents, state_size, action_size):
        """ Initialize a Multi-agent Deep Deterministic Policy Gradient (MADDPG) algorithm wrapper.
        
        Params
        ======
            num_agents (int): Number of agents in the environment
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        
        # Set the parameters.
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        
        # Creat new DDPG agents and add them into a new list.
        self.agents = [Agent(state_size, action_size, i+1) for i in range(num_agents)]
        
        # Use the Replay memory buffer (once per class).
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
                
        # Initialize the time step (until max WEIGHT_FREQUENCY is reached).
        self.t_step = 0
        
        # Set up a working directory to save the model.
        self.model_dir = os.getcwd() + "/maddpg/models"
        os.makedirs(self.model_dir, exist_ok=True)
        

    def get_tensor(self, i):
        """ Get an agent's number as a Torch tensor.

        Params
        ======
            i (int): Agent ID
        """
        
        # Return the agent's tensor.
        return torch.tensor([i]).to(device)
    
    
    def step(self, states, actions, rewards, next_states, dones):
        """ Update the network on each step.
            In other words, save the experience in replay memory,
            and then use random sampling from the buffer to learn.
        """
        
        states = states.reshape(1, -1)
        actions = actions.reshape(1, -1)
        next_states = next_states.reshape(1, -1)
        
        # Save experience in replay memory.
        self.memory.add(states, actions, rewards, next_states, dones)
        
        # Learn every time step till WEIGHT_FREQUENCY is reached.
        self.t_step = (self.t_step+1) % WEIGHT_FREQUENCY
        
        # Save the experience in replay memory, then use random sampling from the buffer to learn.
        self.sample_and_learn()
        
        
    def sample_and_learn(self):
        """ For a specified number of agents,
            use random sampling from the buffer to learn.
        """
        
        # If enough samples are available in memory, get a random subset and learn.
        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            for i, agent in enumerate(self.agents):
                experiences = self.memory.sample()
                self.learn(experiences, i)

    
    def act(self, observations, add_noise=False):
        """ Pick an action for each agent given their individual observations, and the current policy.
        
        Params
        ======
            observations (array_like): Current agent state observations
            add_noise (bool): Toggle for using the stochastic process
        """
        
        # Action list.
        actions = []
        
        for agent, observation in zip(self.agents, observations):
            action = agent.act(observation, add_noise=add_noise)
            actions.append(action)
                
        # Return the actions.
        return np.array(actions)
                

    def reset(self):
        """ Reset the state.
        """
        
        # Reset the internal state (noise) to mean (mu).
        for agent in self.agents:
            agent.reset()
                
        
    def learn(self, experiences, agent_num):
        """ Pick actions from each agent for the experiences tuple, 
            which will be used for updating the weights to agent (with ID = agent_num).
            Note that each observation in the experiences tuple contains observations from each agent,
            so before using the tuple to update the weights of an agent,
            we need all agents to contribute in generating (next_actions), and predictions (i.e. actions_pred).
            This happens because the Critic will take, as its input,
            the combined observations and actions from all agents.
        
        Params
        ======
            experiences (Tuple[torch.Tensor]): Tuple of (s, a, r, s', done, w) tuples 
            agent_num (int): Agent number
        """

        # Set the parameters.
        next_actions = []
        actions_pred = []
        states, _, _, next_states, _ = experiences
        
        next_states = next_states.reshape(-1, self.num_agents, self.state_size)
        states = states.reshape(-1, self.num_agents, self.state_size)
        
        # Pick actions for each agent, provided that each agent is assigned a tensor_id.
        for i, agent in enumerate(self.agents):
            tensor_id = self.get_tensor(i)
            
            # Get the current, and next state for each agent.
            state = states.index_select(1, tensor_id).squeeze(1)
            next_state = next_states.index_select(1, tensor_id).squeeze(1)
            
            # Append them to both action lists.
            next_actions.append(agent.actor_target(next_state))
            actions_pred.append(agent.actor_local(state))
            
        # Concatenate the actions.
        next_actions = torch.cat(next_actions, dim=1).to(device)
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        
        # Update value parameters using given batch of experience tuples.
        agent = self.agents[agent_num]
        agent.learn(experiences, next_actions, actions_pred)
        
        
    def save(self):
        """ Save model weights to file, using the current working directory (i.e. model_dir).
        """
        
        # For each model (specifically, both Actor and Critic instances),
        # save their params and optimizer data to file (depending on the number of agents in the environment).
        for i in range(self.num_agents):
            torch.save(
                self.agents[i].actor_local.state_dict(),
                os.path.join(self.model_dir, 'actor_params_{}.pth'.format(i)))           # Actor params.
            torch.save(
                self.agents[i].actor_optimizer.state_dict(), 
                os.path.join(self.model_dir, 'actor_optim_params_{}.pth'.format(i)))     # Actor optimizer params.
            torch.save(
                self.agents[i].critic_local.state_dict(), 
                os.path.join(self.model_dir, 'critic_params_{}.pth'.format(i)))          # Critic params.
            torch.save(
                self.agents[i].critic_optimizer.state_dict(), 
                os.path.join(self.model_dir, 'critic_optim_params_{}.pth'.format(i)))    # Critic optimizer params.

        
    def load(self):
        """ Load the weights from the saved model (if it exists).
        """
        
        # For each model (specifically, both Actor and Critic instances),
        # load their params and optimizer data from file (depending on the number of agents in the environment).
        for i in range(self.num_agents):
            self.agents[i].actor_local.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'actor_params_{}.pth'.format(i))))          # Actor params.
            self.agents[i].actor_optimizer.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'actor_optim_params_{}.pth'.format(i))))    # Actor optimizer params.
            self.agents[i].critic_local.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'critic_params_{}.pth'.format(i))))         # Critic params.
            self.agents[i].critic_optimizer.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'critic_optim_params_{}.pth'.format(i))))   # Critic optimizer params.
    