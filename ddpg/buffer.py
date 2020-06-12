# Imports.
import torch
import numpy as np

# Utility imports.
import random

# Using deques and named tuples.
from collections import namedtuple, deque

# Other imports.
from ddpg.hyperparams import *


class ReplayBuffer:
    """ Class implementation of a fixed-size (replay) buffer for storing experience tuples.
    """

    def __init__(self, action_size, buffer_size, batch_size):
        """ Initialize a ReplayBuffer instance.

        Params
        ======
            action_size (int): Dimension of each action
            buffer_size (int): Maximum buffer size
            batch_size (int): Size of each training batch
        """
        
        # Set the parameters.
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(RANDOM_SEED)
    
    
    def __len__(self):
        """ Return the current size of the internal memory.
        """

        return len(self.memory)


    def add(self, state, action, reward, next_state, done):
        """ Add a new experience to memory.
        """

        self.memory.append(self.experience(state, action, reward, next_state, done))
    

    def sample(self):
        """ Randomly sample a batch of experiences from memory.
        """

        # Random sample of experiences from memory.
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        # Return the batch.
        return (states, actions, rewards, next_states, dones)
                