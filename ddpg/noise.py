# Imports.
import numpy as np
import random
import copy

# Other imports.
from ddpg.hyperparams import *


class OUNoise():
    """ Class implementation of the Ornstein-Uhlenbeck (stochastic) process.
        See <https://en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process>.
    """
    
    def __init__(self, size, mu=0., theta=.15, sigma=.2):
        """ Initialize parameters and noise process.
        """
        
        # Set the parameters.
        self.size = size
        self.seed = random.seed(RANDOM_SEED)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        
        # Reset.
        self.reset()


    def reset(self):
        """ Reset the internal state (= noise) to mean (mu).
        """
        
        self.state = copy.copy(self.mu)


    def sample(self):
        """ Update internal state and return it as a noise sample.
        """
        
        x = self.state
        # dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        
        # Return the current state.
        return self.state
