# Imports.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Other imports.
from ddpg.hyperparams import *


def hidden_init(layer):
    """ Hidden layer initialization.

    Params
    ======
        layer: Network layer
    """
    
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    
    # Return the layer (tuple).
    return (-lim, lim)


# Actor (Policy) Model.
class Actor(nn.Module):
    """ Class implementation of an Actor (Model) in the environment.
    """
    
    def __init__(self, state_size, action_size, fc1_units=768, fc2_units=512):
        """ Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """

        super(Actor, self).__init__()
        """ Initialize an Actor instance.
        """

        # Manual seeding.
        self.seed = torch.manual_seed(RANDOM_SEED)
        
        # Set the parameters.
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        # Lastly, reset the parameters.
        self.reset_parameters()
        
        
    def reset_parameters(self):
        """ Reset model parameters.
        """
        
        # Use the hidden layer initialization function.
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))        
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)


    def forward(self, state):
        """ Build an actor (policy) network that maps states -> actions.
        
        Params
        ======
            state: State of the environment
        """
        
        # Perform a feed-forward pass through the model.
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Return the result of the pass.
        return F.tanh(self.fc3(x))
        

class Critic(nn.Module):
    """ Critic (Value) Model.
    """

    def __init__(self, state_size, action_size, fcs1_units=512, fc2_units=256, fc3_units=128):
        """ Initialize parameters and build model.
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            fc3_units (int): Number of nodes in the third hidden layer
        """
        
        super(Critic, self).__init__()
        """ Initialize a Critic instance.
        """
        
        # Manual seeding.
        self.seed = torch.manual_seed(RANDOM_SEED)
        
        # self.bn = nn.BatchNorm1d(state_size)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        
        # Lastly, reset the parameters.
        self.reset_parameters()


    def reset_parameters(self):
        """ Reset model parameters.
        """
        
        # Use the hidden initialization function to initialize.
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
                

    def forward(self, state, action):
        """ Build a critic (value) network that maps (state, action) pairs -> Q-values.
        
        Params
        ======
            state: State of the environment
            action: Current action permissible in the environment 
        """
        
        # state = self.bn(state)
        
        # xs = F.leaky_relu(self.fc1(state))
        xs = F.relu(self.fcs1(state))
        
        # Concatenate (states) and (actions) as the input of the first layer (fc1).
        # Note that the DDPG algorithm does this before the second layer (fc2). 
        x = torch.cat((xs, action), dim=1)

        # x = F.leaky_relu(self.fc2(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Return the result of the pass.
        return self.fc4(x)
        