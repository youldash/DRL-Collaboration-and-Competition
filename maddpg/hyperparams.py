# Imports.
import torch


""" Hyperparameter setup.
"""
RANDOM_SEED = 0         # Random seed used for PyTorch, NumPy and random packages.
BUFFER_SIZE = int(1e6)  # Replay buffer size (5e5 | 1e6).
BATCH_SIZE = 1024       # Minibatch size (128 | 256 | 512 | 1024).
GAMMA = 99e-2           # Discount factor.
TAU = 1e-3              # For soft update of target parameters (5e-2 | 1e-3).
LR_ACTOR = 1e-4         # Learning rate of the Actor (1e-3 | 1e-4 | 5e-4).
LR_CRITIC = 1e-3        # Learning rate of the Critic (1e-3 | 1e-4 | 5e-4).
WEIGHT_DECAY = 0.       # L2 weight decay.
WEIGHT_FREQUENCY = 2    # How often to update the weight (i.e. weight frequency).
NOISE_AMPL = 1          # Exploration noise amplification.
NOISE_AMPL_DECAY = 1    # Noise amplification decay.


""" Tennis environment setup.
"""
NUM_AGENTS = 2          # Number of agents in the environment.
STATE_SIZE = 24         # State space size.
ACTION_SIZE = 2         # Action size.


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
""" Set the working device on the NVIDIA Tesla K80 accelerator GPU (if available).
    Otherwise we use the CPU (depending on availability).
"""
