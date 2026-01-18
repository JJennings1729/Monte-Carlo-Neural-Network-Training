
"""
CONNECT4 init.py

"""

from . import Connect4_Env as env
from .Connect4_NeuralNet import Connect4Net as nn_model
from .Connect4_PlayGame import Connect4PlayGame as play_game

# Include relevant constants

# Constants used in training
replay_buffer_size = 30_000
replay_warmup_ratio = 0.25
warmup_episodes = 10
eval_interval = 10
eval_games = 100
promote_thresh = 0.55

# Network initial parameters
net_forward_batch_size = 128
net_training_batch_size = 64
net_max_learning_rate = 1e-2
net_min_learning_rate = 1e-6
num_training_samples = 5000
net_epochs = 5
net_l2_reg = 1e-4
value_loss_weight = 0.5

# Constants used in Monte Carlo Search
num_games = 256
simulations = 100
determ_depth = 6
rand_pos_depth = 10
alpha = 0.3
epsilon = 0.25

__all__ = ["env", "nn_model", "play_game"]