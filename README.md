# Monte Carlo Self-Play Neural Network Training

Alphazero style training with Monte Carlo Tree Search (MCTS) and self-play to learn policy–value neural networks.
No fixed number of episodes — training stops automatically when convergence is detected.

## Training Loop

Run MCTS self-play using the current network.

Collect (state, policy, value) PyTorch tensors for all moves.

Store samples in a replay buffer, pre-filled during warm-up.

Sample batches from the buffer to train the network.

Periodically evaluate the current network against reference networks.

Promote networks exceeding the win-rate threshold.

Stop automatically when KL divergence or losses are low.
