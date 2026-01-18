import torch
import numpy as np


class Buffer:
    def __init__(self, capacity, state_shape, policy_shape, device="cpu"):
        self.capacity = int(capacity)
        self.device = device

        self.size = 0
        self.pos = 0  # next write position

        self.states = torch.zeros(
            (self.capacity, *state_shape), dtype=torch.float32, device=device
        )
        self.policies = torch.zeros(
            (self.capacity, *policy_shape), dtype=torch.float32, device=device
        )
        self.values = torch.zeros(
            (self.capacity, 1), dtype=torch.float32, device=device
        )


    def push(self, training_data):
        """
        Add a batch of (state, policy, value) tensors.
        Oldest samples are overwritten first (FIFO).
        """
        batch_size = len(training_data)
        if batch_size == 0:
            return

        # Keep only most recent samples if batch is too large
        if batch_size > self.capacity:
            training_data = training_data[-self.capacity:]
            batch_size = len(training_data)

        states, policies, values = zip(*training_data)  # tuples of tensors

        states = torch.stack(states)
        policies = torch.stack(policies)
        values = torch.stack([v.unsqueeze(0) for v in values])  # make shape (batch,1)

        end_len = min(batch_size, self.capacity - self.pos)
        start_len = batch_size - end_len

        # Contiguous segment
        if end_len > 0:
            idx = np.arange(self.pos, self.pos + end_len)
            self.states[idx] = states[:end_len]
            self.policies[idx] = policies[:end_len]
            self.values[idx] = values[:end_len]

        # Wrap-around segment
        if start_len > 0:
            idx = np.arange(start_len)
            self.states[idx] = states[end_len:]
            self.policies[idx] = policies[end_len:]
            self.values[idx] = values[end_len:]

        self.pos = (self.pos + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)


    def sample(self, batch_size):
        """
        Uniformly sample a batch from the buffer.
        Returns None if the buffer is empty.
        """
        if self.size == 0:
            return None, None, None

        batch_size = min(batch_size, self.size)
        indices = np.random.randint(0, self.size, size=batch_size)

        states = self.states[indices]
        policies = self.policies[indices]
        values = self.values[indices]

        return states, policies, values


    def __len__(self):
        return self.size