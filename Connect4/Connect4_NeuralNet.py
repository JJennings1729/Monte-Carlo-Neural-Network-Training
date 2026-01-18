import torch
import torch.nn as nn
import torch.nn.functional as F

class Connect4Net(nn.Module):
    """
    AlphaZero-style network for Connect-4 with BatchNorm.
    Legal move masking is applied inside forward().
    """
    def __init__(self, input_channels=2, board_size=6, board_width=7, num_actions=7):
        super().__init__()

        # ---- Convolutional trunk with BatchNorm ----
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.flat_size = 64

        # ---- Policy head ----
        self.policy_fc1 = nn.Linear(self.flat_size, 128)
        self.policy_out = nn.Linear(128, num_actions)

        # ---- Value head ----
        self.value_fc1 = nn.Linear(self.flat_size, 128)
        self.value_fc2 = nn.Linear(128, 64)
        self.value_out = nn.Linear(64, 1)

    def forward(self, x, legal_mask=None):
        """
        x: (B, input_channels, H, W)
        legal_mask: (B, num_actions) boolean mask
        """

        # --- Shared trunk ---
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.mean(dim=[2, 3])  # Global Average Pooling

        # --- Policy head ---
        p = F.relu(self.policy_fc1(out))
        logits = self.policy_out(p)

        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float('-inf'))

        policy_probs = F.softmax(logits, dim=1)

        # --- Value head ---
        v = F.relu(self.value_fc1(out))
        v = F.relu(self.value_fc2(v))
        value = torch.tanh(self.value_out(v))

        return policy_probs, value