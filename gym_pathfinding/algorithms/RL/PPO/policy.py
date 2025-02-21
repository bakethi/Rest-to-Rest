import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(PPOPolicy, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layers for mean and standard deviation
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.log_std_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output action mean
        mean = torch.tanh(self.mean_layer(x))  # Tanh to keep values in (-1, 1)

        # Log standard deviation (learnable)
        log_std = self.log_std_layer(x)
        std = torch.exp(log_std)

        return mean, std
