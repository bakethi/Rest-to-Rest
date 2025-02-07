import torch.nn as nn
import torch.nn.functional as F


class DQNModel(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Initialize the DQN model with input and output sizes.

        Args:
            input_size (int): The number of input features (state space).
            output_size (int): The number of possible actions (output size).
        """
        super(DQNModel, self).__init__()

        # Define the layers of the neural network
        self.layer1 = nn.Linear(input_size, 128)  # First hidden layer
        self.layer2 = nn.Linear(128, 128)         # Second hidden layer
        self.layer3 = nn.Linear(128, output_size) # Output layer (Q-values)

    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Args:
            x (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The predicted Q-values for each action.
        """
        # Pass input through the layers
        x = F.relu(self.layer1(x))  # Apply ReLU activation after first layer
        x = F.relu(self.layer2(x))  # Apply ReLU activation after second layer
        x = self.layer3(x)          # Output layer (no activation function here)

        return x
