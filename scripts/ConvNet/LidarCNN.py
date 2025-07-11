# custom_networks.py

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LidarCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param num_lidar_scans: (int) The number of LiDAR scans in the observation.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, num_lidar_scans: int = 360):
        super().__init__(observation_space, features_dim)
        
        # We assume the LiDAR data is the last part of the observation
        self.num_lidar_scans = num_lidar_scans
        other_obs_dim = observation_space.shape[0] - num_lidar_scans

        # 1D CNN for LiDAR data
        # Input: (batch_size, 1, num_lidar_scans)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # To compute the size of the flattened CNN output, we pass a dummy tensor
        with torch.no_grad():
            dummy_lidar_input = torch.zeros(1, 1, self.num_lidar_scans)
            cnn_output_dim = self.cnn(dummy_lidar_input).shape[1]

        # The total feature dimension will be the CNN output plus the other observations
        combined_dim = cnn_output_dim + other_obs_dim

        # Linear layer to project the combined features to the desired features_dim
        self.linear = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Split observation into LiDAR and other data
        # Assumes LiDAR data is the last 'num_lidar_scans' elements
        lidar_data = observations[:, -self.num_lidar_scans:]
        other_data = observations[:, :-self.num_lidar_scans]
        
        # Reshape LiDAR data for Conv1d: (batch_size, 1, length)
        lidar_data = lidar_data.unsqueeze(1)
        
        # Pass LiDAR data through the CNN
        cnn_out = self.cnn(lidar_data)
        
        # Concatenate the CNN's output with the other observation data
        combined_features = torch.cat((cnn_out, other_data), dim=1)
        
        # Pass through the final linear layer
        return self.linear(combined_features)