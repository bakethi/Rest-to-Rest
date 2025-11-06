import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    A feature extractor that processes LiDAR data with a Transformer and
    concatenates the result with raw agent state data.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # We call the parent constructor with a features_dim that will be the
        # output size of our final MLP
        super().__init__(observation_space, features_dim=features_dim)

        # Extract the shapes of the different observation components
        lidar_shape = observation_space['lidar'].shape
        agent_state_shape = observation_space['agent_state'].shape

        # --- 1. Transformer for LiDAR Data ---
        # The TransformerEncoderLayer is a single layer. We stack 2 of them.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lidar_shape[0],  # The number of expected features in the input
            nhead=4,                 # The number of heads in the multi-head attention
            dim_feedforward=128,     # The dimension of the feedforward network
            dropout=0.1,
            activation='relu',
            batch_first=True         # IMPORTANT: Ensures input/output is (batch, seq, feature)
        )
        self.lidar_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # --- 2. Final MLP to Combine Features ---
        # The input to this MLP will be the output of the transformer PLUS the
        # size of the raw agent state vector.
        combined_input_size = lidar_shape[0] + agent_state_shape[0]

        self.final_mlp = nn.Sequential(
            nn.Linear(combined_input_size, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict) -> torch.Tensor:
        # Separate the observations
        lidar_data = observations['lidar']
        agent_state_data = observations['agent_state']

        # Process LiDAR data with the Transformer
        # The output `lidar_features` will have the same shape as the input: (batch_size, lidar_shape[0])
        lidar_features = self.lidar_transformer(lidar_data)

        # Concatenate the transformer's output with the RAW agent state data
        combined_features = torch.cat([lidar_features, agent_state_data], dim=1)

        # Pass the combined vector through the final MLP to get the final feature representation
        return self.final_mlp(combined_features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    An Actor-Critic policy that uses our custom Transformer-based feature extractor.
    """
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256), # Output size of our extractor
            **kwargs
        )