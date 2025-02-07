import numpy as np
import torch
import torch.nn.functional as F


class DQNAgent:
    def __init__(self, model, optimizer, replay_buffer, epsilon, gamma, epsilon_decay, min_epsilon, batch_size):
        """
        Initializes the DQN agent with a model, optimizer, replay buffer, and hyperparameters.

        Args:
            model (nn.Module): The neural network model (Q-network).
            optimizer (optim.Optimizer): The optimizer used to train the model.
            replay_buffer (ReplayBuffer): The experience replay buffer.
            epsilon (float): Exploration rate for epsilon-greedy strategy.
            gamma (float): Discount factor for future rewards.
            epsilon_decay (float): Rate at which epsilon decays after each episode.
            min_epsilon (float): The minimum value for epsilon.
            batch_size (int): The size of the batches used during training.
        """
        self.model = model
        self.target_model = model
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def act(self, state):
        """
        Select an action based on the epsilon-greedy policy.

        Args:
            state (array): The current state of the environment.

        Returns:
            int: The selected action.
        """
        if np.random.rand() < self.epsilon:
            # Exploration: Choose random action
            return np.random.choice(len(state))
        else:
            # Exploitation: Choose action with max Q-value
            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    def train(self):
        """
        Train the DQN agent by sampling a batch of experiences from the replay buffer,
        calculating the target Q-values, and updating the model.
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough data in the buffer to train

        # Sample a batch of experiences from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Get Q-values for current states
        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get Q-values for next states from target model
        next_q_values = self.target_model(next_states).max(1)[0]

        # Compute the target Q-values
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute the loss between predicted and target Q-values
        loss = F.mse_loss(q_values, target_q_values)

        # Backpropagate and optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        """
        Update the target model by copying weights from the main model.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def store_experience(self, state, action, reward, next_state, done):
        """
        Store the experience (state, action, reward, next_state, done) into the replay buffer.

        Args:
            state (array): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (array): The next state after the action.
            done (bool): Whether the episode is finished.
        """
        self.replay_buffer.add(state, action, reward, next_state, done)

    def decrement_epsilon(self):
        """
        Decrement epsilon to reduce exploration over time.
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
