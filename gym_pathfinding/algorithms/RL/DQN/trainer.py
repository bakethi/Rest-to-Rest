import torch
import torch.optim as optim
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
from dqn_model import DQNModel


class Trainer:
    def __init__(self, env, batch_size, gamma, target_update_frequency):
        """
        Initialize the trainer with environment, agent, and training parameters.

        Args:
            env (gym.Env): The environment to train the agent on.
            batch_size (int): The batch size used for training.
            gamma (float): The discount factor for Q-learning.
            target_update_frequency (int): Frequency (in steps) to update the target network.
        """
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_frequency = target_update_frequency

        # Create the model and the optimizer
        self.model = DQNModel(env.observation_space.shape[0], env.action_space.n)
        self.target_model = DQNModel(env.observation_space.shape[0], env.action_space.n)
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize target model with same weights
        self.optimizer = optim.Adam(self.model.parameters())

        # Create the replay buffer
        self.buffer = ReplayBuffer(max_size=100000)

        # Initialize the agent
        self.agent = DQNAgent(self.model)

    def train(self, num_episodes):
        """
        Train the agent for a number of episodes.

        Args:
            num_episodes (int): The number of episodes to train the agent.
        """
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                # Select an action using epsilon-greedy strategy
                action = self.agent.select_action(state)

                # Step in the environment
                next_state, reward, done, _ = self.env.step(action)

                # Store experience in replay buffer
                self.buffer.add(state, action, reward, next_state, done)

                # Update state and accumulate reward
                state = next_state
                total_reward += reward

                # Train the model if we have enough samples in the buffer
                if len(self.buffer) > self.batch_size:
                    self.optimize_model()

            # Print episode summary
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

            # Update the target network periodically
            if episode % self.target_update_frequency == 0:
                self.update_target_network()

    def optimize_model(self):
        """
        Optimize the model by sampling a batch from the buffer and updating the Q-values.
        """
        # Sample a batch of experiences
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert everything to tensors
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        # Get Q-values for the current states and next states
        current_q_values = self.model(states)
        next_q_values = self.target_model(next_states)

        # Get the Q-values corresponding to the selected actions
        current_q_value = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute the target Q-values
        next_q_value = next_q_values.max(1)[0]
        target_q_value = rewards + (self.gamma * next_q_value * ~dones)

        # Compute the loss
        loss = torch.mean((current_q_value - target_q_value) ** 2)

        # Backpropagate the loss and update the model parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """
        Update the target network with the weights of the model.
        """
        self.target_model.load_state_dict(self.model.state_dict())
