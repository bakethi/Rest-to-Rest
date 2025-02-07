import gym
from trainer import Trainer
from utils import Config, Checkpointer, Plotter


class TrainDQN:
    def __init__(self, env_name):
        """
        Initializes the training setup with the environment, configuration,
        trainer, and utilities for checkpointing and plotting.

        Args:
            env_name (str): The name of the gym environment.
        """
        # Create the environment
        self.env = gym.make(env_name)

        # Initialize the configuration with default values
        self.config = Config(
            batch_size=64, 
            gamma=0.99, 
            epsilon=0.1, 
            learning_rate=1e-3, 
            max_episodes=1000, 
            target_update_frequency=10
            )

        # Initialize the trainer that handles the DQN agent's training
        self.trainer = Trainer(
            env=self.env, 
            batch_size=self.config.batch_size, 
            gamma=self.config.gamma, 
            target_update_frequency=self.config.target_update_frequency
            )

        # Initialize utilities for checkpointing and plotting
        self.checkpointer = Checkpointer(save_path="model_checkpoint.pth")
        self.plotter = Plotter()

        # List to store rewards for plotting
        self.episode_rewards = []

    def train(self, num_episodes):
        """
        Train the DQN agent for a specified number of episodes.

        Args:
            num_episodes (int): The number of episodes to train the agent.
        """
        # Training loop
        for episode in range(num_episodes):
            print(f"Training episode {episode + 1}/{num_episodes}")

            # Train the agent using the trainer
            self.trainer.train(1)  # Train for 1 episode

            # Optionally, save the model checkpoint after each episode
            if episode % 100 == 0:
                self.checkpointer.save_model(self.trainer.model, self.trainer.optimizer, episode)

            # Track rewards for plotting
            self.episode_rewards.append(self.trainer.agent.total_rewards)

        # After training, plot the rewards
        self.plotter.episode_rewards = self.episode_rewards
        self.plotter.plot_rewards()

    def evaluate(self):
        """
        Evaluate the trained DQN agent by running it in the environment.
        Optionally prints average rewards after several test episodes.
        """
        total_rewards = []
        num_test_episodes = 10

        for _ in range(num_test_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.trainer.agent.select_best_action(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
            total_rewards.append(total_reward)

        avg_reward = sum(total_rewards) / num_test_episodes
        print(f"Average reward over {num_test_episodes} evaluation episodes: {avg_reward}")


if __name__ == "__main__":
    # Create the TrainDQN instance with the desired environment
    train_dqn = TrainDQN(env_name="CartPole-v1")

    # Train the agent for 1000 episodes
    train_dqn.train(num_episodes=1000)

    # Evaluate the agent after training
    train_dqn.evaluate()
