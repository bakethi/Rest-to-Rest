import random
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size):
        """
        Initialize the replay buffer with a given max size.

        Args:
            max_size (int): The maximum size of the buffer.
        """
        self.buffer = []
        self.max_size = max_size
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        """
        Store a new experience in the replay buffer.

        Args:
            state (array): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (array): The next state after the action.
            done (bool): Whether the episode has ended.
        """
        experience = (state, action, reward, next_state, done)
        
        # If buffer is full, remove the oldest experience
        if self.size >= self.max_size:
            self.buffer.pop(0)
        else:
            self.size += 1
        
        # Add new experience to buffer
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Sample a random batch of experiences from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            list: A list of random experiences from the buffer.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """
        Return the current size of the buffer.

        Returns:
            int: The current size of the buffer.
        """
        return self.size
