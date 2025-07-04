# train.py (Corrected Version with Domain Randomization)

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import argparse
import random

# --- Make sure your environment is installed: `pip install -e .` ---
try:
    import gym_pathfinding
except ImportError:
    print("Error: gym_pathfinding not installed. Please run 'pip install -e .' from your project root.")
    exit()

# +++ 1. INCLUDE THE RANDOMIZATION WRAPPER FROM YOUR OLD SCRIPT +++
class RandomizedEnvWrapper(gym.Wrapper):
    """
    This wrapper randomizes the environment's parameters on each reset.
    These ranges define the training curriculum.
    """
    def __init__(self, env):
        super().__init__(env)
        # --- Define the curriculum ranges here ---
        self.num_intruders_range = [5, 20]  # Train with 5 to 20 intruders
        self.intruder_speed_range = [0.8, 2.0]   # Train with speeds between 0.8 and 2.0

    def reset(self, **kwargs):
        # Pick random values for the curriculum
        num_intruders = random.randint(self.num_intruders_range[0], self.num_intruders_range[1])
        max_speed = random.uniform(self.intruder_speed_range[0], self.intruder_speed_range[1])

        # Access the underlying environment via `unwrapped` to set the parameters
        self.env.unwrapped.number_of_intruders = num_intruders
        self.env.unwrapped.max_intruder_speed = max_speed
        
        # Now, call the original reset method, which will use these new parameters
        return self.env.reset(**kwargs)


def train_agent(args):
    """
    Trains a PPO agent with PBRS hyperparameters from Optuna
    AND a randomized training curriculum.
    """
    # 1. Collect all PBRS-related hyperparameters into a dictionary
    env_kwargs = {
        'd_safe': args.d_safe,
        'k_bubble': args.k_bubble,
        'C_collision': args.C_collision,
        'k_pos': args.k_pos,
        'k_action': args.k_action,
        'w_safe': args.w_safe,
        'w_pos': args.w_pos
    }

    # 2. Define a function to create and wrap the environment
    #    This is the standard way to apply wrappers before vectorization.
    def make_env():
        # Create the base environment with the PBRS parameters from Optuna
        env = gym.make("gym_pathfinding/IntruderAvoidance-v0", **env_kwargs)
        # Apply the randomization wrapper for a robust curriculum
        env = RandomizedEnvWrapper(env)
        # Apply the time limit wrapper to prevent infinite episodes
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        return env

    # 3. Create the vectorized environment
    vec_env = make_vec_env(
        make_env,
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv
    )

    # 4. Define the PPO model
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                         net_arch=dict(pi=[256, 256], vf=[256, 256]))
                         
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device='cpu'
    )

    # 5. Train the model
    print(f"--- Training model for {args.total_timesteps} timesteps ---")
    model.learn(total_timesteps=args.total_timesteps)

    # 6. Save the final model
    print(f"--- Saving model to {args.save_path} ---")
    model.save(args.save_path)
    vec_env.close()


if __name__ == "__main__":
    # The argparse section remains exactly the same as before
    parser = argparse.ArgumentParser(description="Train a PPO agent for the IntruderAvoidance environment.")
    # ... (rest of the file is identical to the previous train.py) ...
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the trained model.")
    parser.add_argument("--total_timesteps", type=int, default=100000, help="Total timesteps for training.")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments to use for training.")
    parser.add_argument("--d_safe", type=float, required=True, help="Radius of the safety bubble.")
    parser.add_argument("--k_bubble", type=float, required=True, help="Penalty magnitude for entering the bubble.")
    parser.add_argument("--C_collision", type=float, required=True, help="Large terminal penalty for a collision.")
    parser.add_argument("--k_pos", type=float, required=True, help="Scaling constant for position penalty.")
    parser.add_argument("--k_action", type=float, required=True, help="Scaling constant for action magnitude penalty.")
    parser.add_argument("--w_safe", type=float, required=True, help="Weight for the safety potential.")
    parser.add_argument("--w_pos", type=float, required=True, help="Weight for the position-holding potential.")
    args = parser.parse_args()
    train_agent(args)