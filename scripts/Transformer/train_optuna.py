# train_optuna.py (Modified Version)

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import argparse
import random
from transformer import CustomActorCriticPolicy

# --- Make sure your environment is installed: `pip install -e .` ---
try:
    import gym_pathfinding
except ImportError:
    print("Error: gym_pathfinding not installed. Please run 'pip install -e .' from your project root.")
    exit()




class RandomizedEnvWrapper(gym.Wrapper):
    """
    This wrapper randomizes the environment's parameters on each reset.
    These ranges define the training curriculum.
    """
    def __init__(self, env):
        super().__init__(env)
        self.num_intruders_range = [5, 20]
        self.intruder_speed_range = [0.8, 2.0]

    def reset(self, **kwargs):
        self.env.unwrapped.number_of_intruders = random.randint(*self.num_intruders_range)
        self.env.unwrapped.max_intruder_speed = random.uniform(*self.intruder_speed_range)
        return self.env.reset(**kwargs)


def train_agent(args):
    """
    Trains a PPO agent with PBRS hyperparameters from Optuna
    AND a randomized training curriculum.
    """
    env_kwargs = {
        'd_safe': args.d_safe,
        'k_bubble': args.k_bubble,
        'C_collision': args.C_collision,
        'k_pos': args.k_pos,
        'k_action': args.k_action,
        'w_safe': args.w_safe,
        'w_pos': args.w_pos
    }

    def make_env():
        env = gym.make("gym_pathfinding/IntruderAvoidanceTransformer-v0", **env_kwargs)
        env = RandomizedEnvWrapper(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        return env

    vec_env = make_vec_env(
        make_env,
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv
    )


                         
    model = PPO(
        CustomActorCriticPolicy, # Use the custom policy
        vec_env,
        verbose=0,
        device='cpu'
    )

    print(f"--- Training model for {args.total_timesteps} timesteps ---")
    model.learn(total_timesteps=args.total_timesteps)

    print(f"--- Saving model to {args.save_path} ---")
    model.save(args.save_path)
    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO agent for the IntruderAvoidance environment.")
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