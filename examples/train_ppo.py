import numpy as np
import pygame
from gym_pathfinding.envs.pathfinding_env import PathfindingEnv
from gym_pathfinding.algorithms.RL.PPO import PPOTrainer
from gym_pathfinding.envs.visualization import Renderer

# Initialize environment
env = PathfindingEnv(
    number_of_obstacles=50,  
    bounds=np.array([[0, 0], [100, 100]]),
    bounce_factor=1,
    num_lidar_scans=24,
    lidar_max_range=50
)

# Initialize PPO Trainer
trainer = PPOTrainer(env)

# Initialize Renderer
renderer = Renderer(env)
# test render
renderer.render(env.agent, env.obstacle_manager, env.target_position)
pygame.time.wait(5000)

# Train the PPO agent with visualization
max_episodes = 1000
render_interval = 10  # Render every 10 episodes

for episode in range(max_episodes):
    obs, _ = env.reset()
    done = False

    # Start rendering every N episodes
    if episode % render_interval == 0:
        print(f"Rendering Episode {episode}...")

    while not done:
        action, _ = trainer.agent.select_action(obs)
        next_obs, reward, done, _, _ = env.step(action)

        # Render the environment
        if episode % render_interval == 0:
            renderer.render(env.agent, env.obstacle_manager, env.target_position)

        obs = next_obs

# Close the renderer at the end
renderer.close()
pygame.quit()
