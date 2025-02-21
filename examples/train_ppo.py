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
pygame.quit()

# Train the PPO agent with visualization
visualize= True
max_episodes = 1000
train_interval = 10   # Render every 10 episodes

# Training Loop with Periodic Visualization
for episode in range(0, max_episodes, train_interval):
    print(f"\nTraining episodes {episode} to {episode + train_interval - 1}...")
    
    # Train PPO for train_interval episodes
    trainer.train(max_episodes=train_interval, rollout_size=2048, batch_size=64)
    
    # Run **one** inference episode with visualization
    print(f"Running inference visualization after {train_interval} training episodes...")
    obs, _ = env.reset()
    done = False
    
    if visualize:
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()  # Handle window close event

            action, _ = trainer.agent.select_action(obs)  # Select action using trained policy
            next_obs, reward, done, _, _ = env.step(action)

            # Render the environment
            renderer.render(env.agent, env.obstacle_manager, env.target_position)
            pygame.time.delay(30)  # Control visualization speed

            obs = next_obs

# Close the renderer at the end
renderer.close()
pygame.quit()
