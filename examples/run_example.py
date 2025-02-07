import numpy as np
from gym_pathfinding.envs.pathfinding_env import PathfindingEnv
from gym_pathfinding.envs import Renderer


class RunExample:
    def __init__(self):
        # Initialize environment, visualization, and pathfinding instances
        self.env = PathfindingEnv(number_of_obstacles=100, bounds=np.array([[0, 0], [100, 100]]), bounce_factor=1)
        self.visualization = Renderer(self.env)
        self.magnitude = 1

    def run(self):
        # Run the environment
        done = False

        while not done:
            self.visualization.render(self.env.agent, self.env.obstacle_manager, self.env.target_position)
            action = self.env.action_space.sample()
            action * self.magnitude
            self.env.step(action)
            for _ in range(10):
                action = [0, 0]
                self.env.step(action)
                self.visualization.render(self.env.agent, self.env.obstacle_manager, self.env.target_position)
            if np.array_equal(self.env.target_position, self.env.agent.position):
                done = True


if __name__ == "__main__":
    example = RunExample()
    example.run()  # Run the example
