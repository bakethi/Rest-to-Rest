import pygame


class Renderer:
    def __init__(self, env, width=600, height=600, scale=6.0):
        """
        Initialize the Pygame renderer.

        Args:
            width (int): Width of the window in pixels.
            height (int): Height of the window in pixels.
            scale (float): Scale factor for world to screen coordinates.
        """
        pygame.init()
        self.env = env
        self.width = width
        self.height = height
        self.scale = scale
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.colors = {
            "background": (30, 30, 30),
            "agent": (0, 255, 0),
            "obstacle": (255, 0, 0),
            "target": (0, 0, 255),
            "path": (255, 192, 203)
        }

    def render(self, agent, obstacle_manager, target_position):
        """
        Render the environment.

        Args:
            agent (PhysicsObject): The agent to be rendered.
            obstacle_manager (ObstacleManager): Manages the obstacles to be rendered.
            target_position (np.array): Coordinates of the target.
        """
        # Clear the screen
        self.screen.fill(self.colors["background"])

        # Draw the target
        target_screen_pos = self._world_to_screen(target_position)
        pygame.draw.circle(self.screen, self.colors["target"], target_screen_pos, 10)

        # Draw the agent
        agent_screen_pos = self._world_to_screen(agent.position)
        pygame.draw.circle(self.screen, self.colors["agent"], agent_screen_pos, 10)

        # Draw the obstacles
        for obstacle in obstacle_manager.get_obstacles():
            # Convert world coordinates to screen coordinates
            obstacle_screen_pos = self._world_to_screen(obstacle["position"])

            # Scale the obstacle size
            obstacle_size = int(obstacle["size"] * self.scale)

            # Get obstacle type
            obstacle_type = obstacle.get("type", "square")  # Default to "square" if type is missing

            if obstacle_type == "circle":
                # Draw a circle obstacle
                pygame.draw.circle(
                    self.screen,
                    self.colors["obstacle"],
                    (int(obstacle_screen_pos[0]), int(obstacle_screen_pos[1])),
                    obstacle_size
                )
            else:
                # Draw a square obstacle
                pygame.draw.rect(
                    self.screen,
                    self.colors["obstacle"],
                    pygame.Rect(
                        obstacle_screen_pos[0] - obstacle_size // 2,
                        obstacle_screen_pos[1] - obstacle_size // 2,
                        obstacle_size,
                        obstacle_size,
                    ),
                )

        # Draw the path taken by the agent
        if len(agent.path_history) > 1:
            for i in range(1, len(agent.path_history)):
                start_pos = tuple(self._world_to_screen(agent.path_history[i - 1]))
                end_pos = tuple(self._world_to_screen(agent.path_history[i]))
                if start_pos != end_pos:  # Avoid drawing zero-length lines
                    pygame.draw.line(self.screen, self.colors["path"], start_pos, end_pos, 2)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        self.clock.tick(60)

    def _world_to_screen(self, position):
        """
        Convert world coordinates to screen coordinates.

        Args:
            position (np.array): World coordinates [x, y].

        Returns:
            tuple: Screen coordinates (x, y).
        """
        screen_x = int(position[0] * self.scale)
        screen_y = int(self.height - position[1] * self.scale)  # Flip y-axis
        return screen_x, screen_y

    def close(self):
        """
        Close the Pygame window.
        """
        pygame.quit()
