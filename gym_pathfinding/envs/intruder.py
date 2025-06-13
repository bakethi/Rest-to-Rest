import numpy as np
from .physics import PhysicsObject

class Intruder:
    def __init__(self, initial_position, bounds, max_speed=5.0, change_direction_interval=2.0):
        """
        Initializes an Intruder.

        Args:
            initial_position (np.array): The starting position.
            bounds (np.array): The environment boundaries.
            max_speed (float): The maximum speed of the intruder.
            change_direction_interval (float): Time in seconds between changing direction.
        """
        # Each intruder has its own physics object
        self.physics = PhysicsObject(
            position=initial_position,
            max_speed=max_speed,
            bounds=bounds,
            bounce_factor=-1 # Make intruders bounce off walls
        )
        self.physics.size = 10 # Example size for collision detection

        # --- Intruder "Brain" Attributes ---
        self.change_direction_interval = change_direction_interval
        self.time_since_last_change = 0.0
        self.movement_force = 1.0 # The force with which the intruder pushes itself
        self.direction = np.array([0.0, 0.0])

    def update(self, agent_position, dt):
        """
        Updates the intruder's behavior and physics.

        Args:
            agent_position (np.array): The current position of the main agent.
            dt (float): The time step for the physics update.
        """
        # 1. Update the "brain" timer
        self.time_since_last_change += dt

        # 2. Decide on a new direction if the timer is up
        if self.time_since_last_change >= self.change_direction_interval:
            self.time_since_last_change = 0.0  # Reset timer
            
            # This is the logic for deciding the direction
            # 80% chance to move in a random direction
            if np.random.rand() < 0.8:
                # Choose a random angle and create a direction vector
                random_angle = np.random.uniform(0, 2 * np.pi)
                self.direction = np.array([np.cos(random_angle), np.sin(random_angle)])
            
            # 20% chance to move towards the agent
            else:
                # Calculate direction vector towards the agent
                direction_to_agent = agent_position - self.physics.position
                # Normalize the vector to get a unit direction
                norm = np.linalg.norm(direction_to_agent)
                if norm > 0:
                    self.direction = direction_to_agent / norm
                else:
                    # If somehow on top of the agent, move randomly
                    self.direction = np.array([1.0, 0.0])

        # 3. Apply force to move in the chosen direction
        self.physics.apply_force(self.direction * self.movement_force)
        
        # 4. Update the physics simulation for this intruder
        self.physics.update(dt)