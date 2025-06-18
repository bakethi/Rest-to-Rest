import numpy as np
from .physics import PhysicsObject

class Intruder:
    def __init__(self, initial_position, bounds, max_speed=5.0, change_direction_interval=6.0, size=3.0):
        """
        Initializes an Intruder with goal-oriented movement.
        """
        self.physics = PhysicsObject(
            position=initial_position,
            max_speed=max_speed,
            bounds=bounds,
            bounce_factor=-1,
            size=size # Make size part of the constructor
        )
        self.bounds = bounds

        # --- Intruder "Brain" Attributes ---
        self.change_direction_interval = change_direction_interval
        self.time_since_last_change = change_direction_interval
        self.movement_force = 1.0
        self.target_point = None
        
        # +++ ADD: Define a small radius to consider the target "reached" +++
        self.arrival_radius = 5.0

    def update(self, agent_position, dt):
        """
        Updates the intruder's behavior and physics.
        """
        self.time_since_last_change += dt

        # --- THIS IS THE MODIFIED DECISION LOGIC ---
        
        # 1. Check if the intruder has reached its current target
        has_reached_target = False
        if self.target_point is not None:
            distance_to_target = np.linalg.norm(self.target_point - self.physics.position)
            if distance_to_target < self.arrival_radius:
                has_reached_target = True
        
        # 2. Decide on a new target if the timer is up OR the target has been reached
        if self.time_since_last_change >= self.change_direction_interval or has_reached_target:
            self.time_since_last_change = 0.0
            
            # 80% chance to pick a random destination in the environment
            if np.random.rand() < 0.8:
                self.target_point = np.random.uniform(low=self.bounds[0], high=self.bounds[1])
            
            # 20% chance to target the agent's position at this moment in time
            else:
                self.target_point = agent_position

        # 3. Always calculate direction and apply force towards the current target_point
        if self.target_point is not None:
            direction_to_target = self.target_point - self.physics.position
            norm = np.linalg.norm(direction_to_target)
            
            # Only apply force if not already at the target to prevent jittering
            if norm > self.arrival_radius:
                direction = direction_to_target / norm
                self.physics.apply_force(direction * self.movement_force)
        
        # 4. Update the physics simulation for this intruder
        self.physics.update(dt)