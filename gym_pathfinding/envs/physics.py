import numpy as np


class PhysicsObject:
    def __init__(
        self,
        position=[0, 0],
        velocity=None,
        mass=1.0,
        drag=0.1,
        bounds=None,    # bounds [x_min, y_min], [x_max, y_max]
        max_speed=20.0,
        bounce_factor=1,
        obstacleManager=None,
        size = 3,
    ):
        """
        Initializes the PhysicsObject.

        Args:
            position (list or np.array, optional): The initial position of the object in world coordinates (default is [0, 0]).
            velocity (list or np.array, optional): The initial velocity of the object (default is None, meaning zero velocity).
            mass (float, optional): The mass of the object (default is 1.0).
            drag (float, optional): The drag coefficient, which will affect the velocity over time (default is 0.1).
            bounds (tuple, optional): The bounds for the object in the form ((x_min, y_min), (x_max, y_max)) to keep the
            object within these limits.
            max_speed (float, optional): The maximum speed the object can reach. If None, there is no speed limit.
        """
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity if velocity is not None else [0.0, 0.0], dtype=np.float32)
        self.acceleration = np.array([0.0, 0.0], dtype=np.float32)
        self.mass = mass
        self.drag = drag
        self.bounds = bounds
        self.max_speed = max_speed
        self.path_history = [self.position]
        self.bounce_factor = bounce_factor
        self.obstacleManager = obstacleManager
        self.size = size

    def apply_force(self, force):
        """
        Apply a force to the object, updating its acceleration.
        """
        self.velocity += np.array(force, dtype=np.float32)

    def update(self, dt):
        # Apply drag, scaled by dt for consistency
        self.velocity *= (1 - self.drag * dt)

        # Update velocity with acceleration, scaled by dt
        self.velocity += self.acceleration * dt

        # Limit speed (this part doesn't need dt)
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed

        # Update position, scaled by dt
        self.position += self.velocity * dt

        # Reset acceleration for the next step
        self.acceleration = np.zeros(2)

        # Bounds checking (this also doesn't need dt)
        if self.bounds is not None:
            if not (self.bounds[0][0] <= self.position[0] <= self.bounds[1][0]):
                self.velocity[0] *= self.bounce_factor
            if not (self.bounds[0][1] <= self.position[1] <= self.bounds[1][1]):
                self.velocity[1] *= self.bounce_factor
            self.position = np.clip(self.position, self.bounds[0], self.bounds[1])

        self.path_history.append(self.position.copy())


    def reset(self, position=None, velocity=None):
        """
        Reset the object's position and velocity.
        """
        if position is not None:
            self.position = np.array(position, dtype=np.float32)
        if velocity is not None:
            self.velocity = np.array(velocity, dtype=np.float32)
        self.acceleration = np.array([0.0, 0.0], dtype=np.float32)
        self.path_history = []


class PhysicsEngine:
    def __init__(self, objects=None):
        self.objects = objects if objects is not None else []

    def add_object(self, obj):
        """
        Add a PhysicsObject to the engine.
        """
        self.objects.append(obj)

    def update(self, dt):
        """
        Update all managed objects.
        """
        for obj in self.objects:
            obj.update(dt)
