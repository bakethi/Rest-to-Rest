import numpy as np


class PhysicsObject:
    def __init__(
        self,
        position=[0, 0],
        velocity=None,
        mass=1.0,
        drag=0.1,
        bounds=None,  
        #bounds [x_min, y_min], [x_max, y_max]
        max_speed=None,
        bounce_factor=1,
        obstacleManager=None,
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

    def apply_force(self, force):
        """
        Apply a force to the object, updating its acceleration.
        """
        self.velocity += np.array(force, dtype=np.float32)

    def update(self):
        """
        Update the object's position and velocity.
        """
        self.velocity = (self.velocity * (1 - self.drag)) / self.mass

        speed = np.linalg.norm(self.velocity)
        if self.max_speed is not None:
            if speed > self.max_speed:
                self.velocity = (self.velocity / speed) * self.max_speed

        # Set values near zero to zero
        self.velocity[np.abs(self.velocity) < 0.01] = 0

        # update position
        self.position += self.velocity

        if self.obstacleManager is not None:
            if self.obstacleManager.check_collision(self):
                self.velocity *= (self.bounce_factor * -1)
                self.position += self.velocity

        # Apply bounds if provided
        if self.bounds is not None:
            self.position = np.clip(self.position, self.bounds[0], self.bounds[1])

        self.path_history.append(self.position)

    def reset(self, position=None, velocity=None):
        """
        Reset the object's position and velocity.
        """
        if position is not None:
            self.position = np.array(position, dtype=np.float32)
        if velocity is not None:
            self.velocity = np.array(velocity, dtype=np.float32)
        self.acceleration = np.array([0.0, 0.0], dtype=np.float32)


class PhysicsEngine:
    def __init__(self, objects=None):
        self.objects = objects if objects is not None else []

    def add_object(self, obj):
        """
        Add a PhysicsObject to the engine.
        """
        self.objects.append(obj)

    def update(self):
        """
        Update all managed objects.
        """
        for obj in self.objects:
            obj.update()
