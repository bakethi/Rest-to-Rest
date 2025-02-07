import unittest
import numpy as np
from gym_pathfinding.envs.physics import PhysicsObject, PhysicsEngine
from gym_pathfinding.utils.obstacles import ObstacleManager
import copy


class TestPhysics(unittest.TestCase):
    def setUp(self):
        # Initialize Physics object before each test
        self.agent = PhysicsObject()
        self.obstacleManager = ObstacleManager()

    def test_initial_conditions(self):
        # Verify initial conditions
        self.assertTrue(np.array_equal(self.agent.velocity, np.array([0.0, 0.0])))  # Initial velocity
        self.assertTrue(np.array_equal(self.agent.position, np.array([0.0, 0.0])))  # Initial position

    def test_acceleration_update(self):
        # Apply acceleration and test velocity update
        self.agent.apply_force(np.array([1.0, 0.0]))
        self.agent.update()  # Update for 1 second
        self.assertTrue(np.allclose(self.agent.velocity, np.array([0.9, 0.0])))  # Velocity should match acceleration

    def test_drag_effect(self):
        # Test drag reducing velocity over time
        self.agent.velocity = np.array([10.0, 0.0])
        for _ in range(10):
            self.agent.update()  # Update for 1 second each
        self.assertTrue(np.linalg.norm(self.agent.velocity) < 10.0)  # Velocity should decrease due to drag

    def test_acceleration_and_drag(self):
        # Test both acceleration and drag affecting velocity over time
        self.agent.apply_force(np.array([1.0, 0.0]))
        for _ in range(10):
            self.agent.update()  # Update for 1 second each
        self.assertTrue(np.linalg.norm(self.agent.velocity) > 0.0)
        # Velocity should still be positive but less than initial due to drag

    def test_velocity_limits(self):
        # Test if velocity exceeds the max speed limit
        self.agent = PhysicsObject(max_speed=100)
        self.agent.velocity = np.array([1000.0, 0.0])
        self.agent.update()
        self.assertTrue(np.linalg.norm(self.agent.velocity) <= self.agent.max_speed)  # Should be clamped to max speed

    def test_boundary_conditions(self):
        # Test boundary conditions to prevent going outside world bounds
        self.agent = PhysicsObject(bounds=np.array([[0, 0], [100, 100]]))  # 2D bounds for x and y
        self.agent.position = np.array([1000.0, 1000.0])
        self.agent.update()
        self.assertTrue(
            np.all(self.agent.position >= self.agent.bounds[0])
            and np.all(self.agent.position <= self.agent.bounds[1])
        )  # Position should not exceed world bounds

    def test_physics_engine(self):
        self.agent = PhysicsObject()
        self.engine = PhysicsEngine()
        self.engine.add_object(self.agent)
        self.agent.apply_force(np.array([1.0, 0.0]))
        self.engine.update()
        self.assertTrue(np.allclose(self.agent.velocity, np.array([0.9, 0.0])))

    def test_multiple_steps(self):
        self.agent = PhysicsObject(bounds=[[0, 0], [300, 300]])
        self.engine = PhysicsEngine()
        self.engine.add_object(self.agent)
        for _ in range(100):
            self.agent.apply_force(np.array([1.0, 1.0]))
            self.engine.update()
        self.assertTrue(np.allclose(self.agent.position, np.array([300, 300])))

    def test_position_change_without_inputs(self):
        self.agent = PhysicsObject(bounds=[[0, 0], [300, 300]])
        self.engine = PhysicsEngine()
        self.engine.add_object(self.agent)
        self.agent.apply_force(np.array([1.0, 1.0]))
        self.engine.update()
        old_position = copy.deepcopy(self.agent.position)
        self.engine.update()
        new_position = copy.deepcopy(self.agent.position)
        self.assertFalse(np.array_equal(old_position, new_position))

    def test_sliding(self):
        self.agent = PhysicsObject(bounds=[[0, 0], [300, 300]])
        self.engine = PhysicsEngine()
        self.engine.add_object(self.agent)
        self.agent.apply_force(np.array([1.0, 1.0]))
        self.engine.update()
        old_position = copy.deepcopy(self.agent.position)
        for _ in range(100):
            self.engine.update()
        new_position_1 = copy.deepcopy(self.agent.position)
        self.engine.update()
        new_position_2 = copy.deepcopy(self.agent.position)
        self.assertFalse(np.array_equal(old_position, new_position_1), "Old position is the same as new_pos1")
        self.assertFalse(np.array_equal(old_position, new_position_2), "Old position is the same as new_pos2")
        self.assertTrue(np.array_equal(new_position_1, new_position_2), "new_pos1 and new_pos2 are not the same")
        self.assertFalse(np.any(np.isclose(new_position_1, 300, atol=0.1)), "new_pos1 at the env boundaries")
        self.assertFalse(np.any(np.isclose(new_position_2, 300, atol=0.1)), "new_pos2 at the env boundaries")

    def test_collision(self):
        # test square obstacle
        self.agent = PhysicsObject(bounds=[[0, 0], [300, 300]])
        self.engine = PhysicsEngine()
        self.engine.add_object(self.agent)
        self.obstacleManager.add_obstacle([300, 300], 599)
        self.agent.apply_force(np.array([10, 10]))
        self.agent.update()
        self.assertTrue(self.obstacleManager.check_collision(self.agent), "square obstacle collision failing")
        # test circle obstacle
        self.agent = PhysicsObject(bounds=[[0, 0], [300, 300]])
        self.engine = PhysicsEngine()
        self.engine.add_object(self.agent)
        self.obstacleManager.add_obstacle([300, 300], 599, shape_type="circle")
        self.agent.apply_force(np.array([10, 10]))
        self.agent.update()
        self.assertTrue(self.obstacleManager.check_collision(self.agent), "circle obstacle collision failing")


if __name__ == "__main__":
    unittest.main()
