import gymnasium as gym
from gymnasium import spaces
import numpy as np

from physics import CartPole

class CartPoleEnv(gym.Env):
    def __init__(self,
                max_force: float = 20.0,
                dt: float = 0.01,
                max_episode_time: float = 10.0,
                x_limit: float = 3.0):
        
        super().__init__()

        self.max_force = max_force
        self.dt = dt
        self.max_episode_time = max_episode_time
        self.x_limit = x_limit

        self.max_steps = int(max_episode_time / dt)

        # physics model
        self.physics = CartPole(mass_cart=10.0,
                                cart_damping=0.5,
                                pivot_damping=0.02)

        # action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # observation space
        self.observation_space = spaces.Box(low=np.array(
                                                [-np.inf, -np.inf, -1.0, -1.0, -np.inf],
                                                dtype=np.float32
                                            ),
                                            high=np.array(
                                                [np.inf, np.inf, 1.0, 1.0, np.inf],
                                                dtype=np.float32
                                            ),
                                            dtype=np.float32
                                            )
        
        # current internal state
        self.state = np.zeros(4, dtype=np.float64)
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        x = self.np_random.uniform(low=-0.05, high=0.05)
        theta = self.np_random.uniform(low=-0.1, high=0.1)
        x_dot = self.np_random.uniform(low=-0.05, high=0.05)
        theta_dot = self.np_random.uniform(low=-0.05, high=0.05)

        self.state = np.array(
            [x, theta, x_dot, theta_dot],
            dtype=np.float64
        )

        observation = self._get_observation()

        return observation, {}

    def step(self, action):
        # get action as float and clip to [-1, 1]
        a = np.clip(float(action[0]), -1.0, 1.0)

        # scale action to force
        force = a * self.max_force

        # update state using physics model
        self.state = self.physics.rk4_step(
            self.state,
            force,
            self.dt
        )

        # increment step count
        self.step_count += 1

        # get observation
        observation = self._get_observation()

        # calculate theta error
        theta_error = np.atan2(np.sin(self.state[1]), np.cos(self.state[1]))

        # calculate reward
        reward = float(np.cos(theta_error) - 0.1 * (self.state[0]/self.x_limit)**2 - 0.001 * a**2)

        # termination and truncation conditions
        x = self.state[0]

        terminated = bool(
            abs(self.state[0]) > self.x_limit
            or abs(theta_error) > np.deg2rad(45)
        )

        truncated = bool(
            self.step_count >= self.max_steps
        )

        info = {
            "theta_error": theta_error,
            "x": self.state[0],
            "force": force
        }

        return observation, reward, terminated, truncated, info




    def _get_observation(self):
        x, theta, x_dot, theta_dot = self.state

        # build observation here
        observation = np.array([x, x_dot, np.sin(theta), np.cos(theta), theta_dot], dtype=np.float32)
        return observation