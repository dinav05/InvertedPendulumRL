import numpy as np

# state = np.array([
#     x,
#     theta,
#     x_dot,
#     theta_dot
# ])

class CartPole:
    def __init__(self,
                 mass_cart: float = 1.0,
                 mass_pendulum: float = 1.0,
                 length: float = 1.0,
                 gravity: float = 9.81,
                 cart_damping: float = 0.0,
                 pivot_damping: float = 0.0):

        self.mass_cart = mass_cart
        self.mass_pendulum = mass_pendulum
        self.length = length
        self.gravity = gravity
        self.cart_damping = cart_damping
        self.pivot_damping = pivot_damping

    def dynamics(self, state, force):
        x, theta, x_dot, theta_dot = state

        denominator = self.mass_cart + self.mass_pendulum * np.sin(theta)**2

        x_ddot = (force
                   - self.mass_pendulum * self.length * np.sin(theta) * theta_dot**2
                   - self.cart_damping * x_dot
                   + self.mass_pendulum * self.gravity * np.sin(theta) * np.cos(theta)
                   - self.pivot_damping / self.length * np.cos(theta) * theta_dot
                   ) / denominator

        theta_ddot = (x_ddot * np.cos(theta)
                      + self.gravity * np.sin(theta)
                      - self.pivot_damping / (self.mass_pendulum * self.length) * theta_dot
                      ) / self.length

        return np.array([x_dot, theta_dot, x_ddot, theta_ddot])


    def euler_step(self, state, force, dt = 0.01):

        new_state = state + self.dynamics(state, force) * dt

        return new_state

    def rk4_step(self, state, force, dt = 0.01):

        k1 = self.dynamics(state, force)
        k2 = self.dynamics(state + dt/2 * k1, force)
        k3 = self.dynamics(state + dt/2 * k2, force)
        k4 = self.dynamics(state + dt * k3, force)

        new_state = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        return new_state
