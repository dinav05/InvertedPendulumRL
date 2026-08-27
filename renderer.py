import pygame
import numpy as np

class Renderer:
    def __init__(self, pendulum_length: float = 1.0):
        self.pendulum_length = pendulum_length

        # Window settings
        self.width = 800
        self.height = 600

        # Conversion from physical meters to screen pixels
        self.scale = 100  # pixels per meter

        # Screen y-coordinate corresponding to physical y = 0
        self.ground_y = int(self.height * 0.7)

        # Set cart settings
        self.cart_width = 80
        self.cart_height = 40

        # Color settings
        self.background_color = (30, 30, 30)
        self.ground_color = (200, 200, 200)
        self.cart_color = (100, 150, 220)
        self.pendulum_color = (220, 220, 220)
        self.mass_color = (220, 80, 80)
        self.wheel_color = (55, 55, 55)
        self.wheel_hub_color = (170, 170, 170)
        self.pivot_color = (240, 190, 80)
        self.hud_background = (20, 20, 20)
        self.hud_border = (80, 80, 80)
        self.hud_text = (225, 225, 225)
        self.hud_secondary = (150, 150, 150)

        # Wheel settings
        self.wheel_radius = 10
        self.wheel_offset = 24

        # Pivot settings
        self.pivot_radius = 7

        # Force arrow settings
        self.force_color = (80, 220, 120)
        self.force_arrow_scale = 4.0  # pixels per Newton
        self.force_arrow_max_length = 100
        self.force_arrow_head_length = 12
        self.force_arrow_head_width = 10


        pygame.init()

        self.font = pygame.font.SysFont("consolas", 18)
        self.font_small = pygame.font.SysFont("consolas", 15)

        self.screen = pygame.display.set_mode(
            (self.width, self.height)
        )

        pygame.display.set_caption("Cart-Pole Simulation")

    def render(self, state, force):
        # Unpack state
        x, theta, x_dot, theta_dot = state

        # Get cart positions
        cart_center_x = self.width / 2 + x * self.scale

        cart_left = cart_center_x - self.cart_width / 2
        cart_top = (
                self.ground_y
                - 2 * self.wheel_radius
                - self.cart_height
        )

        # Cart Wheels
        wheel_y = self.ground_y - self.wheel_radius

        left_wheel_x = cart_center_x - self.wheel_offset
        right_wheel_x = cart_center_x + self.wheel_offset

        # Pivot for pendulum
        pivot_x = cart_center_x
        pivot_y = cart_top

        # Calculate pendulum length in pixels
        pendulum_length_px = self.pendulum_length * self.scale

        # Calculate position of pendulum end
        pendulum_x = pivot_x - pendulum_length_px * np.sin(theta)
        pendulum_y = pivot_y - pendulum_length_px * np.cos(theta)

        # Background
        self.screen.fill(self.background_color)

        # Ground / rail
        pygame.draw.line(
            self.screen,
            self.ground_color,
            (0, self.ground_y),
            (self.width, self.ground_y),
            2
        )

        # Wheels
        pygame.draw.circle(
            self.screen,
            self.wheel_color,
            (left_wheel_x, wheel_y),
            self.wheel_radius
        )

        pygame.draw.circle(
            self.screen,
            self.wheel_color,
            (right_wheel_x, wheel_y),
            self.wheel_radius
        )

        pygame.draw.circle(
            self.screen,
            self.wheel_hub_color,
            (left_wheel_x, wheel_y),
            4
        )

        pygame.draw.circle(
            self.screen,
            self.wheel_hub_color,
            (right_wheel_x, wheel_y),
            4
        )

        # Cart
        cart_rect = pygame.Rect(
            cart_left,
            cart_top,
            self.cart_width,
            self.cart_height
        )

        pygame.draw.rect(
            self.screen,
            self.cart_color,
            cart_rect,
            border_radius=6
        )

        # Pendulum
        pygame.draw.line(
            self.screen,
            self.pendulum_color,
            (pivot_x, pivot_y),
            (pendulum_x, pendulum_y),
            width=5
        )

        # Pendulum pivot
        pygame.draw.circle(
            self.screen,
            self.pivot_color,
            (pivot_x, pivot_y),
            self.pivot_radius
        )
        pygame.draw.circle(
            self.screen,
            self.background_color,
            (pivot_x, pivot_y),
            3
        )

        # Mass on Pendulum
        pygame.draw.circle(
            self.screen,
            self.mass_color,
            (pendulum_x, pendulum_y),
            radius=10
        )

        # Draw Force Arrow
        self.draw_force_arrow(
            cart_center_x,
            cart_top,
            force
        )

        # Draw HUD
        self.draw_hud(state, force)

        # Show finished frame
        pygame.display.flip()

    def draw_force_arrow(self, cart_center_x, cart_top, force):
        if abs(force) < 1e-6:
            return

        direction = 1 if force > 0 else -1

        arrow_length = min(
            abs(force) * self.force_arrow_scale,
            self.force_arrow_max_length
        )

        cart_center_y = cart_top + self.cart_height / 2

        # Start slightly outside the cart
        start_x = (
                cart_center_x
                + direction * (self.cart_width / 2 + 10)
        )

        end_x = start_x + direction * arrow_length

        # Arrow shaft
        pygame.draw.line(
            self.screen,
            self.force_color,
            (start_x, cart_center_y),
            (end_x, cart_center_y),
            width=4
        )

        # Arrow head
        base_x = end_x - direction * self.force_arrow_head_length

        arrow_head = [
            (end_x, cart_center_y),
            (base_x, cart_center_y - self.force_arrow_head_width / 2),
            (base_x, cart_center_y + self.force_arrow_head_width / 2)
        ]

        pygame.draw.polygon(
            self.screen,
            self.force_color,
            arrow_head
        )

    def draw_hud(self, state, force):
        x, theta, x_dot, theta_dot = state

        theta_deg = np.degrees(theta)

        panel_x = 20
        panel_y = 20
        panel_width = 235
        panel_height = 175

        # HUD panel
        panel = pygame.Rect(
            panel_x,
            panel_y,
            panel_width,
            panel_height
        )

        pygame.draw.rect(
            self.screen,
            self.hud_background,
            panel,
            border_radius=8
        )

        pygame.draw.rect(
            self.screen,
            self.hud_border,
            panel,
            width=1,
            border_radius=8
        )

        # Title
        title = self.font.render(
            "CART-POLE",
            True,
            self.hud_text
        )

        self.screen.blit(
            title,
            (panel_x + 15, panel_y + 12)
        )

        # Values displayed in HUD
        values = [
            ("x", f"{x:+.3f} m"),
            ("v", f"{x_dot:+.3f} m/s"),
            ("theta", f"{theta_deg:+.1f} deg"),
            ("omega", f"{theta_dot:+.3f} rad/s"),
            ("F", f"{force:+.1f} N"),
        ]

        start_y = panel_y + 48
        line_height = 24

        for i, (label, value) in enumerate(values):
            y = start_y + i * line_height

            label_surface = self.font_small.render(
                label,
                True,
                self.hud_secondary
            )

            value_surface = self.font_small.render(
                value,
                True,
                self.hud_text
            )

            self.screen.blit(
                label_surface,
                (panel_x + 15, y)
            )

            self.screen.blit(
                value_surface,
                (panel_x + 80, y)
            )

    def close(self):
        pygame.quit()