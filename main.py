import numpy as np
import pygame

from physics import CartPole
from renderer import Renderer


model = CartPole(mass_cart=10.0,
                 cart_damping=0.5,
                 pivot_damping=0.02)
renderer = Renderer(model.length)

initial_state = np.array([0.0, 0.1, 0.0, 0.0])
state = initial_state.copy()

max_force = 20.0

dt = 0.01
sim_time = 60.0
num_steps = int(sim_time / dt)

clock = pygame.time.Clock()

running = True
paused = False
simulation_step = 0


while running and simulation_step <= num_steps:

    # --------------------------------
    # Discrete keyboard events
    # --------------------------------
    single_step = False

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:
                running = False

            elif event.key == pygame.K_r:
                state = initial_state.copy()
                simulation_step = 0

            elif event.key == pygame.K_SPACE:
                paused = not paused

            elif event.key == pygame.K_n:
                if paused:
                    single_step = True

            elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                max_force += 5.0

            elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                max_force = max(0.0, max_force - 5.0)

    # --------------------------------
    # Continuous keyboard input
    # --------------------------------
    keys = pygame.key.get_pressed()

    force = 0.0

    if keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
        force = max_force

    elif keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
        force = -max_force

    # --------------------------------
    # Rendering
    # --------------------------------
    renderer.render(state, force)

    # --------------------------------
    # Physics
    # --------------------------------
    if not paused or single_step:

        state = model.rk4_step(state, force, dt)
        simulation_step += 1

        if simulation_step % 10 == 0:
            t = simulation_step * dt
            x = state[0]
            theta = state[1]

            print(
                f"t = {t:.2f}, "
                f"x = {x:.4f}, "
                f"theta = {theta:.4f}, "
                f"Fmax = {max_force:.1f}"
            )

    # 100 FPS
    clock.tick(1 / dt)


renderer.close()