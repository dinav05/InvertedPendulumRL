import numpy as np
import pygame

from stable_baselines3 import PPO

from cartpole_env import CartPoleEnv
from renderer import Renderer


# -----------------------------
# Load environment and model
# -----------------------------
env = CartPoleEnv(max_episode_time=60.0)

model = PPO.load(
    "models/ppo_cartpole_final",
    device="cpu"
)

renderer = Renderer(env.physics.length)

clock = pygame.time.Clock()


# -----------------------------
# Start episode
# -----------------------------
observation, info = env.reset(seed=42)

running = True


while running:

    # Window events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # Let PPO choose an action
    action, _ = model.predict(
        observation,
        deterministic=True
    )

    # Calculate force for visualization
    a = float(np.clip(action[0], -1.0, 1.0))
    force = a * env.max_force

    # Render current physical state
    renderer.render(
        env.state,
        force
    )

    # Apply action to environment
    observation, reward, terminated, truncated, info = env.step(action)

    # End episode
    if terminated or truncated:
        print("Episode finished")
        print(f"Steps: {env.step_count}")
        print(f"Reward: {reward:.3f}")
        print(f"terminated: {terminated}")
        print(f"truncated: {truncated}")
        break

    # Real-time playback
    clock.tick(1 / env.dt)


renderer.close()