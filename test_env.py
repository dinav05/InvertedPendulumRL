from cartpole_env import CartPoleEnv

from stable_baselines3.common.env_checker import check_env

env = CartPoleEnv()
check_env(env)

obs, info = env.reset(seed=42)

total_reward = 0.0
steps = 0

while True:
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    steps += 1

    if terminated or truncated:
        break

print(f"Steps: {steps}")
print(f"Total reward: {total_reward:.2f}")
print(f"terminated: {terminated}")
print(f"truncated: {truncated}")
print(f"x: {info.get('x')}")
print(f"theta_error: {info.get('theta_error')}")