import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from cartpole_env import CartPoleEnv


# Create output folders
os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Environment
env = CartPoleEnv()
env = Monitor(env)

# PPO agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/",
    seed=42,
    device="cpu"
)

# Save completely untrained agent
model.save("models/ppo_cartpole_000000")

# Save intermediate training states
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path="./models/checkpoints/",
    name_prefix="ppo_cartpole"
)

# Train
model.learn(
    total_timesteps=200_000,
    callback=checkpoint_callback
)

# Save final model
model.save("models/ppo_cartpole_final")

env.close()