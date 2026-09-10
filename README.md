# Inverted Pendulum Reinforcement Learning

A from-scratch simulation and reinforcement learning project for controlling an inverted pendulum.

The long-term goal is to train an RL agent to control a **double inverted pendulum on a cart**.  
The project currently implements and trains on a single cart-pole system first, with the physics, numerical integration, visualization, and Gymnasium environment built manually.

## Current State

The current implementation includes:

* Nonlinear cart-pole dynamics derived using Lagrangian mechanics
* Custom RK4 numerical integration
* Viscous cart and pivot damping
* Interactive Pygame visualization
* Manual keyboard control
* Continuous force input
* Gymnasium-compatible reinforcement learning environment
* PPO training using Stable-Baselines3
* Model checkpointing and TensorBoard logging
* Deterministic evaluation of trained agents

The first PPO agent is already capable of balancing the pendulum for the full 10-second training episode in most tested initial conditions.

\---

## System Model

The physical state is

\[
s =
\\begin{bmatrix}
x \\
\\theta \\
\\dot{x} \\
\\dot{\\theta}
\\end{bmatrix}
]

with the following convention:

* `x > 0`: cart moves to the right
* `theta = 0`: pendulum is upright
* `theta > 0`: counter-clockwise rotation
* `F > 0`: force acts on the cart to the right

The nonlinear dynamics are derived from

\[
\\mathcal{L} = T - V
]

using the Euler-Lagrange equations.

The resulting continuous-time system is represented as

\[
\\dot{s} = f(s, F)
]

and integrated numerically using fourth-order Runge-Kutta (RK4).

The physical model also supports viscous damping:

\[
F\_{\\text{friction}} = -b\\dot{x}
]

for the cart and

\[
\\tau\_{\\text{friction}} = -c\\dot{\\theta}
]

for the pendulum pivot.

\---

## Reinforcement Learning Environment

The RL environment is implemented using Gymnasium.

### Observation Space

The internal physical angle is not passed directly to the agent. Instead, the observation is

\[
o =
\\begin{bmatrix}
x \\
\\dot{x} \\
\\sin(\\theta) \\
\\cos(\\theta) \\
\\dot{\\theta}
\\end{bmatrix}
]

Using `sin(theta)` and `cos(theta)` avoids the discontinuity between angles such as `0` and `2π`.

### Action Space

The agent outputs one continuous action

\[
a \\in \[-1, 1]
]

which is mapped to force using

\[
F = a F\_{\\max}
]

The current default is

\[
F\_{\\max} = 20,N.
]

### Episode

The pendulum starts close to the upright equilibrium with small random perturbations.

An episode terminates if:

* the cart leaves the allowed horizontal region
* the pendulum exceeds the allowed angular deviation

An episode is truncated after the maximum episode duration.

The current training setup uses 10-second episodes with a simulation timestep of

\[
dt = 0.01,s.
]

\---

## Reward Function

The initial reward function prioritizes keeping the pendulum upright while mildly penalizing cart displacement and control effort.

Conceptually:

\[
r =
\\cos(\\theta)
---

## \\lambda\_x x^2

\\lambda\_u a^2
]

The first trained PPO policy successfully balances the pendulum, but exhibits a small steady-state cart position offset.

This is an interesting consequence of the current reward design: maintaining an almost perfect pendulum angle is rewarded more strongly than returning the cart exactly to the center.

Future versions will investigate improved reward shaping and longer-horizon behavior.

\---

## Visualization

The Pygame renderer displays:

* cart
* wheels
* pendulum
* pivot
* pendulum mass
* applied force arrow
* live state HUD

The force arrow scales with the magnitude and direction of the applied control force.

### Manual Controls

|Key|Action|
|-|-|
|`Left Arrow`|Apply negative force|
|`Right Arrow`|Apply positive force|
|`Space`|Pause / resume|
|`N`|Advance one simulation step while paused|
|`R`|Reset simulation|
|`+`|Increase maximum manual force|
|`-`|Decrease maximum manual force|
|`Esc`|Quit|

\---

## Project Structure

```text
InvertedPendulumRL/
├── physics.py
├── renderer.py
├── main.py
├── cartpole\\\_env.py
├── test\\\_env.py
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

### `physics.py`

Contains the nonlinear physical model, damping, and RK4 integration.

### `renderer.py`

Contains the Pygame visualization.

### `main.py`

Runs the interactive manually controlled simulation.

### `cartpole\\\_env.py`

Wraps the physical simulation as a Gymnasium environment.

### `test\\\_env.py`

Contains basic environment tests and compatibility checks.

### `train.py`

Trains PPO agents and stores intermediate checkpoints.

### `evaluate.py`

Loads trained policies and evaluates them using the Pygame renderer.

\---

## Installation

Create and activate a Python virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The project currently uses:

* NumPy
* Pygame CE
* Gymnasium
* Stable-Baselines3
* TensorBoard

\---

## Running the Manual Simulation

```bash
python main.py
```

\---

## Training PPO

```bash
python train.py
```

Training is performed without rendering so that the simulation can run as fast as possible.

Intermediate model checkpoints are stored locally but excluded from Git.

TensorBoard logs can be viewed using:

```bash
tensorboard --logdir logs
```

\---

## Evaluating a Trained Agent

```bash
python evaluate.py
```

Evaluation uses deterministic actions and renders the physical state in real time.

This makes it possible to compare agents at different stages of training, for example:

```text
0 steps
20k steps
50k steps
100k steps
200k steps
```

One of the main goals of the project is to visualize how the controller gradually improves during training.

\---

## Planned Development

The current single cart-pole serves as the development and validation platform.

Planned extensions include:

* improved reward shaping
* longer-horizon balancing
* automated evaluation across fixed random seeds
* PPO vs. SAC comparison
* PID controller benchmark
* LQR controller benchmark
* disturbance and robustness testing
* parameter variation and sensor noise
* swing-up control
* double inverted pendulum dynamics
* RL control of the double inverted pendulum

The final goal is to compare classical and reinforcement-learning-based control strategies on the same custom nonlinear simulation.

\---

## Motivation

The purpose of this project is not only to train an RL agent, but to build and understand the complete pipeline:

\[
\\text{mechanical modelling}
\\rightarrow
\\text{nonlinear state-space dynamics}
\\rightarrow
\\text{numerical integration}
\\rightarrow
\\text{simulation}
\\rightarrow
\\text{reinforcement learning}
]

Rather than relying on an existing CartPole simulator, the physical model and visualization are implemented from scratch.

